"""Document ingestion pipeline extracted from the legacy RAG service."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from fastapi import HTTPException, UploadFile
from PIL import Image

from api.schemas import (
    DocumentProcessingError,
    DocumentResponseStatus,
    ErrorCode,
    ProcessDocumentResponse,
    ProcessingStatus,
)
from core.domain import DocumentChunk, ProcessedDocument
from core.interfaces import (
    IDocumentRepository,
    IEmbeddingService,
    IFileStorage,
    IVectorStore,
)
from database.session import get_session
from infrastructure.progress_store import progress_store
from infrastructure.repositories import SQLDocumentRepository
from services.async_processor import async_processor
from services.debug_dump import SearchDebugDump
from services.document_processor_factory import DocumentProcessorFactory
from utils.metadata import serialize_metadata_list
from utils.common import (
    get_file_extension,
    get_file_hash,
    sanitize_filename,
    temp_print,
    validate_file_content,
    validate_uploaded_file,
)

logger = logging.getLogger(__name__)


@dataclass
class IngestionConfig:
    uploads_dir: str
    line_exclude_band: float
    line_min_chars: int
    debug_dumps_enabled: bool = False


class DocumentIngestion:
    """Full ingestion flow from upload through vector storage."""

    def __init__(
        self,
        vector_store: IVectorStore,
        sentence_vector_store: Optional[IVectorStore],
        embedding_service: IEmbeddingService,
        file_storage: IFileStorage,
        document_repo: IDocumentRepository,
        processor_factory: DocumentProcessorFactory,
        config: IngestionConfig,
        debug_dump: Optional[SearchDebugDump] = None,
    ) -> None:
        self._vector_store = vector_store
        self._sentence_store = sentence_vector_store
        self._embedding_service = embedding_service
        self._file_storage = file_storage
        self._document_repo = document_repo
        self._processor_factory = processor_factory
        self._debug_dump = debug_dump
        self._config = config

    async def process_document(self, file: UploadFile) -> ProcessDocumentResponse:
        doc_id: Optional[str] = None
        try:
            file_hash, doc_id, stored_name, _ = await self._validate_and_prepare(file)
            assert file.filename is not None

            progress_store.start(doc_id, file.filename)
            progress_store.update(
                doc_id,
                ProcessingStatus.VALIDATING,
                20,
                "Saving file...",
            )
            file_path = await self._save_and_validate_file(file, stored_name)
            file_type = get_file_extension(file.filename)

            async_processor.submit_task(
                self._process_document_background(
                    doc_id,
                    file_path,
                    file_type,
                    file_hash,
                    file.filename,
                    stored_name,
                )
            )

            return ProcessDocumentResponse(
                status=DocumentResponseStatus.PROCESSING,
                filename=file.filename,
                document_id=doc_id,
                chunks=0,
                pages=0,
                message="Document is being processed in background",
            )
        except DocumentProcessingError as error:
            logger.error("Upload validation failed: %s", error.message)
            if doc_id:
                progress_store.fail(doc_id, error.message, error.error_code)
            return ProcessDocumentResponse(
                status=DocumentResponseStatus.ERROR,
                filename=file.filename or "unknown",
                document_id="",
                chunks=0,
                pages=0,
                error=error.message,
                error_code=error.error_code,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("Unexpected error uploading")
            error = DocumentProcessingError(
                f"Unexpected error: {str(exc)}",
                ErrorCode.PROCESSING_FAILED,
            )
            return ProcessDocumentResponse(
                status=DocumentResponseStatus.ERROR,
                filename=file.filename or "unknown",
                document_id="",
                chunks=0,
                pages=0,
                error=error.message,
                error_code=error.error_code,
            )

    async def _process_document_background(
        self,
        doc_id: str,
        file_path: str,
        file_type: str,
        file_hash: str,
        filename: str,
        stored_name: str,
    ) -> None:
        document: Optional[ProcessedDocument] = None

        async with get_session() as session:
            doc_repo = SQLDocumentRepository(session)

            try:
                existing = await doc_repo.get_by_hash(file_hash)
                if existing:
                    raise DocumentProcessingError(
                        "Document already exists",
                        ErrorCode.DUPLICATE_FILE,
                    )

                document = await doc_repo.create(doc_id, filename, file_hash, stored_name)

                processor = self._processor_factory.get_processor(file_type)
                images: List[Image.Image] = await processor.load_images(file_path, file_type)

                page_image_paths: Dict[int, str] = {}
                page_thumbnail_paths: Dict[int, str] = {}
                for page_number, image in enumerate(images, 1):
                    original_rel, thumb_rel = await self._file_storage.save_page_image(
                        image=image,
                        document_id=doc_id,
                        page_number=page_number,
                    )
                    page_image_paths[page_number] = original_rel
                    page_thumbnail_paths[page_number] = thumb_rel

                document.metadata = (document.metadata or {})
                document.metadata["page_image_paths"] = page_image_paths
                document.metadata["page_thumbnail_paths"] = page_thumbnail_paths
                await doc_repo.update_metadata(document.id, document.metadata)

                try:
                    chunks, geometry_by_page = await self._extract_text_chunks(
                        file_path, file_type, document, doc_id
                    )
                except RuntimeError as error:
                    logger.error(f"[PROCESS] Attempting OCR fallback for {filename}")
                    chunks, geometry_by_page = await self._extract_text_chunks_with_fallback(
                        file_path, file_type, document, doc_id
                    )
                temp_print(f"Chunk Sample: {chunks[0]}")
                temp_print(f"metadata: {chunks[0].metadata}")

                for chunk in chunks:
                    page = int(chunk.metadata.get("page", 0))
                    chunk.metadata["image_path"] = page_image_paths.get(page, "")
                    chunk.metadata["thumbnail_path"] = page_thumbnail_paths.get(page, "")

                await self._normalize_and_persist_lines(document, geometry_by_page, page_image_paths, doc_repo)
                await self._create_and_store_sentence_embeddings(document, geometry_by_page)

                chunks = await self._generate_embeddings(chunks, doc_id)
                await self._store_chunks(chunks, doc_id)

                progress_store.complete(doc_id)
                logger.info("[PROCESS] Successfully processed %s", filename)
            except DocumentProcessingError as error:
                logger.error("[PROCESS] Processing failed for '%s': %s", filename, error.message)
                progress_store.fail(doc_id, error.message, error.error_code)
                await self._cleanup_on_failure(
                    document.id if document else None,
                    stored_name,
                    doc_id,
                    doc_repo,
                )
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.exception("[PROCESS] Unexpected error processing '%s'", filename)
                progress_store.fail(
                    doc_id,
                    f"System error: {str(exc)[:100]}",
                    ErrorCode.PROCESSING_FAILED,
                )
                await self._cleanup_on_failure(
                    document.id if document else None,
                    stored_name,
                    doc_id,
                    doc_repo,
                )

    async def _validate_and_prepare(self, file: UploadFile) -> Tuple[str, str, str, bytes]:
        if not file.filename:
            raise DocumentProcessingError(
                "No filename provided",
                ErrorCode.INVALID_FORMAT,
            )

        try:
            validate_uploaded_file(file)
        except HTTPException as exc:
            error_code = (
                ErrorCode.FILE_TOO_LARGE
                if exc.status_code == 413
                else ErrorCode.INVALID_FORMAT
                if exc.status_code == 400
                else ErrorCode.PROCESSING_FAILED
            )
            raise DocumentProcessingError(exc.detail, error_code)

        content = await file.read()
        await file.seek(0)

        doc_id = str(uuid4())
        safe_suffix = sanitize_filename(Path(file.filename).suffix)
        stored_name = f"{doc_id}{safe_suffix}"

        return get_file_hash(content), doc_id, stored_name, content

    async def _save_and_validate_file(self, file: UploadFile, stored_name: str) -> str:
        file_path = await self._file_storage.save(file, stored_name)
        assert file.filename is not None
        try:
            validate_file_content(file_path, file.filename)
        except HTTPException as exc:
            await self._file_storage.delete(stored_name)
            raise DocumentProcessingError(exc.detail, ErrorCode.INVALID_FORMAT)
        return file_path

    async def _extract_text_chunks(
        self,
        file_path: str,
        file_type: str,
        document: ProcessedDocument,
        doc_id: str,
    ) -> Tuple[List[DocumentChunk], Dict[int, Dict[str, Any]]]:
        def update_page_progress(current_page: int, total_pages: int) -> None:
            page_percent = (current_page / total_pages) * 65
            overall_percent = 30 + int(page_percent)
            progress_store.update(
                doc_id,
                ProcessingStatus.EXTRACTING_TEXT,
                overall_percent,
                f"Extracting text from page {current_page}/{total_pages}...",
            )

        progress_store.update(
            doc_id,
            ProcessingStatus.EXTRACTING_TEXT,
            30,
            "Starting text extraction...",
        )

        processor = self._processor_factory.get_processor(file_type)
        chunks, geometry_by_page = await processor.process(
            file_path, file_type, update_page_progress
        )

        if not chunks:
            raise DocumentProcessingError(
                "No content extracted from document",
                ErrorCode.NO_TEXT_FOUND,
            )

        for chunk in chunks:
            chunk.document_id = document.id
            chunk.metadata.update(
                {
                    "document_id": document.id,
                    "document_name": document.filename,
                }
            )
            if chunk.metadata is not None and "line_ids" in chunk.metadata:
                chunk.metadata["line_ids"] = serialize_metadata_list(
                    chunk.metadata.get("line_ids")
                )

        if self._config.debug_dumps_enabled and self._debug_dump:
            self._debug_dump.save_chunks(chunks, document.filename)
        return chunks, geometry_by_page

    async def _extract_text_chunks_with_fallback(
        self,
        file_path: str,
        file_type: str,
        document: ProcessedDocument,
        doc_id: str,
    ) -> Tuple[List[DocumentChunk], Dict[int, Dict[str, Any]]]:
        def update_page_progress(current_page: int, total_pages: int) -> None:
            page_percent = (current_page / total_pages) * 65
            overall_percent = 30 + int(page_percent)
            progress_store.update(
                doc_id,
                ProcessingStatus.EXTRACTING_TEXT,
                overall_percent,
                f"Extracting text from page {current_page}/{total_pages}...",
            )

        try:
            processor = self._processor_factory.get_processor(file_type)
            chunks, geometry_by_page = await processor.process(
                file_path, file_type, update_page_progress
            )
        except RuntimeError as error:
            logger.warning("[OCR] Primary engine failed: %s, trying fallback...", error)
            processor = self._processor_factory.get_fallback_processor(file_type)
            chunks, geometry_by_page = await processor.process(
                file_path, file_type, update_page_progress
            )

        if not chunks:
            raise DocumentProcessingError(
                "No content extracted from document",
                ErrorCode.NO_TEXT_FOUND,
            )

        for chunk in chunks:
            chunk.document_id = document.id
            chunk.metadata.update(
                {
                    "document_id": document.id,
                    "document_name": document.filename,
                }
            )

        if self._config.debug_dumps_enabled and self._debug_dump:
            self._debug_dump.save_chunks(chunks, document.filename)
        return chunks, geometry_by_page

    async def _generate_embeddings(
        self,
        chunks: List[DocumentChunk],
        doc_id: str,
    ) -> List[DocumentChunk]:
        progress_store.update(
            doc_id,
            ProcessingStatus.GENERATING_EMBEDDINGS,
            95,
            f"Generating embeddings for {len(chunks)} chunks...",
        )
        texts = [chunk.content for chunk in chunks]
        embeddings = await self._embedding_service.generate_embeddings(texts)
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        return chunks

    async def _store_chunks(self, chunks: List[DocumentChunk], doc_id: str) -> None:
        progress_store.update(
            doc_id,
            ProcessingStatus.STORING,
            80,
            "Storing in vector database...",
        )
        success = await self._vector_store.add_chunks(chunks)
        if not success:
            raise DocumentProcessingError(
                "Failed to store vectors",
                ErrorCode.PROCESSING_FAILED,
            )
        progress_store.update(
            doc_id,
            ProcessingStatus.STORING,
            95,
            "Storage complete",
        )

    async def _create_and_store_sentence_embeddings(
        self,
        document: ProcessedDocument,
        geometry_by_page: Dict[int, Dict[str, Any]],
    ) -> None:
        if not self._sentence_store:
            logger.info("Sentence vector store not available, skipping sentence embeddings")
            return

        sentence_chunks: List[DocumentChunk] = []
        for page_idx, geometry in geometry_by_page.items():
            for sentence in geometry.get("sentences", []) or []:
                line_ids_value = serialize_metadata_list(sentence.get("line_ids", []))
                sentence["line_ids"] = line_ids_value
                chunk = DocumentChunk(
                    id=sentence["sentence_id"],
                    content=sentence.get("text", ""),
                    document_id=document.id,
                    metadata={
                        "sentence_id": sentence["sentence_id"],
                        "document_id": document.id,
                        "page_index": page_idx,
                        "line_ids": line_ids_value,
                    },
                )
                sentence_chunks.append(chunk)

        if not sentence_chunks:
            logger.warning("No sentences extracted for %s", document.filename)
            return

        if self._config.debug_dumps_enabled and self._debug_dump:
            self._debug_dump.save_sentences(sentence_chunks, document.filename)

        texts = [chunk.content for chunk in sentence_chunks]
        embeddings = await self._embedding_service.generate_embeddings(texts)
        for chunk, embedding in zip(sentence_chunks, embeddings):
            chunk.embedding = embedding

        success = await self._sentence_store.add_chunks(sentence_chunks)
        if success:
            logger.info(
                "Stored %d sentence embeddings for %s",
                len(sentence_chunks),
                document.filename,
            )
        else:
            logger.warning("Failed to store sentence embeddings for %s", document.filename)

    async def _normalize_and_persist_lines(
        self,
        document: ProcessedDocument,
        geometry_by_page: Dict[int, Dict[str, Any]],
        page_image_paths: Dict[int, str],
        repo: IDocumentRepository,
    ) -> None:
        lines_meta: Dict[str, List[Dict[str, Any]]] = {}
        for page_idx, geometry in geometry_by_page.items():
            rel = page_image_paths.get(page_idx)
            if not rel:
                continue
            image_path = Path(self._config.uploads_dir) / rel
            with Image.open(image_path) as image:
                width, height = image.size

            page_lines: List[Dict[str, Any]] = []
            for line in geometry.get("lines", []) or []:
                x, y, w, h = line.get("bbox_px", [0, 0, 0, 0])
                top_band = self._config.line_exclude_band
                bot_band = 1.0 - top_band
                ynorm = y / height
                if ynorm < top_band or ynorm > bot_band:
                    continue

                text = line.get("text", "") or ""
                if len(text) < self._config.line_min_chars:
                    continue

                page_lines.append(
                    {
                        "line_id": line["line_id"],
                        "bbox": [x / width, y / height, w / width, h / height],
                        "text": text,
                    }
                )

            lines_meta[str(page_idx)] = page_lines

        metadata = document.metadata or {}
        metadata["lines"] = lines_meta
        document.metadata = metadata
        await repo.update_metadata(document.id, metadata)
        if self._config.debug_dumps_enabled and self._debug_dump:
            self._debug_dump.save_lines(lines_meta, document.filename)

    async def _cleanup_on_failure(
        self,
        document_id: Optional[str],
        stored_filename: str,
        doc_id: str,
        repo: Optional[IDocumentRepository] = None,
    ) -> None:
        repository = repo or self._document_repo

        if document_id:
            try:
                await self._vector_store.delete_by_document(document_id)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("Vector cleanup failed: %s", exc)

            if self._sentence_store:
                try:
                    await self._sentence_store.delete_by_document(document_id)
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.warning("Sentence vector cleanup failed: %s", exc)

            try:
                await repository.delete(document_id)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("DB cleanup failed: %s", exc)

        if stored_filename:
            try:
                await self._file_storage.delete(stored_filename)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("File cleanup failed: %s", exc)

        progress_store.remove(doc_id)
        logger.info("Cleanup complete for %s", doc_id)
