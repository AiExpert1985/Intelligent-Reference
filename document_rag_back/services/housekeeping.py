"""Maintenance operations for the RAG service."""
from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Request
from pydantic import HttpUrl

from api.schemas import DocumentsListItem
from core.domain import ProcessedDocument
from core.interfaces import (
    IDocumentRepository,
    IFileStorage,
    IMessageRepository,
    IVectorStore,
)

logger = logging.getLogger(__name__)


@dataclass
class HousekeepingConfig:
    uploads_dir: str


class Housekeeping:
    """Encapsulate document lifecycle and status checks."""

    def __init__(
        self,
        document_repo: IDocumentRepository,
        message_repo: IMessageRepository,
        file_storage: IFileStorage,
        chunk_store: IVectorStore,
        config: HousekeepingConfig,
        sentence_store: Optional[IVectorStore] = None,
        line_store: Optional[IVectorStore] = None,
    ) -> None:
        self._documents = document_repo
        self._messages = message_repo
        self._file_storage = file_storage
        self._chunk_store = chunk_store
        self._sentence_store = sentence_store
        self._line_store = line_store
        self._config = config

    async def delete_document(self, document_id: str) -> bool:
        try:
            doc = await self._documents.get_by_id(document_id)
            if not doc:
                return False

            stored_filename = doc.metadata.get("stored_filename") if doc.metadata else None
            if stored_filename:
                try:
                    await self._file_storage.delete(stored_filename)
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.warning("File deletion failed: %s", exc)

            try:
                await self._chunk_store.delete_by_document(document_id)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("Vector deletion failed: %s", exc)

            if self._sentence_store:
                try:
                    await self._sentence_store.delete_by_document(document_id)
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.warning("Sentence vector deletion failed: %s", exc)

            if self._line_store:
                try:
                    await self._line_store.delete_by_document(document_id)
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.warning("Line vector deletion failed: %s", exc)

            page_dir = Path(self._config.uploads_dir) / "page_images" / document_id
            try:
                await asyncio.to_thread(shutil.rmtree, page_dir, ignore_errors=True)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("Page image folder deletion failed: %s", exc)

            return await self._documents.delete(document_id)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("Document deletion failed: %s", exc)
            return False

    async def list_documents(self, request: Request) -> List[DocumentsListItem]:
        documents = await self._documents.list_all()
        base_url = str(request.base_url)
        return [
            DocumentsListItem(
                id=doc.id,
                filename=doc.filename,
                download_url=HttpUrl(f"{base_url}download/{doc.id}"),
            )
            for doc in documents
        ]

    async def get_document_with_path(self, document_id: str) -> Optional[Dict[str, str]]:
        doc = await self._documents.get_by_id(document_id)
        if not doc:
            return None
        stored_filename = (doc.metadata or {}).get("stored_filename")
        if not stored_filename:
            return None
        return {
            "original_filename": doc.filename,
            "path": stored_filename,
        }

    async def get_document(self, document_id: str) -> Optional[ProcessedDocument]:
        return await self._documents.get_by_id(document_id)

    async def clear_all(self) -> bool:
        try:
            docs = await self._documents.list_all()
            for doc in docs:
                stored_filename = doc.metadata.get("stored_filename") if doc.metadata else None
                if stored_filename:
                    try:
                        await self._file_storage.delete(stored_filename)
                    except Exception as exc:  # pragma: no cover - defensive guard
                        logger.warning("Delete original failed: %s", exc)

            base = Path(self._config.uploads_dir) / "page_images"
            try:
                await asyncio.to_thread(shutil.rmtree, base, ignore_errors=True)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("Failed to remove page_images dir: %s", exc)

            ok = await self._chunk_store.clear()
            if self._sentence_store:
                try:
                    ok = ok and await self._sentence_store.clear()
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.warning("Sentence vector store clear failed: %s", exc)
            if self._line_store:
                try:
                    ok = ok and await self._line_store.clear()
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.warning("Line vector store clear failed: %s", exc)

            ok = ok and await self._documents.delete_all()
            ok = ok and await self._messages.clear_history()
            return ok
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("Clear all failed: %s", exc)
            return False

    async def get_status(self) -> Dict[str, Any]:
        try:
            chunk_count = await self._chunk_store.count()
        except Exception:  # pragma: no cover - defensive guard
            chunk_count = 0

        sentence_count = 0
        if self._sentence_store:
            try:
                sentence_count = await self._sentence_store.count()
            except Exception:  # pragma: no cover - defensive guard
                sentence_count = 0

        try:
            documents = await self._documents.list_all()
            document_count = len(documents)
        except Exception:  # pragma: no cover - defensive guard
            document_count = 0

        return {
            "ready_for_queries": chunk_count > 0,
            "chunk_count": chunk_count,
            "sentence_count": sentence_count,
            "document_count": document_count,
        }
