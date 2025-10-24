# api/endpoints.py
"""
API endpoints for document search system.

SECURITY NOTE (Phase 0 - MVP):
==================================
These endpoints currently have NO AUTHENTICATION for pilot testing.
This is an intentional decision documented in SECURITY_POLICY.md.

Before production deployment:
1. Add authentication middleware
2. Implement role-based access control
3. Add document-level permissions
4. Enable audit logging
==================================
"""

import asyncio
from fastapi import APIRouter, HTTPException, Response, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from pathlib import Path
from typing import Dict, Any, List, Tuple

from database.session import get_db
from core.domain import PageSearchResult
from infrastructure.repositories import SQLDocumentRepository
from utils.highlight_token import verify
from infrastructure.image_utils import ImageHighlighter
from config import settings
import os
from pathlib import Path
from typing import List, Dict, Literal

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request
from fastapi.responses import FileResponse

from config import settings
from core.interfaces import IMessageRepository, IRAGService
from services.factory import get_message_repository, get_rag_service
from api.schemas import (
    ProcessDocumentResponse,
    SearchResponse,
    StatusResponse,
    DocumentsListResponse,
    DeleteResponse,
    ChatRequest,
    PageSearchResponse,
    PageSearchResultItem,
    ProcessingProgress,
)
from utils.common import _safe_file_path, validate_document_id
from infrastructure.progress_store import progress_store
import logging

logger = logging.getLogger(settings.LOGGER_NAME)

router = APIRouter()


# ---------- Upload ----------
@router.post("/upload-document", response_model=ProcessDocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    rag_service: IRAGService = Depends(get_rag_service),
) -> ProcessDocumentResponse:
    return await rag_service.process_document(file)


# ---------- Search (CHUNKS) - keep existing behavior for LLM/compat ----------
@router.post("/search", response_model=SearchResponse)
async def search_endpoint(
    request: Request,
    chat_request: ChatRequest,
    rag_service: IRAGService = Depends(get_rag_service),
) -> SearchResponse:
    if not 3 <= len(chat_request.question) <= 2000:
        raise HTTPException(
            status_code=422, detail="Search query must be between 3 and 2000 characters"
        )

    status = await rag_service.get_status()
    if not status["ready_for_queries"]:
        raise HTTPException(status_code=400, detail="Please upload a document first")

    # IMPORTANT: Use chunk-level search to preserve old behavior
    results = await rag_service.search_chunks(
        chat_request.question, top_k=settings.DEFAULT_SEARCH_RESULTS
    )

    base_url = str(request.base_url).rstrip("/")
    search_results = []

    for result in results:
        content = result.chunk.content
        snippet = (
            content[: settings.SNIPPET_LENGTH] + "..."
            if len(content) > settings.SNIPPET_LENGTH
            else content
        )
        search_results.append(
            {
                "document_name": result.chunk.metadata.get("document_name", "Unknown"),
                "page_number": result.chunk.metadata.get("page", 0),
                "content_snippet": snippet,
                "document_id": result.chunk.document_id,
                "download_url": f"{base_url}/download/{result.chunk.document_id}",
            }
        )

    return SearchResponse(
        status="success",
        query=chat_request.question,
        results=search_results,
        total_results=len(search_results),
    )


# ---------- Search (PAGES) - new endpoint for UI with images/highlights ----------
@router.post("/search-pages", response_model=PageSearchResponse)
async def search_pages_endpoint(
    request: Request,
    chat_request: ChatRequest,
    rag_service: IRAGService = Depends(get_rag_service),
) -> PageSearchResponse:
    if not 3 <= len(chat_request.question) <= 2000:
        raise HTTPException(
            status_code=422, detail="Search query must be between 3 and 2000 characters"
        )

    status = await rag_service.get_status()
    if not status["ready_for_queries"]:
        raise HTTPException(status_code=400, detail="Please upload a document first")

    pages: List[PageSearchResult] = await rag_service.search_pages(chat_request.question, top_k=settings.TOP_K)

    print(f" The number of found pages = {len(pages)}")

    base_url = str(request.base_url).rstrip("/")
    results: List[PageSearchResultItem] = []

    for p in pages:
        item = PageSearchResultItem(
            document_id=p.document_id,
            document_name=p.document_name,
            page_number=p.page_number,
            score=round(p.score, 3),
            chunk_count=p.chunk_count,
            image_url=(base_url + p.image_url) if p.image_url else "",
            thumbnail_url=(base_url + p.thumbnail_url) if p.thumbnail_url else "",
            highlights=p.highlights,
            download_url=(base_url + p.download_url) if p.download_url else "",
        )

        # Attach highlight token from rag_service
        if hasattr(p, 'highlight_token') and p.highlight_token:
            item.highlight_token = p.highlight_token

        results.append(item)

    return PageSearchResponse(
        status="success",
        query=chat_request.question,
        results=results,
        total_results=len(results),
    )


# ---------- Delete one document (keep) ----------
@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(
    document_id: str, rag_service: IRAGService = Depends(get_rag_service)
) -> DeleteResponse:
    if not validate_document_id(document_id):
        raise HTTPException(status_code=422, detail="Invalid document ID format")

    success = await rag_service.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")

    return DeleteResponse(status="success", message="Document deleted successfully")


# ---------- Clear all documents (keep) ----------
@router.delete("/documents", response_model=DeleteResponse)
async def clear_all_documents(
    rag_service: IRAGService = Depends(get_rag_service),
) -> DeleteResponse:
    await rag_service.clear_all()
    return DeleteResponse(status="success", message="All documents cleared successfully")


# ---------- List documents (keep) ----------
@router.get("/documents", response_model=DocumentsListResponse)
async def list_documents(
    request: Request, rag_service: IRAGService = Depends(get_rag_service)
) -> DocumentsListResponse:
    documents = await rag_service.list_documents(request)
    return DocumentsListResponse(documents=documents)


# ---------- Service status (keep) ----------
@router.get("/status", response_model=StatusResponse)
async def get_status(rag_service: IRAGService = Depends(get_rag_service)) -> StatusResponse:
    status = await rag_service.get_status()
    return StatusResponse(**status)


# ---------- Search history (keep) ----------
@router.get("/search-history", response_model=List[Dict])
async def get_search_history(
    message_repo: IMessageRepository = Depends(get_message_repository),
) -> List[Dict]:
    return await message_repo.get_search_history(limit=50)


@router.delete("/search-history", response_model=DeleteResponse)
async def clear_search_history(
    message_repo: IMessageRepository = Depends(get_message_repository),
) -> DeleteResponse:
    success = await message_repo.clear_history()
    if success:
        return DeleteResponse(status="success", message="Search history cleared successfully")
    raise HTTPException(status_code=500, detail="Failed to clear search history")


# ---------- Config (keep) ----------
@router.get("/config")
async def get_config():
    return {
        "allowed_extensions": settings.ALLOWED_FILE_EXTENSIONS,
        "max_file_size_mb": settings.MAX_FILE_SIZE // (1024 * 1024),
    }


# ---------- Processing progress (keep) ----------
@router.get("/processing-status/{document_id}", response_model=ProcessingProgress)
async def get_processing_status(document_id: str) -> ProcessingProgress:
    """Get real-time processing progress for a document"""
    progress = progress_store.get(document_id)
    if not progress:
        raise HTTPException(status_code=404, detail="No processing status found for this document")

    return ProcessingProgress(document_id=document_id, **progress)


# ---------- Health Check ----------
@router.get("/health")
async def health_check(rag_service: IRAGService = Depends(get_rag_service)):
    """
    System health check endpoint.
    
    Returns:
        - status: healthy/unhealthy
        - chunks_indexed: Number of chunks in vector store
        - reranker_loaded: Whether reranker model is available
        - ocr_engine: Currently configured OCR backend
        - timestamp: When check was performed
    """
    try:
        from datetime import datetime
        
        # Check vector store
        count = await rag_service.vector_store.count()
        
        # Check reranker availability
        reranker_loaded = False
        if rag_service.reranker:
            reranker_loaded = bool(getattr(rag_service.reranker, "_model", None))
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "chunks_indexed": count,
            "reranker_loaded": reranker_loaded,
            "ocr_engine": settings.OCR_ENGINE,
            "vector_store": settings.VECTOR_STORE_TYPE,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "error": str(e)
        }
    


@router.get("/page-image/by-token/{token}")
async def get_highlighted_image(
    token: str, 
    db: AsyncSession = Depends(get_db)
):
    if not settings.ENABLE_HIGHLIGHT_PREVIEW:
        raise HTTPException(status_code=404, detail="Disabled")
    
    try:
        payload = verify(token)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    doc_id = payload.get("doc_id")
    page_index = int(payload.get("page_index", -1))
    if not doc_id or page_index < 0:
        raise HTTPException(status_code=400, detail="Invalid token")
    
    # Load doc + image
    repo = SQLDocumentRepository(db)
    doc = await repo.get_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    meta = doc.metadata or {}
    page_images = meta.get("page_image_paths", {})

    # ✅ TYPE-TOLERANT LOOKUP: Try str(page_index) then page_index
    rel_path = page_images.get(str(page_index))
    if rel_path is None:
        rel_path = page_images.get(page_index)

    if not rel_path:
        logger.warning(
            f"[IMG] Page image not found: doc={doc_id} page={page_index}. "
            f"Available keys={list(page_images.keys())[:5]}"
        )
        raise HTTPException(status_code=404, detail="Page image not found")

    image_path = Path(settings.UPLOADS_DIR) / rel_path
    if not image_path.exists():
        logger.error(f"[IMG] File missing on disk: {image_path}")
        raise HTTPException(status_code=404, detail="Image file missing")

    # Extract line-based highlights only
    line_ids = payload.get("line_ids") or []

    # ✅ TYPE-TOLERANT METADATA LOOKUP
    lines_dict = meta.get("lines") or {}
    page_lines = lines_dict.get(str(page_index), []) or lines_dict.get(page_index, [])

    bboxes = []
    if line_ids and page_lines:
        lookup = {
            ln.get("line_id"): ln.get("bbox")
            for ln in page_lines
            if ln.get("line_id") and ln.get("bbox")
        }
        for lid in line_ids:
            bbox = lookup.get(lid)
            if bbox:
                bboxes.append(tuple(float(x) for x in bbox))

    logger.info(
        f"[IMG] doc={doc_id} page={page_index} line_ids={len(line_ids)} matched_bboxes={len(bboxes)}"
    )

    if not bboxes:
        return Response(content=image_path.read_bytes(), media_type="image/png")

    img_bytes = await asyncio.to_thread(
        ImageHighlighter.draw_highlights,
        str(image_path),
        bboxes,
        None,
        settings.HIGHLIGHT_MAX_REGIONS,
        settings.HIGHLIGHT_TIMEOUT_SEC,
        "WEBP",
    )

    return Response(content=img_bytes, media_type="image/webp")


# ---------- Serve page image (original or thumbnail) ----------
@router.get("/page-image/{document_id}/{page_number}")
async def get_page_image(
    document_id: str,
    page_number: int,
    size: Literal["original", "thumbnail"] = "original",
    rag_service: IRAGService = Depends(get_rag_service),
):
    if not validate_document_id(document_id):
        raise HTTPException(status_code=422, detail="Invalid document ID format")

    # Confirm doc exists using service API (repo is encapsulated)
    doc = await rag_service.get_document_with_path(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    suffix = "_thumb.webp" if size == "thumbnail" else ".png"
    rel = f"page_images/{document_id}/page_{page_number:03d}{suffix}"
    full = _safe_file_path(settings.UPLOADS_DIR, rel)

    if not full.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    media_type = "image/webp" if size == "thumbnail" else "image/png"
    return FileResponse(
        path=str(full),
        media_type=media_type,
        filename=Path(rel).name,
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )

# ---------- Download original document (uses safe path) ----------
@router.get("/download/{document_id}")
async def download_document(
    document_id: str, rag_service: IRAGService = Depends(get_rag_service)
):
    if not validate_document_id(document_id):
        raise HTTPException(status_code=422, detail="Invalid document ID format")

    doc_details = await rag_service.get_document_with_path(document_id)
    if not doc_details:
        raise HTTPException(status_code=404, detail="Document not found")

    full = _safe_file_path(settings.UPLOADS_DIR, doc_details["path"])
    if not full.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=str(full),
        filename=doc_details["original_filename"],
        media_type="application/octet-stream",
    )
