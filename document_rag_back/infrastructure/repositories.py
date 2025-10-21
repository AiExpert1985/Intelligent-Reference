# infrastructure/repositories.py
"""Database repository implementations"""
import json
import logging
from typing import List, Optional, Dict, Any, Set
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from core.interfaces import IDocumentRepository, IMessageRepository 
from core.domain import ChunkSearchResult, ProcessedDocument 
from database.session import DocumentEntity, MessageEntity
from config import settings

logger = logging.getLogger(settings.LOGGER_NAME)

class SQLDocumentRepository(IDocumentRepository):
    def __init__(self, session: AsyncSession):
        self.session = session

    def _to_domain(self, db_doc: Optional[DocumentEntity]) -> Optional[ProcessedDocument]:
        """Converts an SQLAlchemy entity to a domain model."""
        if db_doc is None:
            return None
        
        # Prevent accidental mutation of DB entity metadata
        md = (db_doc.meta or {}).copy()
        md.setdefault("timestamp", db_doc.timestamp.isoformat())
        md.setdefault("stored_filename", db_doc.stored_filename)
        
        return ProcessedDocument(
            id=db_doc.id, # type: ignore
            filename=db_doc.filename, # type: ignore
            file_hash=db_doc.file_hash, # type: ignore
            metadata=md
        )

    async def create(self, document_id: str, filename: str, file_hash: str, 
                    stored_filename: str) -> ProcessedDocument:
        db_doc = DocumentEntity(
            id=document_id, 
            filename=filename, 
            file_hash=file_hash, 
            stored_filename=stored_filename
        )
        self.session.add(db_doc)
        await self.session.commit()
        await self.session.refresh(db_doc)
        logger.info(f"Created document {document_id} in database")
        
        result = self._to_domain(db_doc)
        assert result is not None, "Created document should never be None"
        return result

    async def get_by_id(self, document_id: str) -> Optional[ProcessedDocument]:
        db_doc = await self.session.get(DocumentEntity, document_id)
        return self._to_domain(db_doc)

    async def get_by_hash(self, file_hash: str) -> Optional[ProcessedDocument]:
        result = await self.session.execute(
            select(DocumentEntity).where(DocumentEntity.file_hash == file_hash)
        )
        return self._to_domain(result.scalar_one_or_none())
    
    async def update_metadata(self, document_id: str, metadata: Dict[str, Any]) -> bool:
        """Persist metadata on the ORM entity and commit."""
        db_doc = await self.session.get(DocumentEntity, document_id)  # ✅ Get ORM entity
        if not db_doc:
            return False
        db_doc.meta = metadata or {}  # ✅ Modify ORM entity
        await self.session.commit()
        return True

    async def list_all(self) -> List[ProcessedDocument]:
        """List all documents"""
        result = await self.session.execute(
            select(DocumentEntity).order_by(DocumentEntity.timestamp.desc())
        )
        docs = [self._to_domain(doc) for doc in result.scalars().all()]
        return [d for d in docs if d is not None]
    
    async def delete(self, document_id: str) -> bool:
        doc = await self.session.get(DocumentEntity, document_id)
        if not doc:
            return False
        await self.session.delete(doc)
        await self.session.commit()
        return True
    
    async def delete_all(self) -> bool:
        try:
            await self.session.execute(delete(DocumentEntity))
            await self.session.commit()
            return True
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to clear documents: {e}")
            return False
        
    async def exists_bulk(self, document_ids: List[str]) -> Set[str]:
        """
        Check which document IDs exist in database.
        Single query replaces N individual lookups.
        """
        if not document_ids:
            return set()
        
        result = await self.session.execute(
            select(DocumentEntity.id).where(DocumentEntity.id.in_(list(document_ids)))
        )
        return {row[0] for row in result}

class SQLMessageRepository(IMessageRepository):
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def save_search(self, query: str, results_count: int) -> None:
        message = MessageEntity(
            sender="search", 
            content=f"Query: {query} | Results: {results_count}"
        )
        self.session.add(message)
        await self.session.commit()
    
    async def save_search_results(self, query: str, 
                                  results: List[ChunkSearchResult]) -> None:
        """Save query and results using same structure as live chat"""
        user_msg = MessageEntity(sender="user", content=query)
        self.session.add(user_msg)
        
        if not results:
            ai_msg = MessageEntity(sender="ai", content="No relevant information found.")
            self.session.add(ai_msg)
        else:
            for result in results:
                result_json = json.dumps({
                    "document_name": result.chunk.metadata.get('document_name'),
                    "page_number": result.chunk.metadata.get('page'),
                    "content_snippet": result.chunk.content[:300] + "..." 
                        if len(result.chunk.content) > 300 else result.chunk.content,
                    "document_id": result.chunk.document_id,
                    "download_url": f"http://127.0.0.1:8000/download/{result.chunk.document_id}"
                })
                ai_msg = MessageEntity(sender="ai_result", content=result_json)
                self.session.add(ai_msg)
        
        await self.session.commit()

    async def get_search_history(self, limit: int = 10) -> List[Dict]:
        """Get complete chat history"""
        result = await self.session.execute(
            select(MessageEntity)
            .where(MessageEntity.sender.in_(["user", "ai", "ai_result"]))
            .order_by(MessageEntity.timestamp.asc())
            .limit(limit)
        )
        messages = result.scalars().all()
        
        return [
            {
                "sender": msg.sender,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            }
            for msg in messages
        ]
    
    async def clear_history(self) -> bool:
        await self.session.execute(
            delete(MessageEntity).where(
                MessageEntity.sender.in_(["user", "ai", "ai_result"])
            )
        )
        await self.session.commit()
        return True