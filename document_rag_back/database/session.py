# database/session.py

import logging
import uuid
from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Integer, String, select
from sqlalchemy.ext.asyncio import (async_sessionmaker, create_async_engine)
from sqlalchemy.orm import declarative_base, Mapped, mapped_column

from config import settings

from typing import Any, AsyncGenerator, Dict
from sqlalchemy.ext.asyncio import AsyncSession

from contextlib import asynccontextmanager
from typing import AsyncGenerator


# ============= Models =============

logger = logging.getLogger(settings.LOGGER_NAME)

# Setup SQLAlchemy async engine and session maker
async_engine = create_async_engine(
    settings.DATABASE_URL, 
    echo=False,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_recycle=settings.DB_POOL_RECYCLE,
    pool_pre_ping=True  # Check connection health before using
)
AsyncSessionLocal = async_sessionmaker(async_engine, expire_on_commit=False)
Base = declarative_base()

# --- SQLAlchemy Models ---

class MessageEntity(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    sender = Column(String, nullable=False)  # 'user' or 'ai'
    content = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

class DocumentEntity(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False) # The original filename
    file_hash = Column(String, unique=True, index=True, nullable=False)
    # CHANGED: Add a column to store the secure filename on disk
    stored_filename = Column(String, nullable=False, unique=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    # Python attr 'meta' → DB column named 'metadata'
    meta: Mapped[Dict[str, Any]] = mapped_column("metadata", JSON, default=dict)


# ============= Dependencies =============

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Provide database session for FastAPI dependency injection"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# ============= Session Factory =============

@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Create a new database session with proper cleanup.
    
    Used by background processors where request-scoped sessions are unavailable.
    Ensures proper rollback on errors and explicit closure.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()