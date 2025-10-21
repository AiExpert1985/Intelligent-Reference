# main.py
"""Main application with async processing cleanup"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from config import settings
from services.logger_config import setup_logging
from database.session import Base, async_engine
from api.endpoints import router
from services.async_processor import async_processor

setup_logging()
logger = logging.getLogger(settings.LOGGER_NAME)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting application...")

    # Database initialization
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized")
    
    logger.info("Services initialized")
    yield
    
    # Cleanup background tasks on shutdown
    logger.info("Shutting down background processor...")
    async_processor.shutdown()
    
    logger.info("Application shutdown complete")

app = FastAPI(
    title=settings.APP_TITLE,
    lifespan=lifespan
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )