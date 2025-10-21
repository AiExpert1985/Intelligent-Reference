# services/async_processor.py
"""Simple async document processor using threading (good for MVP)"""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Coroutine

logger = logging.getLogger(__name__)

class AsyncDocumentProcessor:
    """    
    Background document processor using ThreadPoolExecutor (max 3 concurrent).
    Fire-and-forget task submission. Call shutdown() on app exit.
    """
    
    def __init__(self, max_workers: int = 3):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def submit_task(self, coro: Coroutine) -> None:
        """Submit async task to run in background thread (fire-and-forget)."""
        def run_async():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(coro)
                loop.close()
            except Exception as e:
                logger.exception(f"Background task failed: {e}")
        
        self._executor.submit(run_async)
    
    def shutdown(self):
        """Wait for running tasks to complete, then shutdown executor."""
        self._executor.shutdown(wait=True)

# Global instance
async_processor = AsyncDocumentProcessor(max_workers=3)