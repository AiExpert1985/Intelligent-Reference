# infrastructure/file_storage.py
import os
import logging
from pathlib import Path
from typing import Optional, Union
from PIL import Image
from typing import Tuple

from fastapi import UploadFile
from core.interfaces import IFileStorage

from config import settings
from infrastructure.image_utils import ImageProcessor

logger = logging.getLogger(settings.LOGGER_NAME)

class LocalFileStorage(IFileStorage):
    """Concrete implementation for storing files on the local disk."""

    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
        # Create the directory if it doesn't exist
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Upload directory ensured at: {self.base_path}")
        except Exception as e:
            logger.error(f"Could not create upload directory at {self.base_path}: {e}")
            raise

    async def save(self, file: UploadFile, filename: str) -> str:
        """Saves a file to the configured upload directory."""
        file_path = self.base_path / filename
        try:
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            logger.info(f"Successfully saved file to {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"Failed to save file to {file_path}: {e}")
            raise

    async def get_path(self, filename: str) -> Optional[str]:
        """Gets the full path of a file if it exists."""
        file_path = self.base_path / filename
        if file_path.exists():
            return str(file_path)
        return None

    async def delete(self, filename: str) -> bool:
        """Deletes a file from the upload directory."""
        try:
            file_path = self.base_path / filename
            if file_path.exists():
                os.unlink(file_path)
                logger.info(f"Successfully deleted file: {file_path}")
                return True
            logger.warning(f"Attempted to delete non-existent file: {file_path}")
            return False
        except Exception as e:
            logger.error(f"Error deleting file {filename}: {e}")
            return False
        
    async def save_page_image(
        self,
        image: Image.Image,
        document_id: str,
        page_number: int
    ) -> Tuple[str, str]:
        """
        Save page image (PNG) + thumbnail (WebP).
        Returns (original_relative_path, thumbnail_relative_path).
        """
        images_dir = self.base_path / "page_images" / document_id
        filename_base = f"page_{page_number:03d}"

        # write both files
        original_abs, thumb_abs = await ImageProcessor.save_page_image_with_thumbnail(
            image=image,
            base_path=images_dir,
            filename_base=filename_base,
        )

        # return relative paths (the rest of the code builds absolute paths from settings.UPLOADS_DIR)
        original_rel = f"page_images/{document_id}/{filename_base}.png"
        thumb_rel    = f"page_images/{document_id}/{filename_base}_thumb.webp"
        return original_rel, thumb_rel