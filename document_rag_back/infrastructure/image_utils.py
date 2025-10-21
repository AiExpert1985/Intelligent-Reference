# infrastructure/image_utils.py
"""
Image processing utilities for thumbnails and optimization.
"""
import asyncio
import logging
from pathlib import Path

from PIL import Image

from config import settings

from typing import List, Tuple, Optional
from PIL import Image, ImageDraw
import io, time
from config import settings

logger = logging.getLogger(getattr(settings, "LOGGER_NAME", __name__))


class ImageProcessor:
    """Handle image optimization and thumbnail generation."""

    # Thumbnail constraints
    THUMB_MAX_DIMENSION = 1024  # Max width or height
    THUMB_QUALITY = 85

    # Original image settings
    ORIGINAL_COMPRESS_LEVEL = 6

    @staticmethod
    async def save_page_image_with_thumbnail(
        image: Image.Image,
        base_path: Path,
        filename_base: str
    ) -> Tuple[str, str]:
        """
        Save both original and thumbnail versions of page image.

        Args:
            image: PIL Image object
            base_path: Directory to save images
            filename_base: Base filename (e.g., "page_001")

        Returns:
            Tuple of (original_abs_path, thumbnail_abs_path)
        """
        base_path.mkdir(parents=True, exist_ok=True)

        original_path = base_path / f"{filename_base}.png"
        thumb_path = base_path / f"{filename_base}_thumb.webp"

        # Save original (PNG for fidelity)
        await asyncio.to_thread(
            image.save,
            str(original_path),
            format="PNG",
            optimize=True,
            compress_level=ImageProcessor.ORIGINAL_COMPRESS_LEVEL,
        )

        # Create and save thumbnail (WebP for size)
        thumbnail = await ImageProcessor._create_thumbnail(image)
        await asyncio.to_thread(
            thumbnail.save,
            str(thumb_path),
            format="WEBP",
            quality=ImageProcessor.THUMB_QUALITY,
            method=6,  # Better compression
        )

        try:
            orig_kb = original_path.stat().st_size / 1024
            thumb_kb = thumb_path.stat().st_size / 1024
            logger.info(
                f"[IMAGE] {filename_base}: original={orig_kb:.0f}KB, "
                f"thumbnail={thumb_kb:.0f}KB (saved {orig_kb - thumb_kb:.0f}KB)"
            )
        except Exception as e:
            logger.debug(f"[IMAGE] stat() failed for {filename_base}: {e}")

        return str(original_path), str(thumb_path)

    @staticmethod
    async def _create_thumbnail(image: Image.Image) -> Image.Image:
        """
        Create thumbnail respecting aspect ratio.
        Scales image so longest dimension = THUMB_MAX_DIMENSION
        """
        def resize():
            width, height = image.size
            if width == 0 or height == 0:
                return image.copy()

            if width > height:
                new_w = ImageProcessor.THUMB_MAX_DIMENSION
                new_h = int(height * (new_w / width))
            else:
                new_h = ImageProcessor.THUMB_MAX_DIMENSION
                new_w = int(width * (new_h / height))

            thumb = image.copy()
            thumb.thumbnail((new_w, new_h), Image.Resampling.LANCZOS)
            return thumb

        return await asyncio.to_thread(resize)
    

BBox = Tuple[float,float,float,float]  # normalized [x,y,w,h]

class ImageHighlighter:
    # @staticmethod
    # def _merge_overlaps(bxs: List[BBox], thr: float = 0.10) -> List[BBox]:
    #     merged: List[BBox] = []
    #     def iou(a,b):
    #         ax,ay,aw,ah=a; bx,by,bw,bh=b
    #         ax2,ay2, bx2,by2 = ax+aw,ay+ah, bx+bw,by+bh
    #         ix1,iy1=max(ax,bx),max(ay,by); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    #         if ix2<=ix1 or iy2<=iy1: return 0.0
    #         inter=(ix2-ix1)*(iy2-iy1); area=aw*ah + bw*bh - inter
    #         return inter/area if area>0 else 0.0
    #     for b in bxs:
    #         hit=False
    #         for i,m in enumerate(merged):
    #             if iou(b,m)>=thr:
    #                 ax,ay,aw,ah=b; bx,by,bw,bh=m
    #                 x1,y1=min(ax,bx),min(ay,by); x2=max(ax+aw,bx+bw); y2=max(ay+ah,by+bh)
    #                 merged[i]=(x1,y1,x2-x1,y2-y1); hit=True; break
    #         if not hit: merged.append(b)
    #     return merged

    @staticmethod
    def draw_highlights(image_path: str, normalized_bboxes: List[BBox],
                        style_id: Optional[str]=None, max_regions: Optional[int]=None,
                        timeout_sec: Optional[int]=None, fmt: str="WEBP") -> bytes:
        fill_rgb=(255,235,59); outline_rgb=(255,193,7); fill_a=80; outline_a=160; width=3
        t0=time.time()
        img=Image.open(image_path).convert("RGBA"); W,H=img.size
        boxes = normalized_bboxes[: (max_regions or settings.HIGHLIGHT_MAX_REGIONS)]
        overlay=Image.new("RGBA", img.size, (0,0,0,0)); draw=ImageDraw.Draw(overlay, "RGBA")
        for x,y,w,h in boxes:
            if timeout_sec and (time.time()-t0)>timeout_sec: break
            px,py,pw,ph=int(x*W),int(y*H),max(1,int(w*W)),max(1,int(h*H))
            draw.rectangle([(px,py),(px+pw,py+ph)],
                           fill=(*fill_rgb,fill_a), outline=(*outline_rgb,outline_a), width=width)
        out=Image.alpha_composite(img, overlay).convert("RGB")
        buf=io.BytesIO()
        out.save(buf, "PNG" if fmt.upper()=="PNG" else "WEBP", **({"quality":90,"method":6} if fmt.upper()!="PNG" else {"optimize":True}))
        return buf.getvalue()
