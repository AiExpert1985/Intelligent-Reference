from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
import uvicorn
from PIL import Image
import io
import base64
import time
import os
import torch
from transformers import AutoModel, AutoTokenizer
import tempfile
import sys
from io import StringIO

model = None
tokenizer = None

def load_model():
    global model, tokenizer
    print("Loading DeepSeek-OCR model...")

    tokenizer = AutoTokenizer.from_pretrained(
        'deepseek-ai/DeepSeek-OCR',
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        'deepseek-ai/DeepSeek-OCR',
        _attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
        use_safetensors=True
    ).eval()

    print("Model loaded successfully with Flash Attention 2!")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(title="DeepSeek OCR API", lifespan=lifespan)

class Base64ImageRequest(BaseModel):
    image: str
    prompt_type: str = "free"
    custom_prompt: Optional[str] = None

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    }

def get_prompt(prompt_type: str, custom_prompt: Optional[str] = None) -> str:
    if custom_prompt:
        return f"<image>\n{custom_prompt}"
    return "<image>\nFree OCR."

@app.post("/ocr_base64")
async def ocr_base64(request: Base64ImageRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start = time.time()

        # Decode image
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save image
            img_path = os.path.join(temp_dir, "input.jpg")
            image.save(img_path)

            # Get prompt
            prompt = get_prompt(request.prompt_type, request.custom_prompt)

            # Capture stdout (model prints to stdout, doesn't return)
            captured = StringIO()
            sys.stdout = captured

            try:
                model.infer(
                    tokenizer,
                    prompt=prompt,
                    image_file=img_path,
                    output_path=temp_dir,
                    base_size=1024,
                    image_size=1024,
                    crop_mode=False,
                    save_results=False,
                    test_compress=False,
                )
            finally:
                sys.stdout = sys.__stdout__

            # Extract text from stdout
            raw_text = captured.getvalue()

            # Clean: remove debug lines
            lines = [
                line.strip()
                for line in raw_text.split('\n')
                if line.strip() and not any(x in line for x in ['===', 'BASE:', 'PATCHES', 'torch.Size'])
            ]

            result = '\n'.join(lines)

        elapsed = time.time() - start
        print(f"OCR completed: {len(result)} chars in {elapsed:.1f}s")

        return JSONResponse({
            "success": True,
            "text": result,
            "processing_time": elapsed,
        })

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
