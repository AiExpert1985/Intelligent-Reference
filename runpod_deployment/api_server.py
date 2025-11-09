from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
from PIL import Image
import io
import base64
import time
import os
import torch
from transformers import AutoModel, AutoTokenizer
import tempfile
import shutil

app = FastAPI(title="DeepSeek OCR API")

# Global model and tokenizer
model = None
tokenizer = None

class Base64ImageRequest(BaseModel):
    image: str  # base64 encoded image
    prompt_type: str = "markdown"  # "markdown", "free", or "custom"
    custom_prompt: Optional[str] = None

def load_model():
    """Load DeepSeek-OCR model and tokenizer"""
    global model, tokenizer

    print("Loading DeepSeek-OCR model...")
    model_name = 'deepseek-ai/DeepSeek-OCR'

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        _attn_implementation='flash_attention_2',
        trust_remote_code=True,
        use_safetensors=True
    )
    model = model.eval().cuda().to(torch.bfloat16)

    print("Model loaded successfully!")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
async def root():
    return {
        "status": "DeepSeek OCR API is running",
        "endpoints": ["/ocr", "/ocr_base64", "/health"],
        "model": "deepseek-ai/DeepSeek-OCR"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    }

def get_prompt(prompt_type: str, custom_prompt: Optional[str] = None) -> str:
    """Generate prompt based on type

    Using OFFICIAL DeepSeek-OCR prompt formats from documentation
    """
    if prompt_type == "custom" and custom_prompt:
        return f"<image>\n{custom_prompt}"
    elif prompt_type == "markdown":
        # Official prompt for document to markdown conversion
        return "<image>\n<|grounding|>Convert the document to markdown."
    elif prompt_type == "free":
        # Official prompt for layout-free OCR
        return "<image>\nFree OCR."
    else:
        # Default to markdown for documents
        return "<image>\n<|grounding|>Convert the document to markdown."

@app.post("/ocr")
async def ocr_endpoint(
    file: UploadFile = File(...),
    prompt_type: str = "markdown"
):
    """
    OCR endpoint that accepts image file upload

    Args:
        file: Image file (jpg, png, etc.)
        prompt_type: Type of prompt - "markdown", "free", or "custom"
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start_time = time.time()

        # Create temporary directory for this request
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded file temporarily
            temp_image_path = os.path.join(temp_dir, "input_image.jpg")
            with open(temp_image_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Get prompt
            prompt = get_prompt(prompt_type)

            # Run OCR inference
            result = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=temp_image_path,
                output_path=temp_dir,
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=False,  # Don't save intermediate results
                test_compress=True
            )

        processing_time = time.time() - start_time

        return JSONResponse({
            "success": True,
            "text": result,
            "processing_time": processing_time,
            "prompt_used": prompt
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

@app.post("/ocr_base64")
async def ocr_base64_endpoint(request: Base64ImageRequest):
    """
    OCR endpoint that accepts base64 encoded image

    Args:
        request: JSON with base64 image and optional prompt_type
    """
    print("=" * 80)
    print("🔔 NEW OCR REQUEST RECEIVED")
    print(f"📋 Prompt type: {request.prompt_type}")
    print(f"📦 Image size (base64): {len(request.image) / 1024:.1f} KB")
    print("=" * 80)

    if model is None:
        print("❌ ERROR: Model not loaded!")
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start_time = time.time()

        # Decode base64 image
        print("🔓 Decoding base64 image...")
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        print(f"🖼️  Image decoded: {image.size} pixels, mode={image.mode}")

        # Create temporary directory for this request
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save image temporarily
            temp_image_path = os.path.join(temp_dir, "input_image.jpg")
            image.save(temp_image_path)
            print(f"💾 Image saved to: {temp_image_path}")

            # Get prompt
            prompt = get_prompt(request.prompt_type, request.custom_prompt)
            print(f"📝 Using prompt: {prompt[:100]}...")

            # Run OCR inference
            # Using "Base" mode (1024x1024, crop_mode=False) for balance of speed and quality
            # This avoids the multi-crop overhead of "Gundam" mode while maintaining accuracy
            print("Starting DeepSeek OCR inference...")
            result = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=temp_image_path,
                output_path=temp_dir,
                base_size=1024,  # Base mode: 1024x1024 (256 vision tokens)
                image_size=1024,  # Match base_size for single resolution
                crop_mode=False,  # Single pass processing (faster than Gundam mode)
                save_results=True,  # MUST be True - infer() saves to files, doesn't return text!
                test_compress=False,
            )
            print("OCR inference completed!")
            print(f"DEBUG: infer() return value type: {type(result)}")
            print(f"DEBUG: infer() return value: {repr(result)[:200] if result else 'None'}")

            # The infer() method saves results to files instead of returning text
            # Look for common output file patterns
            import glob
            output_files = glob.glob(f"{temp_dir}/*")
            print(f"DEBUG: Files in output dir: {output_files}")

            # Read the OCR result from saved files
            text_result = None

            # Try common DeepSeek output file patterns
            for pattern in ['result.mmd', 'result_ori.mmd', 'result.txt', 'images']:
                file_path = os.path.join(temp_dir, pattern)
                if os.path.exists(file_path) and os.path.isfile(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            if content and len(content) > 0:
                                text_result = content
                                print(f"✓ Found OCR result in {pattern}: {len(content)} characters")
                                break
                    except Exception as e:
                        print(f"  ! Error reading {pattern}: {e}")

            # If no result found in known files, list all files and try to read them
            if not text_result:
                print("WARNING: No result in expected files. Checking all output files...")
                for fpath in output_files:
                    fname = os.path.basename(fpath)
                    fsize = os.path.getsize(fpath) if os.path.isfile(fpath) else 0
                    print(f"  - {fname} ({fsize} bytes)")

                    # Skip input image
                    if fname == 'input_image.jpg':
                        continue

                    # Try reading any text files
                    if os.path.isfile(fpath) and fsize > 0:
                        try:
                            with open(fpath, 'r', encoding='utf-8') as f:
                                content = f.read().strip()
                                if content and len(content) > 0:
                                    text_result = content
                                    print(f"✓ Found text in {fname}: {len(content)} characters")
                                    break
                        except Exception as e:
                            print(f"  ! Could not read {fname} as text: {e}")

            # Use the result from infer() as fallback (if it returned something)
            if not text_result and result:
                text_result = str(result)
                print("Using infer() return value as result")

            # Final result
            result = text_result or ""

        processing_time = time.time() - start_time
        result_length = len(result) if result else 0

        print(f"Processing time: {processing_time:.2f}s")
        print(f"Result length: {result_length} characters")
        if result_length > 0:
            print(f"First 100 chars: {result[:100]}")
        else:
            print("WARNING: DeepSeek returned EMPTY or None result!")
            print(f"Result value: {result}")
        print("=" * 80)

        return JSONResponse({
            "success": True,
            "text": result,
            "processing_time": processing_time,
            "prompt_used": prompt
        })

    except Exception as e:
        print(f"❌ ERROR during OCR processing: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        print("=" * 80)
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

if __name__ == "__main__":
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000)
