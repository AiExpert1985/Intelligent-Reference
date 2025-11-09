from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
from PIL import Image
import io
import base64
import time
import uvicorn

app = FastAPI(title="DeepSeek OCR API")

# Global vLLM model
llm = None

class Base64ImageRequest(BaseModel):
    image: str
    prompt_type: str = "free"
    custom_prompt: Optional[str] = None

def load_model():
    """Load DeepSeek-OCR with vLLM (production method)"""
    global llm
    print("Loading DeepSeek-OCR with vLLM...")

    llm = LLM(
        model="deepseek-ai/DeepSeek-OCR",
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        logits_processors=[NGramPerReqLogitsProcessor],  # Anti-repetition
    )

    print("Model loaded successfully!")

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": llm is not None,
    }

def get_prompt(prompt_type: str, custom_prompt: Optional[str] = None) -> str:
    """Generate prompt - simple and standard"""
    if prompt_type == "custom" and custom_prompt:
        return f"<image>\n{custom_prompt}"
    # Free OCR for clean text (no bounding boxes)
    return "<image>\nFree OCR."

@app.post("/ocr_base64")
async def ocr_base64_endpoint(request: Base64ImageRequest):
    """OCR endpoint - clean, standard vLLM approach"""

    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start_time = time.time()

        # Decode image
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Get prompt
        prompt = get_prompt(request.prompt_type, request.custom_prompt)

        # Prepare input (standard vLLM format)
        model_input = [{
            "prompt": prompt,
            "multi_modal_data": {"image": image}
        }]

        # Sampling params (production-optimized)
        sampling_params = SamplingParams(
            temperature=0.0,      # Deterministic
            max_tokens=8192,      # Standard for documents
            skip_special_tokens=False,
            extra_args=dict(
                ngram_size=30,
                window_size=90,
                whitelist_token_ids={128821, 128822},  # <td>, </td>
            ),
        )

        print(f"Processing image: {image.size}")

        # Generate - THIS RETURNS TEXT DIRECTLY (no stdout capture needed!)
        outputs = llm.generate(model_input, sampling_params)

        # Extract text - simple and clean
        result_text = outputs[0].outputs[0].text

        processing_time = time.time() - start_time

        print(f"✓ OCR completed: {len(result_text)} characters in {processing_time:.2f}s")

        return JSONResponse({
            "success": True,
            "text": result_text,
            "processing_time": processing_time,
        })

    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
