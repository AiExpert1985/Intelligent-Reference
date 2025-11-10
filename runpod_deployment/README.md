# 🚀 RunPod DeepSeek OCR Deployment Guide (Pods)

This guide shows you how to deploy DeepSeek OCR on RunPod Pods (GPU servers) for cost-effective, on-demand GPU processing.

---

## 📋 Why RunPod Pods?

**Pods (Server) vs Serverless:**
- ✅ **Pods**: SSH access, git pull updates (seconds), start/stop as needed, $5-30/month for testing
- ❌ **Serverless**: Requires Docker upload (10+ hours on slow internet), pay per request

**We chose Pods for:**
- Fast updates via git pull
- Start/stop capability for cost savings
- Direct development and testing
- No Docker image uploads needed

---

## 🎯 Quick Start

### 1. Create RunPod Pod

1. Go to [RunPod Console](https://www.runpod.io/console/pods)
2. Click **"Deploy"** → **"Pods"**
3. Choose a GPU (e.g., RTX 4090, A4000)
4. Select a template with:
   - CUDA 11.8+
   - Python 3.10+
   - Git installed
5. **Expose HTTP Ports**: Add `8000` in the HTTP ports field
6. Click **"Deploy"**

### 2. Setup Environment on RunPod

SSH into your pod and run:

```bash
# Install Miniconda
cd /workspace
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/miniconda3
/workspace/miniconda3/bin/conda init bash
source ~/.bashrc

# Accept conda TOS
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create environment
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr

# Clone your repository
cd /workspace
git clone -b YOUR_BRANCH https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# Install PyTorch with CUDA 11.8
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
cd runpod_deployment
pip install -r requirements.txt

# Install flash-attn (takes a few minutes to compile)
pip install flash-attn==2.7.3 --no-build-isolation

# Install hf_transfer for faster model downloads
pip install hf_transfer
```

### 3. Start the API Server

```bash
# Make sure you're in the environment
conda activate deepseek-ocr

# Navigate to deployment directory
cd /workspace/YOUR_REPO/runpod_deployment

# Start server (will download DeepSeek model ~6.7GB on first run)
python api_server.py
```

The server will:
- Download DeepSeek-OCR model from Hugging Face (~6.7GB, one-time)
- Load model on GPU
- Start FastAPI server on port 8000
- Be accessible via RunPod proxy URL

### 4. Access Your API

RunPod automatically creates a proxy URL:
```
https://[YOUR-POD-ID]-8000.proxy.runpod.net
```

Find your Pod ID in the RunPod interface, then access:
- Status: `https://[POD-ID]-8000.proxy.runpod.net/`
- Health: `https://[POD-ID]-8000.proxy.runpod.net/health`
- OCR: `POST https://[POD-ID]-8000.proxy.runpod.net/ocr_base64`

---

## 🔄 Updating Code

When you push changes to GitHub:

```bash
# SSH into RunPod
cd /workspace/YOUR_REPO
git pull

# Restart server
cd runpod_deployment
conda activate deepseek-ocr
python api_server.py
```

**No Docker uploads needed!** Updates take seconds, not hours.

---

## 💰 Cost Management

**Start/Stop Strategy:**
- Keep Pod stopped when not in use
- Start only for testing/demos
- Typical cost: $5-30/month vs $244/month for 24/7

**To stop Pod:**
1. Go to RunPod Console
2. Find your Pod
3. Click **"Stop"**

**To restart:**
1. Click **"Start"**
2. SSH in and run `python api_server.py`

---

## 📡 API Endpoints

### GET /
Returns server status and available endpoints.

### GET /health
Returns health status, GPU availability, and model status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "cuda_available": true,
  "device": "NVIDIA GeForce RTX 4090"
}
```

### POST /ocr
Upload image file for OCR processing.

**Parameters:**
- `file`: Image file (multipart/form-data)
- `prompt_type`: "markdown" | "free" | "custom" (default: "markdown")

### POST /ocr_base64
Send base64-encoded image for OCR processing.

**Request:**
```json
{
  "image": "base64_encoded_image_string",
  "prompt_type": "markdown"
}
```

**Response:**
```json
{
  "success": true,
  "text": "Extracted text...",
  "processing_time": 1.23,
  "prompt_used": "<image>\n<|grounding|>Convert the document to markdown."
}
```

---

## 🔧 Configuration

### Prompt Types

- **`markdown`**: Converts documents to structured markdown (best for documents)
- **`free`**: Simple OCR without layout preservation
- **`custom`**: Provide your own prompt via `custom_prompt` field

### Model Settings

Edit `api_server.py` to customize:
- `base_size`: Base resolution (default: 1024)
- `image_size`: Dynamic resolution (default: 640)
- `crop_mode`: Enable/disable cropping (default: True)

---

## 🐛 Troubleshooting

### Server won't start
```bash
# Check conda environment
conda activate deepseek-ocr

# Check PyTorch and CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check logs
python api_server.py  # Run in foreground to see errors
```

### Port not accessible
1. Go to RunPod Console → Your Pod
2. Click hamburger menu (☰) → "Edit Pod"
3. Add `8000` to "Expose HTTP Ports"
4. Pod will restart

### Model download fails
```bash
# Install hf_transfer for faster downloads
pip install hf_transfer

# Restart server
python api_server.py
```

---

## 📝 Files in This Directory

- `api_server.py` - FastAPI server for DeepSeek OCR
- `requirements.txt` - Python dependencies
- `README.md` - This file

---

## 🔗 Integrating with Your App

Update your local app's `config.py`:

```python
# GPU / Remote compute configuration
USE_REMOTE_GPU: bool = True
RUNPOD_ENDPOINT: Optional[str] = "https://YOUR-POD-ID-8000.proxy.runpod.net"
OCR_ENGINE: str = "deepseek"
```

Your app will automatically send OCR requests to RunPod!

---

## 📚 Additional Resources

- [RunPod Docs](https://docs.runpod.io/)
- [DeepSeek-OCR GitHub](https://github.com/deepseek-ai/DeepSeek-OCR)
- [FastAPI Docs](https://fastapi.tiangolo.com/)

---

## 💡 Tips

1. **Keep Pod running during active development**, stop overnight
2. **Model is cached** - subsequent starts are faster
3. **Monitor RunPod credits** to avoid unexpected charges
4. **Use git branches** for testing changes before production
5. **Consider snapshot/backup** of /workspace for quick recovery

---

**Need help?** Check RunPod Discord or open an issue in your repository.
