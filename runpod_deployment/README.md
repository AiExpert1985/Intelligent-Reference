# 🚀 RunPod DeepSeek OCR Deployment Guide

This guide walks you through deploying your DeepSeek OCR service to RunPod serverless.

---

## 📋 Prerequisites

1. **RunPod Account**: Sign up at [runpod.io](https://www.runpod.io/)
2. **Docker Hub Account**: Sign up at [hub.docker.com](https://hub.docker.com/)
3. **Docker Installed**: For building and pushing the image locally

---

## 🔧 Step 1: Prepare the Docker Image

### Option A: Build and Push Locally (Recommended)

1. **Navigate to this directory**:
   ```bash
   cd runpod_deployment
   ```

2. **Build the Docker image**:
   ```bash
   docker build -t your-dockerhub-username/deepseek-ocr:latest .
   ```

3. **Test locally** (optional):
   ```bash
   docker run --gpus all -p 8000:8000 your-dockerhub-username/deepseek-ocr:latest
   ```

4. **Push to Docker Hub**:
   ```bash
   docker login
   docker push your-dockerhub-username/deepseek-ocr:latest
   ```

### Option B: Use RunPod's Built-in Build (Easier)

RunPod can build your image directly from GitHub. Skip to Step 2.

---

## 🌐 Step 2: Create RunPod Serverless Endpoint

### 2.1 Log into RunPod

1. Go to [runpod.io](https://www.runpod.io/)
2. Log in to your account
3. Navigate to **"Serverless"** in the left menu

### 2.2 Create New Endpoint

1. Click **"+ New Endpoint"**
2. Fill in the details:

   **Basic Settings:**
   - **Endpoint Name**: `deepseek-ocr-endpoint`
   - **Select GPU**: Choose based on budget
     - RTX 4090 (cheapest, good performance)
     - RTX A6000 (more VRAM)
     - A100 (best performance, more expensive)

   **Docker Configuration:**
   - **Docker Image**: `your-dockerhub-username/deepseek-ocr:latest`
   - Or use **"Build from GitHub"** and connect your repo

   **Container Configuration:**
   - **Container Disk**: 20 GB (minimum)
   - **Idle Timeout**: 5 seconds (saves money)
   - **Max Workers**: 3 (adjust based on load)

   **Environment Variables** (if needed):
   - Add any custom environment variables here

3. Click **"Deploy"**

### 2.3 Wait for Deployment

- RunPod will pull your Docker image and deploy it
- This takes 2-5 minutes
- You'll see the status change from "Deploying" to "Ready"

---

## 🔑 Step 3: Get Your API Credentials

1. Once deployed, click on your endpoint
2. You'll see:
   - **Endpoint ID**: Copy this
   - **API Key**: Click "Reveal" and copy
   - **Endpoint URL**: Something like `https://api.runpod.ai/v2/your-endpoint-id`

---

## ⚙️ Step 4: Configure Your App

Update your `config.py` or create a `.env` file:

```python
# config.py or .env
USE_REMOTE_GPU = True
RUNPOD_API_KEY = "your-api-key-here"
RUNPOD_ENDPOINT = "https://api.runpod.ai/v2/your-endpoint-id"
RUNPOD_TIMEOUT = 300
OCR_ENGINE = "deepseek"
```

Or use environment variables:

```bash
export USE_REMOTE_GPU=True
export RUNPOD_API_KEY="your-api-key-here"
export RUNPOD_ENDPOINT="https://api.runpod.ai/v2/your-endpoint-id"
export OCR_ENGINE="deepseek"
```

---

## 🧪 Step 5: Test the Connection

### Test with RunPod's UI

1. In RunPod dashboard, click your endpoint
2. Go to **"Requests"** tab
3. Click **"Run"** to send a test request
4. Use this test payload:

```json
{
  "input": {
    "operation": "ocr",
    "processor": "deepseek",
    "image": "<paste-base64-encoded-image-here>"
  }
}
```

### Test from Your App

Run a simple test in Python:

```python
import requests
import base64
from PIL import Image
import io

# Load and encode an image
with open("test_image.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

# Send request to RunPod
response = requests.post(
    "https://api.runpod.ai/v2/your-endpoint-id/run",
    headers={
        "Authorization": "Bearer your-api-key",
        "Content-Type": "application/json"
    },
    json={
        "input": {
            "operation": "ocr",
            "processor": "deepseek",
            "image": image_b64
        }
    },
    timeout=300
)

print(response.json())
```

### Test Your Full App

```bash
cd document_rag_back
python -m main
# Upload a document and test OCR
```

---

## 💰 Cost Optimization Tips

### 1. **Use Idle Timeout**
   - Set to 5 seconds
   - Pods shut down when not in use
   - Start automatically on new requests

### 2. **Choose Right GPU**
   - Start with RTX 4090 ($0.00020/sec)
   - Only upgrade if you need more VRAM

### 3. **Batch Processing**
   - Process multiple images in one request
   - Reduces cold start overhead

### 4. **Set Max Workers**
   - Start with 1-3 workers
   - Scale based on concurrent users

### 5. **Monitor Usage**
   - Check RunPod dashboard for costs
   - Set budget alerts

---

## 📊 Monitoring Your Endpoint

### View Logs

1. Go to RunPod dashboard
2. Click your endpoint
3. Click **"Logs"** tab
4. See real-time logs from your handler

### View Metrics

1. Click **"Analytics"** tab
2. See:
   - Request count
   - Success rate
   - Average processing time
   - Cost per request

---

## 🐛 Troubleshooting

### Endpoint Won't Start

**Check logs** for errors:
- Missing dependencies → Update `requirements.txt`
- GPU not detected → Check CUDA version
- Handler errors → Check `handler.py` syntax

### Slow First Request (Cold Start)

**Normal behavior**:
- First request takes 10-30 seconds (loading model)
- Subsequent requests are fast (~1-2 seconds)
- Use "Min Workers: 1" to keep one pod warm

### Timeout Errors

**Increase timeout**:
- In RunPod: Container Settings → Request Timeout
- In your app: `RUNPOD_TIMEOUT = 600`

### High Costs

**Optimize**:
- Lower idle timeout (5 seconds)
- Reduce max workers
- Choose cheaper GPU
- Consider switching to local GPU for high volume

---

## 🔄 Alternative: Secure Cloud Pod (Always Running)

If you have **constant high volume**, rent a dedicated pod:

### Setup:

1. Go to **"Secure Cloud"** in RunPod
2. Click **"+ GPU Cloud"**
3. Select GPU and rent by the hour
4. SSH into the pod:
   ```bash
   ssh root@pod-ip-address
   ```
5. Install dependencies and run a FastAPI server:
   ```bash
   pip install fastapi uvicorn deepseek-ocr
   python api_server.py
   ```
6. Expose port 8000
7. Use the pod's public IP as your endpoint

**Cost**: ~$0.30-$2.00/hour (depending on GPU)

---

## 📝 Summary

**What you did:**
1. ✅ Created Docker image with DeepSeek OCR
2. ✅ Deployed to RunPod serverless
3. ✅ Got API credentials
4. ✅ Configured your app
5. ✅ Tested the connection

**Your app is now using remote GPU for OCR!** 🎉

When you're ready to buy a local GPU, just change:
```python
USE_REMOTE_GPU = False
```

---

## 🆘 Need Help?

- **RunPod Docs**: [docs.runpod.io](https://docs.runpod.io/)
- **RunPod Discord**: [discord.gg/runpod](https://discord.gg/runpod)
- **Support**: support@runpod.io

---

## 📈 Estimated Costs

Assuming **RTX 4090** at $0.00020/sec:

| Usage | Time/Request | Requests/Day | Cost/Day | Cost/Month |
|-------|--------------|--------------|----------|------------|
| Light | 2 seconds | 50 | $0.02 | $0.60 |
| Medium | 2 seconds | 500 | $0.20 | $6.00 |
| Heavy | 2 seconds | 5000 | $2.00 | $60.00 |

**Cold starts** add 10-30 seconds for the first request after idle period.

With idle timeout, you only pay when actually processing!
