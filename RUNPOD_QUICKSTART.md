# 🚀 RunPod Quick Start Guide

This is a quick reference for connecting your app to RunPod. For detailed instructions, see `runpod_deployment/README.md`.

---

## 📋 Quick Overview

Your app now has **modular GPU support** that can switch between:
- **Local GPU** - Run on your own hardware
- **Remote GPU** - Send work to RunPod (or any cloud provider)

---

## ⚡ 5-Minute Setup

### Step 1: Create RunPod Account
1. Go to [runpod.io](https://www.runpod.io/) and sign up
2. Add billing info (get free credits)

### Step 2: Deploy Your Service
1. Navigate to **"Serverless"** → **"+ New Endpoint"**
2. Configure:
   - **Name**: `deepseek-ocr`
   - **GPU**: RTX 4090 (cheapest, good performance)
   - **Docker Image**: `your-dockerhub-username/deepseek-ocr:latest`
   - **Container Disk**: 20 GB
   - **Idle Timeout**: 5 seconds
3. Click **"Deploy"** and wait 2-5 minutes

### Step 3: Get Credentials
After deployment completes:
- Copy **Endpoint URL**: `https://api.runpod.ai/v2/your-endpoint-id`
- Copy **API Key**: Click "Reveal" to see it

### Step 4: Configure Your App
Edit `document_rag_back/config.py` or create `.env`:

```python
USE_REMOTE_GPU = True
RUNPOD_API_KEY = "your-api-key-here"
RUNPOD_ENDPOINT = "https://api.runpod.ai/v2/your-endpoint-id"
OCR_ENGINE = "deepseek"
```

### Step 5: Test Connection
```bash
cd runpod_deployment
python test_connection.py "your-api-key" "https://api.runpod.ai/v2/your-endpoint-id"
```

**That's it!** Your app now uses RunPod for OCR. 🎉

---

## 📁 Files You Need

All RunPod deployment files are in `runpod_deployment/`:

```
runpod_deployment/
├── README.md           ← Full deployment guide
├── handler.py          ← RunPod serverless handler
├── Dockerfile          ← Docker image configuration
├── requirements.txt    ← Python dependencies
├── test_connection.py  ← Connection test script
└── .dockerignore       ← Docker build exclusions
```

---

## 🔄 Building Your Docker Image

### Option A: Local Build (Recommended)

```bash
cd runpod_deployment

# Build
docker build -t your-dockerhub-username/deepseek-ocr:latest .

# Push to Docker Hub
docker login
docker push your-dockerhub-username/deepseek-ocr:latest
```

### Option B: RunPod Auto-Build

1. Push files to GitHub
2. In RunPod, select "Build from GitHub"
3. Connect your repo
4. RunPod builds automatically

---

## 🎛️ Switching Between Local and Remote

Your app is designed to switch seamlessly:

### Use Local GPU:
```python
USE_REMOTE_GPU = False
LOCAL_GPU_DEVICE = "cuda"  # or "cpu"
```

### Use Remote GPU (RunPod):
```python
USE_REMOTE_GPU = True
RUNPOD_API_KEY = "your-key"
RUNPOD_ENDPOINT = "your-endpoint"
```

**No code changes needed!** Just flip the config flag.

---

## 💰 Estimated Costs

RunPod serverless with **RTX 4090** ($0.00020/sec):

| Daily OCR Tasks | Cost/Day | Cost/Month |
|-----------------|----------|------------|
| 50 documents    | $0.02    | $0.60      |
| 500 documents   | $0.20    | $6.00      |
| 5000 documents  | $2.00    | $60.00     |

**You only pay when processing!** Idle pods cost nothing.

---

## 🧪 Testing Your Setup

### Test 1: Basic Connection
```bash
cd runpod_deployment
python test_connection.py "$RUNPOD_API_KEY" "$RUNPOD_ENDPOINT"
```

### Test 2: With Your Own Image
```bash
python test_connection.py "$RUNPOD_API_KEY" "$RUNPOD_ENDPOINT" "/path/to/image.png"
```

### Test 3: Full App Test
```bash
cd document_rag_back
python -m main
# Upload a document through your app
```

---

## 🐛 Common Issues

### ❌ "Endpoint not found"
- Check endpoint URL is correct
- Ensure endpoint is deployed (not stopped)

### ❌ "Authentication failed"
- Check API key is correct
- Ensure no extra spaces in key

### ❌ "First request takes 30 seconds"
- **Normal!** This is cold start (loading model)
- Subsequent requests are ~1-2 seconds
- Set "Min Workers: 1" to keep pod warm

### ❌ "DeepSeek not available"
- Install: `pip install deepseek-ocr`
- Or update `requirements.txt` in Docker image

---

## 📊 Monitoring

View real-time metrics in RunPod dashboard:
- Request count
- Success rate
- Processing time
- Cost per request
- Logs and errors

---

## 🎯 Key Features

✅ **Provider-agnostic** - Easy to switch from RunPod to AWS/Azure/etc.
✅ **Modular** - Swap between local and remote with one config flag
✅ **Cost-effective** - Pay only for actual processing time
✅ **Auto-scaling** - Handles traffic spikes automatically
✅ **Fallback support** - Falls back to PaddleOCR if DeepSeek unavailable

---

## 📚 More Information

- **Full Guide**: `runpod_deployment/README.md`
- **RunPod Docs**: [docs.runpod.io](https://docs.runpod.io/)
- **Support**: [discord.gg/runpod](https://discord.gg/runpod)

---

## 🎉 You're Ready!

Your app is now configured for both **local and remote GPU execution**. Start with RunPod to test with good image quality, then decide if you want to invest in a local GPU based on your usage patterns.

**Happy OCR-ing!** 📄✨
