"""
Test script for RunPod DeepSeek OCR connection.
Run this to verify your RunPod endpoint is working correctly.
"""
import base64
import io
import sys
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont


def create_test_image() -> str:
    """Create a simple test image with text."""
    # Create a white image with text
    img = Image.new('RGB', (800, 200), color='white')
    draw = ImageDraw.Draw(img)

    # Draw some test text
    text = "Hello من الذكاء الاصطناعي 12345"
    draw.text((50, 80), text, fill='black')

    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def test_runpod_connection(api_key: str, endpoint_url: str) -> None:
    """Test connection to RunPod endpoint."""
    print("🧪 Testing RunPod Connection...")
    print(f"📡 Endpoint: {endpoint_url}")
    print()

    # Create test image
    print("🖼️  Creating test image...")
    image_b64 = create_test_image()
    print(f"✅ Image created (size: {len(image_b64)} chars)")
    print()

    # Prepare request
    payload = {
        "input": {
            "operation": "ocr",
            "processor": "deepseek",
            "image": image_b64
        }
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Send request
    print("📤 Sending request to RunPod...")
    print("⏳ This may take 10-30 seconds on first request (cold start)...")

    try:
        response = requests.post(
            f"{endpoint_url}/run",
            headers=headers,
            json=payload,
            timeout=300
        )
        response.raise_for_status()
        result = response.json()

        # Check for errors
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return

        # Display results
        print()
        print("=" * 60)
        print("✅ SUCCESS! RunPod connection is working!")
        print("=" * 60)
        print()

        # Extract data
        output = result.get("output", result)
        text = output.get("text", "")
        lines = output.get("lines", [])
        confidence = output.get("confidence", 0.0)
        processing_time = output.get("processing_time", 0.0)

        print(f"📝 Extracted Text:")
        print(f"   {text}")
        print()
        print(f"📊 Statistics:")
        print(f"   Lines detected: {len(lines)}")
        print(f"   Confidence: {confidence:.2%}")
        print(f"   Processing time: {processing_time:.2f}s")
        print()

        if lines:
            print("📍 First line details:")
            first_line = lines[0]
            print(f"   Text: {first_line.get('text', 'N/A')}")
            print(f"   Confidence: {first_line.get('conf', 0.0):.2%}")
            print(f"   Bounding box: {first_line.get('bbox_px', 'N/A')}")

        print()
        print("=" * 60)
        print("🎉 Your RunPod endpoint is ready to use!")
        print("=" * 60)

    except requests.Timeout:
        print("❌ Request timed out. Check:")
        print("   1. Endpoint is deployed and running")
        print("   2. Increase timeout in config")
        print("   3. Check RunPod logs for errors")

    except requests.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        print(f"   Status code: {response.status_code}")
        print(f"   Response: {response.text}")
        print()
        print("   Check:")
        print("   1. API key is correct")
        print("   2. Endpoint URL is correct")
        print("   3. Endpoint is deployed")

    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
        print("   Check:")
        print("   1. Internet connection")
        print("   2. Endpoint URL is correct")
        print("   3. Firewall settings")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def test_with_local_image(api_key: str, endpoint_url: str, image_path: str) -> None:
    """Test with a local image file."""
    print(f"🧪 Testing RunPod with local image: {image_path}")
    print()

    # Load and encode image
    try:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        print(f"✅ Image loaded (size: {len(image_b64)} chars)")
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        return

    # Prepare request
    payload = {
        "input": {
            "operation": "ocr",
            "processor": "deepseek",
            "image": image_b64
        }
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Send request
    print("📤 Sending request to RunPod...")

    try:
        response = requests.post(
            f"{endpoint_url}/run",
            headers=headers,
            json=payload,
            timeout=300
        )
        response.raise_for_status()
        result = response.json()

        # Check for errors
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return

        # Display results
        output = result.get("output", result)
        text = output.get("text", "")
        lines = output.get("lines", [])
        confidence = output.get("confidence", 0.0)
        processing_time = output.get("processing_time", 0.0)

        print()
        print("=" * 60)
        print("✅ OCR RESULTS")
        print("=" * 60)
        print()
        print("📝 Extracted Text:")
        print("-" * 60)
        print(text)
        print("-" * 60)
        print()
        print(f"📊 Statistics:")
        print(f"   Lines detected: {len(lines)}")
        print(f"   Confidence: {confidence:.2%}")
        print(f"   Processing time: {processing_time:.2f}s")
        print()

    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Main function."""
    print()
    print("=" * 60)
    print("    RunPod DeepSeek OCR Connection Test")
    print("=" * 60)
    print()

    # Get credentials
    if len(sys.argv) >= 3:
        api_key = sys.argv[1]
        endpoint_url = sys.argv[2]
        image_path = sys.argv[3] if len(sys.argv) > 3 else None
    else:
        print("💡 Usage:")
        print("   python test_connection.py <API_KEY> <ENDPOINT_URL> [IMAGE_PATH]")
        print()
        print("   Example:")
        print("   python test_connection.py \\")
        print("       'your-api-key' \\")
        print("       'https://api.runpod.ai/v2/your-endpoint-id'")
        print()
        print("   Or with local image:")
        print("   python test_connection.py \\")
        print("       'your-api-key' \\")
        print("       'https://api.runpod.ai/v2/your-endpoint-id' \\")
        print("       'path/to/image.png'")
        print()

        # Try to read from environment or config
        try:
            import os
            api_key = os.getenv("RUNPOD_API_KEY")
            endpoint_url = os.getenv("RUNPOD_ENDPOINT")

            if not api_key or not endpoint_url:
                print("❌ Please provide API_KEY and ENDPOINT_URL")
                print("   Either as arguments or environment variables:")
                print("   export RUNPOD_API_KEY='your-key'")
                print("   export RUNPOD_ENDPOINT='your-endpoint'")
                return
        except Exception:
            print("❌ Please provide API_KEY and ENDPOINT_URL as arguments")
            return

        image_path = None

    # Clean endpoint URL
    endpoint_url = endpoint_url.rstrip("/")

    # Run test
    if image_path:
        test_with_local_image(api_key, endpoint_url, image_path)
    else:
        test_runpod_connection(api_key, endpoint_url)

    print()


if __name__ == "__main__":
    main()
