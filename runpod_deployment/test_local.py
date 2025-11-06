"""
Local test script for handler.py
Tests OCR functionality without RunPod/Docker
"""
import base64
import io
from PIL import Image, ImageDraw, ImageFont

# Import your handler functions
from handler import load_paddleocr_model, parse_paddleocr_result
import numpy as np


def create_test_image():
    """Create a simple test image with text."""
    img = Image.new('RGB', (800, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw test text
    text = "Hello مرحبا 12345"
    draw.text((50, 80), text, fill='black')
    
    return img


def test_ocr_locally():
    """Test OCR functionality locally."""
    print("🧪 Testing PaddleOCR locally...")
    print()
    
    # Step 1: Load model
    print("1️⃣ Loading PaddleOCR model...")
    try:
        model = load_paddleocr_model()
        print("   ✅ Model loaded successfully!")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return
    
    print()
    
    # Step 2: Create test image
    print("2️⃣ Creating test image...")
    test_img = create_test_image()
    print("   ✅ Test image created")
    print()
    
    # Step 3: Run OCR
    print("3️⃣ Running OCR...")
    try:
        # Convert to numpy array (PaddleOCR format)
        img_array = np.array(test_img)
        
        # Run OCR
        result = model.ocr(img_array, cls=True)
        
        # Parse result
        text, lines = parse_paddleocr_result(result, test_img.size)
        
        print("   ✅ OCR completed!")
        print()
        
        # Step 4: Display results
        print("=" * 60)
        print("📝 RESULTS")
        print("=" * 60)
        print()
        print(f"Extracted Text: {text}")
        print(f"Lines detected: {len(lines)}")
        print()
        
        if lines:
            print("First line details:")
            first = lines[0]
            print(f"  Text: {first['text']}")
            print(f"  Confidence: {first['conf']:.2%}")
            print(f"  Bounding box: {first['bbox_px']}")
        
        print()
        print("=" * 60)
        print("✅ Local test PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"   ❌ OCR failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_ocr_locally()