#!/usr/bin/env python3
"""
Preload all models and cache locally for offline demo operation.
Run this ONCE before the demo to cache everything.

Usage:
    python3 setup_models.py
"""

import os
import sys
import torch
from pathlib import Path
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from ultralytics import YOLO
from paddleocr import PaddleOCR
import shutil

# Setup paths
PROJECT_DIR = Path(__file__).parent
MODELS_CACHE = PROJECT_DIR / ".models_cache"
HF_CACHE = PROJECT_DIR / ".huggingface_cache"


def setup_directories():
    """Create cache directories."""
    MODELS_CACHE.mkdir(exist_ok=True)
    HF_CACHE.mkdir(exist_ok=True)
    print(f"✅ Created cache directories")


def preload_qwen_model():
    """Preload Qwen2-VL model and processor."""
    print("\n📥 Preloading Qwen2-VL-2B-Instruct...")
    print("   This may take 2-3 minutes on first run...")

    os.environ['HF_HOME'] = str(HF_CACHE)

    try:
        model_name = "Qwen/Qwen2-VL-2B-Instruct"

        # Download model
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            cache_dir=str(HF_CACHE)
        )
        processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=str(HF_CACHE)
        )

        # Save to local cache
        local_model_path = MODELS_CACHE / "qwen2vl_2b"
        model.save_pretrained(local_model_path)
        processor.save_pretrained(local_model_path)

        print(f"✅ Qwen2-VL cached at: {local_model_path}")
        del model, processor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    except Exception as e:
        print(f"❌ Failed to preload Qwen2-VL: {e}")
        return False

    return True


def preload_yolo_model():
    """Preload YOLO model."""
    print("\n📥 Preloading YOLOv8 model...")

    try:
        yolo_path = PROJECT_DIR / "model_weights" / "best.pt"

        if not yolo_path.exists():
            print(f"⚠️  YOLO model not found at {yolo_path}")
            print("   Place your trained best.pt in model_weights/ folder")
            return False

        # Load and test
        model = YOLO(str(yolo_path))
        print(f"✅ YOLO model loaded and cached")
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    except Exception as e:
        print(f"❌ Failed to preload YOLO: {e}")
        return False

    return True


def preload_paddleocr_models():
    """Preload PaddleOCR language models."""
    print("\n📥 Preloading PaddleOCR models (Hindi + English)...")
    print("   This may take 1-2 minutes...")

    try:
        # Set cache directory
        paddle_cache = HF_CACHE / "paddle"
        paddle_cache.mkdir(exist_ok=True)
        os.environ['PADDLE_CACHE_HOME'] = str(paddle_cache)

        # Preload Hindi OCR
        paddle_hi = PaddleOCR(use_angle_cls=True, lang='hi')
        print("   ✅ Hindi OCR cached")

        # Preload English OCR
        paddle_en = PaddleOCR(use_angle_cls=True, lang='en')
        print("   ✅ English OCR cached")

        del paddle_hi, paddle_en

    except Exception as e:
        print(f"❌ Failed to preload PaddleOCR: {e}")
        return False

    return True


def preload_easyocr_model():
    """Preload EasyOCR (optional)."""
    print("\n📥 Preloading EasyOCR (optional)...")

    try:
        import easyocr
        reader = easyocr.Reader(['en', 'hi'], gpu=False, verbose=False)
        print("✅ EasyOCR loaded")
        del reader

    except Exception as e:
        print(f"⚠️  EasyOCR not available: {e}")
        print("   (This is optional, pipeline will work with PaddleOCR)")
        return True  # Not critical

    return True


def create_offline_marker():
    """Create marker file to indicate models are preloaded."""
    import json
    from datetime import datetime

    marker = MODELS_CACHE / "OFFLINE_READY"
    metadata = {
        "offline_ready": True,
        "timestamp": datetime.now().isoformat(),
        "models": {
            "qwen2vl": True,
            "yolo": True,
            "paddleocr": True,
            "easyocr": True
        }
    }
    with open(marker, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✅ Created offline-ready marker: {marker}")


def print_summary():
    """Print setup summary."""
    print("\n" + "="*60)
    print("🎉 Model Setup Complete!")
    print("="*60)
    print(f"\nCache locations:")
    print(f"  • Models: {MODELS_CACHE}")
    print(f"  • HuggingFace: {HF_CACHE}")
    print(f"\nYour system is now ready for offline demo!")
    print(f"\nNext steps:")
    print(f"  1. Copy input invoices to: input_images/")
    print(f"  2. Run: python3 demo_runner.py")
    print(f"  3. No internet needed during demo!")
    print("="*60 + "\n")


def main():
    """Run all preload steps."""
    print("\n" + "="*60)
    print("🚀 Invoice Extractor - Model Setup")
    print("="*60)

    setup_directories()

    results = {
        "Qwen2-VL": preload_qwen_model(),
        "YOLO": preload_yolo_model(),
        "PaddleOCR": preload_paddleocr_models(),
        "EasyOCR": preload_easyocr_model(),
    }

    print("\n" + "-"*60)
    print("Setup Summary:")
    print("-"*60)
    for name, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {name}")

    if all(results.values()):
        create_offline_marker()
        print_summary()
        return 0
    else:
        print("\n⚠️  Some models failed to preload. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
