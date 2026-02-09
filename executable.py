"""
Hybrid Invoice Extraction Pipeline
Uses Multi-OCR + Qwen2-VL for best accuracy.
"""

from pathlib import Path
import re
import os
import torch
import json
import logging
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from utils.processing import calculate_perplexity_confidence, sanity_check_hp, validate_bbox, detect_signature_stamp
from utils.multi_ocr import MultiOCR
from utils.post_processing import validate_and_clean_fields

MODEL_PATH = "Qwen/Qwen2-VL-2B-Instruct"

# Automatic device selection: prefer CUDA, then fall back to CPU (skip MPS for stability)
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Setup HuggingFace cache for offline mode
_cache_dir = Path(__file__).parent / ".huggingface_cache"
if _cache_dir.exists():
    os.environ['HF_HOME'] = str(_cache_dir)
    # STRICT OFFLINE MODE: Disable all HuggingFace Hub access
    os.environ['HF_HUB_OFFLINE'] = '1'  # Completely disable internet
    os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Offline for transformers
    os.environ['HF_DATASETS_OFFLINE'] = '1'  # Offline for datasets
    print(f"📦 Using cached models from: {_cache_dir}")
    print("🔒 OFFLINE MODE: All model access from local cache")
else:
    print("⚠️  No model cache found. Run setup_models.py first for offline demo.")


logging.getLogger("ppocr").setLevel(logging.ERROR)

print("👁️ Initializing Hybrid Pipeline...")

# Initialize Multi-OCR Engine
ocr_engine = MultiOCR()

# Global model cache (persistent across function calls)
_loaded_model = None
_loaded_processor = None


def load_model():
    """Load model once and cache it globally to avoid reloading."""
    global _loaded_model, _loaded_processor

    # Return cached model if already loaded
    if _loaded_model is not None and _loaded_processor is not None:
        print(f"✅ CACHE HIT: Using cached model on {device}")
        import sys
        sys.stdout.flush()
        return _loaded_model, _loaded_processor, device

    # Choose dtype: use FP16 on CUDA to accelerate inference and reduce memory
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"🚀 Loading Qwen2-VL on {device} with dtype={torch_dtype}...")

    # Try 8-bit quantization on CUDA for extra speed + memory savings
    if device == "cuda":
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_has_fp16_weight=False
            )
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL_PATH,
                quantization_config=bnb_config,
                device_map="cuda",
                cache_dir=str(_cache_dir),
                local_files_only=True,  # OFFLINE MODE
                trust_remote_code=True
            )
            print("✅ Using 8-bit quantization for faster inference")
        except Exception as e:
            # Fallback to non-quantized
            print(
                f"⚠️ 8-bit quantization failed ({e}), using standard loading")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch_dtype,
                cache_dir=str(_cache_dir),
                local_files_only=True,  # OFFLINE MODE
                trust_remote_code=True
            )
            model.to(device)
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch_dtype,
            cache_dir=str(_cache_dir),
            local_files_only=True,  # OFFLINE MODE
            trust_remote_code=True
        )
        model.to(device)

    model.eval()

    # Load processor with offline-first strategy
    try:
        processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            cache_dir=str(_cache_dir),
            local_files_only=True,  # OFFLINE MODE
            trust_remote_code=True
        )
    except Exception as e:
        print(f"⚠️  Standard processor load failed in offline mode: {e}")
        print("   Trying alternative loading method...")

        # Fallback: Look for processor in the snapshot directory
        snapshot_dir = _cache_dir / "models--Qwen--Qwen2-VL-2B-Instruct" / "snapshots"
        if snapshot_dir.exists():
            # Get the first (and likely only) snapshot directory
            snapshots = list(snapshot_dir.glob("*"))
            if snapshots:
                snapshot_path = snapshots[0]
                print(f"   Loading processor from snapshot: {snapshot_path}")
                processor = AutoProcessor.from_pretrained(
                    snapshot_path,
                    local_files_only=True,
                    trust_remote_code=True
                )
                print("   ✅ Processor loaded from local snapshot")
            else:
                raise RuntimeError(
                    "No model snapshots found in cache. Run setup_models.py first.")
        else:
            raise RuntimeError(
                f"Cache directory not found: {snapshot_dir}. Run setup_models.py first.")

    # Cache globally so we don't reload for subsequent images
    _loaded_model = model
    _loaded_processor = processor

    return model, processor, device


def process_single_image(model, processor, device, image_path, filename):
    """
    Process invoice with Multi-OCR + VLM hybrid approach.
    OPTIMIZED: Reduces image resolution for faster VLM inference.
    """
    # OPTIMIZATION: Reduce image resolution to 1200px max only for very large images
    # (preserves numeric field detection while still saving inference time)
    import cv2
    import tempfile
    img = cv2.imread(image_path)
    vlm_image_path = image_path
    if img is not None:
        h, w = img.shape[:2]
        # Only resize if image is very large (>1200px) to preserve number legibility
        if max(h, w) > 1200:
            scale = 1200 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img_resized = cv2.resize(
                img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            fd, vlm_image_path = tempfile.mkstemp(suffix='.jpg')
            cv2.imwrite(vlm_image_path, img_resized)
            os.close(fd)

    # 1. MULTI-OCR EXTRACTION (fast, always run as fallback)
    spatial_text, raw_text = ocr_engine.extract_text(image_path)

    # 2. OCR-BASED EXTRACTION (reliable fallback)
    ocr_hp = ocr_engine.extract_hp(raw_text)
    ocr_cost = ocr_engine.extract_cost(raw_text)
    ocr_dealer = ocr_engine.extract_dealer(spatial_text)
    ocr_model = ocr_engine.extract_model(raw_text)

    # 3. VLM PROMPT
    # OPTIMIZATION: Use 1500 chars of OCR (balanced for accuracy & speed)
    prompt = f"""Extract from this Indian tractor invoice. Return JSON only.

OCR TEXT:
{spatial_text[:1500]}

CRITICAL RULES:
- dealer_name: Company that ISSUED invoice (has Traders/Motors/Ltd/M/s/ट्रॅक्टर्स/मोटर्स)
  * KEEP HINDI TEXT AS-IS! If dealer is "नेशनल मोटर्स", output "नेशनल मोटर्स" NOT "National Motors"
  * Mahindra/Swaraj/Eicher/Sonalika are BRANDS not dealers!
  * Look at TOP of invoice for dealer name (letterhead)
- model_name: Tractor model (like "575 SP PLUS", "MF-241 DT")  
- horse_power: Number 20-99 BEFORE "HP" or "H.P." or "एच.पी" (A7=47 handwritten)
- asset_cost: Total amount in rupees (एकूण/Total)

DO NOT translate Hindi to English. Preserve original language.

{{"dealer_name":"string","model_name":"string","horse_power":number,"asset_cost":number}}"""

    # 4. RUN VLM (with resized image for speed)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": vlm_image_path},
        {"type": "text", "text": prompt},
    ]}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(device)

    # Use inference mode and autocast when CUDA is available for speed
    # OPTIMIZATION: Reduce max_new_tokens from 256 → 100 (JSON output is short, saves ~2.5x generation time)
    if device == "cuda":
        with torch.inference_mode():
            with torch.cuda.amp.autocast():
                generated_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False,
                                               temperature=0.0, output_scores=True, return_dict_in_generate=True)
    else:
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False,
                                           temperature=0.0, output_scores=True, return_dict_in_generate=True)

    gen_trimmed = [out[len(inp):] for inp, out in zip(
        inputs.input_ids, generated_ids.sequences)]
    output = processor.batch_decode(gen_trimmed, skip_special_tokens=True)[0]

    # 5. PARSE VLM OUTPUT
    try:
        clean = output.replace("```json", "").replace("```", "").strip()
        match = re.search(r'\{[^{}]*\}', clean, re.DOTALL)
        data = json.loads(match.group()) if match else json.loads(clean)
    except:
        data = {}

    # 6. SMART FIELD SELECTION

    # HP: VLM if valid (20-99), else OCR
    vlm_hp = data.get("horse_power", 0)
    if isinstance(vlm_hp, int) and 20 <= vlm_hp <= 99:
        final_hp = vlm_hp
    else:
        final_hp = ocr_hp

    # Cost: VLM if valid range, else OCR
    vlm_cost = data.get("asset_cost", 0)
    try:
        vlm_cost = int(str(vlm_cost).replace(',', ''))
    except:
        vlm_cost = 0
    final_cost = vlm_cost if 300000 <= vlm_cost <= 2000000 else ocr_cost

    # Dealer: Check if VLM returned brand name (common error)
    vlm_dealer = data.get("dealer_name", "") or ""
    vlm_model = data.get("model_name", "") or ""

    # Comprehensive brand list
    brands = [
        'mahindra', 'swaraj', 'eicher', 'sonalika', 'tafe', 'escorts',
        'powertrac', 'massey', 'ferguson', 'john deere', 'new holland',
        'force', 'captain', 'preet', 'farmtrac', 'kubota', 'vst'
    ]
    vlm_dealer_lower = vlm_dealer.lower().strip()

    is_brand = any(
        vlm_dealer_lower == b or
        vlm_dealer_lower == b + " tractors" or
        vlm_dealer_lower == b + " tractor" or
        (vlm_dealer_lower.startswith(b + " ") and "trader" not in vlm_dealer_lower)
        for b in brands
    )

    final_dealer = ocr_dealer if (
        is_brand or not vlm_dealer.strip()) else vlm_dealer

    # Model: VLM if valid, else OCR fallback
    final_model = vlm_model if vlm_model.strip() else ocr_model

    # 7. SIGNATURE/STAMP DETECTION (image-based)
    img_signature, img_stamp = detect_signature_stamp(image_path)

    # Use VLM output if valid, otherwise use image-based detection
    vlm_signature = validate_bbox(
        data.get("signature", {"present": False, "bbox": [0, 0, 0, 0]}))
    vlm_stamp = validate_bbox(
        data.get("stamp", {"present": False, "bbox": [0, 0, 0, 0]}))

    final_signature = vlm_signature if vlm_signature["present"] else img_signature
    final_stamp = vlm_stamp if vlm_stamp["present"] else img_stamp

    # 8. BUILD RESULT
    fields = {
        "dealer_name": final_dealer,
        "model_name": final_model,
        "horse_power": final_hp,
        "asset_cost": final_cost,
        "signature": final_signature,
        "stamp": final_stamp
    }
    fields = validate_and_clean_fields(fields, raw_text, spatial_text)

    # 9. VALIDATE SIGNATURE/STAMP
    fields["signature"] = validate_bbox(fields["signature"])
    fields["stamp"] = validate_bbox(fields["stamp"])

    # 10. EMERGENCY FALLBACKS
    if not fields.get("dealer_name"):
        fields["dealer_name"] = ocr_dealer
    if fields.get("horse_power", 0) == 0:
        fields["horse_power"] = ocr_hp
    if fields.get("asset_cost", 0) == 0:
        fields["asset_cost"] = ocr_cost
    if not fields.get("model_name"):
        fields["model_name"] = ocr_model

    # 11. CONFIDENCE
    confidence = calculate_perplexity_confidence(generated_ids)

    # Cleanup: remove temporary resized image
    if vlm_image_path != image_path:
        try:
            os.remove(vlm_image_path)
        except:
            pass

    return {"doc_id": filename, "fields": fields, "confidence": confidence}
