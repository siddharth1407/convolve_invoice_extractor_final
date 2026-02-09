import re
import torch
import numpy as np
import cv2


def clean_cost(raw_cost):
    """
    Normalizes asset cost to integer.
    Removes currency symbols like '₹', ',', 'Rs' and returns pure int.
    """
    if isinstance(raw_cost, (int, float)):
        return int(raw_cost)
    if isinstance(raw_cost, str):
        # Remove non-numeric characters
        clean = re.sub(r'[^\d]', '', raw_cost)
        return int(clean) if clean else 0
    return 0


def sanity_check_hp(val):
    """
    Validates Horse Power.
    Tractors typically range from 15 HP to 100 HP.
    """
    try:
        if not val:
            return None
        ival = int(val)
        return ival if 15 <= ival <= 100 else None
    except:
        return None


def smart_extract_hp(data):
    """
    Hybrid Fallback Logic for HP:
    1. Tries the 'horse_power' field directly.
    2. If missing/invalid, scans 'model_name' text using Regex patterns (e.g., '50 HP').
    """
    # 1. Try direct field extraction
    raw_hp = data.get("horse_power")
    if isinstance(raw_hp, str):
        match = re.search(r'(\d+)', raw_hp)
        if match:
            raw_hp = int(match.group(1))

    if sanity_check_hp(raw_hp):
        return int(raw_hp)

    # 2. Fallback: Search in Model Description
    model_str = data.get("model_name", "")
    if model_str and isinstance(model_str, str):
        # Look for patterns like "50 HP", "50HP", "50 H.P."
        matches = re.findall(
            r'(\d{2})\s*(?:HP|H\.P\.|hp|Hp)', model_str, re.IGNORECASE)
        for m in matches:
            if sanity_check_hp(m):
                return int(m)
    return None


def validate_bbox(field_data):
    """
    Ensures consistency: If the bbox is empty or [0,0,0,0], 'present' MUST be False.
    This prevents the "Ghost Box" issue where JSON says True but no box exists.
    """
    default = {"present": False, "bbox": [0, 0, 0, 0]}

    if not field_data:
        return default

    # Case 1: Model output just a list [y, x, y, x]
    if isinstance(field_data, list):
        # Check if it has 4 numbers and at least one is > 0
        is_valid = len(field_data) == 4 and any(c > 0 for c in field_data)
        return {"present": is_valid, "bbox": field_data if is_valid else [0, 0, 0, 0]}

    # Case 2: Model output a dict {"present": true, "bbox": [...]}
    if isinstance(field_data, dict):
        bbox = field_data.get("bbox", [0, 0, 0, 0])

        # STRICT CHECK: The box is only valid if it contains non-zero coordinates
        is_valid_box = isinstance(bbox, list) and len(
            bbox) == 4 and any(c > 0 for c in bbox)

        return {
            "present": is_valid_box,  # Force 'False' if the box is empty
            "bbox": bbox if is_valid_box else [0, 0, 0, 0]
        }

    return default


def calculate_perplexity_confidence(outputs):
    """
    STRICT CONFIDENCE SCORING
    - Penalizes any uncertainty heavily.
    - If the model is 99% sure about 9 words but 50% sure about 1 word, 
      the score drops significantly.
    """
    if not hasattr(outputs, 'scores') or not outputs.scores:
        return 0.0

    token_probs = []

    for step_scores in outputs.scores:
        # Softmax to get probabilities
        probs = torch.nn.functional.softmax(step_scores, dim=-1)
        # Get the probability of the chosen token
        max_prob, _ = torch.max(probs, dim=-1)
        token_probs.append(max_prob.item())

    if not token_probs:
        return 0.0

    # THE FIX: Don't ignore the "hard" tokens. Include them all.
    # We calculate the geometric mean of ALL tokens.
    log_sum = np.sum(np.log(token_probs))
    geo_mean = np.exp(log_sum / len(token_probs))

    # Penalize scores that look "too perfect" (often a sign of hallucination)
    # If it's exactly 1.0 or > 0.995, we clamp it slightly to be realistic
    final_score = min(0.98, geo_mean)

    return round(final_score, 4)


def detect_signature_stamp(image_path):
    """
    Detect signature and stamp regions using image processing.
    Returns (signature_result, stamp_result) where each is {"present": bool, "bbox": [x1,y1,x2,y2]}
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"present": False, "bbox": [0, 0, 0, 0]}, {"present": False, "bbox": [0, 0, 0, 0]}

    h, w = img.shape[:2]

    # Focus on bottom-right quadrant where signatures/stamps usually are
    roi_y = int(h * 0.5)
    roi_x = int(w * 0.3)
    roi = img[roi_y:, roi_x:]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Detect blue/purple regions (common stamp colors)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Detect red regions (common stamp colors)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(
        hsv, lower_red2, upper_red2)

    stamp_mask = blue_mask | red_mask

    # Results
    signature_result = {"present": False, "bbox": [0, 0, 0, 0]}
    stamp_result = {"present": False, "bbox": [0, 0, 0, 0]}

    # Find stamp contours
    contours, _ = cv2.findContours(
        stamp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 2000 < area < 100000:  # Reasonable stamp size
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Convert back to full image coordinates
            stamp_result = {
                "present": True,
                "bbox": [roi_x + x, roi_y + y, roi_x + x + bw, roi_y + y + bh]
            }
            break

    # Detect signature (dark ink curves in bottom area)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / max(bh, 1)

        # Signature: wider than tall, reasonable size
        if 1500 < area < 50000 and 0.5 < aspect < 5:
            # Check if not overlapping with stamp
            if not stamp_result["present"] or not _boxes_overlap(
                [roi_x + x, roi_y + y, roi_x + x + bw, roi_y + y + bh],
                stamp_result["bbox"]
            ):
                signature_result = {
                    "present": True,
                    "bbox": [roi_x + x, roi_y + y, roi_x + x + bw, roi_y + y + bh]
                }
                break

    return signature_result, stamp_result


def _boxes_overlap(box1, box2):
    """Check if two bounding boxes overlap."""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    return not (x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1)
