import re


def clean_money_string(value):
    """Normalize cost to integer."""
    if not value:
        return 0
    if isinstance(value, (int, float)):
        amount = float(value)
    else:
        clean = str(value).replace(',', '').replace(
            'Rs', '').replace('₹', '').strip()
        match = re.search(r'(\d+)', clean)
        if not match:
            return 0
        amount = float(match.group(1))

    # Normalize range
    while amount > 2000000:
        amount /= 10

    return int(amount) if amount >= 100000 else 0


def extract_hp_from_text(ocr_text):
    """Extract HP with strict adjacency."""
    if not ocr_text:
        return 0

    # Fix handwriting OCR errors
    text = ocr_text
    text = re.sub(r'\bA7\b', '47', text)
    text = re.sub(r'\ba7\b', '47', text)
    text = re.sub(r'\bA1\b', '41', text)
    text = re.sub(r'\ba1\b', '41', text)
    text = re.sub(r'\b4T\b', '47', text)

    # Strict patterns
    patterns = [
        r'(\d{2})\s*(?:HP|H\.P\.?)\b',
        r'(\d{2})\s*(?:एच\.?\s*पी)',
        r'(\d{2})\s*हॉर्स',
    ]

    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            val = int(match.group(1))
            if 20 <= val <= 99:
                return val
    return 0


def extract_dealer_model_from_ocr(spatial_text):
    """Extract dealer and model from OCR."""
    if not spatial_text:
        return None, None

    lines = [ln.strip()
             for ln in spatial_text.split('\n') if len(ln.strip()) > 4]

    dealer = None
    model = None

    company_words = ['traders', 'motors', 'tractors', 'enterprises', 'corporation',
                     'ltd', 'auto', 'care', 'm/s', 'ट्रॅक्टर्स', 'मोटर्स', 'मे.']
    brands = ['mahindra', 'swaraj', 'eicher', 'sonalika', 'tafe', 'powertrac']

    for ln in lines[:10]:
        ln_lower = ln.lower()
        is_brand_only = any(ln_lower.strip() == b for b in brands)
        if is_brand_only:
            continue
        if any(w in ln_lower for w in company_words):
            dealer = ln
            break

    for ln in lines:
        if re.search(r'\d{2}\s*(?:HP|H\.P)', ln, re.I):
            model = ln
            break

    return dealer, model


def extract_cost_from_ocr(ocr_text):
    """Extract cost from OCR."""
    if not ocr_text:
        return 0

    patterns = [
        r'(?:total|योग|एकुण)\s*(?:rs\.?\s*)?(\d[\d,]*)',
        r'(\d{1,2},\d{2},\d{3})',
    ]

    for pat in patterns:
        match = re.search(pat, ocr_text, re.IGNORECASE)
        if match:
            val = int(match.group(1).replace(',', ''))
            if 300000 <= val <= 2000000:
                return val
    return 0
