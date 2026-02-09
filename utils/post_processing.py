import re
from difflib import SequenceMatcher

# Brand names (NOT dealers)
BRANDS = ['mahindra', 'swaraj', 'eicher', 'sonalika', 'tafe', 'escorts', 'powertrac',
          'massey', 'ferguson', 'new holland', 'john deere', 'force', 'captain',
          'महिंद्रा', 'स्वराज', 'आयशर', 'सोनालिका']


def clean_generic_text(text):
    """Clean text field."""
    if not text:
        return None
    text = str(text).strip()
    text = re.sub(r'^(Me\.|M\.|Mes\.|मे\.?)\s*',
                  'M/s ', text, flags=re.IGNORECASE)
    text = text.strip(" |-_:.,\"'")

    # Fix common OCR errors
    text = re.sub(r'\bNESHANAL\b', 'NATIONAL', text, flags=re.IGNORECASE)
    text = re.sub(r'\bNeshanal\b', 'National', text)

    return text if text else None


def is_garbage_text(text):
    """Check if text is OCR garbage (too short, just labels, etc.)."""
    if not text:
        return True
    t = text.strip().lower()
    # Common garbage patterns
    garbage = ['कटेशन', 'कोटेशन', 'quotation', 'invoice', 'date', 'ref', 'no.']
    if t in garbage or len(t) < 4:
        return True
    return False


def is_brand_name(text):
    """Check if text is just a brand name (not a dealer)."""
    if not text:
        return False
    t = text.lower().strip()
    for brand in BRANDS:
        if t == brand or t == brand + " tractors" or t == brand + " tractor":
            return True
    return False


def extract_dealer_from_header(spatial_text, raw_ocr_text):
    """Extract dealer from header, avoiding brand names and garbage."""
    text = spatial_text or raw_ocr_text or ""
    lines = [ln.strip() for ln in text.split('\n') if len(ln.strip()) > 4]

    company_words = [
        'traders', 'motors', 'tractors', 'enterprises', 'corporation', 'ltd', 'pvt',
        'auto', 'care', 'agencies', 'industries', 'm/s',
        'ट्रॅक्टर्स', 'मोटर्स', 'ट्रेडर्स', 'केयर', 'ऑटो', 'मे.'
    ]

    # Words that indicate NOT a dealer name
    skip_words = ['कटेशन', 'कोटेशन', 'quotation', 'invoice', 'date', 'gstin', 'gst',
                  'mobile', 'mo.', 'मो.', 'दिनांक', 'ref', 'स्वराज']

    for ln in lines[:10]:
        ln_lower = ln.lower()

        # Skip brand-only lines
        if is_brand_name(ln):
            continue

        # Skip date/number/garbage lines
        if re.match(r'^[\d\s.,/\-:]+$', ln):
            continue

        # Skip lines with skip words
        if any(w in ln_lower for w in skip_words):
            continue

        # Has company indicator (prioritize these)
        if any(w in ln_lower for w in company_words):
            return ln

    # Fallback: first non-brand, non-garbage line with reasonable length
    for ln in lines[:5]:
        ln_lower = ln.lower()
        if (not is_brand_name(ln) and
            not re.match(r'^[\d\s.,/\-:]+$', ln) and
            not any(w in ln_lower for w in skip_words) and
                len(ln) >= 5):
            return ln

    return None


def validate_and_clean_fields(fields, raw_ocr_text, spatial_text=None):
    """Validate and clean extracted fields."""
    dealer = fields.get("dealer_name")
    model = fields.get("model_name")

    dealer = clean_generic_text(dealer)
    model = clean_generic_text(model)

    # If dealer is garbage or brand name, get from header
    if is_garbage_text(dealer) or is_brand_name(dealer):
        dealer = extract_dealer_from_header(spatial_text, raw_ocr_text)
        dealer = clean_generic_text(dealer)

    # If still no dealer, extract from header
    if not dealer:
        dealer = extract_dealer_from_header(spatial_text, raw_ocr_text)
        dealer = clean_generic_text(dealer)

    # Anti-duplication
    if dealer and model:
        if SequenceMatcher(None, dealer.lower(), model.lower()).ratio() > 0.8:
            model = None

    fields["dealer_name"] = dealer
    fields["model_name"] = model
    return fields
