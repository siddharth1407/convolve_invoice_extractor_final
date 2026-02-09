import re


def smart_parse_cost(ocr_text):
    """
    Extract cost/total amount from text.
    """
    if not ocr_text:
        return 0

    candidates = []

    # Priority patterns
    patterns = [
        # High priority - explicit total keywords
        (r'(?:grand\s*total|total\s*amount|net\s*payable)\s*[:\-=]?\s*(?:rs\.?\s*|₹\s*)?([\d,]+)', 100),
        (r'(?:योग|एकुण|एकूण|कुल)\s*[:\-=]?\s*(?:रु\.?\s*)?([\d,]+)', 100),
        (r'total\s*[:\-=]?\s*(?:rs\.?\s*|₹\s*)?([\d,]+)', 90),

        # Medium priority
        (r'(?:rs\.?|₹|रु\.?)\s*([\d,]+)\s*(?:only|/-)', 70),
        (r'(\d{1,2},\d{2},\d{3})', 60),  # Indian format: 8,30,000

        # Low priority
        (r'(\d{6,7})\s*[-/]', 40),
    ]

    for pattern, priority in patterns:
        matches = re.findall(pattern, ocr_text, re.IGNORECASE)
        for m in matches:
            clean_num = m.replace(',', '').strip()
            if clean_num.isdigit():
                val = int(clean_num)

                # Normalize
                if val >= 10000000:
                    val = val // 100
                elif val >= 2000000:
                    val = val // 10

                # Valid range: 3L to 20L
                if 300000 <= val <= 2000000:
                    candidates.append((val, priority))

    if candidates:
        candidates.sort(key=lambda x: (-x[1], -x[0]))
        return candidates[0][0]

    return 0


def smart_parse_hp(ocr_text):
    """
    Extract HP with STRICT adjacency requirement.
    Number must be IMMEDIATELY followed by HP indicator.
    """
    if not ocr_text:
        return 0

    text = ocr_text.replace('\n', ' ')

    # Strict patterns - HP indicator must be right after number
    patterns = [
        r'(\d{2})\s*(?:HP|H\.P\.?)\b',
        r'(\d{2})\s*(?:एच\.?\s*पी\.?|हॉर्स\s*पॉवर)',
        r'(?:HP|H\.P\.?)\s*[:\-=]\s*(\d{2})\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = int(match.group(1))
            if 20 <= val <= 99:
                return val

    return 0
