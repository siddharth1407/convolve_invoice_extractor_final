"""
Multi-OCR Engine for better text extraction.
Uses PaddleOCR, with optional EasyOCR support.
"""

from paddleocr import PaddleOCR
import re
import cv2
import logging

logging.getLogger("ppocr").setLevel(logging.ERROR)


class MultiOCR:
    """Ensemble OCR using PaddleOCR (with optional EasyOCR)."""

    def __init__(self):
        print("🔧 Initializing Multi-OCR Engine...")

        # PaddleOCR engines
        self.paddle_hi = PaddleOCR(use_angle_cls=True, lang='hi')
        self.paddle_en = PaddleOCR(use_angle_cls=True, lang='en')

        # Try EasyOCR (optional - may fail due to SSL issues)
        self.easyocr_reader = None
        try:
            import easyocr
            print("  ✅ EasyOCR found - attempting to initialize...")
            self.easyocr_reader = easyocr.Reader(
                ['en', 'hi'], gpu=False, verbose=False)
            print("  ✅ EasyOCR initialized successfully")
        except Exception as e:
            print(f"  ⚠️ EasyOCR not available: {str(e)[:50]}...")
            print("     Continuing with PaddleOCR only")

    def _preprocess_image(self, img):
        """Apply preprocessing to enhance OCR accuracy."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # CLAHE for contrast enhancement (helps with faded/poor quality)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

        return gray, enhanced, denoised

    def extract_text(self, image_path):
        """
        Extract text using available OCR engines with multiple preprocessing.
        Returns (spatial_text, combined_text).
        """
        img = cv2.imread(image_path)
        if img is None:
            return "", ""

        # Get multiple preprocessed versions
        gray, enhanced, denoised = self._preprocess_image(img)

        all_texts = []

        def add_texts(texts):
            for t in texts:
                t_clean = t.strip()
                if t_clean:
                    all_texts.append(t_clean)

        # Run OCR on multiple image versions for robustness
        images_to_try = [gray, enhanced]

        for img_version in images_to_try:
            # 1. PaddleOCR Hindi
            try:
                result = self.paddle_hi.ocr(img_version, cls=True)
                if result and result[0]:
                    texts = [item[1][0]
                             for item in result[0] if item[1][1] > 0.35]
                    add_texts(texts)
            except Exception:
                pass

            # 2. PaddleOCR English
            try:
                result = self.paddle_en.ocr(img_version, cls=True)
                if result and result[0]:
                    texts = [item[1][0]
                             for item in result[0] if item[1][1] > 0.35]
                    add_texts(texts)
            except Exception:
                pass

        # 3. EasyOCR (if available) - on enhanced image
        if self.easyocr_reader:
            try:
                result = self.easyocr_reader.readtext(enhanced)
                texts = [item[1] for item in result if item[2] > 0.25]
                add_texts(texts)
            except Exception:
                pass

        # Deduplicate and combine
        lines = list(dict.fromkeys(all_texts))
        spatial = "\n".join(lines)
        combined = " ".join(lines)

        return spatial, combined

    def extract_hp(self, text):
        """Extract HP with handwriting corrections - robust for unseen invoices."""
        if not text:
            return 0

        # Fix common OCR/handwriting misreads (comprehensive list)
        corrections = [
            (r'\bA7\b', '47'), (r'\ba7\b',
                                '47'), (r'\b4T\b', '47'), (r'\bAT\b', '47'),
            (r'\bA1\b', '41'), (r'\ba1\b', '41'), (r'\b4I\b', '41'),
            (r'\bA2\b', '42'), (r'\ba2\b', '42'), (r'\b4Z\b', '42'),
            (r'\bA5\b', '45'), (r'\ba5\b', '45'), (r'\b4S\b', '45'),
            (r'\bA9\b', '49'), (r'\ba9\b', '49'),
            (r'\bS0\b', '50'), (r'\b5O\b', '50'),
            (r'\bS5\b', '55'), (r'\b5S\b', '55'),
            (r'\b3O\b', '30'), (r'\bS9\b', '39'),
        ]
        for pat, repl in corrections:
            text = re.sub(pat, repl, text, flags=re.IGNORECASE)

        # Comprehensive HP patterns (priority order)
        patterns = [
            # Standard formats
            r'(\d{2})\s*(?:HP|H\.P\.?|hp|Hp|H\s*P)\b',
            r'(?:HP|H\.P\.?)\s*[:\-]?\s*(\d{2})\b',
            # Hindi formats
            r'(\d{2})\s*(?:एच\.?\s*पी|एचपी|एचपी\.)',
            r'(\d{2})\s*(?:हॉर्स|horse|अश्वशक्ति)',
            # In model strings
            r'[,\s/](\d{2})\s*HP',
            r'(?:DI|4WD|2WD|PT|MF|SONALIKA)\s+(\d{2})\s*HP',
            # Marathi formats
            r'(\d{2})\s*(?:एच\.पी\.?)',
            # Bracketed/quoted
            r'[\(\[](\d{2})\s*HP[\)\]]',
            r'["\'](\d{2})\s*HP["\']',
            # Single number patterns (when HP word is separate)
            r'\bHP\s*[-:]?\s*(\d{2})',
            r'\b(\d{2})\b(?=\s*(?:HP|एचपी|हॉर्स))',
        ]

        for pat in patterns:
            matches = re.findall(pat, text, re.IGNORECASE)
            for m in matches:
                try:
                    val = int(m) if isinstance(m, str) else m
                    if 20 <= val <= 99:
                        return val
                except:
                    pass

        return 0

    def extract_cost(self, text):
        """Extract total cost - robust for various Indian invoice formats."""
        if not text:
            return 0

        # Priority patterns (most specific first)
        patterns = [
            # Hindi/Marathi total keywords (very strict)
            r'(?:एकूण|एकुण|योग|कुल|total|grand\s*total|रु\.?|रुपये)\s*[:]?\s*(?:rs\.?\s*|₹\s*|रु\.?\s*)?(\d[\d,\.]*)',
            # With "only" or "/-"
            r'(?:rs\.?|₹|रु\.?)\s*(\d[\d,\.]*)\s*(?:only|/-|/-|मात्र|\()',
            # Indian lakh format (8,50,000 or 8,30,000 or 8,00,000)
            r'(\d{1,2},\d{2},\d{3})',
            # Without comma - looser (850000 or 5500000)
            r'(?:rs\.?|₹|रु\.?)\s*(\d{6,7})(?:\s|$|[-/]|मात्र)',
            # After keywords
            r'(?:amount|रक्कम|किंमत|price|मूल्य|नेट|NET)\s*[:\-]?\s*(?:rs\.?\s*|₹\s*|रु\.?\s*)?(\d[\d,]*)',
            # After dash (830000- or 550000/-)
            r'(\d{6,7})\s*[-/]',
            # Between Rs and parenthesis
            r'Rs\.?\s*(\d[\d,\.]*)\s*\(',
            # In middle of text with clear demarcation
            r'(?:amount|नेट|NET|कुल|total)[\s:]*(\d[\d,\.]*)',
            # Standalone large numbers in invoice area
            r'^\s*(\d{6,7})\s*$',
        ]

        best = 0
        for pat in patterns:
            for m in re.finditer(pat, text, re.IGNORECASE | re.MULTILINE):
                try:
                    val_str = m.group(1).replace(
                        ',', '').replace('.', '').strip()
                    if not val_str:
                        continue
                    val = int(val_str)
                    # Tractor price range: 3L to 20L (expanded to catch more)
                    if 200000 <= val <= 3000000 and val > best:
                        best = val
                except:
                    continue

        return best

    def extract_dealer(self, text):
        """Extract dealer name (not brand) - robust for unseen invoices."""
        if not text:
            return None

        lines = [ln.strip() for ln in text.split('\n') if len(ln.strip()) > 4]

        # Company indicators (English + Hindi/Marathi)
        company_words = [
            # English
            'traders', 'motors', 'tractors', 'enterprises', 'corporation',
            'ltd', 'limited', 'pvt', 'private', 'auto', 'care', 'agencies',
            'm/s', 'agro', 'sales', 'service', 'center', 'centre', 'dealer',
            # Hindi/Marathi
            'ट्रॅक्टर्स', 'ट्रेक्टर्स', 'मोटर्स', 'ट्रेडर्स', 'केयर', 'ऑटो',
            'एजेंसी', 'एजन्सी', 'सर्विस', 'सेंटर', 'सेल्स', 'मे.', 'प्रा.लिमि'
        ]

        # Comprehensive brand list (NOT dealers)
        brands = [
            'mahindra', 'swaraj', 'eicher', 'sonalika', 'tafe', 'escorts',
            'powertrac', 'massey', 'ferguson', 'john deere', 'new holland',
            'force', 'captain', 'preet', 'farmtrac', 'indo farm', 'ace',
            'kubota', 'vst', 'mitsubishi', 'yanmar', 'same', 'deutz',
            'महिंद्रा', 'स्वराज', 'आयशर', 'सोनालिका', 'टाफे', 'पॉवरट्रॅक'
        ]

        # Skip words - labels, not dealer names
        skip_words = [
            'कटेशन', 'कोटेशन', 'quotation', 'invoice', 'date', 'gstin', 'gst',
            'mobile', 'मो.', 'दिनांक', 'स्वराज', 'tax', 'cgst', 'sgst', 'igst',
            'bill', 'receipt', 'serial', 'sl.', 'no.', 'phone', 'email',
            'बिल', 'रसीद', 'दूरभाष', 'फोन'
        ]

        for ln in lines[:12]:
            ln_lower = ln.lower()

            # Skip brand-only
            is_brand_only = any(ln_lower.strip() == b or
                                (ln_lower.strip().startswith(b)
                                 and len(ln_lower.split()) <= 2)
                                for b in brands)
            if is_brand_only:
                continue

            # Skip numbers
            if re.match(r'^[\d\s.,/\-:]+$', ln):
                continue

            # Skip garbage/labels
            if any(w in ln_lower for w in skip_words):
                continue

            # Has company indicator
            if any(w in ln_lower for w in company_words):
                return ln

        # Fallback
        for ln in lines[:5]:
            if len(ln) >= 5 and not re.match(r'^[\d\s.,/\-:]+$', ln):
                ln_lower = ln.lower()
                if not any(b in ln_lower for b in brands):
                    if not any(w in ln_lower for w in skip_words):
                        return ln

        return None

    def extract_model(self, text):
        """Extract tractor model name - robust patterns."""
        if not text:
            return None

        # Common model patterns (priority order)
        patterns = [
            # Mahindra models
            r'((?:ARJUN|YUVO|JIVO|OJA|NOVO)\s*(?:Tech\+?)?\s*\d{3}\s*(?:DI|SP|PLUS|4WD|2WD)?(?:\s*\d{2}\s*HP)?)',
            # Swaraj models
            r'(\d{3}\s*(?:XT|FE|SP)\s*(?:PLUS)?)',
            # Eicher models
            r'((?:EICHER)\s*\d{3,4})',
            # Massey Ferguson
            r'(MF[-\s]?\d{3,4}\s*(?:DI|MAHASHAKTI|DAYNATRACK)?)',
            # Powertrac
            r'(PT\s*\d{3}\s*(?:DS|DI)?\s*(?:PLUS)?(?:\s*HR)?)',
            # Sonalika
            r'((?:SONALIKA|DI)\s*\d{2,3}\s*(?:RX)?)',
            # Generic: 3-digit + suffix
            r'(\d{3}\s*(?:SP|DI|FE|XT|HP)\s*(?:PLUS|4WD|2WD)?)',
        ]

        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                model = match.group(1).strip()
                if len(model) >= 3:
                    return model

        return None
