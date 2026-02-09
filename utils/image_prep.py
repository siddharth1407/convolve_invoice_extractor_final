import cv2
import numpy as np


def preprocess_for_ocr(image_path, method="adaptive"):
    """
    Smart Image Preprocessing Pipeline with multiple strategies.
    Returns the best image for OCR.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # 1. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if method == "adaptive":
        # Adaptive Thresholding - good for printed text with shadows
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
        return denoised

    elif method == "otsu":
        # Otsu's Thresholding - good for clean documents
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    elif method == "raw":
        # Raw grayscale - preserves handwritten text better
        return gray

    elif method == "contrast":
        # CLAHE - enhances contrast, good for faded text
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return enhanced

    return gray


def dual_ocr_preprocess(image_path):
    """
    Returns multiple preprocessed versions for dual OCR strategy.
    Some documents work better with raw, others with processed.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Version 1: Raw grayscale (better for handwritten/colored text)
    raw = gray.copy()

    # Version 2: Light preprocessing (contrast enhancement only)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return raw, enhanced


def get_best_ocr_image(image_path):
    """
    Try multiple preprocessing methods and return the one most likely to work.
    For now, use contrast-enhanced version as it's more robust.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use CLAHE for contrast enhancement - works well for most invoices
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return enhanced
