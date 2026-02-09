"""
PDF Processing Utility for Invoice Extraction
Converts PDF invoices to images for processing
"""

import os
from pathlib import Path
import tempfile
import logging

# Try multiple PDF libraries for robustness
try:
    import fitz  # PyMuPDF - faster, better for scanned PDFs
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handle PDF to image conversion for invoice processing."""

    def __init__(self, dpi=300):
        """
        Initialize PDF processor.

        Args:
            dpi: DPI for PDF conversion (higher = better quality for OCR, but slower)
        """
        self.dpi = dpi
        if not PYMUPDF_AVAILABLE and not PDF2IMAGE_AVAILABLE:
            raise RuntimeError(
                "PDF processing libraries not found. "
                "Install with: pip install pymupdf pdf2image"
            )

    def pdf_to_images(self, pdf_path, output_dir=None, dpi=None):
        """
        Convert PDF to individual page images.

        Args:
            pdf_path: Path to PDF file
            output_dir: Optional directory to save images. If None, uses temp directory
            dpi: Optional DPI override

        Returns:
            List of image paths (one per PDF page)
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if not pdf_path.suffix.lower() == '.pdf':
            raise ValueError(f"File is not a PDF: {pdf_path}")

        dpi = dpi or self.dpi
        output_dir = output_dir or tempfile.mkdtemp(prefix="invoice_pdf_")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Converting PDF to images: {pdf_path} (DPI: {dpi})")

        try:
            if PYMUPDF_AVAILABLE:
                return self._convert_with_pymupdf(pdf_path, output_dir, dpi)
            else:
                return self._convert_with_pdf2image(pdf_path, output_dir, dpi)
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            raise

    def _convert_with_pymupdf(self, pdf_path, output_dir, dpi):
        """Convert using PyMuPDF (fitz) - faster and better for scanned PDFs."""
        images = []
        pdf_name = pdf_path.stem

        # Open PDF
        doc = fitz.open(str(pdf_path))
        num_pages = doc.page_count

        logger.info(f"PDF has {num_pages} page(s)")

        for page_num in range(num_pages):
            # Get page
            page = doc[page_num]

            # Render to image with specified DPI
            # DPI scaling: default is 72 DPI, so multiply by dpi/72
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            # Save as PNG
            image_path = output_dir / f"{pdf_name}_page_{page_num + 1:03d}.png"
            pix.save(str(image_path))
            images.append(str(image_path))

            logger.debug(f"Converted page {page_num + 1}/{num_pages}")

        doc.close()
        return images

    def _convert_with_pdf2image(self, pdf_path, output_dir, dpi):
        """Fallback conversion using pdf2image (requires poppler)."""
        images = []
        pdf_name = pdf_path.stem

        # Convert all pages to PIL Images
        pil_images = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            fmt='png'
        )

        logger.info(f"PDF has {len(pil_images)} page(s)")

        for page_num, pil_image in enumerate(pil_images):
            # Save as PNG
            image_path = output_dir / f"{pdf_name}_page_{page_num + 1:03d}.png"
            pil_image.save(str(image_path), 'PNG')
            images.append(str(image_path))

            logger.debug(f"Converted page {page_num + 1}/{len(pil_images)}")

        return images

    def is_pdf(self, file_path):
        """Check if file is a PDF."""
        return str(file_path).lower().endswith('.pdf')


def get_pdf_processor(dpi=300):
    """Get or create PDF processor instance."""
    return PDFProcessor(dpi=dpi)
