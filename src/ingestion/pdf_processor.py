import os
import fitz
import logging

logger = logging.getLogger(__name__)


class PDFProcessor:
    def __init__(self):
        logger.info("Using PyMuPDF (fitz) for PDF processing")

    def extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF file using PyMuPDF"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                page_text = page.get_text("text")
                text += page_text + "\n\n"  # Page delimiter
            doc.close()
        except Exception as e:
            raise Exception(f"PDF extraction failed: {e}")

        return text.strip()
