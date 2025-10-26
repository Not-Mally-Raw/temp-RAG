"""
Robust PDF text extraction with multiple fallback methods including OCR
"""

import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_sentences(pdf_bytes):
    """
    Main extraction function that uses the robust processor.
    """
    try:
        # Import here to avoid circular imports
        from extractors.robust_pdf_processor import RobustPDFProcessor
        
        processor = RobustPDFProcessor()
        sentences = processor.extract_sentences(pdf_bytes)
        
        if sentences:
            logger.info(f"✅ Successfully extracted {len(sentences)} sentences")
            return sentences
        else:
            logger.error("❌ No sentences extracted - PDF may be image-based or corrupted")
            return []
            
    except ImportError as e:
        logger.error(f"❌ Missing dependencies for robust extraction: {e}")
        # Fallback to basic extraction
        return _basic_extract_sentences(pdf_bytes)
    except Exception as e:
        logger.error(f"❌ Robust extraction failed: {e}")
        # Fallback to basic extraction
        return _basic_extract_sentences(pdf_bytes)


def _basic_extract_sentences(pdf_bytes):
    """
    Basic extraction using only pdfminer (original implementation).
    """
    try:
        from pdfminer.high_level import extract_text
        from io import BytesIO
        import re
        
        logger.info("⏳ Using basic pdfminer extraction...")
        
        # Convert bytes to a file-like object
        pdf_file_like = BytesIO(pdf_bytes)

        # Extract the text as a Unicode string
        text = extract_text(pdf_file_like)

        if not text or len(text.strip()) < 50:
            logger.warning(f"⚠️ Basic extraction got minimal text: {len(text) if text else 0} chars")
            return []

        text = text.replace('(cid:415)', 'ti')
        text = text.replace('(cid:425)', 'tt')

        # Split text into sentences using regex
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Remove extra whitespace and newlines
        sentences = [s.strip().replace('\n', ' ') for s in sentences]

        # Remove empty sentences and very short ones
        sentences = [s for s in sentences if len(s) > 10]

        logger.info(f"✅ Basic extraction: {len(sentences)} sentences")
        return sentences
        
    except Exception as e:
        logger.error(f"❌ Even basic extraction failed: {e}")
        return []