"""
Enhanced text extraction with multiple fallback methods
"""

from pdfminer.high_level import extract_text
from io import BytesIO
import re
import logging

logger = logging.getLogger(__name__)

def extract_sentences(pdf_bytes):
    """
    Extract sentences from PDF with multiple extraction methods.
    
    Tries in order:
    1. pdfminer (best for structured PDFs)
    2. PyPDF2 (fallback for simpler PDFs)
    3. PyMuPDF/fitz (fallback for complex layouts)
    """
    
    # Method 1: pdfminer (current method)
    try:
        pdf_file_like = BytesIO(pdf_bytes)
        text = extract_text(pdf_file_like)
        
        if text and len(text.strip()) > 100:  # At least 100 chars
            logger.info(f"✅ pdfminer extracted {len(text)} characters")
            return _process_text_to_sentences(text)
        else:
            logger.warning(f"⚠️ pdfminer extracted only {len(text) if text else 0} characters")
    except Exception as e:
        logger.error(f"❌ pdfminer failed: {e}")
    
    # Method 2: PyPDF2 (fallback)
    try:
        import PyPDF2
        pdf_file = BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        if text and len(text.strip()) > 100:
            logger.info(f"✅ PyPDF2 extracted {len(text)} characters (fallback)")
            return _process_text_to_sentences(text)
        else:
            logger.warning(f"⚠️ PyPDF2 extracted only {len(text) if text else 0} characters")
    except ImportError:
        logger.warning("⚠️ PyPDF2 not installed (pip install PyPDF2)")
    except Exception as e:
        logger.error(f"❌ PyPDF2 failed: {e}")
    
    # Method 3: PyMuPDF/fitz (second fallback)
    try:
        import fitz  # PyMuPDF
        pdf_file = BytesIO(pdf_bytes)
        pdf_document = fitz.open(stream=pdf_file, filetype="pdf")
        
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text += page.get_text()
        
        if text and len(text.strip()) > 100:
            logger.info(f"✅ PyMuPDF extracted {len(text)} characters (fallback)")
            return _process_text_to_sentences(text)
        else:
            logger.warning(f"⚠️ PyMuPDF extracted only {len(text) if text else 0} characters")
    except ImportError:
        logger.warning("⚠️ PyMuPDF not installed (pip install PyMuPDF)")
    except Exception as e:
        logger.error(f"❌ PyMuPDF failed: {e}")
    
    # All methods failed
    logger.error("❌ All extraction methods failed. PDF may be:")
    logger.error("   - Scanned images (needs OCR)")
    logger.error("   - Password protected")
    logger.error("   - Corrupted or invalid format")
    
    return []


def _process_text_to_sentences(text: str):
    """Process raw text into sentences."""
    # Fix common PDF encoding issues
    text = text.replace('(cid:415)', 'ti')
    text = text.replace('(cid:425)', 'tt')
    
    # Split text into sentences using regex
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Remove extra whitespace and newlines
    sentences = [s.strip().replace('\n', ' ') for s in sentences]
    
    # Remove empty sentences and very short ones (likely headers/footers)
    sentences = [s for s in sentences if len(s) > 10]
    
    return sentences


# Test function
def test_extraction(pdf_path: str):
    """Test extraction on a PDF file."""
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()
    
    print(f"\n{'='*60}")
    print(f"Testing extraction on: {pdf_path}")
    print(f"{'='*60}\n")
    
    sentences = extract_sentences(pdf_bytes)
    
    print(f"\n📊 Results:")
    print(f"   Sentences extracted: {len(sentences)}")
    if sentences:
        print(f"\n   First sentence: {sentences[0][:200]}...")
        print(f"   Last sentence: {sentences[-1][:200]}...")
    
    return sentences


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_extraction(sys.argv[1])
    else:
        print("Usage: python text_enhanced.py <pdf_file_path>")
