"""
Robust PDF text extraction with OCR fallback for scanned documents
"""

from pdfminer.high_level import extract_text
from io import BytesIO
import re
import logging
import tempfile
import os
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class RobustPDFProcessor:
    """
    Multi-method PDF text extraction with OCR fallback
    """
    
    def __init__(self):
        self.extraction_methods = [
            ('pdfminer', self._extract_with_pdfminer),
            ('pdfplumber', self._extract_with_pdfplumber),
            ('pypdf2', self._extract_with_pypdf2),
            ('pymupdf', self._extract_with_pymupdf),
            ('ocr', self._extract_with_ocr)
        ]
    
    def extract_sentences(self, pdf_bytes: bytes) -> List[str]:
        """
        Extract sentences from PDF using multiple fallback methods.
        """
        logger.info(f"🔍 Processing PDF ({len(pdf_bytes)} bytes)")
        
        for method_name, method_func in self.extraction_methods:
            try:
                logger.info(f"⏳ Trying {method_name}...")
                text, char_count = method_func(pdf_bytes)
                
                if text and len(text.strip()) > 50:  # Minimum viable text
                    logger.info(f"✅ {method_name} extracted {char_count} characters")
                    sentences = self._process_text_to_sentences(text)
                    
                    if sentences:
                        logger.info(f"📝 Processed into {len(sentences)} sentences")
                        return sentences
                    else:
                        logger.warning(f"⚠️ {method_name} extracted text but no valid sentences")
                else:
                    logger.warning(f"⚠️ {method_name} extracted insufficient text ({char_count} chars)")
                    
            except ImportError as e:
                logger.warning(f"⚠️ {method_name} not available: {e}")
            except Exception as e:
                logger.error(f"❌ {method_name} failed: {e}")
        
        # All methods failed
        logger.error("❌ ALL EXTRACTION METHODS FAILED")
        logger.error("📋 Possible issues:")
        logger.error("   • PDF contains only scanned images (OCR failed)")
        logger.error("   • PDF is password protected or encrypted")
        logger.error("   • PDF file is corrupted")
        logger.error("   • Missing required libraries for OCR")
        
        return []
    
    def _extract_with_pdfminer(self, pdf_bytes: bytes) -> Tuple[str, int]:
        """Extract using pdfminer (best for structured PDFs)"""
        pdf_file = BytesIO(pdf_bytes)
        text = extract_text(pdf_file)
        return text, len(text) if text else 0
    
    def _extract_with_pdfplumber(self, pdf_bytes: bytes) -> Tuple[str, int]:
        """Extract using pdfplumber (good for tables and complex layouts)"""
        import pdfplumber
        
        pdf_file = BytesIO(pdf_bytes)
        text_parts = []
        
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        
        text = '\n'.join(text_parts)
        return text, len(text)
    
    def _extract_with_pypdf2(self, pdf_bytes: bytes) -> Tuple[str, int]:
        """Extract using PyPDF2 (simple fallback)"""
        import PyPDF2
        
        pdf_file = BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text_parts = []
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        
        text = '\n'.join(text_parts)
        return text, len(text)
    
    def _extract_with_pymupdf(self, pdf_bytes: bytes) -> Tuple[str, int]:
        """Extract using PyMuPDF (good for complex PDFs)"""
        import fitz  # PyMuPDF
        
        pdf_file = BytesIO(pdf_bytes)
        pdf_document = fitz.open(stream=pdf_file, filetype="pdf")
        
        text_parts = []
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            page_text = page.get_text()
            if page_text:
                text_parts.append(page_text)
        
        text = '\n'.join(text_parts)
        return text, len(text)
    
    def _extract_with_ocr(self, pdf_bytes: bytes) -> Tuple[str, int]:
        """Extract using OCR (for scanned/image PDFs)"""
        try:
            import pytesseract
            from pdf2image import convert_from_bytes
            from PIL import Image
        except ImportError as e:
            raise ImportError(f"OCR libraries not installed: {e}. Install with: pip install pytesseract pdf2image pillow")
        
        logger.info("🖼️ Converting PDF to images for OCR...")
        
        # Convert PDF to images
        try:
            images = convert_from_bytes(pdf_bytes, dpi=200)
            logger.info(f"📸 Converted to {len(images)} images")
        except Exception as e:
            raise Exception(f"PDF to image conversion failed: {e}")
        
        # OCR each image
        text_parts = []
        for i, image in enumerate(images):
            try:
                logger.info(f"🔍 OCR processing page {i+1}/{len(images)}...")
                
                # Configure Tesseract for better accuracy
                custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:!?()[]{}+-=<>/%$@#&*'
                page_text = pytesseract.image_to_string(image, config=custom_config)
                
                if page_text.strip():
                    text_parts.append(page_text)
                    logger.info(f"✅ Page {i+1}: {len(page_text)} characters")
                else:
                    logger.warning(f"⚠️ Page {i+1}: No text detected")
                    
            except Exception as e:
                logger.error(f"❌ OCR failed on page {i+1}: {e}")
        
        if not text_parts:
            raise Exception("OCR failed to extract any text from images")
        
        text = '\n'.join(text_parts)
        logger.info(f"🎯 OCR completed: {len(text)} total characters")
        
        return text, len(text)
    
    def _process_text_to_sentences(self, text: str) -> List[str]:
        """Process raw text into clean sentences."""
        if not text:
            return []
        
        # Fix common PDF encoding issues
        text = text.replace('(cid:415)', 'ti')
        text = text.replace('(cid:425)', 'tt')
        text = text.replace('\x0c', ' ')  # Form feed
        text = text.replace('\xa0', ' ')  # Non-breaking space
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Split into sentences (more sophisticated regex)
        sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s+'
        sentences = re.split(sentence_pattern, text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Filter out very short sentences, headers, page numbers
            if (len(sentence) > 15 and 
                not sentence.isdigit() and 
                not re.match(r'^Page \d+', sentence) and
                not re.match(r'^\d+$', sentence)):
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def analyze_pdf(self, pdf_bytes: bytes) -> dict:
        """
        Analyze PDF and return extraction diagnostics.
        """
        analysis = {
            'size_bytes': len(pdf_bytes),
            'methods_tested': [],
            'successful_method': None,
            'total_text_length': 0,
            'sentence_count': 0,
            'issues_detected': []
        }
        
        # Test each method
        for method_name, method_func in self.extraction_methods:
            try:
                text, char_count = method_func(pdf_bytes)
                
                method_result = {
                    'name': method_name,
                    'success': char_count > 50,
                    'text_length': char_count,
                    'error': None
                }
                
                if method_result['success'] and not analysis['successful_method']:
                    analysis['successful_method'] = method_name
                    analysis['total_text_length'] = char_count
                    sentences = self._process_text_to_sentences(text)
                    analysis['sentence_count'] = len(sentences)
                
            except ImportError as e:
                method_result = {
                    'name': method_name,
                    'success': False,
                    'text_length': 0,
                    'error': f"Library not installed: {e}"
                }
            except Exception as e:
                method_result = {
                    'name': method_name,
                    'success': False,
                    'text_length': 0,
                    'error': str(e)
                }
            
            analysis['methods_tested'].append(method_result)
        
        # Detect common issues
        if not analysis['successful_method']:
            analysis['issues_detected'].append("No extraction method succeeded")
        
        if analysis['total_text_length'] < 100:
            analysis['issues_detected'].append("Very little text extracted - may be image-based PDF")
        
        return analysis


def extract_sentences(pdf_bytes: bytes) -> List[str]:
    """
    Main extraction function - maintains compatibility with existing code.
    """
    processor = RobustPDFProcessor()
    return processor.extract_sentences(pdf_bytes)


def analyze_pdf_file(pdf_path: str) -> dict:
    """
    Analyze a PDF file and return diagnostics.
    """
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()
    
    processor = RobustPDFProcessor()
    return processor.analyze_pdf(pdf_bytes)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        print(f"\n{'='*60}")
        print(f"ROBUST PDF ANALYSIS: {pdf_path}")
        print(f"{'='*60}\n")
        
        # Run analysis
        analysis = analyze_pdf_file(pdf_path)
        
        print(f"📊 Analysis Results:")
        print(f"   File size: {analysis['size_bytes']:,} bytes")
        print(f"   Successful method: {analysis['successful_method'] or 'NONE'}")
        print(f"   Text extracted: {analysis['total_text_length']:,} characters")
        print(f"   Sentences found: {analysis['sentence_count']:,}")
        
        if analysis['issues_detected']:
            print(f"\n⚠️  Issues detected:")
            for issue in analysis['issues_detected']:
                print(f"   • {issue}")
        
        print(f"\n🔧 Method Results:")
        for method in analysis['methods_tested']:
            status = "✅" if method['success'] else "❌"
            error_msg = f" ({method['error']})" if method['error'] else ""
            print(f"   {status} {method['name']}: {method['text_length']:,} chars{error_msg}")
        
        # Extract actual sentences if successful
        if analysis['successful_method']:
            print(f"\n📝 Extracting sentences...")
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            
            sentences = extract_sentences(pdf_bytes)
            
            if sentences:
                print(f"\n   First sentence: {sentences[0][:200]}...")
                if len(sentences) > 1:
                    print(f"   Last sentence: {sentences[-1][:200]}...")
            else:
                print("   No valid sentences extracted")
    else:
        print("Usage: python robust_pdf_processor.py <pdf_file_path>")