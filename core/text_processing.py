"""
Enhanced Text Processing Module
Combines the best of Phase-3-Final-master and current RAG system approaches
"""

import re
from typing import List, Optional, Dict, Any
from io import BytesIO

# Try to import pdfminer, fallback to basic text processing
try:
    from pdfminer.high_level import extract_text
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False
    print("Warning: pdfminer not available. PDF text extraction will be limited.")


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract text from PDF bytes with character encoding fixes.
    Incorporates improvements from Phase-3-Final-master.
    """
    if not PDFMINER_AVAILABLE:
        raise ImportError("pdfminer is required for PDF text extraction")
    
    # Convert bytes to a file-like object
    pdf_file_like = BytesIO(pdf_bytes)
    
    # Extract the text as a Unicode string
    text = extract_text(pdf_file_like)
    
    # Apply character encoding fixes from Phase-3-Final-master
    text = text.replace('(cid:415)', 'ti')
    text = text.replace('(cid:425)', 'tt')
    
    return text


def extract_sentences_enhanced(pdf_bytes: bytes) -> List[str]:
    """
    Enhanced sentence extraction with character fixes.
    Based on Phase-3-Final-master implementation with improvements.
    """
    text = extract_text_from_pdf(pdf_bytes)
    
    # Split text into sentences using regex
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Remove extra whitespace and newlines
    sentences = [s.strip().replace('\n', ' ') for s in sentences]
    
    # Remove empty sentences
    sentences = [s for s in sentences if s]
    
    return sentences


def chunk_text_adaptive(text: str, 
                       max_tokens: int = 900, 
                       overlap: int = 100,
                       method: str = "word") -> List[str]:
    """
    Enhanced text chunking with multiple strategies.
    
    Args:
        text: Input text to chunk
        max_tokens: Maximum tokens per chunk
        overlap: Overlap between chunks
        method: Chunking method ('word', 'sentence', 'hybrid')
    
    Returns:
        List of text chunks
    """
    
    if not text or not text.strip():
        return []
    
    if method == "sentence":
        return _chunk_by_sentences(text, max_tokens, overlap)
    elif method == "hybrid":
        return _chunk_hybrid(text, max_tokens, overlap)
    else:  # word method (default)
        return _chunk_by_words(text, max_tokens, overlap)


def _chunk_by_words(text: str, max_tokens: int, overlap: int) -> List[str]:
    """Word-based chunking (current system approach)."""
    words = text.split()
    if len(words) <= max_tokens:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        
        # Move start forward, accounting for overlap
        start = end - overlap
        if start >= len(words):
            break
    
    return chunks


def _chunk_by_sentences(text: str, max_tokens: int, overlap: int) -> List[str]:
    """Sentence-aware chunking."""
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return []
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_words = len(sentence.split())
        
        # If adding this sentence would exceed limit, start new chunk
        if current_length + sentence_words > max_tokens and current_chunk:
            chunks.append(' '.join(current_chunk))
            
            # Start new chunk with overlap
            overlap_sentences = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_sentences + [sentence]
            current_length = sum(len(s.split()) for s in current_chunk)
        else:
            current_chunk.append(sentence)
            current_length += sentence_words
    
    # Add final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def _chunk_hybrid(text: str, max_tokens: int, overlap: int) -> List[str]:
    """Hybrid approach: sentence-aware with word fallback."""
    # Try sentence-based first
    sentence_chunks = _chunk_by_sentences(text, max_tokens, overlap)
    
    # Check if any chunks are still too long
    final_chunks = []
    for chunk in sentence_chunks:
        words = chunk.split()
        if len(words) <= max_tokens:
            final_chunks.append(chunk)
        else:
            # Fallback to word-based chunking for long chunks
            sub_chunks = _chunk_by_words(chunk, max_tokens, overlap)
            final_chunks.extend(sub_chunks)
    
    return final_chunks


def preprocess_text_for_manufacturing(text: str) -> str:
    """
    Manufacturing-specific text preprocessing.
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common PDF extraction issues
    text = text.replace('(cid:415)', 'ti')
    text = text.replace('(cid:425)', 'tt')
    
    # Normalize measurement units
    text = re.sub(r'(\d+)\s*(mm|cm|in|inch|inches)', r'\1 \2', text)
    
    # Normalize temperature units
    text = re.sub(r'(\d+)\s*°\s*([CF])', r'\1°\2', text)
    
    # Normalize percentages
    text = re.sub(r'(\d+)\s*%', r'\1%', text)
    
    return text.strip()


def extract_manufacturing_features_from_text(text: str) -> List[str]:
    """
    Extract manufacturing-specific features from text.
    """
    features = []
    
    # Common manufacturing terms
    manufacturing_patterns = [
        r'\b(?:diameter|thickness|width|height|length|depth)\b',
        r'\b(?:tolerance|clearance|fit|surface\s+finish)\b',
        r'\b(?:material|steel|aluminum|plastic|composite)\b',
        r'\b(?:machining|drilling|milling|turning|grinding)\b',
        r'\b(?:assembly|welding|brazing|soldering)\b',
    ]
    
    for pattern in manufacturing_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        features.extend([match.lower().strip() for match in matches])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_features = []
    for feature in features:
        if feature not in seen:
            seen.add(feature)
            unique_features.append(feature)
    
    return unique_features


# Backwards compatibility
def chunk_text(text: str, max_tokens: int = 900, overlap: int = 100) -> List[str]:
    """Backwards compatible chunk_text function."""
    return chunk_text_adaptive(text, max_tokens, overlap, method="word")


def extract_sentences(pdf_bytes: bytes) -> List[str]:
    """Backwards compatible extract_sentences function."""
    return extract_sentences_enhanced(pdf_bytes)


# Test functions
if __name__ == "__main__":
    # Test text processing
    sample_text = """
    Design Guidelines for Manufacturing. The diameter should be between 10-20mm with tolerance of ±0.1mm.
    Material selection is critical for machining operations. Steel components require different approach than aluminum.
    Assembly process must consider clearances and surface finish requirements.
    """
    
    print("=== Text Processing Test ===")
    
    # Test chunking methods
    print("\n1. Word-based chunking:")
    word_chunks = chunk_text_adaptive(sample_text, max_tokens=15, method="word")
    for i, chunk in enumerate(word_chunks, 1):
        print(f"Chunk {i}: {chunk}")
    
    print("\n2. Sentence-based chunking:")
    sentence_chunks = chunk_text_adaptive(sample_text, max_tokens=15, method="sentence")
    for i, chunk in enumerate(sentence_chunks, 1):
        print(f"Chunk {i}: {chunk}")
    
    print("\n3. Hybrid chunking:")
    hybrid_chunks = chunk_text_adaptive(sample_text, max_tokens=15, method="hybrid")
    for i, chunk in enumerate(hybrid_chunks, 1):
        print(f"Chunk {i}: {chunk}")
    
    # Test feature extraction
    print("\n4. Manufacturing features:")
    features = extract_manufacturing_features_from_text(sample_text)
    print(f"Features: {features}")
    
    # Test preprocessing
    print("\n5. Preprocessed text:")
    processed = preprocess_text_for_manufacturing(sample_text)
    print(f"Processed: {processed}")