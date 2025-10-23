from pdfminer.high_level import extract_text
from io import BytesIO
import re

def extract_sentences(pdf_bytes):
    # Convert bytes to a file-like object
    pdf_file_like = BytesIO(pdf_bytes)

    # Extract the text as a Unicode string
    text = extract_text(pdf_file_like)

    text = text.replace('(cid:415)', 'ti')
    text = text.replace('(cid:425)', 'tt')

    # Split text into sentences using regex
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Remove extra whitespace and newlines
    sentences = [s.strip().replace('\n', ' ') for s in sentences]

    # Remove empty sentences
    sentences = [s for s in sentences if s]

    return sentences