# Troubleshooting Guide

## Common Issues and Solutions

### Import Errors

#### Problem: `ModuleNotFoundError: No module named 'core'`

**Solution:**

```bash
# Option 1: Run from repository root
cd /path/to/temp-RAG
python -m core.dfm_pipeline data/sample_dfm.txt

# Option 2: Install package in development mode
pip install -e .

# Option 3: Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### Problem: `ImportError: cannot import name 'EnhancedManufacturingRAG'`

**Cause**: Missing dependencies or circular imports

**Solution:**

```bash
# Install all dependencies
pip install -r requirements.txt

# Check Python version (3.8+ required)
python --version
```

### Dependency Issues

#### Problem: `ModuleNotFoundError: No module named 'sentence_transformers'`

**Solution:**

```bash
# Install missing dependency
pip install sentence-transformers

# Or install all dependencies
pip install -r requirements.txt
```

#### Problem: `torch not found` or CUDA errors

**Solution:**

```bash
# For CPU-only (default)
pip install torch

# For GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues

#### Problem: Out of memory when processing large documents

**Solution 1**: Use smaller models

```python
# Instead of BAAI/bge-large-en-v1.5
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"  # Much smaller

# Instead of flan-t5-large
llm_model = "google/flan-t5-small"
```

**Solution 2**: Process in smaller chunks

```python
from core.dfm_pipeline import split_text_for_rag

# Smaller chunks
chunks = split_text_for_rag(text, chunk_size=300, overlap=30)
```

**Solution 3**: Batch processing

```python
# Process embeddings in batches
batch_size = 50
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    batch_embeddings = embed_chunks(batch)
```

### Performance Issues

#### Problem: Processing is very slow

**Solution 1**: Use GPU acceleration

```python
from sentence_transformers import SentenceTransformer

# Enable GPU
model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
```

**Solution 2**: Reduce chunk count

```python
# Larger chunks = fewer embeddings
chunks = split_text_for_rag(text, chunk_size=1000, overlap=50)

# Retrieve fewer chunks
contexts = retrieve_context(query, collection, model, top_k=3)
```

**Solution 3**: Disable progress bars

```python
embeddings = model.encode(chunks, show_progress_bar=False)
```

### ChromaDB Issues

#### Problem: `chromadb.errors.InvalidDimensionException`

**Cause**: Embeddings dimension mismatch

**Solution:**

```python
# Ensure consistent embedding model
from config import RAGConfig

embedding_model = RAGConfig.EMBEDDING_MODEL  # Use config
```

#### Problem: `PersistentClientError: Could not connect to database`

**Solution:**

```bash
# Delete corrupted database
rm -rf ./chroma_db/

# Or use a fresh directory
python -m core.dfm_pipeline handbook.pdf --persist-dir ./new_db
```

### PDF Processing Issues

#### Problem: `pdfplumber.PDFSyntaxError`

**Cause**: Corrupted or encrypted PDF

**Solution:**

```bash
# Try decrypting the PDF first
pip install pypdf
python -c "
from pypdf import PdfReader, PdfWriter
reader = PdfReader('encrypted.pdf')
writer = PdfWriter()
for page in reader.pages:
    writer.add_page(page)
with open('decrypted.pdf', 'wb') as f:
    writer.write(f)
"
```

#### Problem: No text extracted from PDF (scanned documents)

**Solution**: Add OCR support

```bash
# Install OCR dependencies
pip install pytesseract pdf2image
# On Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils

# On macOS
brew install tesseract poppler
```

Then use:

```python
from pdf2image import convert_from_path
import pytesseract

def extract_text_with_ocr(pdf_path):
    images = convert_from_path(pdf_path)
    text = []
    for image in images:
        text.append(pytesseract.image_to_string(image))
    return "\n\n".join(text)
```

### Streamlit Issues

#### Problem: `exec() in main_app.py fails`

**Cause**: Old code using exec() pattern

**Solution**: This has been fixed in the latest version. Update your code:

```bash
git pull origin main
```

Or manually update `main_app.py` to use proper imports.

#### Problem: Streamlit page not found

**Solution:**

```bash
# Ensure pages/__init__.py exists
ls pages/__init__.py

# Run from repository root
cd /path/to/temp-RAG
streamlit run main_app.py
```

### Model Download Issues

#### Problem: Model download fails or times out

**Solution:**

```bash
# Pre-download models
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print('Downloaded successfully')
"

# Set custom cache directory
export TRANSFORMERS_CACHE="/path/to/cache"
```

#### Problem: Slow model downloads

**Solution**: Use a mirror or cached models

```bash
# Use Hugging Face mirror (China)
export HF_ENDPOINT=https://hf-mirror.com

# Or download models manually
git lfs install
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
```

### Results Quality Issues

#### Problem: Extracted rules are not accurate

**Solution 1**: Improve query specificity

```python
# Instead of:
query = "Extract rules"

# Use:
query = """
Extract manufacturing design rules including:
- Dimensional tolerances with numeric values and units
- Material specifications with grades and properties
- Surface finish requirements with Ra values
- Assembly constraints with specific measurements
"""
```

**Solution 2**: Increase retrieval context

```python
# Retrieve more chunks
contexts = retrieve_context(query, collection, model, top_k=10)
```

**Solution 3**: Use a better LLM

```python
# Instead of flan-t5-small, use:
llm_model = "google/flan-t5-large"  # Better quality

# Or use API-based models
# (requires separate implementation)
```

#### Problem: Rules are incomplete or missing

**Solution**: Check chunking strategy

```python
# Ensure chunks aren't too small
chunks = split_text_for_rag(text, chunk_size=800, overlap=100)

# Verify chunk quality
for i, chunk in enumerate(chunks[:5]):
    print(f"Chunk {i}:\n{chunk}\n---")
```

### Testing Issues

#### Problem: Tests fail due to missing dependencies

**Solution:**

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run specific test
python tests/test_dfm_pipeline.py

# Or use pytest
pytest tests/
```

#### Problem: Tests pass but integration doesn't work

**Solution**: Check import paths

```python
# Verify imports work
python -c "from core.dfm_pipeline import process_dfm_handbook; print('OK')"

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

## Getting Help

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from core.dfm_pipeline import process_dfm_handbook
results = process_dfm_handbook("handbook.pdf")
```

### Checking System Requirements

```bash
# Check Python version
python --version  # Should be 3.8+

# Check available memory
free -h  # Linux
vm_stat  # macOS

# Check disk space
df -h

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Getting Versions

```bash
# Check installed package versions
pip list | grep -E "torch|transformers|chromadb|sentence"

# Check repository version
git log --oneline -1
```

### Clean Installation

If all else fails:

```bash
# Remove all cached files
rm -rf ~/.cache/huggingface/
rm -rf ./chroma_db/
rm -rf __pycache__/
rm -rf **/__pycache__/

# Reinstall dependencies
pip uninstall -y -r requirements.txt
pip install -r requirements.txt

# Verify installation
python -c "
from core.dfm_pipeline import process_dfm_handbook
print('Installation successful')
"
```

## Still Having Issues?

1. Check the GitHub Issues page for similar problems
2. Ensure you have the latest version: `git pull`
3. Try the minimal example in `tests/test_dfm_pipeline.py`
4. Review the logs carefully for error messages
5. Create a minimal reproducible example

## Common Error Messages

### `RuntimeError: CUDA out of memory`

**Solution**: Use CPU or smaller models

```python
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
```

### `ValueError: Invalid chunk_size`

**Solution**: Ensure chunk_size > chunk_overlap

```python
chunks = split_text_for_rag(text, chunk_size=500, overlap=50)  # Valid
```

### `FileNotFoundError: [Errno 2] No such file or directory`

**Solution**: Use absolute paths or verify file exists

```python
from pathlib import Path
pdf_path = Path("handbook.pdf").resolve()
assert pdf_path.exists(), f"File not found: {pdf_path}"
```

### `JSONDecodeError: Expecting value`

**Cause**: LLM output is not valid JSON

**Solution**: This is handled automatically by the postprocessor

```python
from core.dfm_pipeline import postprocess_extracted_rules

# Will return text format if JSON parsing fails
result = postprocess_extracted_rules(raw_output)
print(result["format"])  # 'json' or 'text'
```
