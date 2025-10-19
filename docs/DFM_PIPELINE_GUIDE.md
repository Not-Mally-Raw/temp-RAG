# DFM Pipeline Usage Guide

## Overview

The DFM (Design for Manufacturing) Pipeline is a complete end-to-end solution for extracting manufacturing rules from handbook documents. It uses RAG (Retrieval-Augmented Generation) to process documents and extract structured manufacturing constraints, tolerances, and requirements.

## Pipeline Architecture

```
PDF Document → Text Extraction → Chunking → Embedding → Vector Index → 
RAG Retrieval → LLM Rule Extraction → Structured Output
```

### Pipeline Steps

1. **Text Extraction**: Extract text from PDF using pdfplumber (or add OCR for scanned documents)
2. **Chunking**: Split text into overlapping chunks for better context
3. **Embedding**: Generate semantic embeddings using sentence-transformers
4. **Indexing**: Store embeddings in ChromaDB vector database
5. **Retrieval**: Find relevant chunks for queries using semantic search
6. **Rule Extraction**: Use LLM to extract structured manufacturing rules
7. **Postprocessing**: Format and validate extracted rules

## Installation

### Basic Installation

```bash
# Install core dependencies
pip install pdfplumber sentence-transformers chromadb transformers torch

# Or install from requirements.txt
pip install -r requirements.txt
```

### Optional Dependencies

```bash
# For OCR support (scanned PDFs)
pip install pytesseract

# For advanced models
pip install openai  # For GPT models
```

## Usage

### 1. Command Line Interface

#### Process a single DFM handbook:

```bash
python -m core.dfm_pipeline handbook.pdf
```

#### With custom options:

```bash
python -m core.dfm_pipeline handbook.pdf \
    --persist-dir ./my_db \
    --query "Extract all dimensional tolerances and surface finish requirements" \
    --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
    --llm-model google/flan-t5-small \
    --output results.json
```

### 2. Python API

#### Basic usage:

```python
from core.dfm_pipeline import process_dfm_handbook

# Process a handbook
results = process_dfm_handbook("path/to/handbook.pdf")

# Print extracted rules
print(results["rules"])
```

#### Advanced usage with custom configuration:

```python
from core.dfm_pipeline import (
    extract_text_from_pdf,
    split_text_for_rag,
    embed_chunks,
    build_vector_index,
    retrieve_context,
    extract_rules_with_llm,
)
from pathlib import Path

# Step-by-step processing
pdf_path = Path("handbook.pdf")

# 1. Extract text
text = extract_text_from_pdf(pdf_path)
print(f"Extracted {len(text)} characters")

# 2. Chunk text
chunks = split_text_for_rag(text, chunk_size=800, overlap=100)
print(f"Created {len(chunks)} chunks")

# 3. Generate embeddings
embeddings = embed_chunks(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Build index
collection = build_vector_index(chunks, embeddings, Path("./chroma_db"))

# 5. Query and retrieve
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
contexts = retrieve_context(
    "What are the dimensional tolerances?",
    collection,
    embedding_model,
    top_k=5
)

# 6. Extract rules
query = "Extract all manufacturing constraints and tolerances"
rules_text = extract_rules_with_llm(query, contexts, llm_model="google/flan-t5-small")
print(rules_text)
```

## Configuration

### Model Selection

#### Embedding Models

Choose based on your requirements:

- **Lightweight** (fast, low memory): `sentence-transformers/all-MiniLM-L6-v2`
- **Balanced**: `sentence-transformers/all-mpnet-base-v2`
- **High Quality** (slower, more memory): `BAAI/bge-large-en-v1.5`

#### LLM Models

- **Small/Fast**: `google/flan-t5-small` (80M params)
- **Medium**: `google/flan-t5-base` (250M params)  
- **Large**: `google/flan-t5-large` (780M params)
- **API-based**: Use OpenAI GPT-4 or similar for best quality

### Chunking Parameters

```python
# For detailed documents with lots of technical specs
chunk_size=800
chunk_overlap=100

# For shorter documents or when context is important
chunk_size=500
chunk_overlap=50

# For very large documents (optimize for speed)
chunk_size=1000
chunk_overlap=50
```

### Retrieval Parameters

```python
# Number of chunks to retrieve (more context vs more noise)
top_k=5  # Good default
top_k=10  # For complex queries needing more context
top_k=3   # For focused, specific queries
```

## Example Queries

### General Rule Extraction

```python
query = "Extract all manufacturing design rules"
```

### Specific Categories

```python
# Dimensional tolerances
query = "List all dimensional tolerances and precision requirements"

# Material specifications
query = "Extract material specifications and properties"

# Surface finish
query = "What are the surface finish requirements?"

# Assembly instructions
query = "Extract assembly procedures and constraints"

# Quality control
query = "List quality control and inspection requirements"
```

## Output Format

The pipeline returns a dictionary with the following structure:

```python
{
    "document": "handbook.pdf",
    "total_chunks": 150,
    "contexts_retrieved": 6,
    "rules": {
        "format": "json",  # or "text"
        "rules": [
            {
                "type": "dimensional_tolerance",
                "constraint": "±0.001 inches",
                "context": "critical dimensions",
                "confidence": 0.85
            },
            # ... more rules
        ]
    },
    "query": "Extract dimensional tolerances...",
    "status": "success"
}
```

## Integration with Existing Code

### Using with Enhanced RAG System

```python
from core.enhanced_rag_db import EnhancedManufacturingRAG
from core.dfm_pipeline import process_dfm_handbook

# Process handbook with DFM pipeline
results = process_dfm_handbook("handbook.pdf")

# Add to enhanced RAG system for further queries
rag = EnhancedManufacturingRAG()
rag.add_document("handbook.pdf", metadata={
    "source": "handbook.pdf",
    "extracted_rules": results["rules"]
})
```

### Using with Streamlit UI

The DFM pipeline integrates seamlessly with the existing Streamlit pages:

```python
# In a Streamlit page
import streamlit as st
from core.dfm_pipeline import process_dfm_handbook

uploaded_file = st.file_uploader("Upload DFM Handbook", type=["pdf"])

if uploaded_file:
    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    
    # Process
    with st.spinner("Processing handbook..."):
        results = process_dfm_handbook(tmp_path)
    
    # Display results
    st.json(results)
```

## Performance Optimization

### For Large Documents

```python
# Use smaller chunks and larger batches
chunks = split_text_for_rag(text, chunk_size=500, overlap=30)

# Process in batches
batch_size = 100
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    batch_embeddings = embed_chunks(batch)
    # ... process batch
```

### For Production Use

1. **Use GPU**: Set `device="cuda"` for embedding models
2. **Cache embeddings**: Reuse vector index across runs
3. **Use API models**: Consider OpenAI API for better quality
4. **Parallel processing**: Process multiple documents concurrently

## Troubleshooting

### Common Issues

#### Import Errors

```bash
# Ensure you're in the repository root
cd temp-RAG

# Add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### Memory Issues

```python
# Use smaller models
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
llm_model = "google/flan-t5-small"

# Or process in smaller batches
```

#### Slow Processing

```python
# Reduce chunk size
chunks = split_text_for_rag(text, chunk_size=300)

# Reduce retrieval count
contexts = retrieve_context(query, collection, model, top_k=3)
```

## Advanced Topics

### Custom Prompts

Modify the prompt in `extract_rules_with_llm` for your specific needs:

```python
def custom_extract_rules(query, contexts):
    prompt = """
    You are an expert in manufacturing design rules. 
    Extract ONLY the following information from the context:
    1. Numerical tolerances with units
    2. Material specifications
    3. Process constraints
    
    Format each rule as JSON with: type, value, unit, context
    """
    # ... rest of function
```

### Adding OCR Support

For scanned PDFs:

```python
import pytesseract
from pdf2image import convert_from_path

def extract_text_with_ocr(pdf_path):
    images = convert_from_path(pdf_path)
    text_parts = []
    for image in images:
        text = pytesseract.image_to_string(image)
        text_parts.append(text)
    return "\n\n".join(text_parts)
```

### Using Different Vector Databases

Replace ChromaDB with FAISS:

```python
import faiss
import numpy as np

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index
```

## Best Practices

1. **Start Small**: Test with sample documents before processing large handbooks
2. **Validate Results**: Always review extracted rules for accuracy
3. **Iterate Queries**: Refine queries based on initial results
4. **Cache Results**: Save processed results to avoid reprocessing
5. **Version Control**: Track which model versions produce which results

## Next Steps

- See `tests/test_dfm_pipeline.py` for more examples
- Check `data/sample_dfm.txt` for sample DFM content
- Review `config.py` for system-wide configuration options
- Explore `core/enhanced_rag_db.py` for advanced RAG features
