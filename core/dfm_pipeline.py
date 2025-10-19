"""
Minimal DFM rule extraction pipeline skeleton.

This module provides a complete end-to-end pipeline for extracting Design for Manufacturing (DFM)
rules from handbook documents. It implements: ingest → text extraction → chunking → embedding → 
indexing → retrieval → LLM rule extraction → postprocessing.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from a PDF file.
    
    Uses pdfplumber for text extraction. For image-based PDFs, OCR can be added.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text content
    """
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n\n".join(text_parts)
    except ImportError:
        logger.error("pdfplumber not installed. Install with: pip install pdfplumber")
        raise
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise


def split_text_for_rag(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks for RAG processing.
    
    Args:
        text: Input text to split
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of overlapping characters between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move to next chunk with overlap
        start = end - overlap if end < text_length else end
        if start <= end - overlap:  # Avoid infinite loop
            start = end
            
    return chunks


def embed_chunks(chunks: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Generate embeddings for text chunks.
    
    Args:
        chunks: List of text chunks
        model_name: Name of the sentence transformer model
        
    Returns:
        Array of embeddings
    """
    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        embeddings = model.encode(chunks, normalize_embeddings=True, show_progress_bar=True)
        return embeddings
    except ImportError:
        logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
        raise


def build_vector_index(chunks: List[str], embeddings, persist_dir: Path):
    """
    Build and persist a vector index using ChromaDB.
    
    Args:
        chunks: List of text chunks
        embeddings: Corresponding embeddings
        persist_dir: Directory to persist the database
        
    Returns:
        ChromaDB collection object
    """
    try:
        import chromadb
        from chromadb.config import Settings
        
        logger.info(f"Creating ChromaDB collection at {persist_dir}")
        client = chromadb.Client(Settings(
            persist_directory=str(persist_dir),
            anonymized_telemetry=False
        ))
        
        # Create or get collection
        collection_name = "dfm_docs"
        try:
            collection = client.get_collection(name=collection_name)
            logger.info(f"Using existing collection: {collection_name}")
        except:
            collection = client.create_collection(name=collection_name)
            logger.info(f"Created new collection: {collection_name}")
        
        # Add documents to collection
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        collection.add(
            documents=chunks,
            embeddings=embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings,
            ids=ids
        )
        
        logger.info(f"Added {len(chunks)} chunks to vector index")
        return collection
        
    except ImportError:
        logger.error("chromadb not installed. Install with: pip install chromadb")
        raise


def retrieve_context(query: str, collection, embedding_model, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve relevant context chunks for a query.
    
    Args:
        query: Query string
        collection: ChromaDB collection
        embedding_model: Embedding model for query
        top_k: Number of results to retrieve
        
    Returns:
        List of retrieved chunks with metadata
    """
    try:
        # Embed the query
        query_embedding = embedding_model.encode([query], normalize_embeddings=True)[0]
        
        # Query the collection
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        # Format results
        contexts = []
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                distance = results['distances'][0][i] if 'distances' in results else 0
                contexts.append({
                    'text': doc,
                    'distance': distance,
                    'similarity_score': 1 - distance  # Convert distance to similarity
                })
        
        return contexts
        
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return []


def extract_rules_with_llm(
    query: str, 
    contexts: List[Dict[str, Any]], 
    llm_model: str = "google/flan-t5-small"
) -> str:
    """
    Extract manufacturing rules using an LLM with retrieved contexts.
    
    Args:
        query: Query describing what rules to extract
        contexts: Retrieved context chunks
        llm_model: Name of the LLM model to use
        
    Returns:
        Generated rule extraction output
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        logger.info(f"Loading LLM model: {llm_model}")
        tokenizer = AutoTokenizer.from_pretrained(llm_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(llm_model)
        
        # Build prompt with context
        prompt = "Extract manufacturing design rules from the following context. "
        prompt += "For each rule, identify: rule type, constraint description, numeric values, and confidence level. "
        prompt += "Format as JSON list.\n\n"
        
        for i, ctx in enumerate(contexts):
            prompt += f"Context {i+1} (similarity: {ctx['similarity_score']:.2f}):\n{ctx['text']}\n\n"
        
        prompt += f"Question: {query}\n\nExtracted Rules (JSON):\n"
        
        # Generate response
        logger.info("Generating rule extraction...")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        outputs = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return decoded
        
    except ImportError:
        logger.error("transformers not installed. Install with: pip install transformers torch")
        raise


def postprocess_extracted_rules(raw_output: str) -> Dict[str, Any]:
    """
    Postprocess LLM output into structured rule objects.
    
    Args:
        raw_output: Raw LLM output text
        
    Returns:
        Structured rules dictionary
    """
    try:
        # Try to parse as JSON
        rules = json.loads(raw_output)
        return {"rules": rules, "format": "json"}
    except json.JSONDecodeError:
        # Fallback: return raw output with basic structure
        logger.warning("Could not parse JSON, returning raw output")
        return {
            "rules": [{"raw_text": raw_output, "format": "text"}],
            "format": "text"
        }


def process_dfm_handbook(
    pdf_path: str, 
    persist_dir: str = "./chroma_db",
    query: Optional[str] = None,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    llm_model_name: str = "google/flan-t5-small"
) -> Dict[str, Any]:
    """
    Complete end-to-end pipeline for processing a DFM handbook.
    
    Args:
        pdf_path: Path to the DFM handbook PDF
        persist_dir: Directory to persist the vector database
        query: Query for rule extraction (default: general DFM rules query)
        embedding_model_name: Name of embedding model
        llm_model_name: Name of LLM model
        
    Returns:
        Dictionary containing extracted rules and metadata
    """
    # Default query if not provided
    if query is None:
        query = (
            "Extract dimensional tolerances, mandatory manufacturing constraints, "
            "surface finish requirements, material specifications, and any numeric requirements."
        )
    
    logger.info(f"Processing DFM handbook: {pdf_path}")
    
    # Step 1: Extract text from PDF
    logger.info("Step 1: Extracting text from PDF...")
    pdf = Path(pdf_path)
    text = extract_text_from_pdf(pdf)
    logger.info(f"Extracted {len(text)} characters")
    
    # Step 2: Split text into chunks
    logger.info("Step 2: Splitting text into chunks...")
    chunks = split_text_for_rag(text)
    logger.info(f"Created {len(chunks)} chunks")
    
    # Step 3: Generate embeddings
    logger.info("Step 3: Generating embeddings...")
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer(embedding_model_name)
    embeddings = embed_chunks(chunks, embedding_model_name)
    
    # Step 4: Build vector index
    logger.info("Step 4: Building vector index...")
    collection = build_vector_index(chunks, embeddings, Path(persist_dir))
    
    # Step 5: Retrieve relevant contexts
    logger.info("Step 5: Retrieving relevant contexts...")
    contexts = retrieve_context(query, collection, embedding_model, top_k=6)
    logger.info(f"Retrieved {len(contexts)} relevant chunks")
    
    # Step 6: Extract rules with LLM
    logger.info("Step 6: Extracting rules with LLM...")
    raw_rules = extract_rules_with_llm(query, contexts, llm_model_name)
    
    # Step 7: Postprocess results
    logger.info("Step 7: Postprocessing results...")
    rules = postprocess_extracted_rules(raw_rules)
    
    result = {
        "document": pdf_path,
        "total_chunks": len(chunks),
        "contexts_retrieved": len(contexts),
        "rules": rules,
        "query": query,
        "status": "success"
    }
    
    logger.info("Pipeline completed successfully")
    return result


# CLI interface
def main():
    """Command-line interface for the DFM pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract DFM rules from handbook PDFs")
    parser.add_argument("pdf_path", help="Path to the DFM handbook PDF")
    parser.add_argument("--persist-dir", default="./chroma_db", help="Vector DB persist directory")
    parser.add_argument("--query", help="Custom extraction query")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--llm-model", default="google/flan-t5-small")
    parser.add_argument("--output", help="Output JSON file path")
    
    args = parser.parse_args()
    
    # Process the handbook
    result = process_dfm_handbook(
        pdf_path=args.pdf_path,
        persist_dir=args.persist_dir,
        query=args.query,
        embedding_model_name=args.embedding_model,
        llm_model_name=args.llm_model
    )
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
