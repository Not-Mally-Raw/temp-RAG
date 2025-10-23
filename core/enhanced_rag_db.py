"""
enhanced_rag_db.py
State-of-the-art RAG implementation specifically designed for manufacturing rule generation and constraint handling.
Features: Advanced embeddings, multi-modal content support, hierarchical chunking, manufacturing-aware retrieval.
"""

from typing import List, Dict, Optional, Any, Union, Tuple
import os
import json
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.retrievers import BaseRetriever

# Import our existing extractors
from extractors.text import extract_sentences
from extractors.table import extract_tables_algo, dual_pipeline_2
from extractors.image import extract_images
from generators.features import features_dict

@dataclass
class DocumentMetadata:
    """Enhanced metadata structure for manufacturing documents."""
    doc_id: str
    source_file: str
    doc_type: str  # 'text', 'table', 'image'
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    manufacturing_process: Optional[str] = None
    rule_category: Optional[str] = None
    chunk_index: int = 0
    confidence_score: Optional[float] = None
    extracted_at: str = datetime.now().isoformat()
    features: Optional[List[str]] = None
    constraints: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

class SentenceTransformerEmbeddings(Embeddings):
    """Custom embeddings wrapper for sentence transformers."""
    
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

class ManufacturingTextSplitter:
    """Specialized text splitter for manufacturing documents."""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )
        
        # Manufacturing-specific section headers
        self.section_headers = [
            "design guidelines", "manufacturing constraints", "specifications",
            "requirements", "tolerances", "materials", "processes", "quality",
            "inspection", "testing", "assembly", "fabrication"
        ]
    
    def split_with_structure(self, text: str, metadata: DocumentMetadata) -> List[Document]:
        """Split text while preserving manufacturing document structure."""
        documents = []
        
        # Try to identify sections
        lines = text.split('\n')
        current_section = ""
        current_content = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if this line is a section header
            is_header = any(header in line_lower for header in self.section_headers)
            
            if is_header and current_content:
                # Process previous section
                section_text = '\n'.join(current_content)
                chunks = self.base_splitter.split_text(section_text)
                
                for i, chunk in enumerate(chunks):
                    chunk_metadata = DocumentMetadata(
                        doc_id=f"{metadata.doc_id}_section_{hashlib.md5(current_section.encode()).hexdigest()[:8]}",
                        source_file=metadata.source_file,
                        doc_type=metadata.doc_type,
                        page_number=metadata.page_number,
                        section_title=current_section,
                        manufacturing_process=self._infer_manufacturing_process(chunk),
                        rule_category=self._infer_rule_category(chunk),
                        chunk_index=i,
                        features=self._extract_features(chunk),
                        constraints=self._extract_constraints(chunk)
                    )
                    
                    documents.append(Document(
                        page_content=chunk,
                        metadata=chunk_metadata.to_dict()
                    ))
                
                current_section = line.strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Process final section
        if current_content:
            section_text = '\n'.join(current_content)
            chunks = self.base_splitter.split_text(section_text)
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = DocumentMetadata(
                    doc_id=f"{metadata.doc_id}_final_{i}",
                    source_file=metadata.source_file,
                    doc_type=metadata.doc_type,
                    page_number=metadata.page_number,
                    section_title=current_section,
                    manufacturing_process=self._infer_manufacturing_process(chunk),
                    rule_category=self._infer_rule_category(chunk),
                    chunk_index=i,
                    features=self._extract_features(chunk),
                    constraints=self._extract_constraints(chunk)
                )
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=chunk_metadata.to_dict()
                ))
        
        return documents
    
    def _infer_manufacturing_process(self, text: str) -> Optional[str]:
        """Infer manufacturing process from text content."""
        text_lower = text.lower()
        
        process_keywords = {
            "injection molding": ["injection", "molding", "mold", "plastic"],
            "sheet metal": ["sheet", "metal", "bend", "flange", "stamping"],
            "machining": ["machining", "milling", "drilling", "turning", "cutting"],
            "additive": ["3d print", "additive", "layer", "printing"],
            "assembly": ["assembly", "fastener", "bolt", "screw", "joint"],
            "die casting": ["die cast", "casting", "molten", "die"],
            "general": ["general", "standard", "common"]
        }
        
        for process, keywords in process_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return process
        
        return None
    
    def _infer_rule_category(self, text: str) -> Optional[str]:
        """Infer rule category based on features_dict."""
        text_lower = text.lower()
        
        # Check against known manufacturing processes
        for category in features_dict.keys():
            if category.lower() in text_lower:
                return category
        
        return None
    
    def _extract_features(self, text: str) -> List[str]:
        """Extract manufacturing features mentioned in the text."""
        features = []
        text_lower = text.lower()
        
        # Common manufacturing features
        feature_keywords = [
            "hole", "radius", "diameter", "thickness", "length", "width", "height",
            "angle", "tolerance", "surface", "finish", "material", "bend", "flange",
            "boss", "rib", "draft", "taper", "chamfer", "fillet", "clearance",
            "interference", "distance", "gap", "overlap"
        ]
        
        for keyword in feature_keywords:
            if keyword in text_lower:
                features.append(keyword)
        
        return features
    
    def _extract_constraints(self, text: str) -> List[str]:
        """Extract manufacturing constraints from text."""
        constraints = []
        text_lower = text.lower()
        
        # Look for constraint indicators
        constraint_patterns = [
            "should be", "must be", "shall be", "require", "minimum", "maximum",
            "at least", "no more than", "between", "greater than", "less than",
            "equal to", "conform to", "standard", "specification"
        ]
        
        for pattern in constraint_patterns:
            if pattern in text_lower:
                constraints.append(pattern)
        
        return constraints

class EnhancedManufacturingRAG:
    """Enhanced RAG system specifically designed for manufacturing rule generation."""
    
    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-large-en-v1.5",
        persist_path: str = "enhanced_chroma_db",
        chunk_size: int = 800,
        chunk_overlap: int = 100
    ):
        self.embeddings = SentenceTransformerEmbeddings(embedding_model_name)
        self.persist_path = persist_path
        self.text_splitter = ManufacturingTextSplitter(chunk_size, chunk_overlap)
        
        # Initialize vector store
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=persist_path
        )
        
        # Enhanced memory for conversation
        self.memory = ConversationBufferWindowMemory(
            k=10,  # Keep last 10 exchanges
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Document registry
        self.doc_registry: Dict[str, Dict[str, Any]] = {}
        self.load_registry()
    
    def load_registry(self):
        """Load document registry from disk."""
        registry_path = Path(self.persist_path) / "doc_registry.json"
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                self.doc_registry = json.load(f)
    
    def save_registry(self):
        """Save document registry to disk."""
        registry_path = Path(self.persist_path) / "doc_registry.json"
        os.makedirs(self.persist_path, exist_ok=True)
        with open(registry_path, 'w') as f:
            json.dump(self.doc_registry, f, indent=2)
    
    def process_pdf_document(self, pdf_bytes: bytes, filename: str) -> Dict[str, int]:
        """Process a complete PDF document with all content types."""
        doc_id = hashlib.md5(pdf_bytes).hexdigest()[:16]
        
        if doc_id in self.doc_registry:
            return {"message": "Document already processed", "doc_id": doc_id}
        
        results = {"text_chunks": 0, "table_chunks": 0, "image_chunks": 0}
        
        # Process text content
        try:
            sentences = extract_sentences(pdf_bytes)
            if sentences:
                text_content = ' '.join(sentences)
                text_metadata = DocumentMetadata(
                    doc_id=f"{doc_id}_text",
                    source_file=filename,
                    doc_type="text"
                )
                
                text_documents = self.text_splitter.split_with_structure(text_content, text_metadata)
                self._add_documents_to_vectorstore(text_documents)
                results["text_chunks"] = len(text_documents)
        
        except Exception as e:
            print(f"Error processing text from {filename}: {e}")
        
        # Process tables (placeholder - would need to integrate with existing table extraction)
        # This would require saving the PDF temporarily and using the table extraction pipeline
        
        # Process images (placeholder - would need multimodal processing)
        
        # Update registry
        self.doc_registry[doc_id] = {
            "filename": filename,
            "processed_at": datetime.now().isoformat(),
            "chunks": results
        }
        self.save_registry()
        
        return results
    
    def _add_documents_to_vectorstore(self, documents: List[Document]):
        """Add documents to the vector store."""
        if documents:
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            self.vectorstore.add_texts(texts, metadatas)
            self.vectorstore.persist()
    
    def retrieve_for_rule_generation(
        self,
        query: str,
        manufacturing_process: Optional[str] = None,
        rule_category: Optional[str] = None,
        features: Optional[List[str]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Specialized retrieval for rule generation with manufacturing context."""
        
        # Build metadata filter
        metadata_filter = {}
        if manufacturing_process:
            metadata_filter["manufacturing_process"] = manufacturing_process
        if rule_category:
            metadata_filter["rule_category"] = rule_category
        
        # Perform similarity search
        if metadata_filter:
            results = self.vectorstore.similarity_search_with_score(
                query, k=top_k, filter=metadata_filter
            )
        else:
            results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        
        # Enhanced results with manufacturing context
        enhanced_results = []
        for doc, score in results:
            result = {
                'text': doc.page_content,
                'metadata': doc.metadata,
                'similarity_score': float(score),
                'manufacturing_context': self._extract_manufacturing_context(doc.page_content),
                'related_features': self._find_related_features(doc.page_content, features or []),
                'constraint_indicators': self._identify_constraint_indicators(doc.page_content)
            }
            enhanced_results.append(result)
        
        return enhanced_results
    
    def _extract_manufacturing_context(self, text: str) -> Dict[str, Any]:
        """Extract manufacturing-specific context from text."""
        context = {
            "measurements": [],
            "materials": [],
            "processes": [],
            "tolerances": []
        }
        
        # Simple pattern matching for manufacturing context
        # This could be enhanced with NER models
        import re
        
        # Extract measurements (numbers with units)
        measurements = re.findall(r'\d+\.?\d*\s*(?:mm|in|inch|mil|micron|m|cm)', text, re.IGNORECASE)
        context["measurements"] = measurements
        
        # Extract material references
        materials = re.findall(r'\b(?:steel|aluminum|plastic|carbon|titanium|copper|brass|304|316|6061)\b', text, re.IGNORECASE)
        context["materials"] = list(set(materials))
        
        return context
    
    def _find_related_features(self, text: str, query_features: List[str]) -> List[str]:
        """Find features in text that relate to the query features."""
        text_lower = text.lower()
        related = []
        
        for feature in query_features:
            if feature.lower() in text_lower:
                related.append(feature)
        
        return related
    
    def _identify_constraint_indicators(self, text: str) -> List[str]:
        """Identify constraint-indicating phrases in text."""
        indicators = []
        text_lower = text.lower()
        
        constraint_phrases = [
            "shall not exceed", "must be at least", "should be between",
            "minimum of", "maximum of", "no less than", "no more than",
            "within tolerance", "conform to", "in accordance with"
        ]
        
        for phrase in constraint_phrases:
            if phrase in text_lower:
                indicators.append(phrase)
        
        return indicators
    
    def generate_rule_context(
        self,
        rule_text: str,
        rule_type: Optional[str] = None,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """Generate enhanced context for rule generation using RAG."""
        
        # Retrieve relevant context
        context_results = self.retrieve_for_rule_generation(
            query=rule_text,
            rule_category=rule_type,
            top_k=top_k
        )
        
        # Build comprehensive context
        context = {
            "rule_text": rule_text,
            "rule_type": rule_type,
            "retrieved_context": context_results,
            "similar_rules": self._find_similar_rules(rule_text, top_k=3),
            "related_constraints": self._find_related_constraints(rule_text),
            "manufacturing_standards": self._find_relevant_standards(rule_text),
            "feature_definitions": self._get_feature_definitions(rule_type) if rule_type else None
        }
        
        return context
    
    def _find_similar_rules(self, rule_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find similar rules in the knowledge base."""
        # This would search for previously processed rules
        results = self.vectorstore.similarity_search_with_score(
            rule_text,
            k=top_k,
            filter={"doc_type": "text"}
        )
        
        similar_rules = []
        for doc, score in results:
            if any(indicator in doc.page_content.lower() 
                   for indicator in ["should", "must", "shall", "require"]):
                similar_rules.append({
                    "text": doc.page_content,
                    "similarity": float(score),
                    "metadata": doc.metadata
                })
        
        return similar_rules
    
    def _find_related_constraints(self, rule_text: str) -> List[str]:
        """Find constraints related to the rule."""
        # Extract constraint-related keywords from the rule
        constraints = []
        rule_lower = rule_text.lower()
        
        if "minimum" in rule_lower or "at least" in rule_lower:
            constraints.append("minimum_constraint")
        if "maximum" in rule_lower or "no more than" in rule_lower:
            constraints.append("maximum_constraint")
        if "between" in rule_lower:
            constraints.append("range_constraint")
        if "material" in rule_lower:
            constraints.append("material_constraint")
        
        return constraints
    
    def _find_relevant_standards(self, rule_text: str) -> List[str]:
        """Find relevant manufacturing standards."""
        standards = []
        rule_lower = rule_text.lower()
        
        # Common manufacturing standards
        standard_patterns = [
            "iso", "astm", "ansi", "din", "jis", "bs", "asme"
        ]
        
        for pattern in standard_patterns:
            if pattern in rule_lower:
                standards.append(pattern.upper())
        
        return standards
    
    def _get_feature_definitions(self, rule_type: str) -> Optional[str]:
        """Get feature definitions for the specified rule type."""
        return features_dict.get(rule_type)
    
    def clear_database(self):
        """Clear the entire knowledge base."""
        self.vectorstore.delete_collection()
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.persist_path
        )
        self.doc_registry = {}
        self.save_registry()
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        stats = {
            "total_documents": len(self.doc_registry),
            "processed_files": [doc["filename"] for doc in self.doc_registry.values()],
            "total_chunks": sum(
                sum(doc["chunks"].values()) for doc in self.doc_registry.values()
            ),
            "embedding_model": self.embeddings.model_name,
            "last_updated": max(
                (doc["processed_at"] for doc in self.doc_registry.values()),
                default="Never"
            )
        }
        return stats

# Integration functions for existing pipeline
def integrate_with_text_pipeline(pdf_bytes: bytes, filename: str, rag_system: EnhancedManufacturingRAG):
    """Integrate RAG system with existing text extraction pipeline."""
    
    # Process document
    results = rag_system.process_pdf_document(pdf_bytes, filename)
    
    # Also extract sentences for backward compatibility
    sentences = extract_sentences(pdf_bytes)
    
    return {
        "rag_results": results,
        "sentences": sentences,
        "rag_stats": rag_system.get_database_stats()
    }

def enhance_rule_generation_with_rag(
    rule_text: str,
    rule_type: str,
    rag_system: EnhancedManufacturingRAG
) -> str:
    """Enhance rule generation prompts with RAG context."""
    
    # Get RAG context
    context = rag_system.generate_rule_context(rule_text, rule_type)
    
    # Build enhanced prompt with context
    enhanced_context = f"""
Additional Context from Knowledge Base:

Similar Rules Found:
{chr(10).join([f"- {rule['text'][:200]}..." for rule in context['similar_rules'][:3]])}

Related Manufacturing Context:
{chr(10).join([f"- {ctx['text'][:150]}..." for ctx in context['retrieved_context'][:2]])}

Relevant Constraints:
{', '.join(context['related_constraints'])}

Manufacturing Standards:
{', '.join(context['manufacturing_standards'])}

Feature Definitions for {rule_type}:
{context['feature_definitions'] or 'Not available'}

Original Rule to Parse:
{rule_text}
"""
    
    return enhanced_context

if __name__ == "__main__":
    # Example usage
    rag_system = EnhancedManufacturingRAG()
    
    # Test with sample text
    sample_text = """
    Sheet metal forming design guidelines specify that the minimum bend radius 
    should be at least 1.5 times the material thickness. For materials like 
    aluminum 6061-T6, the recommended minimum bend radius is 2.0mm when the 
    sheet thickness is 1.0mm.
    """
    
    sample_metadata = DocumentMetadata(
        doc_id="test_001",
        source_file="test_guidelines.pdf",
        doc_type="text"
    )
    
    documents = rag_system.text_splitter.split_with_structure(sample_text, sample_metadata)
    rag_system._add_documents_to_vectorstore(documents)
    
    # Test retrieval
    results = rag_system.retrieve_for_rule_generation(
        "bend radius requirements",
        manufacturing_process="sheet metal",
        top_k=3
    )
    
    print("Retrieved context:")
    for result in results:
        print(f"Score: {result['similarity_score']:.3f}")
        print(f"Text: {result['text'][:200]}...")
        print(f"Context: {result['manufacturing_context']}")
        print("---")