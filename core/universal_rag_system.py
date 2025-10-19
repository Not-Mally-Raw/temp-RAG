"""
Enhanced RAG Database with Implicit Rule Processing
Extends the existing RAG system to handle documents without clear manufacturing keywords
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

# Import the implicit rule extractor
from core.implicit_rule_extractor import ImplicitRuleExtractor, ImplicitRule

# Import existing components
from core.enhanced_rag_db import (
    DocumentMetadata, SentenceTransformerEmbeddings, 
    ManufacturingTextSplitter, EnhancedManufacturingRAG
)

@dataclass
class EnhancedDocumentMetadata(DocumentMetadata):
    """Extended metadata for handling implicit rules."""
    implicit_rules_count: Optional[int] = None
    semantic_features: Optional[List[str]] = None
    manufacturing_relevance_score: Optional[float] = None
    rule_extraction_method: Optional[str] = None  # 'keyword_based', 'semantic_based', 'hybrid'
    confidence_distribution: Optional[Dict[str, int]] = None  # High, medium, low confidence counts

class UniversalManufacturingRAG(EnhancedManufacturingRAG):
    """Enhanced RAG system that can handle any type of document."""
    
    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-large-en-v1.5",
        persist_path: str = "universal_rag_db",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        implicit_rule_threshold: float = 0.6
    ):
        super().__init__(embedding_model_name, persist_path, chunk_size, chunk_overlap)
        
        # Initialize implicit rule extractor
        self.implicit_extractor = ImplicitRuleExtractor()
        self.implicit_rule_threshold = implicit_rule_threshold
        
        # Track processing statistics
        self.processing_stats = {
            'keyword_based_rules': 0,
            'implicit_rules': 0,
            'hybrid_rules': 0,
            'documents_processed': 0
        }
    
    def process_any_document(
        self, 
        pdf_bytes: bytes, 
        filename: str,
        force_implicit_extraction: bool = False
    ) -> Dict[str, Any]:
        """Process any document - with or without manufacturing keywords."""
        
        doc_id = hashlib.md5(pdf_bytes).hexdigest()[:16]
        
        if doc_id in self.doc_registry:
            return {"message": "Document already processed", "doc_id": doc_id}
        
        results = {
            "text_chunks": 0, 
            "implicit_rules": 0,
            "keyword_rules": 0,
            "hybrid_rules": 0,
            "processing_method": "unknown"
        }
        
        try:
            # Extract text content
            from extractors.text import extract_sentences
            sentences = extract_sentences(pdf_bytes)
            
            if not sentences:
                return {"error": "No text content extracted"}
            
            text_content = ' '.join(sentences)
            
            # Determine processing approach
            processing_method = self._determine_processing_method(text_content, force_implicit_extraction)
            results["processing_method"] = processing_method
            
            if processing_method == "keyword_based":
                # Use existing keyword-based processing
                processed_docs = self._process_with_keywords(text_content, filename, doc_id)
                results["keyword_rules"] = len(processed_docs)
                self.processing_stats['keyword_based_rules'] += len(processed_docs)
                
            elif processing_method == "implicit_based":
                # Use implicit rule extraction
                processed_docs = self._process_with_implicit_extraction(text_content, filename, doc_id)
                results["implicit_rules"] = len(processed_docs)
                self.processing_stats['implicit_rules'] += len(processed_docs)
                
            else:  # hybrid
                # Use both approaches and combine results
                keyword_docs = self._process_with_keywords(text_content, filename, doc_id)
                implicit_docs = self._process_with_implicit_extraction(text_content, filename, doc_id)
                
                # Combine and deduplicate
                processed_docs = self._combine_and_deduplicate(keyword_docs, implicit_docs)
                
                results["keyword_rules"] = len(keyword_docs)
                results["implicit_rules"] = len(implicit_docs)
                results["hybrid_rules"] = len(processed_docs)
                
                self.processing_stats['keyword_based_rules'] += len(keyword_docs)
                self.processing_stats['implicit_rules'] += len(implicit_docs)
                self.processing_stats['hybrid_rules'] += len(processed_docs)
            
            # Add to vector store
            if processed_docs:
                self._add_documents_to_vectorstore(processed_docs)
                results["text_chunks"] = len(processed_docs)
            
            # Update registry with enhanced metadata
            self.doc_registry[doc_id] = {
                "filename": filename,
                "processed_at": datetime.now().isoformat(),
                "chunks": results,
                "processing_method": processing_method,
                "stats": self.processing_stats.copy()
            }
            self.save_registry()
            
            self.processing_stats['documents_processed'] += 1
            
        except Exception as e:
            results["error"] = str(e)
        
        return results
    
    def _determine_processing_method(self, text: str, force_implicit: bool = False) -> str:
        """Determine the best processing method based on text analysis."""
        
        if force_implicit:
            return "implicit_based"
        
        # Check for manufacturing keywords
        manufacturing_keyword_density = self._calculate_keyword_density(text)
        
        # Check manufacturing relevance using semantic analysis
        manufacturing_relevance = self.implicit_extractor._calculate_manufacturing_relevance(text)
        
        # Decision logic
        if manufacturing_keyword_density > 0.3:  # High keyword density
            if manufacturing_relevance > 0.7:
                return "hybrid"  # Use both for maximum coverage
            else:
                return "keyword_based"  # Stick with keywords
        elif manufacturing_relevance > 0.5:  # Semantically relevant but few keywords
            return "implicit_based"
        else:
            return "hybrid"  # Try both approaches
    
    def _calculate_keyword_density(self, text: str) -> float:
        """Calculate density of manufacturing keywords in text."""
        
        manufacturing_keywords = [
            'manufacturing', 'machining', 'assembly', 'injection', 'molding', 
            'sheet', 'metal', 'casting', 'welding', 'drilling', 'milling',
            'tolerance', 'dimension', 'specification', 'material', 'process',
            'bend', 'radius', 'thickness', 'diameter', 'surface', 'finish'
        ]
        
        text_lower = text.lower()
        words = text_lower.split()
        
        if not words:
            return 0.0
        
        keyword_count = sum(1 for word in words if any(kw in word for kw in manufacturing_keywords))
        return keyword_count / len(words)
    
    def _process_with_keywords(self, text: str, filename: str, doc_id: str) -> List[Document]:
        """Process document using existing keyword-based approach."""
        
        text_metadata = DocumentMetadata(
            doc_id=f"{doc_id}_keyword",
            source_file=filename,
            doc_type="text"
        )
        
        # Add rule_extraction_method to metadata after creation
        docs = self.text_splitter.split_with_structure(text, text_metadata)
        for doc in docs:
            doc.metadata['rule_extraction_method'] = 'keyword_based'
        
        return docs
    
    def _process_with_implicit_extraction(self, text: str, filename: str, doc_id: str) -> List[Document]:
        """Process document using implicit rule extraction."""
        
        # Extract implicit rules
        implicit_rules = self.implicit_extractor.extract_implicit_rules(
            text, confidence_threshold=self.implicit_rule_threshold
        )
        
        documents = []
        
        for i, rule in enumerate(implicit_rules):
            # Create enhanced metadata
            enhanced_metadata = EnhancedDocumentMetadata(
                doc_id=f"{doc_id}_implicit_{i}",
                source_file=filename,
                doc_type="text",
                manufacturing_process=self._map_rule_type_to_process(rule.rule_type),
                rule_category=rule.rule_type,
                chunk_index=i,
                confidence_score=rule.confidence_score,
                features=rule.semantic_features[:10],  # Top 10 features
                constraints=[rule.constraint_type],
                implicit_rules_count=1,
                semantic_features=rule.semantic_features,
                manufacturing_relevance_score=rule.manufacturing_relevance,
                rule_extraction_method="semantic_based"
            )
            
            document = Document(
                page_content=rule.text,
                metadata=enhanced_metadata.to_dict()
            )
            
            documents.append(document)
        
        return documents
    
    def _map_rule_type_to_process(self, rule_type: str) -> Optional[str]:
        """Map semantic rule types to manufacturing processes."""
        
        mapping = {
            "mechanical design": "General",
            "manufacturing process": "General",
            "quality control": "Assembly",
            "assembly procedure": "Assembly", 
            "material specification": "General",
            "dimensional tolerance": "Machining",
            "surface finish": "Machining",
            "structural integrity": "General",
            "thermal management": "General",
            "electrical specification": "Assembly"
        }
        
        return mapping.get(rule_type, "General")
    
    def _combine_and_deduplicate(
        self, 
        keyword_docs: List[Document], 
        implicit_docs: List[Document]
    ) -> List[Document]:
        """Combine keyword and implicit results, removing duplicates."""
        
        combined = []
        seen_texts = set()
        
        # Add keyword-based documents first (higher priority)
        for doc in keyword_docs:
            text_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                # Mark as hybrid processing
                doc.metadata['rule_extraction_method'] = 'hybrid_keyword_priority'
                combined.append(doc)
        
        # Add implicit documents that don't overlap
        for doc in implicit_docs:
            text_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                # Mark as hybrid processing
                doc.metadata['rule_extraction_method'] = 'hybrid_implicit_addition'
                combined.append(doc)
        
        return combined
    
    def retrieve_with_fallback(
        self,
        query: str,
        manufacturing_process: Optional[str] = None,
        rule_category: Optional[str] = None,
        features: Optional[List[str]] = None,
        top_k: int = 5,
        include_implicit: bool = True
    ) -> List[Dict[str, Any]]:
        """Enhanced retrieval that includes implicit rules when needed."""
        
        # First, try standard retrieval
        results = self.retrieve_for_rule_generation(
            query, manufacturing_process, rule_category, features, top_k
        )
        
        # If few results and implicit rules are available, include them
        if len(results) < top_k // 2 and include_implicit:
            implicit_results = self._retrieve_implicit_rules(query, top_k - len(results))
            results.extend(implicit_results)
        
        return results
    
    def _retrieve_implicit_rules(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Retrieve implicit rules based on semantic similarity."""
        
        # Search specifically for implicitly extracted rules
        implicit_filter = {"rule_extraction_method": "semantic_based"}
        
        try:
            results = self.vectorstore.similarity_search_with_score(
                query, k=max_results, filter=implicit_filter
            )
            
            enhanced_results = []
            for doc, score in results:
                result = {
                    'text': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': float(score),
                    'extraction_method': 'implicit',
                    'manufacturing_relevance': doc.metadata.get('manufacturing_relevance_score', 0.0),
                    'semantic_features': doc.metadata.get('semantic_features', [])
                }
                enhanced_results.append(result)
            
            return enhanced_results
            
        except Exception as e:
            print(f"Error retrieving implicit rules: {e}")
            return []
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics including implicit rule processing."""
        
        base_stats = self.get_database_stats()
        
        enhanced_stats = {
            **base_stats,
            "processing_methods": {
                "keyword_based_rules": self.processing_stats['keyword_based_rules'],
                "implicit_rules": self.processing_stats['implicit_rules'],
                "hybrid_rules": self.processing_stats['hybrid_rules'],
                "total_documents": self.processing_stats['documents_processed']
            },
            "implicit_extraction_threshold": self.implicit_rule_threshold,
            "capabilities": [
                "Keyword-based rule extraction",
                "Semantic rule extraction", 
                "Hybrid processing",
                "Manufacturing relevance scoring",
                "Cross-document learning"
            ]
        }
        
        return enhanced_stats
    
    def analyze_document_type(self, text: str) -> Dict[str, Any]:
        """Analyze document to determine best processing approach."""
        
        analysis = {
            "manufacturing_keyword_density": self._calculate_keyword_density(text),
            "manufacturing_relevance": self.implicit_extractor._calculate_manufacturing_relevance(text),
            "recommended_method": self._determine_processing_method(text),
            "estimated_rules": 0,
            "confidence": "unknown"
        }
        
        # Estimate number of potential rules
        sentences = text.split('.')
        potential_rules = [s for s in sentences if any(
            indicator in s.lower() for indicator in 
            ['should', 'must', 'shall', 'require', 'need', 'ensure', 'avoid', 'prevent']
        )]
        
        analysis["estimated_rules"] = len(potential_rules)
        
        # Determine confidence level
        if analysis["manufacturing_keyword_density"] > 0.2 or analysis["manufacturing_relevance"] > 0.6:
            analysis["confidence"] = "high"
        elif analysis["manufacturing_keyword_density"] > 0.1 or analysis["manufacturing_relevance"] > 0.4:
            analysis["confidence"] = "medium"
        else:
            analysis["confidence"] = "low"
        
        return analysis

# Integration functions
def process_random_document(pdf_bytes: bytes, filename: str, rag_system: UniversalManufacturingRAG):
    """Process any type of document, regardless of content."""
    
    # Analyze document first
    from extractors.text import extract_sentences
    sentences = extract_sentences(pdf_bytes)
    text_content = ' '.join(sentences) if sentences else ""
    
    analysis = rag_system.analyze_document_type(text_content)
    
    # Process based on analysis
    results = rag_system.process_any_document(pdf_bytes, filename)
    
    return {
        "document_analysis": analysis,
        "processing_results": results,
        "recommendations": _generate_processing_recommendations(analysis, results)
    }

def _generate_processing_recommendations(analysis: Dict[str, Any], results: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on processing results."""
    
    recommendations = []
    
    if analysis["confidence"] == "low":
        recommendations.append("Consider adding more manufacturing context to improve rule extraction")
    
    if results.get("implicit_rules", 0) > results.get("keyword_rules", 0):
        recommendations.append("Document processed primarily with semantic analysis - review extracted rules carefully")
    
    if analysis["estimated_rules"] > (results.get("text_chunks", 0) * 0.5):
        recommendations.append("Many potential rules detected - consider manual review for completeness")
    
    if analysis["manufacturing_relevance"] < 0.5:
        recommendations.append("Low manufacturing relevance detected - verify rule applicability")
    
    return recommendations

if __name__ == "__main__":
    # Test the enhanced system
    rag_system = UniversalManufacturingRAG()
    
    # Test with non-manufacturing text
    test_text = """
    Software components should maintain appropriate interfaces to prevent integration issues.
    Database connections must be configured considering network latency effects.
    User interface elements require adequate spacing for accessibility compliance.
    System processes need sufficient memory allocation to handle expected workloads.
    """
    
    analysis = rag_system.analyze_document_type(test_text)
    print("Document Analysis:")
    print(json.dumps(analysis, indent=2))