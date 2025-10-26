"""
RAG Pipeline Integration
Connects the enhanced RAG system with the existing text processing pipeline
"""

import streamlit as st
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import json
import os
import tempfile
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Import the RAG system
from .rag_database import ManufacturingRAG, DocumentMetadata
from .implicit_rule_extractor import ImplicitRuleExtractor
from .rule_extractor import RuleExtractor, ExtractionConfig

# Import extractors
from extractors.text import extract_sentences
from extractors.table import dual_pipeline_2
from extractors.image import extract_images

class RAGIntegratedPipeline:
    """Integrated pipeline that combines existing extraction with RAG capabilities."""
    
    def __init__(self, collection_name: str = "manufacturing_rag", persist_directory: str = "./rag_db"):
        """Initialize the RAG pipeline integration."""
        self.rag_system = ManufacturingRAG(
            persist_path=persist_directory
        )
        self.implicit_extractor = ImplicitRuleExtractor()
        
        # Initialize rule extractor
        try:
            extraction_config = ExtractionConfig(
                max_rule_length=200,
                max_rules_per_chunk=10,
                min_confidence_threshold=0.3,
                enable_rule_refinement=True,
                enable_context_analysis=True
            )
            self.rule_extractor = RuleExtractor(config=extraction_config)
            logger.info("Rule extractor initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize extractor: {e}")
            self.rule_extractor = None
        
        self.temp_dir = None
        
    def process_uploaded_file(self, uploaded_file) -> Dict[str, Any]:
        """Process uploaded PDF with both traditional extraction and RAG integration."""
        
        # Read file bytes
        pdf_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name
        
        # Create temporary directory for file processing
        self.temp_dir = tempfile.mkdtemp()
        temp_pdf_path = os.path.join(self.temp_dir, filename)
        
        with open(temp_pdf_path, 'wb') as f:
            f.write(pdf_bytes)
        
        results = {}
        
        try:
            # 1. Traditional text extraction (for backward compatibility)
            sentences = extract_sentences(pdf_bytes)
            results['text_sentences'] = sentences
            results['text_count'] = len(sentences) if sentences else 0
            
            # 2. Enhanced RAG processing
            rag_results = self.rag_system.process_pdf_document(pdf_bytes, filename)
            results['rag_processing'] = rag_results
            
            # 3. Table extraction (using existing pipeline)
            try:
                dual_pipeline_2(temp_pdf_path)
                results['table_extraction'] = "completed"
            except Exception as e:
                results['table_extraction'] = f"failed: {e}"
            
            # 4. Image extraction
            try:
                images_path = os.path.join(self.temp_dir, "images")
                os.makedirs(images_path, exist_ok=True)
                extract_images(pdf_bytes, images_path)
                
                # Count extracted images
                image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                results['image_extraction'] = {
                    "status": "completed",
                    "images_found": len(image_files),
                    "images_path": images_path
                }
            except Exception as e:
                results['image_extraction'] = {"status": f"failed: {e}"}
            
            # 5. Generate RAG statistics
            results['rag_stats'] = self.rag_system.get_database_stats()
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def enhance_classification_with_rag(
        self, 
        text_sentences: List[str], 
        confidence_threshold: float = 0.90
    ) -> List[Dict[str, Any]]:
        """Enhance classification results with RAG context."""
        
        enhanced_rules = []
        
        for sentence in text_sentences:
            if not sentence.strip():
                continue
                
            # Get RAG context for each potential rule
            rag_context = self.rag_system.retrieve_for_rule_generation(
                query=sentence,
                top_k=3
            )
            
            # Enhance with manufacturing context
            enhanced_rule = {
                "text": sentence,
                "rag_context": rag_context,
                "manufacturing_features": self._extract_manufacturing_features(sentence),
                "constraint_indicators": self._identify_constraints(sentence),
                "suggested_rule_type": self._suggest_rule_type(sentence, rag_context)
            }
            
            enhanced_rules.append(enhanced_rule)
        
        return enhanced_rules
    
    def _extract_manufacturing_features(self, text: str) -> List[str]:
        """Extract manufacturing features from text."""
        features = []
        text_lower = text.lower()
        
        feature_keywords = {
            "dimensional": ["length", "width", "height", "diameter", "radius", "thickness"],
            "geometric": ["angle", "bend", "flange", "hole", "boss", "rib"],
            "surface": ["finish", "roughness", "tolerance", "flatness"],
            "material": ["material", "steel", "aluminum", "plastic", "carbon"],
            "process": ["machining", "molding", "casting", "forming", "assembly"]
        }
        
        for category, keywords in feature_keywords.items():
            found_keywords = [kw for kw in keywords if kw in text_lower]
            if found_keywords:
                features.extend(found_keywords)
        
        return list(set(features))
    
    def _identify_constraints(self, text: str) -> List[str]:
        """Identify constraint types in text."""
        constraints = []
        text_lower = text.lower()
        
        constraint_patterns = {
            "minimum": ["minimum", "at least", "no less than", "greater than"],
            "maximum": ["maximum", "at most", "no more than", "less than"],
            "range": ["between", "from", "to", "within"],
            "equality": ["equal to", "exactly", "precisely"],
            "material": ["material", "grade", "type", "specification"]
        }
        
        for constraint_type, patterns in constraint_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                constraints.append(constraint_type)
        
        return constraints
    
    def _suggest_rule_type(self, text: str, rag_context: List[Dict]) -> Optional[str]:
        """Suggest rule type based on text content and RAG context."""
        
        # First, try to infer from RAG context
        for context in rag_context:
            if context.get('metadata', {}).get('rule_category'):
                return context['metadata']['rule_category']
        
        # Fallback to keyword-based inference
        text_lower = text.lower()
        
        type_keywords = {
            "Sheet Metal": ["sheet", "metal", "bend", "flange", "stamping"],
            "Injection Molding": ["injection", "molding", "plastic", "mold"],
            "Assembly": ["assembly", "fastener", "bolt", "screw"],
            "Machining": ["machining", "drilling", "milling", "turning"],
            "General": ["general", "standard", "common"]
        }
        
        for rule_type, keywords in type_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return rule_type
        
        return None
    
    def generate_enhanced_rule_prompt(
        self, 
        rule_text: str, 
        rule_type: str
    ) -> str:
        """Generate enhanced prompts for rule generation using RAG context."""
        
        try:
            # Query RAG system for context
            rag_results = self.rag_system.retrieve_for_rule_generation(
                query=f"Rules similar to: {rule_text}",
                top_k=5
            )
            
            # Prepare context for enhanced extraction  
            if isinstance(rag_results, list):
                # Handle list response from retrieve_for_rule_generation
                rag_context = {
                    'retrieved_context': [{'text': result.get('text', ''), 'metadata': result.get('metadata', {})} 
                                        for result in rag_results],
                    'manufacturing_features': [],
                    'related_constraints': [],
                    'manufacturing_standards': []
                }
            else:
                # Handle dict response (fallback)
                rag_context = {
                    'retrieved_context': [{'text': result.page_content, 'metadata': result.metadata} 
                                        for result in rag_results.get('documents', [])],
                    'manufacturing_features': rag_results.get('manufacturing_features', []),
                    'related_constraints': rag_results.get('related_constraints', []),
                    'manufacturing_standards': rag_results.get('manufacturing_standards', [])
                }
            
            # Use enhanced rule extractor if available
            if hasattr(self, 'rule_extractor'):
                extraction_result = self.rule_extractor.extract_rules_enhanced(
                    rule_text, rag_context
                )
                
                if extraction_result.rules:
                    # Convert to expected format
                    enhanced_rule = extraction_result.rules[0]  # Take the best one
                    return json.dumps({
                        "success": True,
                        "enhanced_rule": enhanced_rule.dict(),
                        "rag_context": rag_context,
                        "processing_metadata": extraction_result.processing_metadata
                    }, indent=2)
            
            # Fallback to basic enhancement
            return json.dumps({
                "success": True,
                "enhanced_rule": {
                    "rule_category": rule_type,
                    "name": rule_text[:100],
                    "confidence": 0.6,
                    "rag_context_available": bool(rag_context.get('retrieved_context'))
                },
                "rag_context": rag_context
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Rule enhancement error: {e}")
            return json.dumps({"success": False, "error": str(e), "rule_text": rule_text})
    
    def get_knowledge_base_summary(self) -> Dict[str, Any]:
        """Get summary of the knowledge base for display."""
        return self.rag_system.get_database_stats()
    
    def search_knowledge_base(
        self, 
        query: str, 
        manufacturing_process: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search the knowledge base with manufacturing context."""
        
        return self.rag_system.retrieve_for_rule_generation(
            query=query,
            manufacturing_process=manufacturing_process,
            top_k=top_k
        )
    
    def cleanup_temp_files(self):
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None

# Streamlit integration functions
def init_rag_pipeline():
    """Initialize RAG pipeline in Streamlit session state."""
    if 'rag_pipeline' not in st.session_state:
        st.session_state['rag_pipeline'] = RAGIntegratedPipeline()
    return st.session_state['rag_pipeline']

def display_rag_stats(rag_pipeline: RAGIntegratedPipeline):
    """Display RAG system statistics in Streamlit."""
    stats = rag_pipeline.get_knowledge_base_summary()
    
    st.sidebar.subheader("📊 Knowledge Base Stats")
    st.sidebar.metric("Total Documents", stats['total_documents'])
    st.sidebar.metric("Total Chunks", stats['total_chunks'])
    st.sidebar.write(f"**Embedding Model:** {stats['embedding_model']}")
    st.sidebar.write(f"**Last Updated:** {stats['last_updated']}")
    
    if stats['processed_files']:
        with st.sidebar.expander("📁 Processed Files"):
            for filename in stats['processed_files']:
                st.write(f"• {filename}")

def add_rag_search_interface(rag_pipeline: RAGIntegratedPipeline):
    """Add RAG search interface to Streamlit sidebar."""
    st.sidebar.subheader("🔍 Search Knowledge Base")
    
    search_query = st.sidebar.text_input("Search Query", key="rag_search")
    manufacturing_process = st.sidebar.selectbox(
        "Manufacturing Process",
        ["None", "Sheet Metal", "Injection Molding", "Assembly", "Machining", "General"],
        key="rag_process_filter"
    )
    
    if st.sidebar.button("Search", key="rag_search_btn") and search_query:
        process_filter = None if manufacturing_process == "None" else manufacturing_process
        
        results = rag_pipeline.search_knowledge_base(
            query=search_query,
            manufacturing_process=process_filter,
            top_k=5
        )
        
        st.sidebar.subheader("Search Results")
        for i, result in enumerate(results, 1):
            with st.sidebar.expander(f"Result {i} (Score: {result['similarity_score']:.3f})"):
                st.write(f"**Text:** {result['text'][:200]}...")
                st.write(f"**Source:** {result['metadata'].get('source_file', 'Unknown')}")
                if result['manufacturing_context']:
                    st.write(f"**Context:** {result['manufacturing_context']}")

if __name__ == "__main__":
    # Example usage for testing
    pipeline = RAGIntegratedPipeline()
    
    # Test search
    results = pipeline.search_knowledge_base("bend radius requirements")
    print("Search results:", len(results))
    
    # Test feature extraction
    sample_text = "The minimum bend radius should be at least 1.5 times the material thickness"
    features = pipeline._extract_manufacturing_features(sample_text)
    constraints = pipeline._identify_constraints(sample_text)
    
    print(f"Features: {features}")
    print(f"Constraints: {constraints}")