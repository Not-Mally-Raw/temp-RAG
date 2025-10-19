"""
LLM-Integrated Text Processing Pipeline
Combines text extraction with LLM-based context understanding for accurate rule extraction
"""

import os
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Add workspace to path for imports
_current_file = Path(__file__).resolve()
_workspace_root = _current_file.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from extractors.text import extract_sentences
from core.llm_context_analyzer import (
    LLMContextAnalyzer, 
    DocumentContext, 
    check_api_availability,
    get_default_analyzer
)
from core.universal_rag_system import UniversalManufacturingRAG
from core.implicit_rule_extractor import ImplicitRuleExtractor

# Load environment variables
load_dotenv()


class LLMIntegratedPipeline:
    """
    Enhanced pipeline that uses LLM understanding for documents with zero keywords.
    Dramatically improves accuracy on generic documents.
    """
    
    def __init__(
        self,
        use_llm: bool = True,
        llm_provider: Optional[str] = None,
        rag_persist_path: str = None
    ):
        """
        Initialize the integrated pipeline.
        
        Args:
            use_llm: Whether to use LLM for context understanding (requires API key)
            llm_provider: "groq" or "cerebras" (or None for auto-detection)
            rag_persist_path: Path for RAG database
        """
        # LLM Setup
        self.use_llm = use_llm and check_api_availability()
        self.llm_analyzer = None
        
        if self.use_llm:
            try:
                if llm_provider:
                    self.llm_analyzer = LLMContextAnalyzer(api_provider=llm_provider)
                else:
                    self.llm_analyzer = get_default_analyzer()
                print(f"✓ LLM Analyzer initialized: {self.llm_analyzer.api_provider}")
            except Exception as e:
                print(f"⚠️  LLM initialization failed: {e}")
                print("   Falling back to non-LLM mode")
                self.use_llm = False
        
        # RAG System Setup
        rag_path = rag_persist_path or os.getenv("RAG_PERSIST_DIR", "./universal_rag_db")
        self.rag_system = UniversalManufacturingRAG(
            persist_path=rag_path,
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "800")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "100"))
        )
        
        # Fallback implicit extractor (for when LLM is not available)
        self.implicit_extractor = ImplicitRuleExtractor()
        
        # Configuration
        self.llm_confidence_threshold = float(os.getenv("LLM_CONFIDENCE_THRESHOLD", "0.6"))
        self.manufacturing_threshold = float(os.getenv("MANUFACTURING_RELEVANCE_THRESHOLD", "0.5"))
    
    def process_document(
        self, 
        pdf_bytes: bytes, 
        filename: str,
        use_llm_enhancement: bool = True
    ) -> Dict[str, Any]:
        """
        Process a document using LLM-enhanced understanding.
        
        Args:
            pdf_bytes: PDF file as bytes
            filename: Document filename
            use_llm_enhancement: Use LLM for enhanced understanding (overrides instance setting)
        
        Returns:
            Processing results with enhanced context
        """
        results = {
            "filename": filename,
            "extraction_method": [],
            "rules_extracted": 0,
            "manufacturing_relevance": 0.0,
            "document_context": None,
            "processing_stats": {}
        }
        
        # 1. Extract text using existing pipeline
        print(f"Extracting text from {filename}...")
        sentences = extract_sentences(pdf_bytes)
        full_text = " ".join(sentences)
        
        results["text_sentences"] = len(sentences)
        results["text_length"] = len(full_text)
        
        # 2. LLM-based context analysis (if available)
        if self.use_llm and use_llm_enhancement and self.llm_analyzer:
            print("Analyzing document context with LLM...")
            try:
                context = self.llm_analyzer.analyze_document_context(full_text)
                results["document_context"] = {
                    "industry": context.industry,
                    "domain": context.domain,
                    "purpose": context.purpose,
                    "key_concepts": context.key_concepts,
                    "manufacturing_relevance": context.manufacturing_relevance_score,
                    "confidence": context.confidence
                }
                results["manufacturing_relevance"] = context.manufacturing_relevance_score
                results["extraction_method"].append("llm_context_analysis")
                
                # 3. Extract rules using LLM understanding
                print("Extracting rules with LLM...")
                llm_rules = self.llm_analyzer.extract_manufacturing_rules(full_text, context)
                results["llm_rules"] = llm_rules
                results["rules_extracted"] += len(llm_rules)
                results["extraction_method"].append("llm_rule_extraction")
                
                # 4. Enhance text for better RAG indexing
                print("Enhancing text for RAG...")
                enhanced_data = self.llm_analyzer.enhance_text_for_rag(full_text)
                
                # Process with enhanced context
                print("Processing with RAG system...")
                rag_results = self.rag_system.process_any_document(
                    pdf_bytes, 
                    filename,
                    force_implicit_extraction=False
                )
                results["rag_processing"] = rag_results
                results["rules_extracted"] += rag_results.get("text_chunks", 0)
                results["extraction_method"].append("rag_hybrid")
                
            except Exception as e:
                print(f"LLM processing error: {e}")
                print("Falling back to standard processing...")
                results["llm_error"] = str(e)
                # Fall through to standard processing
        
        # Fallback: Standard processing (if LLM not available or failed)
        if not results["extraction_method"] or "error" in results:
            print("Using standard RAG processing...")
            rag_results = self.rag_system.process_any_document(pdf_bytes, filename)
            results["rag_processing"] = rag_results
            results["rules_extracted"] = rag_results.get("text_chunks", 0)
            results["extraction_method"].append("standard_rag")
            
            # Use implicit extractor for relevance estimation
            sample_text = full_text[:2000]  # Sample for analysis
            implicit_rules = self.implicit_extractor.extract_implicit_rules(sample_text, confidence_threshold=0.5)
            if implicit_rules:
                avg_relevance = sum(r.manufacturing_relevance for r in implicit_rules) / len(implicit_rules)
                results["manufacturing_relevance"] = avg_relevance
        
        # Processing statistics
        results["processing_stats"] = {
            "total_sentences": len(sentences),
            "llm_enhanced": self.use_llm and use_llm_enhancement,
            "extraction_methods": results["extraction_method"],
            "rules_per_sentence": results["rules_extracted"] / max(len(sentences), 1)
        }
        
        return results
    
    def analyze_sentences_batch(
        self, 
        sentences: List[str],
        use_llm: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Analyze a batch of sentences for manufacturing relevance.
        
        Args:
            sentences: List of sentences to analyze
            use_llm: Use LLM for analysis (much more accurate)
        
        Returns:
            List of analyzed sentences with relevance scores
        """
        if self.use_llm and use_llm and self.llm_analyzer:
            print(f"Analyzing {len(sentences)} sentences with LLM...")
            return self.llm_analyzer.batch_analyze_sentences(sentences)
        else:
            print(f"Analyzing {len(sentences)} sentences with implicit extraction...")
            results = []
            for sentence in sentences:
                implicit_rules = self.implicit_extractor.extract_implicit_rules(
                    sentence, 
                    confidence_threshold=0.3
                )
                
                if implicit_rules:
                    rule = implicit_rules[0]
                    result = {
                        "sentence": sentence,
                        "manufacturing_relevance": rule.manufacturing_relevance,
                        "implicit_requirements": [rule.text],
                        "domain_context": rule.rule_type,
                        "key_concepts": rule.semantic_features[:5]
                    }
                else:
                    result = {
                        "sentence": sentence,
                        "manufacturing_relevance": 0.0,
                        "implicit_requirements": [],
                        "domain_context": "",
                        "key_concepts": []
                    }
                results.append(result)
            
            return results
    
    def search_with_context(
        self,
        query: str,
        top_k: int = 5,
        use_llm_enhancement: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search RAG database with LLM-enhanced query understanding.
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_llm_enhancement: Use LLM to enhance query understanding
        
        Returns:
            Search results with enhanced context
        """
        # Enhance query with LLM if available
        if self.use_llm and use_llm_enhancement and self.llm_analyzer:
            # Analyze query context
            query_context = self.llm_analyzer.analyze_document_context(query)
            
            # Generate better search keywords
            search_keywords = self.llm_analyzer._generate_search_keywords(query_context)
            enhanced_query = f"{query} {' '.join(search_keywords)}"
        else:
            enhanced_query = query
        
        # Search RAG system
        results = self.rag_system.retrieve_with_fallback(
            enhanced_query,
            top_k=top_k,
            include_implicit=True
        )
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        api_availability = check_api_availability()
        
        status = {
            "llm_available": self.use_llm,
            "llm_provider": self.llm_analyzer.api_provider if self.llm_analyzer else None,
            "llm_model": self.llm_analyzer.model if self.llm_analyzer else None,
            "api_availability": api_availability,
            "rag_stats": self.rag_system.get_enhanced_stats(),
            "configuration": {
                "llm_confidence_threshold": self.llm_confidence_threshold,
                "manufacturing_threshold": self.manufacturing_threshold,
                "rag_persist_path": self.rag_system.persist_path
            }
        }
        
        return status
    
    def setup_instructions(self) -> str:
        """Get setup instructions for API keys."""
        api_availability = check_api_availability()
        
        if all(api_availability.values()):
            return "✅ All LLM APIs are configured and ready!"
        
        instructions = """
╔══════════════════════════════════════════════════════════════════════╗
║                    LLM API SETUP REQUIRED                            ║
╚══════════════════════════════════════════════════════════════════════╝

To enable LLM-enhanced context understanding (dramatically improves accuracy
on generic documents with zero keywords), you need an API key:

📝 OPTION 1: GROQ (Recommended - Fast & Free tier available)
   1. Go to: https://console.groq.com/keys
   2. Sign up (free)
   3. Generate an API key
   4. Set environment variable:
      export GROQ_API_KEY="your-key-here"

📝 OPTION 2: CEREBRAS (Alternative - Also has free tier)
   1. Go to: https://cloud.cerebras.ai/
   2. Sign up
   3. Get your API key
   4. Set environment variable:
      export CEREBRAS_API_KEY="your-key-here"

💡 You only need ONE of the above APIs.

🚀 After setting the API key, restart your application.

Current Status:
"""
        if not api_availability["groq"]:
            instructions += "  ❌ Groq: Not configured\n"
        else:
            instructions += "  ✅ Groq: Ready\n"
        
        if not api_availability["cerebras"]:
            instructions += "  ❌ Cerebras: Not configured\n"
        else:
            instructions += "  ✅ Cerebras: Ready\n"
        
        return instructions


if __name__ == "__main__":
    # Test the pipeline
    print("="*70)
    print("LLM-INTEGRATED PIPELINE TEST")
    print("="*70)
    
    # Check API availability
    availability = check_api_availability()
    print(f"\nAPI Availability: {availability}")
    
    if not any(availability.values()):
        print("\n⚠️  No LLM APIs configured.")
        pipeline = LLMIntegratedPipeline(use_llm=False)
        print(pipeline.setup_instructions())
    else:
        print("\n✅ LLM API available!")
        
        # Initialize pipeline
        pipeline = LLMIntegratedPipeline(use_llm=True)
        
        # Get system status
        status = pipeline.get_system_status()
        print(f"\nSystem Status:")
        print(f"  LLM Provider: {status['llm_provider']}")
        print(f"  LLM Model: {status['llm_model']}")
        print(f"  RAG Database: {status['rag_stats']['total_documents']} documents")
        
        print("\n✅ Pipeline ready for high-accuracy document processing!")
