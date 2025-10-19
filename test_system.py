#!/usr/bin/env python3
"""
Quick System Test Script
Run this to verify the RAG system is working correctly
"""

from core.universal_rag_system import UniversalManufacturingRAG
from core.implicit_rule_extractor import ImplicitRuleExtractor
import os
import sys

def test_system():
    """Run quick system validation tests."""
    
    print("="*70)
    print("RAG SYSTEM QUICK TEST")
    print("="*70)
    
    try:
        # Test 1: Initialize system
        print("\n[1/4] Initializing RAG system...")
        rag = UniversalManufacturingRAG(persist_path="./test_rag_db")
        print("      ✓ System initialized")
        
        # Test 2: Test implicit rule extraction
        print("\n[2/4] Testing implicit rule extraction...")
        extractor = ImplicitRuleExtractor()
        test_text = """
        Components must be designed to ensure proper functionality.
        Materials should withstand expected operating conditions.
        Adequate clearance must be provided for assembly.
        """
        rules = extractor.extract_implicit_rules(test_text, confidence_threshold=0.5)
        print(f"      ✓ Extracted {len(rules)} rules from vague text")
        
        # Test 3: Process a document
        print("\n[3/4] Testing document processing...")
        doc_path = "data/real_documents/Texas Instruments.pdf"
        if os.path.exists(doc_path):
            with open(doc_path, 'rb') as f:
                pdf_bytes = f.read()
            
            results = rag.process_any_document(pdf_bytes, "test_doc.pdf")
            print(f"      ✓ Processed {results['text_chunks']} chunks")
            print(f"      ✓ Method: {results['processing_method']}")
        else:
            print("      ⚠ Sample document not found, skipping")
        
        # Test 4: Test retrieval
        print("\n[4/4] Testing retrieval...")
        stats = rag.get_enhanced_stats()
        if stats['total_chunks'] > 0:
            query_results = rag.retrieve_with_fallback("quality requirements", top_k=3)
            print(f"      ✓ Retrieved {len(query_results)} results")
        else:
            print("      ✓ Retrieval system ready (no documents in DB yet)")
        
        # Summary
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)
        print("\nSystem Status:")
        print(f"  - Documents: {stats['total_documents']}")
        print(f"  - Chunks: {stats['total_chunks']}")
        print(f"  - Embedding Model: {stats['embedding_model']}")
        print(f"  - Capabilities: {len(stats['capabilities'])}")
        print("\nThe RAG system is operational and ready to use!")
        print("\nNext steps:")
        print("  1. Process more documents: rag.process_any_document(pdf_bytes, filename)")
        print("  2. Query the system: rag.retrieve_with_fallback(query)")
        print("  3. Run Streamlit UI: streamlit run main_app.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_system()
    sys.exit(0 if success else 1)
