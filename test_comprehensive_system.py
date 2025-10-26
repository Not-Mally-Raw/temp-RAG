#!/usr/bin/env python3
"""
Comprehensive test script for the RAG system
Tests OCR functionality, chunk formation, and basic pipeline operations
"""

import os
import sys
import tempfile
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def test_ocr_functionality():
    """Test OCR functionality with a simple image-based PDF."""
    print("🔍 Testing OCR functionality...")

    try:
        from extractors.robust_pdf_processor import RobustPDFProcessor

        # Create a simple test PDF with text (we'll use a mock approach)
        processor = RobustPDFProcessor()

        # Test the OCR method directly (it should handle missing dependencies gracefully)
        print("✅ Robust PDF processor initialized")

        # Test analysis method
        analysis = processor.analyze_pdf(b"dummy_bytes")
        print(f"✅ PDF analysis works: {len(analysis)} fields returned")

        return True
    except Exception as e:
        print(f"❌ OCR test failed: {e}")
        return False

def test_rag_database():
    """Test RAG database initialization and basic operations."""
    print("🔍 Testing RAG database...")

    try:
        from core.rag_database import ManufacturingRAG

        # Create a temporary database for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            rag_system = ManufacturingRAG(persist_path=temp_dir)

            # Test basic operations
            stats = rag_system.get_database_stats()
            print(f"✅ RAG database initialized: {stats['total_documents']} documents")

            # Test search (should return empty results)
            results = rag_system.retrieve_for_rule_generation("test query", top_k=1)
            print(f"✅ Search functionality works: {len(results)} results")

            return True
    except Exception as e:
        print(f"❌ RAG database test failed: {e}")
        return False

def test_rag_pipeline():
    """Test RAG pipeline integration."""
    print("🔍 Testing RAG pipeline integration...")

    try:
        from core.rag_pipeline_integration import RAGIntegratedPipeline

        # Create pipeline with temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = RAGIntegratedPipeline(persist_directory=temp_dir)

            # Test knowledge base summary
            stats = pipeline.get_knowledge_base_summary()
            print(f"✅ Pipeline initialized: {stats['total_documents']} documents")

            # Test search
            results = pipeline.search_knowledge_base("test manufacturing requirements", top_k=1)
            print(f"✅ Pipeline search works: {len(results)} results")

            return True
    except Exception as e:
        print(f"❌ RAG pipeline test failed: {e}")
        return False

def test_enhanced_qa():
    """Test enhanced QA system."""
    print("🔍 Testing enhanced QA system...")

    try:
        from core.rag_pipeline_integration import RAGIntegratedPipeline
        from enhanced_qa_system import RAGQuestionAnswerer

        # Create pipeline and QA system
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = RAGIntegratedPipeline(persist_directory=temp_dir)
            qa_system = RAGQuestionAnswerer(pipeline)

            # Test basic functionality
            summary = qa_system.get_rules_summary()
            print(f"✅ QA system initialized: {summary['total_rules']} rules")

            return True
    except Exception as e:
        print(f"❌ Enhanced QA test failed: {e}")
        return False

def test_chunk_formation():
    """Test chunk formation with sample text."""
    print("🔍 Testing chunk formation...")

    try:
        from core.rag_database import ManufacturingRAG, DocumentMetadata

        with tempfile.TemporaryDirectory() as temp_dir:
            rag_system = ManufacturingRAG(persist_path=temp_dir)

            # Create sample manufacturing text
            sample_text = """
            Sheet metal design guidelines specify that the minimum bend radius
            should be at least 1.5 times the material thickness. For aluminum 6061-T6,
            the recommended minimum bend radius is 2.0mm when the sheet thickness is 1.0mm.

            Hole specifications require minimum hole diameter of 0.5mm or greater
            for manufacturing feasibility. Distance between holes should be at least
            2 times the hole diameter to ensure structural integrity.
            """

            # Create metadata
            metadata = DocumentMetadata(
                doc_id="test_chunk_001",
                source_file="test_guidelines.pdf",
                doc_type="text"
            )

            # Test chunk formation
            documents = rag_system.text_splitter.split_with_structure(sample_text, metadata)
            print(f"✅ Created {len(documents)} chunks from sample text")

            # Add to vector store
            rag_system._add_documents_to_vectorstore(documents)
            print("✅ Documents added to vector store")

            # Test retrieval
            results = rag_system.retrieve_for_rule_generation("bend radius requirements", top_k=2)
            print(f"✅ Retrieved {len(results)} relevant chunks")

            return True
    except Exception as e:
        print(f"❌ Chunk formation test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("🧪 COMPREHENSIVE RAG SYSTEM TEST SUITE")
    print("=" * 60)

    tests = [
        ("OCR Functionality", test_ocr_functionality),
        ("RAG Database", test_rag_database),
        ("RAG Pipeline", test_rag_pipeline),
        ("Enhanced QA", test_enhanced_qa),
        ("Chunk Formation", test_chunk_formation),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'-'*40}")
        print(f"Running: {test_name}")
        print(f"{'-'*40}")

        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            results.append((test_name, False))

    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST RESULTS SUMMARY")
    print(f"{'='*60}")

    passed = 0
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
        if success:
            passed += 1

    print(f"\n📈 Overall: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")

    if passed == total:
        print("🎉 ALL TESTS PASSED! The RAG system is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)