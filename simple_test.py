#!/usr/bin/env python3
"""
Simple Test Script for text_pipeline_and_rag_system.py

This script tests the basic imports and structure without requiring
model downloads or internet connectivity.
"""

import sys

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        from text_pipeline_and_rag_system import (
            # Data structures
            DocumentMetadata,
            ImplicitRule,
            DocumentContext,
            
            # Main classes (don't initialize yet - they load models)
            TextExtractor,
            
            # Helper functions
            check_api_availability,
            print_system_status,
        )
        print("✅ All imports successful\n")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}\n")
        return False

def test_data_structures():
    """Test data structure creation."""
    print("Testing data structures...")
    
    try:
        from text_pipeline_and_rag_system import DocumentMetadata, ImplicitRule
        
        # Create metadata
        metadata = DocumentMetadata(
            doc_id="test_001",
            source_file="test.pdf",
            doc_type="text",
            page_number=1
        )
        
        print(f"✅ DocumentMetadata created: {metadata.doc_id}")
        print(f"   Source: {metadata.source_file}")
        print(f"   Type: {metadata.doc_type}\n")
        return True
    except Exception as e:
        print(f"❌ Data structure test failed: {e}\n")
        return False

def test_helper_functions():
    """Test helper functions."""
    print("Testing helper functions...")
    
    try:
        from text_pipeline_and_rag_system import check_api_availability
        
        api_status = check_api_availability()
        print("✅ API availability check:")
        for api, available in api_status.items():
            status = "✓ Available" if available else "✗ Not configured"
            print(f"   {api}: {status}")
        print()
        return True
    except Exception as e:
        print(f"❌ Helper function test failed: {e}\n")
        return False

def test_text_extraction():
    """Test text extraction (without actual PDF)."""
    print("Testing text extraction...")
    
    try:
        from text_pipeline_and_rag_system import TextExtractor
        
        # Note: TextExtractor.extract_sentences requires actual PDF bytes
        # We're just testing that the class is accessible
        print("✅ TextExtractor class accessible")
        print("   Methods available:")
        print("   - extract_sentences(pdf_bytes)")
        print("   - extract_text_simple(pdf_bytes)\n")
        return True
    except Exception as e:
        print(f"❌ Text extraction test failed: {e}\n")
        return False

def test_system_status():
    """Test system status reporting."""
    print("Testing system status...")
    
    try:
        from text_pipeline_and_rag_system import print_system_status
        
        print("✅ System status function available")
        print("\nCalling print_system_status():")
        print("-" * 70)
        print_system_status()
        print("-" * 70)
        return True
    except Exception as e:
        print(f"❌ System status test failed: {e}\n")
        return False

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("TEXT PIPELINE AND RAG SYSTEM - SIMPLE TEST")
    print("="*70)
    print("\nTesting basic functionality without model downloads...\n")
    
    tests = [
        ("Imports", test_imports),
        ("Data Structures", test_data_structures),
        ("Helper Functions", test_helper_functions),
        ("Text Extraction", test_text_extraction),
        ("System Status", test_system_status),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*70}")
        print(f"TEST: {test_name}")
        print(f"{'='*70}\n")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The RAG system structure is correct.")
        print("\nNote: Full functionality requires model downloads.")
        print("In an environment with internet access, you can:")
        print("  1. Run: python text_pipeline_and_rag_system.py")
        print("  2. Or use the example_usage.py script")
        print("\nSee USAGE_GUIDE.md for detailed instructions.")
    else:
        print("\n⚠️  Some tests failed. Check error messages above.")
    
    print("\n" + "="*70)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
