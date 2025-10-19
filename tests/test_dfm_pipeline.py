"""
Integration test for the DFM pipeline.
"""

import sys
from pathlib import Path
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.dfm_pipeline import (
    split_text_for_rag,
    postprocess_extracted_rules,
)


def test_text_splitting():
    """Test text splitting functionality."""
    text = "This is a test document. " * 50  # Create a long text
    chunks = split_text_for_rag(text, chunk_size=100, overlap=20)
    
    assert len(chunks) > 0, "Should create at least one chunk"
    assert all(len(chunk) <= 120 for chunk in chunks), "Chunks should be within size limit"
    print(f"✓ Text splitting test passed: {len(chunks)} chunks created")


def test_postprocessing_json():
    """Test postprocessing with valid JSON."""
    json_output = '[{"type": "tolerance", "value": "±0.001"}]'
    result = postprocess_extracted_rules(json_output)
    
    assert result["format"] == "json", "Should recognize JSON format"
    assert "rules" in result, "Should contain rules key"
    print("✓ JSON postprocessing test passed")


def test_postprocessing_text():
    """Test postprocessing with non-JSON text."""
    text_output = "Some raw text output from the model"
    result = postprocess_extracted_rules(text_output)
    
    assert "rules" in result, "Should contain rules key"
    assert result["format"] == "text", "Should recognize text format"
    print("✓ Text postprocessing test passed")


def test_sample_dfm_file():
    """Test that sample DFM file exists and is readable."""
    sample_file = Path(__file__).parent.parent / "data" / "sample_dfm.txt"
    
    assert sample_file.exists(), f"Sample file should exist at {sample_file}"
    
    with open(sample_file, 'r') as f:
        content = f.read()
    
    assert len(content) > 100, "Sample file should contain substantial content"
    assert "tolerance" in content.lower(), "Sample should contain DFM-related content"
    print(f"✓ Sample DFM file test passed: {len(content)} characters")


def test_dfm_pipeline_basic():
    """Test basic DFM pipeline functionality with sample text."""
    from core.dfm_pipeline import split_text_for_rag
    
    sample_file = Path(__file__).parent.parent / "data" / "sample_dfm.txt"
    
    with open(sample_file, 'r') as f:
        content = f.read()
    
    # Test chunking
    chunks = split_text_for_rag(content, chunk_size=500, overlap=50)
    
    assert len(chunks) > 0, "Should create chunks from sample file"
    print(f"✓ Basic pipeline test passed: processed {len(chunks)} chunks")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Running DFM Pipeline Tests")
    print("="*60 + "\n")
    
    tests = [
        test_text_splitting,
        test_postprocessing_json,
        test_postprocessing_text,
        test_sample_dfm_file,
        test_dfm_pipeline_basic,
    ]
    
    failed = []
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed.append(test.__name__)
    
    print("\n" + "="*60)
    if not failed:
        print("All tests passed! ✓")
    else:
        print(f"Failed tests: {', '.join(failed)}")
    print("="*60 + "\n")
    
    return len(failed) == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
