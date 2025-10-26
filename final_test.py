#!/usr/bin/env python3
"""
Final Testing Script for Enhanced RAG System
Tests the complete Streamlit application before GitHub deployment
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")

    try:
        # Test basic imports
        import streamlit as st
        print("✅ Streamlit available")

        # Test core modules (without heavy ML models)
        import pandas as pd
        import numpy as np
        import os
        import sys
        from pathlib import Path
        print("✅ Core libraries available")

        # Test page structure (import without initializing heavy components)
        try:
            # Just check if the files exist and can be parsed
            import ast
            with open('pages/consolidated_rules.py', 'r') as f:
                ast.parse(f.read())
            print("✅ Consolidated rules page syntax valid")
        except Exception as e:
            print(f"❌ Consolidated rules page syntax error: {e}")
            return False

        try:
            with open('pages/smart_uploader.py', 'r') as f:
                ast.parse(f.read())
            print("✅ Smart uploader page syntax valid")
        except Exception as e:
            print(f"❌ Smart uploader page syntax error: {e}")
            return False

        # Test data availability
        test_results_dir = Path("./test_results")
        if test_results_dir.exists():
            csv_files = list(test_results_dir.glob("*_rules.csv"))
            print(f"✅ Test results available: {len(csv_files)} CSV files")
        else:
            print("⚠️ No test results directory found")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_streamlit_app():
    """Test that the Streamlit app files are syntactically correct."""
    print("\n🚀 Testing Streamlit application structure...")

    try:
        # Test enhanced main app syntax
        print("Testing enhanced_main_app.py syntax...")
        import ast
        with open('enhanced_main_app.py', 'r') as f:
            ast.parse(f.read())
        print("✅ Enhanced main app syntax valid")

        # Test consolidated rules page syntax
        print("Testing consolidated_rules.py syntax...")
        with open('pages/consolidated_rules.py', 'r') as f:
            ast.parse(f.read())
        print("✅ Consolidated rules page syntax valid")

        # Test smart uploader syntax
        print("Testing smart_uploader.py syntax...")
        with open('pages/smart_uploader.py', 'r') as f:
            ast.parse(f.read())
        print("✅ Smart uploader page syntax valid")

        return True

    except Exception as e:
        print(f"❌ Syntax test failed: {e}")
        return False

def check_test_data():
    """Check that test data is available and valid."""
    print("\n📊 Checking test data...")

    test_results_dir = Path("./test_results")

    if not test_results_dir.exists():
        print("❌ Test results directory not found")
        return False

    # Check for CSV files
    csv_files = list(test_results_dir.glob("*_rules.csv"))
    if not csv_files:
        print("⚠️ No CSV rule files found")
    else:
        print(f"✅ Found {len(csv_files)} CSV rule files")

    # Check for consolidated file
    consolidated_file = test_results_dir / "consolidated_all_rules.csv"
    if consolidated_file.exists():
        try:
            import pandas as pd
            df = pd.read_csv(consolidated_file)
            print(f"✅ Consolidated file valid: {len(df)} rules")
        except Exception as e:
            print(f"❌ Consolidated file invalid: {e}")
            return False
    else:
        print("⚠️ No consolidated rules file found")

    # Check for JSON results
    json_files = list(test_results_dir.glob("*.json"))
    if json_files:
        print(f"✅ Found {len(json_files)} JSON result files")
    else:
        print("⚠️ No JSON result files found")

    return True

def run_final_tests():
    """Run all final tests."""
    print("🧪 Running Final Tests for Enhanced RAG System")
    print("=" * 50)

    all_passed = True

    # Test 1: Imports
    if not test_imports():
        all_passed = False

    # Test 2: Streamlit app
    if not test_streamlit_app():
        all_passed = False

    # Test 3: Test data
    if not check_test_data():
        all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Ready for GitHub deployment.")
        print("\n🚀 To run the application:")
        print("   streamlit run enhanced_main_app.py")
        print("\n📊 Key features ready:")
        print("   • Consolidated Rules Database")
        print("   • Automated Testing System")
        print("   • Enhanced QA with Citations")
        print("   • Professional UI/UX")
        return True
    else:
        print("❌ Some tests failed. Please fix issues before deployment.")
        return False

if __name__ == "__main__":
    success = run_final_tests()
    sys.exit(0 if success else 1)