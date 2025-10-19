#!/usr/bin/env python3
"""
Quick Start Script for Enhanced RAG System

This script helps you get started with the DFM pipeline quickly.
It checks dependencies, runs a simple test, and provides next steps.
"""

import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ is required.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies():
    """Check if required dependencies are installed."""
    print("\nChecking dependencies...")
    
    required_packages = {
        "sentence_transformers": "sentence-transformers",
        "transformers": "transformers",
        "chromadb": "chromadb",
        "pdfplumber": "pdfplumber",
        "streamlit": "streamlit",
    }
    
    missing = []
    installed = []
    
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
            installed.append(package_name)
            print(f"  ✅ {package_name}")
        except ImportError:
            missing.append(package_name)
            print(f"  ❌ {package_name} (not installed)")
    
    return missing, installed


def install_dependencies(missing):
    """Install missing dependencies."""
    if not missing:
        return True
    
    print(f"\n{len(missing)} dependencies missing. Would you like to install them? (y/n)")
    response = input("> ").strip().lower()
    
    if response != 'y':
        print("Skipping installation. You can install manually with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    print("\nInstalling dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False


def run_basic_test():
    """Run a basic test of the DFM pipeline."""
    print("\nRunning basic functionality test...")
    
    try:
        from core.dfm_pipeline import split_text_for_rag, postprocess_extracted_rules
        
        # Test text splitting
        test_text = "This is a test. " * 100
        chunks = split_text_for_rag(test_text, chunk_size=100, overlap=20)
        
        if len(chunks) > 0:
            print(f"  ✅ Text chunking works ({len(chunks)} chunks created)")
        else:
            print("  ❌ Text chunking failed")
            return False
        
        # Test postprocessing
        result = postprocess_extracted_rules('{"test": "data"}')
        if "rules" in result:
            print("  ✅ Rule postprocessing works")
        else:
            print("  ❌ Rule postprocessing failed")
            return False
        
        print("\n✅ All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        return False


def check_sample_data():
    """Check if sample data exists."""
    print("\nChecking sample data...")
    sample_file = Path("data/sample_dfm.txt")
    
    if sample_file.exists():
        print(f"  ✅ Sample DFM file found: {sample_file}")
        with open(sample_file, 'r') as f:
            content = f.read()
        print(f"     ({len(content)} characters, {len(content.splitlines())} lines)")
        return True
    else:
        print(f"  ❌ Sample file not found: {sample_file}")
        return False


def show_next_steps():
    """Show next steps to the user."""
    print_header("Next Steps")
    
    print("You're all set! Here's what you can do next:\n")
    
    print("1️⃣  Test the DFM pipeline with sample data:")
    print("   python tests/test_dfm_pipeline.py\n")
    
    print("2️⃣  Process the sample DFM handbook (requires full dependencies):")
    print("   python -m core.dfm_pipeline data/sample_dfm.txt\n")
    
    print("3️⃣  Launch the Streamlit UI:")
    print("   streamlit run main_app.py\n")
    
    print("4️⃣  Read the documentation:")
    print("   - DFM Pipeline Guide: docs/DFM_PIPELINE_GUIDE.md")
    print("   - Troubleshooting: docs/TROUBLESHOOTING.md")
    print("   - Full README: README.md\n")
    
    print("5️⃣  Process your own DFM handbook:")
    print("   python -m core.dfm_pipeline path/to/your/handbook.pdf --output results.json\n")
    
    print("For help and options:")
    print("   python -m core.dfm_pipeline --help\n")


def main():
    """Main quick start process."""
    print_header("Enhanced RAG System - Quick Start")
    
    print("This script will help you get started with the DFM pipeline.")
    print("It will check your environment and run basic tests.\n")
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Please upgrade Python and try again.")
        sys.exit(1)
    
    # Check dependencies
    missing, installed = check_dependencies()
    
    if missing:
        print(f"\n⚠️  {len(missing)} required packages are missing.")
        if not install_dependencies(missing):
            print("\n⚠️  Some dependencies are missing. Install them with:")
            print(f"    pip install {' '.join(missing)}")
            print("\nYou can continue, but some features may not work.")
    else:
        print("\n✅ All required dependencies are installed!")
    
    # Check sample data
    check_sample_data()
    
    # Run basic test (even with missing deps, basic functions should work)
    if run_basic_test():
        show_next_steps()
    else:
        print("\n⚠️  Basic tests failed. Please check:")
        print("  1. All dependencies are installed: pip install -r requirements.txt")
        print("  2. You're in the repository root directory")
        print("  3. Check docs/TROUBLESHOOTING.md for help")
    
    print("\n" + "=" * 70)
    print("  Quick start complete! Happy DFM rule extraction! 🚀")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
