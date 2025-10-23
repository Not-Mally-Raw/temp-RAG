#!/usr/bin/env python3
"""
Startup script for the Universal RAG Testing Simulator
Run this to test the vague document processing capabilities
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages if needed."""
    try:
        import streamlit
        import sentence_transformers
        import transformers
        print("✅ Core packages already installed")
    except ImportError:
        print("📦 Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def run_testing_simulator():
    """Run the testing simulator."""
    script_path = Path(__file__).parent / "pages" / "testing_simulator.py"
    
    if not script_path.exists():
        print("❌ Testing simulator not found!")
        return
    
    print("🚀 Starting Universal RAG Testing Simulator...")
    print("🌐 The app will open in your browser automatically")
    print("📄 Test with vague documents without manufacturing keywords!")
    print("💡 Try the Challenge Mode for extreme test cases")
    print("---")
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        str(script_path),
        "--server.port", "8501",
        "--server.address", "localhost",
        "--browser.serverAddress", "localhost"
    ])

def run_main_app():
    """Run the main application."""
    script_path = Path(__file__).parent / "main_app.py"
    
    if not script_path.exists():
        print("❌ Main app not found!")
        return
    
    print("🚀 Starting Universal RAG System...")
    print("🌐 The app will open in your browser automatically")
    print("---")
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        str(script_path),
        "--server.port", "8502",
        "--server.address", "localhost",
        "--browser.serverAddress", "localhost"
    ])

def main():
    """Main startup function."""
    print("🧪 Universal RAG System - Testing Environment")
    print("=" * 50)
    
    # Install requirements
    install_requirements()
    
    # Choose what to run
    print("\nWhat would you like to run?")
    print("1. 🧪 Testing Simulator (recommended for vague document testing)")
    print("2. 🚀 Full Application (complete system)")
    print("3. 📊 Analytics Only")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        run_testing_simulator()
    elif choice == "2":
        run_main_app()
    elif choice == "3":
        analytics_path = Path(__file__).parent / "pages" / "analytics.py"
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(analytics_path)])
    else:
        print("🧪 Running Testing Simulator by default...")
        run_testing_simulator()

if __name__ == "__main__":
    main()