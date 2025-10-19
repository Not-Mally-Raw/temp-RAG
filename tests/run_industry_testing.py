#!/usr/bin/env python3
"""
Run Industry Document Testing
Launch script for comprehensive industry document testing simulation
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed."""
    print("🔍 Checking requirements...")
    
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'plotly', 'sentence-transformers',
        'transformers', 'torch', 'spacy', 'nltk', 'chromadb'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - MISSING")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("📦 Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All requirements satisfied")
    return True

def download_spacy_model():
    """Download required spaCy model if not present."""
    print("🔍 Checking spaCy models...")
    
    try:
        import spacy
        spacy.load("en_core_web_sm")
        print("✅ spaCy English model available")
        return True
    except OSError:
        print("📥 Downloading spaCy English model...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "spacy", "download", "en_core_web_sm"
            ])
            print("✅ spaCy model downloaded successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to download spaCy model")
            print("💡 Try manually: python -m spacy download en_core_web_sm")
            return False

def run_quick_test():
    """Run quick system test."""
    print("\n🧪 Running quick system test...")
    
    try:
        # Import test modules
        sys.path.append(str(Path(__file__).parent))
        
        from core.implicit_rule_extractor import ImplicitRuleExtractor
        
        # Quick test
        extractor = ImplicitRuleExtractor()
        
        test_content = """
        System components should be designed for optimal performance.
        Materials must be selected considering environmental conditions.
        Assembly procedures require proper alignment and securing.
        """
        
        rules = extractor.extract_implicit_rules(test_content, confidence_threshold=0.4)
        
        if rules:
            print(f"✅ Quick test passed - extracted {len(rules)} rules")
            print(f"📋 Sample rule: '{rules[0].text[:60]}...'")
            return True
        else:
            print("❌ Quick test failed - no rules extracted")
            return False
            
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def launch_industry_testing():
    """Launch the industry document testing simulator."""
    print("\n🚀 Launching Industry Document Testing Simulator...")
    
    # Set environment variables for better performance
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Launch Streamlit app with industry testing page
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "pages/industry_testing_simulator.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Industry testing stopped by user")
    except Exception as e:
        print(f"❌ Failed to launch industry testing: {e}")

def launch_main_app():
    """Launch the main application."""
    print("\n🚀 Launching Main RAG System Application...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "main_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Failed to launch main app: {e}")

def main():
    """Main execution function."""
    print("🏭 Industry Document Testing - RAG System")
    print("=" * 50)
    
    # Check system requirements
    if not check_requirements():
        sys.exit(1)
    
    # Download spaCy model if needed
    if not download_spacy_model():
        print("⚠️  spaCy model missing - some features may not work")
    
    # Run quick test
    if not run_quick_test():
        print("⚠️  Quick test failed - proceeding anyway")
    
    # Ask user preference
    print("\n🎯 Choose testing mode:")
    print("1. 🏭 Industry Document Testing (Comprehensive)")
    print("2. 🚀 Main Application (All Features)")
    print("3. 🧪 Quick Test Only")
    
    choice = input("\nEnter choice (1-3) [default: 1]: ").strip()
    
    if choice == "2":
        launch_main_app()
    elif choice == "3":
        print("✅ Quick test completed successfully!")
        print("\n💡 To run full testing:")
        print("   python run_industry_testing.py")
        print("   Or: streamlit run main_app.py")
    else:  # Default to industry testing
        launch_industry_testing()

if __name__ == "__main__":
    main()