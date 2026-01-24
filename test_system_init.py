#!/usr/bin/env python3
"""
Simple test script to verify the enhanced manufacturing rule extraction system
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def test_system_initialization():
    """Test that the system can be initialized properly."""
    try:
        from core.production_system import ProductionRuleExtractionSystem

        # Get API key from environment
        groq_api_key = os.getenv("GROQ_API_KEY", "")
        if not groq_api_key:
            print("❌ ERROR: GROQ_API_KEY not found in environment variables")
            return False

        print(f"✅ Found GROQ_API_KEY: {groq_api_key[:20]}...")

        # Try to initialize the system
        print("🔄 Initializing production system...")
        system = ProductionRuleExtractionSystem(groq_api_key=groq_api_key, use_qdrant=False)

        print("✅ System initialized successfully!")

        # Get system stats
        stats = system.get_system_stats()
        print("📊 System Stats:")
        print(f"   - Groq Model: {stats['configuration']['groq_model']}")
        print(f"   - Documents Processed: {stats['processing_stats']['documents_processed']}")
        print(f"   - Rules Extracted: {stats['processing_stats']['rules_extracted']}")

        return True

    except Exception as e:
        print(f"❌ ERROR: Failed to initialize system: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing Enhanced Manufacturing Rule Extraction System")
    print("=" * 60)

    success = test_system_initialization()

    if success:
        print("\n🎉 All tests passed! The system is ready to run.")
        print("\nTo run the Streamlit app, use:")
        print("cd /opt/anaconda3/rework-RAG-for-HCLTech")
        print("streamlit run enhanced_streamlit_app.py")
    else:
        print("\n💥 System initialization failed. Please check the errors above.")