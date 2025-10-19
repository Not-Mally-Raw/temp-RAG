#!/usr/bin/env python3
"""
LLM API Setup Script
Helps users configure Groq or Cerebras API keys for enhanced document understanding
"""

import os
import sys
from pathlib import Path

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def check_api_status():
    """Check current API configuration status."""
    groq_key = os.getenv("GROQ_API_KEY")
    cerebras_key = os.getenv("CEREBRAS_API_KEY")
    
    status = {
        "groq": bool(groq_key and groq_key != "your_groq_api_key_here"),
        "cerebras": bool(cerebras_key and cerebras_key != "your_cerebras_api_key_here")
    }
    
    return status

def create_env_file():
    """Create or update .env file with API keys."""
    env_path = Path("/workspace/.env")
    env_example_path = Path("/workspace/.env.example")
    
    print_header("API KEY SETUP")
    
    print("\nThis script will help you set up LLM API keys for enhanced")
    print("document understanding. You need at least ONE of these:")
    print("")
    print("1. Groq (Recommended - Fast, free tier available)")
    print("2. Cerebras (Alternative - Also free tier)")
    print("")
    
    # Check if .env exists
    if env_path.exists():
        print("⚠️  .env file already exists.")
        overwrite = input("Do you want to update it? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("Keeping existing .env file.")
            return
    
    # Read example
    if env_example_path.exists():
        with open(env_example_path, 'r') as f:
            content = f.read()
    else:
        content = "# LLM API Configuration\n"
    
    print("\n" + "-"*70)
    print("GROQ API KEY SETUP")
    print("-"*70)
    print("Groq provides fast, free LLM inference.")
    print("Get your key at: https://console.groq.com/keys")
    print("")
    groq_key = input("Enter your Groq API key (or press Enter to skip): ").strip()
    
    if groq_key:
        # Update or add Groq key
        if "GROQ_API_KEY=" in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith("GROQ_API_KEY="):
                    lines[i] = f"GROQ_API_KEY={groq_key}"
            content = '\n'.join(lines)
        else:
            content += f"\nGROQ_API_KEY={groq_key}\n"
    
    print("\n" + "-"*70)
    print("CEREBRAS API KEY SETUP (Optional)")
    print("-"*70)
    print("Cerebras is an alternative to Groq.")
    print("Get your key at: https://cloud.cerebras.ai/")
    print("")
    cerebras_key = input("Enter your Cerebras API key (or press Enter to skip): ").strip()
    
    if cerebras_key:
        # Update or add Cerebras key
        if "CEREBRAS_API_KEY=" in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith("CEREBRAS_API_KEY="):
                    lines[i] = f"CEREBRAS_API_KEY={cerebras_key}"
            content = '\n'.join(lines)
        else:
            content += f"\nCEREBRAS_API_KEY={cerebras_key}\n"
    
    # Write .env file
    with open(env_path, 'w') as f:
        f.write(content)
    
    print("\n✅ .env file created successfully!")
    print(f"   Location: {env_path}")
    
    return groq_key or cerebras_key

def test_api_connection(api_key, provider="groq"):
    """Test if API key works."""
    print(f"\nTesting {provider.upper()} API connection...")
    
    try:
        if provider == "groq":
            from groq import Groq
            client = Groq(api_key=api_key)
            # Simple test
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "Say 'test successful' if you can read this."}],
                max_tokens=20
            )
            response = completion.choices[0].message.content
            print(f"  ✅ {provider.upper()} API working!")
            print(f"  Response: {response}")
            return True
            
        elif provider == "cerebras":
            from cerebras.cloud.sdk import Cerebras
            client = Cerebras(api_key=api_key)
            completion = client.chat.completions.create(
                model="llama3.1-70b",
                messages=[{"role": "user", "content": "Say 'test successful' if you can read this."}],
                max_tokens=20
            )
            response = completion.choices[0].message.content
            print(f"  ✅ {provider.upper()} API working!")
            print(f"  Response: {response}")
            return True
            
    except Exception as e:
        print(f"  ❌ {provider.upper()} API test failed: {e}")
        return False

def show_quick_start():
    """Show quick start instructions."""
    print_header("QUICK START GUIDE")
    
    print("""
Now that your API is configured, you can:

1. Test the LLM analyzer:
   python3 -c "from core.llm_context_analyzer import get_default_analyzer; \\
               analyzer = get_default_analyzer(); \\
               print('✅ LLM Analyzer ready!')"

2. Test the integrated pipeline:
   python3 -c "from core.llm_integrated_pipeline import LLMIntegratedPipeline; \\
               pipeline = LLMIntegratedPipeline(); \\
               print(pipeline.get_system_status())"

3. Process documents with LLM enhancement:
   python3 core/llm_integrated_pipeline.py

4. Run Streamlit with LLM features:
   streamlit run main_app.py

💡 The system will now understand generic documents with ZERO manufacturing
   keywords using advanced LLM context analysis!
""")

def main():
    """Main setup process."""
    print_header("RAG SYSTEM - LLM API SETUP")
    
    print("""
This script helps you configure LLM APIs for enhanced document understanding.

WHY YOU NEED THIS:
- Understand documents with ZERO manufacturing keywords
- Extract implicit requirements from generic text
- Dramatically improve rule extraction accuracy
- Analyze industry-agnostic documents

FREE TIER AVAILABLE: Both Groq and Cerebras offer free API access!
""")
    
    # Check current status
    current_status = check_api_status()
    
    if current_status["groq"] or current_status["cerebras"]:
        print("\n✅ API keys are already configured!")
        print(f"   Groq: {'✅ Configured' if current_status['groq'] else '❌ Not configured'}")
        print(f"   Cerebras: {'✅ Configured' if current_status['cerebras'] else '❌ Not configured'}")
        
        reconfigure = input("\nDo you want to reconfigure? (y/n): ").strip().lower()
        if reconfigure != 'y':
            print("\nKeeping existing configuration.")
            show_quick_start()
            return
    
    # Create/update .env file
    api_key = create_env_file()
    
    if not api_key:
        print("\n⚠️  No API keys provided.")
        print("   The system will work without LLM enhancement, but accuracy will be lower.")
        print("\n   You can run this script again later: python3 setup_llm_apis.py")
        return
    
    # Test the connection
    print("\n" + "="*70)
    test_provider = "groq" if os.getenv("GROQ_API_KEY") else "cerebras"
    test_key = os.getenv("GROQ_API_KEY") or os.getenv("CEREBRAS_API_KEY")
    
    print("Would you like to test the API connection? (y/n): ", end="")
    if input().strip().lower() == 'y':
        test_api_connection(test_key, test_provider)
    
    # Show next steps
    show_quick_start()
    
    print("\n" + "="*70)
    print("  Setup Complete! 🎉")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
