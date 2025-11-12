#!/usr/bin/env python3
"""
System Validation Script for Enhanced Manufacturing Rule Extraction
Tests API key configuration and basic system functionality
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Add DI orchestrator import and create default system for validation runs
from core.orchestrator import default_production_system

def check_environment():
    """Check environment configuration."""
    print("🔍 Checking Environment Configuration...")
    print("=" * 50)

    # Check if .env file exists
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file found")
        load_dotenv(env_file)
    else:
        print("❌ .env file not found")
        print("   Please copy .env.example to .env and configure your API keys")
        return False

    # Check GROQ_API_KEY
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        print("✅ GROQ_API_KEY environment variable set")
        # Check for known placeholder keys
        placeholder_keys = [

        ]
        if groq_key in placeholder_keys:
            print("❌ GROQ_API_KEY appears to be a placeholder")
            print("   Please replace with your actual Groq API key from https://console.groq.com/")
            return False
        elif len(groq_key) < 20:
            print("❌ GROQ_API_KEY seems too short (might be incomplete)")
            return False
        else:
            print("✅ GROQ_API_KEY format looks valid")
    else:
        print("❌ GROQ_API_KEY environment variable not set")
        print("   Make sure to run: export GROQ_API_KEY=your_actual_key")
        return False

    return True

def test_imports():
    """Test that all required modules can be imported."""
    print("\n📦 Testing Module Imports...")
    print("=" * 50)

    try:
        from core.enhanced_rule_engine import EnhancedRuleEngine, EnhancedConfig
        print("✅ EnhancedRuleEngine imported successfully")
    except Exception as e:
        print(f"❌ Failed to import EnhancedRuleEngine: {e}")
        return False

    try:
        from core.production_system import ProductionRuleExtractionSystem
        print("✅ ProductionRuleExtractionSystem imported successfully")
    except Exception as e:
        print(f"❌ Failed to import ProductionRuleExtractionSystem: {e}")
        return False

    try:
        from core.enhanced_vector_utils import EnhancedVectorManager
        print("✅ EnhancedVectorManager imported successfully")
    except Exception as e:
        print(f"❌ Failed to import EnhancedVectorManager: {e}")
        return False

    return True

def test_configuration():
    """Test configuration loading."""
    print("\n⚙️ Testing Configuration Loading...")
    print("=" * 50)

    try:
        from core.enhanced_rule_engine import EnhancedConfig
        config = EnhancedConfig()
        print("✅ EnhancedConfig loaded successfully")
        print(f"   Model: {config.groq_model}")
        print(f"   API Key Set: {'Yes' if config.groq_api_key else 'No'}")
        print(f"   Chunk Size: {config.chunk_size}")
        return True
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return False

def test_llm_connection():
    """Test LLM connection with a simple query."""
    print("\n🤖 Testing LLM Connection...")
    print("=" * 50)

    try:
        from core.enhanced_rule_engine import EnhancedRuleEngine
        engine = EnhancedRuleEngine()
        print("✅ EnhancedRuleEngine initialized successfully")

        active_model = getattr(engine.config, "groq_model", "unknown")
        preferred_model = getattr(engine, "_preferred_model", active_model)
        if active_model != preferred_model:
            print(f"⚠️ Using fallback model '{active_model}' (requested '{preferred_model}')")
        else:
            print(f"✅ Active model: {active_model}")

        # Test with a simple text
        test_text = "Minimum wall thickness should be 2mm for plastic parts."

        import asyncio
        async def test_extraction():
            try:
                result = await engine.extract_rules_parallel(test_text, "test")
            except Exception as exc:  # pragma: no cover - network/runtime errors
                print(f"❌ Rule extraction request failed: {exc}")
                last_error = getattr(engine, "_last_llm_error", None)
                if last_error and last_error is not exc:
                    print(f"   Last LLM error: {last_error}")
                return False

            rules = result.rules
            if not rules:
                print("❌ Rule extraction returned no rules")
                stats_error = result.extraction_stats.get("error") if isinstance(result.extraction_stats, dict) else None
                if stats_error:
                    print(f"   Extraction error: {stats_error}")
                last_error = getattr(engine, "_last_llm_error", None)
                if last_error:
                    print(f"   LLM reported: {last_error}")
                return False

            print(f"✅ Rule extraction completed: {len(rules)} rule(s) found")
            rule = rules[0]
            print(f"   Sample rule: {rule.rule_text[:50]}...")
            try:
                print(f"   Confidence: {rule.confidence_score:.3f}")
            except AttributeError:
                pass
            return True

        # Run the async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(test_extraction())
        loop.close()

        return success

    except Exception as e:
        print(f"❌ LLM connection test failed: {e}")
        return False

def run_smoke_test(sample_doc_path: str):
    """Run a smoke test on the rule extraction system with a sample document."""
    print(f"\n📂 Running Smoke Test on: {sample_doc_path}")
    print("=" * 50)

    # Replace legacy system instantiation (non-invasive)
    _system = default_production_system()

    # Simple extraction test
    try:
        # payload = system.process_document_advanced(sample_doc_path, ...)
        payload = _system.process_document(sample_doc_path, export_path=None)
        if not payload:
            print("❌ No payload returned from process_document")
            return False

        print("✅ Smoke test completed successfully")
        return True
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        return False

def main():
    """Main validation function."""
    print("🧪 Enhanced Manufacturing Rule Extraction - System Validation")
    print("=" * 70)

    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    checks = [
        ("Environment Configuration", check_environment),
        ("Module Imports", test_imports),
        ("Configuration Loading", test_configuration),
        ("LLM Connection", test_llm_connection),
    ]

    results = []
    for check_name, check_func in checks:
        print(f"\n🔍 Running: {check_name}")
        try:
            result = check_func()
            results.append(result)
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"Result: {status}")
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results.append(False)

    # Run smoke test on a sample document (if available)
    sample_doc = Path("samples/sample_document.txt")
    if sample_doc.exists():
        smoke_test_result = run_smoke_test(str(sample_doc))
        results.append(smoke_test_result)
        status = "✅ PASSED" if smoke_test_result else "❌ FAILED"
        print(f"\n🔍 Running: Smoke Test on Sample Document")
        print(f"Result: {status}")
    else:
        print("\n⚠️ Skipping smoke test - sample document not found")

    print("\n" + "=" * 70)
    print("📊 VALIDATION SUMMARY")
    print("=" * 70)

    passed = sum(results)
    total = len(results)

    for i, (check_name, _) in enumerate(checks):
        status = "✅ PASSED" if results[i] else "❌ FAILED"
        print(f"{check_name}: {status}")

    print(f"\nOverall: {passed}/{total} checks passed")

    if passed == total:
        print("\n🎉 All checks passed! Your system is ready to use.")
        print("   Try running: python core/enhanced_rule_engine.py")
    else:
        print("\n⚠️ Some checks failed. Please fix the issues above before using the system.")
        print("   Common fixes:")
        print("   - Copy .env.example to .env and add your real API key")
        print("   - Run: export GROQ_API_KEY=your_actual_key")
        print("   - Make sure all dependencies are installed: pip install -r requirements.txt")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)