"""
Quick Test Script for Universal RAG System
Verifies that all components are working correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from core.implicit_rule_extractor import ImplicitRuleExtractor
        print("✅ ImplicitRuleExtractor imported successfully")
        
        from core.universal_rag_system import UniversalManufacturingRAG
        print("✅ UniversalManufacturingRAG imported successfully")
        
        from core.enhanced_universal_classifier import EnhancedUniversalClassifier
        print("✅ EnhancedUniversalClassifier imported successfully")
        
        from core.enhanced_rag_db import EnhancedManufacturingRAG
        print("✅ EnhancedManufacturingRAG imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_implicit_extraction():
    """Test implicit rule extraction with vague content."""
    print("\n🔍 Testing implicit rule extraction...")
    
    try:
        from core.implicit_rule_extractor import ImplicitRuleExtractor
        
        extractor = ImplicitRuleExtractor()
        
        # Test with very vague content (no manufacturing keywords)
        vague_content = """
        Items should be arranged properly to avoid issues during operation.
        Components must maintain compatibility across different environments.
        Surfaces need adequate preparation before applying finishes.
        Procedures should be followed consistently for best results.
        """
        
        rules = extractor.extract_implicit_rules(vague_content, confidence_threshold=0.3)
        
        print(f"✅ Extracted {len(rules)} rules from vague content")
        
        if rules:
            print(f"📋 Top rule: '{rules[0].text}'")
            print(f"   Confidence: {rules[0].confidence_score:.3f}")
            print(f"   Type: {rules[0].rule_type}")
            print(f"   Manufacturing Relevance: {rules[0].manufacturing_relevance:.3f}")
        
        return len(rules) > 0
        
    except Exception as e:
        print(f"❌ Implicit extraction test failed: {e}")
        return False

def test_universal_classification():
    """Test universal classification system."""
    print("\n🎯 Testing universal classification...")
    
    try:
        from core.enhanced_universal_classifier import EnhancedUniversalClassifier
        
        classifier = EnhancedUniversalClassifier()
        
        # Test with software requirements (non-manufacturing)
        software_content = [
            "System should provide adequate response times for user interactions.",
            "Database connections must be configured considering network latency.",
            "Security measures should protect against unauthorized access."
        ]
        
        enhanced_rules = classifier.classify_with_multiple_approaches(
            software_content, 
            confidence_threshold=0.4
        )
        
        print(f"✅ Classified {len(enhanced_rules)} rules from software content")
        
        if enhanced_rules:
            rule = enhanced_rules[0]
            print(f"📋 Top rule: '{rule['text'][:50]}...'")
            print(f"   Confidence: {rule['confidence']:.3f}")
            print(f"   Methods: {rule['classification_methods']}")
            print(f"   Suggested Type: {rule['suggested_rule_type']}")
        
        return len(enhanced_rules) > 0
        
    except Exception as e:
        print(f"❌ Universal classification test failed: {e}")
        return False

def test_industry_documents():
    """Test with simulated industry document content."""
    print("\n🏭 Testing industry document processing...")
    
    try:
        from core.universal_rag_system import UniversalManufacturingRAG
        
        universal_rag = UniversalManufacturingRAG()
        
        # Simulate pharmaceutical document content
        pharma_content = """
        Good Manufacturing Practices require validated processes and procedures.
        Contamination control measures must prevent cross-contamination risks.
        Equipment qualification ensures consistent product quality and safety.
        Environmental monitoring programs track critical quality parameters.
        """
        
        # Analyze document type
        analysis = universal_rag.analyze_document_type(pharma_content)
        
        print(f"✅ Document analysis completed")
        print(f"   Manufacturing Relevance: {analysis['manufacturing_relevance']:.3f}")
        print(f"   Recommended Method: {analysis['recommended_method']}")
        print(f"   Confidence Level: {analysis['confidence']}")
        print(f"   Estimated Rules: {analysis['estimated_rules']}")
        
        return analysis['manufacturing_relevance'] > 0.5
        
    except Exception as e:
        print(f"❌ Industry document test failed: {e}")
        return False

def test_system_integration():
    """Test integration between different components."""
    print("\n🔗 Testing system integration...")
    
    try:
        from core.implicit_rule_extractor import ImplicitRuleExtractor
        from core.universal_rag_system import UniversalManufacturingRAG
        
        # Initialize both systems
        extractor = ImplicitRuleExtractor()
        universal_rag = UniversalManufacturingRAG()
        
        # Test with aerospace content
        aerospace_content = """
        System requirements must be traceable through all levels of design.
        Performance specifications should include margins for variations.
        Interface control documents define connection requirements.
        Verification methods must validate compliance with obligations.
        """
        
        # Extract rules with implicit extractor
        implicit_rules = extractor.extract_implicit_rules(aerospace_content)
        
        # Analyze with universal RAG
        analysis = universal_rag.analyze_document_type(aerospace_content)
        
        print(f"✅ Integration test completed")
        print(f"   Implicit Rules: {len(implicit_rules)}")
        print(f"   RAG Analysis Relevance: {analysis['manufacturing_relevance']:.3f}")
        print(f"   Combined Score: {(len(implicit_rules) * analysis['manufacturing_relevance']):.2f}")
        
        return len(implicit_rules) > 0 and analysis['manufacturing_relevance'] > 0.4
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Universal RAG System Tests")
    print("=" * 50)
    
    test_results = []
    
    # Run all tests
    test_results.append(("Import Test", test_imports()))
    test_results.append(("Implicit Extraction", test_implicit_extraction()))
    test_results.append(("Universal Classification", test_universal_classification()))
    test_results.append(("Industry Documents", test_industry_documents()))
    test_results.append(("System Integration", test_system_integration()))
    
    # Summary
    print("\n📊 TEST SUMMARY")
    print("=" * 30)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed ({(passed/total)*100:.0f}%)")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for use.")
        print("\n🚀 Next steps:")
        print("   1. Run: streamlit run main_app.py")
        print("   2. Navigate to 🧪 Testing Simulator")
        print("   3. Try 🏭 Industry Document Testing")
        print("   4. Test with your own documents!")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("💡 Make sure all required packages are installed:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_implicit_rule_extraction():
    """Test implicit rule extraction with vague content."""
    print("🔍 Testing Implicit Rule Extraction...")
    
    try:
        from core.implicit_rule_extractor import ImplicitRuleExtractor
        
        extractor = ImplicitRuleExtractor()
        
        # Test with extremely vague content
        vague_content = """
        Items should be arranged properly to avoid issues during operation.
        Components must maintain compatibility across different environments.
        Surfaces need adequate preparation before applying finishes.
        Procedures should be followed consistently to achieve desired outcomes.
        """
        
        print(f"📄 Testing content: {vague_content[:60]}...")
        
        rules = extractor.extract_implicit_rules(vague_content, confidence_threshold=0.3)
        
        print(f"✅ Found {len(rules)} implicit rules")
        
        for i, rule in enumerate(rules[:3], 1):
            print(f"  Rule {i}: {rule.text[:50]}...")
            print(f"    Type: {rule.rule_type}")
            print(f"    Confidence: {rule.confidence_score:.3f}")
            print(f"    Mfg Relevance: {rule.manufacturing_relevance:.3f}")
        
        return len(rules) > 0
        
    except Exception as e:
        print(f"❌ Implicit extraction test failed: {e}")
        return False

def test_universal_rag_analysis():
    """Test universal RAG document analysis."""
    print("\n🧠 Testing Universal RAG Analysis...")
    
    try:
        from core.universal_rag_system import UniversalManufacturingRAG
        
        rag = UniversalManufacturingRAG(persist_path="./test_rag_db")
        
        # Test with non-manufacturing content
        test_content = """
        Software components should provide adequate response times for user interactions.
        Error handling must be comprehensive and user-friendly for all scenarios.
        Security measures should protect against unauthorized access attempts.
        Data integrity should be ensured throughout all database operations.
        """
        
        print(f"📄 Testing content: {test_content[:60]}...")
        
        analysis = rag.analyze_document_type(test_content)
        
        print(f"✅ Analysis complete")
        print(f"  Manufacturing Relevance: {analysis['manufacturing_relevance']:.3f}")
        print(f"  Recommended Method: {analysis['recommended_method']}")
        print(f"  Confidence Level: {analysis['confidence']}")
        print(f"  Estimated Rules: {analysis['estimated_rules']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Universal RAG test failed: {e}")
        return False

def test_enhanced_classification():
    """Test enhanced universal classification."""
    print("\n🎯 Testing Enhanced Classification...")
    
    try:
        from core.enhanced_universal_classifier import EnhancedUniversalClassifier
        
        classifier = EnhancedUniversalClassifier()
        
        # Test with general business content
        test_sentences = [
            "Processes need to be efficient and scalable for production volumes.",
            "Quality checkpoints should be implemented throughout the project lifecycle.",
            "Regular monitoring ensures everything continues working as expected.",
            "Documentation must be maintained for all procedures and decisions."
        ]
        
        print(f"📄 Testing {len(test_sentences)} sentences...")
        
        enhanced_rules = classifier.classify_with_multiple_approaches(
            test_sentences,
            confidence_threshold=0.4
        )
        
        print(f"✅ Found {len(enhanced_rules)} potential rules")
        
        for i, rule in enumerate(enhanced_rules[:2], 1):
            print(f"  Rule {i}: {rule['text'][:50]}...")
            print(f"    Type: {rule['suggested_rule_type']}")
            print(f"    Confidence: {rule['confidence']:.3f}")
            print(f"    Methods: {', '.join(rule['classification_methods'])}")
        
        return len(enhanced_rules) > 0
        
    except Exception as e:
        print(f"❌ Enhanced classification test failed: {e}")
        return False

def test_extreme_vagueness():
    """Test with extremely vague philosophical content."""
    print("\n💭 Testing Extreme Vagueness (Philosophical Content)...")
    
    try:
        from core.implicit_rule_extractor import ImplicitRuleExtractor
        
        extractor = ImplicitRuleExtractor()
        
        # Extremely abstract content
        extreme_content = """
        Harmony between elements creates sustainable systems and optimal functioning.
        Careful consideration of relationships prevents discord and operational issues.
        Consistent application of principles yields desired results and outcomes.
        Continuous observation maintains equilibrium and proper function over time.
        """
        
        print(f"📄 Testing philosophical content...")
        
        rules = extractor.extract_implicit_rules(extreme_content, confidence_threshold=0.2)
        
        print(f"✅ Even from philosophical content, found {len(rules)} potential rules")
        
        if rules:
            best_rule = max(rules, key=lambda r: r.confidence_score)
            print(f"  Best Rule: {best_rule.text}")
            print(f"    Confidence: {best_rule.confidence_score:.3f}")
            print(f"    Manufacturing Relevance: {best_rule.manufacturing_relevance:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Extreme vagueness test failed: {e}")
        return False

def main():
    """Run all tests and provide summary."""
    print("🧪 Universal RAG System - Quick Functionality Test")
    print("=" * 60)
    
    tests = [
        ("Implicit Rule Extraction", test_implicit_rule_extraction),
        ("Universal RAG Analysis", test_universal_rag_analysis), 
        ("Enhanced Classification", test_enhanced_classification),
        ("Extreme Vagueness Challenge", test_extreme_vagueness)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 TEST SUMMARY")
    print("=" * 30)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The universal RAG system is working correctly.")
        print("🚀 You can now run the Streamlit testing simulator:")
        print("   python run_tests.py")
    elif passed > total // 2:
        print("⚠️  Most tests passed. Some features may need attention.")
        print("🔧 Check the error messages above for details.")
    else:
        print("❌ Multiple tests failed. Check your environment and dependencies.")
        print("📦 Try: pip install -r requirements.txt")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)