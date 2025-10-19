"""
Universal Document Processor
Demo script showing how to handle documents without manufacturing keywords
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.implicit_rule_extractor import ImplicitRuleExtractor
from core.universal_rag_system import UniversalManufacturingRAG, process_random_document
from core.enhanced_universal_classifier import EnhancedUniversalClassifier
import json

def demo_vague_document_processing():
    """Demonstrate processing of documents without clear manufacturing content."""
    
    print("🚀 Universal Document Processing Demo")
    print("=" * 60)
    
    # Initialize systems
    print("📚 Initializing systems...")
    implicit_extractor = ImplicitRuleExtractor()
    universal_rag = UniversalManufacturingRAG()
    enhanced_classifier = EnhancedUniversalClassifier()
    
    # Test with various types of vague documents
    test_documents = {
        "Software Requirements": """
        The system should provide adequate response times for user interactions.
        Components must maintain compatibility across different operating environments.
        Data integrity should be ensured throughout all operations.
        User interfaces need to be accessible and intuitive for all users.
        Error handling must be comprehensive and user-friendly.
        Security measures should protect against unauthorized access.
        """,
        
        "General Guidelines": """
        Products should meet customer expectations for quality and performance.
        Materials must be selected considering cost and environmental impact.
        Processes need to be efficient and scalable for production volumes.
        Safety measures should be implemented at all operational stages.
        Documentation must be maintained for all procedures and decisions.
        Training should be provided to ensure proper implementation.
        """,
        
        "Project Specifications": """
        Deliverables should align with project objectives and timelines.
        Resources must be allocated appropriately across work packages.
        Communication protocols need to be established between teams.
        Quality checkpoints should be implemented throughout the project.
        Risk mitigation strategies must be developed and maintained.
        Performance metrics should track progress against goals.
        """,
        
        "Technical Standards": """
        Equipment should operate within specified environmental conditions.
        Maintenance schedules must ensure continuous operational availability.
        Calibration procedures need to maintain measurement accuracy.
        Spare parts inventory should support operational requirements.
        Operating procedures must be followed to ensure consistency.
        Performance monitoring should detect deviations from normal operation.
        """
    }
    
    results_summary = {}
    
    for doc_type, content in test_documents.items():
        print(f"\n📄 Processing: {doc_type}")
        print("-" * 40)
        
        # 1. Implicit Rule Extraction
        print("🔍 Extracting implicit rules...")
        implicit_rules = implicit_extractor.extract_implicit_rules(content, confidence_threshold=0.4)
        
        # 2. Universal RAG Analysis
        print("🧠 Analyzing document type...")
        doc_analysis = universal_rag.analyze_document_type(content)
        
        # 3. Enhanced Classification
        print("🎯 Enhanced classification...")
        sentences = content.strip().split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        enhanced_rules = enhanced_classifier.classify_with_multiple_approaches(
            sentences, 
            confidence_threshold=0.4,
            rag_pipeline=universal_rag
        )
        
        # Compile results
        results = {
            "document_type": doc_type,
            "implicit_rules_found": len(implicit_rules),
            "enhanced_rules_found": len(enhanced_rules),
            "manufacturing_relevance": doc_analysis["manufacturing_relevance"],
            "recommended_method": doc_analysis["recommended_method"],
            "confidence_level": doc_analysis["confidence"],
            "sample_implicit_rules": [
                {
                    "text": rule.text,
                    "confidence": rule.confidence_score,
                    "type": rule.rule_type,
                    "constraint": rule.constraint_type
                }
                for rule in implicit_rules[:3]  # Top 3
            ],
            "sample_enhanced_rules": [
                {
                    "text": rule["text"],
                    "confidence": rule["confidence"],
                    "methods": rule["classification_methods"],
                    "type": rule["suggested_rule_type"]
                }
                for rule in enhanced_rules[:3]  # Top 3
            ]
        }
        
        results_summary[doc_type] = results
        
        # Display results
        print(f"  ✅ Implicit Rules: {len(implicit_rules)}")
        print(f"  ✅ Enhanced Rules: {len(enhanced_rules)}")
        print(f"  🎯 Manufacturing Relevance: {doc_analysis['manufacturing_relevance']:.3f}")
        print(f"  🔧 Recommended Method: {doc_analysis['recommended_method']}")
        print(f"  💯 Confidence: {doc_analysis['confidence']}")
        
        if implicit_rules:
            print(f"  📋 Top Rule: '{implicit_rules[0].text[:60]}...'")
            print(f"      Type: {implicit_rules[0].rule_type}")
            print(f"      Confidence: {implicit_rules[0].confidence_score:.3f}")
    
    # Summary Analysis
    print("\n📊 PROCESSING SUMMARY")
    print("=" * 60)
    
    total_implicit = sum(r["implicit_rules_found"] for r in results_summary.values())
    total_enhanced = sum(r["enhanced_rules_found"] for r in results_summary.values())
    avg_relevance = sum(r["manufacturing_relevance"] for r in results_summary.values()) / len(results_summary)
    
    print(f"📈 Total Implicit Rules Extracted: {total_implicit}")
    print(f"📈 Total Enhanced Rules Found: {total_enhanced}")
    print(f"📈 Average Manufacturing Relevance: {avg_relevance:.3f}")
    print(f"📈 Documents Successfully Processed: {len(results_summary)}/4")
    
    # Method Analysis
    print("\n🔬 METHOD EFFECTIVENESS")
    print("-" * 30)
    
    method_stats = {}
    for doc_type, results in results_summary.items():
        method = results["recommended_method"]
        method_stats[method] = method_stats.get(method, 0) + 1
    
    for method, count in method_stats.items():
        print(f"  {method}: {count} documents")
    
    # Confidence Distribution
    print("\n💯 CONFIDENCE DISTRIBUTION")
    print("-" * 30)
    
    confidence_stats = {}
    for doc_type, results in results_summary.items():
        conf = results["confidence_level"]
        confidence_stats[conf] = confidence_stats.get(conf, 0) + 1
    
    for conf, count in confidence_stats.items():
        print(f"  {conf} confidence: {count} documents")
    
    return results_summary

def test_specific_vague_case():
    """Test with a very vague document with no obvious manufacturing content."""
    
    print("\n🧪 TESTING EXTREMELY VAGUE DOCUMENT")
    print("=" * 50)
    
    vague_content = """
    Items should be arranged properly to avoid issues during operation.
    Connections need to be secure and reliable for long-term use.
    Surfaces must be prepared appropriately before application of finishes.
    Components should fit together without interference or gaps.
    Procedures must be followed consistently to achieve desired outcomes.
    Regular checks should ensure everything continues working as expected.
    """
    
    print("📄 Content to analyze:")
    print(f"'{vague_content[:100]}...'\n")
    
    # Initialize extractor
    extractor = ImplicitRuleExtractor()
    
    # Extract rules with lower threshold for vague content
    rules = extractor.extract_implicit_rules(vague_content, confidence_threshold=0.3)
    
    print(f"🔍 Rules found: {len(rules)}")
    
    for i, rule in enumerate(rules, 1):
        print(f"\nRule {i}:")
        print(f"  📝 Text: {rule.text}")
        print(f"  🎯 Type: {rule.rule_type}")
        print(f"  🔧 Constraint: {rule.constraint_type}")
        print(f"  💯 Confidence: {rule.confidence_score:.3f}")
        print(f"  🏭 Mfg Relevance: {rule.manufacturing_relevance:.3f}")
        print(f"  🏷️  Features: {rule.semantic_features[:3]}")
        print(f"  ⚠️  Indicators: {rule.context_indicators}")
    
    return rules

if __name__ == "__main__":
    # Run main demo
    summary = demo_vague_document_processing()
    
    # Test extremely vague case
    vague_rules = test_specific_vague_case()
    
    # Save results
    print(f"\n💾 Saving results...")
    with open("universal_processing_results.json", "w") as f:
        # Convert complex objects to serializable format
        serializable_summary = {}
        for doc_type, results in summary.items():
            serializable_summary[doc_type] = {
                k: v for k, v in results.items() 
                if k not in ['sample_implicit_rules', 'sample_enhanced_rules']
            }
        
        json.dump(serializable_summary, f, indent=2)
    
    print(f"✅ Results saved to 'universal_processing_results.json'")
    print(f"🎉 Demo completed successfully!")
    print(f"\n💡 **Key Insight**: The system successfully extracted {sum(r['implicit_rules_found'] for r in summary.values())} rules from vague documents without manufacturing keywords!")