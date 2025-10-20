#!/usr/bin/env python3
"""
Example Usage of text_pipeline_and_rag_system.py

This script demonstrates how to use the consolidated RAG system
with practical examples.
"""

from text_pipeline_and_rag_system import (
    UniversalRAGSystem,
    ImplicitRuleExtractor,
    TextExtractor,
    check_api_availability,
    print_system_status
)

def example_1_basic_usage():
    """Example 1: Basic RAG system usage without LLM."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic RAG System Usage")
    print("="*70)
    
    # Initialize RAG system (without LLM for faster demo)
    print("\nInitializing RAG system...")
    rag = UniversalRAGSystem(
        embedding_model_name="all-MiniLM-L6-v2",  # Fast, lightweight model
        persist_path="./example_rag_db",
        use_llm=False
    )
    
    # Sample manufacturing text (simulating PDF content)
    sample_text = """
    Manufacturing Design Guidelines
    
    Sheet metal forming operations should maintain minimum bend radius 
    of at least 1.5 times the material thickness. For aluminum alloys, 
    ensure proper grain direction alignment during bending operations.
    
    Quality Control Requirements
    
    All components must undergo dimensional inspection before assembly.
    Surface finish should meet Ra 3.2 micrometers or better for mating surfaces.
    Critical dimensions require statistical process control with Cpk >= 1.33.
    """
    
    print("\n✓ RAG system initialized")
    print(f"Sample text length: {len(sample_text)} characters")
    
    # Note: In real usage, you would use:
    # with open('document.pdf', 'rb') as f:
    #     results = rag.process_document(f.read(), 'document.pdf')
    
    print("\n✅ Example 1 complete - RAG system ready for use")
    return rag

def example_2_implicit_extraction():
    """Example 2: Extract rules from text without manufacturing keywords."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Implicit Rule Extraction")
    print("="*70)
    
    # Initialize extractor
    print("\nInitializing implicit rule extractor...")
    extractor = ImplicitRuleExtractor()
    
    # Generic text with NO explicit manufacturing keywords
    generic_text = """
    Components should be designed to facilitate easy maintenance and inspection.
    Adequate clearance must be provided for tool access during servicing.
    Materials should be selected to ensure longevity under operational conditions.
    The system must maintain reliable performance throughout its service life.
    Interfaces should provide sufficient space for thermal expansion.
    """
    
    print(f"\n✓ Extractor initialized")
    print(f"Analyzing text (no explicit manufacturing keywords):")
    print(f"{generic_text.strip()[:200]}...")
    
    # Extract rules
    print("\nExtracting implicit rules...")
    rules = extractor.extract_implicit_rules(generic_text, confidence_threshold=0.6)
    
    print(f"\n✅ Found {len(rules)} manufacturing-relevant rules:\n")
    
    for i, rule in enumerate(rules, 1):
        print(f"Rule {i}:")
        print(f"  Text: {rule.text}")
        print(f"  Type: {rule.rule_type}")
        print(f"  Constraint: {rule.constraint_type}")
        print(f"  Confidence: {rule.confidence_score:.2f}")
        print(f"  Mfg Relevance: {rule.manufacturing_relevance:.2f}")
        if rule.constraint_value:
            print(f"  Value: {rule.constraint_value}")
        print()
    
    return rules

def example_3_text_extraction():
    """Example 3: Text extraction from various sources."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Text Extraction")
    print("="*70)
    
    # Simulate PDF bytes (in real usage, read from actual PDF file)
    sample_pdf_text = """
    Design for Manufacturing Handbook
    
    Section 1: General Guidelines
    All designs should consider manufacturability during the concept phase.
    
    Section 2: Material Selection
    Material selection must account for processing requirements and end-use conditions.
    
    Section 3: Tolerancing
    Apply geometric dimensioning and tolerancing (GD&T) for critical features.
    Tolerance stack-up analysis is required for assemblies with multiple components.
    """
    
    print("\n✓ Sample text prepared")
    print(f"Text length: {len(sample_pdf_text)} characters")
    
    # In real usage:
    # sentences = TextExtractor.extract_sentences(pdf_bytes)
    
    print("\n✅ Example 3 complete - Text extraction methods available")

def example_4_system_status():
    """Example 4: Check system status and capabilities."""
    print("\n" + "="*70)
    print("EXAMPLE 4: System Status and Capabilities")
    print("="*70)
    
    # Check API availability
    print("\nChecking LLM API availability...")
    api_status = check_api_availability()
    
    print("\nLLM API Status:")
    for api, available in api_status.items():
        status = "✅ Available" if available else "❌ Not configured"
        print(f"  {api.upper()}: {status}")
    
    if not any(api_status.values()):
        print("\n💡 To enable LLM features:")
        print("  1. Get free API key from https://console.groq.com/keys")
        print("  2. Set environment variable: export GROQ_API_KEY='your-key'")
        print("  3. Restart the application")
    
    # Print full system status
    print_system_status()
    
    print("\n✅ Example 4 complete - System status checked")

def example_5_practical_workflow():
    """Example 5: Complete practical workflow."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Complete Practical Workflow")
    print("="*70)
    
    print("\nScenario: Processing a manufacturing document")
    
    # Step 1: Initialize system
    print("\n1. Initializing RAG system...")
    rag = UniversalRAGSystem(
        embedding_model_name="all-MiniLM-L6-v2",
        persist_path="./demo_rag_db",
        use_llm=False
    )
    
    # Step 2: Process document (simulated)
    print("\n2. Processing document...")
    print("   (In production: rag.process_document(pdf_bytes, 'handbook.pdf'))")
    
    # Step 3: Extract implicit rules
    print("\n3. Extracting implicit rules...")
    extractor = ImplicitRuleExtractor()
    
    sample_text = """
    Assembly procedures require proper sequencing to ensure quality.
    Fasteners must be torqued to specification using calibrated tools.
    Visual inspection should verify alignment before final assembly.
    """
    
    rules = extractor.extract_implicit_rules(sample_text)
    print(f"   Extracted {len(rules)} rules")
    
    # Step 4: Query system (simulated)
    print("\n4. Querying the system...")
    print("   Query: 'What are the assembly requirements?'")
    print("   (In production: results = rag.query('assembly requirements'))")
    
    # Step 5: Get statistics
    print("\n5. System statistics:")
    stats = rag.get_stats()
    print(f"   Total documents: {stats['total_documents']}")
    print(f"   Embedding model: {stats['embedding_model']}")
    print(f"   LLM enabled: {stats['llm_enabled']}")
    
    print("\n✅ Example 5 complete - Full workflow demonstrated")

def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("TEXT PIPELINE AND RAG SYSTEM - EXAMPLE USAGE")
    print("="*70)
    print("\nThis script demonstrates various features of the RAG system.")
    print("All examples are self-contained and don't require external files.")
    
    try:
        # Run all examples
        example_1_basic_usage()
        example_2_implicit_extraction()
        example_3_text_extraction()
        example_4_system_status()
        example_5_practical_workflow()
        
        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\n✨ The RAG system is fully functional and ready to use!")
        print("\nNext steps:")
        print("  1. Process your own PDF documents")
        print("  2. Configure LLM API for enhanced accuracy (optional)")
        print("  3. Build custom applications using the RAG system")
        print("\nSee USAGE_GUIDE.md for detailed API documentation.")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("\nThis may be due to missing dependencies or model downloads.")
        print("Please ensure all required packages are installed:")
        print("  pip install sentence-transformers transformers torch chromadb")
        print("  pip install langchain langchain-chroma langchain-text-splitters")
        print("  pip install pdfminer.six PyPDF2 nltk numpy pandas")

if __name__ == "__main__":
    main()
