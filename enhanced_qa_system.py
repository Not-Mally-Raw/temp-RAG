#!/usr/bin/env python3
"""
Enhanced RAG Question Answering with LLM and Citation System
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import tempfile
from pathlib import Path

class RAGQuestionAnswerer:
    """Enhanced RAG system for detailed question answering with citations."""
    
    def __init__(self, rag_pipeline):
        self.rag_pipeline = rag_pipeline
        self.extracted_rules_db = []  # Store all extracted rules
        
    def answer_question_with_citations(
        self, 
        question: str, 
        manufacturing_process: Optional[str] = None,
        top_k: int = 5,
        model: str = "llama-3.1-8b-instant"
    ) -> Dict[str, Any]:
        """
        Generate detailed LLM answer with citations for a question.
        
        Returns:
        {
            "question": str,
            "detailed_answer": str (LLM-generated),
            "citations": List[Dict] (source references),
            "extracted_rules": List[Dict] (rules found),
            "confidence_score": float,
            "metadata": Dict
        }
        """
        
        # Step 1: Retrieve relevant context from RAG
        rag_results = self.rag_pipeline.search_knowledge_base(
            query=question,
            manufacturing_process=manufacturing_process,
            top_k=top_k
        )
        
        # Step 2: Build context for LLM
        context_text = self._build_context_text(rag_results)
        citations = self._format_citations(rag_results)
        
        # Step 3: Generate detailed answer using LLM
        detailed_answer = self._generate_llm_answer(question, context_text, model)
        
        # Step 4: Extract rules from the answer and context
        extracted_rules = self._extract_rules_from_answer(detailed_answer, rag_results)
        
        # Step 5: Store rules in database
        self._store_rules_in_db(extracted_rules, question, manufacturing_process)
        
        return {
            "question": question,
            "detailed_answer": detailed_answer,
            "citations": citations,
            "extracted_rules": extracted_rules,
            "confidence_score": self._calculate_confidence(rag_results),
            "metadata": {
                "retrieval_count": len(rag_results),
                "manufacturing_process": manufacturing_process,
                "model_used": model,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _build_context_text(self, rag_results: List[Dict]) -> str:
        """Build context text for LLM from RAG results."""
        if not rag_results:
            return "No relevant context found."
        
        context_parts = []
        for i, result in enumerate(rag_results, 1):
            source = result.get('metadata', {}).get('source_file', 'Unknown')
            text = result.get('text', '')
            score = result.get('similarity_score', 0.0)
            
            context_parts.append(f"""
[SOURCE {i}] {source} (Relevance: {score:.3f})
{text}
""")
        
        return "\\n".join(context_parts)
    
    def _format_citations(self, rag_results: List[Dict]) -> List[Dict]:
        """Format citations in academic style."""
        citations = []
        for i, result in enumerate(rag_results, 1):
            metadata = result.get('metadata', {})
            citations.append({
                "citation_id": i,
                "source_file": metadata.get('source_file', 'Unknown'),
                "page_number": metadata.get('page_number', 'N/A'),
                "section": metadata.get('section_title', 'N/A'),
                "manufacturing_process": metadata.get('manufacturing_process', 'General'),
                "relevance_score": result.get('similarity_score', 0.0),
                "excerpt": result.get('text', '')[:200] + "..." if len(result.get('text', '')) > 200 else result.get('text', ''),
                "citation_format": f"[{i}] {metadata.get('source_file', 'Unknown')}, {metadata.get('section_title', 'Section Unknown')}"
            })
        return citations
    
    def _generate_llm_answer(self, question: str, context: str, model: str) -> str:
        """Generate detailed answer using LLM."""
        
        # Check if Groq API is available
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            return self._generate_fallback_answer(question, context)
        
        try:
            from groq import Groq
            client = Groq(api_key=groq_api_key)
            
            prompt = f"""You are a manufacturing engineering expert. Based on the provided context from technical documents, provide a detailed, comprehensive answer to the question. 

IMPORTANT: 
- Include specific technical details, measurements, and requirements
- Reference the sources using [1], [2], etc. notation where appropriate
- Provide practical guidance and best practices
- If multiple approaches exist, explain the differences
- Include any relevant warnings or considerations

CONTEXT FROM DOCUMENTS:
{context}

QUESTION: {question}

DETAILED ANSWER:"""

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2048,
                top_p=0.95
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"LLM generation failed: {e}")
            return self._generate_fallback_answer(question, context)
    
    def _generate_fallback_answer(self, question: str, context: str) -> str:
        """Generate fallback answer when LLM is not available."""
        return f"""Based on the available technical documentation context:

{context}

This information provides relevant guidance for: {question}

Note: This is a context-based response. For detailed analysis, please ensure LLM integration is configured with GROQ_API_KEY."""
    
    def _extract_rules_from_answer(self, answer: str, rag_results: List[Dict]) -> List[Dict]:
        """Extract manufacturing rules from the LLM answer and context."""
        rules = []
        
        # Extract rules from each source
        for i, result in enumerate(rag_results, 1):
            text = result.get('text', '')
            metadata = result.get('metadata', {})
            
            # Look for rule patterns in the text
            rule_patterns = [
                "must be", "should be", "shall be", "minimum", "maximum",
                "at least", "no more than", "between", "tolerance", "requirement"
            ]
            
            sentences = text.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if any(pattern in sentence.lower() for pattern in rule_patterns) and len(sentence) > 20:
                    rules.append({
                        "rule_text": sentence,
                        "source_file": metadata.get('source_file', 'Unknown'),
                        "source_citation": f"[{i}]",
                        "manufacturing_process": metadata.get('manufacturing_process', 'General'),
                        "rule_type": self._classify_rule_type(sentence),
                        "confidence": result.get('similarity_score', 0.0),
                        "extraction_method": "RAG + Pattern Matching",
                        "timestamp": datetime.now().isoformat()
                    })
        
        return rules
    
    def _classify_rule_type(self, rule_text: str) -> str:
        """Classify the type of manufacturing rule."""
        text_lower = rule_text.lower()
        
        if any(word in text_lower for word in ['bend', 'radius', 'angle']):
            return "Sheet Metal"
        elif any(word in text_lower for word in ['hole', 'diameter', 'drill']):
            return "Machining"
        elif any(word in text_lower for word in ['tolerance', 'dimension']):
            return "Dimensional"
        elif any(word in text_lower for word in ['material', 'grade', 'alloy']):
            return "Material"
        else:
            return "General"
    
    def _calculate_confidence(self, rag_results: List[Dict]) -> float:
        """Calculate confidence score based on retrieval quality."""
        if not rag_results:
            return 0.0
        
        scores = [result.get('similarity_score', 0.0) for result in rag_results]
        return sum(scores) / len(scores)
    
    def _store_rules_in_db(self, rules: List[Dict], question: str, manufacturing_process: Optional[str]):
        """Store extracted rules in the internal database."""
        for rule in rules:
            rule['query_question'] = question
            rule['query_process'] = manufacturing_process
            self.extracted_rules_db.append(rule)
    
    def export_rules_to_csv(self, filename: Optional[str] = None) -> str:
        """Export all extracted rules to CSV file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"extracted_manufacturing_rules_{timestamp}.csv"
        
        if not self.extracted_rules_db:
            print("No rules extracted yet.")
            return ""
        
        df = pd.DataFrame(self.extracted_rules_db)
        df.to_csv(filename, index=False)
        
        print(f"✅ Exported {len(self.extracted_rules_db)} rules to {filename}")
        return filename
    
    def get_rules_summary(self) -> Dict[str, Any]:
        """Get summary of extracted rules."""
        if not self.extracted_rules_db:
            return {
                "total_rules": 0,
                "rules_by_type": {},
                "rules_by_process": {},
                "rules_by_source": {},
                "average_confidence": 0.0,
                "extraction_methods": {}
            }
        
        df = pd.DataFrame(self.extracted_rules_db)
        
        return {
            "total_rules": len(self.extracted_rules_db),
            "rules_by_type": df['rule_type'].value_counts().to_dict(),
            "rules_by_process": df['manufacturing_process'].value_counts().to_dict(),
            "rules_by_source": df['source_file'].value_counts().to_dict(),
            "average_confidence": df['confidence'].mean(),
            "extraction_methods": df['extraction_method'].value_counts().to_dict()
        }

def test_enhanced_qa_system():
    """Test the enhanced question answering system."""
    from core.rag_pipeline_integration import RAGIntegratedPipeline
    
    print("🔬 TESTING ENHANCED QA SYSTEM")
    print("=" * 60)
    
    # Initialize
    pipeline = RAGIntegratedPipeline()
    qa_system = RAGQuestionAnswerer(pipeline)
    
    # Add some test data first (from our previous test)
    from core.rag_database import DocumentMetadata
    
    mock_content = """
    Manufacturing Design Guidelines for Sheet Metal Parts
    
    Bend Radius Requirements:
    The minimum bend radius must be at least 1.5 times the material thickness.
    For aluminum sheets, the bend radius should be 2.0 times thickness minimum.
    
    Hole Specifications:
    Minimum hole diameter must be 0.5mm or greater for manufacturing feasibility.
    Distance between holes should be at least 2 times the hole diameter.
    """
    
    doc_metadata = DocumentMetadata(
        doc_id="test_qa_001",
        source_file="Manufacturing_Guidelines_V2.pdf",
        doc_type="text"
    )
    
    documents = pipeline.rag_system.text_splitter.split_with_structure(
        text=mock_content, metadata=doc_metadata
    )
    pipeline.rag_system._add_documents_to_vectorstore(documents)
    
    # Test question answering
    question = "What are the bend radius requirements for sheet metal parts?"
    
    print(f"\n📋 Question: {question}")
    
    # Debug: Check if documents are in the system
    stats = pipeline.get_knowledge_base_summary()
    print(f"\n🔍 Debug - Database stats: {stats}")
    
    print("\n🔍 Processing with enhanced QA system...")
    
    result = qa_system.answer_question_with_citations(
        question=question,
        manufacturing_process="Sheet Metal",
        top_k=3
    )
    
    print("\\n✅ DETAILED ANSWER:")
    print("-" * 40)
    print(result['detailed_answer'])
    
    print("\\n📚 CITATIONS:")
    print("-" * 40)
    for citation in result['citations']:
        print(f"{citation['citation_format']}")
        print(f"   Process: {citation['manufacturing_process']}")
        print(f"   Relevance: {citation['relevance_score']:.3f}")
        print(f"   Excerpt: {citation['excerpt']}")
        print()
    
    print("\\n⚙️ EXTRACTED RULES:")
    print("-" * 40)
    for rule in result['extracted_rules']:
        print(f"• {rule['rule_text']}")
        print(f"  Type: {rule['rule_type']} | Source: {rule['source_citation']} | Confidence: {rule['confidence']:.3f}")
        print()
    
    # Export to CSV
    csv_file = qa_system.export_rules_to_csv("test_extracted_rules.csv")
    
    # Summary
    summary = qa_system.get_rules_summary()
    print("\\n📊 RULES SUMMARY:")
    print("-" * 40)
    print(f"Total Rules: {summary['total_rules']}")
    print(f"Rules by Type: {summary['rules_by_type']}")
    print(f"Average Confidence: {summary['average_confidence']:.3f}")
    
    return True

if __name__ == "__main__":
    test_enhanced_qa_system()