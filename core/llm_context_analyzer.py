"""
LLM-Based Context Analyzer for Generic Documents
Uses Groq/Cerebras APIs to understand documents with zero manufacturing keywords
"""

import os
from typing import List, Dict, Any, Optional, Tuple
import json
from dataclasses import dataclass
import re

# Try to import LLM clients
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

try:
    from cerebras.cloud.sdk import Cerebras
    CEREBRAS_AVAILABLE = True
except ImportError:
    CEREBRAS_AVAILABLE = False
    Cerebras = None


@dataclass
class DocumentContext:
    """Structured context extracted from document using LLM."""
    industry: str
    domain: str
    purpose: str
    key_concepts: List[str]
    implicit_requirements: List[str]
    constraint_types: List[str]
    manufacturing_relevance_score: float
    extracted_rules: List[Dict[str, Any]]
    confidence: float


class LLMContextAnalyzer:
    """
    Uses LLM APIs to understand generic documents and extract manufacturing rules.
    Works even when documents have ZERO manufacturing keywords.
    """
    
    def __init__(self, api_provider: str = "groq", model: str = None):
        """
        Initialize LLM Context Analyzer.
        
        Args:
            api_provider: "groq" or "cerebras"
            model: Specific model to use (or None for default)
        """
        self.api_provider = api_provider.lower()
        self.client = None
        
        # Setup API client
        if self.api_provider == "groq":
            if not GROQ_AVAILABLE:
                raise ImportError("Groq package not installed. Install with: pip install groq")
            
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set")
            
            self.client = Groq(api_key=api_key)
            self.model = model or "llama-3.3-70b-versatile"
            
        elif self.api_provider == "cerebras":
            if not CEREBRAS_AVAILABLE:
                raise ImportError("Cerebras package not installed. Install with: pip install cerebras-cloud-sdk")
            
            api_key = os.getenv("CEREBRAS_API_KEY")
            if not api_key:
                raise ValueError("CEREBRAS_API_KEY environment variable not set")
            
            self.client = Cerebras(api_key=api_key)
            self.model = model or "llama3.1-70b"
            
        else:
            raise ValueError(f"Unknown API provider: {api_provider}")
    
    def analyze_document_context(self, text: str, max_length: int = 4000) -> DocumentContext:
        """
        Analyze document using LLM to understand context even without keywords.
        
        Args:
            text: Document text (can be generic with zero manufacturing keywords)
            max_length: Maximum text length to analyze
        
        Returns:
            DocumentContext with extracted information
        """
        # Truncate if needed
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        # Create comprehensive prompt for context understanding
        prompt = self._create_context_analysis_prompt(text)
        
        # Get LLM response
        response = self._call_llm(prompt)
        
        # Parse response into structured context
        context = self._parse_context_response(response)
        
        return context
    
    def extract_manufacturing_rules(self, text: str, context: Optional[DocumentContext] = None) -> List[Dict[str, Any]]:
        """
        Extract manufacturing rules using LLM understanding.
        Works on generic text by leveraging context understanding.
        
        Args:
            text: Document text
            context: Pre-analyzed context (or will analyze if None)
        
        Returns:
            List of extracted rules with confidence scores
        """
        # Get context if not provided
        if context is None:
            context = self.analyze_document_context(text)
        
        # Create rule extraction prompt
        prompt = self._create_rule_extraction_prompt(text, context)
        
        # Get LLM response
        response = self._call_llm(prompt)
        
        # Parse rules from response
        rules = self._parse_rules_response(response, context)
        
        return rules
    
    def enhance_text_for_rag(self, text: str) -> Dict[str, Any]:
        """
        Enhance generic text with manufacturing context for better RAG indexing.
        
        Args:
            text: Original text (may have zero keywords)
        
        Returns:
            Enhanced text with manufacturing context tags
        """
        # Analyze context
        context = self.analyze_document_context(text)
        
        # Create enhanced version
        enhanced = {
            "original_text": text,
            "enhanced_text": self._add_context_tags(text, context),
            "context": context,
            "search_keywords": self._generate_search_keywords(context),
            "manufacturing_relevance": context.manufacturing_relevance_score
        }
        
        return enhanced
    
    def batch_analyze_sentences(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze multiple sentences efficiently using LLM.
        
        Args:
            sentences: List of sentences to analyze
        
        Returns:
            List of analyzed sentences with manufacturing context
        """
        # Batch sentences for efficiency
        batch_size = 10
        results = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            batch_text = "\n".join([f"{j+1}. {s}" for j, s in enumerate(batch)])
            
            prompt = f"""Analyze these sentences for manufacturing relevance and extract any implicit requirements or constraints:

{batch_text}

For each sentence, provide:
1. Manufacturing relevance (0-1 score)
2. Implicit requirements or rules (if any)
3. Domain/industry context
4. Key technical concepts

Format as JSON array with one object per sentence."""

            response = self._call_llm(prompt, temperature=0.3)
            batch_results = self._parse_batch_response(response, batch)
            results.extend(batch_results)
        
        return results
    
    def _create_context_analysis_prompt(self, text: str) -> str:
        """Create prompt for document context analysis."""
        return f"""You are an expert at understanding technical documents and extracting manufacturing-related information, even from generic text with no explicit manufacturing keywords.

Analyze the following document text and extract:

1. **Industry/Domain**: What industry or domain does this relate to? (e.g., electronics, mechanical, software, general business)
2. **Purpose**: What is the document's purpose? (requirements, guidelines, specifications, procedures)
3. **Key Concepts**: List the main technical concepts or topics discussed
4. **Implicit Requirements**: What requirements or constraints are implied, even if not explicitly stated?
5. **Constraint Types**: What types of constraints are present? (dimensional, performance, quality, material, environmental, etc.)
6. **Manufacturing Relevance**: How relevant is this to manufacturing? (0.0-1.0 score, where 0.7+ means high relevance)
7. **Extracted Rules**: Any rules, requirements, or guidelines that could apply to manufacturing

Document text:
```
{text}
```

Provide your analysis in JSON format:
{{
    "industry": "...",
    "domain": "...",
    "purpose": "...",
    "key_concepts": ["...", "..."],
    "implicit_requirements": ["...", "..."],
    "constraint_types": ["...", "..."],
    "manufacturing_relevance_score": 0.0-1.0,
    "extracted_rules": [
        {{"rule": "...", "type": "...", "confidence": 0.0-1.0, "rationale": "..."}}
    ],
    "confidence": 0.0-1.0
}}"""

    def _create_rule_extraction_prompt(self, text: str, context: DocumentContext) -> str:
        """Create prompt for rule extraction with context."""
        return f"""You are extracting manufacturing rules from a document.

**Document Context:**
- Industry: {context.industry}
- Domain: {context.domain}
- Purpose: {context.purpose}
- Key Concepts: {', '.join(context.key_concepts)}

**Task**: Extract ALL rules, requirements, constraints, or guidelines from the text that could apply to manufacturing, design, or production. Look for:
- Explicit requirements (must, shall, should)
- Implicit constraints (even if not obviously stated)
- Quality requirements
- Design guidelines
- Performance criteria
- Material specifications
- Environmental conditions
- Safety requirements

**Document Text**:
```
{text}
```

For each rule found, provide:
1. The rule text (verbatim or paraphrased)
2. Rule type (requirement, constraint, guideline, specification, etc.)
3. Confidence score (0.0-1.0)
4. Manufacturing relevance (how it applies to manufacturing)
5. Suggested rule category (quality, design, material, process, safety, etc.)

Format as JSON array:
[
    {{
        "rule_text": "...",
        "rule_type": "...",
        "confidence": 0.0-1.0,
        "manufacturing_relevance": "...",
        "category": "...",
        "original_context": "..."
    }}
]

Extract even vague or implied rules. Be thorough."""

    def _call_llm(self, prompt: str, temperature: float = 0.2, max_tokens: int = 2000) -> str:
        """Call the LLM API and return response."""
        try:
            if self.api_provider == "groq":
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert technical analyst specializing in manufacturing and engineering documents."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"} if "JSON" in prompt else None
                )
                return completion.choices[0].message.content
            
            elif self.api_provider == "cerebras":
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert technical analyst specializing in manufacturing and engineering documents."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return completion.choices[0].message.content
                
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}")
    
    def _parse_context_response(self, response: str) -> DocumentContext:
        """Parse LLM response into DocumentContext."""
        try:
            data = json.loads(response)
            
            return DocumentContext(
                industry=data.get("industry", "unknown"),
                domain=data.get("domain", "general"),
                purpose=data.get("purpose", "unspecified"),
                key_concepts=data.get("key_concepts", []),
                implicit_requirements=data.get("implicit_requirements", []),
                constraint_types=data.get("constraint_types", []),
                manufacturing_relevance_score=float(data.get("manufacturing_relevance_score", 0.5)),
                extracted_rules=data.get("extracted_rules", []),
                confidence=float(data.get("confidence", 0.5))
            )
        except Exception as e:
            # Fallback to basic context
            return DocumentContext(
                industry="unknown",
                domain="general",
                purpose="unspecified",
                key_concepts=[],
                implicit_requirements=[],
                constraint_types=[],
                manufacturing_relevance_score=0.5,
                extracted_rules=[],
                confidence=0.3
            )
    
    def _parse_rules_response(self, response: str, context: DocumentContext) -> List[Dict[str, Any]]:
        """Parse rules from LLM response."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                rules_data = json.loads(json_match.group())
            else:
                rules_data = json.loads(response)
            
            # Enhance rules with context
            rules = []
            for rule_data in rules_data:
                rule = {
                    "text": rule_data.get("rule_text", ""),
                    "type": rule_data.get("rule_type", "general"),
                    "confidence": float(rule_data.get("confidence", 0.5)),
                    "manufacturing_relevance": rule_data.get("manufacturing_relevance", ""),
                    "category": rule_data.get("category", "general"),
                    "original_context": rule_data.get("original_context", ""),
                    "document_industry": context.industry,
                    "document_domain": context.domain,
                    "extraction_method": "llm_analysis"
                }
                rules.append(rule)
            
            return rules
            
        except Exception as e:
            print(f"Error parsing rules response: {e}")
            return []
    
    def _parse_batch_response(self, response: str, sentences: List[str]) -> List[Dict[str, Any]]:
        """Parse batch analysis response."""
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                results_data = json.loads(json_match.group())
            else:
                results_data = json.loads(response)
            
            # Match results with sentences
            results = []
            for i, (sentence, data) in enumerate(zip(sentences, results_data)):
                result = {
                    "sentence": sentence,
                    "manufacturing_relevance": data.get("manufacturing_relevance", 0.0),
                    "implicit_requirements": data.get("implicit_requirements", []),
                    "domain_context": data.get("domain_context", ""),
                    "key_concepts": data.get("key_concepts", [])
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            # Return default results
            return [{"sentence": s, "manufacturing_relevance": 0.5, "implicit_requirements": [], "domain_context": "", "key_concepts": []} for s in sentences]
    
    def _add_context_tags(self, text: str, context: DocumentContext) -> str:
        """Add context tags to text for better RAG indexing."""
        tags = [
            f"[INDUSTRY: {context.industry}]",
            f"[DOMAIN: {context.domain}]",
            f"[PURPOSE: {context.purpose}]",
        ]
        
        if context.key_concepts:
            tags.append(f"[CONCEPTS: {', '.join(context.key_concepts[:5])}]")
        
        return " ".join(tags) + "\n\n" + text
    
    def _generate_search_keywords(self, context: DocumentContext) -> List[str]:
        """Generate search keywords from context for RAG retrieval."""
        keywords = []
        keywords.append(context.industry)
        keywords.append(context.domain)
        keywords.extend(context.key_concepts)
        keywords.extend(context.constraint_types)
        
        # Add generic manufacturing terms based on context
        if context.manufacturing_relevance_score > 0.6:
            keywords.extend(["manufacturing", "design", "quality", "requirements", "specifications"])
        
        return list(set(keywords))


# Helper function to check API availability
def check_api_availability() -> Dict[str, bool]:
    """Check which LLM APIs are available."""
    availability = {
        "groq": False,
        "cerebras": False
    }
    
    if GROQ_AVAILABLE and os.getenv("GROQ_API_KEY"):
        availability["groq"] = True
    
    if CEREBRAS_AVAILABLE and os.getenv("CEREBRAS_API_KEY"):
        availability["cerebras"] = True
    
    return availability


def get_default_analyzer() -> Optional[LLMContextAnalyzer]:
    """Get default LLM analyzer (prefers Groq, falls back to Cerebras)."""
    availability = check_api_availability()
    
    if availability["groq"]:
        return LLMContextAnalyzer(api_provider="groq")
    elif availability["cerebras"]:
        return LLMContextAnalyzer(api_provider="cerebras")
    else:
        return None


if __name__ == "__main__":
    # Test the analyzer
    print("Testing LLM Context Analyzer...")
    
    availability = check_api_availability()
    print(f"API Availability: {availability}")
    
    if not any(availability.values()):
        print("\n⚠️  No LLM APIs available. Set GROQ_API_KEY or CEREBRAS_API_KEY environment variable.")
        print("\nTo get API keys:")
        print("  - Groq: https://console.groq.com/keys")
        print("  - Cerebras: https://cloud.cerebras.ai/")
    else:
        analyzer = get_default_analyzer()
        
        # Test with generic text (zero manufacturing keywords)
        test_text = """
        The system should maintain reliable operation under varying environmental conditions.
        Components must be designed to facilitate easy maintenance and replacement.
        All interfaces should provide adequate clearance for access during servicing.
        Materials should be selected to ensure longevity and durability.
        """
        
        print("\nAnalyzing generic text...")
        context = analyzer.analyze_document_context(test_text)
        print(f"\nExtracted Context:")
        print(f"  Industry: {context.industry}")
        print(f"  Domain: {context.domain}")
        print(f"  Manufacturing Relevance: {context.manufacturing_relevance_score:.2f}")
        print(f"  Key Concepts: {context.key_concepts}")
        
        print("\nExtracting rules...")
        rules = analyzer.extract_manufacturing_rules(test_text, context)
        print(f"  Found {len(rules)} rules")
        for i, rule in enumerate(rules, 1):
            print(f"\n  Rule {i}:")
            print(f"    Text: {rule['text'][:80]}...")
            print(f"    Type: {rule['type']}")
            print(f"    Confidence: {rule['confidence']:.2f}")
