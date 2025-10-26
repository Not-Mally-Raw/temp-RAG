"""
Enhanced LLM Prompt System for Manufacturing Rule Extraction
Integrates Langchain PromptTemplates with structured JSON outputs and context refinement
"""

import time
import random
from typing import List, Dict, Any, Optional, Union, Callable
import json
import re
from dataclasses import dataclass
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field, ValidationError
import logging

logger = logging.getLogger(__name__)

@dataclass
class RetryConfig:
    """Configuration for retry logic with exponential backoff."""
    max_attempts: int = 10
    base_delay: float = 0.5
    max_delay: float = 60.0
    jitter: bool = True
    backoff_factor: float = 2.0

def retry_with_exponential_backoff(
    func: Callable,
    config: RetryConfig = None,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator for implementing retry logic with exponential backoff.
    Industry standard for production RAG systems.
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(*args, **kwargs):
        last_exception = None
        
        for attempt in range(config.max_attempts):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                last_exception = e
                
                if attempt == config.max_attempts - 1:
                    logger.error(f"Function {func.__name__} failed after {config.max_attempts} attempts: {e}")
                    raise e
                
                # Calculate delay with exponential backoff
                delay = min(
                    config.base_delay * (config.backoff_factor ** attempt),
                    config.max_delay
                )
                
                # Add jitter to prevent thundering herd
                if config.jitter:
                    delay *= (0.5 + random.random())
                
                logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay:.2f}s")
                time.sleep(delay)
        
        raise last_exception
    
    return decorator

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Robust JSON extraction from text that might contain explanations or errors.
    Industry-standard implementation with self-healing.
    
    Args:
        text: Input text that may contain JSON
        
    Returns:
        Parsed JSON dict or None if extraction fails
    """
    if not text or not isinstance(text, str):
        return None
        
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # Clean markdown and explanations
    cleaned = text.replace('```json', '').replace('```', '').strip()
    
    # Try direct parse on cleaned text
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Find JSON objects in text using regex
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, cleaned, re.DOTALL)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    # Try to find content between first { and last }
    start_idx = cleaned.find('{')
    end_idx = cleaned.rfind('}')
    
    if start_idx != -1 and end_idx > start_idx:
        try:
            return json.loads(cleaned[start_idx:end_idx + 1])
        except json.JSONDecodeError:
            pass
    
    logger.warning(f"Failed to extract JSON from text: {text[:200]}...")
    return None

def call_llm_to_fix_json(json_string: str, error_msg: str, llm_client=None, max_retries: int = 2) -> Optional[Dict[str, Any]]:
    """
    Industry-standard JSON self-healing using LLM.
    
    Args:
        json_string: Malformed JSON string
        error_msg: Error message from JSON parser
        llm_client: LLM client for fixing (optional)
        max_retries: Maximum retry attempts
        
    Returns:
        Fixed JSON dict or None if unfixable
    """
    if not llm_client:
        # Return None if no LLM available for fixing
        return None
    
    fix_prompt = f"""
Fix this malformed JSON. Return ONLY the corrected JSON, no explanations.

Error: {error_msg}

Malformed JSON:
{json_string}

Rules:
1. Output ONLY valid JSON
2. Preserve all data values
3. Fix syntax errors (missing quotes, commas, brackets)
4. Ensure proper field names

Corrected JSON:
"""
    
    try:
        for attempt in range(max_retries):
            try:
                response = llm_client.generate(fix_prompt)
                return extract_json_from_text(response)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"JSON fixing failed after {max_retries} attempts: {e}")
                    return None
    except Exception as e:
        logger.error(f"LLM JSON fixing error: {e}")
        return None

def safe_pydantic_parse(parser, text: str, fallback_data: Dict[str, Any] = None):
    """
    Safely parse text with Pydantic parser, providing fallback on failure.
    
    Args:
        parser: Pydantic output parser
        text: Text to parse
        fallback_data: Fallback data to use if parsing fails
        
    Returns:
        Parsed object or fallback data
    """
    try:
        # First try direct parsing
        return parser.parse(text)
    except Exception as e:
        logger.warning(f"Direct parsing failed: {e}")
        
        # Try JSON extraction then parsing
        json_data = extract_json_from_text(text)
        if json_data:
            try:
                return parser.pydantic_object(**json_data)
            except Exception as e2:
                logger.warning(f"JSON parsing failed: {e2}")
        
        # Use fallback data if provided
        if fallback_data:
            try:
                return parser.pydantic_object(**fallback_data)
            except Exception as e3:
                logger.error(f"Fallback parsing failed: {e3}")
        
        # Return empty/default instance
        try:
            return parser.pydantic_object()
        except Exception as e4:
            logger.error(f"Default construction failed: {e4}")
            return None

# Pydantic models for structured outputs
class ManufacturingRule(BaseModel):
    """Structured manufacturing rule with all required fields."""
    rule_category: str = Field(description="Manufacturing category (Sheet Metal, Injection Molding, etc.)")
    name: str = Field(description="Descriptive name of the rule")
    feature1: str = Field(description="Primary manufacturing feature")
    feature2: Optional[str] = Field(default="", description="Secondary feature if applicable")
    object1: str = Field(description="Primary object/component")
    object2: Optional[str] = Field(default="", description="Secondary object if applicable") 
    exp_name: str = Field(description="Expression name with parameters")
    operator: str = Field(description="Comparison operator (>=, <=, ==, between)")
    recom: Union[float, str] = Field(description="Recommended value or range")
    constraint: Optional[Dict[str, Any]] = Field(default=None, description="Additional constraints")
    confidence: float = Field(description="Confidence score 0.0-1.0")
    manufacturing_relevance: float = Field(description="Manufacturing relevance score 0.0-1.0")
    extracted_entities: List[str] = Field(default_factory=list, description="Key entities extracted")
    rationale: str = Field(description="Reasoning for rule extraction")

class RuleExtractionResult(BaseModel):
    """Complete result from rule extraction."""
    rules: List[ManufacturingRule] = Field(description="Extracted manufacturing rules")
    document_context: Dict[str, Any] = Field(description="Document analysis context")
    processing_metadata: Dict[str, Any] = Field(description="Processing statistics")

class DocumentContext(BaseModel):
    """Document context analysis result."""
    industry: str = Field(default="Unknown", description="Identified industry sector")
    domain: str = Field(default="General", description="Technical domain")
    purpose: str = Field(default="Unspecified", description="Document purpose")
    key_concepts: List[str] = Field(default_factory=list, description="Key technical concepts")
    manufacturing_relevance_score: float = Field(default=0.5, description="Overall manufacturing relevance")
    implicit_requirements: List[str] = Field(default_factory=list, description="Implicit requirements identified")
    constraint_types: List[str] = Field(default_factory=list, description="Types of constraints present")

class LLMPromptSystem:
    """Enhanced prompting system with Langchain integration and structured outputs."""
    
    def __init__(self):
        """Initialize the enhanced prompting system."""
        self.rule_parser = PydanticOutputParser(pydantic_object=ManufacturingRule)
        self.context_parser = PydanticOutputParser(pydantic_object=DocumentContext)
        self.result_parser = PydanticOutputParser(pydantic_object=RuleExtractionResult)
        
        # Manufacturing categories from Phase-3-Final-master
        self.manufacturing_categories = [
            "Sheet Metal", "Injection Molding", "Machining", "Assembly", 
            "Welding", "Casting", "Electronics", "Quality Control",
            "Material Specification", "Safety Requirement", "Design Guideline"
        ]
        
        # Features dictionary reference (simplified)
        self.features_dict = {
            "Sheet Metal": ["thickness", "bend_radius", "hole_diameter", "flange_width"],
            "Injection Molding": ["wall_thickness", "draft_angle", "rib_thickness", "gate_size"],
            "Machining": ["surface_roughness", "tolerance", "tool_diameter", "feed_rate"],
            "Assembly": ["clearance", "fastener_type", "torque_spec", "alignment"],
            "General": ["dimension", "material", "process", "quality"]
        }
        
    def create_context_analysis_prompt(self) -> ChatPromptTemplate:
        """Create prompt for document context analysis."""
        
        system_message = SystemMessagePromptTemplate.from_template(
            """You are an expert manufacturing document analyzer. Your task is to analyze documents and extract contextual information that will help identify manufacturing rules and requirements.

You must analyze the document for:
1. Industry sector (automotive, aerospace, pharmaceutical, electronics, etc.)
2. Technical domain (design, quality, safety, materials, processes)
3. Document purpose (specification, standard, guideline, procedure)
4. Key technical concepts and terminology
5. Manufacturing relevance (how much it relates to manufacturing)
6. Implicit requirements (things that are implied but not explicitly stated)
7. Types of constraints present (dimensional, material, process, safety, etc.)

Provide analysis in structured JSON format following the exact schema provided."""
        )
        
        human_message = HumanMessagePromptTemplate.from_template(
            """Analyze this document text and provide comprehensive context analysis:

Document Text:
```
{document_text}
```

{format_instructions}

Focus on identifying manufacturing-relevant information even if not explicitly stated. Look for implied requirements, constraints, and guidelines that could impact manufacturing processes."""
        )
        
        return ChatPromptTemplate.from_messages([system_message, human_message])
    
    def create_production_rule_extraction_prompt(self) -> ChatPromptTemplate:
        """
        Industry-standard rule extraction prompt with structured JSON output.
        Fixes field mapping issues and ensures proper rule structure.
        """
        
        system_message = SystemMessagePromptTemplate.from_template(
            """You are a manufacturing rule extraction expert. Extract ONE specific manufacturing rule from the text.

CRITICAL: Return ONLY valid JSON with these EXACT fields:
{{
  "rule_category": "one of: Sheet Metal, Injection Molding, Machining, Quality Control, Assembly, Electronics, General",
  "name": "specific rule name (max 100 characters)",
  "feature1": "primary feature being constrained",
  "feature2": "secondary feature (use '' if none)",
  "object1": "primary object/component",
  "object2": "secondary object (use '' if none)",
  "exp_name": "expression with parameters",
  "operator": "one of: >=, <=, ==, !=, >, <, between",
  "recom": numeric_value_as_number,
  "constraint": {{"additional": "constraints"}},
  "confidence": number_between_0_and_1,
  "manufacturing_relevance": number_between_0_and_1,
  "extracted_entities": ["key", "entities"],
  "rationale": "why this rule matters (max 200 characters)"
}}

EXTRACTION RULES:
1. Extract ONE manufacturing rule per call
2. Use specific categories from the list above
3. recom must be a NUMBER, not a string
4. If no specific value mentioned, estimate based on context
5. Return ONLY the JSON object, no markdown, no explanations
6. All text fields must respect character limits

MANUFACTURING CATEGORIES: {manufacturing_categories}"""
        )
        
        human_message = HumanMessagePromptTemplate.from_template(
            """Extract ONE manufacturing rule from this text:

Text: {document_text}

Context:
- Industry: {industry}
- Domain: {domain}

JSON Output (ONLY):"""
        )
        
        return ChatPromptTemplate.from_messages([system_message, human_message])

    def create_rule_extraction_prompt(self) -> ChatPromptTemplate:
        """Create enhanced prompt for manufacturing rule extraction."""
        
        system_message = SystemMessagePromptTemplate.from_template(
            """You are an expert manufacturing rule extraction system. Extract ALL manufacturing rules, requirements, constraints, and guidelines from documents.

EXTRACTION CRITERIA:
- Look for explicit requirements (must, shall, should, required)
- Identify implicit constraints (even if not obviously manufacturing-related)
- Extract quality requirements and specifications
- Find design guidelines and recommendations
- Identify material specifications and properties
- Detect process parameters and conditions
- Recognize safety requirements and constraints
- Capture performance criteria and limits

RULE CATEGORIES: {manufacturing_categories}

FEATURES BY CATEGORY: {features_reference}

OUTPUT REQUIREMENTS:
- Extract even vague or implied rules
- Assign confidence scores based on clarity and specificity
- Calculate manufacturing relevance for each rule
- Provide rationale for each extraction
- Use structured JSON format with all required fields
- Limit rule text to 200 characters maximum
- Ensure numerical values are properly extracted and formatted"""
        )
        
        human_message = HumanMessagePromptTemplate.from_template(
            """Extract manufacturing rules from this document section.

Document Context:
- Industry: {industry}
- Domain: {domain}  
- Purpose: {purpose}
- Key Concepts: {key_concepts}

RAG Knowledge Base Context:
{rag_context}

Document Text:
```
{document_text}
```

{format_instructions}

Extract ALL rules following the structured format. Be thorough and include even subtle requirements."""
        )
        
        return ChatPromptTemplate.from_messages([system_message, human_message])
    
    def create_rule_refinement_prompt(self) -> ChatPromptTemplate:
        """Create prompt for refining and enhancing extracted rules."""
        
        system_message = SystemMessagePromptTemplate.from_template(
            """You are a manufacturing rule refinement specialist. Your task is to enhance and refine extracted manufacturing rules by:

1. Standardizing terminology and expressions
2. Adding missing technical details
3. Improving clarity and specificity
4. Ensuring proper categorization
5. Validating numerical values and units
6. Adding relevant constraints and conditions
7. Enhancing manufacturing relevance

Use your knowledge of manufacturing standards and best practices to improve rule quality while maintaining accuracy to the source."""
        )
        
        human_message = HumanMessagePromptTemplate.from_template(
            """Refine and enhance these extracted manufacturing rules:

Original Rules:
{extracted_rules}

Document Context:
{document_context}

RAG Context (Similar Rules):
{similar_rules}

Requirements:
- Enhance clarity while preserving original meaning
- Add technical details where appropriate
- Standardize expressions and terminology
- Ensure proper manufacturing categorization
- Validate and improve confidence scores
- Add missing constraints or conditions
- Limit refined rule text to 200 characters

{format_instructions}

Provide enhanced rules in structured JSON format."""
        )
        
        return ChatPromptTemplate.from_messages([system_message, human_message])
    
    def create_quality_assessment_prompt(self) -> PromptTemplate:
        """Create prompt for assessing rule extraction quality."""
        
        return PromptTemplate(
            input_variables=["extracted_rules", "source_text", "extraction_criteria"],
            template="""Assess the quality of these extracted manufacturing rules against the source text.

Extraction Criteria:
{extraction_criteria}

Source Text:
{source_text}

Extracted Rules:
{extracted_rules}

Evaluate each rule on:
1. Accuracy (how well it represents the source)
2. Completeness (are all relevant rules extracted)
3. Clarity (is the rule clearly stated)
4. Manufacturing Relevance (applicability to manufacturing)
5. Consistency (proper categorization and formatting)

Provide assessment as JSON:
{{
    "overall_quality_score": 0.0-1.0,
    "accuracy_score": 0.0-1.0,
    "completeness_score": 0.0-1.0,
    "clarity_score": 0.0-1.0,
    "relevance_score": 0.0-1.0,
    "consistency_score": 0.0-1.0,
    "missing_rules": ["rule1", "rule2"],
    "quality_issues": ["issue1", "issue2"],
    "recommendations": ["rec1", "rec2"]
}}"""
        )
    
    def create_multi_document_synthesis_prompt(self) -> ChatPromptTemplate:
        """Create prompt for synthesizing rules across multiple documents."""
        
        system_message = SystemMessagePromptTemplate.from_template(
            """You are a manufacturing knowledge synthesis expert. Your task is to combine and harmonize manufacturing rules extracted from multiple documents to create a comprehensive, consistent rule set.

Synthesis Objectives:
1. Identify overlapping and conflicting rules
2. Harmonize different expressions of the same rule
3. Create unified rule categories and naming
4. Resolve conflicts using best practices
5. Maintain traceability to source documents
6. Enhance rules with cross-document insights"""
        )
        
        human_message = HumanMessagePromptTemplate.from_template(
            """Synthesize manufacturing rules from multiple documents:

Document Rules:
{multi_document_rules}

Document Contexts:
{document_contexts}

Synthesis Requirements:
- Identify and resolve rule conflicts
- Harmonize terminology and expressions
- Create unified categorization
- Maintain source traceability
- Enhance with cross-document insights
- Flag conflicting requirements
- Provide synthesis confidence scores

{format_instructions}

Output comprehensive synthesized rule set in structured JSON format."""
        )
        
        return ChatPromptTemplate.from_messages([system_message, human_message])
    
    def format_rag_context(self, rag_context: Dict[str, Any]) -> str:
        """Format RAG context for prompt inclusion."""
        if not rag_context:
            return "No RAG context available."
        
        context_parts = []
        
        if 'retrieved_context' in rag_context:
            context_parts.append("Retrieved Context:")
            for i, ctx in enumerate(rag_context['retrieved_context'][:3]):
                context_parts.append(f"  {i+1}. {ctx.get('text', '')[:150]}...")
        
        if 'manufacturing_features' in rag_context:
            context_parts.append(f"Manufacturing Features: {', '.join(rag_context['manufacturing_features'])}")
        
        if 'related_constraints' in rag_context:
            context_parts.append(f"Related Constraints: {', '.join(rag_context['related_constraints'])}")
        
        if 'manufacturing_standards' in rag_context:
            context_parts.append(f"Standards: {', '.join(rag_context['manufacturing_standards'])}")
        
        return "\n".join(context_parts)
    
    def validate_and_clean_json_output(self, llm_output: str) -> Dict[str, Any]:
        """Clean and validate JSON output from LLM using robust extraction."""
        # Use our robust JSON extraction
        parsed = extract_json_from_text(llm_output)
        
        if parsed is not None:
            return parsed
        
        # If extraction failed, return error structure
        logger.error(f"Failed to extract JSON from LLM output: {llm_output[:200]}...")
        return {
            "rules": [],
            "error": f"JSON extraction failed from output",
            "raw_output": llm_output[:500]  # Limit raw output size
        }
    
    def apply_text_limits(self, rule_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply text length limits to rule data."""
        if isinstance(rule_data, dict):
            # Limit rule text to 200 characters
            if 'rule_text' in rule_data and len(rule_data['rule_text']) > 200:
                rule_data['rule_text'] = rule_data['rule_text'][:197] + "..."
            
            # Limit name to 100 characters
            if 'name' in rule_data and len(rule_data['name']) > 100:
                rule_data['name'] = rule_data['name'][:97] + "..."
            
            # Limit rationale to 300 characters
            if 'rationale' in rule_data and len(rule_data['rationale']) > 300:
                rule_data['rationale'] = rule_data['rationale'][:297] + "..."
        
        return rule_data
    
    def get_format_instructions(self, parser_type: str = "rule") -> str:
        """Get format instructions for different parser types."""
        if parser_type == "rule":
            return self.rule_parser.get_format_instructions()
        elif parser_type == "context":
            return self.context_parser.get_format_instructions()
        elif parser_type == "result":
            return self.result_parser.get_format_instructions()
        else:
            return "Provide output in valid JSON format."

# Example usage and integration with existing system
def create_enhanced_llm_processor():
    """Factory function to create enhanced LLM processor."""
    return LLMPromptSystem()