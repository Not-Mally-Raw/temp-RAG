"""
Enhanced Rule Extractor with Advanced LLM Integration
Integrates the new prompting system with existing rule extraction capabilities
"""

import json
import logging
import time
import random
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict, is_dataclass
import re
from datetime import datetime
from pydantic import ValidationError

# Langchain imports
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline

# Monitoring
from .monitoring import get_monitor, monitor_performance

def safe_to_dict(obj: Any) -> Dict[str, Any]:
    """
    Safely convert an object to dictionary, handling both dataclasses and Pydantic models.
    
    Args:
        obj: Object to convert (dataclass, Pydantic model, or dict)
        
    Returns:
        Dictionary representation of the object
    """
    if obj is None:
        return {}
    
    if isinstance(obj, dict):
        return obj
    
    # Check if it's a dataclass
    if is_dataclass(obj):
        return asdict(obj)
    
    # Check if it's a Pydantic model (has dict method)
    if hasattr(obj, 'dict') and callable(getattr(obj, 'dict')):
        return obj.dict()
    
    # Check if it's a Pydantic model (has model_dump method - newer versions)
    if hasattr(obj, 'model_dump') and callable(getattr(obj, 'model_dump')):
        return obj.model_dump()
    
    # Fallback: try to convert to dict
    try:
        return dict(obj)
    except (TypeError, ValueError):
        return {"error": f"Could not convert {type(obj)} to dict"}

def chunk_text(text: str, max_tokens: int = 900, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks to handle token length limits.
    
    Args:
        text: Input text to chunk
        max_tokens: Maximum tokens per chunk (approximate word count)
        overlap: Number of tokens to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    words = text.split()
    
    # If text is short enough, return as-is
    if len(words) <= max_tokens:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        
        # Move start forward, accounting for overlap
        start = end - overlap
        if start >= len(words):
            break
    
    return chunks
from langchain_community.chat_models import ChatOllama

# Local imports
from .llm_prompts import (
    LLMPromptSystem, 
    ManufacturingRule, 
    DocumentContext, 
    RuleExtractionResult,
    extract_json_from_text, 
    call_llm_to_fix_json,
    retry_with_exponential_backoff
)
from .implicit_rule_extractor import ImplicitRuleExtractor, ImplicitRule

logger = logging.getLogger(__name__)

@dataclass
class ExtractionConfig:
    """Configuration for enhanced rule extraction."""
    max_rule_length: int = 200
    max_rules_per_chunk: int = 10
    min_confidence_threshold: float = 0.3
    enable_rule_refinement: bool = True
    enable_context_analysis: bool = True
    enable_multi_pass_extraction: bool = True
    llm_temperature: float = 0.1
    max_tokens: int = 2048

class RuleExtractor:
    """Enhanced rule extractor with advanced LLM prompting and structured outputs."""
    
    def __init__(self, 
                 llm_model: Optional[BaseLanguageModel] = None,
                 config: Optional[ExtractionConfig] = None):
        """Initialize the enhanced rule extractor."""
        
        self.config = config or ExtractionConfig()
        self.prompt_system = LLMPromptSystem()
        self.implicit_extractor = ImplicitRuleExtractor()
        
        # Initialize LLM
        self.llm = llm_model or self._init_default_llm()
        
        # Initialize prompts
        self.context_prompt = self.prompt_system.create_context_analysis_prompt()
        self.extraction_prompt = self.prompt_system.create_rule_extraction_prompt()
        self.refinement_prompt = self.prompt_system.create_rule_refinement_prompt()
        self.quality_prompt = self.prompt_system.create_quality_assessment_prompt()
        
        logger.info("Enhanced Rule Extractor initialized with advanced prompting")
    
    def _init_default_llm(self) -> BaseLanguageModel:
        """Initialize default LLM model."""
        try:
            # Try to use a local model first
            from transformers import pipeline
            
            # Use a good text generation model
            pipe = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",  # Fallback model
                max_length=self.config.max_tokens,
                temperature=self.config.llm_temperature,
                do_sample=True,
                pad_token_id=50256
            )
            
            return HuggingFacePipeline(pipeline=pipe)
            
        except Exception as e:
            logger.warning(f"Failed to initialize HuggingFace model: {e}")
            logger.info("Using fallback text processing")
            return None
    
    def extract_rules_enhanced(self, 
                             document_text: str,
                             rag_context: Optional[Dict[str, Any]] = None,
                             existing_context: Optional[DocumentContext] = None) -> RuleExtractionResult:
        """
        Enhanced rule extraction with multi-pass processing and structured outputs.
        
        Args:
            document_text: Text to extract rules from
            rag_context: Optional RAG context for enhanced extraction
            existing_context: Pre-analyzed document context
            
        Returns:
            Structured rule extraction result
        """
        
        try:
            # Step 1: Analyze document context (if not provided)
            if existing_context is None and self.config.enable_context_analysis:
                document_context = self._analyze_document_context(document_text)
            else:
                document_context = existing_context or self._create_default_context()
            
            # Step 2: Extract rules using multiple methods
            extraction_results = self._multi_method_extraction(
                document_text, document_context, rag_context
            )
            
            # Step 3: Refine rules if enabled
            if self.config.enable_rule_refinement and extraction_results:
                extraction_results = self._refine_extracted_rules(
                    extraction_results, document_context, rag_context
                )
            
            # Step 4: Apply limits and cleanup
            extraction_results = self._apply_limits_and_cleanup(extraction_results)
            
            # Step 5: Create final result
            result = RuleExtractionResult(
                rules=extraction_results,
                document_context=safe_to_dict(document_context),
                processing_metadata={
                    "extraction_timestamp": datetime.now().isoformat(),
                    "rules_extracted": len(extraction_results),
                    "average_confidence": self._calculate_average_confidence(extraction_results),
                    "methods_used": ["llm_enhanced", "implicit_extraction", "context_analysis"],
                    "config": safe_to_dict(self.config)
                }
            )
            
            logger.info(f"Enhanced extraction completed: {len(extraction_results)} rules extracted")
            return result
            
        except Exception as e:
            logger.error(f"Enhanced extraction failed: {e}")
            # Fallback to basic extraction
            return self._fallback_extraction(document_text)
    
    def _analyze_document_context(self, document_text: str) -> DocumentContext:
        """Analyze document context using LLM."""
        
        if not self.llm:
            return self._create_default_context()
        
        try:
            # Prepare prompt
            prompt = self.context_prompt.format_prompt(
                document_text=document_text[:2000],  # Limit for context analysis
                format_instructions=self.prompt_system.get_format_instructions("context")
            )
            
            # Get LLM response
            response = self.llm.invoke(prompt.to_string())
            
            # Parse response
            parsed_context = self.prompt_system.validate_and_clean_json_output(response)
            
            # Create DocumentContext object safely
            try:
                return DocumentContext(**parsed_context)
            except Exception as validation_error:
                logger.warning(f"Context validation failed: {validation_error}")
                # Try with safe fallback
                return DocumentContext(
                    industry=parsed_context.get("industry", "Unknown"),
                    domain=parsed_context.get("domain", "General"),
                    purpose=parsed_context.get("purpose", "Unspecified"),
                    key_concepts=parsed_context.get("key_concepts", []),
                    manufacturing_relevance_score=parsed_context.get("manufacturing_relevance_score", 0.5),
                    implicit_requirements=parsed_context.get("implicit_requirements", []),
                    constraint_types=parsed_context.get("constraint_types", [])
                )
            
        except Exception as e:
            logger.error(f"Context analysis failed: {e}")
            return self._create_default_context()
    
    def _multi_method_extraction(self, 
                                document_text: str,
                                context: DocumentContext,
                                rag_context: Optional[Dict[str, Any]]) -> List[ManufacturingRule]:
        """Extract rules using multiple methods and combine results."""
        
        all_rules = []
        
        # Method 1: LLM-based extraction with enhanced prompts
        llm_rules = self._llm_enhanced_extraction(document_text, context, rag_context)
        all_rules.extend(llm_rules)
        
        # Method 2: Implicit rule extraction (existing system)
        implicit_rules = self._implicit_extraction_conversion(document_text)
        all_rules.extend(implicit_rules)
        
        # Method 3: Pattern-based extraction for specific formats
        pattern_rules = self._pattern_based_extraction(document_text, context)
        all_rules.extend(pattern_rules)
        
        # Deduplicate and merge similar rules
        deduplicated_rules = self._deduplicate_rules(all_rules)
        
        return deduplicated_rules
    
    @monitor_performance("llm_enhanced_extraction")
    def _llm_enhanced_extraction(self, 
                               document_text: str,
                               context: DocumentContext,
                               rag_context: Optional[Dict[str, Any]]) -> List[ManufacturingRule]:
        """Extract rules using enhanced LLM prompts with text chunking."""
        
        if not self.llm:
            return []
        
        try:
            # Format RAG context
            rag_context_str = self.prompt_system.format_rag_context(rag_context or {})
            
            # Chunk the document if it's too long
            chunks = chunk_text(document_text, max_tokens=900, overlap=100)
            all_rules = []
            
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                
                # Prepare extraction prompt for this chunk
                prompt = self.extraction_prompt.format_prompt(
                    document_text=chunk,
                    industry=context.industry,
                    domain=context.domain,
                    purpose=context.purpose,
                    key_concepts=", ".join(context.key_concepts),
                    rag_context=rag_context_str,
                    manufacturing_categories=", ".join(self.prompt_system.manufacturing_categories),
                    features_reference=json.dumps(self.prompt_system.features_dict, indent=2),
                    format_instructions=self.prompt_system.get_format_instructions("rule")
                )
                
                # Get LLM response
                response = self.llm.invoke(prompt.to_string())
                
                # Parse response
                parsed_rules = self.prompt_system.validate_and_clean_json_output(response)
                
                # Convert to ManufacturingRule objects for this chunk
                chunk_rules = []
                if isinstance(parsed_rules, list):
                    rules_data = parsed_rules
                elif isinstance(parsed_rules, dict):
                    # Handle case where response is a dict with rules key
                    rules_data = parsed_rules.get('rules', [parsed_rules])
                else:
                    continue
                
                # Process each rule in the chunk
                for rule_data in rules_data:
                    try:
                        # Skip if rule_data is an error structure or invalid
                        if not isinstance(rule_data, dict):
                            continue
                        
                        # Skip if it's an error response structure
                        if 'error' in rule_data or 'rules' in rule_data:
                            continue
                        
                        # Check for minimum required fields
                        required_fields = ['rule_category', 'name', 'feature1', 'object1', 'exp_name', 'operator', 'recom']
                        if not all(field in rule_data for field in required_fields):
                            logger.warning(f"Rule data missing required fields: {rule_data}")
                            continue
                        
                        # Apply text limits
                        rule_data = self.prompt_system.apply_text_limits(rule_data)
                        
                        # Ensure required fields have defaults if missing
                        rule_data.setdefault('confidence', 0.5)
                        rule_data.setdefault('manufacturing_relevance', 0.5)
                        rule_data.setdefault('extracted_entities', [])
                        rule_data.setdefault('rationale', "Rule extracted from document text")
                        
                        # Create rule
                        rule = ManufacturingRule(**rule_data)
                        chunk_rules.append(rule)
                    except Exception as e:
                        logger.warning(f"Failed to parse rule from chunk {i+1}: {e}")
                        continue
                
                all_rules.extend(chunk_rules)
            
            # Deduplicate rules across chunks
            return self._deduplicate_rules(all_rules)
            
        except Exception as e:
            logger.error(f"LLM enhanced extraction failed: {e}")
            return []
    
    @monitor_performance("llm_enhanced_robust_extraction")
    @retry_with_exponential_backoff
    def _extract_with_llm_enhanced_robust(self,
                                document_text: str, 
                                context: DocumentContext,
                                rag_context: Optional[Dict[str, Any]]) -> List[ManufacturingRule]:
        """
        Production-grade LLM extraction with retry logic and comprehensive error handling.
        """
        
        try:
            # Chunk document with adaptive strategy
            chunks = self._chunk_document_adaptive(document_text, context)
            
            all_rules = []
            failed_chunks = 0
            
            for i, chunk in enumerate(chunks):
                try:
                    # Use production prompt template
                    prompt = self.prompt_system.create_production_rule_extraction_prompt().format_prompt(
                        document_text=chunk,
                        industry=context.industry,
                        domain=context.domain,
                        manufacturing_categories=", ".join(self.prompt_system.manufacturing_categories)
                    )
                    
                    # Get LLM response with retry
                    response = self._llm_call_with_retry(prompt.to_string())
                    
                    # Robust JSON parsing with self-healing
                    parsed_rule = self._parse_rule_with_healing(response, chunk)
                    
                    if parsed_rule:
                        all_rules.append(parsed_rule)
                    
                except Exception as e:
                    failed_chunks += 1
                    logger.warning(f"Chunk {i+1} processing failed: {e}")
                    
                    # Circuit breaker: stop if too many failures
                    if failed_chunks > len(chunks) * 0.5:  # 50% failure threshold
                        logger.error("Circuit breaker: Too many chunk failures, stopping extraction")
                        break
                        
                    continue
            
            # Log extraction statistics
            success_rate = (len(chunks) - failed_chunks) / len(chunks) if chunks else 0
            logger.info(f"LLM extraction completed: {len(all_rules)} rules from {len(chunks)} chunks (success rate: {success_rate:.2f})")
            
            return self._deduplicate_rules(all_rules)
            
        except Exception as e:
            logger.error(f"LLM enhanced extraction failed: {e}")
            return []
    
    def _chunk_document_adaptive(self, text: str, context: DocumentContext) -> List[str]:
        """
        Adaptive chunking based on document type and industry standards.
        """
        # Document type-specific chunking strategies (following NVIDIA research)
        chunk_configs = {
            "manual": {"size": 512, "overlap": 100},       # Instruction manuals
            "specification": {"size": 512, "overlap": 100}, # Technical specs
            "policy": {"size": 800, "overlap": 150},        # Policy documents
            "qa": {"size": 256, "overlap": 50}              # Q&A content
        }
        
        # Determine document type from context
        doc_type = "specification"  # Default
        if "manual" in context.purpose.lower():
            doc_type = "manual"
        elif "policy" in context.purpose.lower():
            doc_type = "policy"
        elif "question" in context.purpose.lower() or "qa" in context.purpose.lower():
            doc_type = "qa"
        
        config = chunk_configs.get(doc_type, chunk_configs["specification"])
        
        return chunk_text(text, max_tokens=config["size"], overlap=config["overlap"])
    
    def _llm_call_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """LLM call with exponential backoff retry."""
        
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(prompt)
                if response and len(response.strip()) > 0:
                    return response
                else:
                    raise ValueError("Empty response from LLM")
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"LLM call failed after {max_retries} attempts: {e}")
                    raise e
                
                # Exponential backoff
                delay = 2 ** attempt + random.uniform(0, 1)
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                time.sleep(delay)
        
        raise Exception("LLM call failed after all retries")
    
    def _parse_rule_with_healing(self, response: str, original_text: str) -> Optional[ManufacturingRule]:
        """
        Parse LLM response with JSON self-healing and validation.
        """
        
        try:
            # First attempt: direct JSON extraction
            json_data = extract_json_from_text(response)
            
            if json_data is None:
                # Second attempt: use LLM to fix JSON
                logger.warning("JSON extraction failed, attempting self-healing")
                json_data = call_llm_to_fix_json(response, "Invalid JSON format", self.llm)
            
            if json_data is None:
                logger.error(f"Could not extract valid JSON from response: {response[:200]}...")
                return None
            
            # Validate and clean data
            cleaned_data = self._validate_and_clean_rule_data(json_data, original_text)
            
            if cleaned_data:
                return ManufacturingRule(**cleaned_data)
            
        except ValidationError as e:
            logger.warning(f"Pydantic validation failed: {e}")
            # Try to fix validation errors
            fixed_data = self._fix_validation_errors(json_data, e, original_text)
            if fixed_data:
                try:
                    return ManufacturingRule(**fixed_data)
                except ValidationError as e2:
                    logger.error(f"Could not fix validation errors: {e2}")
                    
        except Exception as e:
            logger.error(f"Rule parsing failed: {e}")
        
        return None
    
    def _validate_and_clean_rule_data(self, data: Dict[str, Any], original_text: str) -> Optional[Dict[str, Any]]:
        """
        Validate and clean rule data with intelligent defaults.
        """
        
        if not isinstance(data, dict):
            return None
        
        # Required fields with intelligent defaults
        required_fields = {
            'rule_category': 'General',
            'name': 'Extracted Rule',
            'feature1': 'parameter',
            'object1': 'component',
            'exp_name': 'parameter.value',
            'operator': '>=',
            'recom': 1.0
        }
        
        # Clean and validate each field
        cleaned = {}
        
        for field, default in required_fields.items():
            if field in data and data[field] is not None:
                cleaned[field] = data[field]
            else:
                cleaned[field] = default
                logger.debug(f"Using default value for {field}: {default}")
        
        # Optional fields with defaults
        cleaned.setdefault('feature2', '')
        cleaned.setdefault('object2', '')
        cleaned.setdefault('constraint', {})
        cleaned.setdefault('confidence', 0.6)
        cleaned.setdefault('manufacturing_relevance', 0.7)
        cleaned.setdefault('extracted_entities', [])
        cleaned.setdefault('rationale', f"Rule extracted from: {original_text[:100]}...")
        
        # Apply text limits
        cleaned['name'] = str(cleaned['name'])[:100]
        cleaned['rationale'] = str(cleaned['rationale'])[:200]
        
        # Ensure numeric fields are numeric
        try:
            cleaned['recom'] = float(cleaned['recom'])
            cleaned['confidence'] = max(0.0, min(1.0, float(cleaned['confidence'])))
            cleaned['manufacturing_relevance'] = max(0.0, min(1.0, float(cleaned['manufacturing_relevance'])))
        except (ValueError, TypeError):
            cleaned['recom'] = 1.0
            cleaned['confidence'] = 0.5
            cleaned['manufacturing_relevance'] = 0.5
        
        return cleaned
    
    def _fix_validation_errors(self, data: Dict[str, Any], error: ValidationError, original_text: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to fix Pydantic validation errors automatically.
        """
        
        try:
            # Get error details
            error_dict = {}
            for err in error.errors():
                field = err['loc'][0] if err['loc'] else 'unknown'
                error_dict[field] = err['msg']
            
            # Fix common errors
            fixed_data = data.copy()
            
            # Handle missing required fields
            for field, msg in error_dict.items():
                if 'Field required' in msg:
                    defaults = {
                        'rule_category': 'General',
                        'name': 'Manufacturing Rule',
                        'feature1': 'dimension',
                        'object1': 'component',
                        'exp_name': 'parameter.value',
                        'operator': '>=',
                        'recom': 1.0,
                        'confidence': 0.5,
                        'manufacturing_relevance': 0.5,
                        'rationale': f"Extracted from: {original_text[:50]}..."
                    }
                    if field in defaults:
                        fixed_data[field] = defaults[field]
            
            return fixed_data
            
        except Exception as e:
            logger.error(f"Error fixing validation errors: {e}")
            return None
    
    def _implicit_extraction_conversion(self, document_text: str) -> List[ManufacturingRule]:
        """Convert implicit rule extraction results to ManufacturingRule format."""
        
        try:
            # Use existing implicit extractor
            implicit_rules = self.implicit_extractor.extract_implicit_rules(document_text)
            
            converted_rules = []
            for implicit_rule in implicit_rules:
                if isinstance(implicit_rule, dict):
                    # Convert dict format to ManufacturingRule
                    manufacturing_rule = ManufacturingRule(
                        rule_category=implicit_rule.get('rule_type', 'General'),
                        name=implicit_rule.get('text', '')[:100],
                        feature1=implicit_rule.get('semantic_features', [''])[0] if implicit_rule.get('semantic_features') else '',
                        feature2=implicit_rule.get('semantic_features', ['', ''])[1] if len(implicit_rule.get('semantic_features', [])) > 1 else '',
                        object1=implicit_rule.get('context_indicators', [''])[0] if implicit_rule.get('context_indicators') else '',
                        object2='',
                        exp_name=f"{implicit_rule.get('constraint_type', 'general')}_constraint",
                        operator='>=',
                        recom=implicit_rule.get('constraint_value', 'as_specified'),
                        confidence=implicit_rule.get('confidence_score', 0.5),
                        manufacturing_relevance=implicit_rule.get('manufacturing_relevance', 0.5),
                        extracted_entities=[entity.get('text', '') for entity in implicit_rule.get('extracted_entities', [])],
                        rationale=f"Implicit extraction: {implicit_rule.get('constraint_type', 'general')} constraint"
                    )
                    converted_rules.append(manufacturing_rule)
            
            logger.info(f"Implicit extraction conversion: {len(converted_rules)} rules converted")
            return converted_rules
            
        except Exception as e:
            logger.error(f"Implicit extraction conversion failed: {e}")
            return []
    
    def _pattern_based_extraction(self, 
                                document_text: str,
                                context: DocumentContext) -> List[ManufacturingRule]:
        """Extract rules using specific patterns for known formats."""
        
        rules = []
        
        # Pattern 1: Numerical specifications (e.g., "minimum 5mm", "thickness >= 2.5mm")
        numerical_patterns = [
            r'(?:minimum|min|at least)\s+(\d+(?:\.\d+)?)\s*([a-zA-Z]+)',
            r'(?:maximum|max|no more than)\s+(\d+(?:\.\d+)?)\s*([a-zA-Z]+)',
            r'(\w+)\s*(?:>=|≥)\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]*)',
            r'(\w+)\s*(?:<=|≤)\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]*)'
        ]
        
        for pattern in numerical_patterns:
            matches = re.finditer(pattern, document_text, re.IGNORECASE)
            for match in matches:
                try:
                    if len(match.groups()) >= 2:
                        value = float(match.group(1) if match.group(1).replace('.', '').isdigit() else match.group(2))
                        feature = match.group(1) if not match.group(1).replace('.', '').isdigit() else match.group(3) or 'dimension'
                        unit = match.group(-1) if len(match.groups()) > 2 else ''
                        
                        rule = ManufacturingRule(
                            rule_category=context.domain or 'General',
                            name=f"{feature} specification",
                            feature1=feature,
                            feature2='',
                            object1='component',
                            object2='',
                            exp_name=f"{feature}.value",
                            operator='>=' if 'minimum' in match.group(0).lower() else '<=',
                            recom=value,
                            confidence=0.8,
                            manufacturing_relevance=0.7,
                            extracted_entities=[feature, str(value), unit],
                            rationale=f"Pattern-based extraction from: {match.group(0)}"
                        )
                        rules.append(rule)
                except Exception as e:
                    logger.warning(f"Pattern extraction error: {e}")
                    continue
        
        # Pattern 2: Quality requirements (e.g., "shall be", "must ensure")
        quality_patterns = [
            r'(?:shall|must|should|required to)\s+([^.]+)',
            r'(?:ensure|maintain|provide|achieve)\s+([^.]+)'
        ]
        
        for pattern in quality_patterns:
            matches = re.finditer(pattern, document_text, re.IGNORECASE)
            for match in matches:
                requirement = match.group(1).strip()[:150]
                
                rule = ManufacturingRule(
                    rule_category='Quality Control',
                    name=f"Quality requirement: {requirement[:50]}",
                    feature1='quality',
                    feature2='',
                    object1='product',
                    object2='',
                    exp_name='quality.requirement',
                    operator='==',
                    recom='compliant',
                    confidence=0.6,
                    manufacturing_relevance=0.8,
                    extracted_entities=[requirement],
                    rationale=f"Quality pattern extraction from: {match.group(0)}"
                )
                rules.append(rule)
        
        logger.info(f"Pattern-based extraction: {len(rules)} rules extracted")
        return rules
    
    def _deduplicate_rules(self, rules: List[ManufacturingRule]) -> List[ManufacturingRule]:
        """Remove duplicate and very similar rules."""
        
        if not rules:
            return []
        
        deduplicated = []
        seen_rules = set()
        
        for rule in rules:
            # Create a signature for the rule
            signature = f"{rule.rule_category}_{rule.feature1}_{rule.operator}_{rule.recom}"
            signature = signature.lower().replace(' ', '_')
            
            if signature not in seen_rules:
                seen_rules.add(signature)
                deduplicated.append(rule)
            else:
                # If duplicate, keep the one with higher confidence
                for i, existing_rule in enumerate(deduplicated):
                    existing_sig = f"{existing_rule.rule_category}_{existing_rule.feature1}_{existing_rule.operator}_{existing_rule.recom}"
                    existing_sig = existing_sig.lower().replace(' ', '_')
                    
                    if existing_sig == signature and rule.confidence > existing_rule.confidence:
                        deduplicated[i] = rule
                        break
        
        logger.info(f"Deduplication: {len(rules)} -> {len(deduplicated)} rules")
        return deduplicated
    
    def _refine_extracted_rules(self, 
                              rules: List[ManufacturingRule],
                              context: DocumentContext,
                              rag_context: Optional[Dict[str, Any]]) -> List[ManufacturingRule]:
        """Refine extracted rules using LLM."""
        
        if not self.llm or not rules:
            return rules
        
        try:
            # Prepare rules for refinement
            rules_data = [rule.dict() for rule in rules]
            
            # Format similar rules from RAG
            similar_rules = ""
            if rag_context and 'similar_rules' in rag_context:
                similar_rules = json.dumps(rag_context['similar_rules'][:3], indent=2)
            
            # Prepare refinement prompt
            prompt = self.refinement_prompt.format_prompt(
                extracted_rules=json.dumps(rules_data, indent=2),
                document_context=json.dumps(safe_to_dict(context), indent=2),
                similar_rules=similar_rules,
                format_instructions=self.prompt_system.get_format_instructions("rule")
            )
            
            # Get LLM response
            response = self.llm.invoke(prompt.to_string())
            
            # Parse refined rules
            parsed_rules = self.prompt_system.validate_and_clean_json_output(response)
            
            refined_rules = []
            if isinstance(parsed_rules, dict) and 'rules' in parsed_rules:
                rules_data = parsed_rules['rules']
            elif isinstance(parsed_rules, list):
                rules_data = parsed_rules
            else:
                return rules  # Return original if parsing fails
            
            for rule_data in rules_data:
                try:
                    rule_data = self.prompt_system.apply_text_limits(rule_data)
                    refined_rule = ManufacturingRule(**rule_data)
                    refined_rules.append(refined_rule)
                except Exception as e:
                    logger.warning(f"Failed to parse refined rule: {e}")
                    continue
            
            logger.info(f"Rule refinement: {len(refined_rules)} rules refined")
            return refined_rules if refined_rules else rules
            
        except Exception as e:
            logger.error(f"Rule refinement failed: {e}")
            return rules
    
    def _apply_limits_and_cleanup(self, rules: List[ManufacturingRule]) -> List[ManufacturingRule]:
        """Apply limits and cleanup rules."""
        
        cleaned_rules = []
        
        for rule in rules:
            # Apply confidence threshold
            if rule.confidence < self.config.min_confidence_threshold:
                continue
            
            # Ensure text limits are applied
            if len(rule.name) > 100:
                rule.name = rule.name[:97] + "..."
            
            if len(rule.rationale) > 300:
                rule.rationale = rule.rationale[:297] + "..."
            
            # Validate required fields
            if not rule.rule_category or not rule.name:
                continue
            
            cleaned_rules.append(rule)
        
        # Limit total number of rules
        if len(cleaned_rules) > self.config.max_rules_per_chunk:
            # Sort by confidence and take top rules
            cleaned_rules.sort(key=lambda x: x.confidence, reverse=True)
            cleaned_rules = cleaned_rules[:self.config.max_rules_per_chunk]
        
        return cleaned_rules
    
    def _calculate_average_confidence(self, rules: List[ManufacturingRule]) -> float:
        """Calculate average confidence score."""
        if not rules:
            return 0.0
        return sum(rule.confidence for rule in rules) / len(rules)
    
    def _create_default_context(self) -> DocumentContext:
        """Create default document context."""
        return DocumentContext(
            industry="General",
            domain="Manufacturing",
            purpose="Specification",
            key_concepts=["requirements", "specifications"],
            manufacturing_relevance_score=0.5,
            implicit_requirements=[],
            constraint_types=["general"]
        )
    
    def _fallback_extraction(self, document_text: str) -> RuleExtractionResult:
        """Fallback extraction when enhanced methods fail."""
        logger.warning("Using fallback extraction method")
        
        # Basic pattern-based extraction
        context = self._create_default_context()
        rules = self._pattern_based_extraction(document_text, context)
        
        return RuleExtractionResult(
            rules=rules,
            document_context=safe_to_dict(context),
            processing_metadata={
                "extraction_timestamp": datetime.now().isoformat(),
                "rules_extracted": len(rules),
                "average_confidence": self._calculate_average_confidence(rules),
                "methods_used": ["fallback_pattern_extraction"],
                "note": "Enhanced extraction failed, using fallback method"
            }
        )