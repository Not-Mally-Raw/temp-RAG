"""
Enhanced Rule Engine with LangChain Structured Outputs
Production-ready manufacturing rule extraction with 98%+ accuracy target
"""

import asyncio
import json
import logging
import re
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# Core LangChain imports
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_groq import ChatGroq

# Pydantic for structured outputs
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings

# Analytics and monitoring
import structlog
import tiktoken
from textstat import flesch_reading_ease, automated_readability_index
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

# Local imports
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from .enhanced_vector_utils import EnhancedVectorManager
from .ingest import DocumentIngester
from utils import logger

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Structured logger
enhanced_logger = structlog.get_logger()

class ManufacturingCategory(str, Enum):
    """Manufacturing categories enum for validation."""
    SHEET_METAL = "Sheet Metal"
    INJECTION_MOLDING = "Injection Molding"
    MACHINING = "Machining"
    ASSEMBLY = "Assembly"
    WELDING = "Welding"
    CASTING = "Casting"
    QUALITY_CONTROL = "Quality Control"
    ELECTRONICS = "Electronics"
    MATERIAL_SPECIFICATION = "Material Specification"
    SAFETY_REQUIREMENT = "Safety Requirement"
    DESIGN_GUIDELINE = "Design Guideline"
    GENERAL = "General"

class RuleType(str, Enum):
    """Rule type classification."""
    DIMENSIONAL = "dimensional"
    MATERIAL = "material"
    PROCESS = "process"
    QUALITY = "quality"
    SAFETY = "safety"
    ASSEMBLY = "assembly"
    TOLERANCE = "tolerance"
    GENERAL = "general"

class CorrectnessTag(str, Enum):
    """Correctness tagging for rule validation."""
    CORRECT = "correct"
    OUT_OF_SCOPE = "out_of_scope"
    LOW_CONFIDENCE = "low_confidence"
    NEEDS_REVIEW = "needs_review"
    INCOMPLETE = "incomplete"

class ConfidenceLevel(str, Enum):
    """Confidence level categories."""
    HIGH = "high"      # 0.8-1.0
    MEDIUM = "medium"  # 0.6-0.8
    LOW = "low"        # 0.4-0.6
    VERY_LOW = "very_low"  # 0.0-0.4

class ManufacturingRule(BaseModel):
    """Enhanced Pydantic model for manufacturing rules with validation."""
    
    # Core fields
    rule_text: str = Field(..., description="Original rule text", max_length=500)
    rule_category: ManufacturingCategory = Field(..., description="Manufacturing category")
    rule_type: RuleType = Field(..., description="Type of rule")
    
    # Extracted entities
    primary_feature: str = Field(..., description="Primary manufacturing feature")
    secondary_feature: Optional[str] = Field(default="", description="Secondary feature if applicable")
    primary_object: str = Field(..., description="Primary object/component")
    secondary_object: Optional[str] = Field(default="", description="Secondary object if applicable")
    
    # Constraints and specifications
    operator: str = Field(..., description="Comparison operator (>=, <=, ==, between, etc.)")
    value: Optional[Union[float, List[float], str]] = Field(default=None, description="Numeric value or range")
    unit: Optional[str] = Field(default="", description="Unit of measurement")
    tolerance: Optional[Union[float, List[float]]] = Field(default=None, description="Tolerance value (single number or range)")
    
    # Metadata
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence")
    confidence_level: ConfidenceLevel = Field(..., description="Confidence category")
    correctness_tag: CorrectnessTag = Field(..., description="Correctness validation tag")
    manufacturing_relevance: float = Field(..., ge=0.0, le=1.0, description="Manufacturing relevance score")
    
    # Context
    source_document: str = Field(default="", description="Source document filename")
    document_section: Optional[str] = Field(default="", description="Document section/page")
    extraction_method: str = Field(..., description="Method used for extraction")
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Quality indicators
    readability_score: Optional[float] = Field(default=None, description="Text readability score")
    complexity_score: Optional[float] = Field(default=None, description="Rule complexity score")
    
    # Advanced features
    implicit_requirements: List[str] = Field(default_factory=list, description="Implicit requirements")
    related_standards: List[str] = Field(default_factory=list, description="Related industry standards")
    cross_references: List[str] = Field(default_factory=list, description="Cross-referenced rules")
    
    @validator('confidence_level', pre=True, always=True)
    def set_confidence_level(cls, v, values):
        if 'confidence_score' in values:
            score = values['confidence_score']
            if score >= 0.8:
                return ConfidenceLevel.HIGH
            elif score >= 0.6:
                return ConfidenceLevel.MEDIUM
            elif score >= 0.4:
                return ConfidenceLevel.LOW
            else:
                return ConfidenceLevel.VERY_LOW
        return v
    
    @validator('correctness_tag', pre=True, always=True)
    def set_correctness_tag(cls, v, values):
        """Auto-assign correctness tag based on rule characteristics."""
        if 'confidence_score' in values and 'manufacturing_relevance' in values:
            confidence = values['confidence_score']
            relevance = values['manufacturing_relevance']
            
            # Out of scope if very low relevance
            if relevance < 0.2:
                return CorrectnessTag.OUT_OF_SCOPE
            
            # Low confidence needs review
            if confidence < 0.4:
                return CorrectnessTag.LOW_CONFIDENCE
            
            # Incomplete if missing key fields
            rule_text = values.get('rule_text', '')
            if len(rule_text.strip()) < 20 or not any(char.isdigit() for char in rule_text):
                return CorrectnessTag.INCOMPLETE
            
            # Needs review if medium confidence
            if confidence < 0.7:
                return CorrectnessTag.NEEDS_REVIEW
            
            # High confidence and relevance = correct
            return CorrectnessTag.CORRECT
        
        return CorrectnessTag.NEEDS_REVIEW

    @validator('value', pre=True)
    def parse_value(cls, value):
        """Normalize numeric value field."""
        if value is None or value == "":
            return None
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, list):
            cleaned = []
            for item in value:
                if item in (None, ""):
                    continue
                try:
                    cleaned.append(float(item))
                except (TypeError, ValueError):
                    continue
            if not cleaned:
                return None
            if len(cleaned) == 1:
                return cleaned[0]
            return cleaned
        if isinstance(value, str):
            normalized = value.replace(',', ' ')
            numbers = re.findall(r'-?\d*\.?\d+', normalized)
            if not numbers:
                return value
            floats = [float(num) for num in numbers]
            if len(floats) == 1:
                return floats[0]
            return floats
        return value

    @validator('tolerance', pre=True)
    def parse_tolerance(cls, value):
        """Normalize tolerance values into floats or ranges."""
        if value is None or value == "":
            return None
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, list):
            cleaned = []
            for item in value:
                if item in (None, ""):
                    continue
                try:
                    cleaned.append(float(item))
                except (TypeError, ValueError):
                    continue
            if not cleaned:
                return None
            if len(cleaned) == 1:
                return cleaned[0]
            return cleaned
        if isinstance(value, str):
            normalized = value.replace(',', ' ')
            numbers = re.findall(r'-?\d*\.?\d+', normalized)
            if not numbers:
                return None
            floats = [float(num) for num in numbers]
            if len(floats) == 1:
                return floats[0]
            return floats
        return None

    @validator('extracted_at', pre=True, always=True)
    def normalize_extracted_at(cls, value):
        """Ensure ``extracted_at`` is always a valid timestamp."""

        if not value:
            return datetime.utcnow()
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return datetime.utcnow()
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    return datetime.strptime(cleaned, fmt)
                except ValueError:
                    continue
        return datetime.utcnow()

    @validator('rule_category', pre=True, always=True)
    def normalize_rule_category(cls, value):
        """Map freeform categories into the enumerated set."""

        if value is None:
            return ManufacturingCategory.GENERAL
        if isinstance(value, ManufacturingCategory):
            return value

        normalized = str(value).strip().lower()
        category_map = {
            'pharmaceutical': ManufacturingCategory.QUALITY_CONTROL,
            'pharma': ManufacturingCategory.QUALITY_CONTROL,
            'quality control': ManufacturingCategory.QUALITY_CONTROL,
            'occupational health and safety': ManufacturingCategory.SAFETY_REQUIREMENT,
            'occupational health & safety': ManufacturingCategory.SAFETY_REQUIREMENT,
            'occupational safety': ManufacturingCategory.SAFETY_REQUIREMENT,
            'safety': ManufacturingCategory.SAFETY_REQUIREMENT,
            'health and safety': ManufacturingCategory.SAFETY_REQUIREMENT,
            'material': ManufacturingCategory.MATERIAL_SPECIFICATION,
            'materials': ManufacturingCategory.MATERIAL_SPECIFICATION,
            'assembly': ManufacturingCategory.ASSEMBLY,
            'machining': ManufacturingCategory.MACHINING,
            'process': ManufacturingCategory.GENERAL,
        }
        try:
            return ManufacturingCategory[normalized.replace(" ", "_").upper()]
        except KeyError:
            pass

        for key, mapped in category_map.items():
            if key in normalized:
                return mapped

        return ManufacturingCategory.GENERAL

class RuleExtractionResult(BaseModel):
    """Complete result from enhanced rule extraction."""
    rules: List[ManufacturingRule] = Field(description="Extracted manufacturing rules")
    document_metadata: Dict[str, Any] = Field(description="Document analysis metadata")
    extraction_stats: Dict[str, Any] = Field(description="Extraction statistics")
    processing_time: float = Field(description="Processing time in seconds")
    token_usage: Dict[str, int] = Field(description="Token usage statistics")

class DocumentContext(BaseModel):
    """Enhanced document context analysis."""
    industry_sector: str = Field(default="Unknown", description="Identified industry sector")
    technical_domain: str = Field(default="General", description="Technical domain")
    document_type: str = Field(default="Unspecified", description="Document type")
    document_purpose: str = Field(default="General", description="Document purpose")
    
    # Content analysis
    key_concepts: List[str] = Field(default_factory=list, description="Key technical concepts")
    manufacturing_density: float = Field(default=0.0, description="Manufacturing content density")
    technical_complexity: float = Field(default=0.0, description="Technical complexity score")
    
    # Language metrics
    readability_index: float = Field(default=0.0, description="Document readability")
    avg_sentence_length: float = Field(default=0.0, description="Average sentence length")
    
    # Manufacturing specifics
    process_types: List[str] = Field(default_factory=list, description="Manufacturing processes identified")
    material_types: List[str] = Field(default_factory=list, description="Materials mentioned")
    quality_standards: List[str] = Field(default_factory=list, description="Quality standards referenced")

class EnhancedConfig(BaseSettings):
    """Production configuration with validation."""

    # LLM Configuration
    groq_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    groq_api_key: str = ""
    max_tokens: int = 4096
    temperature: float = 0.05

    # Processing Configuration
    chunk_size: int = 800  # Token-aware chunking
    chunk_overlap: int = 400
    max_chunks_per_document: int = 15
    
    # Quality thresholds
    min_confidence_threshold: float = 0.3
    min_manufacturing_relevance: float = 0.2
    max_rules_per_document: int = 100
    
    # RAG Configuration
    rag_top_k: int = 5
    rag_score_threshold: float = 0.7
    enable_rag_enhancement: bool = False
    api_request_delay: float = 2.0
    
    # Advanced features
    enable_semantic_chunking: bool = True
    enable_cross_validation: bool = True
    enable_rule_clustering: bool = False
    enable_quality_scoring: bool = True
    
    # Monitoring
    log_level: str = "INFO"
    enable_metrics: bool = True
    
    class Config:
        env_file = ".env"
        extra = "ignore"  # Allow extra fields in .env file

class EnhancedRuleEngine:
    """Production-ready rule extraction engine with LangChain integration."""
    
    def __init__(self, config: Optional[EnhancedConfig] = None,
                 vector_manager: Optional[Any] = None):
        """Initialize enhanced rule engine."""
        
        self.config = config or EnhancedConfig()
        
        # Validate API key (no hardcoded or placeholder keys allowed)
        if not self.config.groq_api_key or self.config.groq_api_key == "":
            raise ValueError(
                "GROQ_API_KEY is required but not set. "
                "Please set the GROQ_API_KEY environment variable or create a .env file. "
                "Get your API key from https://console.groq.com/"
            )
        # No hardcoded or placeholder keys allowed in code
        if self.config.groq_api_key.lower().startswith("sk-groq") or self.config.groq_api_key == "your_actual_groq_api_key_here":
            raise ValueError("GROQ_API_KEY appears to be a placeholder or hardcoded key. Please use a secure environment variable.")
        
        self.vector_manager = vector_manager
        self.ingester = DocumentIngester()
        
        # Initialize tokenizer for token-aware processing
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Initialize LLM with structured output
        self._preferred_model = self.config.groq_model
        self._fallback_model = "llama-3.1-8b-instant"
        self._last_llm_error: Optional[Exception] = None

        if not self._try_configure_llm(self._preferred_model):
            if (
                self._preferred_model != self._fallback_model
                and self._try_configure_llm(self._fallback_model)
            ):
                enhanced_logger.warning(
                    "groq_model_fallback_startup",
                    requested_model=self._preferred_model,
                    fallback_model=self._fallback_model,
                )
            else:
                raise ValueError(
                    f"Failed to initialize Groq LLM: {self._last_llm_error}. Please check your GROQ_API_KEY."
                )
        
        # Setup structured output parsers
        self.rule_parser = PydanticOutputParser(pydantic_object=ManufacturingRule)
        self.context_parser = PydanticOutputParser(pydantic_object=DocumentContext)
        self.result_parser = PydanticOutputParser(pydantic_object=RuleExtractionResult)
        
        # Setup LangChain chains
        self._setup_extraction_chains()
        
        # Initialize quality assessors
        self._initialize_quality_assessors()
        
        enhanced_logger.info("Enhanced rule engine initialized", 
                           model=self.config.groq_model,
                           has_vector_manager=self.vector_manager is not None)

    def _try_configure_llm(self, model_name: str) -> bool:
        try:
            self.llm = ChatGroq(
                model=model_name,
                api_key=self.config.groq_api_key,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            self.config.groq_model = model_name
            self._last_llm_error = None
            return True
        except Exception as error:  # pragma: no cover - network / access errors
            self._last_llm_error = error
            return False

    def _ensure_llm_available(self, error: Exception) -> bool:
        message = str(error).lower()
        if "model_not_found" not in message and "does not exist" not in message:
            return False
        if self.config.groq_model == self._fallback_model:
            return False
        if self._try_configure_llm(self._fallback_model):
            self._setup_extraction_chains()
            enhanced_logger.warning(
                "groq_model_fallback_runtime",
                requested_model=self._preferred_model,
                fallback_model=self._fallback_model,
            )
            return True
        enhanced_logger.error(
            "groq_model_fallback_failed",
            requested_model=self._preferred_model,
            fallback_model=self._fallback_model,
            error=str(self._last_llm_error),
        )
        return False
    
    async def extract_rules_from_text(
        self,
        text: str,
        filename: str = "ad_hoc.txt",
    ) -> Dict[str, Any]:
        """Public async helper that mirrors the legacy interface."""

        result = await self.extract_rules_parallel(text, filename)
        return self._result_to_dict(result, fallback_filename=filename)

    def _setup_extraction_chains(self):
        """Setup LangChain chains for structured extraction."""
        
        # Document context analysis chain
        context_system_prompt = """You are an expert document analyzer specializing in manufacturing and engineering documents. 
        Analyze the document context and extract comprehensive metadata that will help with accurate rule extraction.
        
        Focus on:
        1. Industry sector identification (automotive, aerospace, electronics, etc.)
        2. Technical domain classification
        3. Document type and purpose
        4. Manufacturing process types mentioned
        5. Material specifications
        6. Quality standards referenced
        7. Technical complexity assessment
        
        Provide detailed analysis in the specified JSON format."""
        
        context_human_prompt = """Analyze this document text and provide comprehensive context analysis:

        Document Text (first 2000 chars):
        {document_text}

        {format_instructions}
        
        Focus on manufacturing relevance and technical content density."""
        
        self.context_chain = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(context_system_prompt),
            HumanMessagePromptTemplate.from_template(context_human_prompt)
        ]) | self.llm | self.context_parser
        
        # Enhanced rule extraction chain with structured output
        rule_system_prompt = """You are a world-class manufacturing rule extraction expert with 20+ years of experience.
        Your task is to extract ONE specific, actionable manufacturing rule from the provided text.
        
        CRITICAL SUCCESS CRITERIA:
        1. Extract ONLY rules that are specific, measurable, and actionable
        2. Focus on dimensional tolerances, material specifications, process parameters
        3. Identify implicit requirements even if not explicitly stated
        4. Assign accurate confidence scores based on specificity and clarity
        5. Properly categorize by manufacturing domain
    6. Output numeric fields in normalized format.
        
    NUMERIC FIELD FORMATTING RULES:
    - Provide plain numbers without units for "value" and "tolerance".
    - If a tolerance or value is a range, return a two-element JSON array of numbers (e.g., [0.003, 0.005]).
    - Keep units exclusively in the "unit" field.

        EXTRACTION PRIORITIES (in order):
        1. Dimensional specifications with numeric values
        2. Material requirements and properties
        3. Process parameters and conditions
        4. Quality control criteria
        5. Safety requirements
        6. Assembly guidelines
        
        {format_instructions}
        
        Return ONLY the JSON object with all required fields populated."""
        
        rule_human_prompt = """Extract ONE manufacturing rule from this text chunk.

        Text Chunk:
        {text_chunk}
        
        Document Context:
        - Industry: {industry_sector}
        - Domain: {technical_domain}
        - Document Type: {document_type}
        
        RAG Context (similar rules):
        {rag_context}
        
        Manufacturing Keywords Detected: {manufacturing_keywords}
        
        Extract the most specific and actionable rule with highest confidence."""
        
        self.rule_extraction_chain = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(rule_system_prompt),
            HumanMessagePromptTemplate.from_template(rule_human_prompt)
        ]) | self.llm | self.rule_parser
        
        # Rule enhancement chain for quality improvement
        enhancement_system_prompt = """You are a manufacturing standards expert. Your task is to enhance and validate extracted rules.
        
        Enhancement objectives:
        1. Improve specificity and clarity
        2. Add missing technical details
        3. Standardize terminology
        4. Validate numeric values and units
        5. Add relevant constraints
        6. Improve manufacturing relevance
        
        Maintain accuracy to the original source while enhancing quality."""
        
        enhancement_human_prompt = """Enhance this extracted manufacturing rule:

        Original Rule:
        {original_rule}
        
        Document Context:
        {document_context}
        
        Similar Rules from Database:
        {similar_rules}
        
        {format_instructions}
        
        Provide enhanced rule with improved quality and specificity."""
        
        self.enhancement_chain = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(enhancement_system_prompt),
            HumanMessagePromptTemplate.from_template(enhancement_human_prompt)
        ]) | self.llm | self.rule_parser
    
    def _initialize_quality_assessors(self):
        """Initialize quality assessment tools."""
        self.manufacturing_keywords = {
            'high_priority': [
                'minimum', 'maximum', 'thickness', 'diameter', 'radius', 'tolerance',
                'clearance', 'specification', 'requirement', 'standard', 'shall',
                'must', 'required', 'not less than', 'not exceed', 'between'
            ],
            'manufacturing': [
                'machining', 'molding', 'assembly', 'welding', 'casting',
                'sheet metal', 'injection', 'drilling', 'milling', 'turning'
            ],
            'materials': [
                'steel', 'aluminum', 'plastic', 'carbon', 'titanium', 'copper',
                'stainless', 'alloy', 'composite', 'polymer'
            ],
            'dimensions': [
                'length', 'width', 'height', 'depth', 'angle', 'surface finish',
                'roughness', 'flatness', 'roundness', 'concentricity'
            ]
        }
        
        # Initialize TF-IDF for content analysis
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3)
        )
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.tokenizer.encode(text))
    
    def semantic_chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Advanced semantic chunking with manufacturing context awareness."""
        
        # Split into sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        manufacturing_score = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # Calculate manufacturing relevance for this sentence
            sentence_lower = sentence.lower()
            mfg_score = sum(1 for kw in self.manufacturing_keywords['high_priority'] 
                           if kw in sentence_lower)
            
            # Check if adding this sentence would exceed token limit
            if current_tokens + sentence_tokens > self.config.chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    'text': current_chunk.strip(),
                    'token_count': current_tokens,
                    'manufacturing_score': manufacturing_score,
                    'sentence_count': len(re.split(r'[.!?]', current_chunk))
                })
                
                # Start new chunk with overlap
                if self.config.chunk_overlap > 0:
                    overlap_sentences = sentences[-2:] if len(sentences) > 2 else sentences
                    current_chunk = " ".join(overlap_sentences) + " " + sentence
                    current_tokens = self.count_tokens(current_chunk)
                else:
                    current_chunk = sentence
                    current_tokens = sentence_tokens
                
                manufacturing_score = mfg_score
            else:
                current_chunk += " " + sentence
                current_tokens += sentence_tokens
                manufacturing_score += mfg_score
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'token_count': current_tokens,
                'manufacturing_score': manufacturing_score,
                'sentence_count': len(re.split(r'[.!?]', current_chunk))
            })
        
        # Sort by manufacturing relevance and return top chunks
        chunks.sort(key=lambda x: x['manufacturing_score'], reverse=True)
        return chunks[:self.config.max_chunks_per_document]
    
    def analyze_document_context(self, text: str, *, _retry: bool = True) -> DocumentContext:
        """Analyze document context using LangChain structured output."""
        
        try:
            # Prepare text for analysis (first 2000 characters)
            analysis_text = text[:2000] if len(text) > 2000 else text
            
            # Run context analysis chain
            context = self.context_chain.invoke({
                'document_text': analysis_text,
                'format_instructions': self.context_parser.get_format_instructions()
            })

            if self.config.api_request_delay:
                time.sleep(self.config.api_request_delay)
            
            # Calculate additional metrics
            context.readability_index = flesch_reading_ease(analysis_text)
            context.avg_sentence_length = len(analysis_text.split()) / len(re.split(r'[.!?]', analysis_text))
            
            # Calculate manufacturing density
            total_words = len(analysis_text.split())
            mfg_words = sum(1 for word in analysis_text.lower().split() 
                           if any(kw in word for kw_list in self.manufacturing_keywords.values() 
                                 for kw in kw_list))
            context.manufacturing_density = mfg_words / total_words if total_words > 0 else 0.0
            
            return context
            
        except Exception as e:
            if _retry and self._ensure_llm_available(e):
                return self.analyze_document_context(text, _retry=False)
            enhanced_logger.error("Context analysis failed", error=str(e))
            return DocumentContext()
    
    def extract_single_rule(self, text_chunk: str, document_context: DocumentContext,
                           rag_context: List[Dict] = None, *, _retry: bool = True) -> Optional[ManufacturingRule]:
        """Extract single rule using LangChain structured output with enhancement limits."""
        
        # Check if chunk is high-value before processing
        if not self._is_high_value_chunk(text_chunk):
            enhanced_logger.debug("Skipping low-value chunk", chunk_preview=text_chunk[:50])
            return None
        
        max_retries = 2  # Limit LLM enhancement calls
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                # Identify manufacturing keywords in chunk
                chunk_lower = text_chunk.lower()
                detected_keywords = []
                for category, keywords in self.manufacturing_keywords.items():
                    found = [kw for kw in keywords if kw in chunk_lower]
                    detected_keywords.extend(found)
                
                # Format RAG context
                rag_context_str = ""
                if rag_context:
                    rag_context_str = "\n".join([
                        f"- {ctx.get('text', '')[:100]}..." 
                        for ctx in rag_context[:3]
                    ])
                
                # Run rule extraction chain
                rule = self.rule_extraction_chain.invoke({
                    'text_chunk': text_chunk,
                    'industry_sector': document_context.industry_sector,
                    'technical_domain': document_context.technical_domain,
                    'document_type': document_context.document_type,
                    'rag_context': rag_context_str,
                    'manufacturing_keywords': ", ".join(detected_keywords),
                    'format_instructions': self.rule_parser.get_format_instructions()
                })

                if self.config.api_request_delay:
                    time.sleep(self.config.api_request_delay)
                
                # Validate extracted rule quality
                if self._is_rule_quality_acceptable(rule):
                    # Add computed fields
                    rule.readability_score = flesch_reading_ease(text_chunk)
                    rule.complexity_score = len(detected_keywords) / len(text_chunk.split()) if text_chunk.split() else 0
                    rule.extraction_method = "langchain_structured"
                    return rule
                
                # If rule quality is poor and we have retries left, try enhancement
                if retry_count < max_retries:
                    retry_count += 1
                    enhanced_logger.debug("Rule quality poor, attempting enhancement", 
                                        retry=retry_count, 
                                        confidence=rule.confidence_score)
                    
                    # Try to enhance the rule
                    rule = self._enhance_rule_quality(rule, text_chunk, document_context)
                    continue
                
                # No more retries, return the rule as-is
                rule.readability_score = flesch_reading_ease(text_chunk)
                rule.complexity_score = len(detected_keywords) / len(text_chunk.split()) if text_chunk.split() else 0
                rule.extraction_method = "langchain_structured"
                return rule
                
            except Exception as e:
                if _retry and self._ensure_llm_available(e):
                    return self.extract_single_rule(
                        text_chunk,
                        document_context,
                        rag_context,
                        _retry=False,
                    )
                enhanced_logger.error("Rule extraction failed", 
                                    chunk_preview=text_chunk[:100],
                                    retry=retry_count,
                                    error=str(e))
                
                if retry_count < max_retries:
                    retry_count += 1
                    continue
                return None
        
        return None
    
    async def extract_rules_parallel(self, text: str, filename: str = "") -> RuleExtractionResult:
        """Extract rules using parallel processing for production speed."""
        
        start_time = datetime.utcnow()
        token_usage = {'input_tokens': 0, 'output_tokens': 0}
        
        try:
            # Step 1: Analyze document context
            document_context = self.analyze_document_context(text)
            token_usage['input_tokens'] += self.count_tokens(text[:2000])
            
            # Step 2: Semantic chunking
            chunks = self.semantic_chunk_text(text)
            enhanced_logger.info("Document chunked", 
                               chunk_count=len(chunks),
                               total_tokens=sum(c['token_count'] for c in chunks))
            
            # Step 3: Parallel rule extraction
            extraction_tasks = []
            for chunk_data in chunks:
                # Get RAG context for chunk if available
                rag_context = []
                if self.vector_manager and self.config.enable_rag_enhancement:
                    rag_context = self.vector_manager.similarity_search(
                        chunk_data['text'], 
                        top_k=self.config.rag_top_k,
                        score_threshold=self.config.rag_score_threshold
                    )
                
                # Create extraction task
                task = self.extract_single_rule(
                    chunk_data['text'], 
                    document_context, 
                    rag_context
                )
                extraction_tasks.append(task)
            
            # Execute extractions (simulated parallel for now)
            rules = []
            for task in extraction_tasks:
                if task:
                    rule = task  # In real async, this would be awaited
                    if rule and rule.confidence_score >= self.config.min_confidence_threshold:
                        rule.source_document = filename
                        rules.append(rule)
            
            # Step 4: Post-processing
            rules = self._post_process_rules(rules)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Create result
            result = RuleExtractionResult(
                rules=rules,
                document_metadata=document_context.dict(),
                extraction_stats={
                    'total_chunks': len(chunks),
                    'rules_extracted': len(rules),
                    'avg_confidence': sum(r.confidence_score for r in rules) / len(rules) if rules else 0,
                    'manufacturing_density': document_context.manufacturing_density,
                    'high_confidence_rules': len([r for r in rules if r.confidence_level == ConfidenceLevel.HIGH])
                },
                processing_time=processing_time,
                token_usage=token_usage
            )
            
            enhanced_logger.info("Rule extraction completed",
                               rules_count=len(rules),
                               processing_time=processing_time,
                               avg_confidence=result.extraction_stats['avg_confidence'])
            
            return result
            
        except Exception as e:
            enhanced_logger.error("Rule extraction failed", error=str(e))
            return RuleExtractionResult(
                rules=[],
                document_metadata={},
                extraction_stats={'error': str(e)},
                processing_time=(datetime.utcnow() - start_time).total_seconds(),
                token_usage=token_usage
            )
    
    def _post_process_rules(self, rules: List[ManufacturingRule]) -> List[ManufacturingRule]:
        """Post-process rules with deduplication and quality enhancement."""
        
        if not rules:
            return rules
        
        # Step 1: Semantic deduplication
        unique_rules = self._deduplicate_semantic(rules)
        
        # Step 2: Quality filtering
        filtered_rules = [
            rule for rule in unique_rules 
            if (rule.confidence_score >= self.config.min_confidence_threshold and
                rule.manufacturing_relevance >= self.config.min_manufacturing_relevance)
        ]
        
        # Step 3: Log escalation cases
        self._log_escalation_cases(unique_rules)
        
        # Step 4: Clustering for similar rules
        if self.config.enable_rule_clustering and len(filtered_rules) > 5:
            filtered_rules = self._cluster_similar_rules(filtered_rules)
        
        # Step 5: Sort by confidence and limit
        filtered_rules.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return filtered_rules[:self.config.max_rules_per_document]
    
    def _deduplicate_semantic(self, rules: List[ManufacturingRule]) -> List[ManufacturingRule]:
        """Advanced semantic deduplication using text similarity."""
        
        if len(rules) <= 1:
            return rules
        
        # Extract rule texts for similarity comparison
        rule_texts = [rule.rule_text for rule in rules]
        
        try:
            # Use TF-IDF for similarity calculation
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(rule_texts)
            similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
            
            # Identify duplicates (similarity > 0.8)
            unique_indices = []
            for i, rule in enumerate(rules):
                is_duplicate = False
                for j in unique_indices:
                    if similarity_matrix[i][j] > 0.8:
                        # Keep the rule with higher confidence
                        if rule.confidence_score > rules[j].confidence_score:
                            unique_indices.remove(j)
                            unique_indices.append(i)
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_indices.append(i)
            
            return [rules[i] for i in unique_indices]
            
        except Exception as e:
            enhanced_logger.warning("Semantic deduplication failed", error=str(e))
            # Fallback to simple text-based deduplication
            seen_texts = set()
            unique_rules = []
            for rule in rules:
                normalized_text = re.sub(r'\W+', '', rule.rule_text.lower())
                if normalized_text not in seen_texts:
                    seen_texts.add(normalized_text)
                    unique_rules.append(rule)
            return unique_rules
    
    def _cluster_similar_rules(self, rules: List[ManufacturingRule]) -> List[ManufacturingRule]:
        """Cluster similar rules and merge if appropriate."""
        
        try:
            rule_texts = [rule.rule_text for rule in rules]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(rule_texts)
            
            # Use KMeans for clustering
            n_clusters = min(5, len(rules) // 3)  # Adaptive cluster count
            if n_clusters < 2:
                return rules
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(tfidf_matrix.toarray())
            
            # For each cluster, keep the highest confidence rule
            clustered_rules = []
            for cluster_id in range(n_clusters):
                cluster_rules = [rules[i] for i, c in enumerate(clusters) if c == cluster_id]
                if cluster_rules:
                    best_rule = max(cluster_rules, key=lambda x: x.confidence_score)
                    clustered_rules.append(best_rule)
            
            return clustered_rules
            
        except Exception as e:
            enhanced_logger.warning("Rule clustering failed", error=str(e))
            return rules
    
    def _log_escalation_cases(self, rules: List[ManufacturingRule]):
        """Log rules that require escalation for review."""
        
        escalation_rules = [
            rule for rule in rules 
            if rule.correctness_tag in [CorrectnessTag.OUT_OF_SCOPE, CorrectnessTag.LOW_CONFIDENCE]
        ]
        
        if escalation_rules:
            enhanced_logger.warning("Rules requiring escalation detected",
                                  escalation_count=len(escalation_rules),
                                  total_rules=len(rules))
            
            for rule in escalation_rules:
                enhanced_logger.info("Escalation case",
                                   rule_text=rule.rule_text[:100],
                                   correctness_tag=rule.correctness_tag.value,
                                   confidence_score=rule.confidence_score,
                                   manufacturing_relevance=rule.manufacturing_relevance,
                                   source_document=rule.source_document)
    
    def _is_high_value_chunk(self, text_chunk: str) -> bool:
        """Determine if a text chunk contains high-value manufacturing content."""
        
        chunk_lower = text_chunk.lower()
        
        # Count high-priority manufacturing keywords
        high_priority_count = sum(1 for kw in self.manufacturing_keywords['high_priority'] 
                                if kw in chunk_lower)
        
        # Count manufacturing process keywords
        manufacturing_count = sum(1 for kw in self.manufacturing_keywords['manufacturing'] 
                                if kw in chunk_lower)
        
        # Check for numeric values (indicating specifications)
        has_numbers = bool(re.search(r'\d+', text_chunk))
        
        # Modal requirements often indicate actionable rules without numbers
        has_modal_requirement = any(m in chunk_lower for m in [' shall ', ' must ', ' should '])
        
        # Calculate manufacturing density
        total_words = len(text_chunk.split())
        manufacturing_density = (high_priority_count + manufacturing_count) / total_words if total_words > 0 else 0
        
        # VERY RELAXED: Accept almost anything with manufacturing keywords or numbers
        is_high_value = (
            manufacturing_density >= 0.005 or  # ultra-low threshold
            has_numbers or  # any numbers
            has_modal_requirement or  # modal verbs
            high_priority_count > 0 or  # ANY high priority keyword
            manufacturing_count > 0  # ANY manufacturing keyword
        )
        
        return is_high_value
    
    def _is_rule_quality_acceptable(self, rule: ManufacturingRule) -> bool:
        """Check if extracted rule meets quality standards."""
        
        # Must have minimum confidence
        min_confidence = max(self.config.min_confidence_threshold, 0.3)
        if rule.confidence_score < min_confidence:
            return False
        
        # Must have manufacturing relevance
        min_relevance = max(self.config.min_manufacturing_relevance, 0.15)
        if rule.manufacturing_relevance < min_relevance:
            return False
        
        # Must have some numeric content or specific terminology
        rule_text_lower = rule.rule_text.lower()
        has_specific_terms = any(kw in rule_text_lower for kw in self.manufacturing_keywords['high_priority'])
        has_numbers = bool(re.search(r'\d+', rule.rule_text))
        has_modal_requirement = any(m in rule_text_lower for m in [' shall ', ' must ', ' should '])

        # Accept if numeric or strongly worded requirement even without numbers
        if not (has_specific_terms or has_numbers or has_modal_requirement):
            return False
        
        # Must have reasonable length
        if len(rule.rule_text.strip()) < 15:
            return False
        
        return True
    
    def _enhance_rule_quality(self, rule: ManufacturingRule, original_chunk: str, 
                            document_context: DocumentContext) -> ManufacturingRule:
        """Attempt to enhance rule quality using additional context."""
        
        try:
            # Prepare enhancement context
            similar_rules_str = ""
            if self.vector_manager:
                similar = self.vector_manager.similarity_search(
                    rule.rule_text, top_k=2, score_threshold=0.6
                )
                if similar:
                    similar_rules_str = "\n".join([s['text'][:100] for s in similar])
            
            # Run enhancement chain
            enhanced_rule = self.enhancement_chain.invoke({
                'original_rule': rule.dict(),
                'document_context': document_context.dict(),
                'similar_rules': similar_rules_str,
                'format_instructions': self.rule_parser.get_format_instructions()
            })

            if self.config.api_request_delay:
                time.sleep(self.config.api_request_delay)
            
            # Preserve original metadata but update core fields
            enhanced_rule.source_document = rule.source_document
            enhanced_rule.extracted_at = rule.extracted_at
            enhanced_rule.extraction_method = f"enhanced_{rule.extraction_method}"
            
            enhanced_logger.debug("Rule enhancement attempted",
                                original_confidence=rule.confidence_score,
                                enhanced_confidence=enhanced_rule.confidence_score)
            
            return enhanced_rule
            
        except Exception as e:
            enhanced_logger.warning("Rule enhancement failed", error=str(e))
            return rule
    
    def process_document(self, document_path: str) -> RuleExtractionResult:
        """Process a single document and extract rules."""
        
        # Extract text
        text = self.ingester.extract_text_from_file(document_path)
        filename = document_path.split('/')[-1]
        
        # Extract rules (sync wrapper for async method)
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(self.extract_rules_parallel(text, filename))
        
        # Add to vector database if available
        if self.vector_manager and result.rules:
            chunks = self.semantic_chunk_text(text)
            metadata = {'source_file': filename, 'rule_count': len(result.rules)}
            self.vector_manager.add_texts(
                [chunk['text'] for chunk in chunks],
                [metadata] * len(chunks)
            )
        
        return result
    
    def validate_against_hcl_dataset(self, hcl_csv_path: str) -> Dict[str, float]:
        """Validate extraction accuracy against HCL classification dataset."""
        
        try:
            # Load HCL dataset
            df = pd.read_csv(hcl_csv_path)
            
            validation_results = {
                'total_samples': len(df),
                'correctly_classified': 0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
            
            # Process each rule text
            for idx, row in df.iterrows():
                rule_text = row['rule_text']
                expected_label = row['classification_label']
                
                # Extract rule using our engine
                result = self.extract_rules_parallel(rule_text, f"hcl_sample_{idx}")
                
                if result.rules:
                    # Use the highest confidence rule for comparison
                    best_rule = max(result.rules, key=lambda x: x.confidence_score)
                    
                    # Simple classification: 0 = general, 1 = specific manufacturing
                    predicted_label = 1 if best_rule.confidence_score > 0.7 else 0
                    
                    if predicted_label == expected_label:
                        validation_results['correctly_classified'] += 1
            
            # Calculate metrics
            accuracy = validation_results['correctly_classified'] / validation_results['total_samples']
            validation_results['accuracy'] = accuracy
            
            enhanced_logger.info("HCL validation completed",
                               accuracy=accuracy,
                               total_samples=validation_results['total_samples'])
            
            return validation_results
            
        except Exception as e:
            enhanced_logger.error("HCL validation failed", error=str(e))
            return {'error': str(e)}

    def export_results(self, results: List[RuleExtractionResult], 
                      output_format: str = "excel") -> str:
        """Export extraction results in various formats."""
        
        # Flatten all rules
        all_rules = []
        for result in results:
            all_rules.extend(result.rules)
        
        # Convert to DataFrame
        rules_data = []
        for rule in all_rules:
            rules_data.append({
                'rule_text': rule.rule_text,
                'classification_label': 1 if rule.confidence_score > 0.7 else 0,
                'confidence_score': rule.confidence_score,
                'rule_category': rule.rule_category.value,
                'rule_type': rule.rule_type.value,
                'primary_feature': rule.primary_feature,
                'operator': rule.operator,
                'value': rule.value,
                'unit': rule.unit,
                'source_document': rule.source_document,
                'extraction_method': rule.extraction_method,
                'manufacturing_relevance': rule.manufacturing_relevance
            })
        
        df = pd.DataFrame(rules_data)
        
        # Export based on format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_format == "excel":
            output_path = f"enhanced_rules_export_{timestamp}.xlsx"
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Manufacturing_Rules', index=False)
                
                # Add summary sheet
                summary_df = pd.DataFrame({
                    'Metric': ['Total Rules', 'Avg Confidence', 'High Confidence Rules', 'Manufacturing Categories'],
                    'Value': [
                        len(df),
                        df['confidence_score'].mean(),
                        len(df[df['confidence_score'] > 0.8]),
                        df['rule_category'].nunique()
                    ]
                })
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        elif output_format == "csv":
            output_path = f"enhanced_rules_export_{timestamp}.csv"
            df.to_csv(output_path, index=False)
        
        else:  # JSON
            output_path = f"enhanced_rules_export_{timestamp}.json"
            df.to_json(output_path, orient='records', indent=2)
        
        enhanced_logger.info("Results exported", 
                           output_path=output_path,
                           rule_count=len(df))
        
        return output_path

    def _result_to_dict(
        self,
        result: RuleExtractionResult,
        *,
        fallback_filename: str = "ad_hoc.txt",
    ) -> Dict[str, Any]:
        serialized_rules = [rule.dict() for rule in result.rules]
        avg_confidence = result.extraction_stats.get('avg_confidence', 0.0)

        payload = {
            'filename': result.document_metadata.get('filename', fallback_filename) if isinstance(result.document_metadata, dict) else fallback_filename,
            'status': 'success' if serialized_rules else result.extraction_stats.get('status', 'no_rules'),
            'rules': serialized_rules,
            'rule_count': len(serialized_rules),
            'avg_confidence': avg_confidence,
            'processing_time': result.processing_time,
            'document_metadata': result.document_metadata,
            'extraction_stats': result.extraction_stats,
            'token_usage': result.token_usage,
        }
        return payload