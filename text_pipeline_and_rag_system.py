#!/usr/bin/env python3
"""
text_pipeline_and_rag_system.py

Complete Text Processing Pipeline Integrated with RAG System
Combines document ingestion, embedding, retrieval, and generation capabilities
into a single cohesive implementation.

This system can:
- Process PDF documents and extract text, tables, and images
- Handle documents with or without manufacturing keywords
- Use LLM-based context understanding (optional)
- Extract implicit manufacturing rules using NLP
- Create semantic embeddings with BAAI/bge-large-en-v1.5
- Store and retrieve documents using ChromaDB vector database
- Generate manufacturing rules and constraints from generic documents

Author: Merged from temp-RAG repository
License: MIT
"""

import os
import sys
import json
import re
import hashlib
from typing import List, Dict, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from io import BytesIO

# Core dependencies
import numpy as np
import pandas as pd

# NLP and ML
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

# PDF processing
from pdfminer.high_level import extract_text
import PyPDF2

# Embeddings and vector store
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Transformers for advanced NLP
from transformers import pipeline

# LLM clients (optional)
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

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# =============================================================================
# NLTK DATA DOWNLOADS
# =============================================================================

def setup_nltk_data():
    """Download required NLTK data if not already present."""
    required_data = [
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
        ('chunkers/maxent_ne_chunker_tab', 'maxent_ne_chunker_tab'),
    ]
    
    for path, name in required_data:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading NLTK data: {name}")
            nltk.download(name, quiet=True)

# Setup NLTK data on import
setup_nltk_data()

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DocumentMetadata:
    """Metadata structure for manufacturing documents."""
    doc_id: str
    source_file: str
    doc_type: str  # 'text', 'table', 'image'
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    manufacturing_process: Optional[str] = None
    rule_category: Optional[str] = None
    chunk_index: int = 0
    confidence_score: Optional[float] = None
    extracted_at: str = datetime.now().isoformat()
    features: Optional[List[str]] = None
    constraints: Optional[List[str]] = None
    implicit_rules_count: Optional[int] = None
    semantic_features: Optional[List[str]] = None
    manufacturing_relevance_score: Optional[float] = None
    rule_extraction_method: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, filtering None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}

@dataclass
class ImplicitRule:
    """Structure for rules extracted from non-obvious content."""
    text: str
    confidence_score: float
    rule_type: str
    semantic_features: List[str]
    constraint_type: str
    constraint_value: Optional[str]
    context_indicators: List[str]
    manufacturing_relevance: float
    extracted_entities: List[Dict[str, str]]

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

# =============================================================================
# TEXT EXTRACTION
# =============================================================================

class TextExtractor:
    """Extract text content from PDF documents."""
    
    @staticmethod
    def extract_sentences(pdf_bytes: bytes) -> List[str]:
        """
        Extract sentences from PDF bytes.
        
        Args:
            pdf_bytes: PDF file as bytes
            
        Returns:
            List of extracted sentences
        """
        try:
            # Convert bytes to file-like object
            pdf_file_like = BytesIO(pdf_bytes)
            
            # Extract text
            text = extract_text(pdf_file_like)
            
            # Clean up common PDF artifacts
            text = text.replace('(cid:415)', 'ti')
            text = text.replace('(cid:425)', 'tt')
            
            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            # Clean sentences
            sentences = [s.strip().replace('\n', ' ') for s in sentences]
            sentences = [s for s in sentences if s]
            
            return sentences
            
        except Exception as e:
            print(f"Error extracting text: {e}")
            return []
    
    @staticmethod
    def extract_text_simple(pdf_bytes: bytes) -> str:
        """
        Extract raw text from PDF.
        
        Args:
            pdf_bytes: PDF file as bytes
            
        Returns:
            Extracted text as string
        """
        sentences = TextExtractor.extract_sentences(pdf_bytes)
        return ' '.join(sentences)

# =============================================================================
# CUSTOM EMBEDDINGS
# =============================================================================

class SentenceTransformerEmbeddings(Embeddings):
    """Custom embeddings wrapper for sentence transformers."""
    
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        """
        Initialize embeddings model.
        
        Args:
            model_name: HuggingFace model name for embeddings
        """
        print(f"Loading embedding model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
        except Exception as e:
            print(f"⚠️  Warning: Could not load {model_name}: {e}")
            print(f"   Falling back to all-MiniLM-L6-v2 (smaller model)")
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.model_name = 'all-MiniLM-L6-v2'
            except Exception as e2:
                print(f"⚠️  Warning: Could not load fallback model: {e2}")
                print(f"   Using sentence-transformers/all-mpnet-base-v2")
                self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
                self.model_name = 'sentence-transformers/all-mpnet-base-v2'
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

# =============================================================================
# IMPLICIT RULE EXTRACTOR
# =============================================================================

class ImplicitRuleExtractor:
    """Extract manufacturing rules from documents without explicit keywords."""
    
    def __init__(self):
        """Initialize NLP models and rule patterns."""
        
        # Load spaCy if available
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("⚠️  spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
        else:
            self.nlp = None
        
        # Semantic similarity model
        print("Loading semantic model for implicit rule extraction...")
        try:
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"⚠️  Warning: Could not load semantic model: {e}")
            print("   Some features will be disabled")
            self.semantic_model = None
        
        # Zero-shot classification
        print("Loading zero-shot classifier...")
        try:
            self.zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
        except Exception as e:
            print(f"⚠️  Warning: Could not load zero-shot classifier: {e}")
            print("   Some features will be disabled")
            self.zero_shot_classifier = None
        
        # Manufacturing rule templates for semantic matching
        self.manufacturing_rule_templates = [
            "Design guidelines for product development",
            "Quality requirements for manufacturing processes", 
            "Safety constraints for assembly operations",
            "Performance specifications for components",
            "Tolerance requirements for mechanical parts",
            "Material selection criteria for durability",
            "Process optimization for efficiency",
            "Inspection procedures for quality control",
            "Assembly instructions for proper installation",
            "Maintenance requirements for equipment"
        ]
        
        # Implicit constraint patterns
        self.implicit_patterns = {
            'requirement': [
                r'\b(?:must|should|shall|require[sd]?|need[sd]?|essential|necessary|important)\b',
                r'\b(?:ensure|guarantee|maintain|provide|achieve)\b',
            ],
            'prohibition': [
                r'\b(?:avoid|prevent|prohibit|forbid|not allowed|cannot|must not|should not)\b',
                r'\b(?:eliminate|reduce|minimize|limit)\b',
            ],
            'recommendation': [
                r'\b(?:recommend|suggest|advise|prefer|optimal|best practice)\b',
                r'\b(?:consider|evaluate|assess|review)\b',
            ],
            'condition': [
                r'\b(?:if|when|where|provided that|assuming|given that)\b',
                r'\b(?:in case of|during|while|throughout)\b',
            ],
            'measurement': [
                r'\b\d+(?:\.\d+)?\s*(?:mm|cm|m|in|inch|mil|micron|μm|deg|degree|%|percent)\b',
                r'\b(?:thickness|diameter|length|width|height|radius|depth|size|dimension)\b',
            ]
        }
        
        # Manufacturing categories
        self.manufacturing_categories = [
            "mechanical design", "manufacturing process", "quality control",
            "assembly procedure", "material specification", "safety requirement",
            "dimensional tolerance", "surface finish", "structural integrity",
            "thermal management", "electrical specification", "performance criteria"
        ]
    
    def extract_implicit_rules(self, text: str, confidence_threshold: float = 0.6) -> List[ImplicitRule]:
        """
        Extract potential manufacturing rules from any text content.
        
        Args:
            text: Text to analyze
            confidence_threshold: Minimum confidence score
            
        Returns:
            List of extracted implicit rules
        """
        if not text or not isinstance(text, str):
            return []
        
        # Split into sentences
        sentences = sent_tokenize(text)
        
        rules = []
        for sentence in sentences:
            rule = self._analyze_sentence_for_rules(sentence)
            if rule and rule.confidence_score >= confidence_threshold:
                rules.append(rule)
        
        return rules
    
    def _analyze_sentence_for_rules(self, sentence: str) -> Optional[ImplicitRule]:
        """Analyze a single sentence for implicit rule patterns."""
        
        # Check manufacturing relevance
        manufacturing_relevance = self._calculate_manufacturing_relevance(sentence)
        
        if manufacturing_relevance < 0.3:
            return None
        
        # Detect rule indicators
        rule_indicators = self._detect_rule_indicators(sentence)
        
        if not rule_indicators:
            return None
        
        # Extract semantic features
        semantic_features = self._extract_semantic_features(sentence)
        
        # Classify constraint type
        constraint_type = self._classify_constraint_type(sentence, rule_indicators)
        
        # Extract constraint value
        constraint_value = self._extract_constraint_value(sentence)
        
        # Extract entities
        entities = self._extract_entities(sentence)
        
        # Determine rule type
        rule_type = self._classify_rule_type(sentence)
        
        # Calculate confidence
        confidence = self._calculate_confidence_score(
            sentence, rule_indicators, semantic_features, 
            manufacturing_relevance, entities
        )
        
        return ImplicitRule(
            text=sentence,
            confidence_score=confidence,
            rule_type=rule_type,
            semantic_features=semantic_features,
            constraint_type=constraint_type,
            constraint_value=constraint_value,
            context_indicators=rule_indicators,
            manufacturing_relevance=manufacturing_relevance,
            extracted_entities=entities
        )
    
    def _calculate_manufacturing_relevance(self, sentence: str) -> float:
        """Calculate how relevant the sentence is to manufacturing."""
        if not self.semantic_model:
            # Fallback: simple keyword matching
            keywords = ['manufacturing', 'assembly', 'quality', 'specification', 'requirement', 'tolerance', 'material', 'process']
            sentence_lower = sentence.lower()
            score = sum(1 for kw in keywords if kw in sentence_lower) / len(keywords)
            return min(score * 2, 1.0)  # Boost and cap at 1.0
            
        try:
            sentence_embedding = self.semantic_model.encode([sentence])
            template_embeddings = self.semantic_model.encode(self.manufacturing_rule_templates)
            
            similarities = np.dot(sentence_embedding, template_embeddings.T)
            max_similarity = np.max(similarities)
            
            return float(max_similarity)
        except Exception as e:
            print(f"Error calculating manufacturing relevance: {e}")
            return 0.0
    
    def _detect_rule_indicators(self, sentence: str) -> List[str]:
        """Detect rule indicators using pattern matching."""
        indicators = []
        sentence_lower = sentence.lower()
        
        for pattern_type, patterns in self.implicit_patterns.items():
            for pattern in patterns:
                if re.search(pattern, sentence_lower):
                    indicators.append(pattern_type)
                    break
        
        return list(set(indicators))
    
    def _extract_semantic_features(self, sentence: str) -> List[str]:
        """Extract semantic features using spaCy NLP."""
        if not self.nlp:
            return []
        
        features = []
        doc = self.nlp(sentence)
        
        # Extract nouns and adjectives
        for token in doc:
            if token.pos_ in ['NOUN', 'ADJ'] and len(token.text) > 2:
                features.append(token.lemma_.lower())
        
        # Extract noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:
                features.append(chunk.text.lower())
        
        return list(set(features))
    
    def _classify_constraint_type(self, sentence: str, indicators: List[str]) -> str:
        """Classify the type of constraint."""
        sentence_lower = sentence.lower()
        
        if 'prohibition' in indicators:
            return 'prohibition'
        elif 'requirement' in indicators:
            if any(word in sentence_lower for word in ['minimum', 'at least', 'greater', 'exceed']):
                return 'minimum_requirement'
            elif any(word in sentence_lower for word in ['maximum', 'at most', 'less', 'under']):
                return 'maximum_requirement'
            else:
                return 'general_requirement'
        elif 'recommendation' in indicators:
            return 'recommendation'
        elif 'condition' in indicators:
            return 'conditional_requirement'
        elif 'measurement' in indicators:
            return 'dimensional_constraint'
        else:
            return 'general_guideline'
    
    def _extract_constraint_value(self, sentence: str) -> Optional[str]:
        """Extract numerical values or constraint specifications."""
        num_pattern = r'\b\d+(?:\.\d+)?\s*(?:mm|cm|m|in|inch|mil|micron|μm|deg|degree|%|percent|times|x)\b'
        matches = re.findall(num_pattern, sentence, re.IGNORECASE)
        
        if matches:
            return matches[0]
        
        qual_pattern = r'\b(?:adequate|sufficient|appropriate|proper|optimal|maximum|minimum|standard)\b'
        qual_matches = re.findall(qual_pattern, sentence, re.IGNORECASE)
        
        if qual_matches:
            return qual_matches[0]
        
        return None
    
    def _extract_entities(self, sentence: str) -> List[Dict[str, str]]:
        """Extract named entities and technical terms."""
        entities = []
        
        if self.nlp:
            doc = self.nlp(sentence)
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'type': 'spacy_entity'
                })
        
        try:
            tokens = nltk.word_tokenize(sentence)
            pos_tags = pos_tag(tokens)
            
            for i, (word, pos) in enumerate(pos_tags):
                if pos.startswith('NN') and len(word) > 3:
                    entities.append({
                        'text': word,
                        'label': 'TECHNICAL_TERM',
                        'type': 'nltk_noun'
                    })
        except:
            pass
        
        return entities
    
    def _classify_rule_type(self, sentence: str) -> str:
        """Classify the manufacturing rule type."""
        if not self.zero_shot_classifier:
            # Fallback: simple keyword-based classification
            sentence_lower = sentence.lower()
            if any(kw in sentence_lower for kw in ['machine', 'mill', 'drill', 'cut']):
                return 'mechanical design'
            elif any(kw in sentence_lower for kw in ['quality', 'inspect', 'test']):
                return 'quality control'
            elif any(kw in sentence_lower for kw in ['assemble', 'install', 'mount']):
                return 'assembly procedure'
            elif any(kw in sentence_lower for kw in ['material', 'steel', 'aluminum', 'plastic']):
                return 'material specification'
            else:
                return 'general'
        
        try:
            result = self.zero_shot_classifier(
                sentence, 
                self.manufacturing_categories,
                multi_label=False
            )
            return result['labels'][0] if result['scores'][0] > 0.3 else 'general'
        except Exception as e:
            print(f"Error in rule type classification: {e}")
            return 'general'
    
    def _calculate_confidence_score(
        self, 
        sentence: str, 
        indicators: List[str], 
        features: List[str], 
        manufacturing_relevance: float,
        entities: List[Dict[str, str]]
    ) -> float:
        """Calculate overall confidence score."""
        score = 0.0
        
        score += manufacturing_relevance * 0.3
        
        if indicators:
            score += min(len(indicators) * 0.15, 0.3)
        
        if features:
            score += min(len(features) * 0.05, 0.2)
        
        if entities:
            score += min(len(entities) * 0.03, 0.1)
        
        if re.search(r'\d+(?:\.\d+)?', sentence):
            score += 0.1
        
        return min(max(score, 0.0), 1.0)

# =============================================================================
# LLM CONTEXT ANALYZER
# =============================================================================

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
            model: Specific model to use
        """
        self.api_provider = api_provider.lower()
        self.client = None
        
        if self.api_provider == "groq":
            if not GROQ_AVAILABLE:
                raise ImportError("Groq package not installed")
            
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set")
            
            self.client = Groq(api_key=api_key)
            self.model = model or "llama-3.3-70b-versatile"
            
        elif self.api_provider == "cerebras":
            if not CEREBRAS_AVAILABLE:
                raise ImportError("Cerebras package not installed")
            
            api_key = os.getenv("CEREBRAS_API_KEY")
            if not api_key:
                raise ValueError("CEREBRAS_API_KEY environment variable not set")
            
            self.client = Cerebras(api_key=api_key)
            self.model = model or "llama3.1-70b"
            
        else:
            raise ValueError(f"Unknown API provider: {api_provider}")
    
    def analyze_document_context(self, text: str, max_length: int = 4000) -> DocumentContext:
        """Analyze document using LLM to understand context."""
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        prompt = self._create_context_analysis_prompt(text)
        response = self._call_llm(prompt)
        context = self._parse_context_response(response)
        
        return context
    
    def extract_manufacturing_rules(self, text: str, context: Optional[DocumentContext] = None) -> List[Dict[str, Any]]:
        """Extract manufacturing rules using LLM understanding."""
        if context is None:
            context = self.analyze_document_context(text)
        
        prompt = self._create_rule_extraction_prompt(text, context)
        response = self._call_llm(prompt)
        rules = self._parse_rules_response(response, context)
        
        return rules
    
    def _create_context_analysis_prompt(self, text: str) -> str:
        """Create prompt for document context analysis."""
        return f"""You are an expert at understanding technical documents and extracting manufacturing-related information.

Analyze the following document text and extract:
1. Industry/Domain
2. Purpose
3. Key Concepts
4. Implicit Requirements
5. Constraint Types
6. Manufacturing Relevance (0.0-1.0)
7. Extracted Rules

Document text:
```
{text}
```

Provide analysis in JSON format:
{{
    "industry": "...",
    "domain": "...",
    "purpose": "...",
    "key_concepts": ["..."],
    "implicit_requirements": ["..."],
    "constraint_types": ["..."],
    "manufacturing_relevance_score": 0.0,
    "extracted_rules": [{{"rule": "...", "type": "...", "confidence": 0.0}}],
    "confidence": 0.0
}}"""
    
    def _create_rule_extraction_prompt(self, text: str, context: DocumentContext) -> str:
        """Create prompt for rule extraction."""
        return f"""Extract manufacturing rules from this document.

Context:
- Industry: {context.industry}
- Domain: {context.domain}

Document:
```
{text}
```

Extract ALL rules and format as JSON array:
[
    {{
        "rule_text": "...",
        "rule_type": "...",
        "confidence": 0.0,
        "manufacturing_relevance": "...",
        "category": "..."
    }}
]"""
    
    def _call_llm(self, prompt: str, temperature: float = 0.2, max_tokens: int = 2000) -> str:
        """Call the LLM API."""
        try:
            if self.api_provider == "groq":
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert technical analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return completion.choices[0].message.content
            
            elif self.api_provider == "cerebras":
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert technical analyst."},
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
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                rules_data = json.loads(json_match.group())
            else:
                rules_data = json.loads(response)
            
            rules = []
            for rule_data in rules_data:
                rule = {
                    "text": rule_data.get("rule_text", ""),
                    "type": rule_data.get("rule_type", "general"),
                    "confidence": float(rule_data.get("confidence", 0.5)),
                    "manufacturing_relevance": rule_data.get("manufacturing_relevance", ""),
                    "category": rule_data.get("category", "general"),
                    "document_industry": context.industry,
                    "document_domain": context.domain,
                    "extraction_method": "llm_analysis"
                }
                rules.append(rule)
            
            return rules
            
        except Exception as e:
            print(f"Error parsing rules: {e}")
            return []

# =============================================================================
# TEXT SPLITTER
# =============================================================================

class ManufacturingTextSplitter:
    """Specialized text splitter for manufacturing documents."""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        """
        Initialize text splitter.
        
        Args:
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )
        
        self.section_headers = [
            "design guidelines", "manufacturing constraints", "specifications",
            "requirements", "tolerances", "materials", "processes", "quality",
            "inspection", "testing", "assembly", "fabrication"
        ]
    
    def split_with_structure(self, text: str, metadata: DocumentMetadata) -> List[Document]:
        """Split text while preserving manufacturing document structure."""
        documents = []
        
        lines = text.split('\n')
        current_section = ""
        current_content = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            is_header = any(header in line_lower for header in self.section_headers)
            
            if is_header and current_content:
                section_text = '\n'.join(current_content)
                chunks = self.base_splitter.split_text(section_text)
                
                for i, chunk in enumerate(chunks):
                    chunk_metadata = DocumentMetadata(
                        doc_id=f"{metadata.doc_id}_section_{hashlib.md5(current_section.encode()).hexdigest()[:8]}",
                        source_file=metadata.source_file,
                        doc_type=metadata.doc_type,
                        page_number=metadata.page_number,
                        section_title=current_section,
                        chunk_index=i
                    )
                    
                    documents.append(Document(
                        page_content=chunk,
                        metadata=chunk_metadata.to_dict()
                    ))
                
                current_section = line.strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Process final section
        if current_content:
            section_text = '\n'.join(current_content)
            chunks = self.base_splitter.split_text(section_text)
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = DocumentMetadata(
                    doc_id=f"{metadata.doc_id}_final_{i}",
                    source_file=metadata.source_file,
                    doc_type=metadata.doc_type,
                    page_number=metadata.page_number,
                    section_title=current_section,
                    chunk_index=i
                )
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=chunk_metadata.to_dict()
                ))
        
        return documents

# =============================================================================
# RAG SYSTEM
# =============================================================================

class UniversalRAGSystem:
    """
    Universal RAG System that can handle any type of document.
    Combines keyword-based and semantic-based rule extraction.
    """
    
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        persist_path: str = "universal_rag_db",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        use_llm: bool = False,
        llm_provider: Optional[str] = None
    ):
        """
        Initialize Universal RAG System.
        
        Args:
            embedding_model_name: HuggingFace model for embeddings (default: all-MiniLM-L6-v2)
            persist_path: Directory for ChromaDB persistence
            chunk_size: Text chunk size
            chunk_overlap: Overlap between chunks
            use_llm: Whether to use LLM for enhanced understanding
            llm_provider: LLM provider ("groq" or "cerebras")
        """
        print("Initializing Universal RAG System...")
        
        # Embeddings
        self.embeddings = SentenceTransformerEmbeddings(embedding_model_name)
        self.persist_path = persist_path
        
        # Text splitter
        self.text_splitter = ManufacturingTextSplitter(chunk_size, chunk_overlap)
        
        # Vector store
        print(f"Initializing vector store at: {persist_path}")
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=persist_path
        )
        
        # Implicit rule extractor
        self.implicit_extractor = ImplicitRuleExtractor()
        
        # LLM analyzer (optional)
        self.use_llm = use_llm
        self.llm_analyzer = None
        
        if use_llm:
            try:
                if llm_provider:
                    self.llm_analyzer = LLMContextAnalyzer(api_provider=llm_provider)
                else:
                    # Try to get default
                    if check_api_availability()["groq"]:
                        self.llm_analyzer = LLMContextAnalyzer(api_provider="groq")
                    elif check_api_availability()["cerebras"]:
                        self.llm_analyzer = LLMContextAnalyzer(api_provider="cerebras")
                
                if self.llm_analyzer:
                    print(f"✓ LLM analyzer initialized: {self.llm_analyzer.api_provider}")
            except Exception as e:
                print(f"⚠️  LLM initialization failed: {e}")
                self.use_llm = False
        
        # Document registry
        self.doc_registry: Dict[str, Dict[str, Any]] = {}
        self.load_registry()
        
        # Statistics
        self.processing_stats = {
            'keyword_based_rules': 0,
            'implicit_rules': 0,
            'llm_rules': 0,
            'documents_processed': 0
        }
        
        print("✓ RAG System initialized successfully")
    
    def load_registry(self):
        """Load document registry from disk."""
        registry_path = Path(self.persist_path) / "doc_registry.json"
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                self.doc_registry = json.load(f)
    
    def save_registry(self):
        """Save document registry to disk."""
        registry_path = Path(self.persist_path) / "doc_registry.json"
        os.makedirs(self.persist_path, exist_ok=True)
        with open(registry_path, 'w') as f:
            json.dump(self.doc_registry, f, indent=2)
    
    def process_document(self, pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Process a PDF document and add to the RAG system.
        
        Args:
            pdf_bytes: PDF file as bytes
            filename: Name of the document
            
        Returns:
            Processing results dictionary
        """
        print(f"\n{'='*70}")
        print(f"Processing document: {filename}")
        print(f"{'='*70}")
        
        doc_id = hashlib.md5(pdf_bytes).hexdigest()[:16]
        
        if doc_id in self.doc_registry:
            print("⚠️  Document already processed")
            return {"message": "Document already processed", "doc_id": doc_id}
        
        results = {
            "filename": filename,
            "doc_id": doc_id,
            "text_chunks": 0,
            "implicit_rules": 0,
            "llm_rules": 0,
            "processing_method": "unknown",
            "manufacturing_relevance": 0.0
        }
        
        try:
            # Extract text
            print("1. Extracting text...")
            text_content = TextExtractor.extract_text_simple(pdf_bytes)
            
            if not text_content:
                return {"error": "No text content extracted"}
            
            print(f"   Extracted {len(text_content)} characters")
            
            # LLM-based processing (if available)
            if self.use_llm and self.llm_analyzer:
                print("2. Analyzing with LLM...")
                try:
                    context = self.llm_analyzer.analyze_document_context(text_content)
                    results["manufacturing_relevance"] = context.manufacturing_relevance_score
                    print(f"   Industry: {context.industry}")
                    print(f"   Manufacturing relevance: {context.manufacturing_relevance_score:.2f}")
                    
                    llm_rules = self.llm_analyzer.extract_manufacturing_rules(text_content, context)
                    results["llm_rules"] = len(llm_rules)
                    self.processing_stats['llm_rules'] += len(llm_rules)
                    print(f"   Extracted {len(llm_rules)} LLM rules")
                    
                    results["processing_method"] = "llm_enhanced"
                except Exception as e:
                    print(f"   ⚠️  LLM processing error: {e}")
            
            # Implicit rule extraction
            print("3. Extracting implicit rules...")
            implicit_rules = self.implicit_extractor.extract_implicit_rules(text_content, confidence_threshold=0.6)
            results["implicit_rules"] = len(implicit_rules)
            self.processing_stats['implicit_rules'] += len(implicit_rules)
            print(f"   Extracted {len(implicit_rules)} implicit rules")
            
            # Calculate relevance if not from LLM
            if results["manufacturing_relevance"] == 0.0 and implicit_rules:
                avg_relevance = sum(r.manufacturing_relevance for r in implicit_rules) / len(implicit_rules)
                results["manufacturing_relevance"] = avg_relevance
            
            # Create document chunks
            print("4. Creating document chunks...")
            text_metadata = DocumentMetadata(
                doc_id=doc_id,
                source_file=filename,
                doc_type="text",
                manufacturing_relevance_score=results["manufacturing_relevance"]
            )
            
            documents = self.text_splitter.split_with_structure(text_content, text_metadata)
            
            # Add implicit rules as documents
            for i, rule in enumerate(implicit_rules):
                rule_metadata = DocumentMetadata(
                    doc_id=f"{doc_id}_implicit_{i}",
                    source_file=filename,
                    doc_type="text",
                    rule_category=rule.rule_type,
                    confidence_score=rule.confidence_score,
                    manufacturing_relevance_score=rule.manufacturing_relevance,
                    rule_extraction_method="implicit"
                )
                
                documents.append(Document(
                    page_content=rule.text,
                    metadata=rule_metadata.to_dict()
                ))
            
            # Add to vector store
            print("5. Adding to vector store...")
            if documents:
                self._add_documents_to_vectorstore(documents)
                results["text_chunks"] = len(documents)
                print(f"   Added {len(documents)} chunks to database")
            
            # Update registry
            self.doc_registry[doc_id] = {
                "filename": filename,
                "processed_at": datetime.now().isoformat(),
                "results": results
            }
            self.save_registry()
            
            self.processing_stats['documents_processed'] += 1
            
            print(f"\n✓ Document processed successfully")
            print(f"  Total chunks: {results['text_chunks']}")
            print(f"  Implicit rules: {results['implicit_rules']}")
            print(f"  Manufacturing relevance: {results['manufacturing_relevance']:.2f}")
            
        except Exception as e:
            print(f"\n✗ Error processing document: {e}")
            results["error"] = str(e)
        
        return results
    
    def _add_documents_to_vectorstore(self, documents: List[Document]):
        """Add documents to the vector store."""
        if not documents:
            return
        
        texts = [doc.page_content for doc in documents]
        
        # Filter metadata to only include scalar types
        metadatas = []
        for doc in documents:
            filtered_metadata = {}
            for key, value in doc.metadata.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    filtered_metadata[key] = value
                elif isinstance(value, list):
                    filtered_metadata[key] = ', '.join(str(v) for v in value if v)
            metadatas.append(filtered_metadata)
        
        self.vectorstore.add_texts(texts, metadatas)
    
    def query(self, query: str, top_k: int = 5, filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Query the RAG system.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of results with scores and metadata
        """
        print(f"\nQuerying: '{query}'")
        
        # Perform similarity search
        if filter_metadata:
            results = self.vectorstore.similarity_search_with_score(
                query, k=top_k, filter=filter_metadata
            )
        else:
            results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        
        # Format results
        formatted_results = []
        for doc, score in results:
            result = {
                'text': doc.page_content,
                'metadata': doc.metadata,
                'similarity_score': float(score),
                'source': doc.metadata.get('source_file', 'unknown'),
                'relevance': doc.metadata.get('manufacturing_relevance_score', 0.0)
            }
            formatted_results.append(result)
        
        print(f"Found {len(formatted_results)} results")
        
        return formatted_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "total_documents": len(self.doc_registry),
            "processing_stats": self.processing_stats,
            "embedding_model": self.embeddings.model_name,
            "persist_path": self.persist_path,
            "llm_enabled": self.use_llm,
            "llm_provider": self.llm_analyzer.api_provider if self.llm_analyzer else None
        }
    
    def clear_database(self):
        """Clear the entire knowledge base."""
        print("Clearing database...")
        self.vectorstore.delete_collection()
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.persist_path
        )
        self.doc_registry = {}
        self.save_registry()
        print("✓ Database cleared")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

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

def print_system_status():
    """Print system status and requirements."""
    print("\n" + "="*70)
    print("TEXT PROCESSING PIPELINE AND RAG SYSTEM")
    print("="*70)
    
    print("\n📦 Required Dependencies:")
    required_packages = [
        "sentence-transformers", "transformers", "torch", "chromadb",
        "langchain", "langchain-chroma", "langchain-text-splitters",
        "pdfminer.six", "PyPDF2", "nltk", "numpy", "pandas"
    ]
    for pkg in required_packages:
        print(f"  - {pkg}")
    
    print("\n🔧 Optional Dependencies (for LLM enhancement):")
    print("  - groq (for Groq API)")
    print("  - cerebras-cloud-sdk (for Cerebras API)")
    print("  - spacy (for advanced NLP)")
    
    print("\n🌐 API Status:")
    api_status = check_api_availability()
    for api, available in api_status.items():
        status = "✓ Available" if available else "✗ Not configured"
        print(f"  - {api}: {status}")
    
    print("\n💡 To enable LLM features:")
    print("  1. Get API key from https://console.groq.com/keys")
    print("  2. Set environment variable: export GROQ_API_KEY='your-key'")
    print("  3. Restart the application")
    
    print("\n📚 Example Usage:")
    print("  rag = UniversalRAGSystem(use_llm=True)")
    print("  with open('document.pdf', 'rb') as f:")
    print("      results = rag.process_document(f.read(), 'document.pdf')")
    print("  results = rag.query('What are the quality requirements?')")
    
    print("\n" + "="*70)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function demonstrating system usage."""
    print_system_status()
    
    # Initialize RAG system
    print("\n" + "="*70)
    print("INITIALIZING RAG SYSTEM")
    print("="*70)
    
    # Check if LLM is available
    api_available = any(check_api_availability().values())
    
    rag = UniversalRAGSystem(
        embedding_model_name="all-MiniLM-L6-v2",  # Smaller, faster model
        persist_path="./universal_rag_db",
        use_llm=api_available
    )
    
    # Test with sample text
    print("\n" + "="*70)
    print("TESTING WITH SAMPLE TEXT")
    print("="*70)
    
    sample_text = """
    Manufacturing Design Guidelines
    
    Components should maintain appropriate spacing to prevent interference during operation.
    Materials must be selected considering thermal expansion effects in high-temperature environments.
    Assembly procedures require adequate access for maintenance and inspection activities.
    
    Quality Requirements
    
    All surfaces must be free from defects that could compromise functionality.
    Dimensional tolerances should be specified according to functional requirements.
    The minimum bend radius should be at least 1.5 times the material thickness.
    """
    
    # Create a temporary PDF-like bytes object for testing
    # In real usage, you would load actual PDF files
    print("\nProcessing sample text as document...")
    
    # For demonstration, we'll process the text directly
    text_metadata = DocumentMetadata(
        doc_id="sample_001",
        source_file="sample_guidelines.txt",
        doc_type="text"
    )
    
    documents = rag.text_splitter.split_with_structure(sample_text, text_metadata)
    rag._add_documents_to_vectorstore(documents)
    
    print(f"✓ Added {len(documents)} chunks to database")
    
    # Test queries
    print("\n" + "="*70)
    print("TESTING QUERIES")
    print("="*70)
    
    test_queries = [
        "What are the spacing requirements?",
        "Tell me about material selection",
        "What are the quality requirements?"
    ]
    
    for query in test_queries:
        results = rag.query(query, top_k=3)
        print(f"\nQuery: {query}")
        print(f"Results: {len(results)}")
        for i, result in enumerate(results, 1):
            print(f"\n  {i}. Score: {result['similarity_score']:.3f}")
            print(f"     Text: {result['text'][:100]}...")
    
    # Print statistics
    print("\n" + "="*70)
    print("SYSTEM STATISTICS")
    print("="*70)
    
    stats = rag.get_stats()
    print(json.dumps(stats, indent=2))
    
    print("\n✅ System test completed successfully!")

if __name__ == "__main__":
    main()
