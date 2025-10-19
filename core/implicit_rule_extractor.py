"""
Implicit Rule Extractor for Random Documents
Handles documents without clear manufacturing keywords by using advanced NLP techniques
"""

import re
import spacy
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

try:
    nltk.data.find('chunkers/maxent_ne_chunker_tab')
except LookupError:
    nltk.download('maxent_ne_chunker_tab')

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

class ImplicitRuleExtractor:
    """Extract manufacturing rules from documents without explicit keywords."""
    
    def __init__(self):
        """Initialize NLP models and rule patterns."""
        
        # Load advanced NLP models
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Please install spaCy English model: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Semantic similarity model for manufacturing relevance
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Zero-shot classification for rule detection
        self.zero_shot_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        
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
        
        # Implicit constraint patterns (not keyword-dependent)
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
        
        # Semantic categories for manufacturing relevance
        self.manufacturing_categories = [
            "mechanical design", "manufacturing process", "quality control",
            "assembly procedure", "material specification", "safety requirement",
            "dimensional tolerance", "surface finish", "structural integrity",
            "thermal management", "electrical specification", "performance criteria"
        ]
    
    def extract_implicit_rules(self, text: str, confidence_threshold: float = 0.6) -> List[ImplicitRule]:
        """Extract potential manufacturing rules from any text content."""
        
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
        
        # Step 1: Check manufacturing relevance using semantic similarity
        manufacturing_relevance = self._calculate_manufacturing_relevance(sentence)
        
        if manufacturing_relevance < 0.3:  # Skip non-manufacturing content
            return None
        
        # Step 2: Detect rule indicators using pattern matching
        rule_indicators = self._detect_rule_indicators(sentence)
        
        if not rule_indicators:
            return None
        
        # Step 3: Extract semantic features using NLP
        semantic_features = self._extract_semantic_features(sentence)
        
        # Step 4: Classify constraint type
        constraint_type = self._classify_constraint_type(sentence, rule_indicators)
        
        # Step 5: Extract numerical values or constraint values
        constraint_value = self._extract_constraint_value(sentence)
        
        # Step 6: Extract named entities
        entities = self._extract_entities(sentence)
        
        # Step 7: Determine rule type using zero-shot classification
        rule_type = self._classify_rule_type(sentence)
        
        # Step 8: Calculate confidence score
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
        """Calculate how relevant the sentence is to manufacturing using semantic similarity."""
        
        try:
            # Encode the sentence and manufacturing templates
            sentence_embedding = self.semantic_model.encode([sentence])
            template_embeddings = self.semantic_model.encode(self.manufacturing_rule_templates)
            
            # Calculate cosine similarities
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
                    break  # Only add each type once
        
        return list(set(indicators))  # Remove duplicates
    
    def _extract_semantic_features(self, sentence: str) -> List[str]:
        """Extract semantic features using spaCy NLP."""
        
        if not self.nlp:
            return []
        
        features = []
        doc = self.nlp(sentence)
        
        # Extract nouns and adjectives (potential features)
        for token in doc:
            if token.pos_ in ['NOUN', 'ADJ'] and len(token.text) > 2:
                features.append(token.lemma_.lower())
        
        # Extract noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Keep short phrases
                features.append(chunk.text.lower())
        
        # Extract dependency relations for technical terms
        for token in doc:
            if token.dep_ in ['compound', 'amod'] and token.head.pos_ == 'NOUN':
                compound_term = f"{token.text} {token.head.text}".lower()
                features.append(compound_term)
        
        return list(set(features))
    
    def _classify_constraint_type(self, sentence: str, indicators: List[str]) -> str:
        """Classify the type of constraint based on indicators and content."""
        
        sentence_lower = sentence.lower()
        
        # Priority-based classification
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
        
        # Extract numerical values with units
        num_pattern = r'\b\d+(?:\.\d+)?\s*(?:mm|cm|m|in|inch|mil|micron|μm|deg|degree|%|percent|times|x)\b'
        matches = re.findall(num_pattern, sentence, re.IGNORECASE)
        
        if matches:
            return matches[0]
        
        # Extract qualitative constraints
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
        
        # Extract technical terms using NLTK
        try:
            tokens = nltk.word_tokenize(sentence)
            pos_tags = pos_tag(tokens)
            
            # Look for technical noun phrases
            for i, (word, pos) in enumerate(pos_tags):
                if pos.startswith('NN') and len(word) > 3:
                    entities.append({
                        'text': word,
                        'label': 'TECHNICAL_TERM',
                        'type': 'nltk_noun'
                    })
        except:
            pass  # Skip if NLTK processing fails
        
        return entities
    
    def _classify_rule_type(self, sentence: str) -> str:
        """Classify the manufacturing rule type using zero-shot classification."""
        
        try:
            result = self.zero_shot_classifier(
                sentence, 
                self.manufacturing_categories,
                multi_label=False
            )
            
            # Return the most likely category
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
        """Calculate overall confidence score for the extracted rule."""
        
        score = 0.0
        
        # Base score from manufacturing relevance
        score += manufacturing_relevance * 0.3
        
        # Bonus for rule indicators
        if indicators:
            score += min(len(indicators) * 0.15, 0.3)
        
        # Bonus for semantic features
        if features:
            score += min(len(features) * 0.05, 0.2)
        
        # Bonus for named entities
        if entities:
            score += min(len(entities) * 0.03, 0.1)
        
        # Bonus for numerical values
        if re.search(r'\d+(?:\.\d+)?', sentence):
            score += 0.1
        
        # Ensure score is between 0 and 1
        return min(max(score, 0.0), 1.0)
    
    def process_document_sections(self, text: str) -> Dict[str, List[ImplicitRule]]:
        """Process document by sections to maintain context."""
        
        # Split document into sections (simple approach)
        sections = self._split_into_sections(text)
        
        results = {}
        for section_name, section_text in sections.items():
            section_rules = self.extract_implicit_rules(section_text)
            if section_rules:
                results[section_name] = section_rules
        
        return results
    
    def _split_into_sections(self, text: str) -> Dict[str, str]:
        """Split text into logical sections."""
        
        # Simple section splitting based on headers and paragraph breaks
        sections = {}
        
        # Split by double newlines (paragraphs)
        paragraphs = text.split('\n\n')
        
        current_section = "main_content"
        current_text = []
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if this looks like a header
            if len(para) < 100 and (para.isupper() or para.endswith(':')):
                # Save previous section
                if current_text:
                    sections[current_section] = '\n\n'.join(current_text)
                
                # Start new section
                current_section = para.lower().replace(' ', '_').replace(':', '')
                current_text = []
            else:
                current_text.append(para)
        
        # Add final section
        if current_text:
            sections[current_section] = '\n\n'.join(current_text)
        
        return sections

# Usage example and testing
if __name__ == "__main__":
    extractor = ImplicitRuleExtractor()
    
    # Test with random document text (no obvious manufacturing keywords)
    test_text = """
    Components should maintain appropriate spacing to prevent interference during operation.
    Materials must be selected considering thermal expansion effects in high-temperature environments.
    Assembly procedures require adequate access for maintenance and inspection activities.
    Items should be designed to facilitate easy replacement of consumable parts.
    Structures need sufficient strength to withstand expected loading conditions.
    Connections should provide reliable electrical continuity throughout the operational lifetime.
    """
    
    print("Testing Implicit Rule Extraction:")
    print("=" * 50)
    
    rules = extractor.extract_implicit_rules(test_text)
    
    for i, rule in enumerate(rules, 1):
        print(f"\nRule {i}:")
        print(f"Text: {rule.text}")
        print(f"Confidence: {rule.confidence_score:.3f}")
        print(f"Type: {rule.rule_type}")
        print(f"Constraint: {rule.constraint_type}")
        print(f"Features: {rule.semantic_features[:5]}")  # Show first 5
        print(f"Manufacturing Relevance: {rule.manufacturing_relevance:.3f}")
        print(f"Entities: {[e['text'] for e in rule.extracted_entities[:3]]}")  # Show first 3
        print("-" * 30)