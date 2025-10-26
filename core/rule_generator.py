"""
Optimized Manufacturing Rule Generator
Streamlined for Streamlit performance and HCL dataset format compatibility
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
import hashlib
import time
from pathlib import Path
import logging

# Lightweight imports for streamlit performance
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class StandardizedRule:
    """Standardized rule format matching HCL classification dataset."""
    rule_text: str
    classification_label: int  # 0 or 1
    confidence: float = 0.0
    source_document: str = ""
    extraction_method: str = ""
    manufacturing_category: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'rule_text': self.rule_text,
            'classification_label': self.classification_label,
            'confidence': self.confidence,
            'source_document': self.source_document,
            'extraction_method': self.extraction_method,
            'manufacturing_category': self.manufacturing_category
        }


class OptimizedRuleClassifier:
    """Lightweight rule classifier for streamlit performance."""
    
    def __init__(self):
        self.manufacturing_keywords = {
            # High-value manufacturing terms (label 1)
            'high_value': [
                'minimum', 'maximum', 'thickness', 'diameter', 'radius', 'tolerance',
                'clearance', 'dimension', 'specification', 'requirement', 'standard',
                'should be', 'must be', 'shall be', 'not less than', 'not exceed',
                'mm', 'inch', 'degree', 'surface finish', 'material', 'strength'
            ],
            # General guidelines (label 0)
            'general': [
                'consideration', 'ensure', 'avoid', 'reduce', 'improve', 'optimize',
                'recommended', 'suggested', 'advisable', 'preferred', 'consider',
                'important', 'necessary', 'appropriate', 'adequate', 'proper'
            ]
        }
        
        # Precompile regex patterns for speed
        self.measurement_pattern = re.compile(r'\d+\.?\d*\s*(mm|cm|m|inch|in|°|deg)')
        self.specification_pattern = re.compile(r'(minimum|maximum|min|max|shall|must|should)\s+(be|not\s+exceed|not\s+less\s+than)')
        
    def classify_rule_fast(self, rule_text: str) -> Tuple[int, float]:
        """
        Fast classification for streamlit responsiveness.
        Returns (label, confidence)
        """
        text_lower = rule_text.lower()
        
        # Quick scoring
        score = 0
        confidence = 0.5
        
        # Check for measurements (strong indicator of specific rules)
        if self.measurement_pattern.search(rule_text):
            score += 3
            confidence += 0.2
        
        # Check for specification language
        if self.specification_pattern.search(text_lower):
            score += 2
            confidence += 0.15
        
        # Count keyword matches
        high_value_matches = sum(1 for kw in self.manufacturing_keywords['high_value'] if kw in text_lower)
        general_matches = sum(1 for kw in self.manufacturing_keywords['general'] if kw in text_lower)
        
        score += high_value_matches * 1
        score -= general_matches * 0.5
        
        # Length-based adjustment (specific rules tend to be more detailed)
        if len(rule_text) > 100:
            score += 0.5
        
        # Final classification
        label = 1 if score > 2 else 0
        confidence = min(0.95, max(0.1, confidence + abs(score) * 0.1))
        
        return label, confidence


class StreamlitOptimizedProcessor:
    """
    Optimized document processor for streamlit demos.
    Focus: Speed, memory efficiency, responsive UI.
    """
    
    def __init__(self, use_embeddings: bool = False):
        self.classifier = OptimizedRuleClassifier()
        self.use_embeddings = use_embeddings and EMBEDDINGS_AVAILABLE
        self.embedding_model = None
        
        # Manufacturing categories for organization
        self.categories = [
            "Additive Manufacturing", "Sheet Metal", "Machining", 
            "Injection Molding", "General Design", "Assembly"
        ]
        
        # Lightweight text patterns for rule extraction
        self.rule_patterns = [
            r'(?:The|A|An)\s+(?:minimum|maximum)[^.!?]*[.!?]',
            r'(?:Should|Must|Shall|Cannot|Should not)[^.!?]*[.!?]',
            r'(?:Ensure|Avoid|Consider|Maintain)[^.!?]*[.!?]',
            r'(?:Thickness|Diameter|Radius|Clearance)[^.!?]*[.!?]',
            r'(?:Material|Surface|Tolerance)[^.!?]*[.!?]'
        ]
        
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.rule_patterns]
    
    def load_embeddings_if_needed(self):
        """Lazy load embeddings for memory efficiency."""
        if self.use_embeddings and self.embedding_model is None:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, lightweight model
            except Exception as e:
                logger.warning(f"Could not load embeddings: {e}")
                self.use_embeddings = False
    
    def extract_rules_fast(self, text: str, source_doc: str = "") -> List[StandardizedRule]:
        """
        Fast rule extraction optimized for streamlit.
        """
        rules = []
        
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Extract sentences that match manufacturing rule patterns
        potential_rules = set()
        
        # Pattern-based extraction
        for pattern in self.compiled_patterns:
            matches = pattern.findall(text)
            potential_rules.update(matches)
        
        # Sentence-based extraction as fallback
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 30 and 
                any(kw in sentence.lower() for kw in ['should', 'must', 'minimum', 'maximum', 'ensure', 'avoid'])):
                potential_rules.add(sentence)
        
        # Process and classify rules
        for rule_text in potential_rules:
            if len(rule_text) < 20 or len(rule_text) > 500:
                continue
                
            # Clean rule text
            rule_text = re.sub(r'\s+', ' ', rule_text).strip()
            
            # Classify rule
            label, confidence = self.classifier.classify_rule_fast(rule_text)
            
            # Determine category
            category = self._categorize_rule(rule_text)
            
            rule = StandardizedRule(
                rule_text=rule_text,
                classification_label=label,
                confidence=confidence,
                source_document=source_doc,
                extraction_method="pattern_based",
                manufacturing_category=category
            )
            
            rules.append(rule)
        
        # Remove duplicates and sort by confidence
        unique_rules = self._deduplicate_rules(rules)
        return sorted(unique_rules, key=lambda x: x.confidence, reverse=True)
    
    def _categorize_rule(self, rule_text: str) -> str:
        """Fast categorization of rules."""
        text_lower = rule_text.lower()
        
        category_keywords = {
            "Additive Manufacturing": ['additive', 'printing', '3d', 'layer', 'support'],
            "Sheet Metal": ['sheet', 'bend', 'forming', 'stamping', 'brake'],
            "Machining": ['machining', 'drilling', 'milling', 'turning', 'cutting'],
            "Injection Molding": ['molding', 'injection', 'mold', 'plastic', 'resin'],
            "Assembly": ['assembly', 'fastener', 'joint', 'connection', 'bolt'],
            "General Design": []  # Default
        }
        
        for category, keywords in category_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return category
        
        return "General Design"
    
    def _deduplicate_rules(self, rules: List[StandardizedRule]) -> List[StandardizedRule]:
        """Fast deduplication using text similarity."""
        if not rules:
            return []
        
        unique_rules = []
        seen_hashes = set()
        
        for rule in rules:
            # Create a hash of normalized text
            normalized = re.sub(r'\W+', '', rule.rule_text.lower())
            text_hash = hashlib.md5(normalized.encode()).hexdigest()[:8]
            
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_rules.append(rule)
        
        return unique_rules
    
    def process_document_optimized(self, 
                                  text: str, 
                                  filename: str = "",
                                  max_rules: int = 50) -> List[StandardizedRule]:
        """
        Optimized document processing for streamlit demos.
        Returns top rules by confidence.
        """
        start_time = time.time()
        
        # Extract rules
        rules = self.extract_rules_fast(text, filename)
        
        # Limit number of rules for performance
        if len(rules) > max_rules:
            rules = rules[:max_rules]
        
        processing_time = time.time() - start_time
        logger.info(f"Processed {filename} in {processing_time:.2f}s, extracted {len(rules)} rules")
        
        return rules
    
    def batch_process_documents(self, 
                               documents: List[Tuple[str, str]], 
                               max_rules_per_doc: int = 30) -> pd.DataFrame:
        """
        Batch process multiple documents for streamlit efficiency.
        Returns DataFrame in HCL dataset format.
        """
        all_rules = []
        
        for text, filename in documents:
            doc_rules = self.process_document_optimized(text, filename, max_rules_per_doc)
            all_rules.extend(doc_rules)
        
        # Convert to DataFrame
        if all_rules:
            rule_data = [rule.to_dict() for rule in all_rules]
            df = pd.DataFrame(rule_data)
            
            # Reorder columns to match HCL format
            df = df[['rule_text', 'classification_label', 'confidence', 
                    'source_document', 'manufacturing_category', 'extraction_method']]
        else:
            df = pd.DataFrame(columns=['rule_text', 'classification_label'])
        
        return df
    
    def export_hcl_format(self, rules_df: pd.DataFrame, output_path: str):
        """Export rules in HCL classification dataset format."""
        # Keep only essential columns for HCL format
        hcl_df = rules_df[['rule_text', 'classification_label']].copy()
        
        # Save in Excel format matching original
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            hcl_df.to_excel(writer, sheet_name='Sheet1', index=False)
        
        logger.info(f"Exported {len(hcl_df)} rules to {output_path}")
        return output_path


# Streamlit-optimized caching and utilities
class StreamlitCache:
    """Simple caching for streamlit performance."""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
    
    def get(self, key: str) -> Any:
        return self.cache.get(key)
    
    def set(self, key: str, value: Any):
        if len(self.cache) >= self.max_size:
            # Remove oldest item
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = value
    
    def clear(self):
        self.cache.clear()


# Global instances for streamlit
_processor_cache = StreamlitCache()
_global_processor = None

def get_optimized_processor() -> StreamlitOptimizedProcessor:
    """Get cached processor instance for streamlit."""
    global _global_processor
    if _global_processor is None:
        _global_processor = StreamlitOptimizedProcessor(use_embeddings=False)  # Fast mode
    return _global_processor


# Example usage and testing
if __name__ == "__main__":
    print("=== Testing Optimized Rule Generator ===")
    
    processor = StreamlitOptimizedProcessor()
    
    # Test with sample manufacturing text
    sample_text = """
    The minimum bend radius of a metal sheet should not be less than 1.5 times the material thickness.
    Ensure adequate clearance between moving parts. Consider the thermal expansion of materials.
    For injection molding, the wall thickness should be between 1-5mm depending on the material.
    Avoid sharp corners in plastic parts to prevent stress concentration.
    """
    
    # Extract and classify rules
    rules = processor.extract_rules_fast(sample_text, "sample_doc.txt")
    
    print(f"Extracted {len(rules)} rules:")
    for i, rule in enumerate(rules, 1):
        print(f"{i}. [{rule.classification_label}] {rule.rule_text[:80]}...")
        print(f"   Confidence: {rule.confidence:.2f}, Category: {rule.manufacturing_category}")
    
    # Test batch processing
    documents = [(sample_text, "test_doc.txt")]
    df = processor.batch_process_documents(documents)
    
    print(f"\nDataFrame shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    print("\nSample output:")
    print(df.head(3).to_string())