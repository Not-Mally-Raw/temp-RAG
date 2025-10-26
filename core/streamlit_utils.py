"""
Streamlit Performance Optimizations
Lightweight imports and caching for the existing RAG system
"""

import functools
import hashlib
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Simple cache decorator for streamlit performance
def streamlit_cache(maxsize: int = 128, ttl: int = 3600):
    """
    Simple caching decorator for streamlit functions.
    Args:
        maxsize: Maximum cache size
        ttl: Time to live in seconds
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_times = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key_data = str(args) + str(sorted(kwargs.items()))
            cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            current_time = time.time()
            
            # Check if cached and not expired
            if (cache_key in cache and 
                cache_key in cache_times and 
                current_time - cache_times[cache_key] < ttl):
                return cache[cache_key]
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            if len(cache) >= maxsize:
                # Remove oldest entry
                oldest_key = min(cache_times.keys(), key=lambda k: cache_times[k])
                del cache[oldest_key]
                del cache_times[oldest_key]
            
            cache[cache_key] = result
            cache_times[cache_key] = current_time
            
            return result
        
        return wrapper
    return decorator


# Lightweight text processing for streamlit
class StreamlitTextProcessor:
    """Optimized text processing for streamlit performance."""
    
    def __init__(self):
        # Minimal imports for speed
        self.chunk_cache = {}
    
    @streamlit_cache(maxsize=50)
    def chunk_text_fast(self, text: str, max_length: int = 1000) -> List[str]:
        """Fast text chunking for streamlit."""
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            
            if current_length + word_length > max_length and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    @streamlit_cache(maxsize=100)
    def extract_sentences_fast(self, text: str) -> List[str]:
        """Fast sentence extraction."""
        import re
        
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


# Streamlit-optimized rule extractor
class StreamlitRuleExtractor:
    """Lightweight rule extractor for streamlit demos."""
    
    def __init__(self):
        self.text_processor = StreamlitTextProcessor()
        self.rule_patterns = [
            r'(?:minimum|maximum|should|must|shall)\s+[^.!?]*[.!?]',
            r'(?:tolerance|clearance|dimension)\s+[^.!?]*[.!?]',
            r'(?:ensure|avoid|consider)\s+[^.!?]*[.!?]'
        ]
    
    @streamlit_cache(maxsize=20)
    def extract_rules_lightweight(self, text: str, max_rules: int = 30) -> List[Dict[str, Any]]:
        """Lightweight rule extraction for streamlit."""
        import re
        
        rules = []
        
        # Extract sentences
        sentences = self.text_processor.extract_sentences_fast(text)
        
        for sentence in sentences:
            if len(sentence) < 20 or len(sentence) > 300:
                continue
            
            # Check if it looks like a manufacturing rule
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in [
                'should', 'must', 'minimum', 'maximum', 'ensure', 'avoid',
                'tolerance', 'clearance', 'thickness', 'diameter'
            ]):
                # Simple classification
                has_measurement = bool(re.search(r'\d+\.?\d*\s*(mm|cm|inch|°)', sentence))
                has_specification = any(spec in sentence_lower for spec in [
                    'shall', 'must', 'should be', 'not exceed', 'not less than'
                ])
                
                classification = 1 if (has_measurement or has_specification) else 0
                confidence = 0.8 if has_measurement else 0.6
                
                rules.append({
                    'rule_text': sentence,
                    'classification_label': classification,
                    'confidence': confidence,
                    'has_measurement': has_measurement,
                    'has_specification': has_specification
                })
        
        # Sort by confidence and return top rules
        rules.sort(key=lambda x: x['confidence'], reverse=True)
        return rules[:max_rules]


# Memory-optimized imports for streamlit
def import_if_available(module_name: str, fallback=None):
    """Import module if available, return fallback otherwise."""
    try:
        import importlib
        return importlib.import_module(module_name)
    except ImportError:
        logger.warning(f"Module {module_name} not available, using fallback")
        return fallback


# Streamlit performance utilities
class StreamlitPerformanceMonitor:
    """Simple performance monitoring for streamlit."""
    
    def __init__(self):
        self.timings = {}
    
    def start_timing(self, operation: str):
        """Start timing an operation."""
        self.timings[operation] = time.time()
    
    def end_timing(self, operation: str) -> float:
        """End timing and return duration."""
        if operation in self.timings:
            duration = time.time() - self.timings[operation]
            del self.timings[operation]
            return duration
        return 0.0
    
    def time_operation(self, operation: str):
        """Context manager for timing operations."""
        class TimingContext:
            def __init__(self, monitor, op_name):
                self.monitor = monitor
                self.op_name = op_name
            
            def __enter__(self):
                self.monitor.start_timing(self.op_name)
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                return self.monitor.end_timing(self.op_name)
        
        return TimingContext(self, operation)


# Global instances for streamlit
_performance_monitor = StreamlitPerformanceMonitor()
_text_processor = StreamlitTextProcessor()
_rule_extractor = StreamlitRuleExtractor()

def get_streamlit_processor():
    """Get global streamlit processor."""
    return _text_processor

def get_streamlit_extractor():
    """Get global streamlit rule extractor."""
    return _rule_extractor

def get_performance_monitor():
    """Get global performance monitor."""
    return _performance_monitor


# Testing
if __name__ == "__main__":
    print("=== Testing Streamlit Optimizations ===")
    
    processor = get_streamlit_processor()
    extractor = get_streamlit_extractor()
    monitor = get_performance_monitor()
    
    # Test text
    sample_text = """
    The minimum bend radius should not be less than 1.5 times the material thickness.
    Ensure adequate clearance between moving parts. Consider thermal expansion.
    For injection molding, wall thickness should be between 1-5mm.
    Avoid sharp corners to prevent stress concentration.
    """
    
    # Test performance
    with monitor.time_operation("rule_extraction"):
        rules = extractor.extract_rules_lightweight(sample_text)
    
    duration = monitor.end_timing("rule_extraction")
    print(f"Extracted {len(rules)} rules in {duration:.3f}s")
    
    for i, rule in enumerate(rules, 1):
        print(f"{i}. [{rule['classification_label']}] {rule['rule_text'][:60]}...")
    
    # Test caching
    print("\nTesting cache performance...")
    
    # First call (no cache)
    start = time.time()
    rules1 = extractor.extract_rules_lightweight(sample_text)
    time1 = time.time() - start
    
    # Second call (cached)
    start = time.time()
    rules2 = extractor.extract_rules_lightweight(sample_text)
    time2 = time.time() - start
    
    print(f"First call: {time1:.4f}s")
    print(f"Cached call: {time2:.4f}s")
    print(f"Speedup: {time1/max(time2, 0.0001):.1f}x")
    
    print("\n=== Streamlit Optimization Test Complete ===")