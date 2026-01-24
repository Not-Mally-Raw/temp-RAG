"""
Lightweight mock implementations for local development and CI.
Provides a MockEnhancedRuleEngine that can be used when GROQ API key is not set
or heavy native deps are unavailable.
"""
from typing import List, Dict, Any
import asyncio
from datetime import datetime


class MockEnhancedRuleEngine:
    """A tiny mock of the EnhancedRuleEngine for offline testing.

    Methods mimic the async interface used by the Streamlit app:
    - extract_rules_from_text(text) -> dict with 'rules' key
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    async def extract_rules_from_text(self, text: str) -> Dict[str, Any]:
        """Return a deterministic small set of mock rules based on the text.

        This is asynchronous to match the real engine's async methods.
        """
        await asyncio.sleep(0)  # yield control, keep it async-compatible

        # Very small heuristic: if there are digits present, produce a dimensional rule
        has_numbers = any(ch.isdigit() for ch in (text or ""))

        if has_numbers:
            rules = [
                {
                    'rule_text': 'Hole diameter shall be 5 ±0.1 mm.',
                    'rule_category': 'Dimensional',
                    'rule_type': 'dimensional',
                    'primary_feature': 'Hole',
                    'primary_object': 'Bracket',
                    'operator': '±',
                    'value': '5',
                    'unit': 'mm',
                    'confidence_score': 0.92,
                    'manufacturing_relevance': 0.95,
                    'extraction_method': 'mock',
                    'extracted_at': datetime.utcnow().isoformat()
                }
            ]
        else:
            rules = [
                {
                    'rule_text': 'Parts shall be free of sharp burrs and debris.',
                    'rule_category': 'Quality Control',
                    'rule_type': 'quality',
                    'primary_feature': 'Surface',
                    'primary_object': 'Part',
                    'operator': 'shall',
                    'value': None,
                    'unit': None,
                    'confidence_score': 0.78,
                    'manufacturing_relevance': 0.7,
                    'extraction_method': 'mock',
                    'extracted_at': datetime.utcnow().isoformat()
                }
            ]

        return {'rules': rules, 'document_metadata': {}, 'extraction_stats': {}, 'processing_time': 0.0}

    # Provide a sync wrapper for code that calls sync methods
    def extract_rules_from_text_sync(self, text: str) -> Dict[str, Any]:
        return asyncio.get_event_loop().run_until_complete(self.extract_rules_from_text(text))
