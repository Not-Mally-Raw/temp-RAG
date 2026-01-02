import pytest

from core.enhanced_rule_engine import EnhancedRuleEngine, EnhancedConfig


def test_heuristic_extractor_multi_rule(monkeypatch):
    # Prevent LLM initialization and chain setup during test (we only test heuristic)
    monkeypatch.setattr(EnhancedRuleEngine, "_try_configure_llm", lambda self, m: True)
    monkeypatch.setattr(EnhancedRuleEngine, "_setup_extraction_chains", lambda self: None)

    config = EnhancedConfig(groq_api_key="test_key", enable_local_heuristic=True)
    engine = EnhancedRuleEngine(config=config)
    # Ensure required attributes used by heuristic exist
    engine.manufacturing_keywords = engine.manufacturing_keywords if hasattr(engine, 'manufacturing_keywords') else {
        'high_priority': ['minimum', 'maximum', 'thickness', 'diameter', 'radius', 'tolerance', 'clearance'],
        'manufacturing': [],
        'materials': [],
        'dimensions': []
    }

    text = "Calibrate tooling before use. Ensure alignment. Use HEPA vacuum for cleaning. For holes, diameter should be >= 3mm."

    candidates = engine._heuristic_extract_from_chunk(text, None)
    assert isinstance(candidates, list)
    assert len(candidates) >= 3
    assert any('Calibrate' in c.rule_text for c in candidates)
    assert any('diameter' in c.rule_text.lower() for c in candidates)
