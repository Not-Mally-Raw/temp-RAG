from core.enhanced_rule_engine import EnhancedRuleEngine, EnhancedConfig


def test_canonicalizer_basic(monkeypatch):
    monkeypatch.setattr(EnhancedRuleEngine, "_try_configure_llm", lambda self, m: True)
    monkeypatch.setattr(EnhancedRuleEngine, "_setup_extraction_chains", lambda self: None)
    config = EnhancedConfig(groq_api_key="test_key")
    engine = EnhancedRuleEngine(config=config)

    inputs = [
        "--- Page 1 ---\nNote: Use HEPA vacuum for cleaning\n",
        "1. Ensure alignment",
        "Use only certified components -",
        "Please recycle. For holes, diameter >= 3mm"
    ]

    outputs = [engine._canonicalize_rule_text(t) for t in inputs]
    assert all(isinstance(o, str) and o.strip() for o in outputs)
    assert outputs[0].lower().startswith("use hepa vacuum")
    assert outputs[1].endswith('.')
    assert "Please recycle" not in outputs[3]
