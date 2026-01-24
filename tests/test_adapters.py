import pytest
from core.adapters import TextLoaderAdapter, ChunkerAdapter, RuleExtractorAdapter, ExporterAdapter
from core.interfaces import DocumentMeta, Rule
from typing import Iterable, List, Dict, Any, Optional

def fake_load_document(path: str) -> dict:
    # legacy-style dict return to validate coercion
    return {"pages": 1, "text": "sample text", "metadata": {"path": path}}

def fake_chunk(document: DocumentMeta) -> Iterable[str]:
    return ["c1", "c2"]

def fake_extract(chunk: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    # legacy returns list of dicts
    return [{"id": f"{chunk}-1", "text": f"rule {chunk}", "confidence": 0.9}]

captured = {}
def fake_export(rules: Iterable[Rule], output_path: str) -> None:
    captured['out'] = list(rules)
    captured['path'] = output_path

def test_textloader_adapter_coerces_dict():
    adapter = TextLoaderAdapter(impl=fake_load_document)
    meta = adapter.load("doc.pdf")
    assert isinstance(meta, DocumentMeta)
    assert meta.text == "sample text"
    assert meta.pages == 1

def test_chunker_adapter_forwards_callable():
    adapter = ChunkerAdapter(impl=fake_chunk)
    doc = DocumentMeta(path="doc.pdf", text="x", pages=1)
    chunks = list(adapter.chunk(doc))
    assert chunks == ["c1", "c2"]

def test_rule_extractor_adapter_coerces_dicts_to_rules():
    adapter = RuleExtractorAdapter(impl=fake_extract)
    rules = adapter.extract("c1", context=None)
    assert isinstance(rules, list)
    assert rules and hasattr(rules[0], "text")
    assert rules[0].confidence == 0.9

def test_exporter_adapter_calls_callable():
    adapter = ExporterAdapter(impl=fake_export)
    r = Rule(id="1", text="r", category="T", confidence=0.5, meta={})
    adapter.export([r], "out.csv")
    assert captured["path"] == "out.csv"
    assert captured["out"][0].text == "r"
