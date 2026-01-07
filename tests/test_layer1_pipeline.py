import os
from pathlib import Path
import csv

from core.production_system import ProductionRuleExtractionSystem


def _make_system():
    # Use a dummy key; we won't call the LLM in these tests
    return ProductionRuleExtractionSystem(groq_api_key="dummy")


def test_enforceable_with_constraint_string(tmp_path: Path):
    system = _make_system()
    payload = {
        "filename": "test.pdf",
        "status": "success",
        "rules": [
            {
                "rule_text": "The minimum bend radius shall be the greater of 0.5×t or 0.80 mm.",
                "constraints": "bend_radius >= MAX(0.5 * material_thickness, 0.80 mm)",
                "severity": "ENFORCEABLE",
                "confidence": 0.9,
            }
        ],
        "rule_count": 1,
    }

    validated = system._apply_validation(payload)
    rule = validated["rules"][0]
    assert rule["validation_state"] == "ENFORCEABLE"
    assert rule["has_constraints"] is True

    out_csv = tmp_path / "out.csv"
    system.export_results([validated], format="csv", schema="dfm_strict", output_path=str(out_csv))
    rows = list(csv.DictReader(out_csv.open()))
    assert rows, "CSV should have at least one row"
    row = rows[0]
    assert "bend_radius" in row.get("constraints", "")
    assert row.get("severity") == "ENFORCEABLE"


def test_advisory_without_constraints(tmp_path: Path):
    system = _make_system()
    payload = {
        "filename": "test.pdf",
        "status": "success",
        "rules": [
            {
                "rule_text": "Avoid large sheet metal parts with small bent flanges.",
                "constraints": None,
                "severity": "ADVISORY",
                "confidence": 0.7,
            }
        ],
        "rule_count": 1,
    }

    validated = system._apply_validation(payload)
    rule = validated["rules"][0]
    assert rule["validation_state"] == "ADVISORY_ONLY"
    assert rule["has_constraints"] is False

    out_csv = tmp_path / "out.csv"
    system.export_results([validated], format="csv", schema="dfm_strict", output_path=str(out_csv))
    rows = list(csv.DictReader(out_csv.open()))
    assert rows, "CSV should have at least one row"
    row = rows[0]
    assert row.get("constraints", "") == ""
    assert row.get("severity") == "ADVISORY"
