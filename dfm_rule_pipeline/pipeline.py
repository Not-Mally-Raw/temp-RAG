# pipeline.py
import json
import time
import os
import csv
from tqdm import tqdm
from typing import List, Union

from stages.stage1_intent_extraction import extract_intent
from stages.stage2a_schema_consistency import check_schema_consistency
from stages.stage2_schema_validation import validate_against_schema
from stages.stage2b_geometry_resolution import resolve_geometry
from stages.stage2c_tolerance_spec import resolve_tolerance
from stages.stage3_formalization import formalize_rule
from stages.stage4_self_validation import self_validate

from ast_engine.ast_builder import ASTBuilder
from ast_engine.ast_validator import ASTValidator

OUTPUT_FILE = r"output\dfm_final_results.csv"
MAX_RETRIES = 3
SLEEP_BETWEEN_RETRIES = 1

CSV_HEADERS = [
    "rule_text",
    "status",
    "resolution_status",
    "formalism",
    "rule_json",
    "equation",
    "ast",
    "reasoning",
    "error"
]

# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def clean_json_string(s):
    if isinstance(s, dict):
        return json.dumps(s)
    s = s.strip()
    if "```" in s:
        s = s.split("```")[1]
    return s.strip()

def safe_load_json(raw, stage):
    try:
        if isinstance(raw, dict):
            return raw
        return json.loads(clean_json_string(raw))
    except Exception as e:
        raise ValueError(f"{stage} JSON parse error: {e}")

def append_result(row: dict):
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    exists = os.path.isfile(OUTPUT_FILE)
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_HEADERS})

# -------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------
def run_pipeline(llm, rules_data: List[Union[str, dict]]):

    for rule in tqdm(rules_data, desc="Processing Rules"):

        rule_text = rule if isinstance(rule, str) else rule.get("rule_text", "")
        rule_text = rule_text.strip()
        if not rule_text:
            continue

        final = None

        for attempt in range(MAX_RETRIES):
            try:
                # ---------------- STAGE 1 ----------------
                intent = extract_intent(llm, rule_text)

                if intent.get("is_quantifiable") is False:
                    final = {
                        "rule_text": rule_text,
                        "status": "Skipped (Advisory)",
                        "resolution_status": "skipped",
                        "reasoning": intent.get("reasoning")
                    }
                    break

                # ---------------- STAGE 2A ----------------
                consistency = safe_load_json(
                    check_schema_consistency(intent), "Stage 2A"
                )
                if not consistency.get("schema_consistent"):
                    final = {
                        "rule_text": rule_text,
                        "status": "Failed (Schema Gap)",
                        "resolution_status": "failed_schema",
                        "error": consistency.get("error")
                    }
                    break

                # ---------------- STAGE 2 ----------------
                schema = safe_load_json(
                    validate_against_schema(llm, intent), "Stage 2"
                )

                if not schema.get("schema_valid"):
                    err = schema.get("error", "")

                    if "Deferred: Geometry Layer" in err:
                        geo = safe_load_json(
                            resolve_geometry(llm, rule_text, intent), "Stage 2B"
                        )
                        if geo.get("geometry_valid"):
                            final = {
                                "rule_text": rule_text,
                                "status": "Success (Geometry)",
                                "resolution_status": "resolved",
                                "formalism": "Geometry",
                                "rule_json": json.dumps(geo.get("geometry")),
                                "reasoning": geo.get("reasoning")
                            }
                        else:
                            final = {
                                "rule_text": rule_text,
                                "status": "Failed (Schema Gap)",
                                "resolution_status": "failed_schema",
                                "error": geo.get("error")
                            }
                        break

                    if "Deferred: Tolerance Spec" in err:
                        tol = safe_load_json(
                            resolve_tolerance(llm, rule_text, intent), "Stage 2C"
                        )
                        if tol.get("tolerance_valid"):
                            final = {
                                "rule_text": rule_text,
                                "status": "Success (Tolerance)",
                                "resolution_status": "resolved",
                                "formalism": "Tolerance",
                                "equation": tol.get("equation"),
                                "reasoning": tol.get("reasoning")
                            }
                        else:
                            final = {
                                "rule_text": rule_text,
                                "status": "Failed (Schema Gap)",
                                "resolution_status": "failed_schema",
                                "error": tol.get("error")
                            }
                        break

                    final = {
                        "rule_text": rule_text,
                        "status": "Failed (Schema Gap)",
                        "resolution_status": "failed_schema",
                        "error": err
                    }
                    break

                # ---------------- STAGE 3 ----------------
                formal = safe_load_json(
                    formalize_rule(llm, schema), "Stage 3"
                )

                # ---------------- STAGE 4 ----------------
                validation = self_validate(llm, formal)

                if validation.get("is_valid"):
                    final = {
                        "rule_text": rule_text,
                        "status": "Success",
                        "resolution_status": "resolved",
                        "formalism": formal.get("formalism"),
                        "equation": formal.get("equation"),
                        "ast": json.dumps(formal.get("ast")) if formal.get("ast") else "",
                        "reasoning": formal.get("reasoning")
                    }
                else:
                    final = {
                        "rule_text": rule_text,
                        "status": "Review Needed",
                        "resolution_status": "failed_schema",
                        "error": "; ".join(validation.get("issues", []))
                    }
                break

            except Exception as e:
                final = {
                    "rule_text": rule_text,
                    "status": "Failed (Schema Gap)",
                    "resolution_status": "failed_schema",
                    "error": str(e)
                }
                break

        if final:
            append_result(final)

    print("✅ Pipeline complete.")
