# pipeline.py (replace the run_pipeline function and supporting helpers as below)
import json
import time
import os
import csv
from tqdm import tqdm
from typing import List, Union

# -------------------------------------------------------------------
# STAGE IMPORTS
# -------------------------------------------------------------------
from stages.stage1_intent_extraction import extract_intent
from stages.stage2a_schema_consistency import check_schema_consistency
from stages.stage2_schema_validation import validate_against_schema
from stages.stage2b_geometry_resolution import resolve_geometry 
from stages.stage2c_tolerance_spec import resolve_tolerance 
from stages.stage3_formalization import formalize_rule
from stages.stage4_self_validation import self_validate

# AST Engine Imports (unchanged)
from ast_engine.ast_builder import ASTBuilder
from ast_engine.ast_validator import ASTValidator

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
OUTPUT_FILE = r"output\dfm_final_results.csv"
MAX_RETRIES = 3
SLEEP_BETWEEN_RETRIES = 1

CSV_HEADERS = [
    "rule_text",
    "status",
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
def ensure_stripped_json_str(s: Union[str, dict]) -> str:
    """If dict -> dump; if string -> return stripped."""
    if isinstance(s, dict):
        return json.dumps(s, indent=2)
    if not isinstance(s, str):
        return "{}"
    return s.strip()

def clean_json_string(s: Union[str, dict]) -> str:
    """Remove fenced code blocks and return JSON-like string."""
    if isinstance(s, dict):
        return json.dumps(s)
    if not isinstance(s, str):
        return "{}"
    s = s.strip()
    if "```json" in s:
        s = s.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in s:
        s = s.split("```", 1)[1].split("```", 1)[0].strip()
    return s

def safe_load_json(raw_str: Union[str, dict], stage_name: str) -> dict:
    """
    Parse and return a dict. Raises ValueError with helpful message on failure.
    This prevents 'str' object has no attribute 'get' throughout the pipeline.
    """
    try:
        if isinstance(raw_str, dict):
            data = raw_str
        else:
            data = json.loads(clean_json_string(raw_str))
        if not isinstance(data, dict):
            raise ValueError(f"Output is {type(data)}, expected dict.")
        return data
    except Exception as e:
        raise ValueError(f"JSON Parse Error in {stage_name}: {e}")

def load_completed_rules(csv_path: str) -> set:
    completed = set()
    if not os.path.exists(csv_path):
        return completed
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rt = (row.get("rule_text") or "").strip()
                if rt:
                    completed.add(rt)
    except Exception as e:
        print(f"⚠️  Warning: Could not read checkpoint CSV ({e})")
    return completed

def append_result(row: dict):
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    file_exists = os.path.isfile(OUTPUT_FILE)
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        safe_row = {k: row.get(k, "") for k in CSV_HEADERS}
        writer.writerow(safe_row)

# -------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------
def run_pipeline(llm, rules_data: List[Union[str, dict]]):
    if isinstance(llm, list):
        print("⚠️  Warning: 'llm' was passed as a list. Unwrapping it automatically.")
        llm = llm[0]

    print(f"📂 Found {len(rules_data)} rules to process.")
    print("🚀 Starting DFM Rule Pipeline (Multi-Formalism Mode)...\n")

    completed_rules = load_completed_rules(OUTPUT_FILE)
    print(f"📂 Checkpoint loaded: {len(completed_rules)} rules already completed.\n")

    for rule in tqdm(rules_data, desc="Processing Rules"):
        # Normalize input
        if isinstance(rule, dict):
            rule_text = (rule.get("rule_text") or rule.get("text") or "").strip()
        elif isinstance(rule, str):
            rule_text = rule.strip()
        else:
            continue

        if not rule_text or rule_text in completed_rules:
            continue

        print("\n" + "=" * 60)
        print(f"🔸 Processing: {rule_text[:80]}...")

        final_result = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # ---------------------------
                # STAGE 1: Intent -> DICT (guaranteed)
                # ---------------------------
                intent = extract_intent(llm, rule_text)
                if not isinstance(intent, dict):
                    raise ValueError("Stage 1 did not return a dict")

                # Quantifiability check
                if str(intent.get("is_quantifiable", "true")).lower() == "false":
                    final_result = {
                        "rule_text": rule_text,
                        "status": "Skipped (Advisory)",
                        "reasoning": intent.get("reasoning", "")
                    }
                    break

                # ---------------------------
                # STAGE 2A: Schema Consistency (local static check)
                # ---------------------------
                consistency_raw = check_schema_consistency(intent)
                consistency = safe_load_json(consistency_raw, "Stage 2A")
                if not consistency.get("schema_consistent", False):
                    final_result = {
                        "rule_text": rule_text,
                        "status": "Failed (Schema Consistency)",
                        "reasoning": consistency.get("error"),
                        "error": consistency.get("error")
                    }
                    break

                # ---------------------------
                # STAGE 2: Schema Mapping (LLM-assisted)
                # Accepts dict or string
                # ---------------------------
                schema_raw = validate_against_schema(llm, intent)
                schema_json = safe_load_json(schema_raw, "Stage 2")

                if not schema_json.get("schema_valid", False):
                    error_msg = schema_json.get("error", "Schema Gap")

                    # Geometry branch
                    if "Deferred: Geometry Layer" in error_msg:
                        print("    🔀 Branching to Geometry Resolver...")
                        geo_raw = resolve_geometry(llm, rule_text, intent)
                        geo_json = safe_load_json(geo_raw, "Stage 2B (Geometry)")
                        if geo_json.get("geometry_valid"):
                            final_result = {
                                "rule_text": rule_text,
                                "status": "Success (Geometry)",
                                "formalism": "Geometry",
                                "rule_json": json.dumps(geo_json.get("geometry", {}), indent=2),
                                "reasoning": geo_json.get("reasoning"),
                            }
                            break
                        else:
                            final_result = {
                                "rule_text": rule_text,
                                "status": "Failed (Geometry)",
                                "error": geo_json.get("error", "Unknown Geometry Error")
                            }
                            break

                    # Tolerance branch
                    if "Deferred: Tolerance Spec" in error_msg:
                        print("    🔀 Branching to Tolerance Resolver...")
                        tol_raw = resolve_tolerance(llm, rule_text, intent)
                        tol_json = safe_load_json(tol_raw, "Stage 2C (Tolerance)")
                        if tol_json.get("tolerance_valid"):
                            final_result = {
                                "rule_text": rule_text,
                                "status": "Success (Tolerance)",
                                "formalism": "Tolerance",
                                "equation": tol_json.get("equation"),
                                "reasoning": tol_json.get("reasoning"),
                            }
                            break
                        else:
                            final_result = {
                                "rule_text": rule_text,
                                "status": "Failed (Tolerance)",
                                "error": tol_json.get("error", "Unknown Tolerance Error")
                            }
                            break

                    final_result = {
                        "rule_text": rule_text,
                        "status": "Failed (Schema)",
                        "reasoning": schema_json.get("reasoning", ""),
                        "rule_json": json.dumps(schema_json),
                        "error": error_msg
                    }
                    break

                # ---------------------------
                # STAGE 3: FORMALIZATION (LLM)
                # ---------------------------
                formal_raw = formalize_rule(llm, schema_raw)
                formal_json = safe_load_json(formal_raw, "Stage 3")

                formalism = formal_json.get("formalism", "null")
                equation = formal_json.get("equation")
                ast_data = formal_json.get("ast")

                # AST integrity if needed
                if formalism == "AST" and ast_data:
                    try:
                        ast_tree = ASTBuilder.build(ast_data)
                        allowed_vars = {'ModuleParams', 'Bend', 'Hole', 'Counterbore', 'Fastener', 'Sheet', 'Shaft', 'Surface', 'PartBody'}
                        validator = ASTValidator(allowed_variables=allowed_vars)
                        if not validator.validate(ast_tree):
                            final_result = {
                                "rule_text": rule_text,
                                "status": "Failed (AST Validation)",
                                "reasoning": formal_json.get("reasoning", ""),
                                "error": f"Invalid AST: {validator.errors}"
                            }
                            break
                    except Exception as e:
                        final_result = {
                            "rule_text": rule_text,
                            "status": "Failed (AST Build)",
                            "reasoning": formal_json.get("reasoning", ""),
                            "error": f"Malformed AST Structure: {str(e)}"
                        }
                        break

                if not ((formalism == "equation" and equation) or (formalism == "AST" and ast_data)):
                    final_result = {
                        "rule_text": rule_text,
                        "status": "Failed (Formalization)",
                        "reasoning": formal_json.get("reasoning", ""),
                        "error": "Null Equation/AST"
                    }
                    break

                # ---------------------------
                # STAGE 4: SELF VALIDATION
                # ---------------------------
                validation = self_validate(llm, formal_raw)

                if validation.get("is_valid") is True:
                    final_result = {
                        "rule_text": rule_text,
                        "status": "Success",
                        "formalism": formalism,
                        "rule_json": json.dumps(formal_json.get("rule_json", {})),
                        "equation": equation,
                        "ast": json.dumps(ast_data) if ast_data else "",
                        "reasoning": formal_json.get("reasoning", ""),
                    }
                else:
                    issues = "; ".join(validation.get("issues", []))
                    final_result = {
                        "rule_text": rule_text,
                        "status": "Review Needed",
                        "formalism": formalism,
                        "rule_json": json.dumps(formal_json.get("rule_json", {})),
                        "equation": equation,
                        "ast": json.dumps(ast_data) if ast_data else "",
                        "reasoning": formal_json.get("reasoning", ""),
                        "error": issues
                    }
                break

            except ValueError as e:
                # This indicates a JSON parsing / validation problem. Don't retry LLM on these.
                print(f"    💥 Data Error (Attempt {attempt}): {e}")
                final_result = {"rule_text": rule_text, "status": "Failed", "error": str(e)}
                break
            except Exception as e:
                # LLM/IO/network-related issues can be retried
                print(f"    💥 Logic Error (Attempt {attempt}): {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(SLEEP_BETWEEN_RETRIES)
                    continue
                final_result = {"rule_text": rule_text, "status": "Failed", "error": str(e)}
                break

        if final_result:
            append_result(final_result)

    print(f"\n✅ Pipeline complete. Results saved to: {OUTPUT_FILE}")
