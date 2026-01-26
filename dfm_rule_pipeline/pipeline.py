import json
import os
import csv
from tqdm import tqdm

from stages.stage1_intent_extraction import extract_intent
from stages.stage2_rule_resolution import resolve_rule_category_and_domain
from stages.stage2b_geometry_resolution import resolve_geometry
from stages.stage2c_tolerance_spec import resolve_tolerance
from stages.stage3_formalization import formalize_rule
from stages.stage4_self_validation import self_validate
from stages.stage3_attribute_formalization import formalize_attribute_rule
from stages.stage2_attribute_resolution import requires_ast
from ast_engine.ast_validator import ASTValidator
from schema.ast_schema import serialize_ast

# 1. NEW IMPORT
from schema.feature_schema import features_dict 

OUTPUT_FILE = "output/dfm_results.csv"

# Added "domain" to headers
CSV_HEADERS = [
    "rule_text", "status", "resolution_status", "formalism", 
    "rule_json", "equation", "ast", "reasoning", "error", "domain"
]

def append_result(row: dict):
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    exists = os.path.isfile(OUTPUT_FILE)
    safe = {}
    for k in CSV_HEADERS:
        v = row.get(k, "")
        if isinstance(v, (dict, list)):
            safe[k] = json.dumps(v)
        elif v is None:
            safe[k] = ""
        else:
            safe[k] = str(v)
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if not exists: writer.writeheader()
        writer.writerow(safe)

INTRINSIC_ATTRS = {"width", "height", "depth", "radius", "diameter", "thickness", "length", "angle"}

def is_intrinsic_dimension(intent: dict) -> bool:
    attrs = intent.get("mentioned_attributes", [])
    return any(a.lower() in INTRINSIC_ATTRS for a in attrs)

def run_pipeline(llm, rules_data):
    print("🔥 ENTERED run_pipeline")

    for entry in tqdm(rules_data, desc="Processing Rules"):
        rule_text = entry.get("rule_text") if isinstance(entry, dict) else str(entry)
        if not rule_text or not rule_text.strip(): continue
        rule_text = rule_text.strip()

        try:
            # STAGE 1
            intent = extract_intent(llm, rule_text)
            rule_intent = intent["rule_intent"]

            if not rule_intent.get("is_quantifiable", False):
                append_result({
                    "rule_text": rule_text,
                    "status": "Skipped",
                    "resolution_status": "skipped",
                    "reasoning": intent.get("reasoning")
                })
                continue

            # STAGE 2
            # Updated to pass rule_text for keyword matching
            resolution = resolve_rule_category_and_domain(intent, rule_text)
            category = resolution["rule_category"]
            intent["domain"] = resolution["primary_domain"]

            if category == "Geometry" and is_intrinsic_dimension(intent):
                category = "Attribute"

            # 2. RESOLVE SCHEMA TEXT
            # Default to 'Sheetmetal' or 'General' if domain is missing/unknown to allow fallback
            domain_key = intent["domain"] if intent["domain"] in features_dict else "Sheetmetal"
            schema_text = features_dict.get(domain_key, "")

            # STAGE 2b: Geometry
            if category == "Geometry":
                geo = resolve_geometry(llm, rule_text, intent)
                append_result({
                    "rule_text": rule_text,
                    "status": "Deferred",
                    "resolution_status": "deferred_geometry",
                    "formalism": "Geometry",
                    "rule_json": geo.get("geometry"),
                    "reasoning": geo.get("reasoning"),
                    "domain": intent["domain"]
                })
                continue

            # STAGE 2c: Tolerance
            if category == "Tolerance":
                tol = resolve_tolerance(llm, rule_text, intent)
                append_result({
                    "rule_text": rule_text,
                    "status": "Deferred",
                    "resolution_status": "deferred_tolerance",
                    "formalism": "Tolerance" if tol.get("tolerance_valid") else None,
                    "equation": tol.get("equation"),
                    "reasoning": tol.get("reasoning"),
                    "domain": intent["domain"]
                })
                continue

            # STAGE 3: Attribute
            if category == "Attribute":
                # 3. PASS REAL SCHEMA TEXT
                result = formalize_attribute_rule(
                    llm,
                    rule_text,
                    intent,
                    schema_context=schema_text  # <--- NOW PASSING REAL DATA
                )

                if result.get("formalism") == "equation":
                    append_result({
                        "rule_text": rule_text,
                        "status": "Success",
                        "resolution_status": "formalized",
                        "formalism": "equation",
                        "equation": result["equation"],
                        "reasoning": result["reasoning"],
                        "domain": intent["domain"]
                    })
                    continue

                # AST fallback (omitted for brevity, keeping your existing logic...)
                validator = ASTValidator(allowed_variables={"ModuleParams", "Bend", "Hole", "Slot", "Emboss", "Counterbore"})
                if result.get("formalism") == "AST" and validator.validate(result["ast"]):
                    append_result({
                        "rule_text": rule_text,
                        "status": "Success",
                        "resolution_status": "formalized",
                        "formalism": "AST",
                        "ast": serialize_ast(result["ast"]),
                        "reasoning": result["reasoning"],
                        "domain": intent["domain"]
                    })
                    continue
                
                append_result({
                    "rule_text": rule_text,
                    "status": "Deferred",
                    "resolution_status": "deferred_attribute",
                    "reasoning": result.get("reasoning"),
                    "domain": intent["domain"]
                })
                continue

            # Fallback (Generic Formalizer)
            # ... (Rest of your fallback logic)

        except Exception as e:
            append_result({
                "rule_text": rule_text,
                "status": "Review Needed",
                "resolution_status": "failed",
                "error": str(e)
            })

    print("✅ Pipeline complete:", OUTPUT_FILE)