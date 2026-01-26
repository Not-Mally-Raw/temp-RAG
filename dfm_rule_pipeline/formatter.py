import csv
import json
import logging
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional

# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("DFM_FORMATTER")

# -------------------------------------------------
# STRICT CONSTANTS
# -------------------------------------------------
ALLOWED_RULE_CATEGORIES = {
    "Assembly", "Additive", "Die Cast", "Drill", "General",
    "Injection Moulding", "Mill", "Model", "Sheetmetal",
    "SMForm", "Tubing", "Turn", "Tolerance"
}

REQUIRED_RULE_KEYS = [
    "RuleCategory", "Name", "Feature1", "Feature2",
    "Object1", "Object2", "Constraints"
]

REQUIRED_CONSTRAINT_KEYS = [
    "ExpName", "Operator", "Recom"
]

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def s(v):
    return v.strip() if isinstance(v, str) else ""

def normalize_name(text: str) -> str:
    # Title Case, Alphanumeric + Space only
    return "".join(c for c in text if c.isalnum() or c == " ").strip().title()

def parse_equation(eq: str):
    """
    Splits 'LHS >= RHS' into (LHS, Operator, RHS_Value).
    Handles 'Distance(...) >= 8 * Thickness' logic.
    """
    if not eq: return "", "", ""
    
    # regex for operators: >=, <=, ==, >, <
    match = re.search(r"(.*?)\s*(>=|<=|==|>|<)\s*(.*)", eq)
    if not match:
        return eq, "", "" # Fallback: put whole string in ExpName
        
    lhs = match.group(1).strip()
    op = match.group(2).strip()
    rhs = match.group(3).strip()

    # Try to extract a pure number from RHS if possible, otherwise keep string
    # (Your schema example showed 'Recom': 3.0 for ratios)
    
    # Case: "3 * ModuleParams.Thickness" -> extract 3.0
    mult_match = re.search(r"(\d+(?:\.\d+)?)\s*\*", rhs)
    if mult_match:
        val = float(mult_match.group(1))
        # Modify LHS to include the variable if needed, but for strict schema
        # usually ExpName="Distance/Thickness" and Recom=3.0
        # For now, we will put the full normalized expression in ExpName if it's complex.
        # But let's stick to the previous working logic:
        # If it is a ratio, ExpName usually needs the denominator.
        return lhs, op, val

    # Case: Pure Number
    try:
        val = float(rhs)
        return lhs, op, val
    except:
        return lhs, op, rhs

# -------------------------------------------------
# FACTORY METHODS (Strict Schema Generation)
# -------------------------------------------------

def _empty_rule(category: str, rule_text: str) -> Dict:
    return {
        "RuleCategory": category,
        "Name": normalize_name(rule_text),
        "Feature1": "",
        "Feature2": "",
        "Object1": "",
        "Object2": "",
        "Constraints": []
    }

def _make_constraint(expname, operator, recom) -> Dict:
    """
    Generates a constraint dict with STRICT PascalCase keys.
    """
    return {
        "ExpName": str(expname) if expname is not None else "",
        "Operator": str(operator) if operator is not None else "",
        "Recom": recom if recom is not None else "" 
    }

def _validate_strict(rule: Dict) -> None:
    """
    Final safety check. Raises error if schema is violated.
    """
    for k in REQUIRED_RULE_KEYS:
        if k not in rule:
            raise ValueError(f"Schema Violation: Missing '{k}'")
    
    if not isinstance(rule["Constraints"], list):
        raise ValueError("Schema Violation: Constraints must be a list")
        
    for c in rule["Constraints"]:
        for k in REQUIRED_CONSTRAINT_KEYS:
            if k not in c:
                raise ValueError(f"Schema Violation: Constraint missing '{k}'")

# -------------------------------------------------
# BUILDERS
# -------------------------------------------------

def build_skipped(category, rule_text, reason):
    rule = _empty_rule(category, rule_text)
    # Skipped rules put the reason in 'Recom' (or ExpName) to be visible, 
    # but strictly they have no math constraints.
    rule["Constraints"].append(_make_constraint("", "", reason))
    return rule

def build_tolerance(category, rule_text, equation, reasoning):
    rule = _empty_rule("Tolerance", rule_text) # Force Tolerance category
    rule["Feature1"] = "Tolerance"
    rule["Constraints"].append(_make_constraint(equation, "", ""))
    if reasoning:
        # Optional: append reasoning as a secondary constraint or ignore
        pass 
    return rule

def build_geometry(category, rule_text, geo_json):
    rule = _empty_rule(category, rule_text)
    
    rule["Feature1"] = geo_json.get("relation", "Distance")
    rule["Object1"] = geo_json.get("from", "")
    rule["Object2"] = geo_json.get("to", "")
    
    rhs_raw = geo_json.get("rhs", "")
    op = geo_json.get("operator", "")
    
    # Parse the RHS to separate value from expression if needed
    lhs_parsed, op_parsed, val_parsed = parse_equation(f"Expression {op} {rhs_raw}")
    
    # Use raw RHS as ExpName if parsing is ambiguous, or map strictly
    # For robust output, we put the full RHS expression in ExpName if it's complex
    # or follow the ratio pattern if detected.
    
    # Simple mapping:
    rule["Constraints"].append(_make_constraint(rhs_raw, op, ""))
    return rule

def build_equation(category, rule_text, equation):
    rule = _empty_rule(category, rule_text)
    rule["Feature1"] = "Attribute"
    
    lhs, op, val = parse_equation(equation)
    
    # Attempt to extract Object from LHS (e.g. "Hole.Diameter")
    if "." in lhs:
        rule["Object1"] = lhs.split(".")[0]
        
    rule["Constraints"].append(_make_constraint(lhs, op, val))
    return rule

# -------------------------------------------------
# CORE LOOP
# -------------------------------------------------

def format_pipeline_csv(input_csv: Path, output_csv: Path):
    output_rows = []

    with input_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for idx, row in enumerate(reader, start=1):
            rule_text = s(row.get("RuleText"))
            decision = s(row.get("DecisionCode"))
            
            # 1. Resolve Domain/Category
            domain = s(row.get("domain"))
            category = domain if domain in ALLOWED_RULE_CATEGORIES else "General"
            
            # 2. Extract Data
            equation = s(row.get("Equation"))
            geo_raw = s(row.get("GeometryJSON"))
            reasoning = s(row.get("Reasoning")) or s(row.get("Reason")) # Handle both keys

            dfm_json = None

            try:
                # --- LOGIC BRANCHING ---
                if category == "General" and decision != "formalized":
                    # General rules that aren't strict equations are skipped
                    dfm_json = build_skipped(category, rule_text, reasoning or "Non-standard domain")
                
                elif decision == "deferred_tolerance" and equation:
                    dfm_json = build_tolerance(category, rule_text, equation, reasoning)
                    
                elif decision == "deferred_geometry" and geo_raw:
                    try:
                        geo_data = json.loads(geo_raw)
                        dfm_json = build_geometry(category, rule_text, geo_data)
                    except:
                        dfm_json = build_skipped(category, rule_text, "Geometry Parse Error")

                elif decision == "formalized" and equation:
                    dfm_json = build_equation(category, rule_text, equation)

                else:
                    # Fallback for skipped/schema gaps
                    dfm_json = build_skipped(category, rule_text, reasoning or "Skipped")

                # --- VALIDATE & SERIALIZE ---
                _validate_strict(dfm_json)
                
                output_rows.append({
                    "RuleText": rule_text,
                    "DecisionCode": decision,
                    "dfm_json": json.dumps(dfm_json, ensure_ascii=False)
                })
                
                logger.info(f"[{idx}] Success: {category} - {rule_text[:30]}...")

            except Exception as e:
                logger.error(f"[{idx}] FAILED: {e}")
                # Create a safe error rule so CSV doesn't break
                error_json = _empty_rule("General", rule_text)
                error_json["Constraints"].append(_make_constraint("", "", f"Formatting Error: {str(e)}"))
                output_rows.append({
                    "RuleText": rule_text,
                    "DecisionCode": "error",
                    "dfm_json": json.dumps(error_json)
                })

    # Write Output
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["RuleText", "DecisionCode", "dfm_json"])
        writer.writeheader()
        writer.writerows(output_rows)

    logger.info(f"✅ Formatter Finished. Wrote {len(output_rows)} rows to {output_csv}")

# -------------------------------------------------
# CLI
# -------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python formatter.py <input_csv> <output_csv>")
        sys.exit(1)

    format_pipeline_csv(Path(sys.argv[1]), Path(sys.argv[2]))