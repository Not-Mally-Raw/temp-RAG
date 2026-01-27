import csv
import json
import logging
import re
import sys
from pathlib import Path

# -----------------------------
# Setup & Logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("DFM_FORMATTER")

# -----------------------------
# Domain Configuration
# -----------------------------
DOMAIN_CONFIG = {
    "sheetmetal": {
        "RuleCategory": "Sheet Metal",
        "ThickVar": "SheetMetal.Thickness", 
        "Aliases": ["ModuleParams.Thickness", "NormalThickness"]
    },
    "sheet metal": { # Added alias
        "RuleCategory": "Sheet Metal",
        "ThickVar": "SheetMetal.Thickness", 
        "Aliases": ["ModuleParams.Thickness"]
    },
    "smform": {
        "RuleCategory": "Sheet Metal Forming",
        "ThickVar": "SheetMetalForm.NominalThickness",
        "Aliases": ["ModuleParams.NormalThickness"]
    },
    "turn": {
        "RuleCategory": "Turning",
        "ThickVar": "PartBody.NominalThickness",
        "Aliases": ["ModuleParams.Thickness"]
    },
    "turning": {
        "RuleCategory": "Turning",
        "ThickVar": "PartBody.NominalThickness",
        "Aliases": ["ModuleParams.Thickness"]
    },
    "mill": {
        "RuleCategory": "Milling",
        "ThickVar": "PartBody.NominalThickness",
        "Aliases": ["ModuleParams.Thickness"]
    },
    "drill": {
        "RuleCategory": "Drilling",
        "ThickVar": "PartBody.NominalThickness",
        "Aliases": ["ModuleParams.Thickness"]
    },
    "injection moulding": {
        "RuleCategory": "Injection Molding",
        "ThickVar": "InjectionMolding.NominalThickness",
        "Aliases": ["ModuleParams.NominalThickness"]
    },
    "die cast": {
        "RuleCategory": "Die Casting",
        "ThickVar": "PartBody.NominalThickness",
        "Aliases": ["WallThickness.MinValue"] 
    },
    "additive": {
        "RuleCategory": "Additive Manufacturing",
        "ThickVar": "AMFace.MinThickness", 
        "Aliases": ["ModuleParams.NominalThickness"]
    },
    "assembly": {
        "RuleCategory": "Assembly",
        "ThickVar": "Component.Thickness",
        "Aliases": []
    },
    "tubing": {
        "RuleCategory": "Tubing",
        "ThickVar": "Tube.OuterDiameter", 
        "Aliases": ["ModuleParams.Thickness"]
    },
    "general": {
        "RuleCategory": "General",
        "ThickVar": "PartBody.NominalThickness",
        "Aliases": ["ModuleParams.Thickness"]
    }
}

# -----------------------------
# Helper Functions
# -----------------------------
def clean_str(s):
    return s.strip() if isinstance(s, str) else ""

def generate_short_name(rule_text):
    clean = re.sub(r"[^a-zA-Z0-9 ]", "", rule_text)
    words = clean.split()
    stop_words = {'should', 'must', 'is', 'are', 'shall', 'be', 'the'}
    name_words = []
    for w in words:
        if w.lower() in stop_words and len(name_words) > 3:
            break
        name_words.append(w)
    return " ".join(name_words[:7]).title()

def substitute_variables(expression, domain_cfg):
    """Swaps generic 'ModuleParams.Thickness' with domain specific vars."""
    if not expression: return expression
    
    target_var = domain_cfg["ThickVar"]
    for alias in domain_cfg["Aliases"]:
        expression = expression.replace(alias, target_var)
    
    # Generic cleanup
    expression = expression.replace("Distance(Hole, Hole)", "Distance.MinValue")
    expression = expression.replace("Distance(", "Distance.MinValue") 
    
    return expression

def normalize_algebra(lhs, operator, rhs, domain_cfg):
    thick_var = domain_cfg["ThickVar"]
    
    lhs = substitute_variables(lhs, domain_cfg)
    rhs = substitute_variables(rhs, domain_cfg)
    
    # Logic for complex functions (max, min) - preserve as string
    if "max(" in rhs.lower() or "min(" in rhs.lower() or "tolerance(" in lhs.lower():
        return lhs, operator, rhs

    # Regex to find: (number) * Thickness OR Thickness * (number)
    esc_var = re.escape(thick_var)
    pattern = rf"(\d+\.?\d*)\s*\*\s*{esc_var}|{esc_var}\s*\*\s*(\d+\.?\d*)"
    
    match = re.search(pattern, rhs)
    
    if match:
        constant = match.group(1) if match.group(1) else match.group(2)
        remainder = re.sub(pattern, "", rhs).strip()
        
        new_lhs = lhs
        if remainder:
            if remainder.startswith("+"):
                term = remainder.strip("+ ")
                new_lhs = f"({lhs} - {term})"
            elif remainder.startswith("-"):
                term = remainder.strip("- ")
                new_lhs = f"({lhs} + {term})"
            else:
                return lhs, operator, rhs

        final_exp = f"{new_lhs}/{thick_var}"
        return final_exp, operator, float(constant)

    try:
        val = float(rhs)
    except ValueError:
        val = rhs 
        
    return lhs, operator, val

# -----------------------------
# Row Processors
# -----------------------------
def parse_equation_string(eq_str):
    """
    Robustly splits an equation string into LHS, Operator, RHS.
    Uses word boundaries (\b) to avoid splitting 'MinRadius' on 'in'.
    """
    # Order matters: >= before >
    # \b ensures 'in' matches only ' in ', not 'Nominal'
    pattern = r"(>=|<=|==|!=|\bbetween\b|\bin\b|>|<)"
    
    match = re.search(pattern, eq_str)
    if match:
        operator = match.group(1)
        # Split only on the first occurrence of the found operator
        parts = eq_str.split(operator, 1)
        # Clean up the operator (remove spaces from ' in ')
        clean_op = operator.strip() 
        return parts[0].strip(), clean_op, parts[1].strip()
    return None, None, None

def process_row(row):
    status = clean_str(row.get("Status")).lower()
    decision = clean_str(row.get("DecisionCode")).lower()
    
    if status == "skipped" or status == "invalid":
        return None
    if not (row.get("GeometryJSON") or row.get("Equation") or row.get("AST")):
        return None

    # Domain Handling
    raw_domain = clean_str(row.get("domain", row.get("RuleCategory", "General"))).lower()
    # Normalize domain keys to handle cases like "Turn" vs "Turning"
    config = DOMAIN_CONFIG.get(raw_domain, DOMAIN_CONFIG["general"])
    
    lhs_raw = ""
    op_raw = ""
    rhs_raw = ""
    
    feature1 = "Attribute"
    feature2 = ""
    obj1 = ""
    obj2 = ""
    
    # Path A: GeometryJSON
    if row.get("GeometryJSON"):
        try:
            geo = json.loads(row.get("GeometryJSON"))
            obj1 = geo.get("from", "")
            obj2 = geo.get("to", "")
            feature1 = "Distance"
            
            # The 'rhs' in JSON often contains the full equation: "Distance(..) >= .."
            eq_str = geo.get("rhs", "")
            lhs_raw, op_raw, rhs_raw = parse_equation_string(eq_str)
            
            if "Distance" in lhs_raw:
                lhs_raw = "Distance.MinValue"

        except json.JSONDecodeError:
            return None 

    # Path B: Equation / AST
    elif row.get("Equation"):
        eq_str = row.get("Equation")
        
        # Special Case: Tolerance Function calls (usually don't have standard operators)
        if eq_str.strip().startswith("Tolerance("):
            # Pass through Tolerance rules as-is
            lhs_raw = eq_str
            op_raw = "Function" # Special marker or keep empty
            rhs_raw = ""
            feature1 = "Tolerance"
            obj1 = "Tolerance"
        else:
            lhs_raw, op_raw, rhs_raw = parse_equation_string(eq_str)
            
            if lhs_raw and "." in lhs_raw:
                obj1 = lhs_raw.split(".")[0]
                feature1 = obj1 

    if not lhs_raw:
        return None

    # Logic to handle the "Tolerance" case gracefully
    if op_raw == "Function":
        exp_name = lhs_raw
        final_op = "True" # Dummy operator for schema compliance
        recom = True
    else:
        exp_name, final_op, recom = normalize_algebra(lhs_raw, op_raw, rhs_raw, config)
    
    dfm_rule = {
        "RuleCategory": config["RuleCategory"],
        "Name": generate_short_name(row.get("RuleText", "")),
        "Feature1": feature1,
        "Feature2": feature2,
        "Object1": obj1,
        "Object2": obj2,
        "ExpName": exp_name,
        "Operator": final_op,
        "Recom": recom
    }
    
    return {
        "RuleText": row.get("RuleText"),
        "Status": "Success",
        "DecisionCode": decision if decision else "formalized",
        "RuleCategory": config["RuleCategory"],
        "dfm_json": json.dumps(dfm_rule) 
    }

# -----------------------------
# Execution
# -----------------------------
def run_pipeline(input_file, output_file):
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    processed_count = 0
    
    with input_path.open("r", encoding="utf-8-sig") as f_in, \
         output_path.open("w", encoding="utf-8", newline="") as f_out:
        
        reader = csv.DictReader(f_in)
        fieldnames = ["RuleText", "Status", "DecisionCode", "RuleCategory", "dfm_json"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in reader:
            result = process_row(row)
            if result:
                writer.writerow(result)
                processed_count += 1
                
    logger.info(f"Processed: {processed_count} | Output: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python formatter.py <input.csv> <output.csv>")
    else:
        run_pipeline(sys.argv[1], sys.argv[2])