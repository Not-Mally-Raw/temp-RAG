"""
Schema-aware DFM rule normalization using features.py
Maps extracted JSON rules → canonical DFM format
"""

import json
import csv
import os
from typing import Dict, Any, List
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from features import features_dict
from dotenv import load_dotenv

load_dotenv()
# ============================================================
# 1. CONFIGURATION
# ============================================================

GROQ_MODEL = "qwen/qwen3-32b"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

INPUT_JSON_PATH = "output/Design For Manufacturability Guidelines - Sheetmetal.json"
OUTPUT_CSV_PATH = "/opt/anaconda3/RAG-RuleSync-Rules-Consolidated/output/dfm_normalized_rules.csv"

TEMPERATURE = 0.0
MAX_TOKENS = 2048


# ============================================================
# 2. RULE TYPE MAPPING (Handle Extraction → Schema Mismatches)
# ============================================================

RULE_TYPE_MAPPING = {
    "Sheetmetal": "Sheetmetal",
    "SMForm": "SMForm",
    "Injection Moulding": "Injection Moulding",
    "Additive": "Additive",
    "Die Cast": "Die Cast",
    "Mill": "Mill",
    "Drill": "Drill",
    "Turn": "Turn",
    "Tubing": "Tubing",
    "Assembly": "Assembly",
    "Model": "Model",
    "General": "General",
    
    # Handle variations/errors
    "Plating": "General",
    "sheet metal": "Sheetmetal",
    "injection molding": "Injection Moulding",
}


def get_schema_for_rule(rule_type: str) -> str:
    """Get features.py schema for a given rule type."""
    mapped_type = RULE_TYPE_MAPPING.get(rule_type, "General")
    schema = features_dict.get(mapped_type, "")
    return schema


# ============================================================
# 3. DFM NORMALIZATION PROMPT TEMPLATE
# ============================================================

def build_normalization_prompt(rule_type: str) -> str:
    """Build schema-aware normalization prompt."""
    
    schema = get_schema_for_rule(rule_type)
    
    prompt = f"""
═══════════════════════════════════════════════════════════
DFM RULE NORMALIZATION ENGINE
═══════════════════════════════════════════════════════════

You normalize extracted manufacturing rules (JSON format) into canonical DFM structured format.

RULE TYPE: {rule_type}

VALID SCHEMA (Use ONLY these Objects.Attributes):
{schema}

═══════════════════════════════════════════════════════════
INPUT FORMAT
═══════════════════════════════════════════════════════════

You will receive a JSON object with:
{{
  "rule_text": "Complete verbatim rule",
  "rule_type": "{rule_type}",
  "applicability_constraints": {{
    "material": "material name or 'any'",
    "process": "process name or 'any'",
    "feature": "feature type or 'any'",
    "location": "location context or 'any'"
  }},
  "dimensional_constraints": ["param: op value unit", ...],
  "relational_constraints": ["relationship", ...]
}}

═══════════════════════════════════════════════════════════
OUTPUT FORMAT (DFM Structured Text - EXACT MATCH REQUIRED)
═══════════════════════════════════════════════════════════

RuleCategory= {rule_type}
Name = <short descriptive name based on rule>

Feature1 = <primary feature from applicability_constraints.feature>
Feature2 = <secondary feature if any, else empty string>
Object1 = <primary object from applicability_constraints.location>
Object2 = <secondary object if any, else empty string>

[If applicability_constraints.material != "any":]
Constraint : ExpName = PartBody.Material , Operator = == , Value = <material>
{{
    <constraints below>
}}

[For each dimensional/relational constraint:]
ExpName = <Object.Attribute from schema>
Operator = <>=, <=, ==, between, in>
Recom = <numeric value or range>

═══════════════════════════════════════════════════════════
MAPPING RULES (CRITICAL)
═══════════════════════════════════════════════════════════

1. RuleCategory: Always use "{rule_type}"

2. Name: Generate descriptive name from rule_text
   - Pattern: "Feature To Object Distance" OR "Recommended Feature Parameters"
   - Examples: "Counterbore to Edge Distance", "Minimum Bend Radius"

3. Feature1/Feature2:
   - Map from applicability_constraints.feature
   - If feature = "counterbore" → Feature1 = "Distance" (if rule about spacing)
   - If feature = "bend" → Feature1 = "Bend"
   - Use empty string "" for Feature2 unless multiple features involved

4. Object1/Object2:
   - Extract from applicability_constraints.location
   - "between counterbores" → Object1="CBHole", Object2="CBHole"
   - "near edge" → Object1="CBHole", Object2="PartEdge"
   - "near bend" → Object1="Feature", Object2="Bend"
   - Use empty string "" if no spatial relationship

5. ExpName Construction:
   - For distance rules: "Distance.MinValue/SheetMetal.Thickness" (ratios)
   - For feature properties: "Feature.Attribute" (e.g., "Bend.Radius")
   - For formulas: "Distance.MinValue -( constant*Feature.Attribute )" 
   - ONLY use Objects.Attributes from the schema above

6. Operator:
   - ">=" for "at least", "minimum"
   - "<=" for "at most", "maximum"
   - "==" for "equal to", "should be"
   - "between" for ranges
   - "in" for list membership

7. Recom:
   - Extract numeric value from dimensional_constraints
   - For ranges: "0.5:1.5"
   - For ratios: "4.5" (from "4.5 times thickness")
   - For lists: "1-64 UNC, 1-72 UNF"

8. Material Constraints:
   - If applicability_constraints.material != "any"
   - Wrap all ExpName/Operator/Recom in Constraint block
   - Format: Constraint : ExpName = PartBody.Material , Operator = == , Value = <material>

═══════════════════════════════════════════════════════════
EXAMPLES ({rule_type} specific)
═══════════════════════════════════════════════════════════

[PLACEHOLDER - Will inject matching examples from DFXRuleSample]

═══════════════════════════════════════════════════════════
STRICT POLICIES
═══════════════════════════════════════════════════════════

✓ DO: Copy rule_text verbatim for understanding
✓ DO: Use ONLY Objects.Attributes from schema above
✓ DO: Extract ALL numeric constraints
✓ DO: Preserve formulas exactly from relational_constraints
✓ DO: Use natural spacing and line breaks as shown in examples
✓ DO: Return ONLY the DFM formatted text (no JSON, no markdown)

✗ DON'T: Invent Object.Attribute combinations not in schema
✗ DON'T: Add constraints not in original rule
✗ DON'T: Paraphrase rule content
✗ DON'T: Use dot notation in Name field
✗ DON'T: Include explanatory text or reasoning

OUTPUT ONLY THE DFM FORMATTED TEXT BELOW:
""".strip()
    
    return prompt


# ============================================================
# 4. LLM INITIALIZATION
# ============================================================

def initialize_llm() -> ChatGroq:
    """Initialize Groq LLM client."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set in environment variables.")
    
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )


# ============================================================
# 5. RULE LOADING
# ============================================================

def load_rules_from_json(path: str) -> List[Dict[str, Any]]:
    """Load extracted rules from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if "rules" in data:
        return data["rules"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("Unsupported JSON structure for rules input.")


# ============================================================
# 6. SINGLE RULE NORMALIZATION
# ============================================================

def normalize_rule(llm: ChatGroq, rule: Dict[str, Any]) -> str:
    """Normalize a single rule using schema-aware prompt."""
    
    rule_type = rule.get("rule_type", "General")
    system_prompt = build_normalization_prompt(rule_type)
    
    rule_payload = json.dumps(rule, indent=2)
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""
Normalize the following rule into DFM format.

RULE JSON:
{rule_payload}

OUTPUT ONLY THE DFM FORMATTED TEXT (no JSON wrapper):
""")
    ]
    
    try:
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"


# ============================================================
# 7. MAIN PIPELINE
# ============================================================

def run_normalization_pipeline():
    """Execute full normalization pipeline."""
    
    print("🚀 Starting DFM Normalization Pipeline...")
    print(f"📂 Input: {INPUT_JSON_PATH}")
    print(f"📂 Output: {OUTPUT_CSV_PATH}\n")
    
    # Load rules
    rules = load_rules_from_json(INPUT_JSON_PATH)
    print(f"✅ Loaded {len(rules)} rules\n")
    
    # Count rule types
    rule_type_counts = {}
    for rule in rules:
        rt = rule.get("rule_type", "Unknown")
        rule_type_counts[rt] = rule_type_counts.get(rt, 0) + 1
    
    print("📊 Rule Type Distribution:")
    for rt, count in sorted(rule_type_counts.items()):
        mapped = RULE_TYPE_MAPPING.get(rt, "General")
        print(f"   {rt}: {count} rules → schema: {mapped}")
    print()
    
    # Initialize LLM
    llm = initialize_llm()
    print("✅ LLM initialized\n")
    
    # Create output directory
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    
    # Process rules
    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["original_rule_json", "dfm_format"])
        
        for idx, rule in enumerate(rules, start=1):
            print(f"[{idx}/{len(rules)}] Processing: {rule.get('rule_type', 'Unknown')} - {rule.get('rule_text', '')[:60]}...")
            
            try:
                dfm_output = normalize_rule(llm, rule)
            except Exception as e:
                dfm_output = f"ERROR: {str(e)}"
                print(f"   ⚠️  Error: {e}")
            
            writer.writerow([
                json.dumps(rule, ensure_ascii=False),
                dfm_output
            ])
    
    print(f"\n✅ Normalization complete!")
    print(f"📄 Output saved to: {OUTPUT_CSV_PATH}")
    print(f"📊 Processed {len(rules)} rules")


# ============================================================
# 8. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_normalization_pipeline()