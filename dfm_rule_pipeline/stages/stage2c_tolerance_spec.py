import json
import re

from schema.feature_schema import features_dict
from schema.tolerance_schema import TOLERANCE_TYPES, GDT_KEYWORDS
from llm.prompts import TOLERANCE_PROMPT


def resolve_tolerance(llm, rule_text: str, intent_json_str: str) -> str:
    print("\n🔹 STAGE 2C: TOLERANCE RESOLUTION")

    # ------------------------------------------------------------------
    # 1. Parse Input
    # ------------------------------------------------------------------
    if isinstance(intent_json_str, dict):
        data = intent_json_str
    else:
        try:
            data = json.loads(intent_json_str)
        except Exception:
            return json.dumps({
                "status": "Failed",
                "tolerance_valid": False,
                "error": "Invalid intent JSON"
            })

    tol_info = data.get("tolerance") or {}

    # ------------------------------------------------------------------
    # 2. LLM Call
    # ------------------------------------------------------------------
    schema_context = features_dict.get("General", "")

    prompt = TOLERANCE_PROMPT.format(
        rule_text=rule_text,
        intent_json=json.dumps(tol_info, indent=2),
        schema_context=schema_context
    )

    try:
        expression = llm.call(prompt).strip()
        expression = (
            expression
            .replace("```python", "")
            .replace("```", "")
            .strip()
            .strip('"')
            .strip("'")
        )

        print(f"    🎚️  LLM Output: {expression}")

        # ------------------------------------------------------------------
        # 3. STRICT SCHEMA VALIDATION
        # ------------------------------------------------------------------
        parsed_type = None

        # ---- Bilateral Tolerance
        if expression.startswith("Tolerance("):
            parsed_type = "Bilateral"

        # ---- Limits
        elif expression.startswith("Limits("):
            parsed_type = "Limits"

        # ---- Single Limit
        elif expression.startswith("Max(") or expression.startswith("Min("):
            parsed_type = "SingleLimit"

        # ---- GD&T
        elif expression.startswith("GDT("):
            match = re.search(r"GDT\s*\(\s*([A-Za-z]+)", expression)
            if not match:
                return json.dumps({
                    "status": "Failed",
                    "tolerance_valid": False,
                    "error": "Malformed GDT expression"
                })

            gdt_type = match.group(1)
            if gdt_type not in GDT_KEYWORDS:
                return json.dumps({
                    "status": "Failed",
                    "tolerance_valid": False,
                    "error": f"Invalid GD&T type '{gdt_type}'"
                })

            parsed_type = "GDT"

        # ---- Final Gate
        if parsed_type not in TOLERANCE_TYPES:
            return json.dumps({
                "status": "Failed",
                "tolerance_valid": False,
                "error": (
                    f"Expression '{expression}' does not match "
                    f"any tolerance schema: {list(TOLERANCE_TYPES.keys())}"
                )
            })

        # ------------------------------------------------------------------
        # 4. SUCCESS
        # ------------------------------------------------------------------
        return json.dumps({
            "status": "Success",
            "tolerance_valid": True,
            "formalism": "Tolerance",
            "equation": expression,
            "reasoning": f"Resolved {parsed_type} tolerance specification."
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "status": "Failed",
            "tolerance_valid": False,
            "error": f"Tolerance resolution error: {str(e)}"
        })
