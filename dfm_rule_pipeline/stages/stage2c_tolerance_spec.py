import json
import re

from schema.feature_schema import features_dict
from schema.tolerance_schema import TOLERANCE_TYPES, GDT_KEYWORDS
from llm.prompts import TOLERANCE_PROMPT


def resolve_tolerance(llm, rule_text: str, intent_input) -> str:
    print("\n🔹 STAGE 2C: TOLERANCE RESOLUTION")

    # ------------------------------------------------------------------
    # 1. Parse Intent (STRICT)
    # ------------------------------------------------------------------
    if isinstance(intent_input, dict):
        intent = intent_input
    else:
        try:
            intent = json.loads(intent_input)
        except Exception:
            return json.dumps({
                "status": "Failed",
                "tolerance_valid": False,
                "error_type": "InvalidInput",
                "error": "Intent is not valid JSON"
            })

    tolerance_context = intent.get("tolerance")
    if not tolerance_context:
        return json.dumps({
            "status": "Failed",
            "tolerance_valid": False,
            "error_type": "MissingTolerance",
            "error": "Tolerance resolver invoked without tolerance intent"
        })

    # ------------------------------------------------------------------
    # 2. LLM FORMALIZATION
    # ------------------------------------------------------------------
    schema_context = features_dict.get("General", "")

    prompt = TOLERANCE_PROMPT.format(
        rule_text=rule_text,
        intent_json=json.dumps(tolerance_context, indent=2),
        schema_context=schema_context
    )

    try:
        raw = llm.call(prompt)
        if not isinstance(raw, str):
            raw = str(raw)

        expression = (
            raw.replace("```python", "")
               .replace("```", "")
               .strip()
               .strip('"')
               .strip("'")
        )

        print(f"    🎚️  LLM Output: {expression}")

    except Exception as e:
        return json.dumps({
            "status": "Failed",
            "tolerance_valid": False,
            "error_type": "LLMFailure",
            "error": str(e)
        })

    # ------------------------------------------------------------------
    # 3. SEMANTIC TOLERANCE TYPE DETECTION (NO SCHEMA ASSUMPTIONS)
    # ------------------------------------------------------------------
    parsed_type = None

    if expression.startswith("Tolerance("):
        parsed_type = "Bilateral"

    elif expression.startswith("Limits("):
        parsed_type = "Limits"

    elif expression.startswith(("Max(", "Min(")):
        parsed_type = "SingleLimit"

    elif expression.startswith("GDT("):
        match = re.search(r"GDT\s*\(\s*([A-Za-z]+)", expression)
        if not match:
            return json.dumps({
                "status": "Deferred",
                "tolerance_valid": False,
                "error_type": "MalformedGDT",
                "error": "GD&T expression malformed but intent recognized"
            })

        gdt_type = match.group(1)
        if gdt_type not in GDT_KEYWORDS:
            return json.dumps({
                "status": "Deferred",
                "tolerance_valid": False,
                "error_type": "UnsupportedGDTType",
                "error": f"Recognized GD&T type '{gdt_type}' is not supported"
            })

        parsed_type = "GDT"

    # ------------------------------------------------------------------
    # 4. FINAL GATE — SUPPORTED VS UNDERSTOOD
    # ------------------------------------------------------------------
    if parsed_type not in TOLERANCE_TYPES:
        return json.dumps({
            "status": "Deferred",
            "tolerance_valid": False,
            "error_type": "UnsupportedToleranceForm",
            "error": (
                f"Tolerance intent recognized but formalism "
                f"'{expression}' is not supported"
            )
        })

    # ------------------------------------------------------------------
    # 5. SUCCESS
    # ------------------------------------------------------------------
    return json.dumps({
        "status": "Success",
        "tolerance_valid": True,
        "formalism": "Tolerance",
        "equation": expression,
        "reasoning": f"Resolved {parsed_type} tolerance specification."
    }, indent=2)
