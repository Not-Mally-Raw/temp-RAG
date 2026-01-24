# stages/stage2_schema_validation.py
import json
from llm.prompts import SCHEMA_PROMPT
from schema.feature_schema import features_dict

def validate_against_schema(llm, intent_input) -> str:
    """
    Accepts either a dict (preferred) or a JSON string for the intent.
    Returns a JSON string (LLM response or early deferral/failure).
    """
    print("\n🔹 STAGE 2: SCHEMA MAPPING")

    # normalize to dict
    if isinstance(intent_input, dict):
        intent = intent_input
    else:
        try:
            intent = json.loads(intent_input)
        except Exception:
            return json.dumps({
                "schema_valid": False,
                "error": "Invalid intent JSON"
            })

    # Hard stops / deferrals
    if intent.get("requires_geometry") is True:
        # Explicit deferral to Geometry Layer
        # Keep original intent fields in return for later stages
        out = {**intent, "schema_valid": False, "error": "Deferred: Geometry Layer"}
        return json.dumps(out)

    if intent.get("requires_tolerance") is True:
        out = {**intent, "schema_valid": False, "error": "Deferred: Tolerance Spec"}
        return json.dumps(out)

    if intent.get("is_quantifiable") is False:
        out = {**intent, "schema_valid": False, "error": "Non-quantifiable rule"}
        return json.dumps(out)

    # Normal mapping: provide schema context for the LLM
    domain = intent.get("domain", "General")
    schema_text = features_dict.get(domain, features_dict.get("General", ""))

    prompt = SCHEMA_PROMPT.format(
        schema=schema_text,
        intent_json=json.dumps(intent, indent=2)
    )

    # Call LLM for mapping
    try:
        raw = llm.call(prompt)
        if not isinstance(raw, str):
            raw = str(raw)
        raw = raw.strip()

        # strip fenced code blocks if present
        if "```" in raw:
            parts = raw.split("```")
            for p in parts:
                p_stripped = p.strip()
                if p_stripped.startswith("{") and p_stripped.endswith("}"):
                    raw = p_stripped
                    break

        print(f"    → Schema Mapping (raw): {raw[:400]}")
    except Exception as e:
        return json.dumps({
            "schema_valid": False,
            "error": f"LLM schema mapping failed: {str(e)}"
        })

    # Validate the LLM output minimally
    try:
        parsed = json.loads(raw)
        if parsed.get("schema_valid") is True:
            # require object and attribute when schema_valid true
            if not parsed.get("object") or not parsed.get("attribute"):
                return json.dumps({
                    "schema_valid": False,
                    "error": "Schema mapping returned schema_valid true but missing object/attribute"
                })
    except Exception as e:
        return json.dumps({
            "schema_valid": False,
            "error": f"Schema mapping parse error: {str(e)}"
        })

    return raw
