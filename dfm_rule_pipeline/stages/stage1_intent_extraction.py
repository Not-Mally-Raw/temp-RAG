# stages/stage1_intent_extraction.py
import json
from llm.prompts import INTENT_PROMPT
from schema.feature_schema import features_dict


def generate_domain_map(features):
    domain_map = {}
    for domain, schema_text in features.items():
        if not schema_text:
            continue
        objects = []
        parts = schema_text.split("Object:")
        for part in parts[1:]:
            lines = part.strip().split("\n")
            if lines and lines[0].strip():
                objects.append(lines[0].strip())
        if objects:
            domain_map[domain] = objects
    return json.dumps(domain_map, indent=2)


def extract_intent(llm, rule_text: str) -> dict:
    """
    Extracts intent from rule text.
    RETURNS a parsed dict (never raw JSON).
    Geometry versioning is DECLARED here, not validated.
    """
    print("\n🔹 STAGE 1: INTENT EXTRACTION")

    domain_map_str = generate_domain_map(features_dict)

    prompt = INTENT_PROMPT.format(
        domain_context=domain_map_str,
        rule_text=rule_text
    )

    try:
        raw = llm.call(prompt)
        if not isinstance(raw, str):
            raw = str(raw)
        raw = raw.strip()

        # Strip fenced blocks
        if "```" in raw:
            for part in raw.split("```"):
                p = part.strip()
                if p.startswith("{") and p.endswith("}"):
                    raw = p
                    break

        print(f"    → Intent Output (raw): {raw[:400]}")

        parsed = json.loads(raw)

        # -------------------------------------------------
        # Enforce base intent contract
        # -------------------------------------------------
        required_keys = [
            "domain",
            "rule_type",
            "is_quantifiable",
            "requires_geometry",
            "requires_tolerance",
            "reasoning"
        ]
        for k in required_keys:
            if k not in parsed:
                raise ValueError(f"Missing key: {k}")

        # -------------------------------------------------
        # Geometry v2 defaults (DECLARATIVE ONLY)
        # -------------------------------------------------
        parsed.setdefault("geometry_version", "v1")
        parsed.setdefault("geometry_scope", None)

        return parsed

    except Exception as e:
        print(f"    ⚠️ Intent parsing failed: {e}")
        return {
            "domain": "General",
            "object": None,
            "attribute": None,
            "rule_type": "advisory",
            "is_quantifiable": False,
            "requires_geometry": False,
            "requires_tolerance": False,
            "geometry_version": "v1",
            "geometry_scope": None,
            "geometry_relation": None,
            "tolerance": None,
            "reasoning": f"Intent parsing failed: {str(e)}"
        }
