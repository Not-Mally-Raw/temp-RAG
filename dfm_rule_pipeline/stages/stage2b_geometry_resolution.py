import json
from schema.feature_schema import features_dict
from llm.prompts import GEO_MATH_PROMPT


def resolve_geometry(llm, rule_text: str, intent_json_str: str) -> str:
    print("\n🔹 STAGE 2B: GEOMETRY RESOLUTION")

    # ---------------------------------------------------------
    # INPUT PARSING
    # ---------------------------------------------------------
    try:
        data = intent_json_str if isinstance(intent_json_str, dict) else json.loads(intent_json_str)
    except Exception:
        return json.dumps({
            "geometry_valid": False,
            "error": "Invalid JSON input to geometry stage"
        })

    geo = data.get("geometry_relation")
    if not isinstance(geo, dict):
        return json.dumps({
            "geometry_valid": False,
            "error": "Missing geometry_relation"
        })

    geometry_version = data.get("geometry_version", "v1")

    # ---------------------------------------------------------
    # ENTITY EXTRACTION
    # ---------------------------------------------------------
    if geometry_version == "v2":
        entity_a = geo.get("from", {}).get("entity")
        entity_b = geo.get("to", {}).get("entity")
    else:
        entity_a = geo.get("from")
        entity_b = geo.get("to")

    if not entity_a or not entity_b:
        return json.dumps({
            "geometry_valid": False,
            "error": "Geometry entities missing"
        })

    # ---------------------------------------------------------
    # OPERATOR RESOLUTION
    # ---------------------------------------------------------
    rule_type = str(data.get("rule_type", "")).lower()
    operator = ">=" if rule_type == "min" else "<=" if rule_type == "max" else "="

    # ---------------------------------------------------------
    # RHS MATH EXTRACTION
    # ---------------------------------------------------------
    schema_context = features_dict.get(data.get("domain", "General"), "")

    prompt = GEO_MATH_PROMPT.format(
        rule_text=rule_text,
        geo_function=f"Distance({entity_a}, {entity_b})",
        entity_a=entity_a,
        entity_b=entity_b,
        schema_context=schema_context
    )

    try:
        rhs = llm.call(prompt).strip().strip("`").strip('"').strip("'")
    except Exception as e:
        return json.dumps({
            "geometry_valid": False,
            "error": f"Math extraction failed: {str(e)}"
        })

    # ---------------------------------------------------------
    # NORMALIZED GEOMETRY OUTPUT
    # ---------------------------------------------------------
    return json.dumps({
        "geometry_valid": True,
        "formalism": "Geometry",
        "geometry": {
            "relation": {
                "type": "distance"
            },
            "entities": {
                "from": entity_a,
                "to": entity_b
            },
            "constraint": {
                "operator": operator,
                "rhs": rhs
            }
        },
        "reasoning": "Resolved distance-based geometry constraint"
    }, indent=2)
