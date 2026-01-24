import json
from schema.geometry_schema import GEOMETRY_RELATIONS
from schema.feature_schema import features_dict
from llm.prompts import GEO_MATH_PROMPT


# ------------------------------------------------------------------
# RELATION LOOKUP
# ------------------------------------------------------------------
RELATION_LOOKUP = {}
for key, spec in GEOMETRY_RELATIONS.items():
    RELATION_LOOKUP[key.lower()] = key
    for alias in spec.get("aliases", []):
        RELATION_LOOKUP[alias.lower()] = key


def resolve_geometry(llm, rule_text: str, intent_json_str: str) -> str:
    print("\n🔹 STAGE 2B: GEOMETRY RESOLUTION")

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

    raw_type = str(geo.get("type", "")).lower()
    schema_key = RELATION_LOOKUP.get(raw_type)

    if not schema_key:
        return json.dumps({
            "geometry_valid": False,
            "error": f"Unknown geometry relation '{raw_type}'"
        })

    # ---------------------------------------------------------
    # ENTITY EXTRACTION
    # ---------------------------------------------------------
    if geometry_version == "v2":
        entity_a = geo["from"]["entity"]
        entity_b = geo["to"]["entity"]
    else:
        entity_a = geo.get("from")
        entity_b = geo.get("to")

    if not entity_a or not entity_b:
        return json.dumps({
            "geometry_valid": False,
            "error": "Geometry entities missing"
        })

    # ---------------------------------------------------------
    # OPERATOR
    # ---------------------------------------------------------
    rule_type = data.get("rule_type", "").lower()
    operator = ">=" if rule_type == "min" else "<=" if rule_type == "max" else "="

    # ---------------------------------------------------------
    # MATH PROMPT
    # ---------------------------------------------------------
    schema_context = features_dict.get(data.get("domain", "General"), "")

    prompt = GEO_MATH_PROMPT.format(
        rule_text=rule_text,
        geo_function=f"{schema_key}({entity_a}, {entity_b})",
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

    return json.dumps({
        "status": "Success",
        "formalism": "Geometry",
        "geometry_valid": True,
        "geometry": {
            "relation": schema_key,
            "entities": {
                "from": entity_a,
                "to": entity_b
            },
            "constraint": {
                "operator": operator,
                "rhs": rhs
            }
        },
        "reasoning": f"Resolved geometry '{schema_key}' with structured constraint"
    }, indent=2)
