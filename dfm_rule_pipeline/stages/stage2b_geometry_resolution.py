import json
import re
from llm.prompts import GEO_MATH_PROMPT
from schema.feature_schema import features_dict

# --------------------------------------------------
# STRONG ENTITY NORMALIZATION
# --------------------------------------------------
# Mapped STRICTLY to schema/feature_schema.py keys
ENTITY_NORMALIZATION = {
    # Holes
    "hole": "Hole",
    "holes": "Hole",
    "adjacent holes": "Hole",

    # Counterbore (Schema: CBHole)
    "counterbore": "CBHole",
    "counterbores": "CBHole",

    # Countersink (Schema: CSHole)
    "countersunk hole": "CSHole",
    "countersunk holes": "CSHole",
    "cs hole": "CSHole",
    "countersink": "CSHole",  # <--- NEW
    "countersinks": "CSHole", # <--- NEW

    # Emboss
    "emboss": "Emboss",
    "embosses": "Emboss",
    "emboss feature": "Emboss",

    # Dimple (Schema: Dimple)
    "dimple": "Dimple",       # <--- NEW
    "dimples": "Dimple",      # <--- NEW

    # Curl (Schema: RolledHem or Curl depending on schema map)
    # Checking your schema: 'Curl' maps to 'RolledHem' in GEOMETRY_ENTITY_CANONICAL_MAP
    # But usually 'RolledHem' is the object name in 'Sheetmetal' string.
    "curl": "RolledHem",      # <--- NEW
    "curls": "RolledHem",     # <--- NEW

    # Bend
    "bend": "Bend",
    "bends": "Bend",
    "bend line": "Bend",

    # Edge (Schema: PartEdge)
    "edge": "PartEdge",
    "edges": "PartEdge",
    "edge of a hole": "PartEdge", # <--- Handles complex edge phrases
}

def normalize_entity(raw: str):
    if not raw:
        return None

    text = raw.lower().strip()

    # exact match
    if text in ENTITY_NORMALIZATION:
        return ENTITY_NORMALIZATION[text]

    # fuzzy recovery: find known token inside phrase
    # Sorted by length descending to catch "countersunk hole" before "hole"
    for key in sorted(ENTITY_NORMALIZATION.keys(), key=len, reverse=True):
        if key in text:
            return ENTITY_NORMALIZATION[key]

    return None


# --------------------------------------------------
# GEOMETRY RESOLUTION
# --------------------------------------------------

def resolve_geometry(llm, rule_text: str, intent: dict) -> dict:
    if isinstance(intent, str):
        intent = json.loads(intent)

    geo = intent.get("raw_geometry_relation") or intent.get("geometry_relation")
    if not geo:
        return {
            "geometry_valid": False,
            "error": "No explicit geometry relation found"
        }

    raw_from = geo.get("from")
    raw_to = geo.get("to")

    entity_a = normalize_entity(raw_from)
    entity_b = normalize_entity(raw_to)

    # 🔒 Adjacent / spacing rules → same-entity geometry
    if entity_a and not entity_b:
        entity_b = entity_a

    # 🔒 Explicit adjacent keyword fallback
    if not entity_b and "adjacent" in rule_text.lower():
        entity_b = entity_a

    if not entity_a or not entity_b:
        # Return error so we can debug, but structure it safely
        return {
            "geometry_valid": False,
            "error": f"Geometry entity could not be resolved: '{raw_from}' -> {entity_a}, '{raw_to}' -> {entity_b}"
        }

    rule_type = intent.get("rule_intent", {}).get("type", "").lower()
    operator = ">=" if rule_type == "min" else "<="

    geo_function = f"Distance({entity_a}, {entity_b})"

    prompt = GEO_MATH_PROMPT.format(
        rule_text=rule_text,
        geo_function=geo_function,
        entity_a=entity_a,
        entity_b=entity_b,
        schema_context=features_dict.get("General", "")
    )

    raw_rhs = llm.call(prompt)
    rhs = str(raw_rhs).replace("```", "").strip()

    if not rhs:
        return {
            "geometry_valid": False,
            "error": "Geometry RHS expression missing"
        }

    return {
        "geometry_valid": True,
        "formalism": "Geometry",
        "geometry": {
            "relation": "distance",
            "from": entity_a,
            "to": entity_b,
            "operator": operator,
            "rhs": rhs
        },
        "reasoning": "Resolved explicit feature-to-feature spatial constraint"
    }