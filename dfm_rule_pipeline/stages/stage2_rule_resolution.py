import json
from schema.feature_schema import features_dict

# -------------------------------------------------------------------
# DOMAIN KEYWORD MAP (STRICT MAPPING)
# -------------------------------------------------------------------
# These keywords strongly suggest a specific domain.
DOMAIN_KEYWORDS = {
    "Sheetmetal": ["sheet metal", "bend", "flange", "hem", "cutout", "emboss", "gusset", "louver", "stamp"],
    "Turn": ["turn", "groove", "bore", "relief"],
    "Mill": ["mill", "pocket", "fillet", "chamfer"],
    "Drill": ["drill", "counterbore", "countersink", "tapping"],
    "Injection Moulding": ["injection", "mould", "mold", "rib", "boss", "draft"],
    "Die Cast": ["die cast", "die-cast", "mold wall"],
    "Tubing": ["tube", "tubing", "uniform bend"],
    "Assembly": ["assembly", "fastener", "bolt", "nut", "interference"],
    "Additive": ["additive", "3d print", "layer", "overhang"],
}

# -------------------------------------------------------------------
# RULE CATEGORY + DOMAIN RESOLUTION
# -------------------------------------------------------------------
def resolve_rule_category_and_domain(intent: dict, rule_text: str = "") -> dict:
    rule_intent = intent.get("rule_intent", {})
    
    # 1. Determine Category (Geometry, Tolerance, Attribute)
    rule_type = rule_intent.get("type", "advisory")
    requires_geometry = rule_intent.get("requires_geometry", False)
    requires_tolerance = rule_intent.get("requires_tolerance", False)

    if rule_type == "advisory" and not (requires_geometry or requires_tolerance):
        category = "Advisory"
    elif requires_tolerance:
        category = "Tolerance"
    elif requires_geometry:
        category = "Geometry"
    else:
        category = "Attribute"

    # 2. Determine Domain (Strict Dictionary Match)
    detected_domain = "General" # Default fallback
    
    rule_lower = rule_text.lower()
    
    # Check for strong keyword matches
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(k in rule_lower for k in keywords):
            detected_domain = domain
            break
    
    # Special Handling: If intent says "Sheetmetal" explicitly (from Stage 1 optional), trust it
    # But for now, keyword matching is safer and deterministic.

    return {
        "rule_category": category,
        "primary_domain": detected_domain
    }