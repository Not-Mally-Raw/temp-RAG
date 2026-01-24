GEOMETRY_RELATIONS = {
    "distance": {
        "arity": 2,
        "allowed_entities": ["Bend", "Hole", "Counterbore", "Edge"],
        "supports_min": True,
        "supports_max": True
    },
    "clearance": {
        "arity": 2,
        "allowed_entities": ["Fastener", "Hole"],
        "supports_min": True
    }
}

def validate_geometry_relation(relation_type, entities):
    if relation_type not in GEOMETRY_RELATIONS:
        return False, "Unknown geometry relation"

    spec = GEOMETRY_RELATIONS[relation_type]
    if len(entities) != spec["arity"]:
        return False, "Wrong number of entities"

    return True, None


def build_distance_expression(entity_a, entity_b):
    return f"Distance({entity_a}, {entity_b})"
