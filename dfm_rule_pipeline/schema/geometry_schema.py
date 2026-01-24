# schema/geometry_schema.py

GEOMETRY_RELATIONS = {
    "DistanceBetween": {
        "description": "Distance between two geometric entities",
        "required_entities": 2,
        "supports_expression": True,
        # ⚡ NEW: Aliases for the LLM to map synonyms automatically
        "aliases": ["distance", "spacing", "gap", "proximity", "clearance"],
        # ⚡ NEW: Template for code generation
        "function_name": "Distance" 
    },
    "OnSamePlane": {
        "description": "Entities lie on the same plane",
        "required_entities": 2,
        "supports_expression": False,
        "aliases": ["plane", "coplanar", "same plane"],
        "function_name": "SamePlane"
    },
    "AdjacentTo": {
        "description": "Entity is adjacent to another",
        "required_entities": 2,
        "supports_expression": False,
        "aliases": ["adjacent", "next to", "touching"],
        "function_name": "IsAdjacent"
    },
    "ClearanceFromEdge": {
        "description": "Distance from edge of a feature",
        "required_entities": 2,
        "supports_expression": True,
        "aliases": ["edge clearance", "edge distance", "from edge"],
        "function_name": "Clearance"
    }
}