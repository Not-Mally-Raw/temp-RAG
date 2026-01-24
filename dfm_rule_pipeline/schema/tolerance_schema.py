# schema/tolerance_schema.py

# Dictionary of allowed Tolerance Formalisms
TOLERANCE_TYPES = {
    "Bilateral": {
        "description": "Plus/Minus tolerance from a nominal value",
        "signature": "Tolerance(Attribute, Target, Plus, Minus)",
        "example": "Tolerance(Bend.Angle, 90, 0.5, 0.5)"
    },
    "Limits": {
        "description": "Absolute Minimum and Maximum values",
        "signature": "Limits(Attribute, Min, Max)",
        "example": "Limits(Hole.Diameter, 9.9, 10.1)"
    },
    "GDT": {
        "description": "Geometric Dimensioning & Tolerancing constraint",
        "signature": "GDT(Type, Feature, Value)",
        "example": "GDT(Flatness, Surface, 0.05)"
    },
    "SingleLimit": {
        "description": "Unilateral max or min specification (often handled by standard equations, but valid here too)",
        "signature": "Max(Attribute, Value) | Min(Attribute, Value)",
        "example": "Max(Surface.Roughness, 1.6)"
    }
}

# Supported GD&T Types (for validation)
GDT_KEYWORDS = [
    "Flatness", "Straightness", "Circularity", "Cylindricity",
    "Profile", "Parallelism", "Perpendicularity", "Angularity",
    "Position", "Concentricity", "Symmetry", "Runout"
]