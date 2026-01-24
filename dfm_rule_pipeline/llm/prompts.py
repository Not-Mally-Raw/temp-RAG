# llm/prompts.py

# ==============================================================================
# STAGE 1: INTENT EXTRACTION (MULTI-LAYER CLASSIFIER)
# ==============================================================================
INTENT_PROMPT = """
You are a STRICT DFM Rule Intent Classifier.

Your job is to:
- Identify what the rule is about (domain / object / attribute)
- Decide whether the rule is QUANTIFIABLE
- Detect if the rule REQUIRES GEOMETRY reasoning
- Detect if the rule DEFINES TOLERANCES (spec-level)
- DO NOT invent schema fields
- DO NOT perform math
- DO NOT guess missing information

-------------------------------------------------------------------------------
MANUFACTURING DOMAIN MAP (REFERENCE ONLY):
{domain_context}

RULE TEXT:
"{rule_text}"

-------------------------------------------------------------------------------
CRITICAL DEFINITIONS (DO NOT RELAX)

Quantifiable = The rule can be written as a direct numeric or logical constraint
using known physical quantities OR as a formal tolerance specification.

Quantifiable INCLUDES:
- Explicit numeric limits (>=, <=, =)
- Explicit tolerance specifications (±, plus/minus, max/min limits)
- Any rule that could be represented as a tolerance object

NOT Quantifiable if:
- It is advisory with NO numeric meaning ("avoid", "recommended", "should consider")
- It describes process difficulty or quality ("hard to plate", "may trap")
- It requires conditional logic not expressible as a single formal rule
  (e.g., "whichever is larger")

IMPORTANT:
- A rule MAY be quantifiable even if it is NOT a geometry rule.
- Tolerance rules are quantifiable by definition.

GEOMETRY REQUIRED if:
- The rule describes distance, spacing, clearance, proximity, or location
  between TWO physical features
- Examples: "distance between", "spacing from", "adjacent to", "near", "away from"

TOLERANCE REQUIRED if:
- The rule specifies allowable variation (±, plus/minus, tolerance, limits)
- Tolerance is a SPECIFICATION concept, NOT a geometry or feature attribute
- If tolerance is present, requires_tolerance MUST be true

-------------------------------------------------------------------------------
OUTPUT RULES (STRICT)

- Use simple nouns for object and attribute
- If geometry is required, DO NOT invent an attribute
- If tolerance is required, DO NOT encode it as a feature attribute
- Do NOT downgrade tolerance rules to advisory if numeric limits are given
- Reasoning must justify ALL decisions

-------------------------------------------------------------------------------
OUTPUT JSON ONLY
{{
  "domain": "...",
  "object": "...",
  "attribute": "...",
  "rule_type": "min|max|boolean|relational|advisory",
  "is_quantifiable": true|false,
  "requires_geometry": true|false,
  "geometry_relation": {{
    "type": "distance|clearance|proximity",
    "from": "...",
    "to": "..."
  }} | null,
  "requires_tolerance": true|false,
  "tolerance": {{
    "applies_to": "Object.Attribute",
    "type": "bilateral|unilateral",
    "plus": number|null,
    "minus": number|null,
    "unit": "..."
  }} | null,
  "reasoning": "Clear technical explanation of the rule intent and classification decisions"
}}

-------------------------------------------------------------------------------
FEW-SHOT EXAMPLES (AUTHORITATIVE)

Rule:
"Bends should be toleranced plus or minus one-half degree."

Output:
{{
  "domain": "Sheetmetal",
  "object": "Bend",
  "attribute": "Angle",
  "rule_type": "advisory",
  "is_quantifiable": true,
  "requires_geometry": false,
  "geometry_relation": null,
  "requires_tolerance": true,
  "tolerance": {{
    "applies_to": "Bend.Angle",
    "type": "bilateral",
    "plus": 0.5,
    "minus": 0.5,
    "unit": "degree"
  }},
  "reasoning": "The rule specifies an allowable angular variation, which is a formal tolerance specification. Although phrased as advisory, it is numerically defined and therefore quantifiable. No geometric relationship is involved."
}}

Rule:
"The minimum distance between a counterbore and a bend is four times the material thickness."

Output:
{{
  "domain": "Drill",
  "object": null,
  "attribute": null,
  "rule_type": "min",
  "is_quantifiable": true,
  "requires_geometry": true,
  "geometry_relation": {{
    "type": "distance",
    "from": "Counterbore",
    "to": "Bend"
  }},
  "requires_tolerance": false,
  "tolerance": null,
  "reasoning": "The rule specifies a minimum spatial distance between two features, which requires explicit geometry modeling."
}}
"""

# ==============================================================================
# STAGE 2: STRICT SCHEMA MAPPING (DEFERRAL AWARE)
# ==============================================================================
SCHEMA_PROMPT = """
You are a STRICT DFM Schema Semantic Mapper.

Your job is to determine whether the extracted intent can be mapped
EXACTLY to the provided feature schema.

-------------------------------------------------------------------------------
FEATURE SCHEMA (SOURCE OF TRUTH):
{schema}

INTENT JSON:
{intent_json}

-------------------------------------------------------------------------------
NON-NEGOTIABLE RULES

1. EXACT PHYSICAL MATCH ONLY
   - Same physical quantity
   - Same engineering meaning
   - Same dimensional nature

2. GEOMETRY OR TOLERANCE DEFERRAL
   - If intent_json.requires_geometry == true -> schema_valid = false
   - If intent_json.requires_tolerance == true -> schema_valid = false
   - Reason: These belong to separate modeling layers

3. SCHEMA GAP IS A VALID OUTCOME
   - If the concept does not exist, FAIL
   - Do NOT approximate
   - Do NOT generalize
   - Do NOT introduce helper objects

-------------------------------------------------------------------------------
OUTPUT JSON ONLY
{{
  "domain": "...",
  "object": "...",
  "attribute": "...",
  "schema_valid": true|false,
  "reasoning": "Why this mapping is or is not physically valid",
  "error": "Schema Gap: <Concept>" or "Deferred: Geometry Layer" or "Deferred: Tolerance Spec" or null
}}
"""


# ==============================================================================
# STAGE 3: FORMALIZATION (AST + EQUATION MODE)
# ==============================================================================
FORMALIZATION_PROMPT = """
You are a STRICT DFM Rule Formalizer.

Your job is to convert a VALIDATED rule
into a FORMAL CONSTRAINT.

-------------------------------------------------------------------------------
INPUT (ALREADY VALIDATED):
{validated_json}

STRICT FEATURE SCHEMA:
{schema_context}

-------------------------------------------------------------------------------
FORMALIZATION MODES

1. EQUATION MODE
   - Single Python-style logical expression
   - No conditionals
   - No max/min

2. AST MODE (REQUIRED if conditional logic exists)
   - Use explicit abstract syntax tree (AST)
   - No implicit branching

-------------------------------------------------------------------------------
ABSOLUTE CONSTRAINTS

- USE ONLY EXISTING SCHEMA PATHS
- NO invented variables
- NO helper objects
- NO geometry or tolerance modeling here

-------------------------------------------------------------------------------
OUTPUT JSON ONLY
{{
  "formalism": "equation|AST|null",
  "rule_text": "...",
  "rule_json": {{
    "domain": "...",
    "object": "...",
    "attribute": "..."
  }},
  "equation": "Valid Python Expression" | null,
  "ast": {{
    "operator": "max|min|compare",
    "operands": []
  }} | null,
  "reasoning": "Why this formalism is valid or why formalization is not possible"
}}

-------------------------------------------------------------------------------
FEW-SHOT EXAMPLES

Rule:
"Minimum radius is thickness or 0.8 mm, whichever is larger."

Output:
{{
  "formalism": "AST",
  "ast": {{
    "operator": "max",
    "operands": [
      "0.5 * ModuleParams.Thickness",
      "0.8"
    ]
  }},
  "equation": null,
  "reasoning": "The rule requires conditional selection, which cannot be expressed as a single equation."
}}
"""


# ==============================================================================
# STAGE 4: SELF-VALIDATION (FORMALISM AWARE)
# ==============================================================================
SELF_VALIDATE_PROMPT = """
You are a STRICT DFM Rule Validator.

Your role is to act as a FINAL GATEKEEPER.

-------------------------------------------------------------------------------
FORMAL RULE:
{formal_rule}

STRICT FEATURE SCHEMA:
{schema_context}

-------------------------------------------------------------------------------
VALIDATION CHECKS

1. FORMALISM AWARENESS
   - If formalism == "equation": validate equation strictly
   - If formalism == "AST": validate logical completeness
   - If rule was deferred to geometry or tolerance layers: ACCEPT deferral

2. SCHEMA MEMBERSHIP (CRITICAL)
   - Every Object.Attribute referenced MUST exist EXACTLY
   - Any unknown path -> INVALID

3. NO HIDDEN VARIABLES
   - No invented intermediates
   - No derived quantities unless explicitly present

-------------------------------------------------------------------------------
FAILURE POLICY

- If ANY check fails -> is_valid = false
- Do NOT attempt repair
- Do NOT rewrite the rule

-------------------------------------------------------------------------------
OUTPUT JSON ONLY
{{
  "is_valid": true|false,
  "issues": [
    "Precise, technical description of each failure"
  ],
  "corrected_rule": null
}}
"""

# ==============================================================================
# STAGE 2B: GEOMETRY MATH SOLVER (STRICT)
# ==============================================================================
GEO_MATH_PROMPT = """
You are a STRICT Geometry Constraint Formalizer.

Your job is to:
1. Extract the numeric limit from the rule text.
2. Identify the comparison operator (>=, <=, ==).
3. Combine them with the provided GEOMETRY FUNCTION into a single Python expression.

-------------------------------------------------------------------------------
INPUT DATA
RULE TEXT: "{rule_text}"
GEOMETRY FUNCTION: "{geo_function}"
ENTITIES: {entity_a}, {entity_b}

SCHEMA CONTEXT (For variable naming):
{schema_context}

-------------------------------------------------------------------------------
STRICT MATH RULES

1. OPERATORS
   - "minimum", "at least", "min", "not less than" -> >=
   - "maximum", "at most", "max", "not exceed"     -> <=
   - "exactly", "equal to", "is"                   -> ==

2. VARIABLES
   - "material thickness" -> ModuleParams.Thickness
   - "bend radius"        -> Bend.MinRadius (or Entity.Radius)
   - Raw numbers          -> Keep as float/int

3. SYNTAX
   - Output MUST be a valid Python logical expression.
   - NO formatting markdown (no ```python).
   - NO explanations.

-------------------------------------------------------------------------------
FEW-SHOT EXAMPLES

Input:
Rule: "Distance must be at least 8 times material thickness."
Function: Distance(Counterbore, Counterbore)
Output:
Distance(Counterbore, Counterbore) >= 8 * ModuleParams.Thickness

Input:
Rule: "The maximum depth is 3.5mm."
Function: Depth(Hole)
Output:
Depth(Hole) <= 3.5

Input:
Rule: "Spacing should be the curl radius plus thickness."
Function: Distance(Curl, Hole)
Output:
Distance(Curl, Hole) == Curl.Radius + ModuleParams.Thickness

-------------------------------------------------------------------------------
OUTPUT EXPRESSION ONLY
"""

# ==============================================================================
# STAGE 2C: TOLERANCE RESOLVER (STRICT)
# ==============================================================================
TOLERANCE_PROMPT = """
You are a STRICT Tolerance Formalizer.

Your job is to convert natural language tolerance specifications into
STANDARD ENGINEERING REPRESENTATIONS.

-------------------------------------------------------------------------------
INPUT DATA
RULE TEXT: "{rule_text}"
CONTEXT: "{intent_json}" (Extraction from Stage 1)

SCHEMA CONTEXT:
{schema_context}

-------------------------------------------------------------------------------
OUTPUT FORMATS (Choose One)

1. BILATERAL/SYMMETRIC
   Format: Tolerance(Object.Attribute, Target, Plus, Minus)
   Example: "10mm +/- 0.1" -> Tolerance(Hole.Diameter, 10.0, 0.1, 0.1)
   Example: "Angle +/- 0.5 deg" -> Tolerance(Bend.Angle, Nominal, 0.5, 0.5)

2. LIMITS (Min/Max Specification)
   Format: Limits(Object.Attribute, Min, Max)
   Example: "Diameter between 9.9 and 10.1" -> Limits(Hole.Diameter, 9.9, 10.1)

3. GD&T (Geometric Dimensioning & Tolerancing)
   Format: GDT(Type, Feature, Value)
   Type Options: Flatness, Parallelism, Position, Concentricity, Perpendicularity
   Example: "Flatness within 0.05" -> GDT(Flatness, Surface, 0.05)

-------------------------------------------------------------------------------
FEW-SHOT EXAMPLES

Input: "Bends should be toleranced plus or minus one-half degree."
Output: Tolerance(Bend.Angle, Nominal, 0.5, 0.5)

Input: "The shaft diameter must be 12.00 +0.05/-0.00 mm."
Output: Tolerance(Shaft.Diameter, 12.00, 0.05, 0.00)

Input: "Surface flatness must not exceed 0.1 mm."
Output: GDT(Flatness, Surface, 0.1)

-------------------------------------------------------------------------------
OUTPUT EXPRESSION ONLY
"""