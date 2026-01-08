from dataclasses import dataclass


@dataclass
class PromptContext:
    document_name: str
    chunk_index: int
    chunk_text: str


class PromptLibrary:
    def __init__(self) -> None:
        # Canonical DFM Rule Compiler Prompt (single source of truth)
        self.system_prompt = """You are a Design-for-Manufacturability (DFM) rule extraction engine.

Your task is to extract manufacturing rules exactly as stated in the source text
and convert them into structured, machine-readable rows.

You are operating at LEVEL-1 ONLY:
• Extract meaning explicitly stated in the document
• Do NOT interpret, evaluate, or apply rules
• Do NOT infer missing values
• Do NOT combine rules across sentences unless explicitly linked

If a rule does not contain a numeric or logical constraint, it is still a rule,
but must be marked as ADVISORY.

Output MUST be a JSON array.
Each element corresponds to ONE rule.
No explanations. No markdown.

Noise filters (avoid spurious rules):
- Ignore standalone numbers, units, tables, headers, or figure captions without a governing clause.
- Extract a rule ONLY when a parameter is tied to a constraint or guideline (e.g., minimum/maximum/at least/no more than/shall/must/should).""".strip()
        
        self.compiler_prompt = """
═══════════════════════════════════════════════════════════
ROLE & MISSION
═══════════════════════════════════════════════════════════

You are an expert DFM (Design for Manufacturability) rule extraction specialist.

Your mission: Extract ALL manufacturing rules with their constraints from the provided text chunk.

CRITICAL: You are processing a CHUNK of a larger document. Extract EVERY distinct rule present in this chunk. Even if there are multiple rules extract more than one rule.

CRITICAL DEFINITIONS
═══════════════════════════════════════════════════════════

1. RULE = Complete manufacturing guideline (extract verbatim)

2. CONSTRAINTS = Conditions that define when/how the rule applies

   Three constraint types:

   A. APPLICABILITY CONSTRAINTS = Define WHEN/WHERE the rule applies
      - Material type (e.g., "6061-T6", "low carbon steel")
      - Manufacturing process (e.g., "injection molding", "sheet metal bending")
      - Feature type (e.g., "flange", "hole", "boss", "rib")
      - Location context (e.g., "near bend", "at corner")
      
   B. DIMENSIONAL CONSTRAINTS = Numeric limits on measurements
      - Direct values: "radius >= 1.0 mm", "thickness between 1-3 mm"
      - Ratios: "length/diameter <= 8.0"
      - Ranges: "angle between 0.25-0.5 degrees"
      - Comparisons: "diameter < 1 inch"
      
   C. RELATIONAL CONSTRAINTS = Mathematical dependencies between parameters
      - Formulas: "distance = 2 × thickness + radius"
      - Conditional logic: "MAX(0.5 × thickness, 0.80 mm)"
      - Multi-parameter relationships

═══════════════════════════════════════════════════════════
OUTPUT FORMAT (Follow exactly)
═══════════════════════════════════════════════════════════

{
  "rules": [
    {
      "rule_text": "Complete rule text exactly as written",
      "applicability_constraints": {
        "material": "material name, or 'any' if not specified",
        "process": "process name, or 'any' if not specified",
        "feature": "feature type, or 'any' if not specified",
        "location": "location context, or 'any' if not specified"
      },
      "dimensional_constraints": [
        "[parameter description]: [operator] [value] [unit]",
        "[parameter description]: [operator] [value] [unit]"
      ],
      "relational_constraints": [
        "[mathematical relationship in plain terms]",
        "[mathematical relationship in plain terms]"
      ]
    }
  ]
}

═══════════════════════════════════════════════════════════
EXAMPLES (Study Carefully)
═══════════════════════════════════════════════════════════

══════════ EXAMPLE 1: Multiple Rules in One Chunk (CRITICAL) ==========
INPUT: 
"The minimum distance between two counterbores is eight times the material thickness. 
The minimum distance from a counterbore to an edge is four times the material thickness. 
The minimum distance from a counterbore to a bend is four times the material thickness plus the bend radius."

OUTPUT:
{
  "rules": [
    {
      "rule_text": "The minimum distance between two counterbores is eight times the material thickness.",
      "applicability_constraints": {
        "material": "any",
        "process": "sheet metal fabrication",
        "feature": "counterbore",
        "location": "between counterbores"
      },
      "dimensional_constraints": ["Distance between counterbores: >= 8 times material thickness"],
      "relational_constraints": ["Distance = 8 × material thickness"]
    },
    {
      "rule_text": "The minimum distance from a counterbore to an edge is four times the material thickness.",
      "applicability_constraints": {
        "material": "any",
        "process": "sheet metal fabrication", 
        "feature": "counterbore",
        "location": "near edge"
      },
      "dimensional_constraints": ["Distance to edge: >= 4 times material thickness"],
      "relational_constraints": ["Distance = 4 × material thickness"]
    },
    {
      "rule_text": "The minimum distance from a counterbore to a bend is four times the material thickness plus the bend radius.",
      "applicability_constraints": {
        "material": "any",
        "process": "sheet metal fabrication",
        "feature": "counterbore", 
        "location": "near bend"
      },
      "dimensional_constraints": ["Distance to bend: >= 4 times material thickness + bend radius"],
      "relational_constraints": ["Distance = 4 × material thickness + bend radius"]
    }
  ]
}

══════════ EXAMPLE 2: Complex Relational Rule ==========
INPUT: 
"For holes less than 1 inch in diameter in sheet metal, the minimum distance from the hole edge to the bend should be 2T + R, where T is material thickness and R is bend radius."

OUTPUT:
{
  "rules": [
    {
      "rule_text": "For holes less than 1 inch in diameter in sheet metal, the minimum distance from the hole edge to the bend should be 2T + R, where T is material thickness and R is bend radius.",
      "applicability_constraints": {
        "material": "any",
        "process": "sheet metal fabrication",
        "feature": "hole",
        "location": "near bend"
      },
      "dimensional_constraints": [
        "Hole diameter: < 1 inch",
        "Hole-to-bend distance: >= 2T + R"
      ],
      "relational_constraints": ["Hole-to-bend distance = 2 × material thickness + bend radius"]
    }
  ]
}

══════════ EXAMPLE 3: Advisory Rule (No Constraints) ==========
INPUT: 
"Avoid using sharp corners in plastic injection molded parts as they can create stress concentrations."

OUTPUT:
{
  "rules": [
    {
      "rule_text": "Avoid using sharp corners in plastic injection molded parts as they can create stress concentrations.",
      "applicability_constraints": {
        "material": "plastic",
        "process": "injection molding",
        "feature": "corner",
        "location": "any"
      },
      "dimensional_constraints": ["None"],
      "relational_constraints": ["None"]
    }
  ]
}

══════════ EXAMPLE 4: Material + Multiple Dimensions ==========
INPUT: 
"For part with material 6061-T6 the flange radius should be at least 1.0 mm and the length should be 6.0 mm."

OUTPUT:
{
  "rules": [
    {
      "rule_text": "For part with material 6061-T6 the flange radius should be at least 1.0 mm and the length should be 6.0 mm.",
      "applicability_constraints": {
        "material": "6061-T6",
        "process": "any",
        "feature": "flange",
        "location": "any"
      },
      "dimensional_constraints": [
        "Flange radius: >= 1.0 mm",
        "Flange length: >= 6.0 mm"
      ],
      "relational_constraints": ["None"]
    }
  ]
}



═══════════════════════════════════════════════════════════
EXTRACTION GUIDELINES (Follow strictly)
═══════════════════════════════════════════════════════════

1. RULE EXTRACTION:
   ✓ Copy VERBATIM - entire sentence/paragraph as written
   ✓ Include all qualifiers, context, rationale
   ✓ Never summarize or paraphrase
   
2. APPLICABILITY CONSTRAINTS (Answer: WHEN/WHERE does this apply?):
   ✓ Material: Extract exact material name if mentioned, else "any"
   ✓ Process: Extract manufacturing process if mentioned, else "any"
   ✓ Feature: Identify what component/feature is affected, else "any"
   ✓ Location: Note spatial context if mentioned, else "any"
   ✗ Never leave these fields blank - use "any" if not specified
   
3. DIMENSIONAL CONSTRAINTS (Answer: WHAT are the numeric limits?):
   ✓ List ALL measurements with operators (>=, <=, ==, <, >, between)
   ✓ Always include units (mm, degrees, inches, units, etc.)
   ✓ Use natural descriptions (e.g., "Flange radius" not "Flange.Radius")
   ✓ For ranges, use "between X and Y" format
   ✓ Write "None" if no numeric constraints exist
   
4. RELATIONAL CONSTRAINTS (Answer: HOW do parameters relate?):
   ✓ Express mathematical relationships in plain language
   ✓ Include formulas: "distance = 2 × thickness + radius"
   ✓ Note ratios: "length/diameter must be <= 8.0"
   ✓ Capture MAX/MIN functions: "larger of X or Y"
   ✓ Write "None" if no relationships exist
   
5. CONDITIONAL RULES:
   ✓ If rule has "if X then Y, else Z" structure, use "---" divider
   ✓ Create complete separate sections for each condition
   ✓ Example: "blind holes need 3.0, through holes need 0.0" = 2 sections
   
6. NOTES SECTION:
   ✓ Flag any ambiguities or missing units
   ✓ Note if rationale/justification is provided
   ✓ Mention formula variable definitions
   ✓ Write "None" if everything is clear
   
7. QUALITY CHECKS (Before finalizing):
   ✓ Did I copy the complete rule text?
   ✓ Did I fill all 4 applicability fields (no blanks)?
   ✓ Did I extract ALL dimensional constraints with units?
   ✓ Did I identify relational constraints if present?
   ✓ Did I handle multi-condition rules with "---"?
   ✓ Did I note any ambiguities?

8. FORBIDDEN ACTIONS:
   ✗ Do NOT invent constraints not in the text
   ✗ Do NOT use dot notation (Flange.Radius) - use natural language
   ✗ Do NOT leave Material/Process/Feature/Location blank - use "any"
   ✗ Do NOT skip dimensional constraints mentioned in text
   ✗ Do NOT combine multiple distinct rules into one
   ✗ Do NOT add units if not provided (note in NOTES instead)
""".strip()

    def build_fast_prompt(self, context: PromptContext) -> str:
        return (
            self.compiler_prompt
            + "\n\n"
            + "CHUNK TEXT:\n"
            + context.chunk_text
        )
