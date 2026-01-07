"""Prompt helpers for GroqCloud GPT-OSS-20B manufacturing extraction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PromptContext:
    """Lightweight container describing the current document segment."""

    document_name: str
    chunk_index: int
    chunk_text: str


class PromptLibrary:
    """Centralised prompts shared across the rule extraction pipeline."""

    def __init__(self) -> None:
        # Level-1 minimalist extraction prompt: no interpretation/evaluation; table-ready rows.
        self.system_prompt = """
    You are a Design-for-Manufacturability (DFM) rule extraction engine.

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
    - Extract a rule ONLY when a parameter is tied to a constraint or guideline (e.g., minimum/maximum/at least/no more than/shall/must/should).
    """.strip()

    def build_fast_prompt(self, context: PromptContext) -> str:
        """Render the high-throughput extraction prompt for a single chunk."""

        return (
            f"Document: {context.document_name}\n"
            f"Segment: {context.chunk_index + 1}\n\n"
            "Return a JSON ARRAY of rule objects using the schema below.\n\n"
            "REQUIRED OUTPUT SCHEMA (KEEP ALL FIELDS):\n"
            "{\n"
            "  \"confidence\": 0.0,\n"
            "  \"rule_text\": \"verbatim or lightly normalized rule sentence\",\n"
            "  \"constraints\": \"explicit constraint(s) only, or null\",\n"
            "  \"applicability\": {\n"
            "    \"process\": \"e.g. sheet metal bending | unknown\",\n"
            "    \"material\": \"e.g. low carbon steel | unknown\",\n"
            "    \"feature\": \"e.g. bend | countersink | unknown\",\n"
            "    \"location\": \"e.g. adjacent to bend | unknown\"\n"
            "  },\n"
            "  \"severity\": \"ENFORCEABLE | ADVISORY\",\n"
            "  \"source_document\": \"" + context.document_name + "\"\n"
            "}\n\n"
            "NON-NEGOTIABLE RULES:\n"
            "1. rule_text MUST always be present.\n"
            "2. constraints MUST contain explicit numeric or logical constraints only (no prose, no interpretation).\n"
            "3. If no explicit constraint exists: constraints = null and severity = ADVISORY.\n"
            "4. Do NOT invent applicability. If not stated, use 'unknown'.\n"
            "5. Do NOT merge multiple rules into one.\n"
            "6. One sentence ≈ one rule unless explicitly combined in the text.\n\n"
            "REAL EXAMPLES (FOLLOW EXACTLY):\n\n"
            "Example — Enforceable (input text → output object):\n"
            "> In low carbon steel sheet metal, the minimum radius of a bend should be one-half the material thickness or 0.80 mm, whichever is larger.\n\n"
            "{\n"
            "  \"confidence\": 0.92,\n"
            "  \"rule_text\": \"The minimum bend radius in low carbon steel sheet metal shall be the greater of 0.5×material thickness or 0.80 mm.\",\n"
            "  \"constraints\": \"bend_radius >= MAX(0.5 * material_thickness, 0.80 mm)\",\n"
            "  \"applicability\": {\n"
            "    \"process\": \"sheet metal bending\",\n"
            "    \"material\": \"low carbon steel\",\n"
            "    \"feature\": \"bend\",\n"
            "    \"location\": \"unknown\"\n"
            "  },\n"
            "  \"severity\": \"ENFORCEABLE\",\n"
            "  \"source_document\": \"" + context.document_name + "\"\n"
            "}\n\n"
            "Example — Advisory (input text → output object):\n"
            "> Avoid large sheet metal parts with small bent flanges.\n\n"
            "{\n"
            "  \"confidence\": 0.65,\n"
            "  \"rule_text\": \"Large sheet metal parts should avoid small bent flanges.\",\n"
            "  \"constraints\": null,\n"
            "  \"applicability\": {\n"
            "    \"process\": \"sheet metal fabrication\",\n"
            "    \"material\": \"unknown\",\n"
            "    \"feature\": \"flange\",\n"
            "    \"location\": \"unknown\"\n"
            "  },\n"
            "  \"severity\": \"ADVISORY\",\n"
            "  \"source_document\": \"" + context.document_name + "\"\n"
            "}\n\n"
            "SOURCE TEXT:\n"
            f"{context.chunk_text}"
        )
