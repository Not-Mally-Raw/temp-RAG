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
        # GroqCloud's GPT-OSS-20B excels at long-context manufacturing analysis.
        self.system_prompt = (
            "You are a senior manufacturing engineer leveraging GroqCloud's GPT-OSS-20B model. "
            "Extract precise, production-ready manufacturing rules. Every response must be valid JSON."
        )

    def build_fast_prompt(self, context: PromptContext) -> str:
        """Render the high-throughput extraction prompt for a single chunk."""

        prefix = (
            f"Document: {context.document_name}\n"
            f"Segment: {context.chunk_index + 1}\n"
            "Instructions: Return a JSON object with a `rules` key. The value must be a list of rule \n"
            "objects describing actionable manufacturing requirements. Each rule object must include \n"
            "rule_text, rule_category, rule_type, confidence (0-1 float), priority (high|medium|low), \n"
            "rationale, primary_feature, unit, value, tolerance_range, supporting_quote. Avoid narration."
        )

        return (
            f"{prefix}\n"
            "Only return JSON.\n"
            "Source text:\n"
            f"{context.chunk_text}"
        )
