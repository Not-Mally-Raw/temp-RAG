"""Compatibility wrapper built on top of :mod:`core.rule_extraction`."""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import Field
from pydantic_settings import BaseSettings

from .rule_extraction import Rule, RuleExtractionPipeline, RuleExtractionSettings


class FastRuleExtractorConfig(BaseSettings):
	"""Legacy configuration mapped onto :class:`RuleExtractionSettings`."""

	groq_api_key: str = Field(default="")
	groq_model: str = Field(default="gpt-oss-20b")
	temperature: float = Field(default=0.1)
	max_output_tokens: int = Field(default=2048)
	max_chunk_tokens: int = Field(default=3200)
	chunk_overlap_tokens: int = Field(default=200)
	max_rules_per_chunk: int = Field(default=12)
	max_rules_total: int = Field(default=80)
	throttle_seconds: float = Field(default=0.0)
	request_timeout: float = Field(default=60.0)
	max_concurrent_calls: int = Field(default=4)
	max_retries: int = Field(default=3)
	retry_backoff_seconds: float = Field(default=2.0)

	class Config:
		env_file = ".env"
		extra = "ignore"


class FastRuleExtractor:
	"""Projected facade used by the CLI and production system."""

	def __init__(self, config: Optional[FastRuleExtractorConfig] = None) -> None:
		self.config = config or FastRuleExtractorConfig()
		self.pipeline = RuleExtractionPipeline(self._build_settings())

	async def extract_from_text(
		self,
		document_text: str,
		document_name: str,
		max_rules: Optional[int] = None,
	) -> Dict[str, object]:
		summary = await self.pipeline.extract_from_text(
			document_text,
			document_name,
			max_rules=max_rules,
		)
		return summary.to_dict()

	def _build_settings(self) -> RuleExtractionSettings:
		return RuleExtractionSettings(
			groq_api_key=self.config.groq_api_key,
			groq_model=self.config.groq_model,
			temperature=self.config.temperature,
			max_output_tokens=self.config.max_output_tokens,
			max_chunk_tokens=self.config.max_chunk_tokens,
			chunk_overlap_tokens=self.config.chunk_overlap_tokens,
			max_rules_per_chunk=self.config.max_rules_per_chunk,
			max_rules_total=self.config.max_rules_total,
			throttle_seconds=self.config.throttle_seconds,
			request_timeout=self.config.request_timeout,
			max_concurrent_calls=self.config.max_concurrent_calls,
			max_retries=self.config.max_retries,
			retry_backoff_seconds=self.config.retry_backoff_seconds,
		)


async def extract_rules_from_text(
	text: str,
	document_name: str,
	config: Optional[FastRuleExtractorConfig] = None,
	max_rules: Optional[int] = None,
) -> Dict[str, object]:
	"""Backward compatible helper used by legacy imports."""
	extractor = FastRuleExtractor(config)
	summary = await extractor.extract_from_text(text, document_name, max_rules=max_rules)
	return summary


# Re-export for consumers that previously imported FastRule directly
FastRule = Rule
