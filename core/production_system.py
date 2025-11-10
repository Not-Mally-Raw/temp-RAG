"""Unified production facade bridging fast and enhanced rule extraction."""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Union

from .rule_extraction import (
    ExtractionSummary,
    RuleExtractionPipeline,
    RuleExtractionSettings,
)

try:  # Optional dependency for rich exports
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - pandas optional
    pd = None  # type: ignore


if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .enhanced_rule_engine import EnhancedConfig  # type: ignore

logger = logging.getLogger(__name__)


class _PipelineRuleEngineFacade:
    """Async-friendly adapter exposing pipeline operations like the legacy engine."""

    def __init__(self, pipeline: RuleExtractionPipeline) -> None:
        self._pipeline = pipeline

    @property
    def settings(self) -> RuleExtractionSettings:
        return self._pipeline.settings

    async def extract_rules_from_text(
        self,
        text: str,
        document_name: str = "ad_hoc.txt",
        *,
        max_rules: Optional[int] = None,
    ) -> Dict[str, Any]:
        summary = await self._pipeline.extract_from_text(
            text,
            document_name,
            max_rules=max_rules,
        )
        return summary.to_dict()

    async def process_document(
        self,
        document_path: str,
        *,
        max_rules: Optional[int] = None,
    ) -> Dict[str, Any]:
        summary = await self._pipeline.process_document(
            document_path,
            max_rules=max_rules,
        )
        return summary.to_dict()

    async def batch_process(
        self,
        document_paths: Sequence[str],
        *,
        max_rules: Optional[int] = None,
        concurrency: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        summaries = await self._pipeline.batch_process(
            document_paths,
            max_rules=max_rules,
            concurrency=concurrency,
        )
        return [summary.to_dict() for summary in summaries]


class ProductionRuleExtractionSystem:
    """High-level orchestrator exposing both fast and enhanced pipelines."""

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        *,
        max_rules: Optional[int] = None,
        use_qdrant: bool = False,
        pipeline_settings: Optional[RuleExtractionSettings] = None,
        enable_enhanced: bool = False,
        enhanced_config: Optional["EnhancedConfig"] = None,
        vector_manager: Optional[Any] = None,
    ) -> None:
        base_settings = pipeline_settings or RuleExtractionSettings()
        settings_data = base_settings.dict()

        if groq_api_key:
            settings_data["groq_api_key"] = groq_api_key
        else:
            settings_data.setdefault("groq_api_key", os.getenv("GROQ_API_KEY", ""))

        if not settings_data.get("groq_api_key"):
            raise ValueError("GROQ_API_KEY missing; set it in the environment or pass explicitly")

        if max_rules is not None:
            settings_data["max_rules_total"] = max_rules

        self.settings = RuleExtractionSettings(**settings_data)
        self.pipeline = RuleExtractionPipeline(self.settings)
        self.max_rules = self.settings.max_rules_total
        self.use_qdrant = use_qdrant

        # Legacy compatibility surfaces
        self.rule_engine = _PipelineRuleEngineFacade(self.pipeline)
        self.prompt_system = self.pipeline.processor.prompts
        self.llm = self.pipeline.processor._client

        self.enhanced_engine = None
        if enable_enhanced:
            try:  # lazy import to avoid heavy dependencies when unused
                from .enhanced_rule_engine import EnhancedConfig, EnhancedRuleEngine  # type: ignore

                config = enhanced_config or EnhancedConfig(groq_api_key=self.settings.groq_api_key)
                self.enhanced_engine = EnhancedRuleEngine(config=config, vector_manager=vector_manager)
            except Exception as exc:  # pragma: no cover - optional feature
                logger.warning("enhanced_engine_initialization_failed", exc_info=exc)

    # ------------------------------------------------------------------
    # Core processing helpers
    # ------------------------------------------------------------------

    async def process_document_advanced(
        self,
        document_path: str,
        enable_enhancement: bool = False,
        enable_validation: bool = False,
    ) -> Dict[str, Any]:
        """Process a document with optional enhanced enrichment."""

        if enable_enhancement and self.enhanced_engine is not None:
            payload = await self._run_enhanced_path(document_path)
            payload.setdefault("document_context", {"mode": "enhanced"})
            return payload

        if enable_enhancement and self.enhanced_engine is None:
            logger.info(
                "enhanced_processing_requested_but_engine_unavailable",
                extra={"document_path": document_path},
            )

        summary = await self.pipeline.process_document(
            document_path,
            max_rules=self.max_rules,
        )
        payload = summary.to_dict()
        payload.setdefault("document_context", {"mode": "fast"})
        if enable_validation:
            payload.setdefault("validation", {"status": "not_configured"})
        return payload

    async def batch_process_documents(
        self,
        document_paths: Sequence[str],
        *,
        concurrency: int = 2,
    ) -> List[Dict[str, Any]]:
        summaries = await self.pipeline.batch_process(
            document_paths,
            max_rules=self.max_rules,
            concurrency=concurrency,
        )
        return [summary.to_dict() for summary in summaries]

    async def extract_rules_from_text(
        self,
        text: str,
        document_name: str = "ad_hoc.txt",
        *,
        max_rules: Optional[int] = None,
    ) -> Dict[str, Any]:
        return await self.rule_engine.extract_rules_from_text(
            text,
            document_name,
            max_rules=max_rules,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def export_results(
        self,
        results: Iterable[Union[Dict[str, Any], ExtractionSummary]],
        format: str = "json",
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Export rule extraction results to JSON, CSV, or Excel."""

        documents = [self._normalise_result(entry) for entry in results]
        if not documents:
            raise ValueError("No results provided for export")

        format_normalised = format.lower()
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        default_name = {
            "json": f"rule_extraction_{timestamp}.json",
            "csv": f"rule_extraction_{timestamp}.csv",
            "tsv": f"rule_extraction_{timestamp}.tsv",
            "xlsx": f"rule_extraction_{timestamp}.xlsx",
        }.get(format_normalised, f"rule_extraction_{timestamp}.{format_normalised}")

        path = Path(output_path) if output_path else Path(default_name)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format_normalised == "json":
            payload = {
                "summary": self._build_summary(documents),
                "results": documents,
            }
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        else:
            rows = self._flatten_rules(documents)
            if format_normalised in {"csv", "tsv"}:
                delimiter = "," if format_normalised == "csv" else "\t"
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}), delimiter=delimiter)
                    writer.writeheader()
                    writer.writerows(rows)
            elif format_normalised in {"xlsx", "excel"}:
                if pd is None:
                    raise RuntimeError("pandas is required for Excel exports; install pandas to enable this format")
                df = pd.DataFrame(rows)
                df.to_excel(path, index=False)  # type: ignore[arg-type]
            else:  # pragma: no cover - safeguard for unsupported formats
                raise ValueError(f"Unsupported export format: {format}")

        logger.info("export_completed", extra={"path": str(path), "format": format_normalised})
        return str(path)

    def get_system_stats(self) -> Dict[str, Any]:
        settings = self.settings
        return {
            "configuration": {
                "max_rules": self.max_rules,
                "max_concurrent_calls": settings.max_concurrent_calls,
                "throttle_seconds": settings.throttle_seconds,
                "enhanced_engine": bool(self.enhanced_engine),
                "use_qdrant": self.use_qdrant,
            }
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_enhanced_path(self, document_path: str) -> Dict[str, Any]:
        """Execute the enhanced rule extraction path for a document."""

        assert self.enhanced_engine is not None  # for type checkers

        payload = await self.pipeline.loader.load(Path(document_path))
        filename = payload.metadata.get("filename") or Path(document_path).name

        enhanced_result = await self.enhanced_engine.extract_rules_from_text(
            payload.text,
            filename=filename,
        )

        enhanced_result.setdefault("document_metadata", payload.metadata)
        enhanced_result.setdefault("filename", filename)
        return enhanced_result

    def _normalise_result(
        self,
        result: Union[Dict[str, Any], ExtractionSummary],
    ) -> Dict[str, Any]:
        if isinstance(result, ExtractionSummary):
            payload = result.to_dict()
        else:
            payload = dict(result)

        payload.setdefault("filename", payload.get("document_name", payload.get("filename", "unknown")))
        payload.setdefault("rules", [])
        payload.setdefault("rule_count", len(payload["rules"]))
        payload.setdefault("avg_confidence", 0.0)
        payload.setdefault("status", "success" if payload["rules"] else payload.get("status", "no_rules"))
        return payload

    def _build_summary(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_rules = sum(len(doc.get("rules", [])) for doc in documents)
        successful = sum(1 for doc in documents if doc.get("status") == "success")
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "documents": len(documents),
            "successful_documents": successful,
            "total_rules": total_rules,
            "average_confidence": (
                sum(doc.get("avg_confidence", 0.0) for doc in documents if doc.get("avg_confidence")) / successful
                if successful
                else 0.0
            ),
        }

    def _flatten_rules(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for doc in documents:
            if not doc.get("rules"):
                rows.append(
                    {
                        "filename": doc.get("filename"),
                        "status": doc.get("status"),
                        "rule_count": doc.get("rule_count", 0),
                        "avg_confidence": doc.get("avg_confidence", 0.0),
                    }
                )
                continue

            for rule in doc["rules"]:
                row = {
                    "filename": doc.get("filename"),
                    "status": doc.get("status"),
                    "rule_count": doc.get("rule_count", 0),
                    "avg_confidence": doc.get("avg_confidence", 0.0),
                }
                row.update(rule)
                rows.append(row)
        return rows


__all__ = ["ProductionRuleExtractionSystem"]