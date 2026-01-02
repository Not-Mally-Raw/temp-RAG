"""Unified production facade bridging fast and enhanced rule extraction."""

from __future__ import annotations

import asyncio
import csv
import hashlib
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
        include_metadata: bool = True,
        schema: str = "full",
    ) -> str:
        """Export rule extraction results to JSON, CSV, or Excel."""

        documents = [self._normalise_result(entry) for entry in results]
        if not documents:
            raise ValueError("No results provided for export")

        format_normalised = format.lower().strip()
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        default_name = {
            "json": f"rule_extraction_{timestamp}.json",
            "csv": f"rule_extraction_{timestamp}.csv",
            "tsv": f"rule_extraction_{timestamp}.tsv",
            "xlsx": f"rule_extraction_{timestamp}.xlsx",
            "excel": f"rule_extraction_{timestamp}.xlsx",
        }.get(format_normalised, f"rule_extraction_{timestamp}.{format_normalised}")

        if output_path:
            path = Path(output_path)
        else:
            path = Path("output") / "exports" / default_name
        path.parent.mkdir(parents=True, exist_ok=True)

        if format_normalised == "json":
            payload = {
                "summary": self._build_summary(documents),
                "results": documents,
            }
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        else:
            schema_norm = (schema or "full").strip().lower()
            if schema_norm in {"dfm", "dfm_strict", "strict_dfm"}:
                rows = self._flatten_rules_dfm_strict(documents)
            else:
                rows = self._flatten_rules(documents, include_metadata=include_metadata)
            if format_normalised in {"csv", "tsv"}:
                delimiter = "," if format_normalised == "csv" else "\t"
                with path.open("w", newline="", encoding="utf-8") as handle:
                    fieldnames = sorted({key for row in rows for key in row})
                    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
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
        # Minimal counters (can be incremented in future enhancements)
        processing_stats = {
            "documents_processed": 0,
            "rules_extracted": 0,
        }
        return {
            "configuration": {
                "max_rules": self.max_rules,
                "max_concurrent_calls": settings.max_concurrent_calls,
                "throttle_seconds": settings.throttle_seconds,
                "enhanced_engine": bool(self.enhanced_engine),
                "use_qdrant": self.use_qdrant,
                "groq_model": getattr(settings, "groq_model", "unknown"),
            },
            "processing_stats": processing_stats,
        }

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def validate_against_hcl_dataset(self, hcl_csv_path: str) -> Dict[str, Any]:
        """Proxy HCL validation to the enhanced engine when available.

        This provides a stable entry point for UI layers that expect the
        legacy ``validate_against_hcl_dataset`` API on the production system.
        """

        if not self.enhanced_engine:
            raise RuntimeError("Enhanced engine is not enabled; HCL validation is unavailable.")

        return self.enhanced_engine.validate_against_hcl_dataset(hcl_csv_path)

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
        failed = sum(1 for doc in documents if doc.get("status") not in {"success", "no_rules"})
        total_processing_time = sum(float(doc.get("processing_time") or 0.0) for doc in documents)
        avg_processing_time = total_processing_time / len(documents) if documents else 0.0
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "documents": len(documents),
            "successful_documents": successful,
            "failed_documents": failed,
            "total_rules": total_rules,
            "total_processing_seconds": round(total_processing_time, 3),
            "avg_processing_seconds": round(avg_processing_time, 3),
            "average_confidence": (
                sum(doc.get("avg_confidence", 0.0) for doc in documents if doc.get("avg_confidence")) / successful
                if successful
                else 0.0
            ),
        }

    def _flatten_rules(self, documents: List[Dict[str, Any]], *, include_metadata: bool) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for doc in documents:
            base_row = {
                "filename": doc.get("filename"),
                "status": doc.get("status"),
                "rule_count": doc.get("rule_count", 0),
                "avg_confidence": doc.get("avg_confidence", 0.0),
                "chunks_processed": doc.get("chunks_processed"),
                "model_calls": doc.get("model_calls"),
                "processing_time": doc.get("processing_time"),
            }
            if include_metadata:
                if doc.get("document_context"):
                    base_row["document_context"] = self._serialise_field(doc.get("document_context"))
                if doc.get("document_metadata"):
                    base_row["document_metadata"] = self._serialise_field(doc.get("document_metadata"))

            if not doc.get("rules"):
                rows.append(base_row)
                continue

            for idx, rule in enumerate(doc["rules"], start=1):
                row = dict(base_row)
                row["rule_index"] = idx
                for key, value in rule.items():
                    row[key] = self._serialise_field(value)
                rows.append(row)
        return rows

    def _flatten_rules_dfm_strict(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Flatten extraction results into a strict, DFM-focused tabular schema.

        This intentionally avoids dumping every raw field. It provides a minimal
        set of columns geared toward downstream DFM triage and governance.

        Columns:
          - rule_id: stable hash of normalized rule text (+ document)
          - rule_text: rule sentence
          - domain: where it applies (best-effort)
          - rule_type: dimensional/material/process/general (when available)
          - complexity: 0..1 (when available)
          - priority: high/medium/low (importance to follow)
          - risk: high/medium/low (impact if violated)
          - confidence: 0..1
          - source_document
          - extraction_method
        """

        def _norm_text(text: str) -> str:
            return " ".join((text or "").strip().split()).lower()

        def _pick_domain(doc: Dict[str, Any], rule: Dict[str, Any]) -> str:
            filename = (doc.get("filename") or "").strip()
            filename_lower = filename.lower()
            ctx = doc.get("document_context") or {}
            tech_domain = (ctx.get("technical_domain") or "").strip()
            if tech_domain:
                base = tech_domain
            elif "sheet" in filename_lower and "metal" in filename_lower:
                base = "sheet metal"
            else:
                base = (ctx.get("industry_sector") or "general").strip() or "general"

            primary_feature = (rule.get("primary_feature") or "").strip()
            if primary_feature:
                return f"{base} / {primary_feature}"
            return base

        def _priority_from_text(text: str) -> str:
            t = _norm_text(text)
            if any(k in t for k in [" must ", " shall ", " required ", " do not ", " never "]):
                return "high"
            if any(k in t for k in [" should ", " recommended "]):
                return "medium"
            return "low"

        def _risk_from_text(text: str) -> str:
            t = _norm_text(text)
            high_markers = [
                "tolerance",
                "±",
                "minimum",
                "maximum",
                "clearance",
                "interference",
                "thickness",
                "radius",
                "failure",
                "crack",
                "fracture",
                "safety",
                "critical",
            ]
            if any(m in t for m in high_markers):
                return "high"
            # Any numeric constraint usually implies measurable defect risk.
            if any(ch.isdigit() for ch in t):
                return "medium"
            if any(k in t for k in ["avoid", "ensure", "prevent"]):
                return "medium"
            return "low"

        rows: List[Dict[str, Any]] = []
        for doc in documents:
            rules = doc.get("rules") or []
            if not rules:
                continue

            source_document = doc.get("filename") or doc.get("document_name") or "unknown"
            for rule in rules:
                rule_text = (rule.get("rule_text") or "").strip()
                if not rule_text:
                    continue

                normalized = _norm_text(rule_text)
                stable_basis = f"{source_document}::{normalized}".encode("utf-8")
                rule_id = hashlib.sha1(stable_basis).hexdigest()

                confidence = rule.get("confidence_score")
                try:
                    confidence_val = float(confidence) if confidence is not None else 0.0
                except Exception:
                    confidence_val = 0.0

                complexity = rule.get("complexity_score")
                try:
                    complexity_val = float(complexity) if complexity is not None else 0.0
                except Exception:
                    complexity_val = 0.0

                rows.append(
                    {
                        "rule_id": rule_id,
                        "rule_text": rule_text,
                        "domain": _pick_domain(doc, rule),
                        "rule_type": (rule.get("rule_type") or "").strip() or "general",
                        "complexity": round(max(0.0, min(1.0, complexity_val)), 4),
                        "priority": _priority_from_text(rule_text),
                        "risk": _risk_from_text(rule_text),
                        "confidence": round(max(0.0, min(1.0, confidence_val)), 4),
                        "source_document": source_document,
                        "extraction_method": (rule.get("extraction_method") or "").strip(),
                    }
                )

        return rows

    @staticmethod
    def _serialise_field(value: Any) -> Any:
        if isinstance(value, set):
            value = sorted(value)
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return value


__all__ = ["ProductionRuleExtractionSystem"]