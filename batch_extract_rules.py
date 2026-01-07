"""Batch extract manufacturing rules from PDFs and export to CSV."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import pandas as pd
from dotenv import load_dotenv

from core.production_system import ProductionRuleExtractionSystem
from core.rule_extraction import RuleExtractionSettings
from core.enhanced_rule_engine import EnhancedConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _format_applicability(rule: Dict) -> str:
    app = rule.get("applicability_structured")
    if not isinstance(app, dict):
        app_list = rule.get("applicability")
        if isinstance(app_list, list) and app_list and isinstance(app_list[0], dict):
            app = app_list[0]
    if not isinstance(app, dict):
        app = {}

    parts: List[str] = []
    for key in ("process", "material", "geometry", "location", "feature", "plane_condition"):
        value = app.get(key)
        if value is None:
            continue
        value_str = str(value).strip()
        if not value_str or value_str.lower() in {"unknown", "null"}:
            continue
        parts.append(f"{key}={value_str}")

    domain = rule.get("domain")
    if isinstance(domain, str) and domain.strip():
        parts.insert(0, f"domain={domain.strip()}")

    scope = rule.get("scope_domain")
    if isinstance(scope, dict):
        scope_bits = [str(scope.get(k) or "").strip() for k in ("process", "item", "feature")]
        scope_bits = [b for b in scope_bits if b]
        if scope_bits:
            parts.append(f"scope={'/'.join(scope_bits)}")

    return "; ".join(parts)


def _format_constraints(rule: Dict) -> str:
    constraints = rule.get("constraints")
    if not isinstance(constraints, list) or not constraints:
        return ""

    rendered: List[str] = []
    for c in constraints:
        if not isinstance(c, dict):
            continue
        param = str(c.get("parameter") or c.get("target") or c.get("subject") or "").strip()
        op = str(c.get("operator") or c.get("constraint_type") or c.get("op") or "").strip()
        value = c.get("value")
        expr = c.get("expression")
        units = str(c.get("units") or c.get("unit") or "").strip()

        rhs = ""
        if isinstance(expr, str) and expr.strip():
            rhs = expr.strip()
        elif isinstance(value, (int, float)):
            rhs = str(value)
        elif isinstance(value, str):
            rhs = value.strip()
        elif value is not None:
            rhs = json.dumps(value, ensure_ascii=False)

        piece = ""
        if param and op and rhs:
            piece = f"{param} {op} {rhs}"
        elif param and rhs:
            piece = f"{param}: {rhs}"
        elif rhs:
            piece = rhs

        if piece and units:
            piece = f"{piece} {units}"

        if piece:
            rendered.append(piece)

    # If compound_logic is present, append a short readable hint.
    compound = rule.get("compound_logic")
    if isinstance(compound, list) and compound:
        for node in compound:
            if not isinstance(node, dict):
                continue
            fn = str(node.get("function") or "").strip()
            operands = node.get("operands")
            if fn and isinstance(operands, list) and operands:
                op_str = ", ".join(str(o) for o in operands if str(o).strip())
                if op_str:
                    rendered.append(f"{fn}({op_str})")

    return "; ".join(rendered)


def find_pdf_documents(root: Path) -> List[Path]:
    """Return every PDF found under ``root`` (recursive)."""

    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")

    if root.is_file():
        return [root] if root.suffix.lower() == ".pdf" else []

    return sorted(p for p in root.rglob("*.pdf") if p.is_file())


def _rule_to_row(rule: Dict, doc_path: Path, status: str) -> Dict:
    return {
        "source_file": doc_path.name,
        "source_path": str(doc_path),
        "status": status,
        "rule_text": rule.get("rule_text") or rule.get("canonical_statement", ""),
        "canonical_statement": rule.get("canonical_statement", ""),
        "rule_category": rule.get("rule_category", rule.get("category", "")),
        "rule_type": rule.get("rule_type", ""),
        "confidence_score": rule.get("confidence_score", rule.get("confidence", 0.0)),
        "confidence": rule.get("confidence_score", rule.get("confidence", 0.0)),
        "manufacturing_relevance": rule.get("manufacturing_relevance"),
        "priority": rule.get("priority"),
        "source_document_section": rule.get("document_section"),
        "operator": rule.get("operator"),
        "value": json.dumps(rule.get("value")) if isinstance(rule.get("value"), list) else rule.get("value"),
        "unit": rule.get("unit"),
        "tolerance": json.dumps(rule.get("tolerance")) if isinstance(rule.get("tolerance"), list) else rule.get("tolerance"),
        # drop extraction_method from CSV to keep Level-1 minimalist schema
        "intent": rule.get("intent"),
        "domain": rule.get("domain"),
        "applicability": _format_applicability(rule),
        "applicability_json": json.dumps(
            rule.get("applicability_structured")
            or rule.get("applicability")
            or rule.get("scope_domain")
            or {},
            ensure_ascii=False,
        ),
        "constraints": _format_constraints(rule),
        "constraints_json": (
            rule.get("constraints") if isinstance(rule.get("constraints"), str) else json.dumps(rule.get("constraints", []), ensure_ascii=False)
        ),
        "compound_logic": json.dumps(rule.get("compound_logic", []), ensure_ascii=False),
        "severity": rule.get("severity"),
        "validation_state": rule.get("validation_state"),
        "source_document": rule.get("source_document") or doc_path.name,
    }


async def extract_rules_from_documents(
    documents: Iterable[Path],
    *,
    enable_enhancement: bool = True,
    pipeline_settings: Optional[RuleExtractionSettings] = None,
    enhanced_config: Optional[EnhancedConfig] = None,
    progress_callback: Optional[Callable[[List[Dict]], None]] = None,
) -> List[Dict]:
    """Extract rules for each document and report incremental progress."""

    system = ProductionRuleExtractionSystem(
        enable_enhanced=enable_enhancement,
        use_qdrant=False,
        pipeline_settings=pipeline_settings,
        enhanced_config=enhanced_config,
    )

    if pipeline_settings:
        logger.info(
            "Rate limiting configured",
            extra={
                "max_concurrent_calls": pipeline_settings.max_concurrent_calls,
                "document_concurrency": pipeline_settings.max_documents_concurrency,
                "throttle_seconds": pipeline_settings.throttle_seconds,
            },
        )

    aggregated: List[Dict] = []
    for doc_path in documents:
        logger.info("Processing %s", doc_path)
        doc_rows: List[Dict] = []
        try:
            payload = await system.process_document_advanced(
                str(doc_path),
                enable_enhancement=enable_enhancement,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.exception("Failed to process %s", doc_path)
            doc_rows.append(
                {
                    "source_file": doc_path.name,
                    "source_path": str(doc_path),
                    "status": "error",
                    "error": str(exc),
                }
            )
        else:
            rules = payload.get("rules", [])
            if not rules:
                doc_rows.append(
                    {
                        "source_file": doc_path.name,
                        "source_path": str(doc_path),
                        "status": payload.get("status", "no_rules"),
                        "rule_text": "",
                    }
                )
            else:
                for rule in rules:
                    doc_rows.append(_rule_to_row(rule, doc_path, payload.get("status", "success")))

        if doc_rows:
            aggregated.extend(doc_rows)
            if progress_callback:
                progress_callback(doc_rows)

    return aggregated


def write_csv(rows: List[Dict], output_path: Path) -> None:
    if not rows:
        return

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Wrote %s rows to %s", len(df), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch extract manufacturing rules from PDFs")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/opt/anaconda3/rework-RAG-for-HCLTech/data"),
        help="Directory or file containing PDF documents",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/opt/anaconda3/rework-RAG-for-HCLTech/output/extracted_rules.csv"),
        help="Path to the CSV file that will be written",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=Path("/opt/anaconda3/rework-RAG-for-HCLTech/.env"),
        help="Optional path to the .env file with GROQ settings",
    )
    parser.add_argument(
        "--throttle",
        type=float,
        default=2.5,
        help="Seconds to wait between Groq requests (applies to fast path)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Maximum concurrent Groq calls per document",
    )
    parser.add_argument(
        "--document-concurrency",
        type=int,
        default=1,
        help="Maximum number of documents processed in parallel",
    )
    parser.add_argument(
        "--enhanced-delay",
        type=float,
        default=3.0,
        help="Seconds to pause between enhanced LLM calls",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=12,
        help="Maximum enhanced chunks per document (lower to reduce token usage)",
    )
    parser.add_argument(
        "--fast-only",
        action="store_true",
        help="Use the fast Level-1 extractor only (disable enhanced engine)",
    )
    args = parser.parse_args()

    if args.env.exists():
        load_dotenv(dotenv_path=args.env)
    else:
        logger.warning("No .env file found at %s", args.env)

    documents = find_pdf_documents(args.input)
    if not documents:
        logger.error("No PDF documents found in %s", args.input)
        return

    pipeline_settings = RuleExtractionSettings(
        throttle_seconds=max(0.0, args.throttle),
        max_concurrent_calls=max(1, args.max_concurrent),
        max_documents_concurrency=max(1, args.document_concurrency),
    )

    enhanced_config = EnhancedConfig(
        api_request_delay=max(0.0, args.enhanced_delay),
        max_chunks_per_document=max(1, args.max_chunks),
    )

    rows: List[Dict] = []

    def on_progress(new_rows: List[Dict]) -> None:
        if not new_rows:
            return
        rows.extend(new_rows)
        write_csv(rows, args.output)

    try:
        final_rows = asyncio.run(
            extract_rules_from_documents(
                documents,
                enable_enhancement=not args.fast_only,
                pipeline_settings=pipeline_settings,
                enhanced_config=enhanced_config,
                progress_callback=on_progress,
            )
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted by user; partial results (if any) saved to %s", args.output)
        return

    if final_rows and final_rows is not rows:
        rows = final_rows
    write_csv(rows, args.output)


# --- Adapter wiring (non-invasive) ---
from core.adapters import TextLoaderAdapter, ChunkerAdapter, RuleExtractorAdapter, ExporterAdapter

# Instantiate default adapters (lazy delegates to legacy implementations)
_loader_adapter = TextLoaderAdapter()
_chunker_adapter = ChunkerAdapter()
_extractor_adapter = RuleExtractorAdapter()
_exporter_adapter = ExporterAdapter()

# Maintain original function names as thin proxies so existing call-sites work unchanged.
def load_document(path):
    return _loader_adapter.load(path)

def chunk_text(document):
    return _chunker_adapter.chunk(document)

def extract_rules_from_chunk(chunk, context=None):
    return _extractor_adapter.extract(chunk, context)

def export_rules(rules, output_path):
    return _exporter_adapter.export(rules, output_path)
# --- end adapter wiring ---

# --- DI wiring: use the new orchestrator with default adapters ---
from core.orchestrator import default_production_system

# create a single system instance for the batch run (preserves legacy behavior)
_system = default_production_system()

# Replace legacy per-document orchestration with the orchestrator's method.
def process_single_document(path, output_path, **kwargs):
    """
    Existing callers expected a function that loads, chunks, extracts and exports.
    Forward to the ProductionRuleExtractionSystem to preserve behaviour while
    enabling dependency injection.
    """
    # keep prior logging/metrics around this call unchanged
    rules = _system.process_document(path, export_path=output_path)
    return rules
# --- end DI wiring ---

if __name__ == "__main__":
    main()
