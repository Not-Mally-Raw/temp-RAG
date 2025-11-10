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
        "rule_text": rule.get("rule_text", ""),
        "rule_category": rule.get("rule_category", rule.get("category", "")),
        "rule_type": rule.get("rule_type", ""),
        "confidence_score": rule.get("confidence_score", rule.get("confidence", 0.0)),
        "manufacturing_relevance": rule.get("manufacturing_relevance"),
        "priority": rule.get("priority"),
        "source_document_section": rule.get("document_section"),
        "operator": rule.get("operator"),
        "value": json.dumps(rule.get("value")) if isinstance(rule.get("value"), list) else rule.get("value"),
        "unit": rule.get("unit"),
        "tolerance": json.dumps(rule.get("tolerance")) if isinstance(rule.get("tolerance"), list) else rule.get("tolerance"),
        "extraction_method": rule.get("extraction_method"),
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
        default=Path("/opt/anaconda3/RAG-System/data/real_documents"),
        help="Directory or file containing PDF documents",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/opt/anaconda3/RAG-System/data/extracted_rules.csv"),
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
                enable_enhancement=True,
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


if __name__ == "__main__":
    main()
