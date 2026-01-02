"""Quick CLI runner for rule extraction.

Usage:
  /opt/anaconda3/.venv/bin/python scripts/quick_extract.py \
    --pdf "/opt/anaconda3/Design For Manufacturability Guidelines - Sheetmetal.pdf" \
    --recall

This script intentionally loads `.env` so secrets do not need to be passed via
command line arguments.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

# Ensure repo root is importable even when running from another CWD.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.enhanced_rule_engine import EnhancedConfig
from core.production_system import ProductionRuleExtractionSystem
from core.rule_extraction import RuleExtractionSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run rule extraction on a single document")
    parser.add_argument("--pdf", required=True, help="Path to the input PDF")
    parser.add_argument(
        "--recall",
        action="store_true",
        help="Enable high-recall settings (bulk extraction, relaxed filters)",
    )
    parser.add_argument(
        "--max-rules",
        type=int,
        default=500,
        help="Max rules to return (caps fast + enhanced outputs)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=120,
        help="Max enhanced chunks to process",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional path to write full JSON result",
    )
    parser.add_argument(
        "--resume-chunks",
        action="store_true",
        help="Resume from cached per-chunk results when available",
    )
    parser.add_argument(
        "--cache-dir",
        default="output/chunk_cache",
        help="Directory for per-chunk cache files",
    )
    return parser


async def run(
    pdf_path: str,
    *,
    recall: bool,
    max_rules: int,
    max_chunks: int,
    out_path: str,
    resume_chunks: bool,
    cache_dir: str,
) -> None:
    # Load `.env` from repo root (or cwd)
    load_dotenv(override=False)

    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        raise SystemExit(
            "Missing GROQ_API_KEY. Add it to your environment or place it in rework-RAG-for-HCLTech/.env"
        )

    pdf = Path(pdf_path).expanduser().resolve()
    if not pdf.exists():
        raise SystemExit(f"PDF not found: {pdf}")

    model = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

    pipeline_settings = RuleExtractionSettings(
        groq_api_key=groq_key,
        groq_model=model,
        max_concurrent_calls=1,
        throttle_seconds=0.5,
        max_retries=3,
        retry_backoff_seconds=3.0,
        request_timeout=120.0,
        max_rules_total=max_rules,
    )

    enhanced_config = EnhancedConfig(
        groq_api_key=groq_key,
        groq_model=model,
        recall_mode=bool(recall),
        enable_local_heuristic=True,
        max_chunks_per_document=max_chunks,
        max_rules_per_document=max_rules,
        api_request_delay=0.0,
        # keep explicit for clarity; recall_mode will also force these
        allow_bulk_extraction=bool(recall),
        extraction_mode="raw" if recall else "structured",

        # resumable extraction
        enable_chunk_cache=True,
        resume_from_cache=bool(resume_chunks),
        chunk_cache_dir=str(cache_dir),
    )

    system = ProductionRuleExtractionSystem(
        groq_api_key=groq_key,
        use_qdrant=False,
        pipeline_settings=pipeline_settings,
        enable_enhanced=True,
        enhanced_config=enhanced_config,
    )

    print(f"Processing: {pdf}")
    result = await system.process_document_advanced(str(pdf), enable_enhancement=True, enable_validation=False)

    rules = result.get("rules", []) or []
    print("Status:", result.get("status"))
    print("Rules:", len(rules))

    stats = result.get("extraction_stats") or {}
    if stats:
        print(
            "Chunks:",
            stats.get("total_chunks"),
            "RawBefore:",
            stats.get("raw_rules_before_postprocessing"),
            "Deduped:",
            stats.get("rules_after_deduplication"),
            "AfterFilter:",
            stats.get("rules_after_quality_filter"),
        )

    # quick sanity preview
    for idx, rule in enumerate(rules[:10], start=1):
        text = (rule.get("rule_text") or "").replace("\n", " ").strip()
        print(f"{idx}.", text[:160])

    if out_path:
        out = Path(out_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print("Wrote:", str(out))


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(
        run(
            args.pdf,
            recall=args.recall,
            max_rules=args.max_rules,
            max_chunks=args.max_chunks,
            out_path=args.out,
            resume_chunks=args.resume_chunks,
            cache_dir=args.cache_dir,
        )
    )


if __name__ == "__main__":
    main()
