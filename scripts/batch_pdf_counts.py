"""Batch-run extraction on a folder of PDFs and report rule counts.

This is intentionally minimal and geared toward quick benchmarking.

Example:
  /opt/anaconda3/.venv/bin/python scripts/batch_pdf_counts.py \
    --inputs /opt/anaconda3/RAG-System/data \
    --inputs /opt/anaconda3/Phase-3-Final-master/data \
    --recall \
    --out output/batch_rule_counts.csv

Notes:
- Uses the enhanced engine.
- Writes a CSV with per-PDF counts + key diagnostics.
- Continues on errors and records them in the CSV.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv

# Ensure repo root is importable even when running from another CWD.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.enhanced_rule_engine import EnhancedConfig
from core.production_system import ProductionRuleExtractionSystem
from core.rule_extraction import RuleExtractionSettings


@dataclass
class Row:
    path: str
    filename: str
    status: str
    rule_count: int
    total_chunks: Optional[int]
    raw_before_post: Optional[int]
    after_dedup: Optional[int]
    after_quality: Optional[int]
    after_cluster: Optional[int]
    avg_confidence: Optional[float]
    processing_time_s: Optional[float]
    processed_chunks: Optional[int]
    tpd_aborted: Optional[bool]
    tpd_blocked_until: Optional[float]
    hit_rule_cap: Optional[bool]
    rules_before_cap: Optional[int]
    max_rules_cap: Optional[int]
    error: str


def _iter_pdfs(inputs: Iterable[str]) -> List[Path]:
    pdfs: List[Path] = []
    for root in inputs:
        p = Path(root).expanduser().resolve()
        if p.is_file() and p.suffix.lower() == ".pdf":
            pdfs.append(p)
            continue
        if p.is_dir():
            pdfs.extend(sorted(p.rglob("*.pdf")))
    # de-dupe
    seen = set()
    out: List[Path] = []
    for p in pdfs:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _load_existing_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load previously-written rows keyed by absolute PDF path."""

    if not path.exists():
        return {}

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: Dict[str, Dict[str, Any]] = {}
        for row in reader:
            key = (row.get("path") or "").strip()
            if not key:
                continue
            rows[key] = dict(row)
        return rows


def _write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write all rows with a stable header that includes all fields."""

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _build_system(*, recall: bool) -> ProductionRuleExtractionSystem:
    load_dotenv(override=False)
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_key:
        raise SystemExit("Missing GROQ_API_KEY. Add it to .env or env vars.")

    model = os.getenv("GROQ_MODEL", EnhancedConfig().groq_model)

    pipeline_settings = RuleExtractionSettings(
        groq_api_key=groq_key,
        groq_model=model,
        max_concurrent_calls=1,
        throttle_seconds=0.5,
        max_retries=4,
        retry_backoff_seconds=3.0,
        request_timeout=180.0,
        max_rules_total=500,
    )

    enhanced_config = EnhancedConfig(
        groq_api_key=groq_key,
        groq_model=model,
        recall_mode=bool(recall),
        allow_bulk_extraction=bool(recall),
        extraction_mode="raw" if recall else "structured",
        enable_local_heuristic=True,
        max_chunks_per_document=120 if recall else 25,
        max_rules_per_document=500 if recall else 150,
        api_request_delay=0.0,

        # resumable extraction
        enable_chunk_cache=True,
    )

    return ProductionRuleExtractionSystem(
        groq_api_key=groq_key,
        use_qdrant=False,
        pipeline_settings=pipeline_settings,
        enable_enhanced=True,
        enhanced_config=enhanced_config,
    )


async def _run_one(system: ProductionRuleExtractionSystem, pdf: Path) -> Dict[str, Any]:
    return await system.process_document_advanced(str(pdf), enable_enhancement=True, enable_validation=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch rule-count extraction for PDFs")
    parser.add_argument("--inputs", action="append", help="Folder or PDF path (repeatable)")
    parser.add_argument("--pdf", action="append", help="Single PDF path (repeatable)")
    parser.add_argument("--recall", action="store_true", help="Use high-recall (raw/bulk) extraction")
    parser.add_argument("--out", default="output/batch_rule_counts.csv", help="Output CSV path")
    parser.add_argument(
        "--max-docs",
        type=int,
        default=1,
        help="How many documents to process in this run (default: 1)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip PDFs already present in the output CSV",
    )
    parser.add_argument(
        "--stop-on-rate-limit",
        action="store_true",
        help="Exit immediately if a TPD/rate-limit abort is detected",
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
    args = parser.parse_args()

    requested: List[str] = []
    if args.inputs:
        requested.extend(args.inputs)
    if args.pdf:
        requested.extend(args.pdf)
    if not requested:
        raise SystemExit("Provide --pdf <file.pdf> and/or --inputs <folder>")

    pdfs = _iter_pdfs(requested)
    if not pdfs:
        raise SystemExit("No PDFs found in the provided inputs")

    system = _build_system(recall=bool(args.recall))
    # Configure enhanced chunk cache settings (if enhanced engine is available)
    if system.enhanced_engine is not None:
        system.enhanced_engine.config.chunk_cache_dir = str(args.cache_dir)
        system.enhanced_engine.config.resume_from_cache = bool(args.resume_chunks)

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_rows = _load_existing_rows(out_path)
    ordered_rows: List[Dict[str, Any]] = list(existing_rows.values())

    start = datetime.now(UTC)
    processed = 0

    for idx, pdf in enumerate(pdfs, start=1):
        if args.resume and str(pdf) in existing_rows:
            continue
        if processed >= int(args.max_docs or 0):
            break

        print(f"[{idx}/{len(pdfs)}] {pdf}")
        try:
            payload = asyncio.run(_run_one(system, pdf))
            stats = payload.get("extraction_stats") or {}

            status = str(payload.get("status") or "unknown")
            # Prefer enhanced-engine status signals when available.
            if stats.get("status"):
                status = str(stats.get("status"))
            elif stats.get("tpd_aborted"):
                status = "partial_rate_limited" if int(stats.get("raw_rules_before_postprocessing") or 0) > 0 else "rate_limited"

            row = asdict(
                Row(
                    path=str(pdf),
                    filename=str(payload.get("filename") or pdf.name),
                    status=status,
                    rule_count=int(payload.get("rule_count") or len(payload.get("rules") or [])),
                    total_chunks=stats.get("total_chunks"),
                    raw_before_post=stats.get("raw_rules_before_postprocessing"),
                    after_dedup=stats.get("rules_after_deduplication"),
                    after_quality=stats.get("rules_after_quality_filter"),
                    after_cluster=stats.get("rules_after_clustering"),
                    avg_confidence=payload.get("avg_confidence"),
                    processing_time_s=payload.get("processing_time"),
                    processed_chunks=stats.get("processed_chunks"),
                    tpd_aborted=stats.get("tpd_aborted"),
                    tpd_blocked_until=stats.get("tpd_blocked_until"),
                    hit_rule_cap=stats.get("hit_rule_cap"),
                    rules_before_cap=stats.get("rules_before_cap"),
                    max_rules_cap=stats.get("max_rules_cap"),
                    error="",
                )
            )
        except Exception as exc:
            row = asdict(
                Row(
                    path=str(pdf),
                    filename=pdf.name,
                    status="failed",
                    rule_count=0,
                    total_chunks=None,
                    raw_before_post=None,
                    after_dedup=None,
                    after_quality=None,
                    after_cluster=None,
                    avg_confidence=None,
                    processing_time_s=None,
                    processed_chunks=None,
                    tpd_aborted=None,
                    tpd_blocked_until=None,
                    hit_rule_cap=None,
                    rules_before_cap=None,
                    max_rules_cap=None,
                    error=str(exc),
                )
            )

        # Upsert + flush incremental progress so you can monitor while it runs
        existing_rows[str(pdf)] = row
        ordered_rows = list(existing_rows.values())
        _write_rows(out_path, ordered_rows)

        processed += 1

        if args.stop_on_rate_limit and row.get("status") in {"rate_limited", "partial_rate_limited"}:
            raise SystemExit(f"Stopped due to rate limiting. blocked_until={row.get('tpd_blocked_until')}")

    dur = (datetime.now(UTC) - start).total_seconds()
    print(f"Done. Wrote {out_path} in {dur:.1f}s")


if __name__ == "__main__":
    main()
