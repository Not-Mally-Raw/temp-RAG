"""Fast smoke test for the streamlined manufacturing rule extractor."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from core.production_system import ProductionRuleExtractionSystem


load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast smoke test for manufacturing rule extraction")
    parser.add_argument("--docs", nargs="*", default=[], help="Document paths (pdf/docx/txt)")
    parser.add_argument("--max-rules", type=int, default=60, help="Cap rules per document")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent documents")
    return parser.parse_args()


def resolve_documents(paths: List[str]) -> List[Path]:
    if paths:
        return [Path(p) for p in paths]
    default_dir = Path("data/smoke")
    if not default_dir.exists():
        default_dir.mkdir(parents=True, exist_ok=True)
        return []
    return [p for p in default_dir.iterdir() if p.suffix.lower() in {".pdf", ".txt", ".docx"}]


async def run_smoke(system: ProductionRuleExtractionSystem, docs: List[Path], concurrency: int) -> List[Dict[str, Any]]:
    return await system.batch_process_documents([str(doc) for doc in docs], concurrency=concurrency)


def summarise(results):
    successes = [r for r in results if r.get("status") == "success"]
    total_rules = sum(r.get("rule_count", 0) for r in successes)
    return {
        "documents": len(results),
        "successful": len(successes),
        "total_rules": total_rules,
        "avg_confidence": (
            sum(r.get("avg_confidence", 0.0) for r in successes) / len(successes)
            if successes
            else 0.0
        ),
        "rules_per_doc": (total_rules / len(successes)) if successes else 0.0,
    }


def main():
    args = parse_args()
    docs = resolve_documents(args.docs)
    if not docs:
        print("[ERROR] No documents found. Provide --docs or place files in data/smoke/.")
        sys.exit(1)

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        print("[ERROR] GROQ_API_KEY missing. Set it in .env or the shell.")
        sys.exit(2)

    system = ProductionRuleExtractionSystem(groq_api_key=api_key, max_rules=args.max_rules)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        results = loop.run_until_complete(run_smoke(system, docs, args.concurrency))
    finally:
        loop.close()

    payload = {"summary": summarise(results), "results": results}
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
