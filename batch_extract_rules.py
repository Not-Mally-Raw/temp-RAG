from __future__ import annotations
import argparse
import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from core.production_system import ProductionRuleExtractionSystem
from core.rule_extraction import RuleExtractionSettings
from core.enhanced_rule_engine import EnhancedConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def find_pdf_documents(root: Path):
    if root.is_file() and root.suffix.lower() == ".pdf":
        return [root]
    return sorted(p for p in root.rglob("*.pdf"))


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("output/compiled_rules"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    args = parser.parse_args()

    if args.env.exists():
        load_dotenv(args.env)

    pdfs = find_pdf_documents(args.input)
    args.out.mkdir(parents=True, exist_ok=True)

    system = ProductionRuleExtractionSystem(enable_enhanced=True)

    for pdf in pdfs:
        logger.info("Processing %s", pdf)
        payload = await system.process_document_advanced(str(pdf), enable_enhancement=True)
        out_file = args.out / f"{pdf.stem}.json"

        final = {
            "source_document": pdf.name,
            "compiled_at": datetime.utcnow().isoformat(),
            "rules": payload.get("rules", [])
        }

        out_file.write_text(json.dumps(final, indent=2), encoding="utf-8")
        logger.info("Wrote %s", out_file)


if __name__ == "__main__":
    asyncio.run(main())
