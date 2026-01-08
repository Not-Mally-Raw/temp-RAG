import os
import json
from pathlib import Path
import asyncio
from datetime import datetime
from core.production_system import ProductionRuleExtractionSystem

PDF_PATH = os.environ.get("DFM_PDF", "input.pdf")
OUT_DIR = Path(os.environ.get("DFM_OUT", "output/compiled_rules"))

async def run():
    system = ProductionRuleExtractionSystem()
    payload = await system.process_document_advanced(PDF_PATH, enable_enhancement=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{Path(PDF_PATH).stem}.json"

    final = {
        "source_document": Path(PDF_PATH).name,
        "compiled_at": datetime.utcnow().isoformat(),
        "rules": payload.get("rules", [])
    }

    out_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print("Saved:", out_path)

if __name__ == "__main__":
    asyncio.run(run())
