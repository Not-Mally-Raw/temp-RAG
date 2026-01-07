import os
from pathlib import Path
import asyncio
from core.production_system import ProductionRuleExtractionSystem

PDF_PATH = os.environ.get("DFM_PDF", "/opt/anaconda3/Design For Manufacturability Guidelines - Sheetmetal.pdf")
OUT_DIR = Path(os.environ.get("DFM_OUT", "/opt/anaconda3/rework-RAG-for-HCLTech/output/exports"))

async def run():
    system = ProductionRuleExtractionSystem()
    payload = await system.process_document_advanced(PDF_PATH, enable_enhancement=False)
    # Export using the minimalist dfm_strict schema we just updated
    out_path = OUT_DIR / "dfm_level1_minimal.csv"
    system.export_results([payload], format="csv", output_path=str(out_path), schema="dfm_strict")
    print(str(out_path))

if __name__ == "__main__":
    asyncio.run(run())
