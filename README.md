# Enhanced Manufacturing Rule Extraction System

This repository contains the end-to-end pipeline we are using to extract structured manufacturing rules from large PDF specifications, reports, and checklists. The implementation favours transparency over marketing claims: every section below reflects the current behaviour observed in the November 2025 builds.

## System Overview

- **Primary entry point**: `ProductionRuleExtractionSystem` orchestrates both the fast pipeline (`core/rule_extraction.py`) and the enhanced Groq-backed engine (`core/enhanced_rule_engine.py`).
- **Batch automation**: `batch_extract_rules.py` walks the `/opt/anaconda3/RAG-System/data` tree, streams rules into `data/extracted_rules.csv`, and checkpoints after each document.
- **Streamlit UI**: `enhanced_streamlit_app.py` exposes interactive processing, analytics, and a live view of the latest CSV export.
- **Validation**: `validate_system.py` confirms environment setup, model availability, and runs a focused extraction sanity test.

## Model Architecture & Justification

| Component | Model / Library | Role | Why it is kept |
|-----------|-----------------|------|----------------|
| Enhanced extraction | `meta-llama/llama-4-scout-17b-16e-instruct` (via Groq) | High-fidelity structured extraction for the enhanced path | Best balance of recall and well-formed JSON on long technical sentences during internal tests. |
| Fallback extraction | `llama-3.1-8b-instant` (Groq) | Automatic failover when the 17B model is unavailable | Keeps the system responsive under rate limits or regional outages; slightly lower confidence but still acceptable for triage. |
| Fast pipeline | Same Groq model, single-call JSON output (`core/rule_extraction.py`) | Lower-latency batch extraction when the enhanced chain is disabled | Enables ~2× faster throughput for bulk scans where enrichment is unnecessary. |
| Embeddings / clustering | `scikit-learn` TF‑IDF and optional FAISS | Deduplication, clustering, and similarity lookups | Lightweight and fully local; no external service required for semantic grouping. |

## Processing Pipeline

1. **Document ingestion** (`DocumentLoader`): converts PDF/DOCX/TXT to text, capturing metadata such as page counts and readability scores.
2. **Chunking** (`TextChunker` + semantic heuristics): selects manufacturing-dense segments while respecting Groq token limits.
3. **Fast extraction** (`ChunkProcessor`): prompts Groq for JSON rules, normalises them, and applies adaptive rate limiting.
4. **Enhanced enrichment** (`EnhancedRuleEngine`): optional LangChain flow adds context analysis, structured Pydantic parsing, and rule quality signals (confidence tags, readability, implicit requirements).
5. **Post-processing**: semantic deduplication, clustering (opt-in), and export via pandas/Streamlit.
6. **Batch export**: CSV table stored at `/opt/anaconda3/RAG-System/data/extracted_rules.csv`; Streamlit surfaces the same table for review.

## Current Results (November 2025)

- `python validate_system.py` using the 17B Groq model extracts one rule from the test sentence in ~8 seconds with confidence 1.0. When the 17B model is unavailable, the fallback 8B model completes in ~4 seconds with 0.9 confidence.
- The batch run against `3M Pharmaceutical-Industry-Best-Practice.pdf` (12 chunks, ~7.9k tokens) currently produces partial CSV output. Remaining issues are Pydantic normalisation (category mapping and timestamps) and Groq 429 throttling; both are handled by configuration flags in `batch_extract_rules.py` but still require conservative settings (1 request/sec) for the longest PDFs.
- Manual prompt tests against the sample text block return four distinct rules with full confidence after normalising `.env` loading.

## Input Documents

Real PDF and Excel sources bundled in `/opt/anaconda3/RAG-System/data/real_documents`:

- `3M Pharmaceutical-Industry-Best-Practice.pdf`
- `DFMA-Sample-Checklist.xlsx`
- `Lockheed Martin engineering flowdown reqguide.pdf`
- `Nestle-answers-forests-2023.pdf`
- `Northrop Grumman Quality Assurance - Quality standards.pdf`
- `Supply-Chain-Solutions-Guide-02-24.pdf`
- `Texas Instruments.pdf`
- `edh_ed_handbook-683689-666980.pdf`

These filenames are read verbatim by the batch runner and will appear in the CSV export.

## Technology Stack

- **LLM orchestration**: LangChain 0.2.x, langchain-groq, Groq hosted models.
- **Data processing**: pandas, numpy, PyMuPDF, python-docx, textstat.
- **Analytics & UI**: Streamlit, Plotly, structlog for JSON-formatted logs.
- **Machine learning utilities**: scikit-learn for TF‑IDF and clustering, FAISS (optional) for similarity search.
- **Configuration**: pydantic / pydantic-settings, python-dotenv for environment management.

## Setup & Usage

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Provide environment variables**
   ```bash
   cp .env.example .env  # if present
   echo "GROQ_API_KEY=your_actual_key" >> .env
   echo "GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct" >> .env
   ```

3. **Validate the installation**
   ```bash
   python validate_system.py
   ```

4. **Run the Streamlit interface**
   ```bash
   streamlit run enhanced_streamlit_app.py
   ```
   The “Document Processing” tab now displays the latest `extracted_rules.csv` and lets you download it directly.

5. **Bulk extraction without UI**
   ```bash
   python batch_extract_rules.py \
     --input /opt/anaconda3/RAG-System/data/real_documents \
     --output /opt/anaconda3/RAG-System/data/extracted_rules.csv \
     --throttle 2.5 --max-concurrent 1 --document-concurrency 1 \
     --enhanced-delay 3.0 --max-chunks 9
   ```

## Known Limitations & Next Steps

- **Rate limits**: Groq free-tier caps (≈30 RPM, 6k TPM) require the conservative throttle settings provided. Increase cautiously if you upgrade your plan.
- **Schema drift**: Some documents emit free-form categories (e.g., “Occupational Health and Safety”). Recent normalization fixes map these back to enum values, but keep an eye on the CSV for new variants.
- **Long documents**: Files above ~10k tokens per chunk can still trigger retries. Consider lowering `--max-chunks` or pre-splitting PDFs in those cases.

## Contributing & Maintenance

- Automated tests live under `tests/`. Use `pytest` before merging changes.
- Structured logging (structlog) keeps API failures and retries discoverable—check logs when diagnosing extraction gaps.
- Keep `requirements.txt` in sync with any new imports so the project is reproducible after cloning.

## License

See `LICENSE.router` (router integration license) or project-level license as applicable.