# FILE_DESCRIPTIONS — RAG-System (stable-oct26-update)

This file contains a concise, unique one-line description for each file and important folder in this project export.

Top-level files
- `comprehensive_app.py`: Aggregated app that runs the full RAG workflow and integrates multiple sub-app pages for end-to-end demos.
- `config.py`: Project configuration and constants (paths, thresholds, environment-specific settings).
- `ENHANCED_INTEGRATION_GUIDE.md`: Integration notes and step-by-step guide for the enhanced RAG components.
- `enhanced_main_app.py`: Streamlit entrypoint for the enhanced RAG UI with rule generation and QA pages wired in.
- `enhanced_qa_system.py`: Orchestrates enhanced question-answering flows, adapter logic between retriever and LLM.
- `final_test.py`: Lightweight final system validator — checks syntax/structure (modified to avoid heavy model loads for CI-friendly runs).
- `main_app.py`: Original/main Streamlit application for the RAG demo (legacy entrypoint).
- `main_rag_app.py`: RAG-focused app wiring, focusing on document ingestion, retrieval, and QA.
- `manufacturing_rules_app.py`: Specialized app page for manufacturing rule extraction and presentation.
- `README.md`: Project overview, installation, and usage notes.
- `requirements.txt`: Python dependencies pinned/required for running the project.
- `rule_processor.py`: Utilities for post-processing extracted rules, normalization and CSV export.
- `run_full_system_test.py`: End-to-end runner for the system to execute full processing and produce test results.
- `simple_test_app.py`: Minimal Streamlit app used for smoke testing or dev previews.
- `test_comprehensive_system.py`: Test harness for executing comprehensive system checks.

`core/` folder
- `core/enhanced_rag_db.py`: Database helpers for the enhanced RAG flow (storage/backing store helpers).
- `core/implicit_rule_extractor.py`: Algorithms to infer implicit rules from parsed text and structured outputs.
- `core/llm_prompts.py`: Centralized LLM prompt templates and prompt-building helpers.
- `core/monitoring.py`: Monitoring and logging utilities used by the RAG pipeline for health checks and telemetry.
- `core/rag_database.py`: Generic RAG database abstraction for storing/retrieving documents and embeddings.
- `core/rag_pipeline_integration.py`: Glue logic connecting extractors, generators, retriever and the LLM pipeline.
- `core/rag_system.py`: High-level RAG orchestration class that wires together ingestion, indexing, and query handling.
- `core/rule_extractor.py`: Primary rule-extraction logic parsing text and recognizing rule-like constructs.
- `core/rule_generator.py`: Generates candidate rules and formats them for display/export.
- `core/streamlit_utils.py`: Streamlit helpers (UI components, state helpers, layout helpers used by pages).
- `core/text_processing.py`: Text normalization, tokenization, and preprocessing utilities.
- `core/universal_classifier.py`: Lightweight classifier utilities used across the pipeline for document labeling.

`data/` folder
- `data/sample_documents.py`: Script or data mapping to sample documents used in demos and tests.
- `data/train_finaldata.csv`: CSV training dataset used for model training or classifier tuning.
- `data/real_documents/`: Directory containing real-world example documents used for extraction/regression tests.
- `data/sample_documents/`: Sample document files used by development and tests.

`extractors/` folder
- `extractors/image.py`: Image extractor and image-specific processing (e.g., OCR or feature extraction from images).
- `extractors/robust_pdf_processor.py`: PDF parsing/processing utilities handling complex layout and tables robustly.
- `extractors/table.py`: Specialized table extraction and conversion to structured data.
- `extractors/text_enhanced.py`: Enhanced text extractor with additional processing and heuristics.
- `extractors/text.py`: Core text extraction functions used for simple document types.

`generators/` folder
- `generators/features.py`: Feature generation utilities used by rule generation and classifiers.

`pages/` folder (Streamlit pages)
- `pages/automated_testing.py`: Page to configure and run automated test scenarios on uploaded documents.
- `pages/consolidated_rules.py`: Page to display consolidated/extracted rules and allow CSV export.
- `pages/enhanced_qa.py`: QA page for interactive question answering using the enhanced RAG pipeline.
- `pages/rule_generation.py`: Page for generating candidate rules from documents and viewing intermediate outputs.
- `pages/smart_uploader.py`: Document uploader page with preprocessing pipeline hooks and status UI.

`rag_db/` folder
- `rag_db/chroma.sqlite3`: ChromaDB SQLite store used for vector/embedding storage (binary DB file — rebuildable).

`test_results/` folder
- `test_results/3M Pharmaceutical-Industry-Best-Practice.pdf_rules.csv`: Example rules CSV from a previous run for that document.
- `test_results/consolidated_all_rules.csv`: Consolidated CSV of rules aggregated across multiple test runs.
- `test_results/full_system_test_20251026_192115.json`: Recorded results from a full system test run (JSON with run metadata).
- `test_results/Injection_Molding_Design_Guidelines_2017.pdf_rules.csv`: Example extracted rules CSV for that PDF.
- `test_results/Nestle-answers-forests-2023.pdf_rules.csv`: Example extracted rules CSV for that PDF.
- `test_results/Xometry Sheet Metal Design Guide 2020.pdf_rules.csv`: Example extracted rules CSV for that PDF.

`tests/` folder
- `tests/test_rag_system.py`: Unit/integration tests for the RAG system components (CI-targeted tests).

Notes
- These one-line descriptions are written to be unique and succinct so the repository commit includes a manifest of what each artifact is for reviewers and for the GitHub repo description requirement.
