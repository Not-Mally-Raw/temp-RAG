"""Minimal Streamlit UI for DFM rule extraction.

This UI is intentionally strict and simple:
- Single document (path or upload)
- Extract
- Show results + per-chunk diagnostics ("show the working")
- Export in strict DFM tabular schema

Run:
  /opt/anaconda3/.venv/bin/streamlit run simple_streamlit_app.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core.enhanced_rule_engine import EnhancedConfig
from core.production_system import ProductionRuleExtractionSystem
from core.rule_extraction import RuleExtractionSettings


def _dfm_strict_rows_from_result(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create a strict DFM-only table for display/export."""

    def _norm_text(text: str) -> str:
        return " ".join((text or "").strip().split()).lower()

    def _priority_from_text(text: str) -> str:
        t = _norm_text(text)
        if any(k in t for k in [" must ", " shall ", " required ", " do not ", " never "]):
            return "high"
        if any(k in t for k in [" should ", " recommended "]):
            return "medium"
        return "low"

    def _risk_from_text(text: str) -> str:
        t = _norm_text(text)
        high_markers = [
            "tolerance",
            "±",
            "minimum",
            "maximum",
            "clearance",
            "interference",
            "thickness",
            "radius",
            "failure",
            "crack",
            "fracture",
            "safety",
            "critical",
        ]
        if any(m in t for m in high_markers):
            return "high"
        if any(ch.isdigit() for ch in t):
            return "medium"
        if any(k in t for k in ["avoid", "ensure", "prevent"]):
            return "medium"
        return "low"

    filename = result.get("filename") or "unknown"
    filename_lower = filename.lower()
    base_domain = "sheet metal" if ("sheet" in filename_lower and "metal" in filename_lower) else "general"

    rows: List[Dict[str, Any]] = []
    for rule in result.get("rules", []) or []:
        rule_text = (rule.get("rule_text") or "").strip()
        if not rule_text:
            continue

        feature = (rule.get("primary_feature") or "").strip()
        domain = f"{base_domain} / {feature}" if feature else base_domain

        try:
            confidence_val = float(rule.get("confidence_score") or 0.0)
        except Exception:
            confidence_val = 0.0

        try:
            complexity_val = float(rule.get("complexity_score") or 0.0)
        except Exception:
            complexity_val = 0.0

        rows.append(
            {
                "rule": rule_text,
                "domain": domain,
                "complexity": round(max(0.0, min(1.0, complexity_val)), 4),
                "priority": _priority_from_text(rule_text),
                "risk": _risk_from_text(rule_text),
                "confidence": round(max(0.0, min(1.0, confidence_val)), 4),
                "rule_type": (rule.get("rule_type") or "").strip() or "general",
                "source_document": filename,
                "extraction_method": (rule.get("extraction_method") or "").strip(),
            }
        )

    return rows


@st.cache_resource
def _initialize_system(profile: str, *, allow_bulk: bool, enable_local_heuristics: bool) -> ProductionRuleExtractionSystem:
    load_dotenv(override=False)

    allow_fake_flag = os.getenv("ALLOW_FAKE_GROQ", "0") == "1"
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()

    # Mock mode should never silently override a real key.
    # If a real key exists, always run the real enhanced engine.
    using_mock = (not groq_api_key) and allow_fake_flag
    if not groq_api_key and not allow_fake_flag:
        raise RuntimeError("GROQ_API_KEY not found. Put it in .env or env vars.")
    if using_mock:
        # Fast pipeline has a built-in mock LLM client when ALLOW_FAKE_GROQ=1.
        groq_api_key = "mock_key"

    model = os.getenv("GROQ_MODEL", EnhancedConfig().groq_model)

    profile_norm = (profile or "balanced").strip().lower()
    if profile_norm == "fast scan":
        pipeline_settings = RuleExtractionSettings(
            groq_api_key=groq_api_key,
            groq_model=model,
            max_concurrent_calls=2,
            throttle_seconds=0.0,
            max_retries=3,
            retry_backoff_seconds=2.0,
            request_timeout=60.0,
            max_rules_total=120,
        )
        enhanced_config = EnhancedConfig(
            groq_api_key=groq_api_key,
            groq_model=model,
            allow_bulk_extraction=allow_bulk,
            extraction_mode="raw" if allow_bulk else "structured",
            recall_mode=bool(allow_bulk),
            enable_local_heuristic=enable_local_heuristics,
            max_chunks_per_document=20,
            max_rules_per_document=200,
        )
    elif profile_norm == "full coverage":
        pipeline_settings = RuleExtractionSettings(
            groq_api_key=groq_api_key,
            groq_model=model,
            max_concurrent_calls=1,
            throttle_seconds=1.0,
            max_retries=5,
            retry_backoff_seconds=5.0,
            request_timeout=120.0,
            max_rules_total=500,
        )
        enhanced_config = EnhancedConfig(
            groq_api_key=groq_api_key,
            groq_model=model,
            recall_mode=True,
            allow_bulk_extraction=True,
            extraction_mode="raw",
            enable_local_heuristic=enable_local_heuristics,
            max_chunks_per_document=120,
            max_rules_per_document=500,
            chunk_size=900,
            chunk_overlap=450,
        )
    else:
        pipeline_settings = RuleExtractionSettings(
            groq_api_key=groq_api_key,
            groq_model=model,
            max_concurrent_calls=1,
            throttle_seconds=0.5,
            max_retries=4,
            retry_backoff_seconds=3.0,
            request_timeout=90.0,
            max_rules_total=500,
        )
        enhanced_config = EnhancedConfig(
            groq_api_key=groq_api_key,
            groq_model=model,
            recall_mode=bool(allow_bulk),
            allow_bulk_extraction=allow_bulk,
            extraction_mode="raw" if allow_bulk else "structured",
            enable_local_heuristic=enable_local_heuristics,
            max_chunks_per_document=80 if allow_bulk else 25,
            max_rules_per_document=500 if allow_bulk else 150,
        )

    return ProductionRuleExtractionSystem(
        groq_api_key=groq_api_key,
        use_qdrant=False,
        pipeline_settings=pipeline_settings,
        enable_enhanced=not using_mock,
        enhanced_config=enhanced_config if not using_mock else None,
    )


def main() -> None:
    st.set_page_config(page_title="DFM Rule Extractor", layout="wide")

    st.title("DFM Rule Extractor")
    st.caption("Simple UI: pick 1 doc → extract → see diagnostics → export.")

    st.sidebar.header("Settings")
    profile = st.sidebar.selectbox("Profile", ["Fast scan", "Balanced", "Full coverage"], index=1)
    profile_norm = (profile or "").strip().lower()

    high_recall = st.sidebar.checkbox(
        "High recall (raw/bulk)",
        value=profile_norm in {"balanced", "full coverage"},
        help="Keeps rule counts stable on long documents.",
    )
    enable_local_heuristics = st.sidebar.checkbox("Local heuristics fallback", value=True)

    max_display = st.sidebar.slider("Max rows to display", 25, 300, 100, 25)
    export_format = st.sidebar.selectbox("Export", ["CSV", "Excel", "JSON"], index=0)

    # Initialize
    try:
        with st.spinner("Initializing engine..."):
            system = _initialize_system(profile, allow_bulk=high_recall, enable_local_heuristics=enable_local_heuristics)
    except Exception as exc:
        st.error(str(exc))
        return

    if os.getenv("ALLOW_FAKE_GROQ", "0") == "1" and not os.getenv("GROQ_API_KEY", "").strip():
        st.warning(
            "Running in MOCK mode (no GROQ_API_KEY). The app will return a small demo output and the rule count is NOT real. "
            "To see the actual rule count, set GROQ_API_KEY in .env and restart without relying on --mock."
        )

    st.subheader("1) Choose document")
    default_path = "/opt/anaconda3/Design For Manufacturability Guidelines - Sheetmetal.pdf"
    doc_path = st.text_input("Local path", value=default_path if os.path.exists(default_path) else "")
    uploaded = st.file_uploader("Or upload PDF", type=["pdf"], accept_multiple_files=False)

    chosen_path = ""
    if uploaded is not None:
        tmp = f"/tmp/{uploaded.name}"
        with open(tmp, "wb") as f:
            f.write(uploaded.getbuffer())
        chosen_path = tmp
    elif doc_path and os.path.exists(doc_path):
        chosen_path = doc_path

    st.subheader("2) Run")
    st.write(f"Profile: {profile} | High recall: {high_recall}")

    if st.button("Extract", type="primary", disabled=not bool(chosen_path)):
        with st.spinner("Extracting..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    system.process_document_advanced(
                        chosen_path,
                        enable_enhancement=True,
                        enable_validation=False,
                    )
                )
            finally:
                loop.close()

        st.session_state["result"] = result

    st.subheader("3) Results")
    result = st.session_state.get("result")
    if not result:
        st.info("Pick a document and click Extract.")
        return

    if result.get("status") != "success":
        st.error(result.get("error") or "Extraction failed")
        return

    stats = result.get("extraction_stats") or {}
    rules = result.get("rules", []) or []

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rules", result.get("rule_count", len(rules)))
    c2.metric("Avg confidence", f"{float(result.get('avg_confidence', 0.0)):.3f}")
    c3.metric("Chunks", stats.get("total_chunks", "-"))
    c4.metric("After dedupe", stats.get("rules_after_deduplication", "-"))

    with st.expander("Show working (diagnostics)", expanded=True):
        st.write(
            {
                "raw_before_post": stats.get("raw_rules_before_postprocessing"),
                "after_dedup": stats.get("rules_after_deduplication"),
                "after_quality": stats.get("rules_after_quality_filter"),
                "after_cluster": stats.get("rules_after_clustering"),
            }
        )
        per_chunk = stats.get("per_chunk_stats", []) or []
        if per_chunk:
            st.table(
                [
                    {
                        "chunk": c.get("chunk_index"),
                        "tokens": c.get("token_count"),
                        "mfg_score": c.get("manufacturing_score"),
                        "candidates": c.get("candidate_count"),
                        "accepted": c.get("accepted_count"),
                    }
                    for c in per_chunk
                ]
            )

    df = pd.DataFrame(_dfm_strict_rows_from_result(result))
    st.dataframe(df.head(max_display), use_container_width=True)
    if len(df) > max_display:
        st.info(f"Showing {max_display} of {len(df)}")

    if st.button("Export results"):
        with st.spinner("Preparing export..."):
            export_path = system.export_results(
                [result],
                export_format.lower(),
                include_metadata=False,
                schema="dfm_strict",
            )
        with open(export_path, "rb") as f:
            st.download_button(
                f"Download {export_format}",
                data=f.read(),
                file_name=Path(export_path).name,
                mime="application/octet-stream",
            )


if __name__ == "__main__":
    main()
