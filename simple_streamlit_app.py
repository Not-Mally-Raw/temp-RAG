from __future__ import annotations
import asyncio
import os
import tempfile
from pathlib import Path
import json
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from core.enhanced_rule_engine import EnhancedConfig
from core.production_system import ProductionRuleExtractionSystem
from core.rule_extraction import RuleExtractionSettings

st.set_page_config(page_title="DFM Rule Compiler", layout="wide")
st.title("DFM Rule Compiler")

load_dotenv(override=False)

st.sidebar.header("Settings")
profile = st.sidebar.selectbox("Profile", ["Fast scan", "Balanced", "Full coverage"], index=1)
high_recall = st.sidebar.checkbox("High recall (raw/bulk)", value=True)

@st.cache_resource
def get_system():
    key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", EnhancedConfig().groq_model)

    pipeline_settings = RuleExtractionSettings(groq_api_key=key, groq_model=model)
    enhanced_config = EnhancedConfig(groq_api_key=key, recall_mode=high_recall)

    return ProductionRuleExtractionSystem(
        groq_api_key=key,
        pipeline_settings=pipeline_settings,
        enable_enhanced=True,
        enhanced_config=enhanced_config,
    )

system = get_system()

st.subheader("Choose Document")
uploaded = st.file_uploader("Upload PDF", type=["pdf"])
doc_path = ""

if uploaded:
    tmp = Path(tempfile.gettempdir()) / uploaded.name
    tmp.write_bytes(uploaded.getbuffer())
    doc_path = str(tmp)

if st.button("Extract", disabled=not bool(doc_path)):
    with st.spinner("Compiling rules..."):
        result = asyncio.run(system.process_document_advanced(doc_path, enable_enhancement=True))
        st.session_state["result"] = result
        st.session_state["doc_name"] = Path(doc_path).name

result = st.session_state.get("result")
doc_name = st.session_state.get("doc_name", "")

if result:
    pdf_name = Path(doc_name).stem if doc_name else "untitled"
    output_file = f"output/{pdf_name}.json"
    
    # Show simple success message - NO TABLE DISPLAY
    st.success(f"✅ Saved to: `{output_file}`")
    st.info(f"📊 Total Rules: **{result.get('rule_count', 0)}**")
    
    # Show first few rules in JSON format
    if st.expander("Preview first 5 rules"):
        st.json(result.get("rules", [])[:5])

# Consolidation section
st.markdown("---")
st.subheader("📦 Consolidate All Rules")
st.write("Merge all individual PDF rule files into a single consolidated JSON file.")

col1, col2 = st.columns([3, 1])
with col1:
    output_dir = Path("output")
    if output_dir.exists():
        json_files = list(output_dir.glob("*.json"))
        # Exclude consolidated files from count
        individual_files = [f for f in json_files if "consolidated" not in f.stem.lower()]
        if individual_files:
            st.info(f"📁 Found **{len(individual_files)}** PDF rule file(s) in output directory")
        else:
            st.warning("No individual PDF rule files found in output directory")
    else:
        st.warning("Output directory does not exist yet. Extract some PDFs first.")

with col2:
    if st.button("🔗 Consolidate", disabled=not output_dir.exists() or not individual_files if output_dir.exists() else True):
        try:
            with st.spinner("Consolidating all rules..."):
                consolidated_path = system.consolidate_all_rules(output_dir="output")
            st.success(f"✅ Consolidated file created!")
            st.code(consolidated_path, language=None)
            
            # Show summary
            with open(consolidated_path, 'r', encoding='utf-8') as f:
                consolidated_data = json.load(f)
            st.metric("Total Documents", consolidated_data.get("total_documents", 0))
            st.metric("Total Rules", consolidated_data.get("total_rules", 0))
            
        except Exception as e:
            st.error(f"❌ Consolidation failed: {str(e)}")
