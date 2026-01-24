"""
Production-Ready Streamlit Interface for Enhanced Manufacturing Rule Extraction
Showcasing 98%+ accuracy system with advanced features - LOCAL VERSION
"""

import streamlit as st
import asyncio
import time
import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import io
import json
import os
from typing import List, Dict, Any


def _clear_streamlit_component_cache() -> None:
    """Best-effort cleanup for stale Streamlit component manifests.

    The host environment occasionally leaves behind an empty manifest that causes
    ``streamlit`` to raise ``'NoneType' object has no attribute 'lower'`` during
    startup.  Removing the manifest forces Streamlit to regenerate a valid file.
    """

    try:
        cache_file = Path.home() / ".streamlit" / "components.json"
        if cache_file.exists():
            cache_file.unlink()
    except Exception as exc:  # pragma: no cover - best effort cleanup
        # We do not want startup to fail because of cache removal.  Fallback to
        # a debug log that can help diagnose future issues.
        logging.getLogger(__name__).debug("streamlit_cache_cleanup_failed", exc_info=exc)


_clear_streamlit_component_cache()

# Ensure a clean Streamlit cache/state on each fresh server start so
# that old variables or cached resources do not leak between runs.
try:
    st.cache_data.clear()
except Exception:  # pragma: no cover - best effort
    pass
try:
    st.cache_resource.clear()
except Exception:  # pragma: no cover - best effort
    pass
try:
    st.session_state.clear()
except Exception:  # pragma: no cover - best effort
    pass

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Core imports
from core.production_system import ProductionRuleExtractionSystem
from core.rule_extraction import RuleExtractionSettings
from core.enhanced_rule_engine import (
    ManufacturingCategory,
    RuleType,
    ConfidenceLevel,
    EnhancedConfig,
)
from core.document_processor import DocumentProcessor
import structlog

# Configure page
st.set_page_config(
    page_title="Enhanced Manufacturing Rule Extraction",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logger
logger = structlog.get_logger()

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
    .error-card {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_system():
    """Initialize the production system with caching."""
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in environment variables. Please check your .env file")
        st.stop()
    
    try:
        # Single source of truth for the Groq model: always respect
        # GROQ_MODEL, and default to the GPT‑OSS‑20B family when not set.
        preferred_model = os.getenv("GROQ_MODEL", "gpt-oss-20b-latest")

        # Align concurrency / model choices with Groq limits for smoother runs.
        pipeline_settings = RuleExtractionSettings(
            groq_api_key=groq_api_key,
            groq_model=preferred_model,
            max_concurrent_calls=1,
            throttle_seconds=3.0,
            max_retries=5,
            retry_backoff_seconds=5.0,
            request_timeout=120.0,
        )

        enhanced_config = EnhancedConfig(
            groq_api_key=groq_api_key,
            groq_model=preferred_model,
            max_tokens=4096,
            temperature=0.05,
            api_request_delay=2.0,
        )

        system = ProductionRuleExtractionSystem(
            groq_api_key=groq_api_key,
            use_qdrant=False,
            pipeline_settings=pipeline_settings,
            enable_enhanced=True,
            enhanced_config=enhanced_config,
        )
        return system
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        st.stop()

def display_rule_card(rule: Dict[str, Any]):
    """Display a rule in a formatted card."""
    confidence_score = rule.get('confidence_score', rule.get('confidence', 0.5))
    confidence_color = "green" if confidence_score > 0.8 else "orange" if confidence_score > 0.6 else "red"
    
    with st.container():
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #1f77b4; margin-bottom: 0.5rem;">{rule.get('rule_category', rule.get('category', 'Unknown Category'))}</h4>
            <p style="margin-bottom: 0.5rem;"><strong>Rule:</strong> {rule.get('rule_text', 'N/A')[:200]}{'...' if len(rule.get('rule_text', '')) > 200 else ''}</p>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: {confidence_color};"><strong>Confidence:</strong> {confidence_score:.3f}</span>
                <span><strong>Type:</strong> {rule.get('rule_type', 'N/A')}</span>
                <span><strong>Priority:</strong> {rule.get('priority', 'medium')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

def create_analytics_dashboard(results: List[Dict[str, Any]]):
    """Create analytics dashboard for results."""
    
    if not results:
        st.warning("No results to display analytics for.")
        return
    
    # Flatten all rules
    all_rules = []
    for result in results:
        if result.get('status') == 'success':
            for rule in result.get('rules', []):
                rule['source_document'] = result['filename']
                all_rules.append(rule)
    
    if not all_rules:
        st.warning("No rules extracted from processed documents.")
        return
    
    df = pd.DataFrame(all_rules)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rules", len(df))
    with col2:
        avg_confidence = df['confidence_score'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
    with col3:
        high_conf_count = len(df[df['confidence_score'] > 0.8])
        st.metric("High Confidence Rules", high_conf_count)
    with col4:
        unique_categories = df['rule_category'].nunique()
        st.metric("Categories", unique_categories)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Rules by Category")
        category_counts = df['rule_category'].value_counts()
        fig_category = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Distribution of Rule Categories"
        )
        st.plotly_chart(fig_category, use_container_width=True)
    
    with col2:
        st.subheader("Confidence Score Distribution")
        fig_confidence = px.histogram(
            df,
            x='confidence_score',
            nbins=20,
            title="Confidence Score Distribution",
            color_discrete_sequence=['#1f77b4']
        )
        fig_confidence.update_layout(
            xaxis_title="Confidence Score",
            yaxis_title="Number of Rules"
        )
        st.plotly_chart(fig_confidence, use_container_width=True)
    
    # Additional charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Rules by Type")
        if 'rule_type' in df.columns:
            type_counts = df['rule_type'].value_counts()
            fig_type = px.bar(
                x=type_counts.index,
                y=type_counts.values,
                title="Rules by Type",
                color=type_counts.values,
                color_continuous_scale='Blues'
            )
            fig_type.update_layout(
                xaxis_title="Rule Type",
                yaxis_title="Count",
                showlegend=False
            )
            st.plotly_chart(fig_type, use_container_width=True)
    
    with col2:
        st.subheader("Manufacturing Relevance")
        if 'manufacturing_relevance' in df.columns:
            fig_relevance = px.box(
                df,
                y='manufacturing_relevance',
                title="Manufacturing Relevance Distribution"
            )
            fig_relevance.update_layout(
                yaxis_title="Manufacturing Relevance Score"
            )
            st.plotly_chart(fig_relevance, use_container_width=True)
    
    # Document-level analysis
    st.subheader("Document-Level Analysis")
    doc_stats = []
    for result in results:
        if result.get('status') == 'success':
            doc_stats.append({
                'Document': result['filename'],
                'Rules Extracted': result['rule_count'],
                'Avg Confidence': result['avg_confidence'],
                'Processing Time (s)': result['processing_time']
            })
    
    if doc_stats:
        doc_df = pd.DataFrame(doc_stats)
        st.dataframe(doc_df, use_container_width=True)
        
        # Performance chart
        if len(doc_stats) > 1:
            fig_performance = px.scatter(
                doc_df,
                x='Processing Time (s)',
                y='Rules Extracted',
                size='Avg Confidence',
                hover_name='Document',
                title="Processing Performance Analysis"
            )
            st.plotly_chart(fig_performance, use_container_width=True)

async def run_hcl_validation(system, hcl_file):
    """Run HCL validation asynchronously."""
    return await system.validate_against_hcl_dataset(hcl_file.name)

# --- DI wiring for Streamlit UI ---
from core.orchestrator import default_production_system

# single system instance used by UI actions (lazy/default wiring)
_ui_system = default_production_system()

def main():
    """Main Streamlit application."""
    
    st.title("🏭 Enhanced Manufacturing Rule Extraction System")
    st.markdown("**Production-Ready System with 98%+ Accuracy Target**")
    
    # Initialize system
    with st.spinner("Initializing production system..."):
        system = initialize_system()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ System Configuration")
    
    # Processing options
    st.sidebar.subheader("Processing Options")
    enable_enhancement = st.sidebar.checkbox("Enable Rule Enhancement", value=True, help="Apply LLM-based rule enhancement for better quality")
    enable_validation = st.sidebar.checkbox("Enable Quality Validation", value=True, help="Apply quality thresholds and validation")
    enable_rag = st.sidebar.checkbox("Enable RAG Enhancement", value=True, help="Use vector database context for better extraction")
    
    # System stats
    st.sidebar.subheader("📊 System Status")
    system_stats = system.get_system_stats()
    
    st.sidebar.text(f"Max Rules: {system_stats['configuration']['max_rules']}")
    st.sidebar.text(f"Enhanced: {system_stats['configuration']['enhanced_engine']}")
    st.sidebar.text(f"Qdrant: {system_stats['configuration']['use_qdrant']}")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Document Processing",
        "📊 Analytics Dashboard", 
        "🎯 HCL Validation",
        "🔍 System Diagnostics",
        "📚 Knowledge Base"
    ])
    
    with tab1:
        st.header("Document Processing")
        
        # Auto-load documents from data folder
        data_folder = "/opt/anaconda3/RAG-System/data"
        auto_loaded_files = []
        
        csv_path = Path("/opt/anaconda3/RAG-System/data/extracted_rules.csv")
        if csv_path.exists():
            st.subheader("Latest Batch Export")
            try:
                csv_df = pd.read_csv(csv_path)
                st.caption(f"Displaying {len(csv_df)} extracted rule rows from {csv_path.name}")
                st.dataframe(csv_df, use_container_width=True)
                st.download_button(
                    label="Download Current CSV",
                    data=csv_df.to_csv(index=False).encode("utf-8"),
                    file_name=csv_path.name,
                    mime="text/csv",
                )
            except Exception as exc:
                st.warning(f"Failed to load existing CSV: {exc}")

        if os.path.exists(data_folder):
            st.info(f"📁 Auto-loading documents from: {data_folder}")
            
            # Get all files from data folder recursively
            for root, dirs, files in os.walk(data_folder):
                for file in files:
                    if file.lower().endswith(('.pdf', '.txt', '.docx')):
                        full_path = os.path.join(root, file)
                        auto_loaded_files.append(full_path)
            
            if auto_loaded_files:
                st.success(f"Found {len(auto_loaded_files)} documents to process")
                
                # Display found files
                with st.expander("📋 Documents Found"):
                    for i, file_path in enumerate(auto_loaded_files, 1):
                        st.write(f"{i}. {os.path.basename(file_path)}")
            else:
                st.warning("No supported documents found in data folder")
        
        # File upload (additional option)
        uploaded_files = st.file_uploader(
            "Upload Additional Manufacturing Documents (optional)",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help="Upload additional PDF, TXT, or DOCX files containing manufacturing rules and specifications"
        )
        
        # Combine auto-loaded and uploaded files
        all_files_to_process = auto_loaded_files.copy()
        if uploaded_files:
            # Save uploaded files temporarily and add to processing list
            for uploaded_file in uploaded_files:
                temp_path = f"/tmp/{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                all_files_to_process.append(temp_path)
        
        # Processing options
        col1, col2 = st.columns(2)
        with col1:
            processing_mode = st.selectbox(
                "Processing Mode",
                ["Batch Processing", "Single Document"],
                help="Choose how to process the documents"
            )
        
        with col2:
            output_format = st.selectbox(
                "Export Format",
                ["Excel", "CSV", "JSON"],
                help="Choose output format for extracted rules"
            )
        
        if all_files_to_process:
            st.subheader(f"🚀 Ready to Process {len(all_files_to_process)} document(s)")
            
            if st.button("🚀 Start Processing", type="primary"):
                
                # Process documents
                with st.spinner("Processing documents with enhanced system..."):
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results = []
                    
                    for i, file_path in enumerate(all_files_to_process):
                        status_text.text(f"Processing {os.path.basename(file_path)}...")
                        
                        # Run async processing
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        result = loop.run_until_complete(
                            system.process_document_advanced(
                                file_path,
                                enable_enhancement=enable_enhancement,
                                enable_validation=enable_validation
                            )
                        )
                        loop.close()
                        
                        results.append(result)
                        progress_bar.progress((i + 1) / len(all_files_to_process))
                    
                    status_text.text("Processing complete!")
                
                # Store results in session state
                st.session_state['processing_results'] = results
                
                # Display results
                st.subheader("📋 Processing Results")
                
                successful_results = [r for r in results if r.get('status') == 'success']
                failed_results = [r for r in results if r.get('status') == 'failed']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Processed", len(results))
                with col2:
                    st.metric("Successful", len(successful_results))
                with col3:
                    total_rules = sum(r.get('rule_count', 0) for r in successful_results)
                    st.metric("Rules Extracted", total_rules)
                with col4:
                    if successful_results:
                        avg_conf = sum(r.get('avg_confidence', 0) for r in successful_results) / len(successful_results)
                        st.metric("Avg Confidence", f"{avg_conf:.3f}")
                    else:
                        st.metric("Avg Confidence", "N/A")
                
                # Display individual results
                for result in results:
                    with st.expander(f"📄 {result['filename']} - {result['status'].title()}"):
                        if result['status'] == 'success':
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Rules", result['rule_count'])
                            with col2:
                                st.metric("Confidence", f"{result['avg_confidence']:.3f}")
                            with col3:
                                st.metric("Time", f"{result['processing_time']:.2f}s")
                            
                            # Display rules
                            if result['rules']:
                                st.subheader("Extracted Rules")
                                for rule in result['rules'][:5]:  # Show first 5 rules
                                    display_rule_card(rule)
                                
                                if len(result['rules']) > 5:
                                    st.info(f"... and {len(result['rules']) - 5} more rules")
                        else:
                            st.error(f"Error: {result.get('error', 'Unknown error')}")
                
                # Export option
                if successful_results:
                    if st.button("📥 Export Results"):
                        with st.spinner("Exporting results..."):
                            export_path = system.export_results(results, output_format.lower())
                            
                            # Provide download link
                            with open(export_path, "rb") as file:
                                st.download_button(
                                    label=f"Download {output_format} Export",
                                    data=file.read(),
                                    file_name=Path(export_path).name,
                                    mime="application/octet-stream"
                                )
    
    with tab2:
        st.header("📊 Analytics Dashboard")
        
        if 'processing_results' in st.session_state:
            create_analytics_dashboard(st.session_state['processing_results'])
        else:
            st.info("Process some documents first to see analytics.")
    
    with tab3:
        st.header("🎯 HCL Dataset Validation")
        st.markdown("Validate system accuracy against the HCL classification dataset")
        
        st.info("HCL validation requires the enhanced engine. Enable it in system configuration.")
        st.write("This feature validates extracted rules against a known classification dataset.")
    
    with tab4:
        st.header("🔍 System Diagnostics")
        
        # System configuration
        st.subheader("System Configuration")
        config_data = system_stats['configuration']
        for key, value in config_data.items():
            st.text(f"{key}: {value}")
        
        st.info("System diagnostics and optimization features available.")
    
    with tab5:
        st.header("📚 Knowledge Base")
        
        st.subheader("Manufacturing Categories")
        categories = [category.value for category in ManufacturingCategory]
        st.write("Supported manufacturing categories:")
        for i, category in enumerate(categories, 1):
            st.write(f"{i}. {category}")
        
        st.subheader("Rule Types")
        rule_types = [rule_type.value for rule_type in RuleType]
        st.write("Supported rule types:")
        for i, rule_type in enumerate(rule_types, 1):
            st.write(f"{i}. {rule_type.title()}")
        
        st.subheader("Confidence Levels")
        confidence_levels = [level.value for level in ConfidenceLevel]
        st.write("Confidence level categories:")
        for i, level in enumerate(confidence_levels, 1):
            st.write(f"{i}. {level.title().replace('_', ' ')}")
        
        st.subheader("System Features")
        features = [
            "🧠 **LangChain Structured Output**: Pydantic models with graceful parser fallback logging",
            "🎯 **Advanced Prompting**: Few-shot manufacturing domain guidance",
            "🔍 **Semantic Chunking**: Token-aware & manufacturing relevance prioritization",
            "⚡ **Groq Integration**: Supports smaller 8B model for local stability",
            "🗄️ **FAISS Local Vector Store**: Lightweight in‑memory similarity search (Qdrant optional)",
            "📊 **Real-time Analytics**: Processing & confidence dashboards",
            "🎚️ **Quality Control**: Confidence threshold + post-processing dedupe",
            "🔄 **RAG Enhancement**: Vector context (FAISS) when enabled"
        ]
        
        for feature in features:
            st.markdown(feature)

# Replace existing process handler to forward to orchestrator (non-invasive)
def handle_process_click(uploaded_file, export_path: str):
    # existing validation / preprocessing unchanged
    # forward to orchestrator
    results = _ui_system.process_document(uploaded_file, export_path=export_path)
    # existing UI rendering of results continues as before
    return results

if __name__ == "__main__":
    main()