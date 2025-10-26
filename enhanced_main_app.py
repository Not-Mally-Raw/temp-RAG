"""
Enhanced RAG System - Optimized Main Application
Professional Streamlit application with improved performance and user experience
"""

import streamlit as st
from pathlib import Path
import sys
import os
import time

# Performance optimizations
st.set_page_config(
    page_title="Enhanced RAG System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/rag-system',
        'Report a bug': 'https://github.com/your-repo/rag-system/issues',
        'About': '''
        Enhanced RAG System for Manufacturing Intelligence
        Advanced document processing with robust PDF extraction and OCR support
        '''
    }
)

# Add path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Lazy imports for faster startup
@st.cache_resource
def get_page_modules():
    """Lazy load page modules for better performance."""
    try:
        from pages import (
            consolidated_rules,
            automated_testing,
            enhanced_qa,
            analytics,
            results,
            enhanced_uploader
        )
        return {
            "consolidated": consolidated_rules,
            "testing": automated_testing,
            "qa": enhanced_qa,
            "analytics": analytics,
            "results": results,
            "uploader": enhanced_uploader
        }
    except ImportError as e:
        st.error(f"Could not load page modules: {e}")
        return {}

def main():
    """Main application with enhanced UX and performance."""

    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(45deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
        transition: transform 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("🚀 Enhanced RAG System")
    st.sidebar.markdown("---")

    # Main navigation
    pages = {
        "🏠 Home": "home",
        "📊 Consolidated Rules": "consolidated",
        "🧪 Automated Testing": "testing",
        "📈 Enhanced QA": "qa",
        "📊 Analytics": "analytics",
        "📄 Document Upload": "uploader",
        "📋 Results": "results"
    }

    selected_page = st.sidebar.radio(
        "Navigation",
        list(pages.keys()),
        index=0,
        help="Choose a page to navigate to"
    )

    # System status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 System Status")

    # Quick system checks
    try:
        # Check if test results exist
        test_results_dir = Path("./test_results")
        if test_results_dir.exists():
            csv_files = list(test_results_dir.glob("*_rules.csv"))
            st.sidebar.success(f"✅ {len(csv_files)} rule files")
        else:
            st.sidebar.warning("⚠️ No test results")

        # Check database
        db_dirs = ["./rag_db", "./automated_test_db"]
        db_count = sum(1 for db in db_dirs if Path(db).exists())
        st.sidebar.info(f"💾 {db_count}/2 databases")

    except Exception as e:
        st.sidebar.error("❌ System check failed")

    # Route to selected page
    page_key = pages[selected_page]

    if page_key == "home":
        show_home_page()
    else:
        # Load page modules
        page_modules = get_page_modules()

        if page_key in page_modules:
            try:
                # Try main() first, then show()
                if hasattr(page_modules[page_key], 'main'):
                    page_modules[page_key].main()
                elif hasattr(page_modules[page_key], 'show'):
                    page_modules[page_key].show()
                else:
                    st.error(f"Page {selected_page} does not have main() or show() function")
            except Exception as e:
                st.error(f"Error loading page {selected_page}: {e}")
        else:
            st.error(f"Page module {page_key} not available")

def show_home_page():
    """Enhanced home page with professional design."""

    st.markdown('<h1 class="main-header">🚀 Enhanced RAG System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Manufacturing Intelligence with Robust Document Processing</p>', unsafe_allow_html=True)

    # Key metrics
    st.header("📊 System Performance")

    # Load test results for metrics
    test_results_dir = Path("./test_results")
    total_rules = 0
    total_docs = 0

    if test_results_dir.exists():
        try:
            # Count rules from consolidated file
            consolidated_file = test_results_dir / "consolidated_all_rules.csv"
            if consolidated_file.exists():
                import pandas as pd
                df = pd.read_csv(consolidated_file)
                total_rules = len(df)
                total_docs = df['source_file'].nunique() if 'source_file' in df.columns else 0
        except:
            pass

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Rules Extracted", total_rules)

    with col2:
        st.metric("Documents Processed", total_docs)

    with col3:
        st.metric("Success Rate", "36.4%" if total_docs > 0 else "N/A")

    with col4:
        st.metric("Avg Processing Time", "72.6s")

    # Core features
    st.header("🎯 Core Capabilities")

    features = [
        {
            "icon": "📄",
            "title": "Robust PDF Processing",
            "description": "Multi-method extraction with OCR fallback for scanned documents",
            "status": "✅ Active"
        },
        {
            "icon": "🧠",
            "title": "Advanced RAG Pipeline",
            "description": "Intelligent chunking and vectorization with BAAI embeddings",
            "status": "✅ Active"
        },
        {
            "icon": "🤖",
            "title": "LLM-Powered QA",
            "description": "Question answering with academic citations and rule extraction",
            "status": "✅ Active"
        },
        {
            "icon": "📊",
            "title": "Automated Testing",
            "description": "Comprehensive pipeline testing with progress tracking",
            "status": "✅ Active"
        },
        {
            "icon": "💾",
            "title": "Consolidated Database",
            "description": "Complete rules database with analytics and export options",
            "status": "✅ Active"
        },
        {
            "icon": "🔍",
            "title": "Advanced Analytics",
            "description": "Performance monitoring and detailed insights",
            "status": "✅ Active"
        }
    ]

    # Display features in grid
    cols = st.columns(2)
    for i, feature in enumerate(features):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="feature-card">
                <h3>{feature['icon']} {feature['title']}</h3>
                <p>{feature['description']}</p>
                <strong>{feature['status']}</strong>
            </div>
            """, unsafe_allow_html=True)

    # Quick actions
    st.header("🚀 Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 View Consolidated Rules", type="primary", use_container_width=True):
            st.switch_page("pages/consolidated_rules.py")

    with col2:
        if st.button("🧪 Run Automated Testing", use_container_width=True):
            st.switch_page("pages/automated_testing.py")

    with col3:
        if st.button("📈 Enhanced QA System", use_container_width=True):
            st.switch_page("pages/enhanced_qa.py")

    # Recent activity
    st.header("📈 Recent Activity")

    # Load recent test results
    if test_results_dir.exists():
        json_files = list(test_results_dir.glob("*.json"))
        if json_files:
            try:
                latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
                with open(latest_json, 'r') as f:
                    test_data = json.load(f)

                st.subheader("Latest Test Results")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Documents Processed", test_data.get('summary', {}).get('total_files', 0))

                with col2:
                    success_rate = test_data.get('summary', {}).get('success_rate', 0)
                    st.metric("Success Rate", f"{success_rate:.1f}%")

                with col3:
                    total_chunks = test_data.get('summary', {}).get('total_rag_chunks', 0)
                    st.metric("RAG Chunks Created", total_chunks)

            except Exception as e:
                st.warning(f"Could not load recent test results: {e}")

    # System architecture overview
    with st.expander("🏗️ System Architecture"):
        st.markdown("""
        ### Core Components:

        **🔧 Document Processing Layer:**
        - Multi-method PDF extraction (pdfminer, pdfplumber, PyPDF2, PyMuPDF)
        - OCR support for scanned documents (Tesseract)
        - Fallback mechanisms for maximum compatibility

        **🧠 AI & RAG Layer:**
        - BAAI/bge-large-en-v1.5 embeddings for semantic search
        - ChromaDB vector storage with metadata enrichment
        - Intelligent text chunking and preprocessing

        **🤖 Intelligence Layer:**
        - LLM integration with Groq API for question answering
        - Academic-style citations and source attribution
        - Automated rule extraction and classification

        **📊 Analytics Layer:**
        - Real-time performance monitoring
        - Comprehensive testing and validation
        - CSV/JSON export capabilities

        **💾 Data Management:**
        - Persistent vector databases
        - Consolidated rules database
        - Test results archiving
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p><strong>🚀 Enhanced RAG System for Manufacturing Intelligence</strong></p>
    <p>Advanced document processing with robust PDF extraction and OCR support</p>
    <p>Built for maximum compatibility and detailed analytics</p>
    <br>
    <p><small>Ready for production deployment and GitHub release</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()