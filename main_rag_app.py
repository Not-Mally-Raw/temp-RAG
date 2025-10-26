"""
Manufacturing Rules RAG System - Main Application
Multi-page Streamlit app with automated testing and comprehensive rule generation
"""

import streamlit as st
from pathlib import Path

# Configure Streamlit
st.set_page_config(
    page_title="Manufacturing Rules RAG System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Navigation
def main():
    """Main application with navigation"""
    
    # Sidebar navigation
    st.sidebar.title("⚙️ Manufacturing Rules RAG System")
    st.sidebar.markdown("---")
    
    # Page selection
    pages = {
        "🏠 Home": "home",
        "📤 Upload Documents": "uploader", 
        "🔍 Enhanced QA": "enhanced_qa",
        "⚡ Rule Generation": "rule_generation",
        "🔄 Automated Testing": "automated_testing",
        "📊 Analytics": "analytics",
        "🏭 Industry Testing": "industry_testing_simulator",
        "📈 Results": "results"
    }
    
    selected_page = st.sidebar.selectbox(
        "Navigate to:",
        options=list(pages.keys()),
        index=0
    )
    
    page_name = pages[selected_page]
    
    # Load the selected page
    if page_name == "home":
        show_home_page()
    elif page_name == "automated_testing":
        # Import and run automated testing page
        import pages.automated_testing
    else:
        # Load other pages dynamically
        try:
            page_module = __import__(f"pages.{page_name}", fromlist=[page_name])
            # Most pages are just imported and run automatically
        except ImportError:
            st.error(f"Page {page_name} not found")
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 Quick Guide")
    st.sidebar.markdown("""
    1. **Upload** PDF documents
    2. **Test** with automated system
    3. **Generate** rules automatically  
    4. **Analyze** with Enhanced QA
    5. **Export** results to CSV
    """)
    
    # System status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 System Status")
    
    try:
        from core.rag_pipeline_integration import init_rag_pipeline
        pipeline = init_rag_pipeline()
        stats = pipeline.get_knowledge_base_summary()
        
        st.sidebar.metric("Documents", stats.get('total_documents', 0))
        st.sidebar.metric("RAG Chunks", stats.get('total_chunks', 0))
        st.sidebar.metric("Vector Embeddings", stats.get('vector_count', 0))
        
    except Exception as e:
        st.sidebar.warning("Database not initialized")

def show_home_page():
    """Home page with system overview"""
    
    st.title("⚙️ Manufacturing Rules RAG System")
    st.markdown("### Advanced Document Processing with Automated Rule Generation")
    
    # Feature overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 📤 Document Processing
        - **Advanced PDF extraction** with manufacturing context
        - **Intelligent RAG chunking** preserving document structure
        - **Cross-document knowledge** building
        - **Manufacturing-aware metadata** extraction
        """)
    
    with col2:
        st.markdown("""
        #### 🤖 AI-Powered Analysis  
        - **LLM integration** with Groq API
        - **Academic citation** format with source attribution
        - **Automated rule extraction** using pattern matching + LLM
        - **Question answering** with detailed responses
        """)
    
    with col3:
        st.markdown("""
        #### 📊 Export & Analysis
        - **CSV export** with complete metadata
        - **Real-time progress** tracking
        - **Comprehensive testing** on document batches
        - **Performance analytics** and success rates
        """)
    
    # Quick start guide
    st.markdown("---")
    st.subheader("🚀 Quick Start Guide")
    
    st.markdown("""
    ### For Testing DFM Handbook Documents:
    
    1. **Go to Automated Testing** 📊
       - Enter your DFM Handbook folder path
       - Click "Start Automated Testing"
       - Watch real-time progress for each document
    
    2. **Monitor Progress** 📈
       - View RAG chunking success for each document
       - See rule generation progress
       - Check success rates and processing times
    
    3. **Review Results** 📋
       - Detailed results for each document
       - RAG chunks created per document
       - Rules extracted and citation counts
       - Export comprehensive test results
    
    ### For Individual Document Processing:
    
    1. **Upload Documents** 📤
       - Use the uploader page for single documents
       - Automatic RAG processing with progress feedback
       - View processing details and warnings
    
    2. **Enhanced QA** 🔍
       - Ask questions about uploaded documents
       - Get detailed LLM-generated answers
       - Academic citations with source attribution
       - Export rules database to CSV
    
    3. **Rule Generation** ⚡
       - Direct rule extraction from documents
       - Manufacturing process filtering
       - Confidence scoring and classification
    """)
    
    # System requirements
    with st.expander("📋 System Requirements & Features"):
        st.markdown("""
        ### Core Features:
        - **RAG System**: ChromaDB with BAAI/bge-large-en-v1.5 embeddings (1024 dimensions)
        - **LLM Integration**: Groq API with multiple model options
        - **Document Support**: PDF with advanced text extraction
        - **Export Formats**: CSV with complete metadata, JSON for detailed results
        
        ### Automated Testing Capabilities:
        - **Batch Processing**: Process entire folders of documents
        - **Progress Tracking**: Real-time status updates and progress bars
        - **Comprehensive Testing**: RAG chunking, rule generation, QA system, CSV export
        - **Performance Metrics**: Processing times, success rates, chunk counts
        - **Error Handling**: Detailed error reporting and fallback mechanisms
        
        ### Quality Assurance:
        - **Citation System**: Academic-style references with source attribution
        - **Rule Validation**: Pattern matching + LLM verification
        - **Metadata Preservation**: Complete document context and source tracking
        - **Cross-Document Analysis**: Knowledge building across multiple documents
        """)
    
    # Current status
    st.markdown("---")
    st.subheader("📊 Current System Status")
    
    try:
        from core.rag_pipeline_integration import init_rag_pipeline
        pipeline = init_rag_pipeline()
        stats = pipeline.get_knowledge_base_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Documents", stats.get('total_documents', 0))
        
        with col2:
            st.metric("RAG Chunks", stats.get('total_chunks', 0))
        
        with col3:
            st.metric("Vector Embeddings", stats.get('vector_count', 0))
        
        with col4:
            if stats.get('total_chunks', 0) > 0:
                st.metric("Status", "✅ Ready", delta="Operational")
            else:
                st.metric("Status", "⚪ Empty", delta="No documents")
    
    except Exception as e:
        st.warning("⚠️ RAG system not initialized. Upload documents first.")
    
    # Next steps
    st.markdown("---")
    st.info("""
    **Ready to test your DFM Handbook documents?** 
    
    Navigate to **🔄 Automated Testing** to process your entire document collection with real-time progress tracking and comprehensive results analysis.
    """)

if __name__ == "__main__":
    main()