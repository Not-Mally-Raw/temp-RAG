"""
Comprehensive RAG System with Enhanced PDF Processing
Main application with all features integrated
"""

import streamlit as st
from pathlib import Path
import sys
import os

# Configure Streamlit
st.set_page_config(
    page_title="Enhanced RAG System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def main():
    """Main application with enhanced features."""
    
    st.sidebar.title("📄 Enhanced RAG System")
    st.sidebar.write("Comprehensive document processing with OCR support")
    
    # Import pages
    try:
        from pages import (
            enhanced_uploader,
            automated_testing, 
            enhanced_qa,
            analytics,
            rule_generation,
            results,
            classification,
            industry_testing_simulator,
            smart_uploader,
            uploader
        )
        
        pages = {
            "🏠 Home": show_home,
            "📁 Enhanced Uploader": enhanced_uploader,
            "🧪 Comprehensive Testing": automated_testing,
            "📈 Enhanced QA": enhanced_qa,
            "📊 Analytics": analytics,
            "⚙️ Rule Generation": rule_generation,
            "📄 Results": results,
            "🔍 Classification": classification,
            "🏭 Industry Testing": industry_testing_simulator,
            "📤 Smart Uploader": smart_uploader,
            "📁 Basic Uploader": uploader
        }
        
    except ImportError as e:
        st.error(f"Could not import pages: {e}")
        st.write("Available modules:")
        try:
            import pages
            st.write(dir(pages))
        except:
            st.write("Pages module not found")
        return
    
    # Navigation
    selected_page = st.sidebar.selectbox(
        "Choose a page:",
        list(pages.keys()),
        index=0
    )
    
    # Show the selected page
    if selected_page == "🏠 Home":
        show_home()
    else:
        try:
            pages[selected_page].show()
        except AttributeError:
            try:
                pages[selected_page].main()
            except AttributeError:
                st.error(f"Page {selected_page} does not have show() or main() function")
        except Exception as e:
            st.error(f"Error loading page {selected_page}: {e}")


def show_home():
    """Enhanced home page with comprehensive system overview."""
    
    st.title("📄 Enhanced RAG System")
    st.write("Comprehensive document processing with robust PDF extraction and OCR support")
    
    # System overview
    st.header("🔧 System Capabilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Document Processing")
        st.markdown("""
        - **Multi-format PDF extraction** (pdfminer, pdfplumber, PyPDF2, PyMuPDF)
        - **OCR support** for scanned documents (Tesseract)
        - **Fallback mechanisms** for maximum compatibility
        - **Detailed diagnostics** and error reporting
        - **Progress tracking** for all operations
        """)
        
        st.subheader("🧠 AI & RAG Features")
        st.markdown("""
        - **Advanced RAG chunking** with intelligent splitting
        - **Vector embeddings** using BAAI/bge-large-en-v1.5
        - **LLM integration** with Groq API
        - **Academic citations** with source attribution
        - **Rule extraction** and CSV export
        """)
    
    with col2:
        st.subheader("🧪 Testing & Validation")
        st.markdown("""
        - **Comprehensive testing** on multiple documents
        - **Real-time progress tracking** with detailed metrics
        - **Automated rule generation** for each document
        - **Performance analytics** and success rates
        - **Detailed error analysis** with suggestions
        """)
        
        st.subheader("💾 Data Management")
        st.markdown("""
        - **Persistent vector storage** with ChromaDB
        - **Metadata enrichment** and document tracking
        - **CSV export** for all extracted rules
        - **Test results archiving** with JSON export
        - **Database analytics** and search capabilities
        """)
    
    # Quick start section
    st.header("🚀 Quick Start Guide")
    
    tab1, tab2, tab3 = st.tabs(["📁 Upload Documents", "🧪 Run Tests", "📈 Ask Questions"])
    
    with tab1:
        st.markdown("""
        ### Upload and Process Documents
        
        1. **Go to Enhanced Uploader** - Handles any PDF type including scanned documents
        2. **Upload multiple files** - Process several documents at once
        3. **Monitor progress** - Real-time tracking with detailed status
        4. **Review results** - Comprehensive analysis and suggestions
        
        **Features:**
        - Automatic OCR for image-based PDFs
        - Detailed extraction diagnostics
        - Fallback methods for maximum compatibility
        - Progress tracking with error handling
        """)
        
        if st.button("📁 Go to Enhanced Uploader", type="primary"):
            st.query_params["page"] = "Enhanced Uploader"
            st.rerun()
    
    with tab2:
        st.markdown("""
        ### Comprehensive System Testing
        
        1. **Go to Comprehensive Testing** - Test the entire pipeline
        2. **Configure test settings** - Choose test depth and options
        3. **Upload test documents** - Or use sample documents
        4. **Monitor execution** - Real-time progress and results
        5. **Review analytics** - Detailed performance metrics
        
        **Test Coverage:**
        - PDF extraction methods
        - RAG chunking and vectorization
        - Rule generation and QA
        - CSV export functionality
        - Performance benchmarking
        """)
        
        if st.button("🧪 Go to Comprehensive Testing", type="primary"):
            st.query_params["page"] = "Comprehensive Testing"
            st.rerun()
    
    with tab3:
        st.markdown("""
        ### Enhanced Question Answering
        
        1. **Go to Enhanced QA** - Ask questions about your documents
        2. **Enter your question** - Natural language queries
        3. **Get detailed answers** - LLM-generated responses
        4. **View citations** - Academic-style source references
        5. **Export rules** - Download complete rule database
        
        **Features:**
        - LLM-powered detailed answers
        - Academic citation format
        - Source attribution and relevance scores
        - Automatic rule extraction
        - CSV export with metadata
        """)
        
        if st.button("📈 Go to Enhanced QA", type="primary"):
            st.query_params["page"] = "Enhanced QA"
            st.rerun()
    
    # System status
    st.header("🔧 System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📚 Dependencies")
        
        # Check key dependencies
        dependencies = {
            "pdfminer": "Basic PDF extraction",
            "pdfplumber": "Advanced PDF parsing", 
            "PyPDF2": "PDF reading fallback",
            "PyMuPDF": "Complex PDF handling",
            "pytesseract": "OCR text extraction",
            "pdf2image": "PDF to image conversion",
            "langchain_chroma": "Vector storage",
            "sentence_transformers": "Text embeddings"
        }
        
        for dep, desc in dependencies.items():
            try:
                __import__(dep)
                st.success(f"✅ {dep}")
            except ImportError:
                st.error(f"❌ {dep}")
    
    with col2:
        st.subheader("🗄️ Database Status")
        
        # Check database directories
        db_dirs = [
            "./rag_db",
            "./automated_test_db", 
            "./test_comprehensive_rag_db",
            "./debug_test"
        ]
        
        for db_dir in db_dirs:
            if os.path.exists(db_dir):
                files = len([f for f in os.listdir(db_dir) if f.endswith('.sqlite3')])
                st.info(f"📁 {db_dir}: {files} DBs")
            else:
                st.warning(f"📁 {db_dir}: Not found")
    
    with col3:
        st.subheader("🧪 Test Results")
        
        # Check for test results
        test_dirs = [
            "./test_results",
            "./processing_temp"
        ]
        
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                files = len(os.listdir(test_dir))
                st.info(f"📊 {test_dir}: {files} files")
            else:
                st.warning(f"📊 {test_dir}: Empty")
    
    # Recent improvements
    st.header("✨ Recent Enhancements")
    
    improvements = [
        "🔧 **Robust PDF Processing** - 5 extraction methods with OCR fallback",
        "📊 **Comprehensive Testing** - Automated testing with progress tracking", 
        "📈 **Enhanced QA System** - LLM-powered answers with academic citations",
        "💾 **CSV Export** - Complete rule database with metadata",
        "🧪 **Real-time Analytics** - Processing metrics and success rates",
        "⚡ **Error Handling** - Detailed diagnostics and suggestions",
        "🎯 **Progress Tracking** - Real-time status for all operations"
    ]
    
    for improvement in improvements:
        st.markdown(f"• {improvement}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p><strong>Enhanced RAG System v2.0</strong></p>
    <p>Comprehensive document processing with robust PDF extraction and OCR support</p>
    <p>Built for maximum compatibility and detailed analytics</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()