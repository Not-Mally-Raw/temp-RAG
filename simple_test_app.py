"""
Simple test app to validate the enhanced PDF processing system
"""

import streamlit as st
import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

st.set_page_config(
    page_title="Enhanced PDF Test", 
    page_icon="📄",
    layout="wide"
)

def main():
    st.title("📄 Enhanced PDF Processing Test")
    st.write("Test the robust PDF extraction system")
    
    # Test PDF processing
    st.header("🧪 Test PDF Processing")
    
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file:
        st.write(f"File: {uploaded_file.name} ({uploaded_file.size:,} bytes)")
        
        if st.button("🔍 Analyze PDF"):
            with st.spinner("Analyzing PDF..."):
                try:
                    from extractors.robust_pdf_processor import RobustPDFProcessor
                    
                    processor = RobustPDFProcessor()
                    pdf_bytes = uploaded_file.getvalue()
                    
                    # Run analysis
                    analysis = processor.analyze_pdf(pdf_bytes)
                    
                    # Display results
                    st.subheader("📊 Analysis Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("File Size", f"{analysis['size_bytes']:,} bytes")
                    with col2:
                        st.metric("Text Length", f"{analysis['total_text_length']:,} chars")
                    with col3:
                        st.metric("Sentences", analysis['sentence_count'])
                    
                    # Successful method
                    if analysis['successful_method']:
                        st.success(f"✅ Best extraction method: {analysis['successful_method']}")
                    else:
                        st.error("❌ No extraction method succeeded")
                    
                    # Method results
                    st.subheader("🔧 Extraction Methods")
                    for method in analysis['methods_tested']:
                        status = "✅" if method['success'] else "❌"
                        st.write(f"{status} **{method['name']}**: {method['text_length']:,} chars")
                        if method['error']:
                            st.caption(f"Error: {method['error']}")
                    
                    # Extract sentences
                    if st.button("📝 Extract Text"):
                        with st.spinner("Extracting text..."):
                            sentences = processor.extract_sentences(pdf_bytes)
                            
                            if sentences:
                                st.success(f"✅ Extracted {len(sentences)} sentences")
                                
                                st.subheader("📝 Sample Text")
                                for i, sentence in enumerate(sentences[:5], 1):
                                    st.write(f"**{i}.** {sentence}")
                                
                                if len(sentences) > 5:
                                    st.write(f"... and {len(sentences) - 5} more sentences")
                            else:
                                st.error("❌ No text could be extracted")
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.write("Full error details:")
                    st.exception(e)
    
    # Test dependencies
    st.header("🔧 System Status")
    
    dependencies = [
        "pdfminer",
        "pdfplumber", 
        "PyPDF2",
        "fitz",  # PyMuPDF
        "pytesseract",
        "pdf2image",
        "PIL"
    ]
    
    for dep in dependencies:
        try:
            if dep == "fitz":
                import fitz
            elif dep == "PIL":
                from PIL import Image
            else:
                __import__(dep)
            st.success(f"✅ {dep}")
        except ImportError:
            st.error(f"❌ {dep}")

if __name__ == "__main__":
    main()