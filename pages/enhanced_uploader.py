"""
Enhanced uploader page with RAG integration
"""

import streamlit as st
import os
from rag_pipeline_integration import init_rag_pipeline, display_rag_stats

# Initialize RAG pipeline
rag_pipeline = init_rag_pipeline()

st.title("📤 PDF Uploader with Enhanced RAG Processing")

# Display knowledge base stats in sidebar
display_rag_stats(rag_pipeline)

# If no file currently uploaded, reset uploaded_file in session state
if "uploaded_file" in st.session_state and st.session_state["uploaded_file"] is None:
    del st.session_state["uploaded_file"]

# File uploader
uploaded_file = st.file_uploader(
    "Upload a PDF file",
    type=["pdf"],
    accept_multiple_files=False,
    help="Upload manufacturing guidelines, specifications, or design documents"
)

# Store in session state if a new file is uploaded
if uploaded_file is not None:
    if "uploaded_file" not in st.session_state or st.session_state["uploaded_file"] != uploaded_file:
        st.session_state["uploaded_file"] = uploaded_file
        st.session_state['file_name'] = uploaded_file.name[:-4]
        
        # Process with enhanced RAG pipeline
        with st.spinner("Processing document with enhanced RAG system..."):
            try:
                processing_results = rag_pipeline.process_uploaded_file(uploaded_file)
                
                # Display results
                st.success("File uploaded and processed successfully!")
                
                # Show processing summary
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Text Sentences", 
                        processing_results.get('text_count', 0)
                    )
                
                with col2:
                    rag_chunks = processing_results.get('rag_processing', {}).get('text_chunks', 0)
                    st.metric("RAG Chunks", rag_chunks)
                
                with col3:
                    images = processing_results.get('image_extraction', {}).get('images_found', 0)
                    st.metric("Images Extracted", images)
                
                # Show detailed results
                with st.expander("📊 Processing Details"):
                    st.json(processing_results)
                
                # RAG Enhancement Notice
                if rag_chunks > 0:
                    st.info(f"""
                    🧠 **Enhanced RAG Processing Complete!** 
                    
                    Your document has been processed with our advanced RAG system:
                    - **{rag_chunks}** intelligent chunks created with manufacturing context
                    - **Advanced embeddings** using BAAI/bge-large-en-v1.5 for better technical understanding  
                    - **Manufacturing-aware metadata** extraction for improved rule generation
                    - **Cross-document reasoning** capabilities enabled
                    
                    This will significantly improve rule generation and classification accuracy!
                    """)
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                # Fallback to basic processing
                st.warning("Falling back to basic processing...")
                st.session_state["uploaded_file"] = uploaded_file
                st.session_state['file_name'] = uploaded_file.name[:-4]
    else:
        st.warning("File already uploaded. Navigate to other pages to process, or upload a new file.")
        
        # Show current knowledge base state
        if st.button("🔄 Refresh Knowledge Base Stats"):
            st.rerun()

else:
    st.info("""
    Please upload a PDF file to continue.
    
    **Supported document types:**
    - Manufacturing design guidelines
    - Technical specifications
    - Quality standards
    - Process documentation
    - Assembly instructions
    
    **Enhanced features:**
    - Advanced text extraction with manufacturing context
    - Intelligent chunking that preserves document structure  
    - Cross-document knowledge building
    - Manufacturing-aware rule classification
    """)

# Knowledge base management
st.subheader("🗄️ Knowledge Base Management")

col1, col2 = st.columns(2)

with col1:
    if st.button("📈 View Knowledge Base Stats"):
        stats = rag_pipeline.get_knowledge_base_summary()
        st.json(stats)

with col2:
    if st.button("🗑️ Clear Knowledge Base", type="secondary"):
        if st.session_state.get('confirm_clear', False):
            rag_pipeline.rag_system.clear_database()
            st.success("Knowledge base cleared!")
            st.session_state['confirm_clear'] = False
            st.rerun()
        else:
            st.session_state['confirm_clear'] = True
            st.warning("Click again to confirm clearing the knowledge base")

# Add search interface in sidebar
from rag_pipeline_integration import add_rag_search_interface
add_rag_search_interface(rag_pipeline)

# Cleanup temp files on session end
import atexit
atexit.register(rag_pipeline.cleanup_temp_files)