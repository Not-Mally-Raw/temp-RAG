"""
Enhanced uploader with automatic rule generation and progress tracking
"""

import streamlit as st
import os
import time
import threading
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from core.rag_pipeline_integration import init_rag_pipeline, display_rag_stats
from enhanced_qa_system import RAGQuestionAnswerer

class ProgressTracker:
    def __init__(self):
        self.stages = [
            "📄 Extracting text from PDF",
            "🧩 Creating RAG chunks", 
            "🔗 Generating embeddings",
            "⚡ Extracting rules automatically",
            "💾 Saving to database"
        ]
        self.current_stage = 0
        self.progress = 0.0
        self.status_message = "Ready"
        self.is_processing = False
        self.results = {}

# Initialize components
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = init_rag_pipeline()

if 'qa_system' not in st.session_state:
    st.session_state.qa_system = RAGQuestionAnswerer(st.session_state.rag_pipeline)

if 'progress_tracker' not in st.session_state:
    st.session_state.progress_tracker = ProgressTracker()

rag_pipeline = st.session_state.rag_pipeline
qa_system = st.session_state.qa_system
tracker = st.session_state.progress_tracker

st.title("📤 Smart Document Uploader")
st.write("Upload documents with automatic RAG processing and rule generation")

# Display knowledge base stats in sidebar
display_rag_stats(rag_pipeline)

# Main upload interface
uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    accept_multiple_files=False,
    help="Upload manufacturing guidelines, specifications, or design documents"
)

if uploaded_file is not None:
    
    # Display file info
    st.success(f"📁 File selected: {uploaded_file.name}")
    file_size = len(uploaded_file.getvalue()) / 1024 / 1024  # MB
    st.write(f"File size: {file_size:.2f} MB")
    
    # Processing options
    col1, col2 = st.columns(2)
    
    with col1:
        auto_generate_rules = st.checkbox("🤖 Auto-generate rules", value=True)
        extract_qa_samples = st.checkbox("❓ Extract QA samples", value=True)
    
    with col2:
        chunk_size = st.selectbox("Chunk size", [500, 1000, 1500], index=1)
        overlap = st.selectbox("Chunk overlap", [50, 100, 200], index=1)
    
    # Start processing button
    if st.button("🚀 Process Document", type="primary"):
        
        # Initialize progress tracking
        tracker.is_processing = True
        tracker.current_stage = 0
        tracker.progress = 0.0
        tracker.status_message = "Starting processing..."
        
        # Create progress display
        progress_bar = st.progress(0)
        status_text = st.empty()
        stage_text = st.empty()
        
        # Metrics display
        metrics_container = st.container()
        
        # Processing stages
        stages = tracker.stages
        total_stages = len(stages)
        
        try:
            # Stage 1: Extract text
            tracker.current_stage = 0
            tracker.progress = 0.2
            progress_bar.progress(tracker.progress)
            status_text.text(f"Stage {tracker.current_stage + 1}/{total_stages}")
            stage_text.text(f"🔄 {stages[tracker.current_stage]}")
            
            # Process with RAG pipeline
            processing_results = rag_pipeline.process_uploaded_file(uploaded_file)
            
            # Stage 2: RAG chunks
            tracker.current_stage = 1
            tracker.progress = 0.4
            progress_bar.progress(tracker.progress)
            status_text.text(f"Stage {tracker.current_stage + 1}/{total_stages}")
            stage_text.text(f"🔄 {stages[tracker.current_stage]}")
            
            rag_chunks = processing_results.get('rag_processing', {}).get('text_chunks', 0)
            
            # Stage 3: Embeddings
            tracker.current_stage = 2
            tracker.progress = 0.6
            progress_bar.progress(tracker.progress)
            status_text.text(f"Stage {tracker.current_stage + 1}/{total_stages}")
            stage_text.text(f"🔄 {stages[tracker.current_stage]}")
            
            time.sleep(0.5)  # Simulate embedding time
            
            # Stage 4: Auto rule generation
            tracker.current_stage = 3
            tracker.progress = 0.8
            progress_bar.progress(tracker.progress)
            status_text.text(f"Stage {tracker.current_stage + 1}/{total_stages}")
            stage_text.text(f"🔄 {stages[tracker.current_stage]}")
            
            rules_data = {}
            qa_samples = []
            
            if auto_generate_rules and rag_chunks > 0:
                # Generate rules automatically
                test_questions = [
                    "What are the key manufacturing requirements in this document?",
                    "What design guidelines are specified?",
                    "What quality standards are mentioned?",
                    "What material specifications are provided?"
                ]
                
                for question in test_questions:
                    try:
                        answer_data = qa_system.answer_question_with_citations(question)
                        if answer_data.get('extracted_rules'):
                            rules_data[question] = {
                                'answer': answer_data.get('answer', ''),
                                'rules': answer_data.get('extracted_rules', []),
                                'citations': answer_data.get('citations', [])
                            }
                    except Exception as e:
                        st.warning(f"Rule generation failed for: {question} - {str(e)}")
            
            # Stage 5: Save to database
            tracker.current_stage = 4
            tracker.progress = 1.0
            progress_bar.progress(tracker.progress)
            status_text.text(f"Stage {tracker.current_stage + 1}/{total_stages}")
            stage_text.text(f"🔄 {stages[tracker.current_stage]}")
            
            time.sleep(0.3)
            
            # Processing complete
            tracker.is_processing = False
            tracker.status_message = "Processing complete!"
            
            # Clear progress display
            progress_bar.empty()
            status_text.empty()
            stage_text.empty()
            
            # Show results
            st.success("✅ Document processed successfully!")
            
            # Display processing metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Text Sentences", 
                    processing_results.get('text_count', 0)
                )
            
            with col2:
                st.metric("RAG Chunks", rag_chunks)
            
            with col3:
                total_rules = sum(len(data.get('rules', [])) for data in rules_data.values())
                st.metric("Rules Generated", total_rules)
            
            with col4:
                images = processing_results.get('image_extraction', {}).get('images_found', 0)
                st.metric("Images Found", images)
            
            # Detailed results
            if rag_chunks > 0:
                st.info(f"""
                🧠 **Enhanced RAG Processing Complete!** 
                
                Your document has been processed with our advanced RAG system:
                - **{rag_chunks}** intelligent chunks created
                - **Advanced embeddings** using BAAI/bge-large-en-v1.5
                - **Manufacturing-aware metadata** extraction
                - **{total_rules}** rules automatically generated
                """)
                
                # Show generated rules
                if rules_data:
                    st.subheader("🤖 Auto-Generated Rules")
                    
                    for question, data in rules_data.items():
                        with st.expander(f"📋 {question}"):
                            st.write("**Answer:**")
                            st.write(data['answer'])
                            
                            if data['rules']:
                                st.write("**Extracted Rules:**")
                                for i, rule in enumerate(data['rules'], 1):
                                    st.write(f"{i}. {rule}")
                            
                            if data['citations']:
                                st.write("**Sources:**")
                                for citation in data['citations']:
                                    st.write(f"• {citation}")
                
                # Export option
                if rules_data:
                    st.subheader("💾 Export Rules")
                    
                    if st.button("📥 Export Rules to CSV"):
                        try:
                            csv_path = f"./exports/rules_{uploaded_file.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                            os.makedirs("./exports", exist_ok=True)
                            qa_system.export_rules_to_csv(csv_path)
                            st.success(f"Rules exported to: {csv_path}")
                        except Exception as e:
                            st.error(f"Export failed: {str(e)}")
            
            else:
                # Handle no chunks case
                warnings = processing_results.get('rag_processing', {}).get('warnings', [])
                if warnings:
                    st.warning(f"""
                    ⚠️ **No RAG Chunks Created**
                    
                    **Possible reasons:**
                    {chr(10).join(f'- {w}' for w in warnings)}
                    
                    **Solutions:**
                    - If scanned: Use OCR software to convert to searchable PDF
                    - If password-protected: Remove security restrictions
                    - Try uploading a text-based version of the document
                    """)
                else:
                    st.warning("⚠️ No text could be extracted from this PDF")
            
            # Store results in session state
            st.session_state.last_processing_results = {
                'file_name': uploaded_file.name,
                'processing_results': processing_results,
                'rules_data': rules_data,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            tracker.is_processing = False
            st.error(f"Processing failed: {str(e)}")
            
            # Clear progress display
            progress_bar.empty()
            status_text.empty()
            stage_text.empty()

# Show last processing results if available
if hasattr(st.session_state, 'last_processing_results'):
    results = st.session_state.last_processing_results
    
    with st.expander(f"📊 Last Processed: {results['file_name']}"):
        st.json(results['processing_results'])

# Knowledge base management
st.subheader("🗄️ Knowledge Base Management")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📈 View Database Stats"):
        stats = rag_pipeline.get_knowledge_base_summary()
        st.json(stats)

with col2:
    if st.button("🔍 Test Search"):
        if 'test_query' not in st.session_state:
            st.session_state.test_query = "manufacturing requirements"
        
        query = st.text_input("Search query:", value=st.session_state.test_query)
        if query:
            results = rag_pipeline.search_knowledge_base(query, top_k=3)
            st.write(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                st.write(f"{i}. {result.page_content[:100]}...")

with col3:
    if st.button("🗑️ Clear Database", type="secondary"):
        if st.session_state.get('confirm_clear', False):
            rag_pipeline.rag_system.clear_database()
            st.success("Knowledge base cleared!")
            st.session_state['confirm_clear'] = False
            st.rerun()
        else:
            st.session_state['confirm_clear'] = True
            st.warning("Click again to confirm")

# Instructions
with st.expander("📖 How to Use"):
    st.markdown("""
    ### Upload Process:
    
    1. **Select File**: Choose a PDF document to upload
    2. **Configure Options**: Set auto-rule generation and chunking parameters
    3. **Process**: Click "Process Document" to start
    4. **Monitor Progress**: Watch real-time progress through 5 processing stages
    5. **Review Results**: Check RAG chunks, generated rules, and processing metrics
    6. **Export**: Save generated rules to CSV for further analysis
    
    ### Processing Stages:
    
    1. **Text Extraction**: Extract text content from PDF
    2. **RAG Chunking**: Create intelligent chunks with context preservation
    3. **Embeddings**: Generate vector embeddings for semantic search
    4. **Rule Generation**: Automatically extract manufacturing rules using LLM
    5. **Database Storage**: Save processed data to vector database
    
    ### Auto Rule Generation:
    
    When enabled, the system automatically:
    - Asks predefined questions about the document
    - Generates detailed answers using LLM
    - Extracts specific manufacturing rules
    - Provides source citations
    - Formats results for easy export
    
    This provides immediate value from uploaded documents!
    """)