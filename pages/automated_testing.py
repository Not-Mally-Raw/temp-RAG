"""
Automated Testing & Rule Generation System
Optimized for performance with real-time progress tracking
"""

import streamlit as st
import os
import time
import threading
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime

from core.rag_pipeline_integration import RAGIntegratedPipeline
from enhanced_qa_system import RAGQuestionAnswerer
from core.rag_database import DocumentMetadata

class OptimizedTestingSystem:
    def __init__(self):
        self.pipeline = RAGIntegratedPipeline(persist_directory='./automated_test_db')
        self.qa_system = RAGQuestionAnswerer(self.pipeline)
        self.test_results = []
        self.current_progress = 0
        self.total_steps = 0
        self.current_status = "Ready"
        self.is_processing = False
        self.processing_thread = None

    def find_test_documents(self, folder_paths: List[str]) -> List[str]:
        """Find all PDF files in the given folders"""
        pdf_files = []

        for folder_path in folder_paths:
            if not os.path.exists(folder_path):
                continue

            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith('.pdf') and not file.lower().endswith('.xlsx'):
                        pdf_files.append(os.path.join(root, file))

        return pdf_files

    def process_single_document(self, file_path: str) -> Dict[str, Any]:
        """Process a single document through the complete pipeline"""
        start_time = time.time()

        # Step 1: Document Processing
        self.current_status = f"Processing: {os.path.basename(file_path)}"

        try:
            # Process document
            result = self.pipeline.process_document(file_path)

            # Step 2: QA Testing
            self.current_status = f"Running QA tests: {os.path.basename(file_path)}"

            qa_results = []
            test_questions = [
                "What are the key manufacturing requirements?",
                "What design guidelines are mentioned?",
                "What are the quality standards?",
                "What material specifications are provided?",
                "What are the tolerance requirements?"
            ]

            for question in test_questions:
                try:
                    answer, citations, rules = self.qa_system.ask_question_with_rules(question)
                    qa_results.append({
                        'question': question,
                        'answer': answer,
                        'citations': len(citations),
                        'rules_found': len(rules)
                    })
                except Exception as e:
                    qa_results.append({
                        'question': question,
                        'error': str(e)
                    })

            processing_time = time.time() - start_time

            return {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'processing_time': processing_time,
                'rag_chunks': result.get('chunks_created', 0),
                'search_results': len(result.get('search_results', [])),
                'qa_results': qa_results,
                'csv_exported': result.get('csv_exported', False),
                'success': True,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            processing_time = time.time() - start_time
            return {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'processing_time': processing_time,
                'error': str(e),
                'success': False,
                'timestamp': datetime.now().isoformat()
            }

    def run_automated_tests(self, folder_paths: List[str], progress_callback=None):
        """Run automated tests on all documents in the folders"""
        self.is_processing = True
        self.test_results = []
        self.current_progress = 0

        # Find all documents
        pdf_files = self.find_test_documents(folder_paths)
        self.total_steps = len(pdf_files)

        if not pdf_files:
            self.current_status = "No PDF files found"
            self.is_processing = False
            return

        self.current_status = f"Starting tests on {len(pdf_files)} documents"

        # Process each document
        for i, file_path in enumerate(pdf_files):
            if not self.is_processing:  # Allow stopping
                break

            self.current_progress = i + 1
            result = self.process_single_document(file_path)
            self.test_results.append(result)

            if progress_callback:
                progress_callback(i + 1, len(pdf_files), result)

        self.current_status = "Testing completed"
        self.is_processing = False

    def get_summary_stats(self):
        """Get summary statistics from test results"""
        if not self.test_results:
            return None

        total_files = len(self.test_results)
        successful = sum(1 for r in self.test_results if r.get('success', False))
        total_chunks = sum(r.get('rag_chunks', 0) for r in self.test_results if r.get('success', False))
        total_rules = sum(sum(qa.get('rules_found', 0) for qa in r.get('qa_results', [])) for r in self.test_results if r.get('success', False))
        avg_processing_time = sum(r.get('processing_time', 0) for r in self.test_results) / total_files if total_files > 0 else 0

        return {
            'total_files': total_files,
            'successful': successful,
            'failed': total_files - successful,
            'success_rate': (successful / total_files) * 100 if total_files > 0 else 0,
            'total_rag_chunks': total_chunks,
            'total_rules_extracted': total_rules,
            'avg_processing_time': avg_processing_time
        }

    def process_single_document(self, file_path: str) -> Dict[str, Any]:
        """Process a single document through the entire pipeline"""
        start_time = time.time()

        try:
            # Step 1: File Processing
            self.current_status = f"Processing: {os.path.basename(file_path)}"

            # Read file and create mock uploaded file object
            with open(file_path, 'rb') as f:
                file_content = f.read()

            # Create a mock uploaded file object
            class MockUploadedFile:
                def __init__(self, content, name):
                    self.content = content
                    self.name = name

                def read(self):
                    return self.content

                def getvalue(self):
                    return self.content

            mock_file = MockUploadedFile(file_content, os.path.basename(file_path))

            # Step 2: RAG Processing
            self.current_status = f"RAG Processing: {os.path.basename(file_path)}"
            processing_results = self.pipeline.process_uploaded_file(mock_file)

            # Check if RAG processing was successful
            rag_chunks = processing_results.get('rag_processing', {}).get('text_chunks', 0)
            if rag_chunks == 0:
                # Try to get chunks from different possible locations
                rag_chunks = processing_results.get('rag_processing', {}).get('chunks_created', 0)

            # Step 3: Rule Generation
            self.current_status = f"Generating Rules: {os.path.basename(file_path)}"

            # Test basic RAG search
            search_results = self.pipeline.search_knowledge_base("manufacturing requirements", top_k=5)

            # Test QA system with better error handling
            test_questions = [
                "What are the key manufacturing requirements?",
                "What design guidelines are mentioned?",
                "What are the quality standards?",
                "What material specifications are provided?",
                "What are the tolerance requirements?"
            ]

            qa_results = []
            for question in test_questions:
                try:
                    answer_data = self.qa_system.answer_question_with_citations(question)
                    qa_results.append({
                        'question': question,
                        'answer': answer_data.get('detailed_answer', answer_data.get('answer', 'No answer generated')),
                        'citations': len(answer_data.get('citations', [])),
                        'rules_found': len(answer_data.get('extracted_rules', []))
                    })
                except Exception as e:
                    qa_results.append({
                        'question': question,
                        'error': str(e)
                    })

            # Step 4: Export Rules
            self.current_status = f"Exporting Rules: {os.path.basename(file_path)}"

            try:
                csv_path = f"./test_results/{os.path.basename(file_path)}_rules.csv"
                os.makedirs("./test_results", exist_ok=True)
                exported_count = self.qa_system.export_rules_to_csv(csv_path)
                csv_exported = exported_count > 0 if isinstance(exported_count, int) else True
            except Exception as e:
                csv_exported = False
                csv_error = str(e)

            processing_time = time.time() - start_time

            # Compile results
            result = {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'processing_time': processing_time,
                'processing_results': processing_results,
                'rag_chunks': rag_chunks,
                'search_results': len(search_results),
                'qa_results': qa_results,
                'csv_exported': csv_exported,
                'success': True,
                'timestamp': datetime.now().isoformat()
            }

            if not csv_exported:
                result['csv_error'] = csv_error

        except Exception as e:
            result = {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'error': str(e),
                'success': False,
                'timestamp': datetime.now().isoformat()
            }

        return result
    
    def run_automated_tests(self, folder_paths: List[str], progress_callback=None):
        """Run automated tests on all documents in folders"""
        self.is_processing = True
        
        # Find all PDF files
        pdf_files = self.find_test_documents(folder_paths)
        self.total_steps = len(pdf_files) * 4  # 4 steps per file
        self.current_progress = 0
        
        if not pdf_files:
            self.current_status = "No PDF files found in specified folders"
            self.is_processing = False
            return
        
        self.current_status = f"Found {len(pdf_files)} PDF files to process"
        
        # Process each file
        for i, file_path in enumerate(pdf_files):
            if not self.is_processing:  # Check for stop signal
                break
                
            result = self.process_single_document(file_path)
            self.test_results.append(result)
            
            # Update progress
            self.current_progress = (i + 1) * 4
            if progress_callback:
                progress_callback(self.current_progress, self.total_steps, result)
        
        self.current_status = "Testing Complete"
        self.is_processing = False


# Initialize testing system
if 'testing_system' not in st.session_state:
    st.session_state.testing_system = OptimizedTestingSystem()

testing_system = st.session_state.testing_system

st.title("🔄 Automated Testing & Rule Generation System")
st.write("Test the entire pipeline on DFM Handbook documents with real-time progress tracking")

# Configuration section
st.subheader("📁 Test Configuration")

col1, col2 = st.columns(2)

with col1:
    # Folder paths input
    dfm_folder = st.text_input(
        "DFM Handbook Folder Path",
        value="/Users/spandankewte/Downloads/DFM Handbook data",
        help="Path to folder containing DFM Handbook PDF documents"
    )
    
    additional_data_folder = st.text_input(
        "Additional Data Folder Path", 
        value="/opt/anaconda3/Phase-3-Final-master/data",
        help="Path to additional manufacturing data folder"
    )
    
    # Combine folders
    folder_paths = []
    if os.path.exists(dfm_folder):
        folder_paths.append(dfm_folder)
    if os.path.exists(additional_data_folder):
        folder_paths.append(additional_data_folder)
    
    # Check folders and show file count
    if folder_paths:
        pdf_files = testing_system.find_test_documents(folder_paths)
        st.success(f"✅ Found folders: {len(pdf_files)} PDF files detected")
        
        if pdf_files:
            with st.expander("📄 Files to Process"):
                for file in pdf_files:
                    folder_name = "DFM Handbook" if dfm_folder in file else "Additional Data"
                    st.write(f"• [{folder_name}] {os.path.basename(file)}")
    else:
        st.error("❌ No valid folders found")
        pdf_files = []

with col2:
    # Test options
    st.write("**Test Options:**")
    clear_db = st.checkbox("Clear database before testing", value=True)
    export_results = st.checkbox("Export detailed results to JSON", value=True)
    
    # Real-time updates
    auto_refresh = st.checkbox("Auto-refresh progress", value=True)

# Progress tracking section
st.subheader("📊 Progress Tracking")

if testing_system.is_processing:
    # Progress bar
    progress = testing_system.current_progress / max(testing_system.total_steps, 1)
    st.progress(progress)
    
    # Status and metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Step", f"{testing_system.current_progress}/{testing_system.total_steps}")
    
    with col2:
        st.metric("Files Processed", len(testing_system.test_results))
    
    with col3:
        if testing_system.test_results:
            successful = sum(1 for r in testing_system.test_results if r.get('success', False))
            st.metric("Success Rate", f"{(successful/len(testing_system.test_results)*100):.1f}%")
    
    # Current status
    st.info(f"🔄 {testing_system.current_status}")
    
    # Stop button
    if st.button("🛑 Stop Testing", type="secondary"):
        testing_system.is_processing = False
        st.warning("Stopping tests...")
        
    # Auto-refresh
    if auto_refresh:
        time.sleep(1)
        st.rerun()

else:
    # Start testing button
    if pdf_files and st.button("🚀 Start Automated Testing", type="primary"):
        # Clear database if requested
        if clear_db:
            testing_system.pipeline.rag_system.clear_database()
            st.success("Database cleared")
        
        # Clear previous results
        testing_system.test_results = []
        
        # Start testing in a separate thread
        def progress_callback(current, total, result):
            # Update session state (will be reflected on next rerun)
            pass
        
        testing_system.run_automated_tests(folder_paths, progress_callback)
        st.rerun()

# Results section
if testing_system.test_results:
    st.subheader("📈 Test Results")
    
    # Summary statistics
    summary_stats = testing_system.get_summary_stats()
    
    if summary_stats:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Files", summary_stats['total_files'])
        
        with col2:
            st.metric("Successful", summary_stats['successful'])
        
        with col3:
            st.metric("Success Rate", f"{summary_stats['success_rate']:.1f}%")
        
        with col4:
            st.metric("RAG Chunks", summary_stats['total_rag_chunks'])
        
        with col5:
            st.metric("Rules Extracted", summary_stats['total_rules_extracted'])
        
        st.metric("Avg Processing Time", f"{summary_stats['avg_processing_time']:.2f}s")
    
    # Detailed results
    st.subheader("📋 Detailed Results")
    
    for i, result in enumerate(testing_system.test_results):
        with st.expander(f"📄 {result['file_name']} - {'✅ Success' if result['success'] else '❌ Failed'}"):
            if result['success']:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Processing Metrics:**")
                    st.write(f"• RAG Chunks: {result['rag_chunks']}")
                    st.write(f"• Search Results: {result['search_results']}")
                    st.write(f"• Processing Time: {result['processing_time']:.2f}s")
                    st.write(f"• CSV Exported: {'✅' if result['csv_exported'] else '❌'}")
                
                with col2:
                    st.write("**QA Test Results:**")
                    for qa in result['qa_results']:
                        if 'error' not in qa:
                            st.write(f"• {qa['question'][:50]}...")
                            st.write(f"  → Citations: {qa['citations']}, Rules: {qa['rules_found']}")
                        else:
                            st.write(f"• ❌ {qa['question'][:50]}... - Error: {qa['error']}")
                
                # Show processing details
                if st.checkbox(f"Show raw data for {result['file_name']}", key=f"raw_{i}"):
                    st.json(result)
            else:
                st.error(f"Error: {result.get('error', 'Unknown error')}")
    
    # Export results
    if export_results and st.button("💾 Export Results to JSON"):
        results_path = f"./test_results/automated_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("./test_results", exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump({
                'summary': summary_stats,
                'detailed_results': testing_system.test_results,
                'test_config': {
                    'dfm_folder': dfm_folder,
                    'additional_data_folder': additional_data_folder,
                    'total_files': len(pdf_files),
                    'timestamp': datetime.now().isoformat()
                }
            }, f, indent=2)
        
        st.success(f"Results exported to: {results_path}")

# System status
st.subheader("🔧 System Status")

col1, col2 = st.columns(2)

with col1:
    if st.button("📊 Check RAG Database Stats"):
        stats = testing_system.pipeline.get_knowledge_base_summary()
        st.json(stats)

with col2:
    if st.button("🧹 Clear Test Results"):
        testing_system.test_results = []
        testing_system.current_progress = 0
        testing_system.total_steps = 0
        testing_system.current_status = "Ready"
        st.success("Test results cleared")
        st.rerun()

# Instructions
with st.expander("📖 How to Use"):
    st.markdown("""
    ### Automated Testing Process:
    
    1. **Configure**: Set the folder path containing your DFM Handbook PDFs
    2. **Start**: Click "Start Automated Testing" to begin
    3. **Monitor**: Watch real-time progress and status updates
    4. **Review**: Check detailed results for each document
    5. **Export**: Save results and generated rules for analysis
    
    ### What Gets Tested:
    
    - **Document Processing**: PDF text extraction and parsing
    - **RAG Chunking**: Intelligent text chunking and vectorization
    - **Knowledge Retrieval**: Search functionality and relevance scoring
    - **Rule Generation**: Automated rule extraction using LLM
    - **QA System**: Question answering with citations
    - **CSV Export**: Rule database export functionality
    
    ### Progress Tracking:
    
    Each document goes through 4 main steps:
    1. Document processing and text extraction
    2. RAG chunking and vectorization
    3. Rule generation and QA testing
    4. Results compilation and CSV export
    
    Real-time updates show current status, progress, and success rates.
    """)