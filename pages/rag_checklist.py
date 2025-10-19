"""
RAG System Checklist Page
Shows the complete RAG pipeline process with visual status indicators
"""

import streamlit as st
from pathlib import Path
import sys
import time
from datetime import datetime
import json

# Add parent directory to path for imports
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

from core.universal_rag_system import UniversalManufacturingRAG
from core.implicit_rule_extractor import ImplicitRuleExtractor
from extractors.text import extract_sentences
import io


class RAGChecklistApp:
    """Complete RAG pipeline checklist application."""
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'rag_system' not in st.session_state:
            st.session_state.rag_system = None
        if 'checklist_status' not in st.session_state:
            st.session_state.checklist_status = {
                'system_initialized': False,
                'document_uploaded': False,
                'text_extracted': False,
                'embeddings_created': False,
                'stored_in_db': False,
                'rules_generated': False
            }
        if 'processed_document' not in st.session_state:
            st.session_state.processed_document = None
        if 'extracted_text' not in st.session_state:
            st.session_state.extracted_text = None
        if 'embeddings_info' not in st.session_state:
            st.session_state.embeddings_info = None
        if 'extracted_rules' not in st.session_state:
            st.session_state.extracted_rules = []
        if 'db_stats' not in st.session_state:
            st.session_state.db_stats = None
    
    def render_status_indicator(self, status: bool, label: str):
        """Render a status indicator with checkmark or pending icon."""
        icon = "✅" if status else "⏳"
        color = "green" if status else "orange"
        st.markdown(f":{color}[{icon} {label}]")
    
    def run(self):
        """Main application runner."""
        st.set_page_config(
            page_title="RAG System Checklist",
            page_icon="✅",
            layout="wide"
        )
        
        st.title("✅ RAG System Implementation Checklist")
        st.subheader("Complete Pipeline Status & Verification")
        
        # Display checklist status
        st.markdown("---")
        st.header("📋 Pipeline Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**System Setup**")
            self.render_status_indicator(
                st.session_state.checklist_status['system_initialized'],
                "RAG System Initialized"
            )
            self.render_status_indicator(
                st.session_state.checklist_status['document_uploaded'],
                "Document Uploaded"
            )
        
        with col2:
            st.markdown("**Processing**")
            self.render_status_indicator(
                st.session_state.checklist_status['text_extracted'],
                "Text Extracted"
            )
            self.render_status_indicator(
                st.session_state.checklist_status['embeddings_created'],
                "Embeddings Created"
            )
        
        with col3:
            st.markdown("**Storage & Analysis**")
            self.render_status_indicator(
                st.session_state.checklist_status['stored_in_db'],
                "Stored in Vector DB"
            )
            self.render_status_indicator(
                st.session_state.checklist_status['rules_generated'],
                "Rules Generated"
            )
        
        st.markdown("---")
        
        # Step 1: Initialize RAG System
        st.header("1️⃣ Initialize RAG System")
        
        if st.button("🚀 Initialize RAG System", type="primary"):
            with st.spinner("Initializing RAG system with embeddings..."):
                try:
                    # Initialize the Universal RAG system
                    st.session_state.rag_system = UniversalManufacturingRAG(
                        embedding_model_name="BAAI/bge-large-en-v1.5",
                        persist_path="./rag_checklist_db",
                        chunk_size=800,
                        chunk_overlap=100
                    )
                    st.session_state.checklist_status['system_initialized'] = True
                    st.success("✅ RAG System initialized successfully!")
                    st.info(f"**Embedding Model:** BAAI/bge-large-en-v1.5")
                    st.info(f"**Chunk Size:** 800 characters")
                    st.info(f"**Chunk Overlap:** 100 characters")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error initializing RAG system: {str(e)}")
        
        if st.session_state.checklist_status['system_initialized']:
            st.success("✅ RAG System is initialized and ready!")
        
        # Step 2: Upload Document
        st.markdown("---")
        st.header("2️⃣ Upload Document")
        
        if st.session_state.checklist_status['system_initialized']:
            uploaded_file = st.file_uploader(
                "Upload a PDF document",
                type=['pdf'],
                help="Upload a PDF document to process"
            )
            
            if uploaded_file is not None:
                st.session_state.processed_document = {
                    'filename': uploaded_file.name,
                    'size': uploaded_file.size,
                    'content': uploaded_file.read()
                }
                st.session_state.checklist_status['document_uploaded'] = True
                st.success(f"✅ Document uploaded: {uploaded_file.name}")
                st.info(f"**File Size:** {uploaded_file.size / 1024:.2f} KB")
        else:
            st.warning("⚠️ Please initialize the RAG system first")
        
        # Step 3: Extract Text
        st.markdown("---")
        st.header("3️⃣ Extract Text from Document")
        
        if st.session_state.checklist_status['document_uploaded']:
            if st.button("📄 Extract Text", type="primary"):
                with st.spinner("Extracting text from document..."):
                    try:
                        doc_bytes = st.session_state.processed_document['content']
                        sentences = extract_sentences(doc_bytes)
                        
                        if sentences:
                            st.session_state.extracted_text = ' '.join(sentences)
                            st.session_state.checklist_status['text_extracted'] = True
                            
                            st.success(f"✅ Extracted {len(sentences)} sentences from document")
                            
                            # Show preview
                            with st.expander("📖 View Extracted Text Preview"):
                                preview_text = st.session_state.extracted_text[:1000]
                                st.text_area(
                                    "First 1000 characters:",
                                    preview_text,
                                    height=200,
                                    disabled=True
                                )
                            st.rerun()
                        else:
                            st.error("❌ No text could be extracted from the document")
                    except Exception as e:
                        st.error(f"❌ Error extracting text: {str(e)}")
        else:
            st.warning("⚠️ Please upload a document first")
        
        if st.session_state.checklist_status['text_extracted']:
            st.success("✅ Text extracted successfully!")
            text_length = len(st.session_state.extracted_text)
            st.info(f"**Total Characters:** {text_length:,}")
            st.info(f"**Estimated Words:** {text_length // 5:,}")
        
        # Step 4: Create Embeddings
        st.markdown("---")
        st.header("4️⃣ Create Vector Embeddings")
        
        if st.session_state.checklist_status['text_extracted']:
            if st.button("🧮 Create Embeddings", type="primary"):
                with st.spinner("Creating embeddings with BAAI/bge-large-en-v1.5..."):
                    try:
                        # Process the document to create embeddings
                        doc_bytes = st.session_state.processed_document['content']
                        filename = st.session_state.processed_document['filename']
                        
                        results = st.session_state.rag_system.process_any_document(
                            doc_bytes,
                            filename
                        )
                        
                        st.session_state.embeddings_info = results
                        st.session_state.checklist_status['embeddings_created'] = True
                        st.session_state.checklist_status['stored_in_db'] = True
                        
                        st.success("✅ Embeddings created and stored in vector database!")
                        
                        # Display embedding statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Text Chunks", results.get('text_chunks', 0))
                        with col2:
                            st.metric("Keyword Rules", results.get('keyword_rules', 0))
                        with col3:
                            st.metric("Implicit Rules", results.get('implicit_rules', 0))
                        
                        st.info(f"**Processing Method:** {results.get('processing_method', 'Unknown')}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error creating embeddings: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        else:
            st.warning("⚠️ Please extract text first")
        
        if st.session_state.checklist_status['embeddings_created']:
            st.success("✅ Vector embeddings created!")
            if st.session_state.embeddings_info:
                with st.expander("📊 View Embedding Details"):
                    st.json(st.session_state.embeddings_info)
        
        # Step 5: Verify Database Storage
        st.markdown("---")
        st.header("5️⃣ Verify Vector Database Storage")
        
        if st.session_state.checklist_status['stored_in_db']:
            if st.button("🔍 Check Database", type="primary"):
                with st.spinner("Querying vector database..."):
                    try:
                        # Get database statistics
                        stats = st.session_state.rag_system.get_enhanced_stats()
                        st.session_state.db_stats = stats
                        
                        st.success("✅ Database verified successfully!")
                        
                        # Display database stats
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📊 Database Statistics")
                            st.metric("Total Documents", stats.get('total_documents', 0))
                            st.metric("Total Chunks", stats.get('total_chunks', 0))
                            st.metric("Embedding Model", stats.get('embedding_model', 'Unknown'))
                        
                        with col2:
                            st.subheader("🔧 Processing Methods")
                            methods = stats.get('processing_methods', {})
                            st.metric("Keyword-Based Rules", methods.get('keyword_based_rules', 0))
                            st.metric("Implicit Rules", methods.get('implicit_rules', 0))
                            st.metric("Hybrid Rules", methods.get('hybrid_rules', 0))
                        
                        with st.expander("📋 View Full Database Stats"):
                            st.json(stats)
                    except Exception as e:
                        st.error(f"❌ Error checking database: {str(e)}")
        else:
            st.warning("⚠️ Please create embeddings first")
        
        # Step 6: Generate Rules
        st.markdown("---")
        st.header("6️⃣ Generate Manufacturing Rules")
        
        if st.session_state.checklist_status['stored_in_db']:
            if st.button("⚙️ Generate Rules", type="primary"):
                with st.spinner("Extracting manufacturing rules from document..."):
                    try:
                        # Extract implicit rules
                        extractor = ImplicitRuleExtractor()
                        rules = extractor.extract_implicit_rules(
                            st.session_state.extracted_text,
                            confidence_threshold=0.5
                        )
                        
                        st.session_state.extracted_rules = rules
                        st.session_state.checklist_status['rules_generated'] = True
                        
                        st.success(f"✅ Generated {len(rules)} manufacturing rules!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error generating rules: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        else:
            st.warning("⚠️ Please store embeddings in database first")
        
        if st.session_state.checklist_status['rules_generated']:
            st.success(f"✅ {len(st.session_state.extracted_rules)} rules generated!")
            
            # Display rules
            st.subheader("📜 Extracted Manufacturing Rules")
            
            for i, rule in enumerate(st.session_state.extracted_rules[:10], 1):  # Show first 10
                with st.expander(f"Rule #{i} - {rule.rule_type} (Confidence: {rule.confidence_score:.2f})"):
                    st.markdown(f"**Text:** {rule.text}")
                    st.markdown(f"**Rule Type:** {rule.rule_type}")
                    st.markdown(f"**Constraint Type:** {rule.constraint_type}")
                    st.markdown(f"**Confidence Score:** {rule.confidence_score:.2f}")
                    st.markdown(f"**Manufacturing Relevance:** {rule.manufacturing_relevance:.2f}")
                    
                    if rule.semantic_features:
                        st.markdown(f"**Semantic Features:** {', '.join(rule.semantic_features[:5])}")
            
            if len(st.session_state.extracted_rules) > 10:
                st.info(f"Showing 10 of {len(st.session_state.extracted_rules)} rules. All rules are stored in the database.")
        
        # Final Summary
        if all(st.session_state.checklist_status.values()):
            st.markdown("---")
            st.success("🎉 **ALL STEPS COMPLETED!** RAG System is fully operational.")
            
            st.balloons()
            
            # Summary card
            st.subheader("📊 Complete Summary")
            
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                st.markdown("**Document Info**")
                if st.session_state.processed_document:
                    st.write(f"Filename: {st.session_state.processed_document['filename']}")
                    st.write(f"Size: {st.session_state.processed_document['size'] / 1024:.2f} KB")
            
            with summary_col2:
                st.markdown("**Processing Stats**")
                if st.session_state.embeddings_info:
                    st.write(f"Text Chunks: {st.session_state.embeddings_info.get('text_chunks', 0)}")
                    st.write(f"Processing: {st.session_state.embeddings_info.get('processing_method', 'N/A')}")
            
            with summary_col3:
                st.markdown("**Results**")
                st.write(f"Rules Generated: {len(st.session_state.extracted_rules)}")
                if st.session_state.db_stats:
                    st.write(f"DB Documents: {st.session_state.db_stats.get('total_documents', 0)}")
        
        # Reset button
        st.markdown("---")
        if st.button("🔄 Reset All", type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


def main():
    """Main entry point."""
    app = RAGChecklistApp()
    app.run()


if __name__ == "__main__":
    main()
