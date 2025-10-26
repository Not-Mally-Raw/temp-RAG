"""
Enhanced Question Answering with RAG and Citations
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any
import os
from datetime import datetime

# Import the enhanced QA system
from core.rag_pipeline_integration import init_rag_pipeline
from enhanced_qa_system import RAGQuestionAnswerer

# Initialize RAG pipeline
rag_pipeline = init_rag_pipeline()

# Initialize QA system
if 'qa_system' not in st.session_state:
    st.session_state['qa_system'] = RAGQuestionAnswerer(rag_pipeline)

qa_system = st.session_state['qa_system']

st.title("🎯 Enhanced RAG Question Answering")

st.markdown("""
Ask detailed questions about manufacturing processes and get:
- **Detailed LLM-generated answers**
- **Academic-style citations** 
- **Extracted manufacturing rules**
- **CSV export** of all rules
""")

# Sidebar for settings
st.sidebar.subheader("⚙️ QA Settings")

# Question input
question = st.text_area(
    "❓ Enter your manufacturing question:",
    placeholder="What are the bend radius requirements for sheet metal parts?",
    height=100
)

# Manufacturing process filter
manufacturing_process = st.sidebar.selectbox(
    "🏭 Manufacturing Process",
    ["None", "Sheet Metal", "Injection Molding", "Machining", "Assembly", "General"]
)

# Advanced settings
with st.sidebar.expander("🔧 Advanced Settings"):
    top_k = st.slider("Number of sources to retrieve", 1, 10, 5)
    model = st.selectbox(
        "LLM Model", 
        ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"]
    )

# Main question answering
if st.button("🔍 Ask Question", type="primary"):
    if not question.strip():
        st.error("Please enter a question.")
    else:
        with st.spinner("🧠 Processing question with RAG + LLM..."):
            try:
                process_filter = None if manufacturing_process == "None" else manufacturing_process
                
                result = qa_system.answer_question_with_citations(
                    question=question,
                    manufacturing_process=process_filter,
                    top_k=top_k,
                    model=model
                )
                
                # Display results
                st.success("✅ Question processed successfully!")
                
                # Main answer
                st.subheader("📝 Detailed Answer")
                st.markdown(result['detailed_answer'])
                
                # Confidence and metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Confidence", f"{result['confidence_score']:.3f}")
                with col2:
                    st.metric("Sources Used", result['metadata']['retrieval_count'])
                with col3:
                    st.metric("Rules Extracted", len(result['extracted_rules']))
                
                # Citations
                if result['citations']:
                    st.subheader("📚 Sources & Citations")
                    for citation in result['citations']:
                        with st.expander(f"📄 {citation['citation_format']} (Relevance: {citation['relevance_score']:.3f})"):
                            st.write(f"**File:** {citation['source_file']}")
                            st.write(f"**Section:** {citation['section']}")
                            st.write(f"**Process:** {citation['manufacturing_process']}")
                            st.write(f"**Page:** {citation['page_number']}")
                            st.write("**Excerpt:**")
                            st.info(citation['excerpt'])
                
                # Extracted Rules
                if result['extracted_rules']:
                    st.subheader("⚙️ Manufacturing Rules Extracted")
                    for i, rule in enumerate(result['extracted_rules'], 1):
                        with st.expander(f"Rule {i}: {rule['rule_type']} ({rule['confidence']:.3f})"):
                            st.write(f"**Rule:** {rule['rule_text']}")
                            st.write(f"**Type:** {rule['rule_type']}")
                            st.write(f"**Source:** {rule['source_citation']} {rule['source_file']}")
                            st.write(f"**Process:** {rule['manufacturing_process']}")
                            st.write(f"**Method:** {rule['extraction_method']}")
                
                # Store result in session for later export
                if 'qa_results' not in st.session_state:
                    st.session_state['qa_results'] = []
                st.session_state['qa_results'].append(result)
                
            except Exception as e:
                st.error(f"Error processing question: {e}")
                st.exception(e)

# Rules Database Management
st.subheader("🗄️ Extracted Rules Database")

col1, col2 = st.columns(2)

with col1:
    if st.button("📊 View Rules Summary"):
        summary = qa_system.get_rules_summary()
        
        if summary['total_rules'] > 0:
            st.json(summary)
            
            # Create charts
            if summary['rules_by_type']:
                st.subheader("📈 Rules by Type")
                type_df = pd.DataFrame(list(summary['rules_by_type'].items()), 
                                     columns=['Rule Type', 'Count'])
                st.bar_chart(type_df.set_index('Rule Type'))
            
            if summary['rules_by_process']:
                st.subheader("🏭 Rules by Manufacturing Process")
                process_df = pd.DataFrame(list(summary['rules_by_process'].items()), 
                                        columns=['Process', 'Count'])
                st.bar_chart(process_df.set_index('Process'))
        else:
            st.info("No rules extracted yet. Ask some questions to build the database!")

with col2:
    if st.button("💾 Export Rules to CSV"):
        if qa_system.extracted_rules_db:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"manufacturing_rules_{timestamp}.csv"
            
            try:
                csv_file = qa_system.export_rules_to_csv(filename)
                
                # Create download button
                with open(filename, 'rb') as f:
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=f.read(),
                        file_name=filename,
                        mime="text/csv"
                    )
                
                st.success(f"✅ Exported {len(qa_system.extracted_rules_db)} rules!")
                
                # Display preview
                df = pd.DataFrame(qa_system.extracted_rules_db)
                st.subheader("📄 CSV Preview")
                st.dataframe(df.head(10))
                
            except Exception as e:
                st.error(f"Export failed: {e}")
        else:
            st.warning("No rules to export yet.")

# Question History
if 'qa_results' in st.session_state and st.session_state['qa_results']:
    st.subheader("📚 Question History")
    
    for i, result in enumerate(reversed(st.session_state['qa_results'][-5:]), 1):
        with st.expander(f"Q{i}: {result['question'][:60]}..."):
            st.write(f"**Answer Preview:** {result['detailed_answer'][:200]}...")
            st.write(f"**Sources:** {len(result['citations'])}")
            st.write(f"**Rules:** {len(result['extracted_rules'])}")
            st.write(f"**Timestamp:** {result['metadata']['timestamp']}")

# Knowledge Base Status
st.sidebar.subheader("📊 Knowledge Base Status")
kb_stats = rag_pipeline.get_knowledge_base_summary()
st.sidebar.metric("Documents", kb_stats['total_documents'])
st.sidebar.metric("Chunks", kb_stats['total_chunks'])

if kb_stats['processed_files']:
    with st.sidebar.expander("📁 Processed Files"):
        for filename in kb_stats['processed_files']:
            st.sidebar.write(f"• {filename}")

# Quick test queries
st.sidebar.subheader("🚀 Quick Test Queries")
test_queries = [
    "What are bend radius requirements?",
    "Minimum hole diameter specifications?",
    "Material tolerance standards?",
    "Assembly fastener requirements?"
]

for query in test_queries:
    if st.sidebar.button(f"📝 {query}", key=f"test_{query}"):
        st.experimental_set_query_params(question=query)
        st.experimental_rerun()

# Instructions
with st.expander("ℹ️ How to Use This System"):
    st.markdown("""
    ### 🎯 Enhanced RAG Question Answering Features:
    
    1. **Detailed Answers**: Get comprehensive LLM-generated responses
    2. **Academic Citations**: See exactly where information comes from
    3. **Rule Extraction**: Automatically extract manufacturing rules
    4. **CSV Export**: Download all extracted rules for analysis
    
    ### 📝 Example Questions:
    - "What are the bend radius requirements for sheet metal parts?"
    - "What tolerances are required for injection molded parts?"
    - "How should holes be spaced in machined components?"
    
    ### 🔧 Features:
    - **Citations**: Each answer includes source references [1], [2], etc.
    - **Rule Database**: All extracted rules are stored and can be exported
    - **Process Filtering**: Filter by manufacturing process for targeted results
    - **Multiple Models**: Choose from different LLM models for varied responses
    """)