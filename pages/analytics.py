import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from typing import Dict, List, Tuple
import os
import sys

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.enhanced_rag_db import EnhancedManufacturingRAG, DocumentMetadata
from core.implicit_rule_extractor import ImplicitRuleExtractor

class RAGAnalyticsApp:
    def __init__(self):
        self.rag_system = None
        self.rule_extractor = None
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'rag_initialized' not in st.session_state:
            st.session_state.rag_initialized = False
        if 'documents_processed' not in st.session_state:
            st.session_state.documents_processed = 0
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = {
                'avg_query_time': 0,
                'total_queries': 0,
                'retrieval_accuracy': 0
            }
    
    def setup_page_config(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="Enhanced RAG Analytics",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🏭 Enhanced Manufacturing RAG Analytics")
        st.markdown("---")
    
    def initialize_rag_system(self):
        """Initialize the RAG system and rule extractor"""
        if not st.session_state.rag_initialized:
            with st.spinner("Initializing Enhanced RAG System..."):
                try:
                    self.rag_system = EnhancedManufacturingRAG(
                        persist_path="./chroma_enhanced_db"
                    )
                    self.rule_extractor = ImplicitRuleExtractor()
                    st.session_state.rag_initialized = True
                    st.success("✅ RAG System Initialized Successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to initialize RAG system: {str(e)}")
                    return False
        return True
    
    def sidebar_controls(self):
        """Create sidebar controls"""
        st.sidebar.header("📊 System Controls")
        
        # Document upload section
        st.sidebar.subheader("📁 Document Upload")
        uploaded_files = st.sidebar.file_uploader(
            "Upload Manufacturing Documents",
            type=['txt', 'pdf', 'docx'],
            accept_multiple_files=True,
            help="Upload documents to add to the RAG database"
        )
        
        if uploaded_files and st.sidebar.button("Process Documents"):
            self.process_uploaded_documents(uploaded_files)
        
        # System settings
        st.sidebar.subheader("⚙️ Settings")
        chunk_size = st.sidebar.slider("Chunk Size", 100, 2000, 500)
        overlap = st.sidebar.slider("Chunk Overlap", 0, 200, 50)
        k_results = st.sidebar.slider("Top-K Results", 1, 20, 5)
        
        # Performance monitoring
        st.sidebar.subheader("📈 Performance")
        if st.session_state.performance_metrics['total_queries'] > 0:
            st.sidebar.metric(
                "Avg Query Time", 
                f"{st.session_state.performance_metrics['avg_query_time']:.2f}s"
            )
            st.sidebar.metric(
                "Total Queries", 
                st.session_state.performance_metrics['total_queries']
            )
            st.sidebar.metric(
                "Documents Processed", 
                st.session_state.documents_processed
            )
        
        return chunk_size, overlap, k_results
    
    def process_uploaded_documents(self, uploaded_files):
        """Process uploaded documents"""
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            
            try:
                # Save uploaded file temporarily
                with open(f"temp_{uploaded_file.name}", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process with RAG system
                if self.rag_system:
                    self.rag_system.add_document(
                        file_path=f"temp_{uploaded_file.name}",
                        metadata=DocumentMetadata(
                            source=uploaded_file.name,
                            document_type="uploaded",
                            manufacturing_domain="general"
                        )
                    )
                    st.session_state.documents_processed += 1
                
                # Clean up temp file
                os.remove(f"temp_{uploaded_file.name}")
                
            except Exception as e:
                st.sidebar.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.text("✅ All documents processed!")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
    
    def query_interface(self):
        """Create the main query interface"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🔍 Query Interface")
            
            # Query input
            query = st.text_area(
                "Enter your manufacturing query:",
                placeholder="e.g., What are the quality control requirements for automotive parts?",
                height=100
            )
            
            # Query options
            col_a, col_b = st.columns(2)
            with col_a:
                use_implicit_extraction = st.checkbox("Use Implicit Rule Extraction", value=True)
            with col_b:
                include_metadata = st.checkbox("Include Metadata Analysis", value=True)
            
            if st.button("🚀 Execute Query", type="primary"):
                if query and self.rag_system:
                    self.execute_query(query, use_implicit_extraction, include_metadata)
        
        with col2:
            st.subheader("📋 Query History")
            if st.session_state.query_history:
                for i, hist_query in enumerate(reversed(st.session_state.query_history[-5:])):
                    with st.expander(f"Query {len(st.session_state.query_history) - i}"):
                        st.write(f"**Query:** {hist_query['query'][:100]}...")
                        st.write(f"**Time:** {hist_query['timestamp']}")
                        st.write(f"**Results:** {len(hist_query['results'])} documents")
    
    def execute_query(self, query: str, use_implicit_extraction: bool, include_metadata: bool):
        """Execute a query and display results"""
        start_time = time.time()
        
        with st.spinner("Processing query..."):
            try:
                # Execute RAG query
                results = self.rag_system.query(
                    query_text=query,
                    n_results=5,
                    include_metadata=include_metadata
                )
                
                # Extract implicit rules if requested
                implicit_rules = []
                if use_implicit_extraction and self.rule_extractor:
                    for result in results:
                        rules = self.rule_extractor.extract_rules(result.get('content', ''))
                        implicit_rules.extend(rules)
                
                query_time = time.time() - start_time
                
                # Update performance metrics
                self.update_performance_metrics(query_time)
                
                # Store in query history
                st.session_state.query_history.append({
                    'query': query,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'results': results,
                    'query_time': query_time,
                    'implicit_rules': implicit_rules
                })
                
                # Display results
                self.display_query_results(query, results, implicit_rules, query_time)
                
            except Exception as e:
                st.error(f"Query execution failed: {str(e)}")
    
    def update_performance_metrics(self, query_time: float):
        """Update performance metrics"""
        total_queries = st.session_state.performance_metrics['total_queries']
        avg_time = st.session_state.performance_metrics['avg_query_time']
        
        # Calculate new average
        new_avg = (avg_time * total_queries + query_time) / (total_queries + 1)
        
        st.session_state.performance_metrics.update({
            'avg_query_time': new_avg,
            'total_queries': total_queries + 1
        })
    
    def display_query_results(self, query: str, results: List[dict], 
                            implicit_rules: List[dict], query_time: float):
        """Display query results with analytics"""
        st.subheader(f"📊 Results for: {query[:50]}...")
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Query Time", f"{query_time:.3f}s")
        with col2:
            st.metric("Documents Found", len(results))
        with col3:
            st.metric("Implicit Rules", len(implicit_rules))
        
        # Results tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📄 Documents", "🔍 Rules", "📊 Analytics", "🎯 Relevance"])
        
        with tab1:
            self.display_documents_tab(results)
        
        with tab2:
            self.display_rules_tab(implicit_rules)
        
        with tab3:
            self.display_analytics_tab(results)
        
        with tab4:
            self.display_relevance_tab(results)
    
    def display_documents_tab(self, results: List[dict]):
        """Display documents tab"""
        for i, result in enumerate(results):
            with st.expander(f"Document {i+1} - Score: {result.get('score', 0):.3f}"):
                content = result.get('content', 'No content available')
                metadata = result.get('metadata', {})
                
                st.write("**Content:**")
                st.write(content[:500] + "..." if len(content) > 500 else content)
                
                if metadata:
                    st.write("**Metadata:**")
                    for key, value in metadata.items():
                        st.write(f"- **{key}:** {value}")
    
    def display_rules_tab(self, implicit_rules: List[dict]):
        """Display implicit rules tab"""
        if not implicit_rules:
            st.info("No implicit rules extracted. Try enabling implicit rule extraction.")
            return
        
        for i, rule in enumerate(implicit_rules):
            with st.expander(f"Rule {i+1} - Confidence: {rule.get('confidence', 0):.2f}"):
                st.write(f"**Type:** {rule.get('type', 'Unknown')}")
                st.write(f"**Content:** {rule.get('content', 'No content')}")
                st.write(f"**Context:** {rule.get('context', 'No context')}")
                
                if 'keywords' in rule:
                    st.write(f"**Keywords:** {', '.join(rule['keywords'])}")
    
    def display_analytics_tab(self, results: List[dict]):
        """Display analytics tab with visualizations"""
        if not results:
            st.info("No results to analyze.")
            return
        
        # Score distribution
        scores = [r.get('score', 0) for r in results]
        fig_scores = px.bar(
            x=range(1, len(scores) + 1),
            y=scores,
            title="Document Relevance Scores",
            labels={'x': 'Document Rank', 'y': 'Relevance Score'}
        )
        st.plotly_chart(fig_scores, use_container_width=True)
        
        # Metadata analysis
        domains = [r.get('metadata', {}).get('manufacturing_domain', 'Unknown') 
                  for r in results]
        if domains and any(d != 'Unknown' for d in domains):
            domain_counts = pd.Series(domains).value_counts()
            fig_domains = px.pie(
                values=domain_counts.values,
                names=domain_counts.index,
                title="Manufacturing Domains Distribution"
            )
            st.plotly_chart(fig_domains, use_container_width=True)
    
    def display_relevance_tab(self, results: List[dict]):
        """Display relevance analysis tab"""
        if not results:
            st.info("No results to analyze.")
            return
        
        st.subheader("🎯 Relevance Analysis")
        
        # Create relevance heatmap
        doc_lengths = [len(r.get('content', '')) for r in results]
        scores = [r.get('score', 0) for r in results]
        
        fig = go.Figure(data=go.Scatter(
            x=doc_lengths,
            y=scores,
            mode='markers+text',
            text=[f"Doc {i+1}" for i in range(len(results))],
            textposition="top center",
            marker=dict(
                size=10,
                color=scores,
                colorscale='Viridis',
                colorbar=dict(title="Relevance Score")
            )
        ))
        
        fig.update_layout(
            title="Document Length vs Relevance Score",
            xaxis_title="Document Length (characters)",
            yaxis_title="Relevance Score"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Top performing excerpts
        st.subheader("🏆 Top Performing Excerpts")
        for i, result in enumerate(results[:3]):
            score = result.get('score', 0)
            content = result.get('content', '')
            
            st.write(f"**Rank {i+1} (Score: {score:.3f})**")
            st.write(content[:200] + "..." if len(content) > 200 else content)
            st.write("---")
    
    def system_health_dashboard(self):
        """Display system health dashboard"""
        st.subheader("🏥 System Health Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Database Status", 
                "🟢 Online" if st.session_state.rag_initialized else "🔴 Offline"
            )
        
        with col2:
            st.metric("Documents in DB", st.session_state.documents_processed)
        
        with col3:
            queries_today = len([q for q in st.session_state.query_history 
                               if q['timestamp'].startswith(time.strftime("%Y-%m-%d"))])
            st.metric("Queries Today", queries_today)
        
        with col4:
            if st.session_state.query_history:
                avg_results = sum(len(q['results']) for q in st.session_state.query_history) / len(st.session_state.query_history)
                st.metric("Avg Results per Query", f"{avg_results:.1f}")
            else:
                st.metric("Avg Results per Query", "0.0")
        
        # Query performance over time
        if len(st.session_state.query_history) > 1:
            query_times = [q['query_time'] for q in st.session_state.query_history]
            timestamps = [q['timestamp'] for q in st.session_state.query_history]
            
            fig = px.line(
                x=range(len(query_times)),
                y=query_times,
                title="Query Performance Over Time",
                labels={'x': 'Query Number', 'y': 'Query Time (seconds)'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """Main application runner"""
        self.setup_page_config()
        
        # Initialize RAG system
        if not self.initialize_rag_system():
            st.stop()
        
        # Sidebar controls
        chunk_size, overlap, k_results = self.sidebar_controls()
        
        # Main interface tabs
        tab1, tab2 = st.tabs(["🔍 Query Interface", "🏥 System Health"])
        
        with tab1:
            self.query_interface()
        
        with tab2:
            self.system_health_dashboard()


def main():
    """Main function to run the Streamlit app"""
    app = RAGAnalyticsApp()
    app.run()


if __name__ == "__main__":
    main()