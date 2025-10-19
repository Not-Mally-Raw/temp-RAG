"""
Universal RAG Testing Simulation
Interactive Streamlit app to test and validate the universal document processing capabilities
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
import numpy as np
from typing import Dict, List, Any, Optional
import tempfile
import os

# Import our core systems
from core.implicit_rule_extractor import ImplicitRuleExtractor, ImplicitRule
from core.universal_rag_system import UniversalManufacturingRAG, process_random_document
from core.enhanced_universal_classifier import EnhancedUniversalClassifier

class UniversalRAGTestSimulator:
    """Interactive testing environment for universal RAG capabilities."""
    
    def __init__(self):
        self.implicit_extractor = None
        self.universal_rag = None
        self.enhanced_classifier = None
        self.test_results = []
        self.initialize_systems()
    
    def initialize_systems(self):
        """Initialize all processing systems with error handling."""
        try:
            with st.spinner("🚀 Initializing Universal RAG Systems..."):
                self.implicit_extractor = ImplicitRuleExtractor()
                self.universal_rag = UniversalManufacturingRAG(
                    persist_path="./test_universal_rag_db"
                )
                self.enhanced_classifier = EnhancedUniversalClassifier()
                st.success("✅ All systems initialized successfully!")
        except Exception as e:
            st.error(f"❌ System initialization failed: {str(e)}")
            return False
        return True
    
    def run_simulation(self):
        """Main simulation interface."""
        st.set_page_config(
            page_title="Universal RAG Testing Simulator",
            page_icon="🧪",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🧪 Universal RAG Testing Simulator")
        st.subheader("Test document processing capabilities with vague content and no manufacturing keywords")
        
        # Sidebar configuration
        self.setup_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🧪 Live Testing", 
            "📊 Batch Analysis", 
            "🎯 Challenge Mode", 
            "📈 Performance Metrics",
            "🔍 System Diagnostics"
        ])
        
        with tab1:
            self.live_testing_interface()
        
        with tab2:
            self.batch_analysis_interface()
        
        with tab3:
            self.challenge_mode_interface()
        
        with tab4:
            self.performance_metrics_interface()
        
        with tab5:
            self.system_diagnostics_interface()
    
    def setup_sidebar(self):
        """Setup sidebar controls and configuration."""
        st.sidebar.header("⚙️ Testing Configuration")
        
        # System status
        st.sidebar.subheader("🔧 System Status")
        if self.implicit_extractor and self.universal_rag and self.enhanced_classifier:
            st.sidebar.success("🟢 All Systems Online")
        else:
            st.sidebar.error("🔴 Systems Offline")
            if st.sidebar.button("🔄 Reinitialize Systems"):
                self.initialize_systems()
        
        # Testing parameters
        st.sidebar.subheader("🎛️ Parameters")
        
        self.confidence_threshold = st.sidebar.slider(
            "Confidence Threshold", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.5, 
            step=0.1,
            help="Minimum confidence score for rule extraction"
        )
        
        self.manufacturing_relevance_threshold = st.sidebar.slider(
            "Manufacturing Relevance", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.3, 
            step=0.1,
            help="Minimum manufacturing relevance score"
        )
        
        self.enable_ensemble = st.sidebar.checkbox(
            "Enable Ensemble Classification", 
            value=True,
            help="Use multiple classification methods"
        )
        
        self.include_rag_context = st.sidebar.checkbox(
            "Include RAG Context", 
            value=True,
            help="Use RAG database for enhanced context"
        )
        
        # Quick stats
        if hasattr(st.session_state, 'test_results_history'):
            st.sidebar.subheader("📊 Session Stats")
            total_tests = len(st.session_state.test_results_history)
            avg_rules = np.mean([r['rules_found'] for r in st.session_state.test_results_history]) if total_tests > 0 else 0
            st.sidebar.metric("Tests Run", total_tests)
            st.sidebar.metric("Avg Rules Found", f"{avg_rules:.1f}")
    
    def live_testing_interface(self):
        """Interactive testing interface for real-time analysis."""
        st.header("🧪 Live Document Testing")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📝 Input Document Content")
            
            # Document type selection
            doc_type = st.selectbox(
                "Document Type (for reference)",
                [
                    "Random/Unknown",
                    "Software Requirements",
                    "General Guidelines", 
                    "Project Specifications",
                    "Technical Standards",
                    "Safety Procedures",
                    "Quality Guidelines",
                    "Process Documentation"
                ]
            )
            
            # Text input
            test_content = st.text_area(
                "Enter document content to test:",
                height=200,
                placeholder="""Example vague content:
Items should be arranged properly to avoid issues during operation.
Components must maintain compatibility across different environments.
Surfaces need adequate preparation before applying finishes.
Procedures should be followed consistently to achieve desired outcomes.""",
                help="Enter any type of content - the system will attempt to extract manufacturing rules regardless of keywords"
            )
            
            # Processing options
            col_a, col_b = st.columns(2)
            with col_a:
                process_implicit = st.checkbox("Extract Implicit Rules", value=True)
                process_universal = st.checkbox("Universal RAG Analysis", value=True)
            with col_b:
                process_ensemble = st.checkbox("Ensemble Classification", value=True)
                detailed_analysis = st.checkbox("Detailed Analysis", value=False)
            
            # Process button
            if st.button("🚀 Process Document", type="primary", disabled=not test_content.strip()):
                self.process_live_document(test_content, doc_type, {
                    'implicit': process_implicit,
                    'universal': process_universal,
                    'ensemble': process_ensemble,
                    'detailed': detailed_analysis
                })
        
        with col2:
            st.subheader("📋 Test Scenarios")
            
            # Predefined test scenarios
            scenarios = {
                "Ultra Vague": """Items should work properly in normal conditions. 
Components need adequate spacing for proper function. 
Materials must be suitable for intended applications.""",
                
                "Software-like": """The system should provide adequate response times. 
Error handling must be comprehensive and user-friendly. 
Security measures should protect against unauthorized access.""",
                
                "General Business": """Processes need to be efficient and scalable. 
Quality checkpoints should be implemented throughout. 
Documentation must be maintained for all procedures.""",
                
                "Technical (No Keywords)": """Equipment should operate within specified conditions. 
Maintenance schedules must ensure continuous availability. 
Performance monitoring should detect deviations from normal operation."""
            }
            
            for scenario_name, scenario_content in scenarios.items():
                if st.button(f"📄 Load {scenario_name}", key=f"scenario_{scenario_name}"):
                    st.session_state['test_content'] = scenario_content
                    st.rerun()
            
            # Show sample results if available
            if hasattr(st.session_state, 'latest_test_result'):
                st.subheader("🎯 Latest Result")
                result = st.session_state.latest_test_result
                st.metric("Rules Found", result.get('total_rules', 0))
                st.metric("Confidence", f"{result.get('avg_confidence', 0):.3f}")
                st.metric("Processing Time", f"{result.get('processing_time', 0):.2f}s")
    
    def process_live_document(self, content: str, doc_type: str, options: Dict[str, bool]):
        """Process a document in real-time and display results."""
        start_time = time.time()
        
        with st.spinner("🔄 Processing document..."):
            results = {
                'content': content,
                'doc_type': doc_type,
                'timestamp': time.time(),
                'processing_options': options
            }
            
            try:
                # 1. Implicit Rule Extraction
                if options['implicit'] and self.implicit_extractor:
                    st.write("🔍 **Step 1: Extracting implicit rules...**")
                    implicit_rules = self.implicit_extractor.extract_implicit_rules(
                        content, confidence_threshold=self.confidence_threshold
                    )
                    results['implicit_rules'] = [
                        {
                            'text': rule.text,
                            'confidence': rule.confidence_score,
                            'type': rule.rule_type,
                            'constraint': rule.constraint_type,
                            'mfg_relevance': rule.manufacturing_relevance,
                            'features': rule.semantic_features[:5]
                        }
                        for rule in implicit_rules
                    ]
                    st.success(f"✅ Found {len(implicit_rules)} implicit rules")
                
                # 2. Universal RAG Analysis
                if options['universal'] and self.universal_rag:
                    st.write("🧠 **Step 2: Universal RAG analysis...**")
                    doc_analysis = self.universal_rag.analyze_document_type(content)
                    results['universal_analysis'] = doc_analysis
                    st.success(f"✅ Document analysis complete - Recommended method: {doc_analysis['recommended_method']}")
                
                # 3. Ensemble Classification
                if options['ensemble'] and self.enhanced_classifier:
                    st.write("🎯 **Step 3: Ensemble classification...**")
                    sentences = [s.strip() for s in content.split('.') if s.strip()]
                    rag_context = self.universal_rag if self.include_rag_context else None
                    
                    enhanced_rules = self.enhanced_classifier.classify_with_multiple_approaches(
                        sentences,
                        confidence_threshold=self.confidence_threshold,
                        rag_pipeline=rag_context
                    )
                    results['ensemble_rules'] = enhanced_rules
                    st.success(f"✅ Ensemble classification found {len(enhanced_rules)} potential rules")
                
                # Calculate summary metrics
                total_rules = 0
                total_rules += len(results.get('implicit_rules', []))
                total_rules += len(results.get('ensemble_rules', []))
                
                all_confidences = []
                all_confidences.extend([r['confidence'] for r in results.get('implicit_rules', [])])
                all_confidences.extend([r['confidence'] for r in results.get('ensemble_rules', [])])
                
                results['total_rules'] = total_rules
                results['avg_confidence'] = np.mean(all_confidences) if all_confidences else 0
                results['processing_time'] = time.time() - start_time
                
                # Store results
                if 'test_results_history' not in st.session_state:
                    st.session_state.test_results_history = []
                
                st.session_state.test_results_history.append({
                    'doc_type': doc_type,
                    'rules_found': total_rules,
                    'avg_confidence': results['avg_confidence'],
                    'processing_time': results['processing_time'],
                    'timestamp': results['timestamp']
                })
                
                st.session_state.latest_test_result = results
                
                # Display results
                self.display_processing_results(results, options['detailed'])
                
            except Exception as e:
                st.error(f"❌ Processing failed: {str(e)}")
                results['error'] = str(e)
    
    def display_processing_results(self, results: Dict[str, Any], detailed: bool = False):
        """Display processing results in an organized manner."""
        st.subheader("📊 Processing Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rules Found", results.get('total_rules', 0))
        with col2:
            st.metric("Average Confidence", f"{results.get('avg_confidence', 0):.3f}")
        with col3:
            st.metric("Processing Time", f"{results.get('processing_time', 0):.2f}s")
        with col4:
            mfg_relevance = 0
            if 'universal_analysis' in results:
                mfg_relevance = results['universal_analysis'].get('manufacturing_relevance', 0)
            st.metric("Mfg Relevance", f"{mfg_relevance:.3f}")
        
        # Detailed results tabs
        if results.get('total_rules', 0) > 0:
            res_tab1, res_tab2, res_tab3 = st.tabs(["🔍 Implicit Rules", "🎯 Ensemble Rules", "📈 Analysis"])
            
            with res_tab1:
                if 'implicit_rules' in results and results['implicit_rules']:
                    st.subheader("🔍 Implicit Rule Extraction Results")
                    
                    for i, rule in enumerate(results['implicit_rules'], 1):
                        with st.expander(f"Rule {i}: {rule['text'][:50]}... (Confidence: {rule['confidence']:.3f})"):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.write(f"**Full Text:** {rule['text']}")
                                st.write(f"**Rule Type:** {rule['type']}")
                                st.write(f"**Constraint Type:** {rule['constraint']}")
                            with col_b:
                                st.write(f"**Confidence:** {rule['confidence']:.3f}")
                                st.write(f"**Mfg Relevance:** {rule['mfg_relevance']:.3f}")
                                st.write(f"**Features:** {', '.join(rule['features'])}")
                else:
                    st.info("No implicit rules extracted with current settings")
            
            with res_tab2:
                if 'ensemble_rules' in results and results['ensemble_rules']:
                    st.subheader("🎯 Ensemble Classification Results")
                    
                    for i, rule in enumerate(results['ensemble_rules'], 1):
                        with st.expander(f"Rule {i}: {rule['text'][:50]}... (Score: {rule['confidence']:.3f})"):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.write(f"**Text:** {rule['text']}")
                                st.write(f"**Suggested Type:** {rule['suggested_rule_type']}")
                                st.write(f"**Primary Method:** {rule['primary_method']}")
                            with col_b:
                                st.write(f"**Confidence:** {rule['confidence']:.3f}")
                                st.write(f"**Methods Used:** {', '.join(rule['classification_methods'])}")
                                st.write(f"**Features:** {', '.join(rule['manufacturing_features'][:3])}")
                                
                            if detailed:
                                st.write("**Method Details:**")
                                for method, details in rule['method_details'].items():
                                    st.write(f"- {method}: {details.get('confidence', 0):.3f}")
                else:
                    st.info("No ensemble rules found with current settings")
            
            with res_tab3:
                if 'universal_analysis' in results:
                    st.subheader("📈 Universal RAG Analysis")
                    analysis = results['universal_analysis']
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Keyword Density", f"{analysis['manufacturing_keyword_density']:.3f}")
                        st.metric("Manufacturing Relevance", f"{analysis['manufacturing_relevance']:.3f}")
                    with col_b:
                        st.metric("Estimated Rules", analysis['estimated_rules'])
                        st.write(f"**Recommended Method:** {analysis['recommended_method']}")
                        st.write(f"**Confidence Level:** {analysis['confidence']}")
        else:
            st.warning("⚠️ No rules found. Try adjusting the confidence threshold or using different content.")
    
    def batch_analysis_interface(self):
        """Interface for batch testing multiple documents."""
        st.header("📊 Batch Document Analysis")
        
        # Predefined test suite
        test_suite = {
            "Extremely Vague": [
                "Items should be arranged properly to avoid issues during operation.",
                "Components must maintain compatibility across different environments.", 
                "Surfaces need adequate preparation before applying finishes.",
                "Procedures should be followed consistently to achieve desired outcomes."
            ],
            "Software Requirements": [
                "The system should provide adequate response times for user interactions.",
                "Error handling must be comprehensive and user-friendly.",
                "Security measures should protect against unauthorized access.",
                "Data integrity should be ensured throughout all operations."
            ],
            "General Guidelines": [
                "Products should meet customer expectations for quality and performance.",
                "Materials must be selected considering cost and environmental impact.",
                "Processes need to be efficient and scalable for production volumes.",
                "Safety measures should be implemented at all operational stages."
            ],
            "Mixed Content": [
                "Equipment should operate within specified environmental conditions.",
                "User interfaces need to be accessible and intuitive for all users.", 
                "Quality checkpoints should be implemented throughout the project.",
                "Calibration procedures need to maintain measurement accuracy."
            ]
        }
        
        if st.button("🚀 Run Batch Analysis", type="primary"):
            self.run_batch_analysis(test_suite)
        
        # Display previous batch results if available
        if hasattr(st.session_state, 'batch_results'):
            self.display_batch_results(st.session_state.batch_results)
    
    def run_batch_analysis(self, test_suite: Dict[str, List[str]]):
        """Run batch analysis on multiple document types."""
        batch_results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_docs = sum(len(docs) for docs in test_suite.values())
        processed = 0
        
        for doc_type, documents in test_suite.items():
            for doc_content in documents:
                status_text.text(f"Processing {doc_type}: {doc_content[:50]}...")
                
                # Process with all methods
                result = {
                    'doc_type': doc_type,
                    'content': doc_content,
                    'processing_time': 0
                }
                
                start_time = time.time()
                
                try:
                    # Implicit extraction
                    implicit_rules = self.implicit_extractor.extract_implicit_rules(
                        doc_content, confidence_threshold=self.confidence_threshold
                    )
                    result['implicit_count'] = len(implicit_rules)
                    result['implicit_avg_confidence'] = np.mean([r.confidence_score for r in implicit_rules]) if implicit_rules else 0
                    
                    # Universal analysis
                    analysis = self.universal_rag.analyze_document_type(doc_content)
                    result['manufacturing_relevance'] = analysis['manufacturing_relevance']
                    result['recommended_method'] = analysis['recommended_method']
                    
                    # Ensemble classification
                    sentences = [doc_content]  # Treat as single sentence for this test
                    enhanced_rules = self.enhanced_classifier.classify_with_multiple_approaches(
                        sentences, confidence_threshold=self.confidence_threshold
                    )
                    result['ensemble_count'] = len(enhanced_rules)
                    result['ensemble_avg_confidence'] = np.mean([r['confidence'] for r in enhanced_rules]) if enhanced_rules else 0
                    
                    result['total_rules'] = result['implicit_count'] + result['ensemble_count']
                    result['processing_time'] = time.time() - start_time
                    result['success'] = True
                    
                except Exception as e:
                    result['error'] = str(e)
                    result['success'] = False
                
                batch_results.append(result)
                processed += 1
                progress_bar.progress(processed / total_docs)
        
        status_text.text("✅ Batch analysis complete!")
        st.session_state.batch_results = batch_results
        
        # Clear progress indicators
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
    
    def display_batch_results(self, batch_results: List[Dict[str, Any]]):
        """Display batch analysis results."""
        st.subheader("📊 Batch Analysis Results")
        
        # Success rate
        successful = sum(1 for r in batch_results if r.get('success', False))
        total = len(batch_results)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Success Rate", f"{successful}/{total}")
        with col2:
            avg_rules = np.mean([r.get('total_rules', 0) for r in batch_results])
            st.metric("Avg Rules/Doc", f"{avg_rules:.1f}")
        with col3:
            avg_time = np.mean([r.get('processing_time', 0) for r in batch_results])
            st.metric("Avg Processing Time", f"{avg_time:.2f}s")
        with col4:
            avg_relevance = np.mean([r.get('manufacturing_relevance', 0) for r in batch_results])
            st.metric("Avg Mfg Relevance", f"{avg_relevance:.3f}")
        
        # Results by document type
        df_results = pd.DataFrame(batch_results)
        
        if not df_results.empty:
            # Chart: Rules found by document type
            fig_rules = px.box(
                df_results, 
                x='doc_type', 
                y='total_rules',
                title="Rules Found by Document Type"
            )
            st.plotly_chart(fig_rules, use_container_width=True)
            
            # Chart: Manufacturing relevance by type
            fig_relevance = px.scatter(
                df_results,
                x='manufacturing_relevance',
                y='total_rules',
                color='doc_type',
                title="Manufacturing Relevance vs Rules Found",
                hover_data=['processing_time']
            )
            st.plotly_chart(fig_relevance, use_container_width=True)
            
            # Detailed results table
            st.subheader("📋 Detailed Results")
            display_df = df_results[[
                'doc_type', 'total_rules', 'implicit_count', 'ensemble_count',
                'manufacturing_relevance', 'recommended_method', 'processing_time'
            ]].round(3)
            
            st.dataframe(display_df, use_container_width=True)
    
    def challenge_mode_interface(self):
        """Challenge mode with increasingly difficult test cases."""
        st.header("🎯 Challenge Mode: Ultimate Vagueness Test")
        
        challenges = {
            "Level 1: Slightly Vague": {
                "description": "Documents with some structure but no manufacturing keywords",
                "content": """
                Components should fit together properly without interference.
                Materials must be selected based on application requirements.
                Assembly procedures should ensure reliable connections.
                Quality checks must verify all specifications are met.
                """
            },
            "Level 2: Very Vague": {
                "description": "Generic business language with hidden technical requirements",
                "content": """
                Items need proper arrangement to prevent operational issues.
                Compatibility should be maintained across different conditions.
                Procedures must be followed to achieve consistent results.
                Regular monitoring ensures everything functions as expected.
                """
            },
            "Level 3: Extremely Vague": {
                "description": "Almost no technical content, maximum abstraction",
                "content": """
                Things should work well together in normal situations.
                Proper preparation helps avoid problems later.
                Following guidelines leads to better outcomes.
                Regular attention keeps everything running smoothly.
                """
            },
            "Level 4: Ultra Challenge": {
                "description": "Philosophical/abstract content with hidden manufacturing rules",
                "content": """
                Harmony between elements creates sustainable systems.
                Careful consideration of relationships prevents discord.
                Consistent application of principles yields desired results.
                Continuous observation maintains equilibrium and function.
                """
            }
        }
        
        selected_level = st.selectbox(
            "Select Challenge Level:",
            list(challenges.keys())
        )
        
        challenge = challenges[selected_level]
        
        st.info(f"**{selected_level}**\n{challenge['description']}")
        
        st.text_area(
            "Challenge Content:",
            value=challenge['content'],
            height=150,
            disabled=True
        )
        
        if st.button(f"🎯 Accept Challenge: {selected_level}", type="primary"):
            st.write("🚀 **Processing challenge...**")
            self.process_challenge(selected_level, challenge['content'])
    
    def process_challenge(self, level: str, content: str):
        """Process a challenge and evaluate performance."""
        start_time = time.time()
        
        results = {}
        
        with st.spinner("Processing challenge..."):
            try:
                # All processing methods
                implicit_rules = self.implicit_extractor.extract_implicit_rules(content, confidence_threshold=0.2)
                analysis = self.universal_rag.analyze_document_type(content)
                
                sentences = [s.strip() for s in content.split('.') if s.strip()]
                enhanced_rules = self.enhanced_classifier.classify_with_multiple_approaches(
                    sentences, confidence_threshold=0.2
                )
                
                results = {
                    'level': level,
                    'implicit_rules': len(implicit_rules),
                    'enhanced_rules': len(enhanced_rules),
                    'total_rules': len(implicit_rules) + len(enhanced_rules),
                    'manufacturing_relevance': analysis['manufacturing_relevance'],
                    'recommended_method': analysis['recommended_method'],
                    'processing_time': time.time() - start_time,
                    'confidence_scores': [r.confidence_score for r in implicit_rules] + [r['confidence'] for r in enhanced_rules]
                }
                
                # Evaluate performance
                score = self.calculate_challenge_score(results)
                results['score'] = score
                
                self.display_challenge_results(results, implicit_rules, enhanced_rules)
                
            except Exception as e:
                st.error(f"Challenge failed: {str(e)}")
    
    def calculate_challenge_score(self, results: Dict[str, Any]) -> int:
        """Calculate a score for challenge performance."""
        score = 0
        
        # Base points for finding any rules
        score += results['total_rules'] * 10
        
        # Bonus for high confidence
        if results['confidence_scores']:
            avg_confidence = np.mean(results['confidence_scores'])
            score += int(avg_confidence * 50)
        
        # Bonus for manufacturing relevance
        score += int(results['manufacturing_relevance'] * 30)
        
        # Speed bonus (inverse of processing time)
        if results['processing_time'] > 0:
            speed_bonus = max(0, 20 - int(results['processing_time'] * 5))
            score += speed_bonus
        
        return score
    
    def display_challenge_results(self, results: Dict[str, Any], implicit_rules: List, enhanced_rules: List):
        """Display challenge results with scoring."""
        st.subheader(f"🏆 Challenge Results: {results['level']}")
        
        # Score display
        score = results['score']
        if score >= 80:
            st.success(f"🥇 EXCELLENT! Score: {score}/100")
        elif score >= 60:
            st.info(f"🥈 GOOD! Score: {score}/100")
        elif score >= 40:
            st.warning(f"🥉 FAIR! Score: {score}/100")
        else:
            st.error(f"💪 CHALLENGE! Score: {score}/100 - Try adjusting parameters")
        
        # Detailed metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rules Found", results['total_rules'])
        with col2:
            st.metric("Mfg Relevance", f"{results['manufacturing_relevance']:.3f}")
        with col3:
            avg_conf = np.mean(results['confidence_scores']) if results['confidence_scores'] else 0
            st.metric("Avg Confidence", f"{avg_conf:.3f}")
        with col4:
            st.metric("Processing Time", f"{results['processing_time']:.2f}s")
        
        # Rules breakdown
        if results['total_rules'] > 0:
            st.subheader("📋 Extracted Rules")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.write("**🔍 Implicit Rules:**")
                for i, rule in enumerate(implicit_rules[:3], 1):
                    st.write(f"{i}. {rule.text[:60]}... (Conf: {rule.confidence_score:.3f})")
            
            with col_b:
                st.write("**🎯 Enhanced Rules:**")
                for i, rule in enumerate(enhanced_rules[:3], 1):
                    st.write(f"{i}. {rule['text'][:60]}... (Conf: {rule['confidence']:.3f})")
    
    def performance_metrics_interface(self):
        """Display comprehensive performance metrics."""
        st.header("📈 Performance Metrics & Analytics")
        
        if hasattr(st.session_state, 'test_results_history') and st.session_state.test_results_history:
            results = st.session_state.test_results_history
            df = pd.DataFrame(results)
            
            # Overall statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Tests", len(results))
            with col2:
                st.metric("Avg Rules/Test", f"{df['rules_found'].mean():.1f}")
            with col3:
                st.metric("Avg Confidence", f"{df['avg_confidence'].mean():.3f}")
            with col4:
                st.metric("Avg Processing Time", f"{df['processing_time'].mean():.2f}s")
            
            # Performance over time
            fig_time = px.line(
                df.reset_index(), 
                x='index', 
                y='rules_found',
                title="Rules Found Over Time",
                labels={'index': 'Test Number', 'rules_found': 'Rules Found'}
            )
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Performance by document type
            if 'doc_type' in df.columns:
                fig_type = px.box(
                    df,
                    x='doc_type',
                    y='rules_found', 
                    title="Performance by Document Type"
                )
                st.plotly_chart(fig_type, use_container_width=True)
            
            # Processing time vs rules found
            fig_scatter = px.scatter(
                df,
                x='processing_time',
                y='rules_found',
                color='avg_confidence',
                title="Processing Time vs Rules Found",
                hover_data=['doc_type'] if 'doc_type' in df.columns else None
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        else:
            st.info("Run some tests first to see performance metrics!")
    
    def system_diagnostics_interface(self):
        """System diagnostics and health checks."""
        st.header("🔍 System Diagnostics")
        
        if st.button("🔍 Run System Diagnostics"):
            with st.spinner("Running diagnostics..."):
                self.run_system_diagnostics()
    
    def run_system_diagnostics(self):
        """Run comprehensive system diagnostics."""
        st.subheader("🔧 System Health Check")
        
        # Test each component
        components = {
            "Implicit Rule Extractor": self.implicit_extractor,
            "Universal RAG System": self.universal_rag,
            "Enhanced Classifier": self.enhanced_classifier
        }
        
        for name, component in components.items():
            if component:
                st.success(f"✅ {name}: Online")
            else:
                st.error(f"❌ {name}: Offline")
        
        # Test with sample content
        st.subheader("🧪 Functionality Test")
        
        test_content = "Items should be arranged properly to ensure proper operation."
        
        try:
            # Test implicit extraction
            if self.implicit_extractor:
                rules = self.implicit_extractor.extract_implicit_rules(test_content)
                st.success(f"✅ Implicit Extraction: Found {len(rules)} rules")
            
            # Test universal analysis
            if self.universal_rag:
                analysis = self.universal_rag.analyze_document_type(test_content)
                st.success(f"✅ Universal Analysis: {analysis['recommended_method']}")
            
            # Test classification
            if self.enhanced_classifier:
                enhanced = self.enhanced_classifier.classify_with_multiple_approaches([test_content])
                st.success(f"✅ Enhanced Classification: Found {len(enhanced)} rules")
            
        except Exception as e:
            st.error(f"❌ Functionality test failed: {str(e)}")
        
        # Memory and performance info
        st.subheader("💾 System Resources")
        import psutil
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CPU Usage", f"{psutil.cpu_percent()}%")
        with col2:
            memory = psutil.virtual_memory()
            st.metric("Memory Usage", f"{memory.percent}%")

def main():
    """Main function to run the testing simulator."""
    simulator = UniversalRAGTestSimulator()
    simulator.run_simulation()

if __name__ == "__main__":
    main()