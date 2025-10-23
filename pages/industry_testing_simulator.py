"""
Comprehensive Industry Document Testing Simulator
Tests the RAG system with real-world industry documents and scenarios
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import json
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.enhanced_rag_db import EnhancedManufacturingRAG
from core.implicit_rule_extractor import ImplicitRuleExtractor
from core.universal_rag_system import UniversalManufacturingRAG
from core.enhanced_universal_classifier import EnhancedUniversalClassifier

class IndustryDocumentTester:
    """Comprehensive testing system for industry documents."""
    
    def __init__(self):
        """Initialize all testing systems."""
        self.test_documents = self._load_test_documents()
        self.test_queries = self._load_test_queries()
        self.results_history = []
        
        # Initialize systems
        self.rag_system = None
        self.universal_rag = None
        self.implicit_extractor = None
        self.universal_classifier = None
        
    def _load_test_documents(self) -> Dict[str, Dict[str, Any]]:
        """Load the top 10 industry test documents."""
        return {
            "Siemens_PCB_DFM": {
                "title": "DFM Analysis for Better PCBs E-Book",
                "industry": "Electronics",
                "complexity": "High",
                "sample_content": """
                PCB fabrication guidelines require careful consideration of trace width and spacing.
                Yield optimization depends on proper via sizing and placement strategies.
                Component placement should minimize thermal stress during reflow soldering.
                Design rules must account for manufacturing tolerances and process variations.
                Signal integrity requires controlled impedance and proper layer stackup design.
                Assembly processes need adequate clearances for pick and place operations.
                """,
                "test_queries": [
                    "PCB yield optimization rules",
                    "trace width requirements",
                    "via placement guidelines",
                    "thermal stress reduction"
                ],
                "expected_rules": 15,
                "manufacturing_relevance": 0.95
            },
            
            "Lockheed_Engineering_Requirements": {
                "title": "Engineering Requirements Flowdown Guide",
                "industry": "Aerospace",
                "complexity": "Very High",
                "sample_content": """
                System requirements must be traceable through all levels of design decomposition.
                Performance specifications should include margins for manufacturing variations.
                Interface control documents define mechanical and electrical connection requirements.
                Verification methods must validate compliance with contractual obligations.
                Design constraints should consider production capabilities and limitations.
                Configuration management ensures consistency across development phases.
                """,
                "test_queries": [
                    "embedded manufacturing specs",
                    "system requirements traceability",
                    "interface control requirements",
                    "verification methods"
                ],
                "expected_rules": 12,
                "manufacturing_relevance": 0.85
            },
            
            "Northrop_Quality_Assurance": {
                "title": "Supplier Quality Assurance Requirements",
                "industry": "Aerospace",
                "complexity": "High",
                "sample_content": """
                First article inspection procedures validate initial production capability.
                Supplier processes must demonstrate statistical control and capability.
                Non-conforming material requires immediate containment and disposition.
                Quality documentation should provide complete traceability records.
                Corrective action systems must address root causes effectively.
                Continuous improvement programs ensure ongoing process optimization.
                """,
                "test_queries": [
                    "first article inspection rules",
                    "supplier process control",
                    "quality documentation requirements",
                    "corrective action procedures"
                ],
                "expected_rules": 18,
                "manufacturing_relevance": 0.90
            },
            
            "Boeing_Supply_Chain": {
                "title": "Supply Chain Solutions Guide",
                "industry": "Aerospace",
                "complexity": "Medium",
                "sample_content": """
                Material procurement strategies should balance cost and quality considerations.
                Supplier evaluation criteria include technical capability and delivery performance.
                Inventory management requires optimization of carrying costs and stockouts.
                Logistics planning must coordinate with production scheduling requirements.
                Risk management strategies should address supply chain disruptions.
                Performance metrics must track key supply chain indicators.
                """,
                "test_queries": [
                    "supply chain material rules",
                    "supplier evaluation criteria",
                    "inventory optimization",
                    "logistics coordination"
                ],
                "expected_rules": 10,
                "manufacturing_relevance": 0.75
            },
            
            "3M_Pharmaceutical": {
                "title": "Pharmaceutical Industry Best Practice Guide",
                "industry": "Pharmaceutical",
                "complexity": "Very High",
                "sample_content": """
                Good Manufacturing Practices require validated processes and procedures.
                Contamination control measures must prevent cross-contamination risks.
                Equipment qualification ensures consistent product quality and safety.
                Batch records should document all manufacturing steps and controls.
                Environmental monitoring programs track critical quality parameters.
                Change control procedures must evaluate manufacturing impact assessments.
                """,
                "test_queries": [
                    "exposure banding guidelines",
                    "contamination control measures",
                    "equipment qualification",
                    "batch record requirements"
                ],
                "expected_rules": 20,
                "manufacturing_relevance": 0.92
            },
            
            "NIST_Security": {
                "title": "Systems Security Engineering",
                "industry": "Government/Security",
                "complexity": "Very High",
                "sample_content": """
                Secure hardware design requires protection against physical tampering.
                Cryptographic modules should implement appropriate key management.
                Side-channel analysis resistance must be built into circuit design.
                Supply chain security addresses risks from component sourcing.
                Verification and validation ensure security requirements compliance.
                Life cycle security considerations span development through disposal.
                """,
                "test_queries": [
                    "tamper-proof design rules",
                    "cryptographic module requirements",
                    "side-channel protection",
                    "supply chain security"
                ],
                "expected_rules": 8,
                "manufacturing_relevance": 0.60
            },
            
            "SAE_FMEA": {
                "title": "FMEA Recommended Practice",
                "industry": "Automotive",
                "complexity": "High",
                "sample_content": """
                Failure mode analysis should identify potential product or process failures.
                Risk assessment considers severity, occurrence, and detection ratings.
                Prevention controls aim to reduce failure occurrence probability.
                Detection controls enable early identification of failure conditions.
                Recommended actions should address highest risk priority numbers.
                Review cycles ensure FMEA remains current with design changes.
                """,
                "test_queries": [
                    "failure mode DFM rules",
                    "risk assessment criteria",
                    "prevention controls",
                    "detection methods"
                ],
                "expected_rules": 14,
                "manufacturing_relevance": 0.88
            },
            
            "Intel_Assembly": {
                "title": "Assembly and Test Technology Handbook",
                "industry": "Semiconductor",
                "complexity": "Very High",
                "sample_content": """
                Die attach processes require precise temperature and pressure control.
                Wire bonding parameters must ensure reliable electrical connections.
                Packaging materials should provide adequate protection and thermal management.
                Test coverage must validate all functional and parametric requirements.
                Yield enhancement strategies focus on defect reduction and process optimization.
                Reliability testing ensures long-term performance under operating conditions.
                """,
                "test_queries": [
                    "die attach guidelines",
                    "wire bonding parameters",
                    "packaging requirements",
                    "test coverage validation"
                ],
                "expected_rules": 16,
                "manufacturing_relevance": 0.94
            },
            
            "Caterpillar_Quality": {
                "title": "Supplier Quality Requirements Manual",
                "industry": "Heavy Equipment",
                "complexity": "Medium",
                "sample_content": """
                Welding procedures must be qualified according to applicable standards.
                Material certifications should verify chemical composition and properties.
                Dimensional inspection requirements ensure proper fit and function.
                Surface treatment specifications provide corrosion protection.
                Packaging and shipping methods must prevent damage during transit.
                Documentation requirements support traceability and warranty claims.
                """,
                "test_queries": [
                    "weld imperfection rules",
                    "material certification requirements",
                    "dimensional inspection",
                    "surface treatment specs"
                ],
                "expected_rules": 12,
                "manufacturing_relevance": 0.86
            },
            
            "Autodesk_CNC": {
                "title": "DFM for CNC Machining Guide",
                "industry": "Manufacturing",
                "complexity": "Medium",
                "sample_content": """
                Part design should consider tool accessibility and workholding requirements.
                Tolerance specifications must be achievable with available machining processes.
                Feature geometry affects machining time and tool wear characteristics.
                Material selection influences cutting parameters and surface finish quality.
                Fixturing design ensures part stability during machining operations.
                Programming efficiency depends on proper toolpath optimization strategies.
                """,
                "test_queries": [
                    "tolerance stack-up rules",
                    "tool accessibility requirements",
                    "workholding design",
                    "machining time optimization"
                ],
                "expected_rules": 11,
                "manufacturing_relevance": 0.91
            }
        }
    
    def _load_test_queries(self) -> Dict[str, List[str]]:
        """Load comprehensive test queries for each document type."""
        return {
            "manufacturing_specific": [
                "What are the key manufacturing requirements?",
                "How should quality be ensured during production?",
                "What are the critical process parameters?",
                "How can defects be prevented or minimized?"
            ],
            "vague_general": [
                "What guidelines should be followed?",
                "How can performance be optimized?",
                "What are the important considerations?",
                "How should compliance be achieved?"
            ],
            "cross_domain": [
                "What are the cost optimization strategies?",
                "How can reliability be improved?",
                "What safety measures are recommended?",
                "How should documentation be maintained?"
            ]
        }
    
    def initialize_systems(self) -> bool:
        """Initialize all RAG and classification systems."""
        try:
            st.info("🚀 Initializing RAG systems...")
            
            # Initialize RAG systems
            self.rag_system = EnhancedManufacturingRAG(
                collection_name="industry_test",
                persist_directory="./test_industry_db"
            )
            
            self.universal_rag = UniversalManufacturingRAG(
                persist_path="./test_universal_db"
            )
            
            # Initialize extractors and classifiers
            self.implicit_extractor = ImplicitRuleExtractor()
            self.universal_classifier = EnhancedUniversalClassifier()
            
            st.success("✅ All systems initialized successfully!")
            return True
            
        except Exception as e:
            st.error(f"❌ System initialization failed: {e}")
            return False
    
    def run_comprehensive_test(self, selected_docs: List[str]) -> Dict[str, Any]:
        """Run comprehensive testing on selected documents."""
        
        if not self.rag_system:
            if not self.initialize_systems():
                return {"error": "Failed to initialize systems"}
        
        results = {
            "test_summary": {},
            "document_results": {},
            "query_performance": {},
            "method_comparison": {},
            "processing_stats": {}
        }
        
        total_docs = len(selected_docs)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, doc_key in enumerate(selected_docs):
            doc_info = self.test_documents[doc_key]
            
            status_text.text(f"Testing {doc_info['title']}...")
            
            # Test document processing
            doc_results = self._test_single_document(doc_key, doc_info)
            results["document_results"][doc_key] = doc_results
            
            progress_bar.progress((i + 1) / total_docs)
        
        # Compile summary statistics
        results["test_summary"] = self._compile_test_summary(results["document_results"])
        
        status_text.text("✅ Testing completed!")
        return results
    
    def _test_single_document(self, doc_key: str, doc_info: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single document with all methods."""
        
        content = doc_info["sample_content"]
        
        # Test 1: Implicit Rule Extraction
        implicit_start = time.time()
        implicit_rules = self.implicit_extractor.extract_implicit_rules(
            content, confidence_threshold=0.4
        )
        implicit_time = time.time() - implicit_start
        
        # Test 2: Universal RAG Analysis
        universal_start = time.time()
        doc_analysis = self.universal_rag.analyze_document_type(content)
        universal_time = time.time() - universal_start
        
        # Test 3: Enhanced Classification
        classifier_start = time.time()
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        enhanced_rules = self.universal_classifier.classify_with_multiple_approaches(
            sentences, confidence_threshold=0.4
        )
        classifier_time = time.time() - classifier_start
        
        # Test 4: Query Testing
        query_results = {}
        for query in doc_info["test_queries"]:
            query_start = time.time()
            
            # Test with implicit extraction
            query_implicit = self.implicit_extractor.extract_implicit_rules(
                f"{query} {content}", confidence_threshold=0.3
            )
            
            query_time = time.time() - query_start
            
            query_results[query] = {
                "implicit_matches": len(query_implicit),
                "processing_time": query_time,
                "relevance_score": doc_analysis["manufacturing_relevance"]
            }
        
        return {
            "document_info": doc_info,
            "implicit_extraction": {
                "rules_found": len(implicit_rules),
                "processing_time": implicit_time,
                "top_rules": [
                    {
                        "text": rule.text,
                        "confidence": rule.confidence_score,
                        "type": rule.rule_type
                    }
                    for rule in implicit_rules[:3]
                ]
            },
            "universal_analysis": {
                "manufacturing_relevance": doc_analysis["manufacturing_relevance"],
                "recommended_method": doc_analysis["recommended_method"],
                "confidence_level": doc_analysis["confidence"],
                "processing_time": universal_time
            },
            "enhanced_classification": {
                "rules_found": len(enhanced_rules),
                "processing_time": classifier_time,
                "method_distribution": self._analyze_methods(enhanced_rules)
            },
            "query_testing": query_results
        }
    
    def _analyze_methods(self, enhanced_rules: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze distribution of classification methods used."""
        method_counts = {}
        for rule in enhanced_rules:
            for method in rule.get("classification_methods", []):
                method_counts[method] = method_counts.get(method, 0) + 1
        return method_counts
    
    def _compile_test_summary(self, document_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compile summary statistics from all test results."""
        
        total_docs = len(document_results)
        total_implicit_rules = sum(
            result["implicit_extraction"]["rules_found"] 
            for result in document_results.values()
        )
        total_enhanced_rules = sum(
            result["enhanced_classification"]["rules_found"] 
            for result in document_results.values()
        )
        
        avg_manufacturing_relevance = sum(
            result["universal_analysis"]["manufacturing_relevance"]
            for result in document_results.values()
        ) / total_docs if total_docs > 0 else 0
        
        avg_processing_time = sum(
            result["implicit_extraction"]["processing_time"] +
            result["universal_analysis"]["processing_time"] +
            result["enhanced_classification"]["processing_time"]
            for result in document_results.values()
        ) / total_docs if total_docs > 0 else 0
        
        return {
            "total_documents_tested": total_docs,
            "total_implicit_rules": total_implicit_rules,
            "total_enhanced_rules": total_enhanced_rules,
            "avg_manufacturing_relevance": avg_manufacturing_relevance,
            "avg_processing_time": avg_processing_time,
            "success_rate": 100.0  # All documents that complete processing are successful
        }

def main():
    """Main Streamlit application for industry document testing."""
    
    st.set_page_config(
        page_title="Industry Document Testing Simulator",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧪 Industry Document Testing Simulator")
    st.subtitle("Test RAG System with Real-World Industry Documents")
    
    # Initialize tester
    if 'tester' not in st.session_state:
        st.session_state.tester = IndustryDocumentTester()
    
    tester = st.session_state.tester
    
    # Sidebar controls
    st.sidebar.header("🎛️ Test Configuration")
    
    # Document selection
    st.sidebar.subheader("📋 Select Documents to Test")
    available_docs = list(tester.test_documents.keys())
    
    # Industry filter
    industries = list(set(doc["industry"] for doc in tester.test_documents.values()))
    selected_industries = st.sidebar.multiselect(
        "Filter by Industry",
        industries,
        default=industries
    )
    
    # Complexity filter
    complexities = ["Medium", "High", "Very High"]
    selected_complexities = st.sidebar.multiselect(
        "Filter by Complexity",
        complexities,
        default=complexities
    )
    
    # Filter documents
    filtered_docs = [
        doc_key for doc_key, doc_info in tester.test_documents.items()
        if doc_info["industry"] in selected_industries
        and doc_info["complexity"] in selected_complexities
    ]
    
    selected_docs = st.sidebar.multiselect(
        "Documents to Test",
        filtered_docs,
        default=filtered_docs[:3] if len(filtered_docs) >= 3 else filtered_docs
    )
    
    # Test parameters
    st.sidebar.subheader("⚙️ Test Parameters")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.4)
    test_queries = st.sidebar.checkbox("Include Query Testing", value=True)
    detailed_analysis = st.sidebar.checkbox("Detailed Method Analysis", value=True)
    
    # Main content area
    if not selected_docs:
        st.warning("⚠️ Please select at least one document to test.")
        
        # Show document overview
        st.subheader("📚 Available Test Documents")
        
        doc_overview = []
        for doc_key, doc_info in tester.test_documents.items():
            doc_overview.append({
                "Document": doc_info["title"],
                "Industry": doc_info["industry"],
                "Complexity": doc_info["complexity"],
                "Expected Rules": doc_info["expected_rules"],
                "Mfg Relevance": f"{doc_info['manufacturing_relevance']:.2f}"
            })
        
        st.dataframe(pd.DataFrame(doc_overview), use_container_width=True)
        
    else:
        # Run testing
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(f"🧪 Testing {len(selected_docs)} Documents")
        
        with col2:
            if st.button("🚀 Run Comprehensive Test", type="primary"):
                with st.spinner("Running comprehensive tests..."):
                    results = tester.run_comprehensive_test(selected_docs)
                    st.session_state.test_results = results
        
        # Display results if available
        if 'test_results' in st.session_state:
            results = st.session_state.test_results
            
            # Summary metrics
            st.subheader("📊 Test Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            summary = results["test_summary"]
            
            with col1:
                st.metric("Documents Tested", summary["total_documents_tested"])
            
            with col2:
                st.metric("Implicit Rules Found", summary["total_implicit_rules"])
            
            with col3:
                st.metric("Enhanced Rules Found", summary["total_enhanced_rules"])
            
            with col4:
                st.metric(
                    "Avg Mfg Relevance", 
                    f"{summary['avg_manufacturing_relevance']:.3f}"
                )
            
            # Performance visualization
            st.subheader("📈 Performance Analysis")
            
            # Create performance charts
            doc_names = []
            implicit_counts = []
            enhanced_counts = []
            processing_times = []
            relevance_scores = []
            
            for doc_key, result in results["document_results"].items():
                doc_info = tester.test_documents[doc_key]
                doc_names.append(doc_info["title"][:30] + "...")
                
                implicit_counts.append(result["implicit_extraction"]["rules_found"])
                enhanced_counts.append(result["enhanced_classification"]["rules_found"])
                
                total_time = (
                    result["implicit_extraction"]["processing_time"] +
                    result["universal_analysis"]["processing_time"] +
                    result["enhanced_classification"]["processing_time"]
                )
                processing_times.append(total_time)
                relevance_scores.append(result["universal_analysis"]["manufacturing_relevance"])
            
            # Rules extraction comparison
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = go.Figure(data=[
                    go.Bar(name='Implicit Rules', x=doc_names, y=implicit_counts),
                    go.Bar(name='Enhanced Rules', x=doc_names, y=enhanced_counts)
                ])
                fig1.update_layout(
                    title="Rules Extracted by Method",
                    xaxis_title="Document",
                    yaxis_title="Number of Rules",
                    barmode='group'
                )
                fig1.update_xaxes(tickangle=45)
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = go.Figure(data=[
                    go.Scatter(
                        x=processing_times,
                        y=relevance_scores,
                        mode='markers+text',
                        text=[name[:15] for name in doc_names],
                        textposition="top center",
                        marker=dict(
                            size=10,
                            color=relevance_scores,
                            colorscale='Viridis',
                            colorbar=dict(title="Relevance Score")
                        )
                    )
                ])
                fig2.update_layout(
                    title="Processing Time vs Manufacturing Relevance",
                    xaxis_title="Processing Time (seconds)",
                    yaxis_title="Manufacturing Relevance Score"
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Detailed results
            if detailed_analysis:
                st.subheader("🔍 Detailed Analysis")
                
                for doc_key, result in results["document_results"].items():
                    doc_info = tester.test_documents[doc_key]
                    
                    with st.expander(f"📄 {doc_info['title']} - {doc_info['industry']}"):
                        
                        # Document info
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write(f"**Industry:** {doc_info['industry']}")
                            st.write(f"**Complexity:** {doc_info['complexity']}")
                        
                        with col2:
                            st.write(f"**Expected Rules:** {doc_info['expected_rules']}")
                            st.write(f"**Actual Implicit:** {result['implicit_extraction']['rules_found']}")
                        
                        with col3:
                            st.write(f"**Manufacturing Relevance:** {result['universal_analysis']['manufacturing_relevance']:.3f}")
                            st.write(f"**Recommended Method:** {result['universal_analysis']['recommended_method']}")
                        
                        # Top extracted rules
                        if result["implicit_extraction"]["top_rules"]:
                            st.write("**Top Extracted Rules:**")
                            for i, rule in enumerate(result["implicit_extraction"]["top_rules"], 1):
                                st.write(f"{i}. *{rule['text']}* (Confidence: {rule['confidence']:.3f}, Type: {rule['type']})")
                        
                        # Method distribution
                        if result["enhanced_classification"]["method_distribution"]:
                            st.write("**Classification Method Distribution:**")
                            method_df = pd.DataFrame(
                                list(result["enhanced_classification"]["method_distribution"].items()),
                                columns=["Method", "Count"]
                            )
                            st.bar_chart(method_df.set_index("Method"))
                        
                        # Query testing results
                        if test_queries and result["query_testing"]:
                            st.write("**Query Testing Results:**")
                            query_df = pd.DataFrame([
                                {
                                    "Query": query,
                                    "Matches": data["implicit_matches"],
                                    "Processing Time": f"{data['processing_time']:.3f}s",
                                    "Relevance": f"{data['relevance_score']:.3f}"
                                }
                                for query, data in result["query_testing"].items()
                            ])
                            st.dataframe(query_df, use_container_width=True)
            
            # Export results
            st.subheader("💾 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Export summary as CSV
                summary_df = pd.DataFrame([summary])
                csv_data = summary_df.to_csv(index=False)
                st.download_button(
                    "📄 Download Summary CSV",
                    data=csv_data,
                    file_name="industry_test_summary.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Export detailed results as JSON
                json_data = json.dumps(results, indent=2, default=str)
                st.download_button(
                    "🔍 Download Detailed JSON",
                    data=json_data,
                    file_name="industry_test_detailed.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()