"""
Manufacturing Rules RAG System - Streamlit Application
Advanced RAG system with automated testing and rule generation
"""

import streamlit as st
import pandas as pd
import time
from pathlib import Path
from typing import List

# Import the proper PDF processor
from extractors.text import extract_sentences
from core.rule_generator import StreamlitOptimizedProcessor, StandardizedRule

# Streamlit config
st.set_page_config(
    page_title="Manufacturing Rules RAG System", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize processor
@st.cache_resource
def get_processor():
    """Get cached processor instance."""
    return StreamlitOptimizedProcessor()

# Cache for performance
@st.cache_data
def load_test_docs() -> List[str]:
    """Load test document paths."""
    docs = []
    data_path = Path("/opt/anaconda3/Phase-3-Final-master/data")
    if data_path.exists():
        docs.extend([str(f) for f in data_path.glob("*.pdf")])
    return docs

@st.cache_data
def extract_rules_from_pdf(pdf_path: str, max_rules: int = 50) -> List[dict]:
    """
    Extract rules directly from PDF using proper text extraction.
    Returns list of dicts in HCL format.
    """
    try:
        # Read PDF
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        
        # Extract sentences using proper extractor
        sentences = extract_sentences(pdf_bytes)
        text = " ".join(sentences)
        
        # Use optimized processor
        processor = get_processor()
        rules = processor.extract_rules_fast(text, source_doc=Path(pdf_path).name)
        
        # Convert to dict format and sort by confidence
        rules_sorted = sorted(rules, key=lambda x: x.confidence, reverse=True)
        
        return [rule.to_dict() for rule in rules_sorted[:max_rules]]
        
    except Exception as e:
        st.error(f"Error extracting from {pdf_path}: {e}")
        return []

def main():
    st.title("⚙️ Manufacturing Rules Generator")
    st.markdown("*Optimized for test document processing*")
    
    # Sidebar
    st.sidebar.header("Settings")
    max_rules = st.sidebar.slider("Max Rules", 10, 100, 30)
    min_conf = st.sidebar.slider("Min Confidence", 0.0, 1.0, 0.4)
    
    # Load documents
    docs = load_test_docs()
    st.sidebar.write(f"Found {len(docs)} test documents")
    
    # Document selection
    selected = []
    for doc in docs:
        name = Path(doc).name
        if st.sidebar.checkbox(name):
            selected.append(doc)
    
    # Process button
    if st.button("🚀 Process Documents", type="primary"):
        if not selected:
            st.warning("Select documents first!")
            return
        
        # Processing
        progress = st.progress(0)
        all_rules = []
        
        for i, doc_path in enumerate(selected):
            doc_name = Path(doc_path).name
            st.write(f"Processing: {doc_name}")
            
            # Extract rules directly from PDF
            start = time.time()
            rules = extract_rules_from_pdf(doc_path, max_rules)
            
            # Filter by confidence
            filtered = [r for r in rules if r['confidence'] >= min_conf]
            all_rules.extend(filtered)
            
            # Update progress
            progress.progress((i + 1) / len(selected))
            
            # Show stats
            proc_time = time.time() - start
            st.metric(f"📊 {doc_name}", f"{len(filtered)} rules", f"{proc_time:.1f}s")
        
        # Results
        st.header(f"📈 Results: {len(all_rules)} Total Rules")
        
        if all_rules:
            # Create DataFrame
            df = pd.DataFrame(all_rules)
            
            # Stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rules", len(all_rules))
            with col2:
                high_conf = len([r for r in all_rules if r['confidence'] > 0.7])
                st.metric("High Confidence", high_conf)
            with col3:
                specific_rules = len([r for r in all_rules if r['classification_label'] == 1])
                st.metric("Specific Rules", specific_rules)
            
            # Classification chart
            st.subheader("📊 Rule Classification")
            label_counts = df['classification_label'].value_counts()
            st.bar_chart(label_counts)
            
            # Top rules
            st.subheader("🌟 Top Rules")
            top_rules = sorted(all_rules, key=lambda x: x['confidence'], reverse=True)[:10]
            
            for i, rule in enumerate(top_rules, 1):
                emoji = "✅" if rule['classification_label'] == 1 else "ℹ️"
                st.write(f"**{i}. {emoji}** {rule['rule_text'][:100]}...")
                st.caption(f"Confidence: {rule['confidence']:.2f}")
            
            # Download
            st.subheader("💾 Download")
            
            # HCL format
            hcl_df = df[['rule_text', 'classification_label']].copy()
            csv_data = hcl_df.to_csv(index=False)
            
            st.download_button(
                "📥 Download HCL Format",
                csv_data,
                f"rules_{int(time.time())}.csv",
                "text/csv"
            )
            
            # Full data
            full_csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Full Data",
                full_csv,
                f"full_rules_{int(time.time())}.csv",
                "text/csv"
            )
        
        else:
            st.warning("No rules found. Try lowering confidence threshold.")
    
    # Reference
    with st.expander("📚 HCL Dataset Reference"):
        try:
            hcl_path = "/opt/anaconda3/Phase-3-Final-master/data/hcl classificiation dataset.xlsx"
            hcl_df = pd.read_excel(hcl_path, skiprows=1, header=None)
            hcl_df.columns = ['rule_text', 'classification_label']
            
            st.write(f"**HCL Dataset:** {len(hcl_df)} rules")
            st.write("**Labels:**", hcl_df['classification_label'].value_counts().to_dict())
            
            # Sample
            st.write("**Sample Rules:**")
            sample = hcl_df.sample(min(5, len(hcl_df)))
            for _, row in sample.iterrows():
                emoji = "✅" if row['classification_label'] == 1 else "ℹ️"
                st.write(f"{emoji} {row['rule_text'][:80]}...")
        
        except Exception as e:
            st.error(f"Could not load HCL dataset: {e}")

if __name__ == "__main__":
    main()