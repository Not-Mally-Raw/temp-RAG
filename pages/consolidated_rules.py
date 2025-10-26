"""
Consolidated Rules Display - Shows all extracted rules from test results
"""

import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import json

st.set_page_config(
    page_title="📊 Consolidated Rules Database",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_all_rules_csvs(test_results_dir: str = "./test_results") -> pd.DataFrame:
    """Load and consolidate all rules CSV files from test results."""
    all_rules = []

    if not os.path.exists(test_results_dir):
        st.error(f"Test results directory not found: {test_results_dir}")
        return pd.DataFrame()

    # Load individual rule files
    for file in os.listdir(test_results_dir):
        if file.endswith('_rules.csv'):
            file_path = os.path.join(test_results_dir, file)
            try:
                df = pd.read_csv(file_path)
                df['source_csv'] = file
                all_rules.append(df)
            except Exception as e:
                st.warning(f"Could not load {file}: {e}")

    # Load consolidated file if it exists
    consolidated_file = os.path.join(test_results_dir, "consolidated_all_rules.csv")
    if os.path.exists(consolidated_file):
        try:
            df = pd.read_csv(consolidated_file)
            df['source_csv'] = 'consolidated_all_rules.csv'
            all_rules.append(df)
        except Exception as e:
            st.warning(f"Could not load consolidated file: {e}")

    if not all_rules:
        return pd.DataFrame()

    # Combine all dataframes
    combined_df = pd.concat(all_rules, ignore_index=True)

    # Clean and standardize columns
    column_mapping = {
        'rule_text': 'rule_text',
        'source_file': 'source_file',
        'source_citation': 'source_citation',
        'manufacturing_process': 'manufacturing_process',
        'rule_type': 'rule_type',
        'confidence': 'confidence',
        'extraction_method': 'extraction_method',
        'timestamp': 'timestamp',
        'query_question': 'query_question',
        'query_process': 'query_process'
    }

    # Rename columns if they exist
    for old_col, new_col in column_mapping.items():
        if old_col in combined_df.columns and old_col != new_col:
            combined_df = combined_df.rename(columns={old_col: new_col})

    # Ensure required columns exist
    required_cols = ['rule_text', 'source_file', 'confidence']
    for col in required_cols:
        if col not in combined_df.columns:
            combined_df[col] = 'N/A'

    # Clean data
    combined_df['confidence'] = pd.to_numeric(combined_df['confidence'], errors='coerce').fillna(0.5)
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], errors='coerce')

    # Remove duplicates based on rule_text and source_file
    combined_df = combined_df.drop_duplicates(subset=['rule_text', 'source_file'], keep='first')

    return combined_df

def create_rules_summary_stats(df: pd.DataFrame) -> dict:
    """Create summary statistics for the rules database."""
    if df.empty:
        return {}

    stats = {
        'total_rules': len(df),
        'unique_sources': df['source_file'].nunique(),
        'avg_confidence': df['confidence'].mean(),
        'high_confidence_rules': len(df[df['confidence'] > 0.8]),
        'manufacturing_processes': df['manufacturing_process'].value_counts().to_dict() if 'manufacturing_process' in df.columns else {},
        'extraction_methods': df['extraction_method'].value_counts().to_dict() if 'extraction_method' in df.columns else {},
        'rule_types': df['rule_type'].value_counts().to_dict() if 'rule_type' in df.columns else {}
    }

    return stats

def main():
    st.title("📊 Consolidated Rules Database")
    st.markdown("*Complete collection of all extracted manufacturing rules from test results*")

    # Load data
    with st.spinner("Loading consolidated rules database..."):
        rules_df = load_all_rules_csvs()

    if rules_df.empty:
        st.error("❌ No rules data found. Please run the automated testing first.")
        return

    # Sidebar filters
    st.sidebar.header("🔍 Filters")

    # Confidence filter
    min_conf = st.sidebar.slider(
        "Minimum Confidence",
        0.0, 1.0,
        0.0,
        step=0.1
    )

    # Source file filter
    if 'source_file' in rules_df.columns:
        source_files = sorted(rules_df['source_file'].unique())
        selected_sources = st.sidebar.multiselect(
            "Source Files",
            source_files,
            default=source_files[:3] if len(source_files) > 3 else source_files
        )
    else:
        selected_sources = []

    # Manufacturing process filter
    if 'manufacturing_process' in rules_df.columns:
        processes = sorted([p for p in rules_df['manufacturing_process'].unique() if pd.notna(p)])
        selected_processes = st.sidebar.multiselect(
            "Manufacturing Processes",
            processes,
            default=processes[:5] if len(processes) > 5 else processes
        )
    else:
        selected_processes = []

    # Apply filters
    filtered_df = rules_df.copy()

    if min_conf > 0:
        filtered_df = filtered_df[filtered_df['confidence'] >= min_conf]

    if selected_sources:
        filtered_df = filtered_df[filtered_df['source_file'].isin(selected_sources)]

    if selected_processes:
        filtered_df = filtered_df[filtered_df['manufacturing_process'].isin(selected_processes)]

    # Main content
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Rules", len(filtered_df))

    with col2:
        st.metric("Source Documents", filtered_df['source_file'].nunique() if 'source_file' in filtered_df.columns else 0)

    with col3:
        st.metric("Avg Confidence", f"{filtered_df['confidence'].mean():.2f}")

    with col4:
        high_conf = len(filtered_df[filtered_df['confidence'] > 0.8])
        st.metric("High Confidence", high_conf)

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Rules Table", "📊 Analytics", "🔍 Search & Filter", "💾 Export"
    ])

    with tab1:
        st.header("📋 Extracted Rules Database")

        # Display options
        col1, col2 = st.columns([3, 1])
        with col1:
            sort_by = st.selectbox(
                "Sort by:",
                ["confidence", "source_file", "timestamp"] if "timestamp" in filtered_df.columns else ["confidence", "source_file"],
                index=0
            )
        with col2:
            ascending = st.checkbox("Ascending", value=False)

        # Sort dataframe
        if sort_by in filtered_df.columns:
            filtered_df = filtered_df.sort_values(sort_by, ascending=ascending)

        # Display dataframe
        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=600,
            column_config={
                "rule_text": st.column_config.TextColumn("Rule Text", width="large"),
                "confidence": st.column_config.NumberColumn("Confidence", format="%.3f"),
                "source_file": st.column_config.TextColumn("Source File", width="medium"),
                "manufacturing_process": st.column_config.TextColumn("Process", width="medium"),
                "extraction_method": st.column_config.TextColumn("Method", width="medium")
            }
        )

    with tab2:
        st.header("📊 Rules Analytics")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 Confidence Distribution")
            fig = px.histogram(
                filtered_df,
                x="confidence",
                nbins=20,
                title="Rule Confidence Scores"
            )
            fig.update_layout(xaxis_title="Confidence Score", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("📄 Rules by Source")
            if 'source_file' in filtered_df.columns:
                source_counts = filtered_df['source_file'].value_counts()
                fig = px.bar(
                    x=source_counts.index,
                    y=source_counts.values,
                    title="Rules Extracted by Document"
                )
                fig.update_layout(xaxis_title="Source Document", yaxis_title="Rule Count")
                st.plotly_chart(fig, use_container_width=True)

        # Additional analytics
        if 'manufacturing_process' in filtered_df.columns:
            st.subheader("🏭 Manufacturing Process Distribution")
            process_counts = filtered_df['manufacturing_process'].value_counts()
            if not process_counts.empty:
                fig = px.pie(
                    values=process_counts.values,
                    names=process_counts.index,
                    title="Rules by Manufacturing Process"
                )
                st.plotly_chart(fig, use_container_width=True)

        if 'extraction_method' in filtered_df.columns:
            st.subheader("🔧 Extraction Methods")
            method_counts = filtered_df['extraction_method'].value_counts()
            if not method_counts.empty:
                fig = px.bar(
                    x=method_counts.index,
                    y=method_counts.values,
                    title="Rules by Extraction Method"
                )
                fig.update_layout(xaxis_title="Extraction Method", yaxis_title="Rule Count")
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("🔍 Search & Filter Rules")

        # Search functionality
        search_term = st.text_input("🔍 Search rules containing:", placeholder="Enter keywords...")

        if search_term:
            search_results = filtered_df[
                filtered_df['rule_text'].str.contains(search_term, case=False, na=False)
            ]
            st.write(f"Found {len(search_results)} rules matching '{search_term}'")

            if not search_results.empty:
                st.dataframe(
                    search_results[['rule_text', 'confidence', 'source_file']],
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("No rules found matching your search term.")
        else:
            st.info("Enter a search term above to find specific rules.")

        # Advanced filters
        with st.expander("🔧 Advanced Filters"):
            col1, col2 = st.columns(2)

            with col1:
                if 'rule_type' in filtered_df.columns:
                    rule_types = sorted(filtered_df['rule_type'].dropna().unique())
                    selected_types = st.multiselect("Rule Types", rule_types)
                    if selected_types:
                        filtered_df = filtered_df[filtered_df['rule_type'].isin(selected_types)]

            with col2:
                if 'query_question' in filtered_df.columns:
                    questions = sorted(filtered_df['query_question'].dropna().unique())
                    selected_questions = st.multiselect("Query Questions", questions)
                    if selected_questions:
                        filtered_df = filtered_df[filtered_df['query_question'].isin(selected_questions)]

    with tab4:
        st.header("💾 Export Options")

        st.subheader("📄 Download Filtered Results")

        # CSV export
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            "📥 Download as CSV",
            data=csv_data,
            file_name=f"consolidated_rules_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            type="primary"
        )

        # JSON export
        json_data = filtered_df.to_json(orient="records", indent=2)
        st.download_button(
            "📥 Download as JSON",
            data=json_data,
            file_name=f"consolidated_rules_{datetime.now().strftime('%Y%m%d_%H%M%M%S')}.json",
            mime="application/json"
        )

        # Summary export
        summary_stats = create_rules_summary_stats(filtered_df)
        if summary_stats:
            summary_json = json.dumps(summary_stats, indent=2)
            st.download_button(
                "📊 Download Summary Stats",
                data=summary_json,
                file_name=f"rules_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

        # Display summary
        st.subheader("📊 Dataset Summary")
        if summary_stats:
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Total Rules", summary_stats.get('total_rules', 0))
                st.metric("Unique Sources", summary_stats.get('unique_sources', 0))
                st.metric("High Confidence Rules", summary_stats.get('high_confidence_rules', 0))

            with col2:
                st.metric("Average Confidence", f"{summary_stats.get('avg_confidence', 0):.3f}")

                if summary_stats.get('manufacturing_processes'):
                    top_process = max(summary_stats['manufacturing_processes'].items(), key=lambda x: x[1])
                    st.metric("Top Process", f"{top_process[0]} ({top_process[1]})")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p><strong>📊 Consolidated Rules Database</strong></p>
    <p>Complete collection of manufacturing rules extracted by the Enhanced RAG System</p>
    <p>Generated from automated testing on DFM Handbook and industry documents</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()