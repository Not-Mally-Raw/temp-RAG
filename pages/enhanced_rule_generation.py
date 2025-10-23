"""
Enhanced rule generation with RAG integration
"""

import os
from groq import Groq
from groq.types.chat import ChatCompletion as GroqChatCompletion
import pandas as pd
from tqdm import tqdm
from generators.features import features_dict
import time
from cerebras.cloud.sdk import Cerebras
from cerebras.cloud.sdk.types.chat import ChatCompletion as CerebrasChatCompletion
from typing import Any
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import json

from rag_pipeline_integration import init_rag_pipeline, display_rag_stats, add_rag_search_interface

def _create_enhanced_prompt_with_rag(rule_type: str|None, rule_text: str, rag_context: dict) -> str:
    """Create an enhanced prompt with RAG context for better rule parsing."""
    if not rule_text or not isinstance(rule_text, str):
        raise ValueError("Invalid rule text provided")

    if rule_type is None:
        raise ValueError("Rule type cannot be None. Please provide a valid rule type.")
    if rule_type not in features_dict:
        raise ValueError(f"Invalid type provided: {rule_type}. \nThe possible types are: {features_dict.keys()}")

    # Build RAG context section
    rag_context_section = ""
    if rag_context and rag_context.get('retrieved_context'):
        rag_context_section = f"""
KNOWLEDGE BASE CONTEXT:
The following similar rules and contexts have been found in the knowledge base:

Similar Rules:
{chr(10).join([f"- {rule['text'][:200]}..." for rule in rag_context.get('similar_rules', [])[:3]])}

Related Manufacturing Context:
{chr(10).join([f"- {ctx['text'][:150]}..." for ctx in rag_context.get('retrieved_context', [])[:2]])}

Manufacturing Features Detected: {', '.join(rag_context.get('manufacturing_features', []))}
Constraint Types: {', '.join(rag_context.get('related_constraints', []))}
Standards Referenced: {', '.join(rag_context.get('manufacturing_standards', []))}
"""

    prompt = f"""<s>[INST] You are a precise {rule_type} manufacturing rule parser enhanced with knowledge base context. Your task is to parse the following rule into a specific structured JSON format using both the rule content and the provided context.

Available Features and Parameters for reference:
{features_dict[rule_type]}

{rag_context_section}

You are currently parsing a {rule_type} rule. Use the knowledge base context above to better understand the rule and improve parsing accuracy.

Important parsing rules:
1. Create descriptive names based on the features involved
2. Use exact feature and attribute names mentioned above
3. Use proper operators: >=, <=, ==, between
4. For ranges, use format: min:max
5. For multiple constraints, list them in the Constraints array
6. For distance rules, always use Distance.MinValue or Distance.MaxValue in ExpName depending upon the expression
7. For part edge references, use "Edge" as Object
8. Recom should never be empty.
9. Recom should always contain a numeric value unless and until the rule doesn't have any numeric value. In case there is no numeric value available in the rule, the Recom should be filled with the RHS of the said expression.
10. USE THE KNOWLEDGE BASE CONTEXT to inform your parsing - if similar rules exist, follow their patterns
11. If manufacturing features are detected, ensure they are properly represented in the parsed output

Parse this rule using this JSON format:
{{
    "RuleCategory": "{rule_type}",
    "Name": "[Descriptive name based on the rule]",
    "Feature1": "[Primary feature mentioned]",
    "Feature2": "[Secondary feature if any, else empty string]",
    "Object1": "[Primary object]",
    "Object2": "[Secondary object if any, else empty string]",
    "ExpName": "[Expression name with proper parameter]",
    "Operator": "[One of: >=, <=, ==, between]",
    "Recom": "[Numerical value or range]",
    "Constraint": {{  // Optional, for material/condition constraints
        "ExpName": "[condition]",
        "Operator": "[operator]",
        "Value": "[value]"
    }},
    "Constraints": [  // Optional, for multiple parameter constraints
        {{
            "ExpName": "[expression]",
            "Operator": "[operator]",
            "Recom": "[value]"
        }}
    ],
    "KnowledgeBaseContext": {{ // Enhanced with RAG context
        "SimilarRulesFound": {len(rag_context.get('similar_rules', []))},
        "ContextScore": "{rag_context.get('context_confidence', 0.0):.3f}",
        "DetectedFeatures": {rag_context.get('manufacturing_features', [])},
        "ConstraintTypes": {rag_context.get('related_constraints', [])}
    }}
}}

Reference Examples:

1. Input: "Distance between card guide form and bend should be at least 5.0 times sheet thickness"

Output:
    RuleCategory= Sheet Metal
    Name = Card Guide Form to Bend Distance

    Feature1 = Distance
    Feature2 = ""
    Object1 = CardGuide
    Object2 = Bend

    ExpName = Distance.MinValue/SheetMetal.Thickness
    Operator = >=
    Recom = 5.0

[... more examples ...]

Output the result as a properly formatted JSON object or array of objects if multiple rules are present.

Rule to parse: {rule_text}
[/INST]"""
    return prompt

class EnhancedGroqParser:
    def __init__(self):
        # Initialize Groq client without hardcoded API key
        self.client = Groq(api_key=None)  # Set API key securely elsewhere

    def parse_rule_with_rag(self, rule_type: str, rule_text: str, model, rag_context: dict):
        """Process a rule with RAG context and return the enhanced LLM output."""
        prompt = _create_enhanced_prompt_with_rag(rule_type, rule_text, rag_context)
        response: GroqChatCompletion = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,  # Lower temperature for more consistent parsing
            max_completion_tokens=4096,
            top_p=0.95,
            stream=False,
            response_format={"type": "json_object"},
            reasoning_format="hidden"
        )

        time.sleep(2)
        return response.choices[0].message.content

class EnhancedCerebrasParser:
    def __init__(self):
        cerebras_api_key = os.getenv('CEREBRAS_API_KEY')
        self.client = Cerebras(api_key=cerebras_api_key)

    def parse_rule_with_rag(self, rule_type: str, rule_text: str, model, rag_context: dict):
        """Process a rule with RAG context and return the enhanced LLM output."""
        prompt = _create_enhanced_prompt_with_rag(rule_type, rule_text, rag_context)
        completion: CerebrasChatCompletion|Any = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Rule to parse: {rule_text}"}
            ],
            max_completion_tokens=4096,
            response_format={"type": "json_object"},
            stream=False,
            temperature=0.1,
        )
        time.sleep(6)
        return completion.choices[0].message.content

# Initialize RAG pipeline
rag_pipeline = init_rag_pipeline()

st.title("⚙️ Enhanced Rule Generation with RAG")

# Display RAG stats
display_rag_stats(rag_pipeline)
add_rag_search_interface(rag_pipeline)

# RAG Enhancement Settings
st.subheader("🧠 RAG Enhancement Settings")

col1, col2, col3 = st.columns(3)
with col1:
    enable_rag = st.checkbox("Enable RAG Context", value=True, help="Use knowledge base to enhance rule generation")
with col2:
    rag_top_k = st.slider("Context Results", 1, 10, 3, help="Number of similar rules to retrieve")
with col3:
    show_rag_details = st.checkbox("Show RAG Details", value=False, help="Display retrieved context")

has_data = False
rules_df = None

tab1, tab2 = st.tabs(["Excel Upload", "Classification Import"])

with tab1:
    st.subheader("Process Rules from Excel File")
    rules_file = st.file_uploader("Upload Excel file", type=["xlsx"], key="excel_uploader")
    if rules_file:
        try:
            rules_df = pd.read_excel(rules_file)
            st.success("Excel file loaded!")
            has_data = True
            st.session_state["rules_df"] = rules_df
        except Exception as e:
            st.error(f"Failed to read Excel: {e}")

with tab2:
    st.subheader("Import Classification Rules")
    rules_df = st.session_state.get("rules_df", None)
    if rules_df is not None:
        st.write("Previously uploaded rules loaded.")
        st.dataframe(rules_df.head())
        has_data = True
    else:
        st.info("No rules loaded yet from Excel upload.")

# Only show processing once we have a valid DataFrame
if has_data and isinstance(rules_df, pd.DataFrame) and not rules_df.empty:
    parser_choice = st.selectbox("Select Parser", ["Enhanced Groq with RAG", "Enhanced Cerebras with RAG", "Standard Groq", "Standard Cerebras"])
    model = None

    if "Groq" in parser_choice:
        if "Enhanced" in parser_choice:
            parser = EnhancedGroqParser()
        else:
            from pages.rule_generation import GroqParser
            parser = GroqParser()
        model = st.selectbox("Select Groq Model", [
            "deepseek-r1-distill-llama-70b",
            "qwen-qwq-32b",
            "qwen/qwen3-32b",
            "meta-llama/llama-guard-4-12b",
        ])
    else:
        if "Enhanced" in parser_choice:
            parser = EnhancedCerebrasParser()
        else:
            from pages.rule_generation import CerebrasParser
            parser = CerebrasParser()
        model = st.selectbox("Select Cerebras Model", [
            "llama-4-scout-17b-16e-instruct",
            "qwen-3-32b",
        ])

    # Preview rules before processing
    st.subheader("📋 Rule Preview")
    st.dataframe(rules_df.head(10))

    if st.button("🚀 Process Rules with Enhanced RAG"):
        if not enable_rag and "Enhanced" in parser_choice:
            st.warning("RAG enhancement is disabled but you selected an enhanced parser. Enabling RAG automatically.")
            enable_rag = True
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Performance tracking
        rag_enhanced_count = 0
        total_processing_time = 0
        
        for idx, row in tqdm(rules_df.iterrows(), total=len(rules_df), desc="Processing"):
            rule = str(row.get("rules", ""))
            rule_type = str(row.get("rule_type", ""))
            
            status_text.text(f"Processing rule {idx + 1}/{len(rules_df)}: {rule[:50]}...")
            
            try:
                start_time = time.time()
                
                if enable_rag and "Enhanced" in parser_choice:
                    # Get RAG context for the rule
                    rag_context = rag_pipeline.rag_system.generate_rule_context(
                        rule_text=rule,
                        rule_type=rule_type,
                        top_k=rag_top_k
                    )
                    
                    # Use enhanced parser with RAG context
                    converted = parser.parse_rule_with_rag(rule_type, rule, model, rag_context)
                    rag_enhanced_count += 1
                    
                    # Parse the JSON to add RAG metadata
                    try:
                        parsed_json = json.loads(converted)
                        if isinstance(parsed_json, dict):
                            parsed_json["ProcessingMetadata"] = {
                                "RAGEnhanced": True,
                                "SimilarRulesFound": len(rag_context.get('similar_rules', [])),
                                "ContextQuality": len(rag_context.get('retrieved_context', [])),
                                "ProcessingTime": time.time() - start_time
                            }
                        converted = json.dumps(parsed_json, indent=2)
                    except json.JSONDecodeError:
                        pass  # Keep original if JSON parsing fails
                    
                else:
                    # Use standard parser
                    converted = parser.parse_rule(rule_type, rule, model)
                
                processing_time = time.time() - start_time
                total_processing_time += processing_time
                
            except Exception as e:
                converted = f"Error: {e}"
                processing_time = 0
            
            results.append({
                "rules": rule,
                "rule_type": rule_type,
                "Converted Rule": converted,
                "processing_time": processing_time,
                "rag_enhanced": enable_rag and "Enhanced" in parser_choice
            })
            
            progress_bar.progress((idx + 1) / len(rules_df))
        
        status_text.text("Processing complete!")
        
        output_df = pd.DataFrame(results)
        
        # Display processing statistics
        st.success(f"✅ Processed {len(output_df)} rules successfully!")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rules", len(output_df))
        with col2:
            st.metric("RAG Enhanced", rag_enhanced_count)
        with col3:
            avg_time = total_processing_time / len(output_df) if len(output_df) > 0 else 0
            st.metric("Avg Time/Rule", f"{avg_time:.2f}s")
        with col4:
            errors = output_df[output_df['Converted Rule'].str.startswith('Error:', na=False)].shape[0]
            st.metric("Errors", errors)

        # Show enhanced rules summary
        if rag_enhanced_count > 0:
            st.info(f"""
            🧠 **RAG Enhancement Applied!**
            - {rag_enhanced_count} rules processed with knowledge base context
            - Cross-document learning enabled
            - Manufacturing constraint awareness active
            - Similar rule patterns used for improved accuracy
            """)

        # Display results with RAG context if requested
        if show_rag_details and enable_rag:
            st.subheader("🔍 RAG Context Details (First 3 Rules)")
            
            for idx in range(min(3, len(output_df))):
                row = output_df.iloc[idx]
                if row['rag_enhanced']:
                    with st.expander(f"Rule {idx + 1}: {row['rules'][:50]}..."):
                        st.write(f"**Rule Type:** {row['rule_type']}")
                        st.write(f"**Processing Time:** {row['processing_time']:.2f}s")
                        
                        # Try to parse and display RAG metadata
                        try:
                            parsed = json.loads(row['Converted Rule'])
                            if 'KnowledgeBaseContext' in parsed:
                                st.json(parsed['KnowledgeBaseContext'])
                            if 'ProcessingMetadata' in parsed:
                                st.json(parsed['ProcessingMetadata'])
                        except:
                            st.write("RAG context embedded in converted rule")

        st.subheader("✏️ Preview & Edit Converted Rules")
        display_df = output_df[['rules', 'rule_type', 'Converted Rule']].copy()
        edited = st.data_editor(display_df, num_rows="dynamic", use_container_width=True)

        # Export options
        st.subheader("💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_out = edited.to_csv(index=False)
            st.download_button(
                "📄 Download CSV",
                data=csv_out,
                file_name=f"enhanced_processed_rules_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            excel_out = edited.to_excel(index=False)
            st.download_button(
                "📊 Download Excel",
                data=excel_out,
                file_name=f"enhanced_processed_rules_{time.strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.ms-excel"
            )
        
        with col3:
            # Export with metadata
            full_export = output_df.to_json(orient='records', indent=2)
            st.download_button(
                "🔍 Download Full JSON",
                data=full_export,
                file_name=f"enhanced_rules_with_metadata_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

else:
    st.info("""
    📤 Upload an Excel file with manufacturing rules or import from the Classification page.
    
    **Enhanced RAG Features:**
    - 🧠 **Context-Aware Parsing**: Uses knowledge base to improve rule interpretation
    - 🔗 **Cross-Document Learning**: Learns from previously processed documents
    - 🎯 **Pattern Recognition**: Identifies similar rule structures for consistency
    - 📊 **Manufacturing Intelligence**: Understands technical constraints and requirements
    - ⚡ **Quality Indicators**: Provides confidence scores and context quality metrics
    
    **Expected Excel Format:**
    - Column 'rules': The rule text to parse
    - Column 'rule_type': The manufacturing process category
    """)

# Knowledge base insights
kb_stats = rag_pipeline.get_knowledge_base_summary()
if kb_stats['total_documents'] > 0:
    with st.expander("📊 Knowledge Base Insights"):
        st.write(f"**Documents in KB:** {kb_stats['total_documents']}")
        st.write(f"**Total Chunks:** {kb_stats['total_chunks']}")
        st.write(f"**Last Updated:** {kb_stats['last_updated']}")
        
        if st.button("🔍 Test Knowledge Base Search"):
            test_query = st.text_input("Test Query", "bend radius requirements")
            if test_query:
                results = rag_pipeline.search_knowledge_base(test_query, top_k=3)
                for i, result in enumerate(results, 1):
                    st.write(f"**Result {i}:** {result['text'][:100]}... (Score: {result['similarity_score']:.3f})")