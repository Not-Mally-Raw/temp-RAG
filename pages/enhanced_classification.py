"""
Enhanced classification page with RAG integration
"""

import streamlit as st
import os
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from core.rag_pipeline_integration import init_rag_pipeline, display_rag_stats, add_rag_search_interface

def load_model(model_path, tokenizer_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, padding=True, truncation=True, max_length=512, return_tensor="pt")
    return model, tokenizer

def predict(text, tokenizer, classifier_model):
    if not isinstance(text, str):
        text = str(text)
    
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = classifier_model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_label = torch.argmax(probabilities, dim=1).item()
    confidence = torch.max(probabilities).item()
    
    return predicted_label, confidence

def enhanced_classifier_with_rag(pdf_text_data: list[str], model, tokenizer, confidence_threshold, rag_pipeline) -> list:
    """Enhanced classifier that incorporates RAG context."""
    rules_list = []
    
    for line in pdf_text_data:
        if not line.strip():
            continue
            
        label, conf = predict(line, tokenizer, model)
        if label == 1 and conf > confidence_threshold:
            # Get RAG enhancement for this potential rule
            rag_context = rag_pipeline.search_knowledge_base(
                query=line,
                top_k=2
            )
            
            # Extract manufacturing features and constraints
            features = rag_pipeline._extract_manufacturing_features(line)
            constraints = rag_pipeline._identify_constraints(line)
            suggested_type = rag_pipeline._suggest_rule_type(line, rag_context)
            
            enhanced_rule = {
                'text': line,
                'confidence': conf,
                'features': features,
                'constraints': constraints,
                'suggested_type': suggested_type,
                'rag_context': rag_context,
                'similar_rules_count': len(rag_context)
            }
            
            rules_list.append(enhanced_rule)
    
    # Process rules with additional cleaning
    processed_rules = []
    for rule_data in rules_list:
        text = rule_data['text']
        try:
            if "     " in text:
                parts = text.split("     ", 1)
                candidate = parts[0].strip()
                if candidate:
                    label, new_conf = predict(candidate, tokenizer, model)
                    if label == 1 and new_conf > confidence_threshold:
                        rule_data['text'] = candidate
                        rule_data['confidence'] = new_conf
                        processed_rules.append(rule_data)
            elif ".html" in text:
                parts = text.split(".html", 1)
                candidate = parts[1].strip() if len(parts) > 1 else ""
                if candidate:
                    label, new_conf = predict(candidate, tokenizer, model)
                    if label == 1 and new_conf > confidence_threshold:
                        rule_data['text'] = candidate
                        rule_data['confidence'] = new_conf
                        processed_rules.append(rule_data)
            else:
                processed_rules.append(rule_data)
        except Exception as e:
            st.error(f"Error processing rule: {text[:50]}... Error: {str(e)}")
            continue

    # Filter short sentences
    return [rule_data for rule_data in processed_rules if len(rule_data['text'].split()) > 2]

# Initialize RAG pipeline
rag_pipeline = init_rag_pipeline()

st.title("📝 Enhanced PDF Text Classifier & Rule Extractor with RAG")

# Display RAG stats in sidebar
display_rag_stats(rag_pipeline)
add_rag_search_interface(rag_pipeline)

if 'text' in st.session_state and st.session_state['text'] is not None:
    pdf_text_data = list(st.session_state['text'])
    models_dir = "./models/"
    model_names = [name for name in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, name))]

    selected_model = st.selectbox("Select Classification Model", model_names)
    tokenizer_dict = {
        "DeBERTa-v3-small-Binary-Classifier": "microsoft/DeBERTa-v3-small",
        "miniLM-L6-v2-Binary-Classifier": "sentence-transformers/all-MiniLM-L6-v2",
        "TinyBERT_General_4L_213D-Binary-Classifier": "huawei-noah/TinyBERT_General_4L_312D",
        "Qwen3-0.6B-Binary-Classifier": "Qwen/Qwen3-0.6B",
        "ElectraLarge-Binary-Classifier": "google/electra-large-discriminator"
    }

    confidence_threshold = st.slider("Confidence Threshold", 0.5, 1.0, 0.90, 0.01)

    # RAG Enhancement Options
    st.subheader("🧠 RAG Enhancement Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        enable_rag = st.checkbox("Enable RAG Enhancement", value=True, help="Use knowledge base context to improve classification")
    with col2:
        show_rag_details = st.checkbox("Show RAG Context", value=False, help="Display retrieved context for each rule")

    if st.button("🚀 Classify Text with RAG"):
        model_path = os.path.join(models_dir, selected_model)
        model, tokenizer = load_model(model_path, tokenizer_dict[selected_model])
        st.info(f"Model: {selected_model} | Tokenizer: {tokenizer_dict[selected_model]}")
        model.eval()
        
        if enable_rag:
            enhanced_rules = enhanced_classifier_with_rag(pdf_text_data, model, tokenizer, confidence_threshold, rag_pipeline)
            st.success(f"🎯 {len(enhanced_rules)} lines classified as Rules with RAG enhancement.")
            
            if enhanced_rules:
                # Create enhanced DataFrame
                data = []
                for rule_data in enhanced_rules:
                    row = {
                        "rules": rule_data['text'],
                        "confidence": rule_data['confidence'],
                        "suggested_rule_type": rule_data['suggested_type'] or '',
                        "manufacturing_features": ', '.join(rule_data['features'][:3]) if rule_data['features'] else '',
                        "constraints": ', '.join(rule_data['constraints']) if rule_data['constraints'] else '',
                        "similar_rules_found": rule_data['similar_rules_count']
                    }
                    data.append(row)
                
                df = pd.DataFrame(data)
                
                # Display results with enhancement details
                st.subheader("📊 Enhanced Classification Results")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Rules", len(df))
                with col2:
                    auto_typed = df[df['suggested_rule_type'] != ''].shape[0]
                    st.metric("Auto-Typed Rules", auto_typed)
                with col3:
                    avg_confidence = df['confidence'].mean()
                    st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                with col4:
                    with_context = df[df['similar_rules_found'] > 0].shape[0]
                    st.metric("Rules with Context", with_context)
                
                # Rule type distribution
                if auto_typed > 0:
                    st.subheader("📈 Suggested Rule Type Distribution")
                    type_counts = df[df['suggested_rule_type'] != '']['suggested_rule_type'].value_counts()
                    st.bar_chart(type_counts)
                
                # Editable results
                st.subheader("✏️ Review and Edit Results")
                edited_df = st.data_editor(
                    df, 
                    num_rows="dynamic", 
                    use_container_width=True,
                    column_config={
                        "confidence": st.column_config.NumberColumn("Confidence", format="%.3f"),
                        "similar_rules_found": st.column_config.NumberColumn("Similar Rules"),
                        "suggested_rule_type": st.column_config.SelectboxColumn(
                            "Suggested Rule Type",
                            options=["", "Sheet Metal", "Injection Molding", "Assembly", "Machining", "General", "Die Cast", "Additive"]
                        )
                    }
                )
                
                # Show detailed RAG context if requested
                if show_rag_details:
                    st.subheader("🔍 RAG Context Details")
                    
                    for i, rule_data in enumerate(enhanced_rules[:5]):  # Show first 5
                        with st.expander(f"Rule {i+1}: {rule_data['text'][:50]}..."):
                            st.write(f"**Confidence:** {rule_data['confidence']:.3f}")
                            st.write(f"**Features:** {', '.join(rule_data['features'])}")
                            st.write(f"**Constraints:** {', '.join(rule_data['constraints'])}")
                            st.write(f"**Suggested Type:** {rule_data['suggested_type']}")
                            
                            if rule_data['rag_context']:
                                st.write("**Similar Context Found:**")
                                for j, ctx in enumerate(rule_data['rag_context'][:2]):
                                    st.write(f"  {j+1}. {ctx['text'][:100]}... (Score: {ctx['similarity_score']:.3f})")
                
                # Download options
                st.subheader("💾 Export Results")
                
                # Prepare final DataFrame
                final_df = edited_df.copy()
                final_df['rule_type'] = final_df['suggested_rule_type']  # For backward compatibility
                
                st.session_state['rules_df'] = final_df
                
                # Export options
                col1, col2 = st.columns(2)
                
                with col1:
                    excel_file = f"{st.session_state['file_name']}_enhanced_rules.xlsx"
                    final_df.to_excel(excel_file, index=False)
                    with open(excel_file, "rb") as file:
                        st.download_button(
                            "📊 Download Enhanced Excel",
                            data=file.read(),
                            file_name=excel_file,
                            mime="application/vnd.ms-excel"
                        )
                
                with col2:
                    csv_data = final_df.to_csv(index=False)
                    st.download_button(
                        "📄 Download CSV",
                        data=csv_data,
                        file_name=f"{st.session_state['file_name']}_enhanced_rules.csv",
                        mime="text/csv"
                    )
        
        else:
            # Fallback to original classifier
            st.info("RAG enhancement disabled - using standard classification")
            # ... (original classification code would go here)

else:
    st.info("""
    📤 Upload a PDF file first to get started.
    
    **Enhanced Features with RAG:**
    - 🧠 **Context-Aware Classification**: Uses knowledge base to improve accuracy
    - 🏭 **Manufacturing Feature Detection**: Automatically identifies technical features
    - 🔗 **Cross-Document Learning**: Learns from previously processed documents  
    - 🎯 **Intelligent Rule Typing**: Suggests appropriate rule categories
    - 📊 **Constraint Analysis**: Identifies manufacturing constraints and requirements
    
    Navigate to the Upload page to process your documents.
    """)

# Additional RAG insights
if st.session_state.get('rules_df') is not None and not st.session_state['rules_df'].empty:
    st.subheader("🔬 RAG Insights")
    
    df = st.session_state['rules_df']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Top Manufacturing Features:**")
        all_features = []
        for features_str in df['manufacturing_features'].dropna():
            if features_str:
                all_features.extend(features_str.split(', '))
        
        if all_features:
            feature_counts = pd.Series(all_features).value_counts().head(10)
            st.bar_chart(feature_counts)
    
    with col2:
        st.write("**Constraint Types:**")
        all_constraints = []
        for constraints_str in df['constraints'].dropna():
            if constraints_str:
                all_constraints.extend(constraints_str.split(', '))
        
        if all_constraints:
            constraint_counts = pd.Series(all_constraints).value_counts().head(10)
            st.bar_chart(constraint_counts)