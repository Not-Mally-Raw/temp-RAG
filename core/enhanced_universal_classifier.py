"""
Enhanced Classification for Random Documents
Improves classification accuracy for documents without clear manufacturing keywords
"""

import os
import streamlit as st
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

from universal_rag_system import UniversalManufacturingRAG
from implicit_rule_extractor import ImplicitRuleExtractor

class EnhancedUniversalClassifier:
    """Advanced classifier that handles any type of document content."""
    
    def __init__(self):
        """Initialize multiple classification approaches."""
        
        # Load traditional classifier
        self.traditional_model = None
        self.traditional_tokenizer = None
        
        # Load zero-shot classifier for flexible rule detection
        self.zero_shot_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        
        # Load sentence transformer for semantic similarity
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize implicit rule extractor
        self.implicit_extractor = ImplicitRuleExtractor()
        
        # Rule indicators for any type of content
        self.universal_rule_indicators = [
            "design requirement", "safety guideline", "quality standard",
            "performance specification", "operational procedure", "maintenance instruction",
            "assembly guideline", "material requirement", "dimensional constraint",
            "process requirement", "inspection procedure", "testing standard"
        ]
        
        # Manufacturing rule templates for similarity matching
        self.manufacturing_templates = [
            "Components must meet minimum strength requirements for safe operation",
            "Materials should be selected based on environmental conditions",
            "Assembly procedures require proper alignment and fastening",
            "Quality standards must be maintained throughout production",
            "Safety guidelines ensure worker protection during operations",
            "Dimensional tolerances affect part fit and function",
            "Surface treatments improve corrosion resistance",
            "Testing procedures validate design requirements",
            "Maintenance schedules ensure continued performance",
            "Process parameters control final product quality"
        ]
    
    def load_traditional_model(self, model_path: str, tokenizer_name: str):
        """Load the traditional classification model."""
        self.traditional_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.traditional_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, use_fast=True, padding=True, truncation=True, 
            max_length=512, return_tensor="pt"
        )
        self.traditional_model.eval()
    
    def classify_with_multiple_approaches(
        self, 
        text_sentences: List[str], 
        confidence_threshold: float = 0.7,
        rag_pipeline: Optional[UniversalManufacturingRAG] = None
    ) -> List[Dict[str, Any]]:
        """Classify text using multiple approaches for maximum coverage."""
        
        enhanced_rules = []
        
        for sentence in text_sentences:
            if not sentence.strip() or len(sentence.split()) < 3:
                continue
            
            # Approach 1: Traditional model (if available and text seems manufacturing-related)
            traditional_result = self._classify_traditional(sentence, confidence_threshold)
            
            # Approach 2: Zero-shot classification
            zero_shot_result = self._classify_zero_shot(sentence)
            
            # Approach 3: Semantic similarity to manufacturing rules
            similarity_result = self._classify_similarity(sentence)
            
            # Approach 4: Implicit rule extraction
            implicit_result = self._classify_implicit(sentence)
            
            # Approach 5: RAG-enhanced classification (if available)
            rag_result = None
            if rag_pipeline:
                rag_result = self._classify_with_rag(sentence, rag_pipeline)
            
            # Combine results using ensemble approach
            combined_result = self._combine_classification_results(
                sentence, traditional_result, zero_shot_result, 
                similarity_result, implicit_result, rag_result
            )
            
            if combined_result:
                enhanced_rules.append(combined_result)
        
        return enhanced_rules
    
    def _classify_traditional(self, sentence: str, threshold: float) -> Optional[Dict[str, Any]]:
        """Use traditional model if available and text seems manufacturing-related."""
        
        if not self.traditional_model or not self.traditional_tokenizer:
            return None
        
        try:
            inputs = self.traditional_tokenizer(
                sentence, padding=True, truncation=True, 
                max_length=512, return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = self.traditional_model(**inputs)
            
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_label = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities).item()
            
            if predicted_label == 1 and confidence > threshold:
                return {
                    'method': 'traditional',
                    'confidence': confidence,
                    'is_rule': True
                }
        
        except Exception as e:
            print(f"Traditional classification error: {e}")
        
        return None
    
    def _classify_zero_shot(self, sentence: str) -> Dict[str, Any]:
        """Use zero-shot classification to identify rule-like content."""
        
        try:
            result = self.zero_shot_classifier(
                sentence,
                self.universal_rule_indicators,
                multi_label=False
            )
            
            # Check if the highest scoring label suggests this is a rule
            top_score = result['scores'][0]
            top_label = result['labels'][0]
            
            is_rule = top_score > 0.3 and any(
                keyword in top_label.lower() 
                for keyword in ['requirement', 'guideline', 'standard', 'specification', 'procedure']
            )
            
            return {
                'method': 'zero_shot',
                'confidence': top_score,
                'is_rule': is_rule,
                'category': top_label,
                'all_scores': dict(zip(result['labels'], result['scores']))
            }
        
        except Exception as e:
            print(f"Zero-shot classification error: {e}")
            return {'method': 'zero_shot', 'confidence': 0.0, 'is_rule': False}
    
    def _classify_similarity(self, sentence: str) -> Dict[str, Any]:
        """Use semantic similarity to manufacturing rule templates."""
        
        try:
            # Encode sentence and templates
            sentence_embedding = self.semantic_model.encode([sentence])
            template_embeddings = self.semantic_model.encode(self.manufacturing_templates)
            
            # Calculate similarities
            similarities = np.dot(sentence_embedding, template_embeddings.T).flatten()
            max_similarity = np.max(similarities)
            best_template_idx = np.argmax(similarities)
            
            # Determine if this looks like a manufacturing rule
            is_rule = max_similarity > 0.5
            
            return {
                'method': 'similarity',
                'confidence': float(max_similarity),
                'is_rule': is_rule,
                'best_template': self.manufacturing_templates[best_template_idx],
                'similarity_score': float(max_similarity)
            }
        
        except Exception as e:
            print(f"Similarity classification error: {e}")
            return {'method': 'similarity', 'confidence': 0.0, 'is_rule': False}
    
    def _classify_implicit(self, sentence: str) -> Dict[str, Any]:
        """Use implicit rule extraction to identify rule-like content."""
        
        try:
            rules = self.implicit_extractor.extract_implicit_rules(sentence, confidence_threshold=0.3)
            
            if rules:
                rule = rules[0]  # Take the best one
                return {
                    'method': 'implicit',
                    'confidence': rule.confidence_score,
                    'is_rule': True,
                    'rule_type': rule.rule_type,
                    'constraint_type': rule.constraint_type,
                    'semantic_features': rule.semantic_features[:5],
                    'manufacturing_relevance': rule.manufacturing_relevance
                }
            else:
                return {'method': 'implicit', 'confidence': 0.0, 'is_rule': False}
        
        except Exception as e:
            print(f"Implicit classification error: {e}")
            return {'method': 'implicit', 'confidence': 0.0, 'is_rule': False}
    
    def _classify_with_rag(self, sentence: str, rag_pipeline: UniversalManufacturingRAG) -> Dict[str, Any]:
        """Use RAG system to enhance classification with context."""
        
        try:
            # Search for similar content in knowledge base
            results = rag_pipeline.retrieve_with_fallback(
                query=sentence,
                top_k=3,
                include_implicit=True
            )
            
            if results:
                # Calculate average similarity to known rules
                avg_similarity = np.mean([r['similarity_score'] for r in results])
                
                # Check if any results are high-confidence manufacturing rules
                manufacturing_context = any(
                    r.get('metadata', {}).get('manufacturing_process') 
                    for r in results
                )
                
                is_rule = avg_similarity > 0.7 or (avg_similarity > 0.5 and manufacturing_context)
                
                return {
                    'method': 'rag',
                    'confidence': float(avg_similarity),
                    'is_rule': is_rule,
                    'context_count': len(results),
                    'manufacturing_context': manufacturing_context,
                    'best_match': results[0]['text'][:100] + "..." if results else None
                }
            else:
                return {'method': 'rag', 'confidence': 0.0, 'is_rule': False}
        
        except Exception as e:
            print(f"RAG classification error: {e}")
            return {'method': 'rag', 'confidence': 0.0, 'is_rule': False}
    
    def _combine_classification_results(
        self,
        sentence: str,
        traditional: Optional[Dict[str, Any]],
        zero_shot: Dict[str, Any],
        similarity: Dict[str, Any],
        implicit: Dict[str, Any],
        rag: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Combine results from all classification methods using ensemble approach."""
        
        # Collect all valid methods
        methods = []
        if traditional and traditional['is_rule']:
            methods.append(traditional)
        if zero_shot['is_rule']:
            methods.append(zero_shot)
        if similarity['is_rule']:
            methods.append(similarity)
        if implicit['is_rule']:
            methods.append(implicit)
        if rag and rag['is_rule']:
            methods.append(rag)
        
        # If no method says it's a rule, return None
        if not methods:
            return None
        
        # Calculate ensemble confidence
        method_weights = {
            'traditional': 0.3,
            'zero_shot': 0.2,
            'similarity': 0.2,
            'implicit': 0.2,
            'rag': 0.1
        }
        
        ensemble_confidence = sum(
            method['confidence'] * method_weights.get(method['method'], 0.1)
            for method in methods
        ) / len(methods)
        
        # Determine rule type and features
        suggested_type = self._determine_ensemble_rule_type(methods)
        features = self._extract_ensemble_features(methods)
        constraints = self._extract_ensemble_constraints(methods)
        
        # Determine the primary classification method used
        primary_method = max(methods, key=lambda x: x['confidence'])['method']
        
        return {
            'text': sentence,
            'confidence': ensemble_confidence,
            'suggested_rule_type': suggested_type,
            'manufacturing_features': features,
            'constraints': constraints,
            'classification_methods': [m['method'] for m in methods],
            'primary_method': primary_method,
            'method_details': {m['method']: m for m in methods},
            'ensemble_score': ensemble_confidence,
            'rule_indicators': self._identify_rule_indicators(sentence)
        }
    
    def _determine_ensemble_rule_type(self, methods: List[Dict[str, Any]]) -> str:
        """Determine rule type from ensemble of methods."""
        
        # Priority order for rule type determination
        type_priorities = ['rag', 'implicit', 'zero_shot', 'similarity', 'traditional']
        
        for priority in type_priorities:
            for method in methods:
                if method['method'] == priority:
                    # Extract rule type based on method
                    if priority == 'implicit' and 'rule_type' in method:
                        return method['rule_type']
                    elif priority == 'zero_shot' and 'category' in method:
                        return method['category']
                    elif priority == 'rag' and method.get('manufacturing_context'):
                        return 'Manufacturing Process'
        
        return 'General'
    
    def _extract_ensemble_features(self, methods: List[Dict[str, Any]]) -> List[str]:
        """Extract manufacturing features from ensemble results."""
        
        features = []
        
        for method in methods:
            if method['method'] == 'implicit' and 'semantic_features' in method:
                features.extend(method['semantic_features'])
        
        # Add general features based on text analysis
        for method in methods:
            if method['method'] == 'similarity' and method['confidence'] > 0.6:
                # Infer features from high-similarity manufacturing templates
                features.extend(['quality', 'specification', 'requirement'])
        
        return list(set(features))  # Remove duplicates
    
    def _extract_ensemble_constraints(self, methods: List[Dict[str, Any]]) -> List[str]:
        """Extract constraint types from ensemble results."""
        
        constraints = []
        
        for method in methods:
            if method['method'] == 'implicit' and 'constraint_type' in method:
                constraints.append(method['constraint_type'])
        
        return list(set(constraints))
    
    def _identify_rule_indicators(self, sentence: str) -> List[str]:
        """Identify rule-indicating phrases in the sentence."""
        
        indicators = []
        sentence_lower = sentence.lower()
        
        rule_patterns = [
            'must', 'should', 'shall', 'require', 'need', 'ensure', 'maintain',
            'provide', 'achieve', 'avoid', 'prevent', 'consider', 'design',
            'implement', 'follow', 'comply', 'conform', 'meet', 'exceed'
        ]
        
        for pattern in rule_patterns:
            if pattern in sentence_lower:
                indicators.append(pattern)
        
        return indicators

# Enhanced Streamlit interface function
def enhanced_classification_interface():
    """Streamlit interface for enhanced classification."""
    
    st.title("🎯 Universal Document Classifier")
    st.subtitle("Handles any type of document - with or without manufacturing keywords")
    
    # Initialize systems
    classifier = EnhancedUniversalClassifier()
    rag_pipeline = UniversalManufacturingRAG()
    
    # Configuration options
    st.subheader("⚙️ Classification Settings")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        enable_ensemble = st.checkbox("Enable Ensemble Classification", value=True)
    with col2:
        confidence_threshold = st.slider("Confidence Threshold", 0.3, 0.9, 0.5)
    with col3:
        use_rag_context = st.checkbox("Use RAG Context", value=True)
    
    # Load traditional model if needed
    if st.checkbox("Use Traditional Model (for manufacturing content)"):
        models_dir = "./models/"
        if os.path.exists(models_dir):
            model_names = [name for name in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, name))]
            if model_names:
                selected_model = st.selectbox("Select Model", model_names)
                tokenizer_dict = {
                    "DeBERTa-v3-small-Binary-Classifier": "microsoft/DeBERTa-v3-small",
                    "miniLM-L6-v2-Binary-Classifier": "sentence-transformers/all-MiniLM-L6-v2",
                }
                if selected_model in tokenizer_dict:
                    model_path = os.path.join(models_dir, selected_model)
                    classifier.load_traditional_model(model_path, tokenizer_dict[selected_model])
                    st.success(f"Loaded traditional model: {selected_model}")
    
    # Document input
    if 'text' in st.session_state and st.session_state['text'] is not None:
        pdf_text_data = list(st.session_state['text'])
        
        st.info(f"📄 Processing {len(pdf_text_data)} sentences from uploaded document")
        
        if st.button("🚀 Classify with Universal Approach"):
            with st.spinner("Processing with multiple classification methods..."):
                
                # Classify using ensemble approach
                rag_context = rag_pipeline if use_rag_context else None
                
                enhanced_rules = classifier.classify_with_multiple_approaches(
                    pdf_text_data,
                    confidence_threshold=confidence_threshold,
                    rag_pipeline=rag_context
                )
                
                st.success(f"✅ Found {len(enhanced_rules)} potential rules using ensemble classification")
                
                if enhanced_rules:
                    # Display classification statistics
                    st.subheader("📊 Classification Method Analysis")
                    
                    # Count methods used
                    method_counts = {}
                    for rule in enhanced_rules:
                        for method in rule['classification_methods']:
                            method_counts[method] = method_counts.get(method, 0) + 1
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Rules", len(enhanced_rules))
                    with col2:
                        avg_conf = np.mean([r['confidence'] for r in enhanced_rules])
                        st.metric("Avg Confidence", f"{avg_conf:.3f}")
                    with col3:
                        ensemble_rules = sum(1 for r in enhanced_rules if len(r['classification_methods']) > 1)
                        st.metric("Ensemble Rules", ensemble_rules)
                    with col4:
                        auto_typed = sum(1 for r in enhanced_rules if r['suggested_rule_type'] != 'General')
                        st.metric("Auto-Typed", auto_typed)
                    
                    # Method usage chart
                    if method_counts:
                        st.subheader("🔍 Classification Methods Used")
                        method_df = pd.DataFrame(list(method_counts.items()), columns=['Method', 'Count'])
                        st.bar_chart(method_df.set_index('Method'))
                    
                    # Display results
                    st.subheader("📋 Classification Results")
                    
                    # Prepare data for display
                    display_data = []
                    for rule in enhanced_rules:
                        row = {
                            'text': rule['text'],
                            'confidence': rule['confidence'],
                            'suggested_type': rule['suggested_rule_type'],
                            'primary_method': rule['primary_method'],
                            'methods_used': ', '.join(rule['classification_methods']),
                            'features': ', '.join(rule['manufacturing_features'][:3]),
                            'constraints': ', '.join(rule['constraints']),
                            'rule_indicators': ', '.join(rule['rule_indicators'][:3])
                        }
                        display_data.append(row)
                    
                    df = pd.DataFrame(display_data)
                    
                    # Editable results
                    edited_df = st.data_editor(
                        df,
                        use_container_width=True,
                        column_config={
                            'confidence': st.column_config.NumberColumn("Confidence", format="%.3f"),
                            'suggested_type': st.column_config.SelectboxColumn(
                                "Rule Type",
                                options=["General", "Sheet Metal", "Injection Molding", "Assembly", 
                                        "Machining", "Quality Control", "Safety", "Material"]
                            )
                        }
                    )
                    
                    # Detailed method analysis
                    if st.checkbox("Show Detailed Method Analysis"):
                        st.subheader("🔬 Method-by-Method Analysis")
                        
                        for i, rule in enumerate(enhanced_rules[:3]):  # Show first 3
                            with st.expander(f"Rule {i+1}: {rule['text'][:50]}..."):
                                for method, details in rule['method_details'].items():
                                    st.write(f"**{method.title()} Method:**")
                                    st.json(details)
                    
                    # Export results
                    st.subheader("💾 Export Enhanced Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        csv_data = edited_df.to_csv(index=False)
                        st.download_button(
                            "📄 Download CSV",
                            data=csv_data,
                            file_name="universal_classification_results.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        # Full detailed export
                        import json
                        detailed_export = json.dumps(enhanced_rules, indent=2)
                        st.download_button(
                            "🔍 Download Detailed JSON",
                            data=detailed_export,
                            file_name="detailed_classification_results.json",
                            mime="application/json"
                        )
                    
                    # Store in session state
                    st.session_state['enhanced_rules_df'] = edited_df
    
    else:
        st.info("""
        📤 **Upload a document first to get started**
        
        **Universal Classification Features:**
        - 🎯 **Multiple Classification Methods**: Traditional, zero-shot, similarity, and implicit extraction
        - 🧠 **Ensemble Approach**: Combines results from multiple methods for better accuracy
        - 🔍 **Semantic Analysis**: Works with any content, not just manufacturing keywords
        - 📊 **Confidence Scoring**: Provides detailed confidence metrics for each method
        - 🏭 **Manufacturing Intelligence**: Maintains specialized manufacturing knowledge
        - 🔗 **RAG Integration**: Uses knowledge base context when available
        
        **Handles Documents Like:**
        - General technical specifications
        - Software requirements 
        - Safety procedures
        - Quality guidelines
        - Process documentation
        - Design standards
        """)

if __name__ == "__main__":
    enhanced_classification_interface()