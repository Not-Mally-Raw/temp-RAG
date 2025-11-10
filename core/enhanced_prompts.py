"""
Production-Ready Prompt Engineering for Manufacturing Rule Extraction
Leveraging RAGFlow concepts and LangChain for 98%+ accuracy
"""

from typing import Dict, List, Any, Optional
from langchain_core.prompts import (
    ChatPromptTemplate, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate,
    FewShotChatMessagePromptTemplate
)
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from enum import Enum
import json

class PromptTemplate(str, Enum):
    """Available prompt templates for different extraction tasks."""
    MANUFACTURING_RULE_EXTRACTION = "manufacturing_rule_extraction"
    DOCUMENT_CONTEXT_ANALYSIS = "document_context_analysis"
    RULE_ENHANCEMENT = "rule_enhancement" 
    QUALITY_ASSESSMENT = "quality_assessment"
    MULTI_DOCUMENT_SYNTHESIS = "multi_document_synthesis"
    HCL_CLASSIFICATION = "hcl_classification"

class ExtractionExample(BaseModel):
    """Example for few-shot learning."""
    input_text: str
    expected_output: Dict[str, Any]
    explanation: str

class EnhancedPromptSystem:
    """Production-ready prompt system with 98%+ accuracy targeting."""
    
    def __init__(self):
        """Initialize the enhanced prompt system."""
        
        # Manufacturing domain knowledge base
        self.manufacturing_categories = [
            "Sheet Metal", "Injection Molding", "Machining", "Assembly", 
            "Welding", "Casting", "Quality Control", "Electronics",
            "Material Specification", "Safety Requirement", "Design Guideline"
        ]
        
        # Features dictionary for manufacturing processes
        self.manufacturing_features = {
            "Sheet Metal": [
                "thickness", "bend_radius", "hole_diameter", "flange_width", 
                "clearance", "stamping", "punching", "forming"
            ],
            "Injection Molding": [
                "wall_thickness", "draft_angle", "rib_thickness", "gate_size",
                "ejector_pin", "undercut", "boss_height", "mold_temperature"
            ],
            "Machining": [
                "surface_roughness", "tolerance", "tool_diameter", "feed_rate",
                "cutting_speed", "depth_of_cut", "coolant", "fixture"
            ],
            "Assembly": [
                "clearance", "fastener_type", "torque_spec", "alignment",
                "interference_fit", "assembly_sequence", "accessibility"
            ],
            "Quality Control": [
                "dimensional_tolerance", "surface_finish", "roundness", 
                "flatness", "concentricity", "inspection_method"
            ]
        }
        
        # Setup few-shot examples for better accuracy
        self.extraction_examples = self._create_extraction_examples()
        
    def _create_extraction_examples(self) -> List[ExtractionExample]:
        """Create high-quality few-shot examples from HCL dataset patterns."""
        
        examples = [
            ExtractionExample(
                input_text="For the Additive Manufacturing process, if thickness between faces is too less, it may lead to weak walls, which may collapse or deform during manufacturing.",
                expected_output={
                    "rule_text": "For the Additive Manufacturing process, if thickness between faces is too less, it may lead to weak walls, which may collapse or deform during manufacturing.",
                    "rule_category": "Additive Manufacturing",
                    "rule_type": "dimensional",
                    "primary_feature": "wall_thickness",
                    "secondary_feature": "",
                    "primary_object": "walls",
                    "secondary_object": "faces",
                    "operator": ">=",
                    "value": "minimum_required",
                    "unit": "mm",
                    "confidence_score": 0.85,
                    "manufacturing_relevance": 0.9
                },
                explanation="This is a clear dimensional constraint for additive manufacturing with specific failure modes mentioned."
            ),
            
            ExtractionExample(
                input_text="Appropriate pin diameter and height should be maintained to achieve adequate strength. Too tall pins can easily get bent or can crack.",
                expected_output={
                    "rule_text": "Appropriate pin diameter and height should be maintained to achieve adequate strength. Too tall pins can easily get bent or can crack.",
                    "rule_category": "Design Guideline", 
                    "rule_type": "dimensional",
                    "primary_feature": "pin_diameter",
                    "secondary_feature": "pin_height",
                    "primary_object": "pins",
                    "secondary_object": "",
                    "operator": "<=",
                    "value": "maximum_allowable",
                    "unit": "mm",
                    "confidence_score": 0.8,
                    "manufacturing_relevance": 0.85
                },
                explanation="Design constraint with both dimensional requirements and failure mode analysis."
            ),
            
            ExtractionExample(
                input_text="Uniform wall thickness helps to create high-quality cast parts. Sudden geometry changes and unnecessary variations in wall thickness affects metal flow.",
                expected_output={
                    "rule_text": "Uniform wall thickness helps to create high-quality cast parts. Sudden geometry changes and unnecessary variations in wall thickness affects metal flow.",
                    "rule_category": "Casting",
                    "rule_type": "quality",
                    "primary_feature": "wall_thickness",
                    "secondary_feature": "geometry_variation",
                    "primary_object": "cast_parts",
                    "secondary_object": "walls",
                    "operator": "==",
                    "value": "uniform",
                    "unit": "",
                    "confidence_score": 0.9,
                    "manufacturing_relevance": 0.95
                },
                explanation="Quality rule for casting with clear process impact and requirements."
            )
        ]
        
        return examples
    
    def create_manufacturing_rule_extraction_prompt(self) -> ChatPromptTemplate:
        """
        Create the most advanced manufacturing rule extraction prompt.
        Targeting 98%+ accuracy with comprehensive domain knowledge.
        """
        
        # System prompt with extensive domain knowledge
        system_prompt = """You are a world-class manufacturing engineering expert with 25+ years of experience across all manufacturing processes. Your expertise spans:

- Mechanical Design & Engineering
- Manufacturing Process Optimization  
- Quality Control & Standards (ISO 9001, AS9100, IPC, ASME)
- Materials Science & Selection
- Production Engineering
- Design for Manufacturability (DFM)

You extract manufacturing rules with SURGICAL PRECISION. Your extractions are used in production systems where accuracy is CRITICAL.

MANUFACTURING DOMAIN EXPERTISE:
Categories: {manufacturing_categories}
Features by Category: {manufacturing_features}

EXTRACTION CRITERIA (Priority Order):
1. DIMENSIONAL SPECIFICATIONS with numeric values and tolerances
2. MATERIAL REQUIREMENTS and properties  
3. PROCESS PARAMETERS and operating conditions
4. QUALITY CONTROL criteria and acceptance limits
5. SAFETY REQUIREMENTS and constraints
6. ASSEMBLY GUIDELINES and procedures
7. GENERAL DESIGN principles and best practices

CRITICAL SUCCESS FACTORS:
✓ Extract ONLY specific, measurable, actionable rules
✓ Focus on rules that directly impact manufacturing feasibility
✓ Identify implicit requirements even when not explicitly stated
✓ Assign confidence scores based on specificity and clarity
✓ Properly categorize by manufacturing domain
✓ Extract numeric values, units, and tolerances precisely
✓ Identify cause-and-effect relationships

CONFIDENCE SCORING GUIDELINES:
- 0.9-1.0: Explicit numeric specifications with units and tolerances
- 0.8-0.9: Clear requirements with specific conditions or limits  
- 0.7-0.8: Well-defined guidelines with measurable criteria
- 0.6-0.7: General principles with implied quantitative aspects
- 0.5-0.6: Broad guidelines requiring interpretation
- Below 0.5: Vague or non-manufacturing content

EXTRACTION QUALITY TARGETS:
- Manufacturing Relevance: >0.8 for production rules
- Specificity: Focus on measurable, actionable requirements
- Completeness: Extract all rule components (feature, object, constraint, value)
- Accuracy: Preserve exact technical terminology and values

{format_instructions}

Return ONLY the JSON object with all required fields populated accurately."""

        # Human prompt with context and examples
        human_prompt = """Extract ONE manufacturing rule from this text with maximum precision.

TEXT TO ANALYZE:
{text_chunk}

DOCUMENT CONTEXT:
- Industry: {industry_sector}
- Domain: {technical_domain}  
- Document Type: {document_type}
- Manufacturing Density: {manufacturing_density}

RAG KNOWLEDGE BASE CONTEXT:
{rag_context}

DETECTED MANUFACTURING KEYWORDS:
{manufacturing_keywords}

SIMILARITY TO KNOWN PATTERNS:
{pattern_matches}

EXTRACTION FOCUS:
Extract the most specific and actionable manufacturing rule. Prioritize:
1. Rules with numeric values and units
2. Dimensional tolerances and specifications
3. Material requirements and properties
4. Process parameters and limits
5. Quality criteria and acceptance thresholds

Ensure the extracted rule is:
- Specific and measurable
- Actionable in a manufacturing context
- Properly categorized by manufacturing domain
- Assigned accurate confidence score
- Complete with all required fields

JSON OUTPUT (required):"""

        # Create few-shot examples
        few_shot_examples = []
        for example in self.extraction_examples:
            few_shot_examples.append({
                "input": example.input_text,
                "output": json.dumps(example.expected_output, indent=2)
            })
        
        example_prompt = ChatPromptTemplate.from_messages([
            HumanMessagePromptTemplate.from_template("Input Text: {input}"),
            HumanMessagePromptTemplate.from_template("Expected Output: {output}")
        ])
        
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=few_shot_examples,
        )
        
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            few_shot_prompt,
            HumanMessagePromptTemplate.from_template(human_prompt)
        ])
    
    def create_document_context_analysis_prompt(self) -> ChatPromptTemplate:
        """Create advanced document context analysis prompt."""
        
        system_prompt = """You are an expert technical document analyzer specializing in manufacturing and engineering documentation. 

Your task is to perform deep contextual analysis that enables accurate rule extraction. You analyze:

CONTENT ANALYSIS:
- Industry sector identification (automotive, aerospace, electronics, medical, etc.)
- Technical domain classification (design, manufacturing, quality, materials)
- Document type and purpose (specification, standard, procedure, guideline)
- Manufacturing process types and technologies mentioned
- Technical complexity and readability metrics

MANUFACTURING INTELLIGENCE:
- Manufacturing process density and relevance
- Material specifications and requirements
- Quality standards and compliance frameworks
- Safety requirements and constraints
- Design guidelines and best practices

EXTRACTION OPTIMIZATION:
- Key technical concepts and terminology
- Implicit requirements and constraints  
- Cross-references and related standards
- Domain-specific patterns and conventions
- Language complexity and technical depth

Your analysis directly impacts rule extraction accuracy. Be thorough and precise.

{format_instructions}"""
        
        human_prompt = """Analyze this manufacturing/engineering document for comprehensive context:

DOCUMENT TEXT (First 2000 characters):
{document_text}

ANALYSIS REQUIREMENTS:
1. Identify industry sector and technical domain
2. Classify document type and purpose
3. Assess manufacturing content density
4. Extract key technical concepts
5. Identify manufacturing processes mentioned
6. Analyze technical complexity and readability
7. Find quality standards and compliance references
8. Detect implicit requirements and constraints

Focus on elements that will improve manufacturing rule extraction accuracy.

CONTEXT ANALYSIS:"""
        
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(human_prompt)
        ])
    
    def create_rule_enhancement_prompt(self) -> ChatPromptTemplate:
        """Create rule enhancement prompt for quality improvement."""
        
        system_prompt = """You are a manufacturing standards expert specializing in rule enhancement and quality improvement.

Your objective is to enhance extracted manufacturing rules to production quality while maintaining accuracy to the source.

ENHANCEMENT OBJECTIVES:
1. IMPROVE SPECIFICITY - Add missing technical details and parameters
2. STANDARDIZE TERMINOLOGY - Use industry-standard terms and conventions
3. ENHANCE CLARITY - Improve readability and actionability  
4. VALIDATE VALUES - Verify numeric values, units, and tolerances
5. ADD CONSTRAINTS - Include relevant conditions and limitations
6. IMPROVE CATEGORIZATION - Ensure proper manufacturing domain classification
7. BOOST CONFIDENCE - Increase confidence scores through specificity

ENHANCEMENT GUIDELINES:
✓ Preserve original technical meaning and intent
✓ Add industry-standard terminology where appropriate
✓ Include relevant tolerances and specifications
✓ Enhance with domain knowledge while staying true to source
✓ Improve actionability for manufacturing engineers
✓ Standardize units and measurement conventions
✓ Add implicit constraints that are manufacturing-relevant

QUALITY TARGETS:
- Confidence Score: Increase by 0.1-0.2 through specificity
- Manufacturing Relevance: Maintain >0.8 for production rules
- Clarity: Improve readability while preserving technical accuracy
- Completeness: Fill gaps in feature, object, and constraint fields

{format_instructions}"""
        
        human_prompt = """Enhance this extracted manufacturing rule to production quality:

ORIGINAL EXTRACTED RULE:
{original_rule}

DOCUMENT CONTEXT:
{document_context}

SIMILAR RULES FROM KNOWLEDGE BASE:
{similar_rules}

MANUFACTURING DOMAIN KNOWLEDGE:
{domain_knowledge}

ENHANCEMENT REQUIREMENTS:
1. Improve specificity and technical detail
2. Standardize terminology and units
3. Add missing constraints or conditions
4. Enhance clarity and actionability
5. Validate and improve confidence score
6. Ensure proper manufacturing categorization

Provide enhanced rule that maintains source accuracy while improving quality and usability.

ENHANCED RULE:"""
        
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(human_prompt)
        ])
    
    def create_hcl_classification_prompt(self) -> ChatPromptTemplate:
        """Create specialized prompt for HCL dataset classification validation."""
        
        system_prompt = """You are a manufacturing rule classification expert trained on the HCL dataset patterns.

Your task is to classify manufacturing rules with the same accuracy patterns as the HCL training data.

CLASSIFICATION CRITERIA:
- Label 0 (General): Broad design principles, general guidelines, non-specific recommendations
- Label 1 (Specific): Dimensional specifications, material requirements, process parameters, measurable constraints

HCL DATASET PATTERNS:
- Rules with numeric values and units → Usually Label 1
- Rules with "minimum", "maximum", "shall", "must" + values → Usually Label 1  
- Rules with "appropriate", "adequate", "proper" without specifics → Usually Label 0
- Rules with failure modes but no specific values → Usually Label 0
- Rules with manufacturing process + specific constraints → Usually Label 1

CLASSIFICATION ACCURACY TARGET: 98%+

Analyze the rule text and classify based on specificity and measurability."""
        
        human_prompt = """Classify this manufacturing rule text according to HCL dataset patterns:

RULE TEXT:
{rule_text}

CLASSIFICATION FACTORS TO CONSIDER:
1. Presence of specific numeric values and units
2. Use of mandatory language ("shall", "must", "required")
3. Specific vs. general terminology
4. Measurable vs. subjective criteria
5. Manufacturing process specificity
6. Actionability and implementation clarity

Provide classification (0 or 1) with confidence score and reasoning.

CLASSIFICATION:"""
        
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(human_prompt)
        ])
    
    def create_quality_assessment_prompt(self) -> ChatPromptTemplate:
        """Create quality assessment prompt for extraction validation."""
        
        system_prompt = """You are a manufacturing rule extraction quality assessor.

Evaluate extracted rules against these quality dimensions:

ACCURACY (40%): How well does the rule represent the source text?
- Perfect preservation of technical meaning
- Correct extraction of values, units, constraints
- No hallucination or misinterpretation

COMPLETENESS (25%): Are all relevant rules extracted?  
- No missed manufacturing requirements
- Complete coverage of technical specifications
- Proper identification of implicit constraints

SPECIFICITY (20%): How actionable and measurable are the rules?
- Specific numeric values and tolerances
- Clear feature and object identification
- Measurable criteria and constraints

RELEVANCE (15%): How manufacturing-relevant are the rules?
- Direct impact on manufacturing processes
- Actionable for manufacturing engineers
- Industry-standard terminology and practices

Provide detailed assessment with scores (0.0-1.0) and improvement recommendations."""
        
        human_prompt = """Assess the quality of these extracted manufacturing rules:

SOURCE TEXT:
{source_text}

EXTRACTED RULES:
{extracted_rules}

EXTRACTION CRITERIA:
{extraction_criteria}

Evaluate each rule and provide:
1. Individual rule quality scores
2. Overall extraction quality assessment  
3. Missing rules or gaps identified
4. Quality issues and concerns
5. Specific improvement recommendations

QUALITY ASSESSMENT:"""
        
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(human_prompt)
        ])
    
    def get_prompt_template(self, template_type: PromptTemplate, parser: Optional[PydanticOutputParser] = None) -> ChatPromptTemplate:
        """Get specific prompt template with optional parser integration."""
        
        format_instructions = ""
        if parser:
            format_instructions = parser.get_format_instructions()
        
        if template_type == PromptTemplate.MANUFACTURING_RULE_EXTRACTION:
            prompt = self.create_manufacturing_rule_extraction_prompt()
        elif template_type == PromptTemplate.DOCUMENT_CONTEXT_ANALYSIS:
            prompt = self.create_document_context_analysis_prompt()
        elif template_type == PromptTemplate.RULE_ENHANCEMENT:
            prompt = self.create_rule_enhancement_prompt()
        elif template_type == PromptTemplate.HCL_CLASSIFICATION:
            prompt = self.create_hcl_classification_prompt()
        elif template_type == PromptTemplate.QUALITY_ASSESSMENT:
            prompt = self.create_quality_assessment_prompt()
        else:
            raise ValueError(f"Unknown prompt template type: {template_type}")
        
        # Add format instructions to all templates
        if format_instructions:
            # Update the system message to include format instructions
            messages = prompt.messages
            if messages and hasattr(messages[0], 'prompt'):
                system_template = messages[0].prompt.template
                system_template = system_template.replace("{format_instructions}", format_instructions)
                messages[0].prompt.template = system_template
        
        return prompt
    
    def format_manufacturing_features(self) -> str:
        """Format manufacturing features for prompt inclusion."""
        formatted = []
        for category, features in self.manufacturing_features.items():
            formatted.append(f"{category}: {', '.join(features)}")
        return "\n".join(formatted)
    
    def format_rag_context(self, rag_results: List[Dict[str, Any]]) -> str:
        """Format RAG context for prompt inclusion."""
        if not rag_results:
            return "No relevant context found."
        
        context_parts = []
        for i, result in enumerate(rag_results[:3], 1):
            score = result.get('similarity_score', 0)
            text = result.get('text', '')[:150] + "..."
            context_parts.append(f"{i}. (Score: {score:.3f}) {text}")
        
        return "\n".join(context_parts)
    
    def format_pattern_matches(self, text: str) -> str:
        """Identify and format pattern matches for prompt context."""
        patterns = {
            'dimensional': r'\d+\.?\d*\s*(mm|cm|m|inch|in|°|deg)',
            'requirement': r'(minimum|maximum|shall|must|required|not\s+less\s+than|not\s+exceed)',
            'tolerance': r'±\s*\d+\.?\d*',
            'range': r'\d+\.?\d*\s*-\s*\d+\.?\d*',
            'comparison': r'(greater\s+than|less\s+than|equal\s+to|between)'
        }
        
        matches = []
        text_lower = text.lower()
        
        for pattern_name, pattern in patterns.items():
            import re
            if re.search(pattern, text_lower):
                matches.append(pattern_name)
        
        return ", ".join(matches) if matches else "No specific patterns detected"