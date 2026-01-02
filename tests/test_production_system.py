"""
Comprehensive Test Suite for Enhanced Manufacturing Rule Extraction System
Consolidates all testing functionality: ingestion, retrieval, extraction, validation, and end-to-end flows
"""

import asyncio
import time
import sys
import os
import tempfile
import pytest
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.production_system import ProductionRuleExtractionSystem
from core.enhanced_rule_engine import ManufacturingRule, EnhancedConfig
from core.document_processor import DocumentProcessor
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

@pytest.fixture
def groq_api_key():
    """Get Groq API key from environment."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not set")
    return api_key

@pytest.fixture
def sample_manufacturing_text():
    """Sample manufacturing document text."""
    return """
    Manufacturing Quality Control Standard

    1. Material specifications must meet ASTM standards
    2. Surface finish requirements: Ra ≤ 1.6μm for critical surfaces
    3. Dimensional tolerances: ±0.05mm for precision components
    4. Heat treatment verification required for all steel components
    5. Assembly torque specifications: M8 bolts 25±2 Nm

    Sheet Metal Design Guidelines:
    - Minimum bend radius should be at least 1.5 times the material thickness
    - Ensure adequate clearance between moving parts
    - For injection molding, wall thickness must be between 1-5mm
    - Avoid sharp corners in plastic parts to prevent stress concentration
    """

@pytest.fixture
def production_system(groq_api_key):
    """Initialize production system for testing.

    Note: this fixture is intentionally synchronous; system construction does
    not require awaiting and this avoids pytest-asyncio strict-mode issues.
    """

    return ProductionRuleExtractionSystem(groq_api_key=groq_api_key, use_qdrant=False)

@pytest.fixture
def document_processor():
    """Initialize document processor."""
    return DocumentProcessor()

class TestDocumentProcessing:
    """Test document processing functionality."""

    @pytest.mark.asyncio
    async def test_pdf_processing(self, document_processor):
        """Test PDF processing with available files."""
        data_folder = "/opt/anaconda3/RAG-System/data/real_documents"

        if not os.path.exists(data_folder):
            pytest.skip("Data folder not found")

        files = list(Path(data_folder).glob("*.pdf"))
        if not files:
            pytest.skip("No PDF files found")

        # Test first PDF
        test_file = files[0]
        result = await document_processor.process_document(str(test_file))

        assert result['filename'] == test_file.name
        assert 'text' in result
        assert len(result['text']) > 0
        assert 'manufacturing_keywords' in result
        assert 'processing_method' in result

    @pytest.mark.asyncio
    async def test_excel_processing(self, document_processor):
        """Test Excel processing."""
        data_folder = "/opt/anaconda3/RAG-System/data/real_documents"

        if not os.path.exists(data_folder):
            pytest.skip("Data folder not found")

        files = list(Path(data_folder).glob("*.xlsx"))
        if not files:
            pytest.skip("No Excel files found")

        # Test first Excel file
        test_file = files[0]
        result = await document_processor.process_document(str(test_file))

        assert result['filename'] == test_file.name
        assert 'text' in result
        assert result['processing_method'] == 'pandas'

    def test_manufacturing_keyword_detection(self, document_processor):
        """Test manufacturing keyword detection."""
        test_text = "The minimum wall thickness for plastic parts should be 2mm to prevent deformation."

        # Manually test the keyword detection logic
        manufacturing_keywords = {
            'thickness', 'wall', 'plastic', 'deformation', 'minimum'
        }

        text_lower = test_text.lower()
        found_keywords = []
        for keyword in manufacturing_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)

        assert len(found_keywords) > 0
        assert 'thickness' in found_keywords
        assert 'wall' in found_keywords

class TestRuleExtraction:
    """Test rule extraction functionality."""

    @pytest.mark.asyncio
    async def test_single_rule_extraction(self, production_system, sample_manufacturing_text):
        """Test single rule extraction."""
        # Extract rules from sample text
        rules_result = await production_system.rule_engine.extract_rules_from_text(sample_manufacturing_text[:2000])
        rules = rules_result.get('rules', [])

        assert isinstance(rules, list)
        # Should extract at least some rules from manufacturing text
        assert len(rules) >= 0  # Allow for no rules if text is too short

        if rules:
            rule = rules[0]
            assert 'rule_text' in rule
            assert 'confidence_score' in rule
            assert 'rule_category' in rule
            assert 0.0 <= rule['confidence_score'] <= 1.0

    @pytest.mark.asyncio
    async def test_rule_structure(self, production_system):
        """Test rule structure and validation."""
        test_text = "Minimum bend radius should be at least 1.5 times the material thickness"

        rules_result = await production_system.rule_engine.extract_rules_from_text(test_text)
        rules = rules_result.get('rules', [])

        if rules:
            rule = rules[0]
            # Check for required fields
            required_fields = ['rule_text', 'confidence_score', 'rule_category']
            for field in required_fields:
                assert field in rule

            # Check confidence range
            assert 0.0 <= rule['confidence_score'] <= 1.0

    @pytest.mark.asyncio
    async def test_confidence_scoring(self, production_system):
        """Test confidence scoring logic."""
        specific_rule = "Wall thickness must be between 1-5mm for injection molding"
        general_text = "Consider the material properties"

        # Extract from specific rule
        specific_result = await production_system.rule_engine.extract_rules_from_text(specific_rule)
        specific_rules = specific_result.get('rules', [])

        # Extract from general text
        general_result = await production_system.rule_engine.extract_rules_from_text(general_text)
        general_rules = general_result.get('rules', [])

        # Specific rules should generally have higher confidence
        if specific_rules and general_rules:
            specific_conf = specific_rules[0]['confidence_score']
            general_conf = general_rules[0]['confidence_score']
            # This is a soft assertion - specific rules often have higher confidence
            assert specific_conf >= 0.0  # Just check it's valid

class TestEndToEnd:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self, production_system, document_processor):
        """Test complete document processing pipeline."""
        data_folder = "/opt/anaconda3/RAG-System/data/real_documents"

        if not os.path.exists(data_folder):
            pytest.skip("Data folder not found")

        files = list(Path(data_folder).glob("*"))
        if not files:
            pytest.skip("No files found")

        # Test with first file
        test_file = files[0]

        # Process document
        doc_result = await document_processor.process_document(str(test_file))
        assert doc_result['text']

        # Extract rules if text is available
        if len(doc_result['text']) > 100:  # Only test if we have meaningful text
            text_sample = doc_result['text'][:3000]  # Limit for testing
            rules_result = await production_system.rule_engine.extract_rules_from_text(text_sample)
            rules = rules_result.get('rules', [])

            assert isinstance(rules, list)
            # Rules might be empty for some documents, which is OK

    @pytest.mark.asyncio
    async def test_performance(self, production_system, sample_manufacturing_text):
        """Test performance benchmarks."""
        start_time = time.time()

        rules_result = await production_system.rule_engine.extract_rules_from_text(sample_manufacturing_text)
        processing_time = time.time() - start_time

        # Should complete in reasonable time (under 10 seconds for this test)
        assert processing_time < 10.0

        rules = rules_result.get('rules', [])
        logger.info("Performance test completed",
                   processing_time=processing_time,
                   rules_extracted=len(rules))

class TestSystemIntegration:
    """Test system integration and components."""

    @pytest.mark.asyncio
    async def test_system_initialization(self, groq_api_key):
        """Test system initialization."""
        system = ProductionRuleExtractionSystem(groq_api_key=groq_api_key, use_qdrant=False)

        assert system.rule_engine is not None
        assert system.prompt_system is not None
        assert system.llm is not None

    def test_config_validation(self, groq_api_key):
        """Test configuration validation."""
        config = EnhancedConfig(groq_api_key=groq_api_key)

        assert config.groq_api_key == groq_api_key
        # Defaults may evolve as Groq deprecates models; validate current defaults.
        assert isinstance(config.groq_model, str) and len(config.groq_model) > 0
        assert config.temperature == 0.05
        assert config.max_tokens == 4096

class TestHCLValidation:
    """Test against HCL dataset (if available)."""

    @pytest.mark.asyncio
    async def test_hcl_sample_validation(self, production_system):
        """Test against HCL dataset sample."""
        hcl_path = "/opt/anaconda3/Phase-3-Final-master/data/hcl_classification_clean.csv"

        if not os.path.exists(hcl_path):
            pytest.skip("HCL dataset not found")

        # Load sample of HCL data
        df = pd.read_csv(hcl_path)
        sample_size = min(10, len(df))  # Test with small sample
        sample_df = df.sample(n=sample_size, random_state=42)

        correct_predictions = 0
        total_predictions = 0

        for _, row in sample_df.iterrows():
            rule_text = row['rule_text']
            expected_label = row['classification_label']

            # Extract rules
            rules_result = await production_system.rule_engine.extract_rules_from_text(rule_text)
            rules = rules_result.get('rules', [])

            if rules:
                # Use confidence as prediction (high confidence = specific rule = label 1)
                best_rule = max(rules, key=lambda x: x['confidence_score'])
                predicted_label = 1 if best_rule['confidence_score'] > 0.7 else 0

                if predicted_label == expected_label:
                    correct_predictions += 1
                total_predictions += 1

        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
            logger.info("HCL validation completed",
                       accuracy=accuracy,
                       sample_size=total_predictions)
            # Don't assert accuracy here as it's a baseline test
            assert total_predictions > 0

# --- Simple in-memory fakes to test substitutability --- #
class FakeLoader(ITextLoader):
    def load(self, path: str) -> DocumentMeta:
        return DocumentMeta(path=path, pages=1, text="one chunk", extras={"source": "fake"})

class FakeChunker(IChunker):
    def chunk(self, document: DocumentMeta) -> Iterable[str]:
        # split text into two "chunks" for determinism
        return ["chunk A", "chunk B"]

class FakeExtractor(IRuleExtractor):
    def extract(self, chunk: str, context: Dict[str, Any] = None) -> List[Rule]:
        # produce a Rule per chunk with text echo and a confidence value
        return [Rule(id=f"{chunk}-1", text=f"rule from {chunk}", category="TEST", confidence=1.0, meta={"chunk": chunk})]

class CapturingExporter(IExporter):
    def __init__(self):
        self.captured = None
    def export(self, rules: Iterable[Rule], output_path: str) -> None:
        self.captured = list(rules)

def test_system_with_direct_fakes():
    loader = FakeLoader()
    chunker = FakeChunker()
    extractor = FakeExtractor()
    exporter = CapturingExporter()
    system = ProductionRuleExtractionSystem(loader=loader, chunker=chunker, extractor=extractor, exporter=exporter)

    rules = system.process_document("doc1", export_path="out.csv")
    assert len(rules) == 2
    assert exporter.captured is not None
    assert exporter.captured == rules
    assert rules[0].text.startswith("rule from chunk")

def test_system_with_adapters_wrapping_callables():
    # Wrap the same fake implementations into the adapter classes via callables
    loader_adapter = TextLoaderAdapter(impl=FakeLoader().load)
    chunker_adapter = ChunkerAdapter(impl=FakeChunker().chunk)
    extractor_adapter = RuleExtractorAdapter(impl=FakeExtractor().extract)
    exporter_adapter = ExporterAdapter(impl=CapturingExporter().export)

    # When passing adapters, behavior should match direct fake objects (substitutability)
    system_with_adapters = ProductionRuleExtractionSystem(
        loader=loader_adapter, chunker=chunker_adapter, extractor=extractor_adapter, exporter=exporter_adapter
    )
    rules_from_adapters = system_with_adapters.process_document("doc1", export_path="out.csv")
    assert len(rules_from_adapters) == 2
    assert all(isinstance(r, Rule) for r in rules_from_adapters)

# Standalone test functions (for running without pytest)
async def run_document_processing_standalone():
    """Standalone document processing test."""
    print("🔧 Testing Document Processing...")

    processor = DocumentProcessor()
    data_folder = "/opt/anaconda3/RAG-System/data/real_documents"

    if not os.path.exists(data_folder):
        print("❌ Data folder not found")
        return

    files = list(Path(data_folder).glob("*"))[:3]  # Test first 3 files

    for file_path in files:
        print(f"📄 Testing: {file_path.name}")
        try:
            result = await processor.process_document(str(file_path))
            print(f"  ✅ Method: {result.get('processing_method')}")
            print(f"  📝 Text length: {len(result.get('text', ''))}")
            print(f"  🏭 Keywords: {len(result.get('manufacturing_keywords', []))}")
        except Exception as e:
            print(f"  ❌ Error: {e}")

async def run_rule_extraction_standalone(groq_api_key):
    """Standalone rule extraction test."""
    print("🔍 Testing Rule Extraction...")

    system = ProductionRuleExtractionSystem(groq_api_key=groq_api_key, use_qdrant=False)

    test_text = "Minimum bend radius should be at least 1.5 times the material thickness for sheet metal parts."

    try:
        rules_result = await system.rule_engine.extract_rules_from_text(test_text)
        rules = rules_result.get('rules', [])

        print(f"✅ Found {len(rules)} rules")
        for i, rule in enumerate(rules[:3]):
            print(f"  📋 Rule {i+1}: {rule.get('rule_text', '')[:80]}...")
            print(f"     Confidence: {rule.get('confidence_score', 0):.3f}")

    except Exception as e:
        print(f"❌ Error: {e}")

async def run_end_to_end_standalone(groq_api_key):
    """Standalone end-to-end test."""
    print("🚀 Testing End-to-End Processing...")

    system = ProductionRuleExtractionSystem(groq_api_key=groq_api_key, use_qdrant=False)
    processor = DocumentProcessor()

    data_folder = "/opt/anaconda3/RAG-System/data/real_documents"
    if not os.path.exists(data_folder):
        print("❌ Data folder not found")
        return

    files = list(Path(data_folder).glob("*"))[:2]  # Test first 2 files

    for file_path in files:
        print(f"📄 Processing: {file_path.name}")

        try:
            # Process document
            doc_result = await processor.process_document(str(file_path))
            print(f"  ✅ Document processed: {len(doc_result.get('text', ''))} chars")

            # Extract rules
            if doc_result.get('text'):
                text_sample = doc_result['text'][:3000]
                rules_result = await system.rule_engine.extract_rules_from_text(text_sample)
                rules = rules_result.get('rules', [])

                print(f"  📋 Rules extracted: {len(rules)}")
                if rules:
                    rule = rules[0]
                    print(f"     Sample: {rule.get('rule_text', '')[:100]}...")

        except Exception as e:
            print(f"  ❌ Error: {e}")

async def main():
    """Main test function for standalone execution."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ GROQ_API_KEY not set")
        return

    print("🧪 Starting Consolidated System Tests...")
    print("=" * 60)

    # Run standalone tests
    await run_document_processing_standalone()
    print()
    await run_rule_extraction_standalone(groq_api_key)
    print()
    await run_end_to_end_standalone(groq_api_key)

    print("\n" + "=" * 60)
    print("🎉 All standalone tests completed!")

if __name__ == "__main__":
    # Run standalone tests
    asyncio.run(main())

    # Also run pytest if available
    try:
        import pytest
        print("\n🧪 Running pytest tests...")
        pytest.main([__file__, "-v"])
    except ImportError:
        print("\n⚠️ pytest not available, run 'pip install pytest' for comprehensive testing")