import pytest
import sys
import os
from unittest.mock import Mock, patch

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.enhanced_rag_db import EnhancedManufacturingRAG, DocumentMetadata
from core.implicit_rule_extractor import ImplicitRuleExtractor

class TestEnhancedRAGDB:
    """Test suite for Enhanced RAG Database functionality"""
    
    @pytest.fixture
    def rag_system(self):
        """Create a test RAG system instance"""
        return EnhancedManufacturingRAG(
            collection_name="test_manufacturing",
            persist_directory="./test_chroma_db"
        )
    
    @pytest.fixture
    def sample_document_metadata(self):
        """Create sample document metadata"""
        return DocumentMetadata(
            source="test_document.txt",
            document_type="procedure",
            manufacturing_domain="quality_control",
            processing_date="2024-01-01",
            industry="automotive"
        )
    
    def test_initialization(self, rag_system):
        """Test RAG system initialization"""
        assert rag_system is not None
        assert rag_system.collection_name == "test_manufacturing"
        assert rag_system.embedding_model is not None
    
    def test_document_metadata_creation(self, sample_document_metadata):
        """Test document metadata creation"""
        assert sample_document_metadata.source == "test_document.txt"
        assert sample_document_metadata.document_type == "procedure"
        assert sample_document_metadata.manufacturing_domain == "quality_control"
    
    @patch('builtins.open', create=True)
    def test_add_document(self, mock_open, rag_system, sample_document_metadata):
        """Test adding a document to the RAG system"""
        # Mock file content
        mock_open.return_value.__enter__.return_value.read.return_value = (
            "This is a test manufacturing document with quality control procedures."
        )
        
        # Test adding document
        with patch.object(rag_system, '_extract_text_content', return_value="Test content"):
            result = rag_system.add_document(
                file_path="test_document.txt",
                metadata=sample_document_metadata
            )
            assert result is True
    
    def test_text_splitter_initialization(self, rag_system):
        """Test text splitter configuration"""
        splitter = rag_system._get_text_splitter()
        assert splitter is not None
        assert hasattr(splitter, 'split_text')
    
    def test_manufacturing_keywords_detection(self, rag_system):
        """Test manufacturing keywords detection"""
        test_text = "Quality control procedures require dimensional tolerances within specifications."
        keywords = rag_system._extract_manufacturing_keywords(test_text)
        assert len(keywords) > 0
        assert any(keyword in ["quality", "control", "dimensional", "tolerances"] for keyword in keywords)
    
    def test_query_functionality(self, rag_system):
        """Test query functionality with mock data"""
        # Mock the collection query method
        with patch.object(rag_system.collection, 'query', return_value={
            'documents': [["Test document content"]],
            'metadatas': [[{"source": "test.txt"}]],
            'distances': [[0.5]]
        }):
            results = rag_system.query("test query", n_results=1)
            assert len(results) > 0
            assert 'content' in results[0]
            assert 'metadata' in results[0]


class TestImplicitRuleExtractor:
    """Test suite for Implicit Rule Extractor functionality"""
    
    @pytest.fixture
    def rule_extractor(self):
        """Create a test rule extractor instance"""
        return ImplicitRuleExtractor()
    
    def test_initialization(self, rule_extractor):
        """Test rule extractor initialization"""
        assert rule_extractor is not None
        assert hasattr(rule_extractor, 'nlp')
        assert hasattr(rule_extractor, 'zero_shot_classifier')
    
    def test_manufacturing_relevance_scoring(self, rule_extractor):
        """Test manufacturing relevance scoring"""
        manufacturing_text = "The component must maintain dimensional tolerances within ±0.05mm"
        non_manufacturing_text = "The weather is nice today"
        
        mfg_score = rule_extractor._calculate_manufacturing_relevance(manufacturing_text)
        non_mfg_score = rule_extractor._calculate_manufacturing_relevance(non_manufacturing_text)
        
        assert mfg_score > non_mfg_score
        assert mfg_score > 0.5
    
    def test_constraint_detection(self, rule_extractor):
        """Test constraint detection in text"""
        constraint_text = "Temperature must not exceed 150°C during the process"
        constraints = rule_extractor._extract_constraints(constraint_text)
        
        assert len(constraints) > 0
        assert any("150°C" in str(constraint) for constraint in constraints)
    
    def test_rule_extraction(self, rule_extractor):
        """Test full rule extraction process"""
        sample_text = """
        Manufacturing Process Requirements:
        - All components must pass dimensional inspection
        - Surface finish should be Ra ≤ 1.6μm
        - Heat treatment required for steel parts
        - Quality control documentation mandatory
        """
        
        rules = rule_extractor.extract_rules(sample_text)
        assert len(rules) > 0
        assert all('confidence' in rule for rule in rules)
        assert all('type' in rule for rule in rules)
    
    def test_keyword_extraction(self, rule_extractor):
        """Test manufacturing keyword extraction"""
        text = "Machining operations require proper coolant flow and spindle speed control"
        keywords = rule_extractor._extract_manufacturing_keywords(text)
        
        assert len(keywords) > 0
        expected_keywords = ["machining", "coolant", "spindle", "speed"]
        assert any(keyword in keywords for keyword in expected_keywords)
    
    def test_batch_processing(self, rule_extractor):
        """Test batch processing of multiple documents"""
        documents = [
            "Quality control procedures for automotive parts",
            "Safety guidelines for manufacturing environment",
            "Maintenance schedule for CNC equipment"
        ]
        
        batch_results = rule_extractor.process_batch(documents)
        assert len(batch_results) == len(documents)
        assert all('rules' in result for result in batch_results)


class TestIntegration:
    """Integration tests for the complete RAG system"""
    
    @pytest.fixture
    def integrated_system(self):
        """Create integrated RAG system with rule extractor"""
        rag = EnhancedManufacturingRAG(
            collection_name="test_integration",
            persist_directory="./test_integration_db"
        )
        extractor = ImplicitRuleExtractor()
        return rag, extractor
    
    def test_end_to_end_processing(self, integrated_system):
        """Test end-to-end document processing and querying"""
        rag, extractor = integrated_system
        
        # Sample manufacturing document
        document_content = """
        Quality Control Standard Operating Procedure
        
        1. All incoming materials must be inspected within 24 hours
        2. Dimensional tolerances: ±0.05mm for critical features
        3. Surface finish requirements: Ra ≤ 1.6μm
        4. Documentation must be maintained for all lots
        """
        
        # Mock file operations
        with patch('builtins.open', create=True) as mock_file:
            mock_file.return_value.__enter__.return_value.read.return_value = document_content
            
            # Add document to RAG system
            metadata = DocumentMetadata(
                source="test_sop.txt",
                document_type="procedure",
                manufacturing_domain="quality_control"
            )
            
            with patch.object(rag, '_extract_text_content', return_value=document_content):
                rag.add_document("test_sop.txt", metadata)
        
        # Query the system
        with patch.object(rag.collection, 'query', return_value={
            'documents': [[document_content]],
            'metadatas': [[metadata.__dict__]],
            'distances': [[0.3]]
        }):
            results = rag.query("What are the quality control requirements?")
            assert len(results) > 0
        
        # Extract rules from results
        if results:
            rules = extractor.extract_rules(results[0]['content'])
            assert len(rules) > 0
    
    def test_performance_metrics(self, integrated_system):
        """Test performance metrics collection"""
        rag, extractor = integrated_system
        
        # Simulate multiple queries and measure performance
        import time
        
        query_times = []
        for i in range(3):
            start_time = time.time()
            
            # Mock query execution
            with patch.object(rag.collection, 'query', return_value={
                'documents': [["Sample manufacturing content"]],
                'metadatas': [[{"source": f"doc_{i}.txt"}]],
                'distances': [[0.4]]
            }):
                results = rag.query(f"test query {i}")
                
            query_time = time.time() - start_time
            query_times.append(query_time)
        
        # Verify performance metrics
        avg_query_time = sum(query_times) / len(query_times)
        assert avg_query_time > 0
        assert all(t > 0 for t in query_times)


# Test configuration
if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])