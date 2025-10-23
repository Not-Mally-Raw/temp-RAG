# Configuration file for the Enhanced RAG System

import os
from typing import Dict, List, Any

class RAGConfig:
    """Configuration settings for the Enhanced RAG System"""
    
    # Model Configuration
    EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
    EMBEDDING_DIMENSION = 1024
    ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
    
    # Database Configuration
    DEFAULT_COLLECTION_NAME = "manufacturing_enhanced"
    DEFAULT_PERSIST_DIRECTORY = "./chroma_enhanced_db"
    
    # Text Processing Configuration
    DEFAULT_CHUNK_SIZE = 500
    DEFAULT_CHUNK_OVERLAP = 50
    MIN_CHUNK_SIZE = 100
    MAX_CHUNK_SIZE = 2000
    
    # Query Configuration
    DEFAULT_N_RESULTS = 5
    MAX_N_RESULTS = 20
    RELEVANCE_THRESHOLD = 0.7
    
    # Rule Extraction Configuration
    MANUFACTURING_CONFIDENCE_THRESHOLD = 0.6
    RULE_CONFIDENCE_THRESHOLD = 0.7
    MAX_RULES_PER_DOCUMENT = 50
    
    # Manufacturing Keywords and Domains
    MANUFACTURING_KEYWORDS = [
        # Quality Control
        "quality", "inspection", "tolerance", "specification", "standard",
        "defect", "compliance", "certification", "audit", "control",
        
        # Manufacturing Processes
        "machining", "welding", "assembly", "fabrication", "casting",
        "forging", "molding", "stamping", "cutting", "drilling",
        
        # Materials and Properties
        "material", "steel", "aluminum", "plastic", "composite",
        "hardness", "strength", "durability", "corrosion", "temperature",
        
        # Measurements and Dimensions
        "dimension", "diameter", "thickness", "length", "width",
        "radius", "angle", "surface", "finish", "roughness",
        
        # Safety and Compliance
        "safety", "hazard", "protection", "regulation", "standard",
        "environmental", "emission", "waste", "disposal", "handling",
        
        # Equipment and Tools
        "equipment", "machine", "tool", "spindle", "coolant",
        "maintenance", "calibration", "operation", "setup", "fixture"
    ]
    
    MANUFACTURING_DOMAINS = [
        "automotive",
        "aerospace",
        "electronics",
        "medical_devices",
        "machinery",
        "consumer_goods",
        "energy",
        "chemical",
        "pharmaceutical",
        "food_processing",
        "textiles",
        "packaging"
    ]
    
    DOCUMENT_TYPES = [
        "specification",
        "procedure",
        "guideline",
        "standard",
        "manual",
        "report",
        "drawing",
        "checklist",
        "form",
        "certificate"
    ]
    
    # Rule Types for Classification
    RULE_TYPES = [
        "quality_requirement",
        "safety_constraint",
        "process_parameter",
        "material_specification",
        "dimensional_tolerance",
        "environmental_condition",
        "testing_procedure",
        "maintenance_schedule",
        "compliance_standard",
        "performance_metric"
    ]
    
    # Manufacturing Processes for Context
    MANUFACTURING_PROCESSES = {
        "machining": [
            "turning", "milling", "drilling", "grinding", "boring",
            "threading", "reaming", "tapping", "broaching", "shaping"
        ],
        "forming": [
            "stamping", "bending", "drawing", "rolling", "forging",
            "extrusion", "spinning", "hydroforming", "stretch_forming"
        ],
        "joining": [
            "welding", "brazing", "soldering", "bonding", "fastening",
            "riveting", "crimping", "assembly", "integration"
        ],
        "casting": [
            "sand_casting", "die_casting", "investment_casting",
            "permanent_mold", "centrifugal_casting", "shell_molding"
        ],
        "finishing": [
            "polishing", "grinding", "coating", "painting", "plating",
            "anodizing", "passivation", "heat_treatment", "surface_treatment"
        ],
        "inspection": [
            "dimensional_inspection", "visual_inspection", "ndt_testing",
            "functional_testing", "performance_testing", "quality_audit"
        ]
    }
    
    # Performance Monitoring
    PERFORMANCE_METRICS = {
        "query_time_threshold": 5.0,  # seconds
        "memory_usage_threshold": 1024,  # MB
        "accuracy_threshold": 0.8,
        "recall_threshold": 0.7
    }
    
    # File Processing Configuration
    SUPPORTED_FILE_TYPES = [
        ".txt", ".pdf", ".docx", ".doc", ".rtf",
        ".xlsx", ".xls", ".csv", ".json", ".xml"
    ]
    
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    
    # Logging Configuration
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = "rag_system.log"
    
    # API Configuration
    API_HOST = "localhost"
    API_PORT = 8000
    API_WORKERS = 1
    
    # Streamlit Configuration
    STREAMLIT_HOST = "localhost"
    STREAMLIT_PORT = 8501
    
    @classmethod
    def get_embedding_config(cls) -> Dict[str, Any]:
        """Get embedding model configuration"""
        return {
            "model_name": cls.EMBEDDING_MODEL,
            "dimension": cls.EMBEDDING_DIMENSION,
            "device": "cpu",  # Change to "cuda" if GPU available
            "normalize_embeddings": True
        }
    
    @classmethod
    def get_text_splitter_config(cls) -> Dict[str, Any]:
        """Get text splitter configuration"""
        return {
            "chunk_size": cls.DEFAULT_CHUNK_SIZE,
            "chunk_overlap": cls.DEFAULT_CHUNK_OVERLAP,
            "length_function": len,
            "separators": ["\n\n", "\n", " ", ""]
        }
    
    @classmethod
    def get_database_config(cls) -> Dict[str, Any]:
        """Get database configuration"""
        return {
            "collection_name": cls.DEFAULT_COLLECTION_NAME,
            "persist_directory": cls.DEFAULT_PERSIST_DIRECTORY,
            "embedding_function": None  # Will be set during initialization
        }
    
    @classmethod
    def get_rule_extraction_config(cls) -> Dict[str, Any]:
        """Get rule extraction configuration"""
        return {
            "manufacturing_threshold": cls.MANUFACTURING_CONFIDENCE_THRESHOLD,
            "rule_threshold": cls.RULE_CONFIDENCE_THRESHOLD,
            "max_rules": cls.MAX_RULES_PER_DOCUMENT,
            "rule_types": cls.RULE_TYPES,
            "keywords": cls.MANUFACTURING_KEYWORDS
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings"""
        try:
            # Check chunk size constraints
            assert cls.MIN_CHUNK_SIZE <= cls.DEFAULT_CHUNK_SIZE <= cls.MAX_CHUNK_SIZE
            
            # Check overlap constraints
            assert 0 <= cls.DEFAULT_CHUNK_OVERLAP < cls.DEFAULT_CHUNK_SIZE
            
            # Check threshold constraints
            assert 0 <= cls.RELEVANCE_THRESHOLD <= 1
            assert 0 <= cls.MANUFACTURING_CONFIDENCE_THRESHOLD <= 1
            assert 0 <= cls.RULE_CONFIDENCE_THRESHOLD <= 1
            
            # Check result limits
            assert 1 <= cls.DEFAULT_N_RESULTS <= cls.MAX_N_RESULTS
            
            return True
            
        except AssertionError:
            return False
    
    @classmethod
    def update_from_env(cls):
        """Update configuration from environment variables"""
        # Model configuration
        cls.EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", cls.EMBEDDING_MODEL)
        cls.ZERO_SHOT_MODEL = os.getenv("RAG_ZERO_SHOT_MODEL", cls.ZERO_SHOT_MODEL)
        
        # Database configuration
        cls.DEFAULT_PERSIST_DIRECTORY = os.getenv(
            "RAG_PERSIST_DIR", cls.DEFAULT_PERSIST_DIRECTORY
        )
        
        # Performance thresholds
        if os.getenv("RAG_RELEVANCE_THRESHOLD"):
            cls.RELEVANCE_THRESHOLD = float(os.getenv("RAG_RELEVANCE_THRESHOLD"))
        
        if os.getenv("RAG_MANUFACTURING_THRESHOLD"):
            cls.MANUFACTURING_CONFIDENCE_THRESHOLD = float(
                os.getenv("RAG_MANUFACTURING_THRESHOLD")
            )
        
        # File size limits
        if os.getenv("RAG_MAX_FILE_SIZE"):
            cls.MAX_FILE_SIZE = int(os.getenv("RAG_MAX_FILE_SIZE"))
        
        # API configuration
        cls.API_HOST = os.getenv("RAG_API_HOST", cls.API_HOST)
        cls.API_PORT = int(os.getenv("RAG_API_PORT", cls.API_PORT))
        
        # Logging
        cls.LOG_LEVEL = os.getenv("RAG_LOG_LEVEL", cls.LOG_LEVEL)


# Initialize configuration
RAGConfig.update_from_env()

# Validate configuration
if not RAGConfig.validate_config():
    raise ValueError("Invalid RAG configuration settings")

# Export configuration instance
config = RAGConfig()