"""
Utils for the Refactored RAG System
Contains configurations, logging, and shared utilities
"""

import logging
import os
from dataclasses import dataclass
from typing import Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RuleEngineConfig:
    """Unified configuration for rule extraction."""
    # Extraction thresholds
    min_confidence: float = 0.3
    max_rules_per_document: int = 50

    # Processing flags
    enable_llm_enhancement: bool = False  # Default off for speed
    enable_implicit_extraction: bool = True
    enable_pattern_extraction: bool = True

    # Chunking (use tiktoken if available)
    chunk_size: int = 800  # Conservative for actual tokens
    chunk_overlap: int = 100

    # Classification
    high_value_threshold: float = 0.8

    # RAG settings
    rag_top_k: int = 3
    rag_score_threshold: float = 0.2

    # LLM settings
    groq_model: str = "llama-3.1-8b-instant"
    max_llm_retries: int = 2


@dataclass
class VectorConfig:
    """Configuration for vector operations."""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "manufacturing_rag"


@dataclass
class IngestConfig:
    """Configuration for document ingestion."""
    supported_formats: list = None

    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['.pdf', '.txt', '.docx']


# Global configurations
DEFAULT_RULE_CONFIG = RuleEngineConfig()
DEFAULT_VECTOR_CONFIG = VectorConfig()
DEFAULT_INGEST_CONFIG = IngestConfig()


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('rag_system.log')
        ]
    )


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_env_vars():
    """Load environment variables from .env file if it exists."""
    env_path = get_project_root() / '.env'
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)


def validate_config(config: Any) -> bool:
    """Validate configuration object."""
    if hasattr(config, '__dataclass_fields__'):
        # It's a dataclass, check required fields
        required_fields = ['min_confidence', 'max_rules_per_document'] if isinstance(config, RuleEngineConfig) else []
        for field in required_fields:
            if hasattr(config, field):
                value = getattr(config, field)
                if field == 'min_confidence' and not (0.0 <= value <= 1.0):
                    logger.error(f"Invalid {field}: {value}")
                    return False
        return True
    return False


def safe_to_dict(obj: Any) -> Dict[str, Any]:
    """Safely convert object to dictionary."""
    if obj is None:
        return {}

    if isinstance(obj, dict):
        return obj

    if hasattr(obj, 'dict'):
        return obj.dict()

    if hasattr(obj, '__dict__'):
        return obj.__dict__

    return {}


def log_performance(func_name: str, start_time: float, **kwargs):
    """Log performance metrics."""
    duration = time.time() - start_time
    logger.info(f"{func_name} completed in {duration:.2f}s", extra=kwargs)


# Import time for performance logging
import time