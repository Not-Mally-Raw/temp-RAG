"""
Document extraction modules for text, tables, and images.
"""

from .text import extract_sentences
from .table import dual_pipeline_2
from .image import extract_images

__all__ = [
    'extract_sentences',
    'dual_pipeline_2',
    'extract_images',
]
