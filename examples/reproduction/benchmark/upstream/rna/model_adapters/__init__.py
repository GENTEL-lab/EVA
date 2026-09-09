"""Model adapters for unified benchmark system"""

from .base_adapter import BaseAdapter
from .huggingface_adapter import HuggingFaceAdapter

__all__ = ['BaseAdapter', 'HuggingFaceAdapter']
