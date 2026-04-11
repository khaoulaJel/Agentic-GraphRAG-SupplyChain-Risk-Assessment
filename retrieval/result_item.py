"""
RetrieverResultItem: Standardized output for retrieval results.
"""
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class RetrieverResultItem:
    content: str
    metadata: Dict[str, Any]
