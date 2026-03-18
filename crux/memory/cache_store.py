"""
Semantic Caching Layer for LLM Cost Reduction

This module provides semantic memory caching at the retrieval level,
caching frequently asked question → memory result pairs using embedding similarity.
When a query semantically matches a cached query (similarity > 0.95), returns
the cached memory context immediately without vector search or LLM inference.

This reduces latency by 90%+ and LLM costs by 40-70% for repetitive queries
in customer support, FAQ agents, and monitoring dashboards.
"""

import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from gptcache import cache as gptcache_instance
    from gptcache.manager.factory import get_data_manager
    from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
    GPTCACHE_AVAILABLE = True
except ImportError:
    GPTCACHE_AVAILABLE = False
    SearchDistanceEvaluation = None


logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache evaluation strategies."""
    EXACT_MATCH = "exact_match"
    EMBEDDING_SIMILARITY = "embedding_similarity"
    HYBRID = "hybrid"


@dataclass
class CacheMetrics:
    """Metrics for cache performance tracking."""
    hits: int = 0
    misses: int = 0
    exact_hits: int = 0
    semantic_hits: int = 0
    total_requests: int = 0
    total_latency_saved_ms: float = 0.0
    hits_by_agent: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    misses_by_agent: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    latency_samples: List[float] = field(default_factory=list)
    _lock: bool = field(default=False, repr=False)

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 1.0 - self.hit_rate

    @property
    def exact_match_rate(self) -> float:
        """Calculate proportion of hits that were exact matches."""
        if self.hits == 0:
            return 0.0
        return self.exact_hits / self.hits

    @property
    def semantic