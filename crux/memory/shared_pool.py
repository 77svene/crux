from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Protocol, Type
from datetime import datetime
from collections import defaultdict
import threading
import uuid
import logging

logger = logging.getLogger(__name__)


class ResolutionStrategy(Enum):
    LAST_WRITE_WINS = "last_write_wins"
    AGENT_PRIORITY = "agent_priority"
    MERGE = "merge"
    ESCALATE = "escalate"


class Permission(Enum):
    READ = "read"
    WRITE = "write"


@dataclass
class VersionedMemory:
    memory_id: str
    content: str
    agent_id: str
    pool_id: str
    timestamp: datetime
    version: int
    confidence: float = 1.0
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    is_resolved: bool = False
    resolution_strategy: Optional[ResolutionStrategy] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "agent_id": self.agent_id,
            "pool_id": self.pool_id,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "is_resolved": self.is_resolved,
            "resolution_strategy": self.resolution_strategy.value if self.resolution_strategy else None,
            "tags": self.tags,
        }


@dataclass
class MemoryConflict:
    conflict_id: str
    entries: List[VersionedMemory]
    similarity_score: float
    detected_at: datetime
    resolved: bool = False
    resolution: Optional[str] = None
    resolved_by: Optional[str] = None
    resolution_strategy: Optional[ResolutionStrategy] = None
    escalated_to_human: bool = False
    human_resolution: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "entries": [e.to_dict() for e in self.entries],
            "similarity_score": self.similarity_score,
            "detected_at": self.detected_at.isoformat(),
            "resolved": self.resolved,
            "resolution": self.resolution,
            "resolved_by": self.resolved_by,
            "resolution_strategy": self.resolution_strategy.value if self.resolution_strategy else None,
            "escalated_to_human": self.escalated_to_human,
            "human_resolution": self.human_resolution,
        }


class ConflictDetector:
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        normalize_fn: Optional[Callable[[str], str]] = None,
    ):
        self.similarity_threshold = similarity_threshold
        self.embedding_fn = embedding_fn
        self.normalize_fn = normalize_fn or (lambda x: x.lower().strip())
        self._cache: Dict[str, List[float]] = {}
        self._cache_lock = threading.Lock()

    def compute_similarity(self, text1: str, text2: str) -> float:
        if self.embedding_fn:
            emb1 = self._get_embedding(text1)
            emb2 = self._get_embedding(text2)
            if emb1 is not None and emb2 is not None:
                return self._cosine_similarity(emb1, emb2)
        return self._jaccard_similarity(text1, text2)

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        cache_key = hash(text)
        with self._cache_lock:
            if cache_key in self._cache:
                return self._cache[cache_key]
        embedding = self.embedding_fn(text)
        with self._cache_lock:
            self._cache[cache_key] = embedding
        return embedding

    def _cosine_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = sum(a * a for a in emb1) ** 0.5
        norm2 = sum(b * b for b in emb2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        norm1 = self.normalize_fn(text1)
        norm2 = self.normalize_fn(text2)
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)

    def detect_conflicts(self, entries: List[VersionedMemory]) -> List[MemoryConflict]:
        conflicts = []
        checked_pairs = set()
        for i, entry1 in enumerate(entries):
            for entry2 in entries[i + 1:]:
                pair_key = tuple(sorted([entry1.memory_id, entry2.memory_id]))
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)
                similarity = self.compute_similarity(entry1.content, entry2.content)
                if similarity >= self.similarity_threshold:
                    conflicts.append(MemoryConflict(
                        conflict_id=str(uuid.uuid4()),
                        entries=[entry1, entry2],
                        similarity_score=similarity,
                        detected_at=datetime.utcnow(),
                    ))
                    logger.info(f"Conflict detected between {entry1.memory_id} and {entry2.memory_id} with similarity {similarity:.3f}")
        return conflicts

    def find_similar_entries(
        self, query: str, entries: List[VersionedMemory], top_k: int = 5
    ) -> List[tuple]:
        similarities = []
        for entry in entries:
            sim = self.compute_similarity(query, entry.content)
            similarities.append((entry, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class ResolutionStrategyHandler(Protocol):
    def resolve(self, conflict: MemoryConflict, context: Dict[str, Any]) -> str: ...
    @property
    def strategy(self) -> ResolutionStrategy: ...


class LastWriteWinsHandler:
    @property
    def strategy(self) -> ResolutionStrategy:
        return ResolutionStrategy.LAST_WRITE_WINS

    def resolve(self, conflict: MemoryConflict, context: Dict[str, Any]) -> str:
        sorted_entries = sorted(conflict.entries, key=lambda e: e.timestamp, reverse=True)
        winner = sorted_entries[0]
        logger.info(f"LastWriteWins: selected memory {winner.memory_id} from agent {winner.agent_id}")
        return winner.content


class AgentPriorityHandler:
    def __init__(self, agent_priorities: Optional[Dict[str, float]] = None):
        self.agent_priorities: Dict[str, float] = agent_priorities or {}

    @property
    def strategy(self) -> ResolutionStrategy:
        return ResolutionStrategy.AGENT_PRIORITY

    def resolve(self, conflict: MemoryConflict, context: Dict[str, Any]) -> str:
        priorities = context.get("agent_priorities", self.agent_priorities)
        scored_entries = []
        for entry in conflict.entries:
            priority = priorities.get(entry.agent_id, 1.0)
            score = priority * entry.confidence
            if entry.timestamp == max(e.timestamp for e in conflict.entries):
                score += 0.1
            scored_entries.append((entry, score))
        scored_entries.sort(key=lambda x: x[1], reverse=True)
        winner = scored_entries[0][0]
        logger.info(f"AgentPriority: selected memory {winner.memory_id} from agent {winner.agent_id} with priority {priorities.get(winner.agent_id, 1.0)}")
        return winner.content


class MergeStrategyHandler:
    def __init__(self