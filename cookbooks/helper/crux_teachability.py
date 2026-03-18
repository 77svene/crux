# Copyright (c) 2023 - 2024, Owners of https://github.com/autogen-ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
# forked from autogen.agentchat.contrib.capabilities.teachability.Teachability

import asyncio
from typing import Dict, Optional, Union, AsyncGenerator, List, Any
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import threading
from difflib import SequenceMatcher

from autogen.agentchat.assistant_agent import ConversableAgent
from autogen.agentchat.contrib.capabilities.agent_capability import AgentCapability
from autogen.agentchat.contrib.text_analyzer_agent import TextAnalyzerAgent
from termcolor import colored

from crux import Memory


class ResolutionStrategy(Enum):
    """Enumeration of conflict resolution strategies for multi-agent memory arbitration."""
    LAST_WRITE_WINS = "last_write_wins"       # Most recent write wins
    AGENT_PRIORITY = "agent_priority"          # Agent with highest priority weight wins
    MERGE = "merge"                             # Merge both perspectives into one
    FLAG_FOR_HUMAN = "flag_for_human"          # Flag for human review
    VOTE = "vote"                               # Majority vote among agents


@dataclass
class MemoryVersion:
    """Represents a versioned memory entry with metadata for conflict resolution."""
    memory_id: str
    content: str
    agent_id: str
    timestamp: datetime
    version: int
    pool_id: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_version_id: Optional[str] = None
    is_deleted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "pool_id": self.pool_id,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "parent_version_id": self.parent_version_id,
            "is_deleted": self.is_deleted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryVersion":
        """Create MemoryVersion from dictionary."""
        data = data.copy()
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class ConflictReport:
    """Report detailing a detected memory conflict."""
    conflict_id: str
    memory_versions: List[MemoryVersion]
    conflicting_facts: List[str]
    similarity_score: float
    resolution_applied: Optional[ResolutionStrategy] = None
    resolved_content: Optional[str] = None
    flagged_for_human: bool = False
    human_review_notes: Optional[str] = None
    resolved_at: Optional[datetime] = None


class VersionedMemory:
    """
    Memory wrapper that tracks version history for every memory entry.
    Enables temporal ordering and confidence scoring for conflict resolution.
    """

    def __init__(self, memory: Memory):
        """Initialize VersionedMemory with an underlying Memory instance."""
        self._memory = memory
        self._version_history: Dict[str, List[MemoryVersion]] = {}
        self._lock = threading.RLock()
        self._version_counter = 0

    def add(
        self,
        messages: List[Dict],
        agent_id: str,
        pool_id: Optional[str] = None,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Add a new memory with version tracking.
        
        Args:
            messages: List of message dicts to store
            agent_id: ID of the agent adding the memory
            pool_id: Optional pool ID for shared memory
            confidence: Confidence score (0.0-1.0)
            metadata: Additional metadata
            **kwargs: Additional arguments for underlying add
            
        Returns:
            Result from underlying memory add with version info
        """
        with self._lock:
            result = self._memory.add(messages, agent_id=agent_id, **kwargs)
            
            # Extract memory IDs from result
            memory_ids = []
            if isinstance(result, dict):
                memory_ids = result.get("memory_ids", [])
            elif isinstance(result, list):
                memory_ids = result
            
            # Create version entries for each memory
            for mem_id in memory_ids:
                content = ""
                if isinstance(messages, list) and len(messages) > 0:
                    content = messages[0].get("content", "")
                
                version = MemoryVersion(
                    memory_id=mem_id,
                    content=content,
                    agent_id=agent_id,
                    timestamp=datetime.now(),
                    version=self._version_counter,
                    pool_id=pool_id,
                    confidence=confidence,
                    metadata=metadata or {},
                )
                
                if mem_id not in self._version_history:
                    self._version_history[mem_id] = []
                self._version_history[mem_id].append(version)
                self._version_counter += 1
            
            if "version_info" not in result:
                result["version_info"] = {}
            result["version_info"]["pool_id"] = pool_id
            result["version_info"]["confidence"] = confidence
            result["version_info"]["version"] = self._version_counter - len(memory_ids)
            
            return result

    def get_versions(self, memory_id: str) -> List[MemoryVersion]:
        """Get all versions of a specific memory."""
        with self._lock:
            return self._version_history.get(memory_id, []).copy()

    def get_latest_version(self, memory_id: str) -> Optional[MemoryVersion]:
        """Get the most recent version of a memory."""
        versions = self.get_versions(memory_id)
        return versions[-1] if versions else None

    def get_all_versions(self) -> Dict[str, List[MemoryVersion]]:
        """Get complete version history."""
        with self._lock:
            return {k: v.copy() for k, v in self._version_history.items()}

    def update_confidence(self, memory_id: str, new_confidence: float) -> bool:
        """Update confidence score for a memory version."""
        with self._lock:
            versions = self._version_history.get(memory_id, [])
            if versions:
                versions[-1].confidence = new_confidence
                return True
            return False

    def mark_deleted(self, memory_id: str, agent_id: str) -> bool:
        """Mark a memory as deleted without removing it."""
        with self._lock:
            versions = self._version_history.get(memory_id, [])
            if versions:
                version = MemoryVersion(
                    memory_id=memory_id,
                    content=versions[-1].content,
                    agent_id=agent_id,
                    timestamp=datetime.now(),
                    version=self._version_counter,
                    pool_id=versions[-1].pool_id,
                    confidence=0.0,
                    metadata={"deleted_by": agent_id},
                    parent_version_id=versions[-1].memory_id,
                    is_deleted=True,
                )
                versions.append(version)
                self._version_counter += 1
                return True
            return False

    def get_by_agent(self, agent_id: str) -> List[MemoryVersion]:
        """Get all memory versions created by a specific agent."""
        with self._lock:
            result = []
            for versions in self._version_history.values():
                for v in versions:
                    if v.agent_id == agent_id:
                        result.append(v)
            return result

    def get_by_pool(self, pool_id: str) -> List[MemoryVersion]:
        """Get all memory versions in a specific pool."""
        with self._lock:
            result = []
            for versions in self._version_history.values():
                for v in versions:
                    if v.pool_id == pool_id:
                        result.append(v)
            return result

    def __getattr__(self, name: str):
        """Delegate all other attributes to underlying memory."""
        return getattr(self._memory, name)


class ConflictDetector:
    """
    Detects conflicting memories using semantic similarity.
    Uses a threshold-based approach to identify overlapping facts.
    """

    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize ConflictDetector.
        
        Args:
            similarity_threshold: Minimum similarity score to flag as conflict (0.0-1.0)
        """
        self.similarity_threshold = similarity_threshold
        self._conflict_history: List[ConflictReport] = []
        self._lock = threading.Lock()

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts.
        
        Uses SequenceMatcher for basic similarity. In production,
        this should be replaced with embeddings-based similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Basic string similarity
        ratio = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        
        # Additional heuristics for better matching
        # Check for common keywords
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if words1 and words2:
            word_overlap = len(words1 & words2) / len(words1 | words2)
            # Combine string ratio with word overlap
            ratio = 0.6 * ratio + 0.4 * word_overlap
        
        return min(ratio, 1.0)

    def detect_conflicts(
        self,
        memory1: MemoryVersion,
        memory2: MemoryVersion,
        check_temporal: bool = True,
        check_content: bool = True,
    ) -> Optional[ConflictReport]:
        """
        Detect if two memory versions conflict.
        
        Args:
            memory1: First memory version
            memory2: Second memory version
            check_temporal: Whether to check temporal proximity
            check_content: Whether to check content similarity
            
        Returns:
            ConflictReport if conflict detected, None otherwise
        """
        conflicting_facts = []
        similarity_score = 0.0
        
        # Check content similarity
        if check_content:
            similarity_score = self.compute_similarity(memory1.content, memory2.content)
            
            if similarity_score >= self.similarity_threshold:
                # Extract conflicting fact summaries
                conflicting_facts.append(
                    f"Similarity: {similarity_score:.2f} between '{memory1.content[:50]}...' and '{memory2.content[:50]}...'"
                )
        
        # Check temporal proximity (conflicting updates within short timeframe)
        if check_temporal and not conflicting_facts:
            time_diff = abs((memory1.timestamp - memory2.timestamp).total_seconds())
            # Consider temporal conflict if same agent updates within 5 seconds
            if time_diff < 5 and memory1.agent_id == memory2.agent_id:
                similarity_score = 0.5
                conflicting_facts.append(
                    f"Temporal proximity: {time_diff:.1f}s between updates"
                )
        
        if not conflicting_facts:
            return None
        
        conflict_id = f"conflict_{memory1.memory_id}_{memory2.memory_id}_{datetime.now().timestamp()}"
        
        report = ConflictReport(
            conflict_id=conflict_id,
            memory_versions=[memory1, memory2],
            conflicting_facts=conflicting_facts,
            similarity_score=similarity_score,
        )
        
        with self._lock:
            self._conflict_history.append(report)
        
        return report

    def scan_for_conflicts(
        self,
        memories: List[MemoryVersion],
        compare_with_existing: bool = True,
    ) -> List[ConflictReport]:
        """
        Scan a list of memories for conflicts with each other.
        
        Args:
            memories: List of memory versions to check
            compare_with_existing: Whether to compare with historical conflicts
            
        Returns:
            List of detected conflicts
        """
        conflicts = []
        checked_pairs = set()
        
        # Check pairwise for conflicts
        for i, mem1 in enumerate(memories):
            for mem2 in memories[i + 1:]:
                pair_key = tuple(sorted([mem1.memory_id, mem2.memory_id]))
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)
                
                conflict = self.detect_conflicts(mem1, mem2)
                if conflict:
                    conflicts.append(conflict)
        
        return conflicts

    def get_conflict_history(self) -> List[ConflictReport]:
        """Get all historical conflict reports."""
        with self._lock:
            return self._conflict_history.copy()

    def get_unresolved_conflicts(self) -> List[ConflictReport]:
        """Get conflicts that haven't been resolved yet."""
        with self._lock:
            return [c for c in self._conflict_history if c.resolution_applied is None]


class ConflictResolver:
    """
    Resolves memory conflicts using configurable strategies.
    Supports pluggable resolution strategies for different use cases.
    """

    def __init__(
        self,
        default_strategy: ResolutionStrategy = ResolutionStrategy.LAST_WRITE_WINS,
        agent_priorities: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize ConflictResolver.
        
        Args:
            default_strategy: Default resolution strategy to use
            agent_priorities: Map of agent_id to priority weight (higher = more priority)
        """
        self.default_strategy = default_strategy
        self.agent_priorities = agent_priorities or {}
        self._resolution_callbacks: Dict[ResolutionStrategy, callable] = {}

    def register_strategy(self, strategy: ResolutionStrategy, callback: callable):
        """Register a custom resolution callback for a strategy."""
        self._resolution_callbacks[strategy] = callback

    def resolve(
        self,
        conflict: ConflictReport,
        strategy: Optional[ResolutionStrategy] = None,
        **kwargs
    ) -> MemoryVersion:
        """
        Resolve a conflict using the specified strategy.
        
        Args:
            conflict: The conflict report to resolve
            strategy: Resolution strategy to use (defaults to configured default)
            **kwargs: Additional arguments for strategy-specific resolution
            
        Returns:
            The winning memory version
        """
        strategy = strategy or self.default_strategy
        
        # Check for custom resolver
        if strategy in self._resolution_callbacks:
            return self._resolution_callbacks[strategy](conflict, **kwargs)
        
        # Built-in strategies
        if strategy == ResolutionStrategy.LAST_WRITE_WINS:
            return self._resolve_last_write_wins(conflict)
        elif strategy == ResolutionStrategy.AGENT_PRIORITY:
            return self._resolve_agent_priority(conflict, **kwargs)
        elif strategy == ResolutionStrategy.MERGE:
            return self._resolve_merge(conflict, **kwargs)
        elif strategy == ResolutionStrategy.FLAG_FOR_HUMAN:
            return self._flag_for_human(conflict, **kwargs)
        elif strategy == ResolutionStrategy.VOTE:
            return self._resolve_vote(conflict, **kwargs)
        else:
            raise ValueError(f"Unknown resolution strategy: {strategy}")

    def _resolve_last_write_wins(self, conflict: ConflictReport) -> MemoryVersion:
        """Select the most recently written memory."""
        versions = sorted(
            conflict.memory_versions,
            key=lambda v: v.timestamp,
            reverse=True
        )
        winner = versions[0]
        
        conflict.resolution_applied = ResolutionStrategy.LAST_WRITE_WINS
        conflict.resolved_content = winner.content
        conflict.resolved_at = datetime.now()
        
        return winner

    def _resolve_agent_priority(
        self,
        conflict: ConflictReport,
        **kwargs
    ) -> MemoryVersion:
        """Select memory from agent with highest priority weight."""
        def get_priority(v: MemoryVersion) -> float:
            return self.agent_priorities.get(v.agent_id, kwargs.get("default_priority", 0.5))
        
        versions = sorted(
            conflict.memory_versions,
            key=get_priority,
            reverse=True
        )
        winner = versions[0]
        
        conflict.resolution_applied = ResolutionStrategy.AGENT_PRIORITY
        conflict.resolved_content = winner.content
        conflict.resolved_at = datetime.now()
        
        return winner

    def _resolve_merge(
        self,
        conflict: ConflictReport,
        **kwargs
    ) -> MemoryVersion:
        """Merge both perspectives into a consolidated memory."""
        combined_content = kwargs.get(
            "merge_template",
            "{mem1}\n\n---\n\n{mem2}"
        ).format(
            mem1=conflict.memory_versions[0].content,
            mem2=conflict.memory_versions[1].content
        )
        
        # Create a merged version based on the most recent
        base = sorted(
            conflict.memory_versions,
            key=lambda v: v.timestamp,
            reverse=True
        )[0]
        
        merged = MemoryVersion(
            memory_id=base.memory_id,
            content=combined_content,
            agent_id="merge_resolution",
            timestamp=datetime.now(),
            version=base.version + 1,
            pool_id=base.pool_id,
            confidence=sum(v.confidence for v in conflict.memory_versions) / len(conflict.memory_versions),
            metadata={
                "merged_from": [v.memory_id for v in conflict.memory_versions],
                "merge_strategy": "conflict_resolution_merge",
            },
            parent_version_id=base.memory_id,
        )
        
        conflict.resolution_applied = ResolutionStrategy.MERGE
        conflict.resolved_content = combined_content
        conflict.resolved_at = datetime.now()
        
        return merged

    def _flag_for_human(
        self,
        conflict: ConflictReport,
        **kwargs
    ) -> MemoryVersion:
        """Flag the