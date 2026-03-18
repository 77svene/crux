"""
Memory Arbitration Layer for Multi-Agent Systems

This module provides memory arbitration capabilities for multi-agent systems,
enabling agents to share, negotiate, and resolve conflicting memories.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID, uuid4

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning("sentence-transformers not installed. Using fallback similarity.")


class ResolutionStrategy(str, Enum):
    """
    Enum defining the available conflict resolution strategies.
    
    LAST_WRITE_WINS: Most recent update wins
    AGENT_PRIORITY: Agent with higher priority weight wins
    MERGE: Combine both perspectives into a unified memory
    ESCALATE: Flag for human review
    """
    LAST_WRITE_WINS = "last_write_wins"
    AGENT_PRIORITY = "agent_priority"
    MERGE = "merge"
    ESCALATE = "escalate"


class ConflictStatus(str, Enum):
    """Status of a detected conflict."""
    DETECTED = "detected"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class MemoryType(str, Enum):
    """Type of memory in the pool."""
    PRIVATE = "private"
    SHARED = "shared"


@dataclass
class AgentPermissions:
    """Defines read/write permissions for an agent in a memory pool."""
    agent_id: str
    can_read: bool = True
    can_write: bool = True
    priority_weight: float = 1.0


@dataclass
class VersionedMemory:
    """Represents a memory entry with full versioning metadata."""
    memory_id: UUID
    content: