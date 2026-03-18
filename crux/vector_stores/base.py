import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class VectorDBType(Enum):
    QDRANT = "qdrant"
    WEAVIATE = "weaviate"
    PGVECTOR = "pgvector"
    PINECONE = "pinecone"
    CHROMADB = "chromadb"


@dataclass
class VectorStoreConfig:
    db_type: VectorDBType
    host: str = "localhost"
    port: int = 6333
    url: Optional[str] = None
    api_key: Optional[str] = None
    index_name: str = "crux"
    dimension: int = 1536
    distance_func: str = "cosine"
    collection_config: Optional[Dict[str, Any]] = None
    ssl: bool = False
    timeout: int = 30
    connection_string: Optional[str] = None
    primary: bool = True
    priority: int = 0
    region: Optional[str] = None
    environment: Optional[str] = None


@dataclass
class SearchResult:
    id: str
    vector: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    score: float = 0.0
    document: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "vector": self.vector,
            "metadata": self.metadata,
            "score": self.score,
            "document": self.document,
            "payload": self.payload,
        }


class VectorStoreException(Exception):
    pass


class CircuitBreakerOpen(Exception):
    pass


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
        half_open_max_calls: int = 3,
        exponential_base: float = 2.0,
        max_reset_timeout: float = 300.0,
    ):
        self._failure_threshold = failure_threshold
        self._reset_timeout = reset_timeout
        self._half_open_max_calls = half_open_max_calls
        self._exponential_base = exponential_base
        self._max_reset_timeout = max_reset_timeout
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._state = CircuitState.CLOSED
        self._last_failure_time: Optional[float] = None
        self._lock = threading.RLock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
            return self._state

    def _should_attempt_reset(self) -> bool:
        if self._last_failure_time is None:
            return True
        return (time.time() - self._last_failure_time) >= self._get_current_timeout()

    def _get_current_timeout(self) -> float:
        timeout = self._reset_timeout * (self._exponential_base ** (self._failure_count - self._failure_threshold))
        return min(timeout, self._max_reset_timeout)

    def allow_request(self) -> bool:
        with self._lock:
            state = self.state
            if state == CircuitState.CLOSED:
                return True
            if state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self._half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            return False

    def record_success(self):
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= 2:
                    self._reset()
            else:
                self._failure_count = 0
                self._success_count = 0

    def record_failure(self):
        with self._lock:
            self._failure_count += 1
            self._success_count = 0
            self._last_failure_time = time.time()
            if self._state == CircuitState.HALF_OPEN:
                self._trip()
            elif self._failure_count >= self._failure_threshold:
                self._trip()

    def _trip(self):
        self._state = CircuitState.OPEN
        logger.warning(f"Circuit breaker opened after {self._failure_count} failures")

    def _reset(self):
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time = None
        logger.info("Circuit breaker reset to closed state")

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "state": self.state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "failure_threshold": self._failure_threshold,
                "reset_timeout": self._reset_timeout,
                "last_failure_time": self._last_failure_time,
            }

    def force_close(self):
        with self._lock:
            self._reset()

    def force_open(self):
        with self._lock:
            self._state = CircuitState.OPEN
            self._last_failure_time = time.time()


class AbstractVectorDB(ABC):
    def __init__(
        self,
        config: VectorStoreConfig,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        self.config = config
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self._is_healthy = True
        self._last_health_check = time.time()
        self._consecutive_failures = 0
        self._consecutive_successes = 0

    @property
    def name(self) -> str:
        return f"{self.config.db_type.value}_{self.config.index_name}"

    @property
    def is_healthy(self) -> bool:
        return self._is_healthy

    @property
    def is_primary(self) -> bool:
        return self.config.primary

    def mark_healthy(self):
        self._is_healthy = True
        self._consecutive_failures = 0
        self._last_health_check = time.time()

    def mark_unhealthy(self):
        self._is_healthy = False
        self._consecutive_failures += 1
        self._consecutive_successes = 0
        if self._consecutive_failures >= 3:
            self.circuit_breaker.record_failure()

    def mark_success(self):
        self._consecutive_successes += 1
        self._consecutive_failures = 0
        if self._consecutive_successes >= 5:
            self.circuit_breaker.record_success()

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def insert(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        batch_size: int = 100,
    ) -> List[str]:
        pass

    @abstractmethod
    def search(
        self,
        query: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        with_vectors: bool = False,
    ) -> List[SearchResult]:
        pass

    @abstractmethod
    def delete(self, ids: List[str], filters: Optional[Dict[str, Any]] = None):
        pass

    @abstractmethod
    def update(
        self,
        ids: List[str],
        vectors: Optional[List[List[float]]] = None,
        payloads: Optional[List[Dict[str, Any]]] = None,
    ):
        pass

    @abstractmethod
    def get(
        self,
        ids: List[str],
        with_vectors: bool = False,
    ) -> List[SearchResult]:
        pass

    @abstractmethod
    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        pass

    @abstractmethod
    def exists(self) -> bool:
        pass

    @abstractmethod
    def delete_collection(self):
        pass

    def health_check(self) -> bool:
        try:
            self._last_health_check = time.time()
            return self.exists()
        except Exception:
            return False


class QdrantVectorDB(AbstractVectorDB):
    def __init__(self, config: VectorStoreConfig, circuit_breaker: Optional[CircuitBreaker] = None):
        super().__init__(config, circuit_breaker)
        self._client = None
        self._collection_name = config.index_name

    def _get_client(self):
        if self._client is None:
            try:
                import qdrant_client
                from qdrant_client import QdrantClient
                from qdrant_client.models import Distance, VectorParams, OptimizersConfigDiff

                if self.config.url:
                    self._client = QdrantClient(url=self.config.url, api_key=self.config.api_key, timeout=self.config.timeout)
                else:
                    self._client = QdrantClient(
                        host=self.config.host,
                        port=self.config.port,
                        timeout=self.config.timeout,
                        ssl=self.config.ssl,
                    )
            except ImportError:
                raise VectorStoreException("qdrant-client not installed. Run: pip install qdrant-client")
        return self._client

    def setup(self):
        client = self._get_client()
        distance_map = {
            "cosine": "Cosine",
            "euclidean": "Euclid",
            "dot": "Dot",
        }
        distance = distance_map.get(self.config.distance_func.lower(), "Cosine")

        try:
            from qdrant_client.models import Distance, VectorParams

            if not self.exists():
                client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(
                        size=self.config.dimension,
                        distance=Distance[distance],
                    ),
                )
                logger.info(f"Created Qdrant collection: {self._collection_name}")
        except Exception as e:
            logger.error(f"Failed to setup Qdrant collection: {e}")
            raise VectorStoreException(f"Failed to setup Qdrant: {e}")

    def insert(self, vectors: List[List[float]], payloads: List[Dict[str, Any]],