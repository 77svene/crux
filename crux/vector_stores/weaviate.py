import os
import time
import logging
import threading
import queue
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import hashlib

try:
    import weaviate
    from weaviate.exceptions import WeaviateConnectionError, WeaviateQueryError
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False

try:
    import qdrant_client
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

try:
    import psycopg2
    import numpy as np
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False

try:
    import pinecone
    from pinecone import Pinecone, ServerlessSpecification
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

from .exceptions import VectorDBError, VectorDBConnectionError, VectorDBTimeoutError
from .config import VectorDBConfig

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout: float = 30.0
    half_open_max_calls: int = 1
    exponential_base: float = 2.0
    max_backoff: float = 60.0


class CircuitBreaker:
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = threading.RLock()
        self._state_listeners: List[Callable[[str, CircuitState, CircuitState], None]] = []

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to(CircuitState.HALF_OPEN)
            return self._state

    def _should_attempt_reset(self) -> bool:
        if self._last_failure_time is None:
            return True
        backoff = min(
            self.config.timeout * (self.config.exponential_base ** (self._failure_count - 1)),
            self.config.max_backoff
        )
        return time.time() - self._last_failure_time >= backoff

    def _transition_to(self, new_state: CircuitState):
        old_state = self._state
        self._state = new_state
        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
        elif new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
        for listener in self._state_listeners:
            try:
                listener(self.name, old_state, new_state)
            except Exception as e:
                logger.error(f"Circuit breaker state listener error: {e}")

    def add_state_listener(self, listener: Callable[[str, CircuitState, CircuitState], None]):
        self._state_listeners.append(listener)

    def record_success(self):
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    def record_failure(self, exception: Optional[Exception] = None):
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.OPEN:
                backoff = min(
                    self.config.timeout * (self.config.exponential_base ** max(0, self._failure_count - 1)),
                    self.config.max_backoff
                )
                self._last_failure_time = time.time() - self.config.timeout + backoff

    def allow_request(self) -> bool:
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            if self._state == CircuitState.OPEN:
                return False
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            return False

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            if not self.allow_request():
                raise VectorDBConnectionError(
                    f"Circuit breaker '{self.name}' is OPEN. Primary vector store unavailable."
                )
            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure(e)
                raise
        return wrapper


class VectorSearchResult:
    def __init__(
        self,
        id: str,
        score: float,
        payload: Dict[str, Any],
        vector: Optional[List[float]] = None
    ):
        self.id = id
        self.score = score
        self.payload = payload
        self.vector = vector

    def __repr__(self):
        return f"VectorSearchResult(id={self.id}, score={self.score:.4f})"


class AbstractVectorDB(ABC):
    def __init__(self, config: VectorDBConfig):
        self.config = config
        self._initialized = False
        self._circuit_breaker = CircuitBreaker(
            name=f"{self.__class__.__name__}_cb",
            config=config.circuit_breaker_config
        )

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def disconnect(self) -> None:
        pass

    @abstractmethod
    def health_check(self) -> bool:
        pass

    @abstractmethod
    def upsert(
        self,
        vectors: Dict[str, List[float]],
        payloads: Optional[Dict[str, Dict[str, Any]]] = None,
        namespace: Optional[str] = None
    ) -> bool:
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        namespace: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        include_vectors: bool = False
    ) -> List[VectorSearchResult]:
        pass

    @abstractmethod
    def search_by_text(
        self,
        text: str,
        top_k: int = 10,
        namespace: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        embed_func: Optional[Callable[[str], List[float]]] = None
    ) -> List[VectorSearchResult]:
        pass

    @abstractmethod
    def delete(
        self,
        ids: Optional[List[str]] = None,
        delete_all: bool = False,
        namespace: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> bool:
        pass

    @abstractmethod
    def get_by_id(
        self,
        ids: List[str],
        namespace: Optional[str] = None,
        include_vectors: bool = False
    ) -> Dict[str, VectorSearchResult]:
        pass

    @abstractmethod
    def count(self, namespace: Optional[str] = None) -> int:
        pass

    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        pass

    @abstractmethod
    def create_collection(
        self,
        collection_name: str,
        vector_dim: int,
        distance_metric: str = "cosine",
        **kwargs
    ) -> bool:
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str) -> bool:
        pass

    def _generate_id(self, text: str, namespace: Optional[str] = None) -> str:
        prefix = f"{namespace}:" if namespace else ""
        return f"{prefix}{hashlib.sha256(text.encode()).hexdigest()[:16]}"

    def _with_circuit_breaker(self, func, *args, **kwargs):
        if not self._circuit_breaker.allow_request():
            raise VectorDBConnectionError(
                f"Circuit breaker open for {self.__class__.__name__}"
            )
        try:
            result = func(*args, **kwargs)
            self._circuit_breaker.record_success()
            return result
        except Exception as e:
            self._circuit_breaker.record_failure(e)
            raise


class WeaviateAdapter(AbstractVectorDB):
    def __init__(self, config: VectorDBConfig):
        super().__init__(config)
        self._client = None
        self._async_client = None

    def initialize(self) -> None:
        if not WEAVIATE_AVAILABLE:
            raise VectorDBError("weaviate package not installed. Install with: pip install weaviate-client")
        
        auth_config = None
        if self.config.api_key:
            if self.config.use_embedded:
                auth_config = weaviate.AuthApiKey(self.config.api_key)
            else:
                from weaviate.auth import AuthClientPassword
                if hasattr(self.config, 'username') and hasattr(self.config, 'password'):
                    auth_config = AuthClientPassword(
                        self.config.username,
                        self.config.password
                    )

        connection_params = weaviate.ConnectionParams(
            url=self.config.url,
            timeout_config=(self.config.timeout, self.config.timeout),
            headers={"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else None
        )

        if self.config.use_embedded:
            self._client = weaviate.Client(
                embedded_options=weaviate.EmbeddedOptions(
                    persistence_data_path=self.config.local_data_path or "./weaviate_data"
                ),
                auth_client_secret=auth_config,
                timeout_config=(self.config.timeout, self.config.timeout)
            )
        else:
            self._client = weaviate.Client(
                connection_params=connection_params,
                auth_client_secret=auth_config,
                timeout_config=(self.config.timeout, self.config.timeout)
            )

        self._initialized = True
        logger.info(f"Weaviate client initialized for {self.config.url}")

    def connect(self) -> bool:
        if not self._initialized:
            self.initialize()
        try:
            meta = self._client.get_meta()
            logger.info(f"Weaviate connected: version {meta.get('version', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Weaviate connection failed: {e}")
            raise VectorDBConnectionError(f"Failed to connect to Weaviate: {e}")

    def disconnect(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
        self._initialized = False

    def health_check(self) -> bool:
        try:
            if not self._client:
                return False
            self._client.get_meta()
            return True
        except Exception:
            return False

    def _get_collection_name(self, namespace: Optional[str] = None) -> str:
        base = self.config.collection_name or "crux_vectors"
        return f"{base}_{namespace}" if namespace else base

    def upsert(
        self,
        vectors: Dict[str, List[float]],
        payloads: Optional[Dict[str, Dict[str, Any]]] = None,
        namespace: Optional[str] = None
    ) -> bool:
        collection_name = self._get_collection_name(namespace)
        
        if not self._client.schema.exists(collection_name):
            sample_vector = next(iter(vectors.values()))
            self.create_collection(collection_name, len(sample_vector))

        data_objects = []
        for vector_id, vector in vectors.items():
            payload = (payloads.get(vector_id, {}) if payloads else {}).copy()
            payload["_id"] = vector_id
            data_objects.append({
                "uuid": vector_id,
                "vector": vector,
                "properties": payload
            })

        with self._client.batch as batch:
            for obj in data_objects:
                batch.add_object(
                    collection=collection_name,
                    uuid=obj["uuid"],
                    vector=obj["vector"],
                    properties=obj["properties"]
                )

        logger.info(f"Upserted {len(vectors)} vectors to {collection_name}")
        return True

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        namespace: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        include_vectors: bool = False
    ) -> List[VectorSearchResult]:
        collection_name = self._get_collection_name(namespace)
        
        where_filter = None
        if filters:
            where_filter = self._convert_filters_to_weaviate(filters)

        try:
            response = self._client.query.get(
                collection_name,
                properties=["*"] + (["vector"] if include_vectors else [])
            ).with_near_vector({
                "vector": query_vector
            }).with_limit(top_k).with_additional(["id", "vector", "distance"]).do()

            results = []
            if response and "data" in response and "Get" in response["data"]:
                items = response["data"]["Get"].get(collection_name, [])
                for item in items:
                    vector_data = item.get("_additional", {})
                    results.append(VectorSearchResult(
                        id=vector_data.get("id", ""),
                        score=1.0 - vector_data.get("distance", 1.0),
                        payload={k: v for k, v in item.items() if k != "_additional"},
                        vector=vector_data.get("vector") if include_vectors else None
                    ))
            
            return results
        except Exception as e:
            logger.error(f"Weaviate search failed: {e}")
            raise VectorDBError(f"Search failed: {e}")

    def _convert_filters_to_weaviate(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        if "operator" in filters and "operands" in filters:
            return {
                "operator": filters["operator"],
                "operands": [self._convert_filters_to_weaviate(op) for op in filters["operands"]]
            }
        return {
            "path": [filters["field"]],
            "operator": filters.get("operator", "Equal"),
            "valueText": filters.get("value", filters.get("valueText"))
        }

    def search_by_text(
        self,
        text: str,
        top_k: int = 10,
        namespace: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        embed_func: Optional[Callable[[str], List[float]]] = None
    ) -> List[VectorSearchResult]:
        if embed_func is None:
            raise VectorDBError("embed_func required for text search")
        query_vector = embed_func(text)
        return self.search(query_vector, top_k, namespace, filters)

    def delete(
        self,
        ids: Optional[List[str]] = None,
        delete_all: bool = False,
        namespace: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> bool:
        collection_name = self._get_collection_name(namespace)
        
        if delete_all:
            self._client.schema.delete_class(collection_name)
            return True
        
        if ids:
            for vector_id in ids:
                try:
                    self._client.data_object.delete(uuid=vector_id, class_name=collection_name)
                except Exception as e:
                    logger.warning(f"Failed to delete {vector_id}: {e}")
        
        return True

    def get_by_id(
        self,
        ids: List[str],
        namespace: Optional[str] = None,
        include_vectors: bool = False
    ) -> Dict[str, VectorSearchResult]:
        collection_name = self._get_collection_name(namespace)
        results = {}
        
        for vector_id in ids:
            try:
                obj = self._client.data_object.get_by_id(
                    vector_id,
                    class_name=collection_name,
                    with_vector=include_vectors
                )
                if obj:
                    results[vector_id] = VectorSearchResult(
                        id=vector_id,
                        score=1.0,
                        payload=obj.get("properties", {}),
                        vector=obj.get("vector")
                    )
            except Exception as e:
                logger.warning(f"Failed to get {vector_id}: {e}")
        
        return results

    def count(self, namespace: Optional[str] = None) -> int:
        collection_name = self._get_collection_name(namespace)
        try:
            count = self._client.query.aggregate(collection_name).with_meta_count().do()
            return count.get("data", {}).get("Aggregate", {}).get(collection_name, [{}])[0].get("meta", {}).get("count", 0)
        except Exception:
            return 0

    def collection_exists(self, collection_name: str) -> bool:
        return self._client.schema.exists(collection_name)

    def create_collection(
        self,
        collection_name: str,
        vector_dim: int,
        distance_metric: str = "cosine",
        **kwargs
    ) -> bool:
        distance_map = {
            "cosine": "cosine",
            "dot": "dot",
            "l2": "l2-squared",
            "manhattan": "manhattan"
        }
        
        class_obj = {
            "class": collection_name,
            "vectorizer": kwargs.get("vectorizer", "none"),
            "vectorIndex