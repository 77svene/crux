# crux/vector_stores/base.py

"""
Abstract base class for vector database backends.
Defines the interface that all vector store implementations must follow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum
import threading
import time
import logging

logger = logging.getLogger(__name__)


class VectorStoreStatus(Enum):
    """Status indicators for vector store health."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class VectorStoreConfig:
    """Configuration for a vector store instance."""
    collection_name: str = "crux_vectors"
    dimension: int = 1536
    distance_metric: str = "cosine"  # cosine, euclidean, dotproduct
    batch_size: int = 100
    max_retries: int = 3
    timeout: float = 30.0
    metadata: dict = field(default_factory=dict)
    
    # Connection settings
    host: Optional[str] = None
    port: Optional[int] = None
    url: Optional[str] = None
    api_key: Optional[str] = None
    index_params: dict = field(default_factory=dict)
    
    # Failover settings
    is_primary: bool = True
    priority: int = 0


@dataclass
class SearchResult:
    """Represents a single search result from the vector store."""
    id: str
    score: float
    vector: Optional[Any] = None
    payload: Optional[dict] = None
    metadata: Optional[dict] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "score": self.score,
            "payload": self.payload,
            "metadata": self.metadata
        }


@dataclass
class SearchResults:
    """Container for multiple search results."""
    results: list[SearchResult]
    query: Optional[Any] = None
    total_count: int = 0
    
    def __post_init__(self):
        self.total_count = len(self.results)
    
    def __len__(self) -> int:
        return len(self.results)
    
    def __iter__(self):
        return iter(self.results)
    
    def to_list(self) -> list[dict]:
        return [r.to_dict() for r in self.results]


@dataclass 
class VectorEntry:
    """Represents a vector entry to be stored."""
    id: str
    vector: list[float]
    payload: Optional[dict] = None
    metadata: Optional[dict] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "vector": self.vector,
            "payload": self.payload or {},
            "metadata": self.metadata or {}
        }


class AbstractVectorDB(ABC):
    """
    Abstract base class defining the vector database interface.
    All concrete implementations must inherit from this class.
    """
    
    def __init__(self, config: VectorStoreConfig):
        """
        Initialize the vector database.
        
        Args:
            config: Configuration object for the vector store
        """
        self.config = config
        self._initialized = False
        self._status = VectorStoreStatus.UNKNOWN
        self._lock = threading.RLock()
        
    @property
    def status(self) -> VectorStoreStatus:
        """Get current status of the vector store."""
        return self._status
    
    @property
    def is_initialized(self) -> bool:
        """Check if the vector store is initialized."""
        return self._initialized
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the vector store connection and create schema.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if the vector store is reachable and healthy.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        pass
    
    @abstractmethod
    def upsert(
        self,
        entries: list[VectorEntry],
        batch_size: Optional[int] = None
    ) -> dict:
        """
        Insert or update vectors in the store.
        
        Args:
            entries: List of VectorEntry objects to upsert
            batch_size: Optional batch size override
            
        Returns:
            dict: Operation result with counts and status
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filters: Optional[dict] = None,
        with_payload: bool = True,
        with_vectors: bool = False
    ) -> SearchResults:
        """
        Search for similar vectors.
        
        Args:
            query_vector: The vector to search for
            limit: Maximum number of results to return
            filters: Optional metadata filters
            with_payload: Include payload in results
            with_vectors: Include vectors in results
            
        Returns:
            SearchResults: Container with search results
        """
        pass
    
    @abstractmethod
    def get(
        self,
        ids: list[str],
        with_payload: bool = True,
        with_vectors: bool = False
    ) -> list[VectorEntry]:
        """
        Retrieve vectors by their IDs.
        
        Args:
            ids: List of vector IDs to retrieve
            with_payload: Include payload in results
            with_vectors: Include vectors in results
            
        Returns:
            list[VectorEntry]: Retrieved vectors
        """
        pass
    
    @abstractmethod
    def delete(
        self,
        ids: list[str],
        filters: Optional[dict] = None
    ) -> bool:
        """
        Delete vectors by ID or filters.
        
        Args:
            ids: List of vector IDs to delete
            filters: Optional filters for deletion
            
        Returns:
            bool: True if deletion successful
        """
        pass
    
    @abstractmethod
    def count(self, filters: Optional[dict] = None) -> int:
        """
        Count vectors in the store.
        
        Args:
            filters: Optional filters for counting
            
        Returns:
            int: Number of vectors matching criteria
        """
        pass
    
    @abstractmethod
    def collection_exists(self) -> bool:
        """Check if the collection exists."""
        pass
    
    @abstractmethod
    def create_collection(self) -> bool:
        """Create the collection if it doesn't exist."""
        pass
    
    @abstractmethod
    def delete_collection(self) -> bool:
        """Delete the collection and all its data."""
        pass
    
    @abstractmethod
    def get_collections(self) -> list[str]:
        """List all collections."""
        pass
    
    def close(self):
        """Clean up resources. Override in subclasses as needed."""
        with self._lock:
            self._initialized = False
            self._status = VectorStoreStatus.UNKNOWN
    
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# crux/vector_stores/qdrant.py

"""
Qdrant vector store implementation.
High-performance vector search engine with gRPC support.
"""

import json
import hashlib
from typing import Optional, Any
import threading

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.exceptions import UnexpectedResponse
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    models = None

from .base import (
    AbstractVectorDB, VectorStoreConfig, VectorStoreStatus,
    SearchResult, SearchResults, VectorEntry
)

logger = logging.getLogger(__name__)


class QdrantVectorDB(AbstractVectorDB):
    """
    Qdrant vector database implementation.
    Supports local and cloud deployments with automatic schema management.
    """
    
    DISTANCE_MAP = {
        "cosine": Distance.COSINE,
        "euclidean": Distance.EUCLID,
        "dotproduct": Distance.DOT,
    }
    
    def __init__(self, config: VectorStoreConfig):
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client is required. Install with: pip install qdrant-client"
            )
        super().__init__(config)
        self._client: Optional[QdrantClient] = None
        self._semaphore = threading.Semaphore(10)
        
    def _get_client(self) -> QdrantClient:
        """Get or create the Qdrant client."""
        if self._client is None:
            if self.config.url:
                self._client = QdrantClient(
                    url=self.config.url,
                    api_key=self.config.api_key,
                    timeout=self.config.timeout,
                    prefer_grpc=True,
                )
            elif self.config.host and self.config.port:
                self._client = QdrantClient(
                    host=self.config.host,
                    port=self.config.port,
                    timeout=self.config.timeout,
                    prefer_grpc=True,
                )
            else:
                # Local mode
                self._client = QdrantClient(":memory:")
        return self._client
    
    def initialize(self) -> bool:
        """Initialize connection and create collection if needed."""
        with self._lock:
            try:
                client = self._get_client()
                
                if not self.collection_exists():
                    self.create_collection()
                
                self._initialized = True
                self._status = VectorStoreStatus.HEALTHY
                logger.info(f"Qdrant initialized: {self.config.collection_name}")
                return True
                
            except Exception as e:
                logger.error(f"Qdrant initialization failed: {e}")
                self._status = VectorStoreStatus.UNHEALTHY
                self._initialized = False
                return False
    
    def health_check(self) -> bool:
        """Check Qdrant server health."""
        try:
            client = self._get_client()
            collections = client.get_collections()
            self._status = VectorStoreStatus.HEALTHY
            return True
        except Exception as e:
            logger.warning(f"Qdrant health check failed: {e}")
            self._status = VectorStoreStatus.UNHEALTHY
            return False
    
    def collection_exists(self) -> bool:
        """Check if collection exists in Qdrant."""
        try:
            client = self._get_client()
            client.get_collection(self.config.collection_name)
            return True
        except Exception:
            return False
    
    def create_collection(self) -> bool:
        """Create collection with specified configuration."""
        try:
            client = self._get_client()
            distance = self.DISTANCE_MAP.get(
                self.config.distance_metric, 
                Distance.COSINE
            )
            
            client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.config.dimension,
                    distance=distance,
                    on_disk=self.config.metadata.get("on_disk", False),
                ),
                optimizers_config=models.OptimizersConfig(
                    indexing_threshold=self.config.metadata.get(
                        "indexing_threshold", 20000
                    ),
                    memmap_threshold=self.config.metadata.get(
                        "memmap_threshold", 50000
                    ),
                ),
                # shard_number=self.config.metadata.get("shards", 1),
                # replication_factor=self.config.metadata.get("replication_factor", 1),
            )
            logger.info(f"Created Qdrant collection: {self.config.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False
    
    def delete_collection(self) -> bool:
        """Delete the collection and all data."""
        try:
            client = self._get_client()
            client.delete_collection(self.config.collection_name)
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False
    
    def get_collections(self) -> list[str]:
        """List all collections."""
        try:
            client = self._get_client()
            return [c.name for c in client.get_collections().collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    def upsert(
        self,
        entries: list[VectorEntry],
        batch_size: Optional[int] = None
    ) -> dict:
        """
        Upsert vectors in batches to Qdrant.
        
        Args:
            entries: List of VectorEntry objects
            batch_size: Override batch size for this operation
            
        Returns:
            dict: Operation result with counts
        """
        batch_size = batch_size or self.config.batch_size
        
        with self._semaphore:
            total_inserted = 0
            total_updated = 0
            failed = 0
            
            try:
                client = self._get_client()
                
                for i in range(0, len(entries), batch_size):
                    batch = entries[i:i + batch_size]
                    points = []
                    
                    for entry in batch:
                        try:
                            payload = entry.payload or {}
                            if entry.metadata:
                                payload["_metadata"] = entry.metadata
                            
                            point = PointStruct(
                                id=entry.id,
                                vector=entry.vector,
                                payload=payload,
                            )
                            points.append(point)
                            
                        except Exception as e:
                            logger.warning(f"Failed to prepare point {entry.id}: {e}")
                            failed += 1
                    
                    if points:
                        operation_info = client.upsert(
                            collection_name=self.config.collection_name,
                            points=points,
                            wait=True,
                        )
                        
                        if operation_info.status == "completed":
                            total_inserted += len(points)
                        elif operation_info.status == "updated":
                            total_updated += len(points)
                        else:
                            failed += len(points)
                
                return {
                    "status": "success",
                    "inserted": total_inserted,
                    "updated": total_updated,
                    "failed": failed,
                    "total": len(entries),
                }
                
            except Exception as e:
                logger.error(f"Qdrant upsert failed: {e}")
                return {
                    "status": "error",
                    "error": str(e),
                    "inserted": total_inserted,
                    "failed": failed + (len(entries) - total_inserted),
                    "total": len(entries),
                }
    
    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filters: Optional[dict] = None,
        with_payload: bool = True,
        with_vectors: bool = False
    ) -> SearchResults:
        """
        Search for similar vectors in Qdrant.
        
        Args:
            query_vector: Query vector to search for
            limit: Maximum results to return
            filters: Metadata filters (converted to Qdrant filter)
            with_payload: Include payload in results
            with_vectors: Include vectors in results
            
        Returns:
            SearchResults: Container with search results
        """
        try:
            client = self._get_client()
            
            qdrant_filter = self._build_filter(filters) if filters else None
            
            search_params = models.SearchParams(
                hnsw_ef=128,
                exact=False,
            )
            
            results = client.search(
                collection_name=self.config.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=qdrant_filter,
                with_payload=with_payload,
                with_vectors=with_vectors,
                search_params=search_params,
            )
            
            search_results = []
            for hit in results:
                metadata = None
                payload = hit.payload or {}
                if "_metadata" in payload:
                    metadata = payload.pop("_metadata")
                
                search_results.append(SearchResult(
                    id=str(hit.id),
                    score=hit.score,
                    vector=hit.vector if with_vectors else None,
                    payload=payload,
                    metadata=metadata,
                ))
            
            return SearchResults(
                results=search_results,
                query=query_vector,
            )
            
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return SearchResults(results=[], query=query_vector)
    
    def get(
        self,
        ids: list[str],
        with_payload: bool = True,
        with_vectors: bool = False
    ) -> list[VectorEntry]:
        """
        Retrieve vectors by ID.
        
        Args:
            ids: List of vector IDs
            with_payload: Include payload
            with_vectors: Include vectors
            
        Returns:
            list[VectorEntry]: Retrieved entries
        """
        try:
            client = self._get_client()
            
            results = client.retrieve(
                collection_name=self.config.collection_name,
                ids=ids,
                with_payload=with_payload,
                with_vectors=with_vectors,
            )
            
            entries = []
            for record in results:
                payload = record.payload or {}
                metadata = None
                if "_metadata" in payload:
                    metadata = payload.pop("_metadata")
                
                entries.append(VectorEntry(
                    id=str(record.id),
                    vector=record.vector if with_vectors else record.vector,
                    payload=payload,
                    metadata=metadata,
                ))
            
            return entries
            
        except Exception as e:
            logger.error(f"Qdrant get failed: {e}")
            return []
    
    def delete(
        self,
        ids: list[str],
        filters: Optional[dict] = None
    ) -> bool:
        """
        Delete vectors by ID or filters.
        
        Args:
            ids: Vector IDs to delete
            filters: Alternative filter-based deletion
            
        Returns:
            bool: True if successful
        """
        try:
            client = self._get_client()
            
            if ids:
                client.delete(
                    collection_name=self.config.collection_name,
                    points_selector=models.PointIdsList(
                        points=ids,
                    ),
                    wait=True,
                )
            elif filters:
                qdrant_filter = self._build_filter(filters)
                client.delete(
                    collection_name=self.config.collection_name,
                    points_selector=models.FilterSelector(
                        filter=qdrant_filter,
                    ),
                    wait=True,
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Qdrant delete failed: {e}")
            return False
    
    def count(self, filters: Optional[dict] = None) -> int:
        """
        Count vectors in collection.
        
        Args:
            filters: Optional filters
            
        Returns: