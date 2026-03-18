import os
import logging
from typing import Optional, List, Dict, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from threading import Lock

logger = logging.getLogger(__name__)


class RerankerType(Enum):
    NONE = "none"
    LOCAL = "local"
    API = "api"
    CUSTOM = "custom"


@dataclass
class RerankerConfig:
    type: RerankerType = RerankerType.NONE
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cpu"
    batch_size: int = 32
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    top_k: int = 10
    custom_fn: Optional[Callable] = None
    
    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = RerankerType(self.type.lower())


@dataclass
class HybridRetrievalConfig:
    enabled: bool = True
    alpha: float = 0.7
    bm25_tokenizer: str = "default"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    rrf_k: int = 60
    dense_top_k: int = 50
    sparse_top_k: int = 50
    final_top_k: int = 10
    reranker: Optional[RerankerConfig] = None
    
    def __post_init__(self):
        if self.alpha < 0 or self.alpha > 1:
            raise ValueError(f"alpha must be between 0 and 1, got {self.alpha}")
        if self.reranker is None:
            self.reranker = RerankerConfig()
        elif isinstance(self.reranker, dict):
            self.reranker = RerankerConfig(**self.reranker)


class BM25Indexer:
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: str = "default"
    ):
        self.k1 = k1
        self.b = b
        self.tokenizer_name = tokenizer
        self._index = None
        self._doc_lengths = []
        self._avgdl = 0
        self._doc_freqs = []
        self._idf = {}
        self._doc_count = 0
        self._corpus_size = 0
        self._lock = Lock()
        self._tokenized_docs = []
        
    def _tokenize(self, text: str) -> List[str]:
        if self.tokenizer_name == "default":
            import re
            text = text.lower()
            tokens = re.findall(r'\b\w+\b', text)
            return tokens
        elif self.tokenizer_name == "whitespace":
            return text.lower().split()
        elif self.tokenizer_name == "nltk":
            try:
                from nltk.tokenize import word_tokenize
                from nltk.corpus import stopwords
                tokens = word_tokenize(text.lower())
                stop_words = set(stopwords.words('english'))
                return [t for t in tokens if t.isalnum() and t not in stop_words]
            except ImportError:
                logger.warning("NLTK not available, falling back to default tokenizer")
                return self._tokenize(text)
        else:
            return text.lower().split()
    
    def _calculate_idf(self, doc_freqs: List[Dict[str, int]], doc_count: int) -> Dict[str, float]:
        idf = {}
        for freq_dict in doc_freqs:
            for doc_term, freq in freq_dict.items():
                n_docs_with_term = freq
                idf[doc_term] = idf.get(doc_term, 0) + 1
        
        for term in idf:
            idf[term] = np.log((doc_count - idf[term] + 0.5) / (idf[term] + 0.5) + 1)
        return idf
    
    def fit(self, documents: List[str]) -> "BM25Indexer":
        with self._lock:
            self._tokenized_docs = [self._tokenize(doc) for doc in documents]
            self._doc_count = len(documents)
            self._corpus_size = self._doc_count
            
            self._doc_freqs = []
            self._doc_lengths = []
            
            for doc_tokens in self._tokenized_docs:
                self._doc_lengths.append(len(doc_tokens))
                freq_dict = {}
                for token in doc_tokens:
                    freq_dict[token] = freq_dict.get(token, 0) + 1
                self._doc_freqs.append(freq_dict)
            
            self._avgdl = sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 0
            self._idf = self._calculate_idf(self._doc_freqs, self._doc_count)
            
        return self
    
    def get_scores(self, query: str) -> np.ndarray:
        query_tokens = self._tokenize(query)
        scores = np.zeros(self._doc_count)
        
        for q_term in query_tokens:
            if q_term not in self._idf:
                continue
            
            q_idf = self._idf[q_term]
            
            for i, doc_tokens in enumerate(self._tokenized_docs):
                if q_term in self._doc_freqs[i]:
                    tf = self._doc_freqs[i][q_term]
                    doc_len = self._doc_lengths[i]
                    
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self._avgdl)
                    
                    scores[i] += q_idf * (numerator / denominator)
        
        return scores
    
    def get_top_k(self, query: str, k: int = 10) -> List[tuple]:
        scores = self.get_scores(query)
        top_indices = np.argsort(scores)[::-1][:k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        batch_size: int = 32
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None
        self._initialized = False
        self._lock = Lock()
        
    def _initialize(self):
        if self._initialized:
            return
            
        with self._lock:
            if self._initialized:
                return
                
            try:
                from sentence_transformers import CrossEncoder as STCrossEncoder
                self._model = STCrossEncoder(self.model_name, device=self.device)
                self._initialized = True
                logger.info(f"Loaded cross-encoder model: {self.model_name}")
            except ImportError:
                logger.warning("sentence-transformers not installed. Using fallback scoring.")
                self._initialized = True
    
    def score(self, query: str, documents: List[str]) -> np.ndarray:
        self._initialize()
        
        if self._model is None:
            return self._fallback_score(query, documents)
        
        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs, batch_size=self.batch_size)
        
        if isinstance(scores, np.ndarray):
            return scores
        return np.array(scores)
    
    def _fallback_score(self, query: str, documents: List[str]) -> np.ndarray:
        query_tokens = set(query.lower().split())
        scores = []
        
        for doc in documents:
            doc_tokens = set(doc.lower().split())
            overlap = len(query_tokens & doc_tokens)
            union = len(query_tokens | doc_tokens)
            score = overlap / union if union > 0 else 0
            scores.append(score)
        
        return np.array(scores)
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10,
        return_scores: bool = True
    ) -> Union[List[Dict[str, Any]], List[str]]:
        if not documents:
            return [] if return_scores else []
        
        scores = self.score(query, documents)
        indices = np.argsort(scores)[::-1][:top_k]
        
        if return_scores:
            return [
                {
                    "document": documents[int(idx)],
                    "score": float(scores[int(idx)]),
                    "original_index": int(idx)
                }
                for idx in indices
            ]
        else:
            return [documents[int(idx)] for idx in indices]


class ReciprocalRankFusion:
    @staticmethod
    def fuse(
        dense_results: List[tuple],
        sparse_results: List[tuple],
        k: int = 60,
        alpha: float = 0.7
    ) -> List[Dict[str, Any]]:
        if not dense_results and not sparse_results:
            return []
        
        scores = {}
        doc_ids = set()
        
        for rank, (doc_id, dense_score) in enumerate(dense_results, 1):
            rrf_score = 1 / (k + rank)
            weighted_score = alpha * rrf_score
            scores[doc_id] = scores.get(doc_id, 0) + weighted_score
            doc_ids.add(doc_id)
        
        for rank, (doc_id, sparse_score) in enumerate(sparse_results, 1):
            rrf_score = 1 / (k + rank)
            weighted_score = (1 - alpha) * rrf_score
            scores[doc_id] = scores.get(doc_id, 0) + weighted_score
            doc_ids.add(doc_id)
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        fused_results = []
        for doc_id, combined_score in ranked:
            original_dense = next(
                (s for d, s in dense_results if d == doc_id), None
            )
            original_sparse = next(
                (s for d, s in sparse_results if d == doc_id), None
            )
            
            fused_results.append({
                "id": doc_id,
                "combined_score": combined_score,
                "dense_score": original_dense,
                "sparse_score": original_sparse,
                "dense_rank": next(
                    (i + 1 for i, (d, _) in enumerate(dense_results) if d == doc_id), None
                ),
                "sparse_rank": next(
                    (i + 1 for i, (d, _) in enumerate(sparse_results) if d == doc_id), None
                )
            })
        
        return fused_results


class APICrossEncoderReranker:
    def __init__(self, api_url: str, api_key: Optional[str] = None, batch_size: int = 32):
        self.api_url = api_url
        self.api_key = api_key
        self.batch_size = batch_size
        
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        try:
            import requests
            
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            payload = {
                "query": query,
                "documents": documents,
                "top_k": top_k
            }
            
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            return response.json().get("results", [])
            
        except Exception as e:
            logger.error(f"API reranking failed: {e}")
            return [
                {"document": doc, "score": 0.0}
                for doc in documents[:top_k]
            ]


class HybridRetriever:
    def __init__(
        self,
        vector_store: Any,
        embedding_model: Any,
        config: Optional[HybridRetrievalConfig] = None,
        memory_id: Optional[str] = None
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.config = config or HybridRetrievalConfig()
        self.memory_id = memory_id or "default"
        
        self.bm25_indexer = BM25Indexer(
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
            tokenizer=self.config.bm25_tokenizer
        )
        
        self._reranker = None
        self._bm25_documents = {}
        self._document_cache = {}
        self._indexed = False
        self._lock = Lock()
        
    @property
    def reranker(self):
        if self._reranker is None:
            self._initialize_reranker()
        return self._reranker
    
    def _initialize_reranker(self):
        reranker_config = self.config.reranker
        
        if reranker_config.type == RerankerType.LOCAL:
            self._reranker = CrossEncoderReranker(
                model_name=reranker_config.model_name,
                device=reranker_config.device,
                batch_size=reranker_config.batch_size
            )
        elif reranker_config.type == RerankerType.API:
            if reranker_config.api_url:
                self._reranker = APICrossEncoderReranker(
                    api_url=reranker_config.api_url,
                    api_key=reranker_config.api_key,
                    batch_size=reranker_config.batch_size
                )
            else:
                logger.warning("API reranker configured but no URL provided")
        elif reranker_config.type == RerankerType.CUSTOM:
            if reranker_config.custom_fn:
                self._reranker = reranker_config.custom_fn
            else:
                logger.warning("Custom reranker configured but no function provided")
        else:
            self._reranker = None
    
    def index_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> "HybridRetriever":
        with self._lock:
            if not documents:
                return self
            
            metadata = metadata or [{}] * len(documents)
            
            self._document_cache = {
                str(i): {"text": doc, "metadata": meta}
                for i, (doc, meta) in enumerate(zip(documents, metadata))
            }
            
            self.bm25_indexer.fit(documents)
            self._indexed = True
            
            logger.info(f"Indexed {len(documents)} documents for hybrid retrieval")
        
        return self
    
    def _get_dense_results(
        self,
        query: str,
        top_k: int
    ) -> List[tuple]:
        try:
            query_embedding = self.embedding_model.embed(query)
            
            namespace = f"bm25_{self.memory_id}"
            
            if hasattr(self.vector_store, 'similarity_search_with_score'):
                results = self.vector_store.similarity_search_with_score(
                    query,
                    k=top_k,
                    namespace=namespace
                )
                return [(str(i), score) for i, (_, score) in enumerate(results) if score is not None]
            
            elif hasattr(self.vector_store, 'get'):
                all_results = self.vector_store.get(namespace=namespace)
                if all_results and "vectors" in all_results:
                    vectors = all_results["vectors"]
                    embeddings = [v.get("embedding") for v in vectors.values() if "embedding" in v]
                    
                    if embeddings and query_embedding is not None:
                        scores = self._compute_similarities(query_embedding, embeddings)
                        indices = np.argsort(scores)[::-1][:top_k]
                        return [(str(i), float(scores[i])) for i in indices]
            
            return []
            
        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            return []
    
    def _compute_similarities(
        self,
        query_embedding: Union[List[float], np.ndarray],
        document_embeddings: List[Union[List[float], np.ndarray]]
    ) -> np.ndarray:
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)
        
        doc_embeds = []
        for emb in document_embeddings:
            if isinstance(emb, list):
                doc_embeds.append(np.array(emb))
            else:
                doc_embeds.append(emb)
        
        doc_embeds = np.array(doc_embeds)
        
        query_norm = np.linalg.norm(query_embedding)
        doc_norms = np.linalg.norm(doc_embeds, axis=1)
        
        similarities = np.dot(doc_embeds, query_embedding) / (doc_norms * query_norm + 1e-8)
        
        return similarities
    
    def _get_sparse_results(
        self,
        query: str,
        top_k: int
    ) -> List[tuple]:
        if not self._indexed:
            return []
        
        try:
            return self.bm25_indexer.get_top_k(query, top_k)
        except Exception as e:
            logger.error(f"Sparse retrieval failed: {e}")
            return []
    
    def _fuse_and_rank(
        self,
        dense_results: List[tuple],
        sparse_results: List[tuple]
    ) -> List[Dict[str, Any]]:
        fused = ReciprocalRankFusion.fuse(
            dense_results=dense_results,
            sparse_results=sparse_results,
            k=self.config.rrf_k,
            alpha=self.config.alpha
        )
        return fused
    
    def _apply_reranking(
        self,
        query: str,
        fused_results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        if self.reranker is None:
            return fused_results[:top_k]
        
        if not fused_results:
            return []
        
        doc_ids = [r["id"] for r in fused_results]
        documents = []
        
        for doc_id in