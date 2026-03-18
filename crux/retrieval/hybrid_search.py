import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from functools import lru_cache

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class RerankerType(Enum):
    LOCAL = "local"
    API = "api"
    NONE = "none"


class RetrievalMethod(Enum):
    DENSE_ONLY = "dense_only"
    SPARSE_ONLY = "sparse_only"
    HYBRID = "hybrid"


@dataclass
class RerankerConfig:
    reranker_type: RerankerType = RerankerType.NONE
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    api_base_url: Optional[str] = None
    api_key: Optional[str] = None
    device: Optional[str] = None
    max_seq_length: int = 512
    top_k: int = 10


@dataclass
class HybridSearchConfig:
    alpha: float = 0.7
    enable_reranking: bool = True
    reranker_config: RerankerConfig = field(default_factory=RerankerConfig)
    k1: float = 1.5
    b: float = 0.75
    rrf_k: int = 60
    min_score_threshold: float = 0.0
    normalize_scores: bool = True
    retrieval_method: RetrievalMethod = RetrievalMethod.HYBRID


class BM25Indexer:
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        lowercase: bool = True,
        tokenize_method: str = "default"
    ):
        self.k1 = k1
        self.b = b
        self.lowercase = lowercase
        self.tokenize_method = tokenize_method
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_ids: List[str] = []
        self.corpus_texts: List[str] = []
        self._tokenized_corpus: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        if self.tokenize_method == "default":
            import re
            tokens = re.findall(r'\w+', text.lower() if self.lowercase else text)
            return tokens
        elif self.tokenize_method == "whitespace":
            return text.split()
        elif self.tokenize_method == "nltk":
            try:
                from nltk.tokenize import word_tokenize
                return word_tokenize(text.lower() if self.lowercase else text)
            except ImportError:
                logger.warning("NLTK not available, falling back to default tokenization")
                return self._tokenize(text)
        else:
            return self._tokenize(text)

    def index(self, texts: List[str], ids: List[str]) -> None:
        if not texts or not ids:
            raise ValueError("Texts and IDs lists cannot be empty")
        if len(texts) != len(ids):
            raise ValueError("Texts and IDs must have the same length")

        self.corpus_ids = ids
        self.corpus_texts = texts
        self._tokenized_corpus = [self._tokenize(text) for text in texts]
        
        if BM25_AVAILABLE:
            self.bm25 = BM25Okapi(self._tokenized_corpus, k1=self.k1, b=self.b)
        else:
            logger.warning("rank_bm25 not installed. Sparse search will be disabled.")
            self.bm25 = None

    def add_documents(self, texts: List[str], ids: List[str]) -> None:
        start_idx = len(self.corpus_ids)
        self.corpus_ids.extend(ids)
        self.corpus_texts.extend(texts)
        new_tokenized = [self._tokenize(text) for text in texts]
        self._tokenized_corpus.extend(new_tokenized)
        
        if self.bm25 is not None:
            self.bm25.add(new_tokenized)

    def remove_documents(self, ids: List[str]) -> None:
        ids_set = set(ids)
        kept_indices = [i for i, cid in enumerate(self.corpus_ids) if cid not in ids_set]
        
        self.corpus_ids = [self.corpus_ids[i] for i in kept_indices]
        self.corpus_texts = [self.corpus_texts[i] for i in kept_indices]
        self._tokenized_corpus = [self._tokenized_corpus[i] for i in kept_indices]
        
        if self.bm25 is not None and self._tokenized_corpus:
            self.bm25 = BM25Okapi(self._tokenized_corpus, k1=self.k1, b=self.b)
        elif not self._tokenized_corpus:
            self.bm25 = None

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, int]]:
        if self.bm25 is None:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        if not self.normalize_scores:
            top_indices = np.argsort(scores)[::-1][:top_k]
            return [(self.corpus_ids[idx], float(scores[idx]), idx) for idx in top_indices]
        
        if len(scores) == 0 or np.max(scores) == 0:
            return [(self.corpus_ids[i], 0.0, i) for i in range(min(top_k, len(self.corpus_ids)))]
        
        min_score = np.min(scores[scores > 0]) if np.any(scores > 0) else 1e-10
        max_score = np.max(scores)
        if max_score == min_score:
            normalized_scores = np.ones_like(scores) * 0.5
        else:
            normalized_scores = (scores - min_score) / (max_score - min_score)
            normalized_scores = np.clip(normalized_scores, 0.0, 1.0)
        
        top_indices = np.argsort(normalized_scores)[::-1][:top_k]
        return [(self.corpus_ids[idx], float(normalized_scores[idx]), idx) for idx in top_indices]

    def get_scores(self, query: str) -> Dict[str, float]:
        if self.bm25 is None:
            return {}

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        if not self.normalize_scores:
            return {self.corpus_ids[i]: float(scores[i]) for i in range(len(scores))}
        
        max_score = np.max(scores) if np.max(scores) > 0 else 1e-10
        normalized_scores = scores / max_score
        
        return {self.corpus_ids[i]: float(normalized_scores[i]) for i in range(len(scores))}

    def update(self, doc_id: str, text: str) -> None:
        try:
            idx = self.corpus_ids.index(doc_id)
            self.corpus_texts[idx] = text
            self._tokenized_corpus[idx] = self._tokenize(text)
            
            if self.bm25 is not None and self._tokenized_corpus:
                self.bm25 = BM25Okapi(self._tokenized_corpus, k1=self.k1, b=self.b)
        except ValueError:
            self.add_documents([text], [doc_id])

    def clear(self) -> None:
        self.bm25 = None
        self.corpus_ids = []
        self.corpus_texts = []
        self._tokenized_corpus = []


class CrossEncoderReranker:
    def __init__(self, config: RerankerConfig):
        self.config = config
        self.model = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        if self.config.reranker_type == RerankerType.LOCAL and CROSS_ENCODER_AVAILABLE:
            try:
                device = self.config.device
                if device is None:
                    device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
                
                self.model = CrossEncoder(
                    self.config.model_name,
                    max_length=self.config.max_seq_length,
                    device=device
                )
                logger.info(f"Loaded cross-encoder model: {self.config.model_name} on {device}")
            except Exception as e:
                logger.error(f"Failed to load cross-encoder model: {e}")
                self.model = None
        elif self.config.reranker_type == RerankerType.API:
            if not self.config.api_base_url:
                logger.warning("API reranker requested but no api_base_url provided")
            else:
                logger.info(f"Configured API reranker with base URL: {self.config.api_base_url}")

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        if self.config.reranker_type == RerankerType.NONE:
            return documents[:top_k] if top_k else documents

        if not documents:
            return []

        if self.config.reranker_type == RerankerType.LOCAL and self.model is not None:
            return self._rerank_local(query, documents, top_k)
        elif self.config.reranker_type == RerankerType.API:
            return self._rerank_api(query, documents, top_k)
        else:
            logger.warning("Reranker not available, returning original order")
            return documents[:top_k] if top_k else documents

    def _rerank_local(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        try:
            doc_texts = []
            for doc in documents:
                text = doc.get("text", doc.get("content", doc.get("content_text", "")))
                doc_texts.append(text)

            pairs = [(query, doc_text) for doc_text in doc_texts]
            scores = self.model.predict(pairs, show_progress_bar=False)

            if isinstance(scores, np.ndarray):
                scores = scores.tolist()
            
            doc_with_scores = []
            for i, doc in enumerate(documents):
                doc_copy = doc.copy()
                doc_copy["rerank_score"] = float(scores[i]) if i < len(scores) else 0.0
                doc_copy["original_index"] = i
                doc_with_scores.append(doc_copy)

            doc_with_scores.sort(key=lambda x: x["rerank_score"], reverse=True)
            return doc_with_scores[:top_k]

        except Exception as e:
            logger.error(f"Error during local reranking: {e}")
            return documents[:top_k] if top_k else documents

    def _rerank_api(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        try:
            import requests

            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            payload = {
                "query": query,
                "documents": [doc.get("text", doc.get("content", "")) for doc in documents]
            }

            response = requests.post(
                f"{self.config.api_base_url}/rerank",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            results = response.json().get("results", [])
            
            reranked_docs = []
            for result in results[:top_k]:
                doc_copy = documents[result["index"]].copy()
                doc_copy["rerank_score"] = result.get("score", 0.0)
                reranked_docs.append(doc_copy)
            
            return reranked_docs

        except Exception as e:
            logger.error(f"Error during API reranking: {e}")
            return documents[:top_k] if top_k else documents


class ReciprocalRankFusion:
    @staticmethod
    def fuse(
        results_list: List[List[Tuple[str, float, int]]],
        k: int = 60
    ) -> List[Tuple[str, float]]:
        if not results_list:
            return []

        score_map: Dict[str, float] = {}
        rank_map: Dict[str, List[int]] = {}

        for results in results_list:
            for rank, (doc_id, score, _) in enumerate(results):
                if doc_id not in score_map:
                    score_map[doc_id] = 0.0
                    rank_map[doc_id] = []
                
                score_map[doc_id] += score
                rank_map[doc_id].append(rank)

        fused_scores = []
        for doc_id in score_map:
            ranks = rank_map[doc_id]
            rrf_score = sum(1.0 / (k + r + 1) for r in ranks)
            fused_scores.append((doc_id, rrf_score))

        fused_scores.sort(key=lambda x: x[1], reverse=True)
        return fused_scores

    @staticmethod
    def weighted_fuse(
        results_with_alpha: List[Tuple[float, List[Tuple[str, float, int]]]],
        k: int = 60
    ) -> List[Tuple[str, float]]:
        score_map: Dict[str, float] = {}
        rank_map: Dict[str, List[int]] = {}
        weight_map: Dict[str, float] = {}

        for alpha, results in results_with_alpha:
            if results is None:
                continue
            for rank, (doc_id, score, _) in enumerate(results):
                if doc_id not in score_map:
                    score_map[doc_id] = 0.0
                    rank_map[doc_id] = []
                    weight_map[doc_id] = 0.0
                
                rrf_contribution = (1.0 / (k + rank + 1)) * alpha
                score_map[doc_id] += rrf_contribution
                rank_map[doc_id].append(rank)
                weight_map[doc_id] += alpha

        fused_scores = []
        for doc_id in score_map:
            if weight_map[doc_id] > 0:
                normalized_score = score_map[doc_id] / weight_map[doc_id]
            else:
                normalized_score = score_map[doc_id]
            fused_scores.append((doc_id, normalized_score))

        fused_scores.sort(key=lambda x: x[1], reverse=True)
        return fused_scores


@dataclass
class RetrievalResult:
    id: str
    score: float
    rerank_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    text: Optional[str] = None


class HybridSearchRetriever:
    def __init__(
        self,
        vector_store: Any,
        config: Optional[HybridSearchConfig] = None
    ):
        self.vector_store = vector_store
        self.config = config or HybridSearchConfig()
        
        self.bm25_indexer = BM25Indexer(
            k1=self.config.k1,
            b=self.config.b
        )
        
        self.reranker = None
        if self.config.enable_reranking:
            self.reranker = CrossEncoderReranker(self.config.reranker_config)
        
        self._indexed_ids: set = set()
        self._dense_results_cache: Dict[str, List] = {}
        self._sparse_results_cache: Dict[str, List] = {}

    def index_documents(
        self,
        documents: List[Dict[str, Any]],
        namespace: str = "default"
    ) -> None:
        if not documents:
            return

        ids = [doc.get("id", str(i)) for i, doc in enumerate(documents)]
        texts = [doc.get("text", doc.get("content", "")) for doc in documents]
        embeddings = [doc.get("embedding") for doc in documents if "embedding" in doc]

        self.bm25_indexer.index(texts, ids)

        for doc_id in ids:
            self._indexed_ids.add(f"{namespace}:{doc_id}")

        if embeddings and len(embeddings) == len(documents):
            self.vector_store.upsert(
                vectors=[
                    {"id": id_, "vector": emb, "text": text}
                    for id_, emb, text in zip(ids, embeddings, texts)
                ],
                namespace=namespace
            )
        elif hasattr(self.vector_store, "add_texts"):
            self.vector_store.add_texts(texts, ids=ids, namespace=namespace)

    def search(
        self,
        query: str,
        namespace: str = "default",
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        alpha: Optional[float] = None
    ) -> List[RetrievalResult]:
        if self.config.retrieval_method == RetrievalMethod.DENSE_ONLY:
            return self._dense_search(query, namespace, top_k, filters)
        elif self.config.retrieval_method == RetrievalMethod.SPARSE_ONLY:
            return self._sparse_search(query, namespace, top_k)
        else:
            return self._hybrid_search(query, namespace, top_k, filters, alpha)

    def _dense_search(
        self,
        query: str,
        namespace: str,
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[Retrieval