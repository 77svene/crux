import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import hashlib
import json
from functools import lru_cache
import warnings

try:
    from rank_bm25 import BM25Okapi, BM25Plus, BM25L
except ImportError:
    raise ImportError("rank_bm25 is required. Install with: pip install rank-bm25")

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)


class BM25Variant(Enum):
    OKAPI = "okapi"
    PLUS = "plus"
    L = "l"


class RerankerType(Enum):
    CROSS_ENCODER = "cross_encoder"
    COHERE = "cohere"
    OPENAI = "openai"
    NONE = "none"


@dataclass
class RetrievalConfig:
    alpha: float = 0.7
    use_reranker: bool = True
    reranker_type: RerankerType = RerankerType.CROSS_ENCODER
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 20
    final_top_k: int = 10
    bm25_variant: BM25Variant = BM25Variant.OKAPI
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    rrf_k: int = 60
    vector_weight: float = 0.7
    sparse_weight: float = 0.3
    cohere_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    device: Optional[str] = None
    batch_size: int = 32
    max_seq_length: int = 512
    show_progress: bool = False


class BM25SparseEncoder:
    def __init__(
        self,
        variant: BM25Variant = BM25Variant.OKAPI,
        k1: float = 1.5,
        b: float = 0.75,
        show_progress: bool = False
    ):
        self.variant = variant
        self.k1 = k1
        self.b = b
        self.show_progress = show_progress
        self.bm25 = None
        self.corpus_tokens = []
        self.doc_ids = []
        self.doc_id_to_index = {}
        self._is_fitted = False

    def fit(self, documents: List[Dict[str, Any]], text_field: str = "text") -> "BM25SparseEncoder":
        if not documents:
            logger.warning("No documents provided to BM25 encoder")
            return self

        self.corpus_tokens = []
        self.doc_ids = []
        self.doc_id_to_index = {}

        for idx, doc in enumerate(documents):
            doc_id = doc.get("id", self._generate_doc_id(doc, idx))
            text = doc.get(text_field, "")
            
            if not text:
                logger.warning(f"Document {doc_id} has empty text field")
                continue

            tokens = self._tokenize(text)
            self.corpus_tokens.append(tokens)
            self.doc_ids.append(doc_id)
            self.doc_id_to_index[doc_id] = idx

        if not self.corpus_tokens:
            logger.warning("No valid documents after filtering")
            return self

        if self.variant == BM25Variant.OKAPI:
            self.bm25 = BM25Okapi(self.corpus_tokens, k1=self.k1, b=self.b)
        elif self.variant == BM25Variant.PLUS:
            self.bm25 = BM25Plus(self.corpus_tokens, k1=self.k1, b=self.b)
        elif self.variant == BM25Variant.L:
            self.bm25 = BM25L(self.corpus_tokens, k1=self.k1, b=self.b)
        else:
            self.bm25 = BM25Okapi(self.corpus_tokens, k1=self.k1, b=self.b)

        self._is_fitted = True
        logger.info(f"BM25 encoder fitted with {len(self.corpus_tokens)} documents")
        return self

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = "".join(c if c.isalnum() or c.isspace() else " " for c in text)
        tokens = text.split()
        tokens = [t for t in tokens if len(t) > 1]
        return tokens

    def _generate_doc_id(self, doc: Dict[str, Any], idx: int) -> str:
        doc_str = json.dumps(doc, sort_keys=True)
        return hashlib.md5(doc_str.encode()).hexdigest()[:16]

    def add_documents(self, documents: List[Dict[str, Any]], text_field: str = "text") -> "BM25SparseEncoder":
        if not self._is_fitted:
            return self.fit(documents, text_field)

        new_tokens = []
        new_doc_ids = []
        
        start_idx = len(self.corpus_tokens)
        for idx, doc in enumerate(documents):
            doc_id = doc.get("id", self._generate_doc_id(doc, start_idx + idx))
            text = doc.get(text_field, "")
            
            if not text:
                continue

            tokens = self._tokenize(text)
            new_tokens.append(tokens)
            new_doc_ids.append(doc_id)
            self.doc_id_to_index[doc_id] = start_idx + idx

        self.corpus_tokens.extend(new_tokens)
        self.doc_ids.extend(new_doc_ids)

        all_tokens = self.corpus_tokens
        if self.variant == BM25Variant.OKAPI:
            self.bm25 = BM25Okapi(all_tokens, k1=self.k1, b=self.b)
        elif self.variant == BM25Variant.PLUS:
            self.bm25 = BM25Plus(all_tokens, k1=self.k1, b=self.b)
        elif self.variant == BM25Variant.L:
            self.bm25 = BM25L(all_tokens, k1=self.k1, b=self.b)
        else:
            self.bm25 = BM25Okapi(all_tokens, k1=self.k1, b=self.b)

        return self

    def get_scores(self, query: str) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("BM25 encoder must be fitted before scoring")
        
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        return scores

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not self._is_fitted:
            raise RuntimeError("BM25 encoder must be fitted before searching")

        query_tokens = self._tokenize(query)
        doc_scores = self.bm25.get_scores(query_tokens)
        
        top_indices = np.argsort(doc_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if doc_scores[idx] > 0:
                results.append({
                    "doc_id": self.doc_ids[idx],
                    "score": float(doc_scores[idx]),
                    "index": int(idx)
                })
        
        return results

    def get_scores_batch(self, queries: List[str]) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("BM25 encoder must be fitted before batch scoring")

        all_scores = np.zeros((len(queries), len(self.corpus_tokens)))
        
        for i, query in enumerate(queries):
            query_tokens = self._tokenize(query)
            all_scores[i] = self.bm25.get_scores(query_tokens)
        
        return all_scores

    def delete_document(self, doc_id: str) -> bool:
        if not self._is_fitted:
            return False
        
        if doc_id not in self.doc_id_to_index:
            return False

        idx = self.doc_id_to_index[doc_id]
        
        del self.corpus_tokens[idx]
        del self.doc_ids[idx]
        del self.doc_id_to_index[doc_id]
        
        for doc_id_key, doc_idx in list(self.doc_id_to_index.items()):
            if doc_idx > idx:
                self.doc_id_to_index[doc_id_key] = doc_idx - 1

        all_tokens = self.corpus_tokens
        if self.variant == BM25Variant.OKAPI:
            self.bm25 = BM25Okapi(all_tokens, k1=self.k1, b=self.b)
        elif self.variant == BM25Variant.PLUS:
            self.bm25 = BM25Plus(all_tokens, k1=self.k1, b=self.b)
        elif self.variant == BM25Variant.L:
            self.bm25 = BM25L(all_tokens, k1=self.k