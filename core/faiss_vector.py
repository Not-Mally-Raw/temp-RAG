"""
FAISS-based Vector Manager for simple, local RAG
Keeps API similar to EnhancedVectorManager used by the system.
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Any, Optional
import structlog

try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

from .enhanced_vector_utils import ManufacturingEmbeddingModel, EmbeddingConfig


enhanced_logger = structlog.get_logger()


class FaissVectorManager:
    """Minimal FAISS-backed vector store with optional TF-IDF hybrid.

    Methods mirror a subset of EnhancedVectorManager used by production_system:
    - add_texts(texts, metadatas)
    - similarity_search(query, top_k, score_threshold, filter_conditions)
    - get_collection_stats()
    - optimize_collection() (no-op)
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        if not FAISS_AVAILABLE:
            raise RuntimeError("faiss is not installed. Install faiss-cpu to use FaissVectorManager.")

        self.config = config or EmbeddingConfig()
        self.embedding_model = ManufacturingEmbeddingModel(self.config)

        # We'll use cosine similarity via inner product on L2-normalized vectors
        self.dim = self.config.vector_size
        self.index = faiss.IndexFlatIP(self.dim)

        # Store payloads alongside vectors
        self._texts: List[str] = []
        self._metadatas: List[Dict[str, Any]] = []

        # Optional hybrid search support
        self._tfidf_vectorizer = None
        self._corpus: List[str] = []

        enhanced_logger.info("FAISS vector manager initialized", dim=self.dim)

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        return X / norms

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        if not texts:
            return []

        if metadatas is None:
            metadatas = [{} for _ in texts]

        # Compute embeddings and normalize for cosine via inner product
        embs = self.embedding_model.encode_batch(texts)
        embs = self._normalize(embs.astype(np.float32))

        # Add to FAISS
        self.index.add(embs)

        # Store payloads
        start_id = len(self._texts)
        self._texts.extend(texts)
        self._metadatas.extend(metadatas)

        # Optional: update TF-IDF corpus for simple hybrid
        self._corpus.extend(texts)

        ids = [str(start_id + i) for i in range(len(texts))]

        enhanced_logger.info("Added texts to FAISS", count=len(texts), total=len(self._texts))
        return ids

    def similarity_search(self, query: str, top_k: int = 5,
                           score_threshold: float = 0.0,
                           filter_conditions: Optional[Dict] = None) -> List[Dict[str, Any]]:
        if len(self._texts) == 0 or self.index.ntotal == 0:
            return []

        # For now ignore filter_conditions (no payload index)
        q = self.embedding_model.encode_batch([query]).astype(np.float32)
        q = self._normalize(q)

        scores, idxs = self.index.search(q, top_k)
        scores = scores[0]
        idxs = idxs[0]

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores, idxs):
            if idx == -1:
                continue
            if score < score_threshold:
                continue
            payload = {
                'text': self._texts[idx],
                'similarity_score': float(score),
                'metadata': self._metadatas[idx] if idx < len(self._metadatas) else {},
                'id': str(idx)
            }
            results.append(payload)

        enhanced_logger.info("FAISS semantic search completed",
                              query_preview=query[:50],
                              results_count=len(results))
        return results

    def get_collection_stats(self) -> Dict[str, Any]:
        return {
            'total_points': int(self.index.ntotal),
            'vector_size': self.dim,
            'distance_metric': 'cosine(inner_product)',
            'sample_size': min(10, int(self.index.ntotal))
        }

    def optimize_collection(self):
        # No-op for simple IndexFlatIP
        pass
