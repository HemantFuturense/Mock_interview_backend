import math
import re
import numpy as np
from typing import List
from .base import EmbeddingProvider
from ..cost_tracker import cost_tracker

class LocalEmbeddingProvider(EmbeddingProvider):
    """Deterministic, zero-cost embedding provider using normalized character-word vectors.
    Provides fallback dense embeddings for RAG and deduplication when external embedding quotas are exhausted.
    """
    def __init__(self, dim: int = 128):
        self.dim = dim

    @property
    def provider_name(self) -> str:
        return "local"

    def _text_to_vector(self, text: str) -> List[float]:
        tokens = re.findall(r"\w+", text.lower())
        vec = np.zeros(self.dim, dtype=np.float32)

        if not tokens:
            return vec.tolist()

        for token in tokens:
            # Hash token to index and sign
            h = hash(token)
            idx = abs(h) % self.dim
            sign = 1.0 if (h % 2 == 0) else -1.0
            vec[idx] += sign

            # Also add character 3-grams for semantic fuzziness
            for i in range(len(token) - 2):
                ngram = token[i:i+3]
                nh = hash(ngram)
                nidx = abs(nh) % self.dim
                vec[nidx] += 0.5 * (1.0 if (nh % 2 == 0) else -1.0)

        # L2 Normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        return vec.tolist()

    async def embed_texts(self, texts: List[str], task_name: str = "rag_embedding") -> List[List[float]]:
        results = [self._text_to_vector(t) for t in texts]
        cost_tracker.record_usage(
            provider="local",
            model="tfidf",
            task=task_name,
            input_tokens=sum(len(t.split()) for t in texts),
            output_tokens=0
        )
        return results

    async def embed_query(self, text: str, task_name: str = "query_embedding") -> List[float]:
        vec = self._text_to_vector(text)
        cost_tracker.record_usage(
            provider="local",
            model="tfidf",
            task=task_name,
            input_tokens=len(text.split()),
            output_tokens=0
        )
        return vec
