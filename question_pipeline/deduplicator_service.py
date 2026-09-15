import re
import numpy as np
from typing import List, Tuple, Dict, Optional
from .config import config
from .models import QuestionObject
from .providers.factory import get_embedding_provider

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further",
    "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's",
    "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over",
    "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd",
    "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}

def normalize_text(text: str) -> str:
    """Strip punctuation, lowercase, and remove stopwords."""
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
    tokens = [w for w in cleaned.split() if w and w not in STOPWORDS]
    return " ".join(tokens)

def token_jaccard_similarity(text1: str, text2: str) -> float:
    s1 = set(normalize_text(text1).split())
    s2 = set(normalize_text(text2).split())
    if not s1 or not s2:
        return 0.0
    return float(len(s1.intersection(s2))) / float(len(s1.union(s2)))

class DeduplicatorService:
    """3-Tier Deduplication Engine: Exact Match -> Normalized Overlap -> Semantic Cosine Similarity."""
    def __init__(self):
        self.embedding_provider = get_embedding_provider()
        # In-memory cache of question embeddings for rapid cosine comparisons
        self._cached_embeddings: Dict[str, np.ndarray] = {}

    def _get_exact_key(self, text: str) -> str:
        return " ".join(text.lower().strip().split())

    async def check_duplicate(
        self,
        candidate: QuestionObject,
        existing_bank: List[QuestionObject]
    ) -> Tuple[bool, str]:
        """Check if candidate question is duplicate against existing bank."""
        if not existing_bank:
            return (False, "")

        cand_text = candidate.question
        cand_key = self._get_exact_key(cand_text)

        # Tier 1: Exact Match
        for item in existing_bank:
            if self._get_exact_key(item.question) == cand_key:
                return (True, "EXACT_DUPLICATE")

        # Tier 2: Normalized Overlap
        cand_norm = normalize_text(cand_text)
        for item in existing_bank:
            sim = token_jaccard_similarity(cand_text, item.question)
            if sim >= config.NORMALIZED_OVERLAP_THRESHOLD:
                return (True, f"NORMALIZED_DUPLICATE (overlap={sim:.2f})")

        # Tier 3: Semantic Cosine Similarity via Embeddings
        try:
            cand_vec = await self.embedding_provider.embed_query(cand_text, task_name="dedup_embedding")
            cand_arr = np.array(cand_vec, dtype=np.float32)
            c_norm = np.linalg.norm(cand_arr)
            if c_norm > 1e-6:
                cand_arr = cand_arr / c_norm

            # Compare against existing bank embeddings
            for item in existing_bank:
                item_text = item.question
                if item_text not in self._cached_embeddings:
                    vec = await self.embedding_provider.embed_query(item_text, task_name="dedup_embedding")
                    arr = np.array(vec, dtype=np.float32)
                    norm = np.linalg.norm(arr)
                    if norm > 1e-6:
                        arr = arr / norm
                    self._cached_embeddings[item_text] = arr

                existing_arr = self._cached_embeddings[item_text]
                cosine_sim = float(np.dot(cand_arr, existing_arr))

                if cosine_sim >= config.SEMANTIC_SIMILARITY_THRESHOLD:
                    return (True, f"SEMANTIC_DUPLICATE (similarity={cosine_sim:.2f})")

            # Cache candidate embedding
            self._cached_embeddings[cand_text] = cand_arr

        except Exception as e:
            print(f"[DEDUPLICATOR] Semantic embedding check warning: {e}. Falling back to normalized check.")

        return (False, "")

deduplicator_service = DeduplicatorService()
