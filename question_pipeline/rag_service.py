import json
import threading
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from .config import config
from .models import KnowledgeChunk, VectorStoreData
from .providers.factory import get_embedding_provider
from .governor import ProviderBlockedException, QuotaExhaustedException
from .state_manager import state_manager

class RAGService:
    """Vector indexing and semantic retrieval engine for question generation."""
    def __init__(self, store_file: Optional[Path] = None):
        self.store_file = store_file or config.VECTOR_STORE_FILE
        self.embedding_provider = get_embedding_provider()
        self.chunks: List[KnowledgeChunk] = []
        self._lock = threading.RLock()
        self._load()

    def _get_current_model_info(self) -> tuple[str, str]:
        provider = self.embedding_provider.provider_name
        model = getattr(self.embedding_provider, "model_name", None) or config.EMBEDDING_MODEL
        return provider, model

    def _load(self):
        if not self.store_file.exists():
            self.chunks = []
            return

        try:
            with open(self.store_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            curr_provider, curr_model = self._get_current_model_info()

            if isinstance(data, dict) and "embedding_model" in data:
                stored_model = data.get("embedding_model")
                stored_provider = data.get("embedding_provider")

                # Invalidate if embedding model or provider changed
                if stored_model != curr_model or stored_provider != curr_provider:
                    print(f"[RAG] Embedding model changed: '{stored_model}' ({stored_provider}) -> '{curr_model}' ({curr_provider}).")
                    print("[RAG] Invalidating vector store to prevent mixing incompatible embedding spaces.")
                    self.chunks = []
                    self.save()
                    return

                self.chunks = [KnowledgeChunk(**item) for item in data.get("chunks", [])]
            elif isinstance(data, list):
                print(f"[RAG] Migrating unversioned vector store to model-versioned store ('{curr_model}').")
                self.chunks = []
                self.save()
        except Exception as e:
            print(f"[RAG] Warning: Could not load vector store: {e}")
            self.chunks = []

    def save(self):
        with self._lock:
            try:
                curr_provider, curr_model = self._get_current_model_info()
                store_data = VectorStoreData(
                    embedding_provider=curr_provider,
                    embedding_model=curr_model,
                    chunks=self.chunks
                )
                with open(self.store_file, "w", encoding="utf-8") as f:
                    json.dump(store_data.model_dump(), f, indent=2)
            except Exception as e:
                print(f"[RAG] Warning: Failed to save vector store: {e}")

    async def index_chunks(self, chunks: List[KnowledgeChunk]) -> None:
        """Embed and store synthesized knowledge chunks with model versioning."""
        if not chunks:
            return

        curr_provider, curr_model = self._get_current_model_info()
        texts = [f"{c.topic}\n{c.text}" for c in chunks]
        print(f"[RAG] Embedding {len(texts)} chunks using {curr_provider} ({curr_model})...")

        try:
            embeddings = await self.embedding_provider.embed_texts(texts, task_name="rag_embedding")
        except QuotaExhaustedException as qe:
            state_manager.mark_waiting_for_quota(chunks[0].role, f"RAG embedding quota exhausted: {qe.message}")
            raise
        except ProviderBlockedException as pbe:
            state_manager.mark_blocked(chunks[0].role, f"RAG embedding blocked: {pbe.message}")
            raise

        with self._lock:
            # Avoid duplicate chunk IDs
            existing_ids = {c.chunk_id for c in self.chunks}
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb
                chunk.embedding_model = curr_model
                if chunk.chunk_id in existing_ids:
                    # Replace existing
                    self.chunks = [c if c.chunk_id != chunk.chunk_id else chunk for c in self.chunks]
                else:
                    self.chunks.append(chunk)

        self.save()
        print(f"[RAG] Indexed {len(chunks)} chunks. Total in store: {len(self.chunks)}.")

    def clear_role_chunks(self, role: str) -> int:
        """Remove indexed chunks ONLY for the specified role (used on forced rerun)."""
        with self._lock:
            before = len(self.chunks)
            self.chunks = [c for c in self.chunks if c.role.lower() != role.lower()]
            removed = before - len(self.chunks)
            if removed > 0:
                self.save()
            return removed

    @staticmethod
    def _cosine_rank(query_arr: np.ndarray, pool: List[KnowledgeChunk]) -> List[Tuple[float, KnowledgeChunk]]:
        """Shared similarity-ranking core used by both role and company
        retrieval, so the two paths can never silently drift out of sync."""
        scored: List[Tuple[float, KnowledgeChunk]] = []
        for c in pool:
            c_arr = np.array(c.embedding, dtype=np.float32)
            c_norm = np.linalg.norm(c_arr)
            if c_norm > 1e-6:
                c_arr = c_arr / c_norm
            similarity = float(np.dot(query_arr, c_arr))
            scored.append((similarity, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    async def _embed_normalized_query(self, query: str) -> np.ndarray:
        query_vec = await self.embedding_provider.embed_query(query, task_name="query_embedding")
        query_arr = np.array(query_vec, dtype=np.float32)
        q_norm = np.linalg.norm(query_arr)
        if q_norm > 1e-6:
            query_arr = query_arr / q_norm
        return query_arr

    async def retrieve_context(self, role: str, experience_band: str, paradigm: str, top_k: int = 3) -> List[KnowledgeChunk]:
        """Retrieve top-k relevant ROLE knowledge chunks using cosine
        similarity. Unchanged behavior from before company-context retrieval
        existed. `c.company is None` is the explicit guard that keeps
        standalone company chunks (which reuse the role column for the
        company name, since research_chunks.role is NOT NULL) out of
        role-only retrieval -- company is the real discriminator, not role."""
        role_chunks = [c for c in self.chunks if c.company is None and c.role and c.role.lower() == role.lower() and c.embedding is not None]
        if not role_chunks:
            # Fallback to all ROLE chunks (still excludes company-only chunks)
            # if this specific role has none indexed.
            role_chunks = [c for c in self.chunks if c.company is None and c.role and c.embedding is not None]

        if not role_chunks:
            return []

        query = f"{role} engineering {paradigm} design troubleshooting level {experience_band}"
        query_arr = await self._embed_normalized_query(query)
        scored = self._cosine_rank(query_arr, role_chunks)
        return [item[1] for item in scored[:top_k]]

    async def retrieve_company_context(
        self, role: str, experience_band: str, paradigm: str,
        companies: Optional[List[str]] = None, top_k: int = 1,
    ) -> List[KnowledgeChunk]:
        """Retrieve company knowledge chunks (chunk.company is set, indexed
        via synthesize_and_index_company_research()) that are SEMANTICALLY
        RELEVANT to this role/band/paradigm query -- using the exact same
        cosine-similarity mechanism as retrieve_context(), not a fixed
        company list. A company whose research doesn't actually relate to
        this role (e.g. a consulting firm's audit practices vs. an ML
        Engineer query) will simply score below the relevance threshold and
        never surface -- this IS the "evidence-driven, not injected" rule,
        enforced structurally rather than by a hardcoded allow-list.

        `companies`, if given, restricts the candidate pool to those company
        names (still subject to the same relevance threshold); if omitted,
        all indexed company chunks are eligible and relevance alone decides
        which (if any) come back.
        """
        pool = [c for c in self.chunks if c.company and c.embedding is not None]
        if companies:
            wanted = {co.lower() for co in companies}
            pool = [c for c in pool if c.company.lower() in wanted]
        if not pool:
            return []

        query = f"{role} engineering {paradigm} level {experience_band} company-specific context"
        query_arr = await self._embed_normalized_query(query)
        scored = self._cosine_rank(query_arr, pool)

        relevant = [(sim, c) for sim, c in scored if sim >= config.COMPANY_CONTEXT_MIN_SIMILARITY]
        return [c for _, c in relevant[:top_k]]

rag_service = RAGService()
