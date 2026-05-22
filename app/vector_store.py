"""
vector_store.py — Pinecone-backed RAG layer for AI Talent Agent.

Features:
  - Upserts candidate embeddings to Pinecone on each run
  - Semantic search: given a JD embedding, retrieves top-K similar candidates
  - Namespaced by job_id so multiple JDs don't pollute each other
  - Graceful degradation when Pinecone is not configured
"""

import os
import logging
import hashlib
from typing import Optional

from dotenv import load_dotenv
load_dotenv()  # ensure .env is loaded even when module is imported before main

logger = logging.getLogger(__name__)

# Try to import Pinecone; degrade gracefully if not installed
try:
    from pinecone import Pinecone, ServerlessSpec
    _PINECONE_AVAILABLE = True
except ImportError:
    _PINECONE_AVAILABLE = False
    logger.warning("pinecone-client not installed — vector store disabled.")

INDEX_NAME = "talent-agent"
DIMENSION  = 384   # matches all-MiniLM-L6-v2 (sentence-transformers default)


class VectorStore:
    """Wraps Pinecone index for candidate semantic search."""

    def __init__(self):
        self.enabled = False
        self.index   = None

        if not _PINECONE_AVAILABLE:
            return

        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            logger.warning("PINECONE_API_KEY not set — vector store disabled.")
            return

        try:
            pc = Pinecone(api_key=api_key)

            # Create index if it doesn't exist
            existing = [i.name for i in pc.list_indexes()]
            if INDEX_NAME not in existing:
                logger.info("Creating Pinecone index: %s", INDEX_NAME)
                pc.create_index(
                    name=INDEX_NAME,
                    dimension=DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )

            self.index   = pc.Index(INDEX_NAME)
            self.enabled = True
            logger.info("✅  Pinecone vector store ready (index=%s)", INDEX_NAME)

        except Exception as e:
            logger.error("Pinecone init failed: %s", e)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _candidate_id(name: str, email: Optional[str], source_file: str = "") -> str:
        # Use source filename as primary key — prevents two resumes from the
        # same person (different files) colliding on the same vector ID.
        if source_file:
            key = source_file.lower().strip()
        else:
            key = f"{name}|{email or ''}".lower().strip()
        return hashlib.md5(key.encode()).hexdigest()

    @staticmethod
    def _candidate_text(candidate: dict) -> str:
        parts = [
            candidate.get("name", ""),
            candidate.get("summary", ""),
            " ".join(candidate.get("skills", [])),
            " ".join(candidate.get("projects", [])),
            candidate.get("education", ""),
        ]
        return " ".join(p for p in parts if p).strip()

    # ── Upsert ────────────────────────────────────────────────────────────────

    def upsert_candidates(
        self,
        candidates: list,
        embed_fn,          # callable: str -> list[float]
        namespace: str = "default",
    ):
        """Embed and upsert a list of parsed candidate dicts."""
        if not self.enabled:
            return

        vectors = []
        for c in candidates:
            text = self._candidate_text(c)
            if not text:
                continue
            try:
                embedding = embed_fn(text)
                vid = self._candidate_id(
                    c.get("name", ""),
                    c.get("email"),
                    source_file=c.get("_source_file", ""),
                )
                metadata = {
                    "name":             c.get("name", ""),
                    "email":            c.get("email") or "",
                    "experience_years": c.get("experience_years") or 0,
                    "skills":           ", ".join(c.get("skills", [])[:20]),
                    "summary":          (c.get("summary") or "")[:500],
                    "source_file":      c.get("_source_file", ""),
                }
                vectors.append({"id": vid, "values": embedding, "metadata": metadata})
            except Exception as e:
                logger.warning("Embed failed for %s: %s", c.get("name"), e)

        if vectors:
            try:
                self.index.upsert(vectors=vectors, namespace=namespace)
                logger.info("📌  Upserted %d candidates to Pinecone (ns=%s)", len(vectors), namespace)
            except Exception as e:
                logger.error("Pinecone upsert error: %s", e)

    # ── Query ─────────────────────────────────────────────────────────────────

    def query_similar(
        self,
        jd_text: str,
        embed_fn,
        top_k: int = 10,
        namespace: str = "default",
    ) -> list:
        """
        Return up to top_k candidate metadata dicts ranked by semantic
        similarity to the JD text.
        Returns empty list if vector store is disabled.
        """
        if not self.enabled:
            return []

        try:
            jd_embedding = embed_fn(jd_text)
            result = self.index.query(
                vector=jd_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace=namespace,
            )
            matches = result.get("matches", [])
            logger.info("🔍  Pinecone retrieved %d RAG matches", len(matches))
            return [
                {**m["metadata"], "rag_score": round(m["score"], 4)}
                for m in matches
            ]
        except Exception as e:
            logger.error("Pinecone query error: %s", e)
            return []

    # ── Delete namespace (clean slate per job run) ────────────────────────────

    def clear_namespace(self, namespace: str = "default"):
        if not self.enabled:
            return
        try:
            self.index.delete(delete_all=True, namespace=namespace)
            logger.info("🗑️  Cleared Pinecone namespace: %s", namespace)
        except Exception as e:
            logger.warning("Pinecone clear error: %s", e)


# Singleton
_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store