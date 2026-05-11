"""
retriever.py
Hybrid retrieval pipeline:
  1. Multi-query expansion  — improves recall on vague inputs
  2. FAISS semantic search  — dense vector similarity
  3. Metadata filtering     — structured field constraints
  4. Deduplication + merge  — union of all retrieval paths
  5. LLM reranking          — final top-N selection

Index is built once at startup and held in memory.
No external vector store service required.

Embeddings use FastEmbed BGE (`passage_embed` / `query_embed`). Rebuild the FAISS
files after any change to ``EMBEDDING_MODEL`` in this module.
"""

import os
import logging
import pickle
from typing import Optional

import faiss
import numpy as np
from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

# Paths
INDEX_PATH = "data/faiss.index"
META_PATH = "data/faiss_meta.pkl"

# ONNX embeddings via FastEmbed — BGE tuned for retrieval (passage vs query prefixes)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# How many candidates to pull before reranking
CANDIDATE_POOL = 20


class SHLRetriever:
    """
    Encapsulates the full hybrid retrieval pipeline.
    Call build_index() once on startup, then retrieve() per request.
    """

    def __init__(self):
        self.model: Optional[TextEmbedding] = None
        self.index: Optional[faiss.Index] = None
        self.metadata: list[dict] = []   # parallel list to FAISS vectors
        self.catalog: list[dict] = []    # full catalog for metadata filtering
        self.name_index: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def build_index(
        self,
        catalog: list[dict],
        name_index: dict[str, dict],
        force_rebuild: bool = False,
    ) -> None:
        """
        Build (or load cached) FAISS index from catalog.
        Called once during FastAPI lifespan startup.
        """
        self.catalog = catalog
        self.name_index = name_index

        logger.info("Loading FastEmbed model %r ...", EMBEDDING_MODEL)
        self.model = TextEmbedding(model_name=EMBEDDING_MODEL)

        if not force_rebuild and os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            logger.info("Loading cached FAISS index from disk...")
            self.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "rb") as f:
                self.metadata = pickle.load(f)
            logger.info(f"Index loaded: {self.index.ntotal} vectors.")
            return

        logger.info(f"Building FAISS index for {len(catalog)} assessments...")
        texts = [entry["embedding_text"] for entry in catalog]
        # Corpus side: passage_embed applies BGE "passage:" prefix for retrieval
        embeddings = np.array(
            list(self.model.passage_embed(texts, batch_size=64)),
            dtype="float32",
        )

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner product = cosine on normalized vecs
        self.index.add(embeddings)

        # metadata parallel to vectors: store only what retrieval needs
        self.metadata = [
            {
                "name": e["name"],
                "url": e["link"],
                "test_type": e["test_type"],
                "all_test_types": e.get("all_test_types", []),
                "keys": e.get("keys", []),
                "job_levels": e.get("job_levels", []),
                "duration_minutes": e.get("duration_minutes"),
                "remote": e.get("remote", ""),
                "adaptive": e.get("adaptive", ""),
                "description": e.get("description", ""),
            }
            for e in catalog
        ]

        # Persist to disk
        os.makedirs("data", exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump(self.metadata, f)

        logger.info(f"FAISS index built and saved: {self.index.ntotal} vectors.")

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    def _vector_search(self, query: str, top_k: int = CANDIDATE_POOL) -> list[dict]:
        """Single query → top_k nearest neighbors by cosine similarity."""
        # Query side: query_embed applies BGE "query:" prefix
        vec = np.array(
            list(self.model.query_embed([query])),
            dtype="float32",
        )
        faiss.normalize_L2(vec)

        scores, indices = self.index.search(vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            item = dict(self.metadata[idx])
            item["_score"] = float(score)
            results.append(item)
        return results

    # ------------------------------------------------------------------
    # Metadata filtering
    # ------------------------------------------------------------------

    def _apply_filters(
        self,
        candidates: list[dict],
        filters: dict,
    ) -> list[dict]:
        """
        Apply structured metadata filters to a candidate list.
        Filters dict may contain any subset of:
          job_levels: list[str]
          keys: list[str]          (test categories)
          remote: bool
          adaptive: bool
          max_duration: int        (minutes)
        All filters are soft — if a filter eliminates everything, it is relaxed.
        """
        if not filters:
            return candidates

        filtered = list(candidates)

        # Job level filter (OR — any match is sufficient)
        if filters.get("job_levels"):
            required_levels = {jl.lower() for jl in filters["job_levels"]}
            level_matched = [
                c for c in filtered
                if any(jl.lower() in required_levels for jl in c.get("job_levels", []))
            ]
            if level_matched:
                filtered = level_matched

        # Test category filter (OR)
        if filters.get("keys"):
            required_keys = {k.lower() for k in filters["keys"]}
            key_matched = [
                c for c in filtered
                if any(k.lower() in required_keys for k in c.get("keys", []))
            ]
            if key_matched:
                filtered = key_matched

        # Remote filter
        if filters.get("remote") is True:
            remote_matched = [
                c for c in filtered if c.get("remote", "").lower() == "yes"
            ]
            if remote_matched:
                filtered = remote_matched

        # Adaptive filter
        if filters.get("adaptive") is True:
            adaptive_matched = [
                c for c in filtered if c.get("adaptive", "").lower() == "yes"
            ]
            if adaptive_matched:
                filtered = adaptive_matched

        # Max duration filter
        if filters.get("max_duration") is not None:
            dur_matched = [
                c for c in filtered
                if c.get("duration_minutes") is not None
                and c["duration_minutes"] <= filters["max_duration"]
            ]
            if dur_matched:
                filtered = dur_matched

        return filtered

    # ------------------------------------------------------------------
    # Multi-query expansion
    # ------------------------------------------------------------------

    def _expand_queries(self, context: str, extracted: dict) -> list[str]:
        """
        Generate multiple search queries from the context to improve recall.
        This is purely string-based (no LLM call) to stay within timeout.
        Combines the raw context with structured field expansions.
        """
        queries = [context]  # always include original

        # Expand from extracted skills/role
        if extracted.get("role"):
            queries.append(f"{extracted['role']} skills assessment test")
            queries.append(f"{extracted['role']} aptitude evaluation")

        if extracted.get("skills"):
            skills_str = " ".join(extracted["skills"])
            queries.append(f"{skills_str} knowledge test")
            queries.append(f"technical assessment {skills_str}")

        if extracted.get("test_categories"):
            for cat in extracted["test_categories"]:
                queries.append(f"{cat} assessment")

        if extracted.get("seniority"):
            queries.append(
                f"{extracted['seniority']} level {extracted.get('role', '')} assessment"
            )

        if extracted.get("assessment_goals"):
            queries.append(str(extracted["assessment_goals"]))

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for q in queries:
            q_clean = q.strip()
            if q_clean and q_clean not in seen:
                seen.add(q_clean)
                unique.append(q_clean)

        return unique[:5]  # cap at 5 queries to control latency

    # ------------------------------------------------------------------
    # Main retrieve entry point
    # ------------------------------------------------------------------

    def retrieve(
        self,
        context: str,
        extracted: dict,
        filters: dict,
        top_n: int = 20,
    ) -> list[dict]:
        """
        Full hybrid retrieval pipeline.
        Returns up to top_n deduplicated candidates for LLM reranking.

        Args:
            context:   Natural language summary of what the user wants.
            extracted: Structured fields pulled from conversation
                       (role, skills, seniority, test_categories, etc.)
            filters:   Hard metadata constraints (job_levels, keys, remote, etc.)
            top_n:     Max candidates to return before reranking.
        """
        if self.index is None:
            raise RuntimeError("Retriever not initialized. Call build_index() first.")

        queries = self._expand_queries(context, extracted)
        logger.debug(f"Expanded to {len(queries)} queries: {queries}")

        # Union results from all queries
        seen_names: set[str] = set()
        all_candidates: list[dict] = []

        for q in queries:
            results = self._vector_search(q, top_k=CANDIDATE_POOL)
            for r in results:
                if r["name"] not in seen_names:
                    seen_names.add(r["name"])
                    all_candidates.append(r)

        logger.debug(f"Candidates before filtering: {len(all_candidates)}")

        # Apply metadata filters
        filtered = self._apply_filters(all_candidates, filters)
        logger.debug(f"Candidates after filtering: {len(filtered)}")

        # Sort by vector score descending, take top_n
        filtered.sort(key=lambda x: x.get("_score", 0), reverse=True)
        return filtered[:top_n]

    # ------------------------------------------------------------------
    # Direct lookup (for compare queries)
    # ------------------------------------------------------------------

    def get_by_name(self, name: str) -> Optional[dict]:
        """
        Exact name lookup for compare queries.
        Case-insensitive. Returns full metadata dict or None.
        """
        return self.name_index.get(name.lower())

    def resolve_catalog_name(self, raw: str) -> Optional[dict]:
        """
        Map a user- or model-typed title to a catalog entry.

        Handles trailing punctuation (e.g. 'Simulation?') and common SHL
        renames where the catalog uses a suffix such as ' (New)' but the
        user omits it.
        """
        if not raw or not self.name_index:
            return None
        key = raw.strip().lower().rstrip("?.!,:;").lstrip("¿¡")
        if not key:
            return None
        if key in self.name_index:
            return self.name_index[key]
        with_new = f"{key} (new)"
        if with_new in self.name_index:
            return self.name_index[with_new]
        hits = [e for nk, e in self.name_index.items() if nk.startswith(f"{key} (")]
        if not hits:
            return None
        if len(hits) == 1:
            return hits[0]
        hits.sort(key=lambda e: (len(e["name"]), e["name"]))
        return hits[0]

    def fuzzy_name_search(self, name: str, top_k: int = 3) -> list[dict]:
        """
        Fuzzy name search for when the user misspells an assessment name.
        Uses vector search on just the name string.
        Returns top_k closest matches.
        """
        results = self._vector_search(name, top_k=top_k)
        return results
