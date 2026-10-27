"""Secure Vector Store Wrapper for ChromaDB and Local Embeddings.

Features:
- Enforces strict server-side metadata filtering based on UserRole.
- Collection confinement: prevents arbitrary collection creation or switching.
- Graceful dual-backend: uses official ChromaDB + SentenceTransformers if installed,
  and provides a deterministic, zero-dependency local vector store fallback for
  environments where heavy ML binaries are not yet installed.
- Redundant post-retrieval authorization validation.
"""

import hashlib
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from src.access_control import UserRole, build_chroma_access_filter, is_authorized_for_level, normalize_role
from src.config import SecurityConfig
from src.guardrails import sanitize_document_text_for_rag
from src.logger import audit_logger, log_operational_event, log_security_event

# Try importing real ChromaDB and SentenceTransformer
_CHROMA_AVAILABLE = False
_SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import chromadb
    from chromadb.config import Settings
    _CHROMA_AVAILABLE = True
except ImportError:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None


class LightweightLocalEmbedder:
    """Deterministic token/frequency embedder used when sentence-transformers is offline."""

    def __init__(self, dim: int = 128):
        self.dim = dim

    def encode(self, text: str) -> List[float]:
        tokens = re.findall(r"\w+", text.lower())
        vec = [0.0] * self.dim
        if not tokens:
            return vec

        for token in tokens:
            # Deterministic hash bucket
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            vec[idx] += 1.0

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    return dot


class SecureVectorStore:
    """ChromaDB wrapper enforcing collection boundaries and RBAC metadata filters."""

    COLLECTION_NAME = "hospital_rag_approved_docs"

    def __init__(self):
        SecurityConfig.ensure_directories()
        self.persist_dir = str(SecurityConfig.CHROMA_PERSIST_DIR)
        self.is_chroma_active = _CHROMA_AVAILABLE
        self.embedder = None
        self.client = None
        self.collection = None

        # Fallback local in-memory storage (doc_id -> list of chunks)
        self._local_chunks: List[Dict[str, Any]] = []
        self._fallback_embedder = LightweightLocalEmbedder()

        self._initialize_backends()
        self._sync_approved_documents()

    def _sync_approved_documents(self):
        """Loads all approved documents from disk into the store if local chunks are empty."""
        try:
            from src.ingestion import document_ingestion_service
            approved_docs = document_ingestion_service.list_documents(status="APPROVED")
            for doc in approved_docs:
                text = document_ingestion_service.get_document_text(doc.doc_id)
                if text:
                    self.index_document(
                        doc_id=doc.doc_id,
                        text=text,
                        metadata={
                            "filename": doc.filename,
                            "access_level": doc.access_level,
                            "version": doc.version,
                            "effective_date": doc.effective_date,
                        },
                        admin_role="ADMIN"
                    )
        except Exception as e:
            audit_logger.warning(f"Could not auto-sync approved documents: {e}")

    def _initialize_backends(self):
        if _SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedder = SentenceTransformer(SecurityConfig.EMBEDDING_MODEL_NAME)
            except Exception as e:
                audit_logger.warning(f"Could not load SentenceTransformer: {e}. Using lightweight embedder.")
                self.embedder = None

        if self.is_chroma_active:
            try:
                self.client = chromadb.PersistentClient(path=self.persist_dir)
                self.collection = self.client.get_or_create_collection(
                    name=self.COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception as e:
                audit_logger.warning(f"ChromaDB initialization failed: {e}. Falling back to local store.")
                self.is_chroma_active = False

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> List[str]:
        """Deterministic text chunking with clean sentence/boundary splits."""
        cleaned, _ = sanitize_document_text_for_rag(text)
        if not cleaned:
            return []

        chunks = []
        start = 0
        text_len = len(cleaned)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            # Find nearest space/newline if not at end
            if end < text_len:
                last_space = cleaned.rfind(" ", start, end)
                if last_space > start + (chunk_size // 2):
                    end = last_space

            chunk = cleaned[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= text_len:
                break
            start = end - chunk_overlap

        return chunks

    def index_document(
        self,
        doc_id: str,
        text: str,
        metadata: Dict[str, Any],
        admin_role: str = "ADMIN"
    ) -> int:
        """Indexes an approved document. Restricted to authorized administrators."""
        if admin_role.upper() != "ADMIN":
            log_security_event(
                "UNAUTHORIZED_INDEX_ATTEMPT", "system", admin_role, "HIGH", {"doc_id": doc_id}
            )
            raise PermissionError("Only ADMIN users can index documents into ChromaDB.")

        chunks = self.chunk_text(text)
        if not chunks:
            return 0

        # Remove previous chunks for this document if updating
        self.delete_document(doc_id, admin_role="ADMIN")

        chunk_ids = []
        chunk_metadatas = []
        chunk_texts = []
        embeddings = []

        access_level = metadata.get("access_level", SecurityConfig.ACCESS_LEVEL_PUBLIC)
        source_name = metadata.get("filename", "unknown_source")
        doc_version = metadata.get("version", "1.0")
        eff_date = metadata.get("effective_date", "")

        for idx, chunk in enumerate(chunks):
            cid = f"{doc_id}_chunk_{idx}"
            meta = {
                "doc_id": str(doc_id),
                "source": str(source_name),
                "chunk_idx": int(idx),
                "access_level": str(access_level),
                "version": str(doc_version),
                "effective_date": str(eff_date),
            }
            chunk_ids.append(cid)
            chunk_metadatas.append(meta)
            chunk_texts.append(chunk)

            # Store in local fallback
            vec = self._fallback_embedder.encode(chunk)
            self._local_chunks.append({
                "id": cid,
                "text": chunk,
                "metadata": meta,
                "embedding": vec,
            })

        if self.is_chroma_active and self.collection is not None:
            try:
                if self.embedder is not None:
                    embeddings = self.embedder.encode(chunk_texts).tolist()
                    self.collection.upsert(
                        ids=chunk_ids,
                        documents=chunk_texts,
                        metadatas=chunk_metadatas,
                        embeddings=embeddings
                    )
                else:
                    self.collection.upsert(
                        ids=chunk_ids,
                        documents=chunk_texts,
                        metadatas=chunk_metadatas
                    )
            except Exception as e:
                audit_logger.error(f"Error persisting to ChromaDB: {e}")

        log_operational_event(
            "DOCUMENT_INDEXED",
            "system",
            admin_role,
            "SUCCESS",
            {"doc_id": doc_id, "chunk_count": len(chunks), "access_level": access_level}
        )
        return len(chunks)

    def delete_document(self, doc_id: str, admin_role: str = "ADMIN") -> bool:
        """Removes a document from the vector store."""
        if admin_role.upper() != "ADMIN":
            raise PermissionError("Only ADMIN users can delete documents from vector store.")

        # Remove from local in-memory
        self._local_chunks = [c for c in self._local_chunks if c["metadata"].get("doc_id") != doc_id]

        if self.is_chroma_active and self.collection is not None:
            try:
                self.collection.delete(where={"doc_id": doc_id})
            except Exception:
                pass
        return True

    def search(
        self,
        query: str,
        user_role: UserRole | str = UserRole.CITIZEN,
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """Retrieves chunks strictly filtered by user's access level."""
        role_enum = normalize_role(user_role if isinstance(user_role, str) else user_role.value)
        meta_filter = build_chroma_access_filter(role_enum)

        results: List[Dict[str, Any]] = []

        if self.is_chroma_active and self.collection is not None:
            try:
                if self.embedder is not None:
                    q_vec = self.embedder.encode([query]).tolist()
                    chroma_res = self.collection.query(
                        query_embeddings=q_vec,
                        n_results=top_k,
                        where=meta_filter
                    )
                else:
                    chroma_res = self.collection.query(
                        query_texts=[query],
                        n_results=top_k,
                        where=meta_filter
                    )

                if chroma_res and chroma_res.get("documents"):
                    docs = chroma_res["documents"][0]
                    metas = chroma_res["metadatas"][0]
                    ids = chroma_res["ids"][0]
                    for d, m, i in zip(docs, metas, ids):
                        results.append({"id": i, "text": d, "metadata": m})
            except Exception as e:
                audit_logger.warning(f"ChromaDB search failed, fallback engaged: {e}")

        # If Chroma returned no results or was inactive, use local fallback
        if not results and self._local_chunks:
            q_vec = self._fallback_embedder.encode(query)
            scored = []
            for item in self._local_chunks:
                # Enforce RBAC filter deterministically
                item_lvl = item["metadata"].get("access_level", SecurityConfig.ACCESS_LEVEL_PUBLIC)
                if not is_authorized_for_level(role_enum, item_lvl):
                    continue

                sim = cosine_similarity(q_vec, item["embedding"])
                scored.append((sim, item))

            scored.sort(key=lambda x: x[0], reverse=True)
            for sim, item in scored[:top_k]:
                results.append({"id": item["id"], "text": item["text"], "metadata": item["metadata"], "score": sim})

        # Redundant post-retrieval validation: guarantee zero unauthorized leakage
        authorized_results = []
        for r in results:
            lvl = r.get("metadata", {}).get("access_level", SecurityConfig.ACCESS_LEVEL_PUBLIC)
            if is_authorized_for_level(role_enum, lvl):
                authorized_results.append(r)
            else:
                log_security_event(
                    "UNAUTHORIZED_CHUNK_SUPPRESSED",
                    "retrieval",
                    role_enum.value,
                    "CRITICAL",
                    {"chunk_id": r.get("id"), "chunk_level": lvl}
                )

        log_operational_event(
            "RETRIEVAL_COMPLETED",
            "search",
            role_enum.value,
            "SUCCESS",
            {"retrieved_count": len(authorized_results)}
        )
        return authorized_results


vector_store_service = SecureVectorStore()
