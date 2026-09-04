"""LlamaIndex ingestion helpers.

This module keeps ML Tutor's indexing path thin by delegating parsing
transformations and embedding to LlamaIndex's official IngestionPipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode

logger = logging.getLogger(__name__)


def build_ingestion_pipeline() -> IngestionPipeline:
    """Create the default ML Tutor ingestion pipeline.

    The embedding step uses ``Settings.embed_model``, which is configured by
    ``embedding_adapter.configure_llamaindex_settings`` to call ML Tutor's
    configured embedding service rather than any local model.
    """

    return IngestionPipeline(
        transformations=[
            SentenceSplitter(
                chunk_size=Settings.chunk_size,
                chunk_overlap=Settings.chunk_overlap,
            ),
            Settings.embed_model,
        ],
    )


def _has_precomputed_embedding(document: Any) -> bool:
    """Return True only for non-Document nodes that already carry a vector.

    LlamaIndex's ``Document`` class inherits from ``BaseNode``, so a naive
    ``isinstance(doc, BaseNode)`` check incorrectly classifies every Document
    as pre-embedded, bypassing the chunking pipeline entirely. This helper
    distinguishes genuinely pre-embedded nodes (e.g. ImageNode produced by
    multimodal loaders) from regular Documents that still need splitting and
    embedding. The embedding may be a list or a numpy array, so we check
    ``len(...) > 0`` rather than ``bool(...)`` (ambiguous for ndarrays).
    """
    if isinstance(document, Document):
        return False
    if not isinstance(document, BaseNode):
        return False
    embedding = getattr(document, "embedding", None)
    if embedding is None:
        return False
    try:
        return len(embedding) > 0
    except TypeError:
        return True


def documents_to_nodes(documents: list[Any], *, show_progress: bool = True) -> list[Any]:
    """Convert LlamaIndex documents into embedded nodes.

    Pre-embedded nodes, such as ImageNode instances produced by the document
    loader, pass through unchanged so they are not re-embedded as text.
    """
    text_documents = [
        document for document in documents if not _has_precomputed_embedding(document)
    ]
    preembedded_nodes = [document for document in documents if _has_precomputed_embedding(document)]

    nodes: list[Any] = []
    if text_documents:
        pipeline = build_ingestion_pipeline()
        nodes.extend(pipeline.run(documents=text_documents, show_progress=show_progress))
    nodes.extend(preembedded_nodes)
    return nodes


def _try_create_faiss_vector_store(nodes: list[Any]) -> Any | None:
    """Create a FaissVectorStore with HNSW index if the library is available."""
    try:
        import faiss
        from llama_index.vector_stores.faiss import FaissVectorStore
    except ImportError:
        return None

    for node in nodes:
        embedding = node.get_embedding()
        if embedding is not None:
            dim = len(embedding)
            break
    else:
        return None

    faiss_index = faiss.IndexHNSWFlat(dim, 32)
    faiss_index.hnsw.efConstruction = 200
    faiss_index.hnsw.efSearch = 128
    return FaissVectorStore(faiss_index)


def create_index_from_documents(
    documents: list[Any], storage_dir: Path, *, show_progress: bool = True
) -> tuple[VectorStoreIndex, int]:
    """Create a VectorStoreIndex from documents.

    Uses FaissVectorStore with HNSW index when available, falling back to
    SimpleVectorStore otherwise. Persistence is handled by the caller.
    """
    nodes = documents_to_nodes(documents, show_progress=show_progress)

    vector_store = _try_create_faiss_vector_store(nodes)
    if vector_store is not None:
        from llama_index.core import StorageContext

        logger.info("Using FaissVectorStore (HNSW) for index creation.")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(
            nodes=nodes, storage_context=storage_context, show_progress=show_progress
        )
    else:
        logger.info("FaissVectorStore not available; using SimpleVectorStore.")
        index = VectorStoreIndex(nodes=nodes, show_progress=show_progress)

    return index, len(documents)


def insert_documents_into_index(
    index: Any, documents: list[Any], *, show_progress: bool = True
) -> int:
    """Transform documents once, then insert nodes into an existing index."""
    nodes = documents_to_nodes(documents, show_progress=show_progress)
    index.insert_nodes(nodes)
    return len(documents)


__all__ = [
    "build_ingestion_pipeline",
    "create_index_from_documents",
    "documents_to_nodes",
    "insert_documents_into_index",
]
