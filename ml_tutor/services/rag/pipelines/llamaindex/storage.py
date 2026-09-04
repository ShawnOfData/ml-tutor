"""Storage operations for the LlamaIndex RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import shutil
import threading
import time
from typing import Any

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.vector_stores.simple import DEFAULT_VECTOR_STORE, NAMESPACE_SEP, DEFAULT_PERSIST_FNAME

from ml_tutor.services.embedding.validation import validate_embedding_batch
from ml_tutor.services.rag.index_versioning import (
    EmbeddingSignature,
    find_matching_version,
    resolve_storage_dir_for_read,
    resolve_storage_dir_for_write,
    resolve_storage_dir_for_rebuild,
    write_version_meta,
)

from . import ingestion, retrievers

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AddStoragePlan:
    existing_storage: Path | None
    storage_dir: Path


def _is_faiss_vector_store(storage_dir: Path) -> bool:
    fname = f"{DEFAULT_VECTOR_STORE}{NAMESPACE_SEP}{DEFAULT_PERSIST_FNAME}"
    vs_path = storage_dir / fname
    if not vs_path.exists():
        return False
    try:
        with open(vs_path, encoding="utf-8") as f:
            json.load(f)
        return False
    except (json.JSONDecodeError, UnicodeDecodeError):
        return True


def _faiss_available() -> bool:
    try:
        from llama_index.vector_stores.faiss import FaissVectorStore  # noqa: F401

        return True
    except ImportError:
        return False


def _load_faiss_vector_store(storage_dir: Path) -> Any:
    import tempfile as _tempfile
    from llama_index.vector_stores.faiss import FaissVectorStore

    faiss_fname = f"{DEFAULT_VECTOR_STORE}{NAMESPACE_SEP}{DEFAULT_PERSIST_FNAME}"
    src_path = storage_dir / faiss_fname
    # faiss SWIG on Windows chokes on Unicode paths, so copy to ASCII temp path
    if not str(src_path).isascii():
        with _tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as _tmp:
            _tmp_path = _tmp.name
        try:
            shutil.copy2(str(src_path), _tmp_path)
            return FaissVectorStore.from_persist_path(persist_path=_tmp_path)
        finally:
            Path(_tmp_path).unlink(missing_ok=True)
    return FaissVectorStore.from_persist_dir(str(storage_dir))


def _load_storage_context(storage_dir: Path) -> StorageContext:
    if _is_faiss_vector_store(storage_dir):
        vector_store = _load_faiss_vector_store(storage_dir)
        return StorageContext.from_defaults(
            persist_dir=str(storage_dir),
            vector_store=vector_store,
        )
    return StorageContext.from_defaults(persist_dir=str(storage_dir))


def _persist_storage_context(storage_context: StorageContext, persist_dir: Path) -> None:
    """Persist a storage context, working around FAISS Unicode path issue on Windows."""
    faiss_fname = f"{DEFAULT_VECTOR_STORE}{NAMESPACE_SEP}{DEFAULT_PERSIST_FNAME}"
    has_faiss = any(
        type(vs).__name__ == "FaissVectorStore"
        for vs in storage_context.vector_stores.values()
    )
    needs_workaround = has_faiss and not str(persist_dir / faiss_fname).isascii()
    if needs_workaround:
        import tempfile as _tempfile

        with _tempfile.TemporaryDirectory() as _tmp:
            storage_context.persist(persist_dir=str(_tmp))
            persist_dir.mkdir(parents=True, exist_ok=True)
            for child in Path(_tmp).iterdir():
                _dst = persist_dir / child.name
                if child.is_dir():
                    shutil.copytree(str(child), str(_dst), dirs_exist_ok=True)
                else:
                    shutil.copy2(str(child), str(_dst))
    else:
        storage_context.persist(persist_dir=str(persist_dir))


def _preserve_embedding_model(embedding_dict: dict[str, list[float]]) -> Any:
    """Create an embed model that returns existing embeddings from a dict."""
    from llama_index.core.embeddings import BaseEmbedding

    class _PreserveEmbedding(BaseEmbedding):
        def __init__(self, embed_map: dict[str, list[float]]):
            super().__init__()
            self._embed_map = embed_map

        def _get_text_embedding(self, text: str) -> list[float]:
            return [0.0]

        def _get_query_embedding(self, query: str) -> list[float]:
            return [0.0]

        async def _aget_text_embedding(self, text: str) -> list[float]:
            return [0.0]

        async def _aget_query_embedding(self, query: str) -> list[float]:
            return [0.0]

    return _PreserveEmbedding(embedding_dict)


def convert_simple_to_faiss(
    src_storage_dir: Path,
    dst_storage_dir: Path,
) -> None:
    """Convert a SimpleVectorStore index to FaissVectorStore in a new directory.

    The source dir is never modified. If FAISS is not installed or the source
    is already in FAISS format, this is a no-op.
    """
    if not _faiss_available():
        logger.info("FAISS not available; skipping conversion.")
        return
    if _is_faiss_vector_store(src_storage_dir):
        logger.info("Source is already FAISS format; skipping conversion.")
        return

    import faiss
    from llama_index.vector_stores.faiss import FaissVectorStore
    from llama_index.core.schema import TextNode

    logger.info(
        "Converting SimpleVectorStore at %s to FaissVectorStore at %s",
        src_storage_dir,
        dst_storage_dir,
    )

    old_ctx = StorageContext.from_defaults(persist_dir=str(src_storage_dir))
    index = load_index_from_storage(old_ctx)

    old_vs = index.vector_store
    old_vs_data = getattr(old_vs, "_data", None) or getattr(old_vs, "data", None)
    embedding_dict: dict[str, list[float]] = {}
    if old_vs_data and hasattr(old_vs_data, "embedding_dict"):
        embedding_dict = old_vs_data.embedding_dict

    if not embedding_dict:
        logger.warning("No embeddings found in SimpleVectorStore; copying dir unchanged.")
        _copy_storage_dir(src_storage_dir, dst_storage_dir)
        return

    all_nodes: list[Any] = []
    for node_id, embedding in embedding_dict.items():
        node = index.docstore.get_node(node_id)
        if node is None:
            node = TextNode(text="", id_=node_id)
        node_copy = node.model_copy()
        node_copy.embedding = embedding
        all_nodes.append(node_copy)

    if not all_nodes:
        logger.warning("No nodes created from embeddings; copying dir unchanged.")
        _copy_storage_dir(src_storage_dir, dst_storage_dir)
        return

    dim = len(next(iter(embedding_dict.values())))
    faiss_index = faiss.IndexHNSWFlat(dim, 32)
    faiss_index.hnsw.efConstruction = 200
    faiss_index.hnsw.efSearch = 128

    vector_store = FaissVectorStore(faiss_index)
    dst_ctx = StorageContext.from_defaults(vector_store=vector_store)

    from llama_index.core import VectorStoreIndex

    embed_model = _preserve_embedding_model(embedding_dict)
    new_index = VectorStoreIndex(
        nodes=all_nodes,
        storage_context=dst_ctx,
        embed_model=embed_model,
        show_progress=False,
    )

    _persist_storage_context(new_index.storage_context, dst_storage_dir)

    bm25_persist_dir = src_storage_dir / "bm25_retriever"
    if bm25_persist_dir.is_dir():
        dst_bm25 = dst_storage_dir / "bm25_retriever"
        shutil.copytree(str(bm25_persist_dir), str(dst_bm25), dirs_exist_ok=True)

    logger.info("FAISS conversion complete at %s", dst_storage_dir)


def maybe_convert_to_faiss(
    kb_dir: Path,
    storage_dir: Path,
    signature: EmbeddingSignature | None,
) -> Path:
    """If storage_dir contains SimpleVectorStore, convert to FAISS in a new version dir.

    Returns the (possibly new) storage_dir to use. The original storage_dir
    is never modified.
    """
    if not _faiss_available():
        return storage_dir
    if _is_faiss_vector_store(storage_dir):
        return storage_dir

    new_storage_dir = resolve_storage_dir_for_rebuild(kb_dir, signature)
    try:
        convert_simple_to_faiss(storage_dir, new_storage_dir)
        if signature is not None:
            write_version_meta(kb_dir, signature, storage_dir=new_storage_dir)
        logger.info("FAISS conversion complete; new storage dir: %s", new_storage_dir)
        return new_storage_dir
    except Exception as exc:
        logger.warning(
            "FAISS conversion failed (%s); falling back to SimpleVectorStore at %s",
            exc,
            storage_dir,
        )
        cleanup_failed_version_dir(new_storage_dir)
        return storage_dir


def _copy_storage_dir(src: Path, dst: Path) -> None:
    for child in src.iterdir():
        if child.name == "meta.json":
            continue
        if child.is_dir():
            shutil.copytree(str(child), str(dst / child.name), dirs_exist_ok=True)
        else:
            shutil.copy2(str(child), str(dst / child.name))


def cleanup_failed_version_dir(storage_dir: Path) -> bool:
    """Remove an empty flat version dir created by a failed indexing attempt."""
    if not storage_dir.is_dir() or not storage_dir.name.startswith("version-"):
        return False
    storage_empty = not any(child for child in storage_dir.iterdir() if child.name != "meta.json")
    meta_path = storage_dir / "meta.json"
    if storage_empty and not meta_path.exists():
        shutil.rmtree(storage_dir, ignore_errors=True)
        return True
    return False


def resolve_add_storage_plan(kb_dir: Path, signature: EmbeddingSignature | None) -> AddStoragePlan:
    """Choose existing/new storage dirs for incremental adds."""
    matching_version = find_matching_version(kb_dir, signature) if signature is not None else None
    existing_storage = Path(str(matching_version["storage_path"])) if matching_version else None

    if matching_version and matching_version.get("layout") == "flat":
        return AddStoragePlan(existing_storage=existing_storage, storage_dir=existing_storage)

    if matching_version:
        return AddStoragePlan(
            existing_storage=existing_storage,
            storage_dir=resolve_storage_dir_for_write(kb_dir, signature),
        )

    fallback_storage = resolve_storage_dir_for_read(kb_dir, signature)
    existing_storage = fallback_storage
    fallback_is_flat = (
        fallback_storage is not None
        and fallback_storage.parent == kb_dir
        and fallback_storage.name.startswith("version-")
    )
    storage_dir = (
        fallback_storage if fallback_is_flat else resolve_storage_dir_for_write(kb_dir, signature)
    )
    return AddStoragePlan(existing_storage=existing_storage, storage_dir=storage_dir)


def create_index(documents: list[Any], storage_dir: Path, *, show_progress: bool = True) -> int:
    index, count = ingestion.create_index_from_documents(
        documents, storage_dir, show_progress=show_progress
    )
    _persist_storage_context(index.storage_context, storage_dir)
    retrievers.persist_bm25_retriever(index, storage_dir, top_k=20)
    return count


def insert_documents(existing_storage: Path, storage_dir: Path, documents: list[Any]) -> int:
    storage_context = _load_storage_context(existing_storage)
    index = load_index_from_storage(storage_context)
    _validate_persisted_embeddings(index, existing_storage)
    if hasattr(index, "insert_nodes"):
        count = ingestion.insert_documents_into_index(index, documents, show_progress=True)
    else:
        # Some tests use a tiny fake index that only implements insert().
        for document in documents:
            index.insert(document)
        count = len(documents)
    _persist_storage_context(index.storage_context, storage_dir)
    retrievers.persist_bm25_retriever(index, storage_dir, top_k=20)
    return count


def _validate_embedding_dict(embedding_dict: Any, *, label: str) -> None:
    if not isinstance(embedding_dict, dict) or not embedding_dict:
        return

    validate_embedding_batch(
        list(embedding_dict.values()),
        expected_count=len(embedding_dict),
        binding="llamaindex",
        model=f"persisted-index:{label}",
    )


def _iter_index_embedding_dicts(index: Any):
    """Yield embedding dictionaries exposed by loaded LlamaIndex vector stores."""
    seen: set[int] = set()

    def _yield_store(label: str, vector_store: Any):
        if vector_store is None:
            return
        store_id = id(vector_store)
        if store_id in seen:
            return
        seen.add(store_id)
        data = getattr(vector_store, "data", None)
        embedding_dict = getattr(data, "embedding_dict", None)
        if isinstance(embedding_dict, dict):
            yield label, embedding_dict

    yield from _yield_store("default", getattr(index, "vector_store", None))

    storage_context = getattr(index, "storage_context", None)
    vector_stores = getattr(storage_context, "vector_stores", None)
    if isinstance(vector_stores, dict):
        for namespace, vector_store in vector_stores.items():
            yield from _yield_store(str(namespace), vector_store)


def _embedding_dict_from_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("embedding_dict"), dict):
        return payload["embedding_dict"]
    data = payload.get("data")
    if isinstance(data, dict) and isinstance(data.get("embedding_dict"), dict):
        return data["embedding_dict"]
    return None


def _iter_file_embedding_dicts(storage_dir: Path):
    """Yield embedding dictionaries from persisted vector-store files.

    Skips FAISS binary files (which have a .json extension but cannot be
    parsed as JSON).
    """
    for path in sorted(storage_dir.glob("*vector_store.json")):
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        embedding_dict = _embedding_dict_from_payload(payload)
        if isinstance(embedding_dict, dict):
            yield path.name, embedding_dict


def _validate_persisted_embeddings(index: Any, storage_dir: Path | None = None) -> None:
    """Fail early when a persisted vector store contains unusable vectors."""
    try:
        for label, embedding_dict in _iter_index_embedding_dicts(index):
            _validate_embedding_dict(embedding_dict, label=label)
        if storage_dir is not None:
            for label, embedding_dict in _iter_file_embedding_dicts(storage_dir):
                _validate_embedding_dict(embedding_dict, label=label)
    except ValueError as exc:
        raise ValueError(
            "RAG index contains invalid embedding vectors. Re-index the "
            "knowledge base with the current embedding provider/model before "
            f"querying it again. Details: {exc}"
        ) from exc


def validate_storage_embeddings(storage_dir: Path) -> None:
    """Validate persisted vector-store files without running a retrieval."""
    _validate_persisted_embeddings(None, storage_dir)


# ---------------------------------------------------------------------------
# In-process index cache
#
# ``load_index_from_storage`` re-parses the persisted docstore + vector files
# on every call, which is the dominant retrieval cost once a KB grows. We keep
# the last-loaded ``StorageContext``/index per storage dir in memory and reload
# only when the index files change (monitored via a signature file mtime).
#
# ``_validated_mtimes`` tracks which on-disk states have already passed the
# expensive full-vector validation, so steady-state queries skip it.
# ---------------------------------------------------------------------------
_CACHE_LOCK = threading.Lock()
_INDEX_CACHE: dict[str, tuple[float, tuple[Any, Any]]] = {}   # storage_dir -> (mtime, (storage_context, index))
_VALIDATED_MTIMES: dict[str, float] = {}                      # storage_dir -> mtime already validated
_MAX_CACHED_INDEXES = 8


def _storage_cursor(storage_dir: Path) -> float:
    """Fingerprint of the on-disk index state for cache invalidation.

    Uses the newest modification time among persisted files; any KB mutation
    (add_documents, delete, rebuild) bumps it, forcing a reload.
    """
    newest = 0.0
    for p in storage_dir.rglob("*"):
        if p.is_file():
            try:
                newest = max(newest, p.stat().st_mtime)
            except OSError:
                continue
    return newest


def _cached_index(storage_dir: Path):
    """Return a cached (storage_context, index) if still fresh, else None."""
    cursor = _storage_cursor(storage_dir)
    key = str(storage_dir)
    with _CACHE_LOCK:
        entry = _INDEX_CACHE.get(key)
        if entry and entry[0] == cursor:
            return entry[1]
    return None


def _cache_index(storage_dir: Path, context_and_index: tuple[Any, Any]) -> None:
    key = str(storage_dir)
    cursor = _storage_cursor(storage_dir)
    with _CACHE_LOCK:
        if len(_INDEX_CACHE) >= _MAX_CACHED_INDEXES:
            _INDEX_CACHE.pop(next(iter(_INDEX_CACHE)), None)
        _INDEX_CACHE[key] = (cursor, context_and_index)


def _needs_validation(storage_dir: Path) -> bool:
    key = str(storage_dir)
    cursor = _storage_cursor(storage_dir)
    with _CACHE_LOCK:
        return _VALIDATED_MTIMES.get(key) != cursor


def _mark_validated(storage_dir: Path) -> None:
    key = str(storage_dir)
    cursor = _storage_cursor(storage_dir)
    with _CACHE_LOCK:
        _VALIDATED_MTIMES[key] = cursor


def clear_index_cache() -> None:
    """Drop all cached indexes + validation marks (used on KB mutation)."""
    with _CACHE_LOCK:
        _INDEX_CACHE.clear()
        _VALIDATED_MTIMES.clear()


def retrieve_nodes(storage_dir: Path, query: str, *, top_k: int = 5) -> list[Any]:
    cache_hit = False
    context_and_index = _cached_index(storage_dir)
    if context_and_index is None:
        storage_context = _load_storage_context(storage_dir)
        index = load_index_from_storage(storage_context)
        context_and_index = (storage_context, index)
        _cache_index(storage_dir, context_and_index)
    else:
        cache_hit = True
        storage_context, index = context_and_index

    # Full-vector validation runs once per on-disk state, not per query.
    if _needs_validation(storage_dir):
        _validate_persisted_embeddings(index, storage_dir)
        _mark_validated(storage_dir)

    if cache_hit:
        logger.info(f"Index cache hit for {storage_dir} (query='{query[:40]}').")
    retriever = retrievers.build_retriever(index, storage_dir, top_k=top_k)
    return retriever.retrieve(query)


def delete_kb_dir(kb_dir: Path) -> bool:
    if kb_dir.exists():
        shutil.rmtree(kb_dir)
        return True
    return False
