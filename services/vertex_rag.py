"""
Operacje na Vertex AI RAG: tworzenie/pobieranie corpusa, import plików, retrieval query.
Wzorce zaczerpnięte z istniejących tools/ (bez zależności od google.adk).
"""

import logging
from typing import List, Optional

from vertexai import rag

logger = logging.getLogger(__name__)

# ── Ustawienia domyślne ──────────────────────────────────────────────
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_EMBEDDING_MODEL = "publishers/google/models/text-embedding-005"
DEFAULT_EMBEDDING_REQUESTS_PER_MIN = 1000
DEFAULT_DISTANCE_THRESHOLD = 0.5


# ── Pomocnicze ────────────────────────────────────────────────────────

def _corpus_resource_name(display_name: str, project: str, location: str) -> Optional[str]:
    """Zwraca pełny resource_name corpusa o podanej display_name lub None."""
    try:
        for c in rag.list_corpora():
            if getattr(c, "display_name", None) == display_name:
                return c.name
    except Exception as e:
        logger.warning(f"Błąd listowania corpusów: {e}")
    return None


# ── CRUD corpusa ──────────────────────────────────────────────────────

def get_or_create_corpus(display_name: str, project: str, location: str) -> str:
    """
    Zwraca resource_name istniejącego corpusa lub tworzy nowy.

    Returns:
        Pełny resource_name corpusa.
    """
    existing = _corpus_resource_name(display_name, project, location)
    if existing:
        logger.info(f"Corpus '{display_name}' już istnieje: {existing}")
        return existing

    logger.info(f"Tworzę nowy corpus: {display_name}")
    corpus = rag.create_corpus(
        display_name=display_name,
        description=f"RAG corpus: {display_name}",
    )
    logger.info(f"Utworzono corpus: {corpus.name}")
    return corpus.name


def delete_corpus(display_name: str, project: str, location: str) -> bool:
    """Usuwa corpus o podanej display_name. Zwraca True jeśli usunięto."""
    resource_name = _corpus_resource_name(display_name, project, location)
    if not resource_name:
        logger.warning(f"Corpus '{display_name}' nie znaleziony – nic do usunięcia")
        return False

    logger.info(f"Usuwam corpus: {resource_name}")
    rag.delete_corpus(name=resource_name)
    return True


# ── Import plików ─────────────────────────────────────────────────────

def import_files(corpus_resource_name: str, gcs_uris: List[str]) -> int:
    """
    Importuje pliki z GCS do corpusa RAG.

    Returns:
        Liczba zaimportowanych plików.
    """
    logger.info(f"Importuję {len(gcs_uris)} plików do {corpus_resource_name}")

    transformation_config = rag.TransformationConfig(
        chunking_config=rag.ChunkingConfig(
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        ),
    )

    result = rag.import_files(
        corpus_resource_name,
        gcs_uris,
        transformation_config=transformation_config,
        max_embedding_requests_per_min=DEFAULT_EMBEDDING_REQUESTS_PER_MIN,
    )

    count = result.imported_rag_files_count
    logger.info(f"Zaimportowano {count} plików")
    return count


# ── Retrieval query ───────────────────────────────────────────────────

def retrieval_query(corpus_resource_name: str, text: str, top_k: int = 5) -> List[dict]:
    """
    Wykonuje retrieval query na corpusie RAG.

    Returns:
        Lista wyników: [{"text": ..., "score": ..., "source_uri": ...}, ...]
    """
    rag_retrieval_config = rag.RagRetrievalConfig(
        top_k=top_k,
        filter=rag.Filter(vector_distance_threshold=DEFAULT_DISTANCE_THRESHOLD),
    )

    response = rag.retrieval_query(
        rag_resources=[
            rag.RagResource(rag_corpus=corpus_resource_name),
        ],
        text=text,
        rag_retrieval_config=rag_retrieval_config,
    )

    results = []
    if hasattr(response, "contexts") and response.contexts:
        for ctx in response.contexts.contexts:
            results.append({
                "text": getattr(ctx, "text", ""),
                "score": getattr(ctx, "score", 0.0),
                "source_uri": getattr(ctx, "source_uri", ""),
            })
    logger.info(f"Retrieval query zakończony sukcesem, wyniki: {results[:5]}")
    return results
