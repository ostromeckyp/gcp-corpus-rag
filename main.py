"""
FastAPI backend do klasyfikacji wydatków z wykorzystaniem Vertex AI RAG + Gemini.
Deploy: Google Cloud Run.
"""

import logging
import os
from typing import List, Optional

import vertexai
from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel

from services.drive import download_file
from services.convert import csv_bytes_to_records, records_to_jsonl, normalize_description
from services.gcs import upload_bytes
from services.vertex_rag import (
    get_or_create_corpus,
    delete_corpus,
    import_files,
    retrieval_query,
)
from services.classifier import classify_batch

# ── Konfiguracja ──────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
API_KEY = os.environ.get("API_KEY", "")
RAG_CORPUS_DEFAULT = os.environ.get("RAG_CORPUS_DEFAULT", "wydatki-pl")
GCS_BUCKET = os.environ.get("GCS_BUCKET", "")
GCS_PREFIX = os.environ.get("GCS_PREFIX", "rag-data/")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
TOP_K = int(os.environ.get("TOP_K", "5"))

# ── Vertex AI init ────────────────────────────────────────────────────

vertexai.init(project=PROJECT, location=LOCATION)
logger.info(f"Vertex AI zainicjalizowany: project={PROJECT}, location={LOCATION}")

# ── FastAPI app ───────────────────────────────────────────────────────

app = FastAPI(title="Expense Classifier API", version="1.0.0")


# ── Auth middleware ───────────────────────────────────────────────────

def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Prosta autoryzacja: nagłówek X-API-Key musi zgadzać się z env API_KEY."""
    if not API_KEY:
        return  # brak klucza w env = autoryzacja wyłączona
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Nieprawidłowy lub brakujący klucz API (X-API-Key)")


# ── Modele request/response ──────────────────────────────────────────

class DriveFile(BaseModel):
    file_id: str
    name: str = "file.csv"

class SyncRequest(BaseModel):
    corpus_name: Optional[str] = None
    drive_files: List[DriveFile]

class SyncResponse(BaseModel):
    status: str
    corpus_name: str
    corpus_resource_name: str
    total_records: int
    files_gcs: List[str]
    files_imported: int

class ClassifyRequest(BaseModel):
    descriptions: List[str]
    corpus_name: Optional[str] = None

class ClassifyResponse(BaseModel):
    categories: List[str]


# ── Endpointy ─────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/sync", response_model=SyncResponse)
def sync(
    body: SyncRequest,
    reset: bool = Query(False, description="Jeśli true – usuwa corpus i tworzy od nowa"),
    x_api_key: Optional[str] = Header(None),
):
    """
    Pobiera CSV z Google Drive, konwertuje do JSONL, wgrywa do GCS,
    importuje do Vertex AI RAG Corpus.
    """
    verify_api_key(x_api_key)

    corpus_name = body.corpus_name or RAG_CORPUS_DEFAULT
    logger.info(f"[SYNC] Start – corpus={corpus_name}, reset={reset}, pliki={len(body.drive_files)}")

    # Opcjonalny reset corpusa
    if reset:
        logger.info(f"[SYNC] Reset corpusa '{corpus_name}'")
        try:
            delete_corpus(corpus_name, PROJECT, LOCATION)
        except Exception as e:
            logger.warning(f"[SYNC] Błąd usuwania corpusa: {e}")

    # Utwórz / pobierz corpus
    try:
        corpus_rn = get_or_create_corpus(corpus_name, PROJECT, LOCATION)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd tworzenia/pobierania corpusa: {e}")

    all_records = []
    gcs_uris = []

    for df in body.drive_files:
        # 1. Pobierz CSV z Drive
        try:
            raw = download_file(df.file_id)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Błąd pobierania pliku '{df.name}' (id={df.file_id}) z Drive: {e}")

        # 2. Konwertuj CSV → rekordy
        try:
            records = csv_bytes_to_records(raw, source_name=df.name)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        all_records.extend(records)

        # 3. Serializuj do JSONL i wgraj do GCS
        jsonl_bytes = records_to_jsonl(records)
        blob_name = df.name.rsplit(".", 1)[0] + ".jsonl"
        blob_path = f"{GCS_PREFIX}{blob_name}"

        try:
            uri = upload_bytes(GCS_BUCKET, blob_path, jsonl_bytes)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Błąd uploadu do GCS: {e}")

        gcs_uris.append(uri)
        logger.info(f"[SYNC] {df.name} → {uri} ({len(records)} rekordów)")

    # 4. Import do RAG
    try:
        imported = import_files(corpus_rn, gcs_uris)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd importu do RAG: {e}")

    logger.info(f"[SYNC] Zakończono – {len(all_records)} rekordów, {imported} plików zaimportowanych")

    return SyncResponse(
        status="ok",
        corpus_name=corpus_name,
        corpus_resource_name=corpus_rn,
        total_records=len(all_records),
        files_gcs=gcs_uris,
        files_imported=imported,
    )


@app.post("/classify", response_model=ClassifyResponse)
def classify(
    body: ClassifyRequest,
    x_api_key: Optional[str] = Header(None),
):
    """
    Klasyfikuje listę opisów transakcji do kategorii wydatków.
    Retrieval per opis, jedno wywołanie LLM dla całej listy.
    """
    verify_api_key(x_api_key)

    if len(body.descriptions) > 200:
        raise HTTPException(status_code=400, detail="Maksymalna liczba opisów to 200")

    if not body.descriptions:
        return ClassifyResponse(categories=[])

    corpus_name = body.corpus_name or RAG_CORPUS_DEFAULT
    logger.info(f"[CLASSIFY] Start – {len(body.descriptions)} opisów, corpus={corpus_name}")

    # Pobierz resource_name corpusa
    try:
        corpus_rn = get_or_create_corpus(corpus_name, PROJECT, LOCATION)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd dostępu do corpusa: {e}")

    # Retrieval per opis (z normalizacją)
    examples_per_desc = []
    for desc in body.descriptions:
        norm = normalize_description(desc)
        try:
            results = retrieval_query(corpus_rn, norm, top_k=TOP_K)
        except Exception as e:
            logger.warning(f"[CLASSIFY] Błąd retrieval dla '{desc}': {e}")
            results = []
        examples_per_desc.append(results)

    # Jedno wywołanie Gemini
    try:
        categories = classify_batch(
            descriptions=body.descriptions,
            examples_per_desc=examples_per_desc,
            model_name=GEMINI_MODEL,
            project=PROJECT,
            location=LOCATION,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd klasyfikacji (Gemini): {e}")

    logger.info(f"[CLASSIFY] Wynik: {categories}")
    return ClassifyResponse(categories=categories)
