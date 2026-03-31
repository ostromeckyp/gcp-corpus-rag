"""
Upload plików do Google Cloud Storage.
"""

import logging
from google.cloud import storage

logger = logging.getLogger(__name__)


def upload_bytes(bucket_name: str, blob_path: str, data: bytes, content_type: str = "application/jsonl") -> str:
    """
    Wgrywa bajty do GCS i zwraca URI gs://.

    Args:
        bucket_name: Nazwa bucketu GCS.
        blob_path: Ścieżka obiektu w buckecie (np. rag-data/wydatki.jsonl).
        data: Zawartość pliku.
        content_type: MIME type.

    Returns:
        URI gs://bucket/blob_path
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_string(data, content_type=content_type)

    uri = f"gs://{bucket_name}/{blob_path}"
    logger.info(f"Wgrano {len(data)} bajtów → {uri}")
    return uri
