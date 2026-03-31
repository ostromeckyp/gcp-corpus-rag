
"""
Pobieranie plików CSV z Google Drive (Drive API v3).
Używa ADC (Application Default Credentials) – na Cloud Run działa automatycznie.
"""

import logging
from io import BytesIO

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import google.auth

logger = logging.getLogger(__name__)


def _get_drive_service():
    """Tworzy klienta Drive API v3 z domyślnymi credentials (ADC)."""
    creds, project = google.auth.default(scopes=["https://www.googleapis.com/auth/drive.readonly"])
    
    # Debug: log which credentials are being used
    if hasattr(creds, 'service_account_email'):
        logger.info(f"Using service account: {creds.service_account_email}")
    else:
        logger.info(f"Using credentials type: {type(creds).__name__}, project: {project}")
    
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def download_file(file_id: str) -> bytes:
    """
    Pobiera zawartość pliku z Google Drive po file_id (alt=media).

    Args:
        file_id: Identyfikator pliku na Google Drive.

    Returns:
        Surowe bajty pliku.

    Raises:
        Exception: Gdy plik nie istnieje lub brak dostępu.
    """
    logger.info(f"Pobieranie pliku z Drive: {file_id}")
    service = _get_drive_service()
    
    # Try to get file metadata first to see if we have any access
    try:
        metadata = service.files().get(fileId=file_id, fields="id,name,mimeType").execute()
        logger.info(f"File accessible - name: {metadata.get('name')}, type: {metadata.get('mimeType')}")
    except Exception as e:
        logger.error(f"Cannot access file metadata: {e}")
        raise Exception(f"File not accessible. Make sure the file is shared with the service account. Error: {e}")

    request = service.files().get_media(fileId=file_id)
    buffer = BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)

    done = False
    while not done:
        _, done = downloader.next_chunk()

    content = buffer.getvalue()
    logger.info(f"Pobrano {len(content)} bajtów z Drive (file_id={file_id})")
    return content