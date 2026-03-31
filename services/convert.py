"""
Konwersja CSV → JSONL z normalizacją opisów transakcji.
"""

import csv
import io
import json
import logging
import re
from typing import List

logger = logging.getLogger(__name__)

# Dozwolone kategorie
CATEGORIES = [
    "Samochód", "Rozrywka", "Jedzenie", "Dom", "Zdrowie", "Edukacja",
    "Rzeczy osobiste", "Podróże", "Media", "Inne", "Kredyt", "Podatki",
]


def normalize_description(desc: str) -> str:
    """
    Normalizacja opisu transakcji:
    - trim + uppercase
    - usuń ciągi ≥3 cyfr (ID, numery kart itp.)
    - zamień wielokrotne spacje na jedną
    """
    text = desc.strip().upper()
    text = re.sub(r"\b\d{3,}\b", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def csv_bytes_to_records(raw: bytes, source_name: str = "") -> List[dict]:
    """
    Parsuje surowe bajty CSV na listę rekordów JSONL.

    CSV musi mieć kolumny: opis, kategoria (wielkość liter dowolna).

    Returns:
        Lista słowników gotowych do serializacji jako JSONL.
    """
    # Próbuj zdekodować UTF-8, potem latin-1
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Nie udało się zdekodować pliku {source_name}")

    # Automatyczne wykrywanie separatora (przecinek lub średnik)
    try:
        dialect = csv.Sniffer().sniff(text, delimiters=",;")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","

    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)

    # Normalizuj nazwy kolumn (strip + lower)
    if reader.fieldnames is None:
        raise ValueError(f"Plik {source_name} nie zawiera nagłówków CSV")

    col_map = {col.strip().lower(): col for col in reader.fieldnames}
    opis_col = col_map.get("opis")
    kat_col = col_map.get("kategoria")

    if opis_col is None or kat_col is None:
        raise ValueError(
            f"Plik {source_name}: brak wymaganych kolumn 'opis' i 'kategoria'. "
            f"Znaleziono: {list(reader.fieldnames)}"
        )

    records = []
    for i, row in enumerate(reader):
        opis = (row.get(opis_col) or "").strip()
        kategoria = (row.get(kat_col) or "").strip()

        if not opis:
            logger.warning(f"{source_name} wiersz {i+2}: pusty opis – pomijam")
            continue

        opis_norm = normalize_description(opis)

        record = {
            "text": f"Opis transakcji: {opis}. Poprawna kategoria: {kategoria}.",
            "opis": opis,
            "kategoria": kategoria,
            "opis_norm": opis_norm,
        }
        records.append(record)

    logger.info(f"Skonwertowano {len(records)} rekordów z {source_name}")
    return records


def records_to_jsonl(records: List[dict]) -> bytes:
    """Serializuje listę rekordów do JSONL (bajty UTF-8)."""
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    return ("\n".join(lines) + "\n").encode("utf-8")
