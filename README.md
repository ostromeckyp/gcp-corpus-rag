# Expense Classifier API

FastAPI backend for expense classification using **Vertex AI RAG** and **Gemini**.
It fetches training data (CSV) from Google Drive, converts it to JSONL, stores it in GCS,
and imports it into a Vertex AI RAG Corpus. The `/classify` endpoint performs RAG retrieval
and classifies transaction descriptions using the Gemini model.

## Architecture

```
Google Drive (CSV) → /sync → GCS (JSONL) → Vertex AI RAG Corpus
                                                     ↓
                     /classify ← Gemini ← RAG retrieval
```

## Required Environment Variables

| Variable               | Description                                | Example           |
|------------------------|--------------------------------------------|-------------------|
| `GOOGLE_CLOUD_PROJECT` | GCP project ID                             | `my-project-123`  |
| `GOOGLE_CLOUD_LOCATION`| GCP region                                 | `us-central1`     |
| `API_KEY`              | Authorization key (X-API-Key header)       | `secret-key`      |
| `RAG_CORPUS_DEFAULT`   | Default RAG corpus name                    | `expenses-en`     |
| `GCS_BUCKET`           | GCS bucket name                            | `my-rag-bucket`   |
| `GCS_PREFIX`           | Prefix in the bucket                       | `rag-data/`       |
| `GEMINI_MODEL`         | Gemini model (optional)                    | `gemini-2.5-flash` |
| `TOP_K`                | Number of retrieval results (optional)     | `5`               |

## Local Setup

```bash
# Set environment variables
export GOOGLE_CLOUD_PROJECT=my-project
export GOOGLE_CLOUD_LOCATION=us-central1
export API_KEY=test-key
export GCS_BUCKET=my-bucket
export GCS_PREFIX=rag-data/
export RAG_CORPUS_DEFAULT=expenses-en

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn main:app --reload --port 8080
```

## Deploy to Cloud Run

```bash
# Build the image
gcloud builds submit --tag gcr.io/$GOOGLE_CLOUD_PROJECT/expense-classifier

# Deploy
gcloud run deploy expense-classifier \
  --image gcr.io/$GOOGLE_CLOUD_PROJECT/expense-classifier \
  --region $GOOGLE_CLOUD_LOCATION \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT,GOOGLE_CLOUD_LOCATION=$GOOGLE_CLOUD_LOCATION,API_KEY=$API_KEY,GCS_BUCKET=$GCS_BUCKET,GCS_PREFIX=$GCS_PREFIX,RAG_CORPUS_DEFAULT=$RAG_CORPUS_DEFAULT" \
  --allow-unauthenticated
```

> **Note:** The Cloud Run service account must have permissions for:
> - Google Drive API (read access to files shared with the service account)
> - Google Cloud Storage (write access to the bucket)
> - Vertex AI (RAG API, Gemini)

## Endpoints

### `GET /health`
Health check.

### `POST /sync`
Synchronizes CSV data from Google Drive to a Vertex AI RAG Corpus.

**Request:**
```bash
curl -X POST http://localhost:8080/sync \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-key" \
  -d '{
    "corpus_name": "expenses-en",
    "drive_files": [
      {"file_id": "1AbCdEfGhIjKlMnOpQrStUvWxYz", "name": "expenses.csv"},
      {"file_id": "2BcDeFgHiJkLmNoPqRsTuVwXyZa", "name": "other.csv"}
    ]
  }'
```

**With corpus reset:**
```bash
curl -X POST "http://localhost:8080/sync?reset=true" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-key" \
  -d '{
    "drive_files": [
      {"file_id": "1AbCdEfGhIjKlMnOpQrStUvWxYz", "name": "expenses.csv"}
    ]
  }'
```

**Response:**
```json
{
  "status": "ok",
  "corpus_name": "expenses-en",
  "corpus_resource_name": "projects/my-project/locations/us-central1/ragCorpora/123456",
  "total_records": 150,
  "files_gcs": ["gs://my-bucket/rag-data/expenses.jsonl"],
  "files_imported": 1
}
```

### `POST /classify`
Classifies transaction descriptions into expense categories.

**Allowed categories:** Car, Entertainment, Food, Home, Health, Personal Items, Travel, Media, Other, Credit, Education, Taxes.

**Request:**
```bash
curl -X POST http://localhost:8080/classify \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-key" \
  -d '{
    "descriptions": [
      "BIEDRONKA 123 WROCLAW",
      "ORLEN STACJA 456",
      "NETFLIX.COM",
      "APTEKA CENTRUM"
    ]
  }'
```

**Response:**
```json
{
  "categories": ["Food", "Car", "Media", "Health"]
}
```

## CSV Format

The CSV file must contain `opis` and `kategoria` columns (case-insensitive):

```csv
opis,kategoria
BIEDRONKA 123 WROCLAW,Food
ORLEN STACJA 456,Car
NETFLIX.COM,Media
```

## Project Structure

```
├── main.py                  # FastAPI – /sync, /classify, /health endpoints
├── services/
│   ├── __init__.py
│   ├── drive.py             # Fetching files from Google Drive
│   ├── convert.py           # CSV → JSONL + description normalization
│   ├── gcs.py               # Upload to Google Cloud Storage
│   ├── vertex_rag.py        # Vertex AI RAG: corpus CRUD, import, query
│   └── classifier.py        # Prompt + Gemini invocation
├── requirements.txt
├── Dockerfile
└── README.md
```
