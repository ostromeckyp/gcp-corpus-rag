FROM python:3.12-slim

WORKDIR /app

# Zależności
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kod aplikacji
COPY main.py .
COPY services/ services/

# Cloud Run ustawia PORT (domyślnie 8080)
ENV PORT=8080

EXPOSE ${PORT}

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
