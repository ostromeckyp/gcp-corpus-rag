# 1. Build the Docker image
gcloud builds submit --tag gcr.io/vertex-ai-apps-script-488713/expense-classifier

# 2. Deploy to Cloud Run
gcloud run deploy expense-classifier \
  --image gcr.io/vertex-ai-apps-script-488713/expense-classifier \
  --region=europe-central2 \
  --platform=managed \
  --allow-unauthenticated