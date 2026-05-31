# Deploy Virtual Mouse to Google Cloud Run

## Prerequisites
- Google Cloud SDK installed and logged in (`gcloud auth login`)
- A GCP project created

## Steps

### 1. Set your project
```bash
gcloud config set project YOUR_PROJECT_ID
```

### 2. Enable required APIs
```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com containerregistry.googleapis.com
```

### 3. Deploy (one command)
```bash
gcloud builds submit --config cloudbuild.yaml .
```

That's it! Cloud Build will:
1. Build the Docker image
2. Push it to Container Registry
3. Deploy to Cloud Run

### 4. Get your URL
```bash
gcloud run services describe virtual-mouse --region=us-central1 --format='value(status.url)'
```

Open the URL in your browser — allow camera access and click **Start Camera**.

---

## Local testing (optional)
```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8080
# Open http://localhost:8080
```
