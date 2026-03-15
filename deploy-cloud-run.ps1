param(
  [Parameter(Mandatory = $true)][string]$ProjectId,
  [Parameter(Mandatory = $false)][string]$Region = "africa-south1",
  [Parameter(Mandatory = $false)][string]$Repository = "haki",
  [Parameter(Mandatory = $false)][string]$ImageName = "hakirag",
  [Parameter(Mandatory = $false)][string]$ServiceName = "hakirag-api",
  [Parameter(Mandatory = $false)][string]$EnvFileSecret = "HakiRAG",
  [Parameter(Mandatory = $false)][bool]$UseEnvFileSecret = $true,
  [Parameter(Mandatory = $false)][string]$MongoUriSecret = "ATLAS_MONGO_DB_URI",
  [Parameter(Mandatory = $false)][string]$GoogleApiKeySecret = "RAG_GOOGLE_API_KEY",
  [Parameter(Mandatory = $false)][string]$CollectionName = "rag",
  [Parameter(Mandatory = $false)][string]$AtlasSearchIndex = "rag",
  [Parameter(Mandatory = $false)][string]$EmbeddingProvider = "google_genai",
  [Parameter(Mandatory = $false)][string]$EmbeddingModel = "gemini-embedding-2-preview"
)

$ErrorActionPreference = "Continue"

$image = "$Region-docker.pkg.dev/$ProjectId/$Repository/${ImageName}:latest"

Write-Host "Setting gcloud project..."
gcloud config set project $ProjectId | Out-Null

Write-Host "Ensuring Artifact Registry repository exists..."
$null = gcloud artifacts repositories describe $Repository `
  --location=$Region `
  --project=$ProjectId 2>&1

if ($LASTEXITCODE -ne 0) {
  gcloud artifacts repositories create $Repository `
    --repository-format=docker `
    --location=$Region `
    --description="Haki containers" `
    --project=$ProjectId 2>&1
}

Write-Host "Building and pushing image with Cloud Build..."
gcloud builds submit . `
  --config=cloudbuild.yaml `
  "--substitutions=_REGION=$Region,_REPO=$Repository,_IMAGE=$ImageName" `
  --project=$ProjectId

if ($LASTEXITCODE -ne 0) {
  Write-Error "Cloud Build failed. Aborting deployment."
  exit 1
}

Write-Host "Deploying to Cloud Run..."
$deployArgs = @(
  $ServiceName
  "--image=$image"
  "--region=$Region"
  "--platform=managed"
  "--allow-unauthenticated"
  "--port=8000"
  "--cpu=2"
  "--memory=2Gi"
  "--min-instances=0"
  "--max-instances=10"
)

if ($UseEnvFileSecret) {
  # Mount the full .env file from Secret Manager so python-dotenv can load it automatically.
  # Your secret value should be plain dotenv content (KEY=value per line).
  $deployArgs += "--set-env-vars=RAG_HOST=0.0.0.0,RAG_PORT=8000"
  $deployArgs += "--update-secrets=/secrets/.env=${EnvFileSecret}:latest"
} else {
  $deployArgs += "--set-env-vars=VECTOR_DB_TYPE=atlas-mongo,COLLECTION_NAME=$CollectionName,ATLAS_SEARCH_INDEX=$AtlasSearchIndex,EMBEDDINGS_PROVIDER=$EmbeddingProvider,EMBEDDINGS_MODEL=$EmbeddingModel,RAG_HOST=0.0.0.0,RAG_PORT=8000"
  $deployArgs += "--set-secrets=ATLAS_MONGO_DB_URI=${MongoUriSecret}:latest,RAG_GOOGLE_API_KEY=${GoogleApiKeySecret}:latest"
}

$deployArgs += "--project=$ProjectId"

& gcloud run deploy @deployArgs

Write-Host "Done. Fetching service URL..."
gcloud run services describe $ServiceName `
  --region=$Region `
  --format='value(status.url)' `
  --project=$ProjectId
