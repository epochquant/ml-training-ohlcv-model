#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# gcp/deploy_workflow.sh
# One-shot deployment: Cloud Workflows + Eventarc GCS trigger
#
# Prerequisites:
#   - gcloud CLI installed and authenticated (gcloud auth login or ADC)
#   - Billing enabled on project dev-gemini-ai
#   - Bucket gs://epochquant-training must already exist
#
# Run from repository root:
#   bash gcp/deploy_workflow.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ID="dev-gemini-ai"
REGION="us-central1"
BUCKET="epochquant-training"
TRIGGER_PREFIX="training-data-workflow"
SA="kronos-notebook-sa@dev-gemini-ai.iam.gserviceaccount.com"
WORKFLOW_NAME="kronos-training-workflow"
TRIGGER_NAME="trigger-kronos-gcs-workflow-upload"
# Resolve workflow.yaml path relative to this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_SRC="${SCRIPT_DIR}/workflow.yaml"

echo "======================================================"
echo "  Deploying Kronos Event-Driven Training Trigger"
echo "  Project  : ${PROJECT_ID}"
echo "  Region   : ${REGION}"
echo "  Bucket   : gs://${BUCKET}/${TRIGGER_PREFIX}/"
echo "  Workflow : ${WORKFLOW_NAME}"
echo "  Trigger  : ${TRIGGER_NAME}"
echo "======================================================"
echo ""

# ── Step 1: Enable required GCP APIs ─────────────────────────────────────────
echo "=== [1/6] Enabling required GCP APIs ==="
gcloud services enable \
  workflows.googleapis.com \
  eventarc.googleapis.com \
  storage.googleapis.com \
  compute.googleapis.com \
  eventarcpublishing.googleapis.com \
  --project="${PROJECT_ID}"
echo "  ✅ APIs enabled."

# ── Step 2: Grant IAM roles to Service Account ───────────────────────────────
echo ""
echo "=== [2/6] Granting IAM roles to service account ==="
echo "  Service Account: ${SA}"
echo ""

# Role 1: Allow Eventarc to invoke the Cloud Workflow on behalf of this SA
echo "  [2a] Granting roles/workflows.invoker ..."
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA}" \
  --role="roles/workflows.invoker" \
  --condition=None
echo "       ✅ roles/workflows.invoker granted."

# Role 2: Allow the Cloud Workflow to create Compute Engine VM instances
echo "  [2b] Granting roles/compute.instanceAdmin.v1 ..."
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA}" \
  --role="roles/compute.instanceAdmin.v1" \
  --condition=None
echo "       ✅ roles/compute.instanceAdmin.v1 granted."

# Role 3: Allow the GCS-managed service account to publish Pub/Sub messages
#         This is required for Eventarc to receive GCS OBJECT_FINALIZE events.
#         The GCS SA is in the format: service-<PROJECT_NUMBER>@gs-project-accounts.iam.gserviceaccount.com
echo "  [2c] Resolving GCS service account project number..."
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")
GCS_SA="service-${PROJECT_NUMBER}@gs-project-accounts.iam.gserviceaccount.com"
echo "       GCS SA: ${GCS_SA}"
echo "  [2c] Granting roles/pubsub.publisher to GCS SA ..."
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${GCS_SA}" \
  --role="roles/pubsub.publisher" \
  --condition=None
echo "       ✅ roles/pubsub.publisher granted to GCS SA."

echo ""
echo "  ✅ All IAM roles granted."

# ── Step 3: Create GCS trigger folder if it does not exist ───────────────────
echo ""
echo "=== [3/6] Ensuring GCS trigger folder exists ==="
echo "  Target: gs://${BUCKET}/${TRIGGER_PREFIX}/"
PLACEHOLDER_OBJECT="gs://${BUCKET}/${TRIGGER_PREFIX}/.keep"

if gcloud storage ls "${PLACEHOLDER_OBJECT}" > /dev/null 2>&1; then
  echo "  ✅ Folder already exists — no action needed."
else
  echo "  Folder not found — creating placeholder object..."
  echo -n "" | gcloud storage cp - "${PLACEHOLDER_OBJECT}"
  echo "  ✅ Folder gs://${BUCKET}/${TRIGGER_PREFIX}/ created."
fi

# ── Step 4: Deploy the Cloud Workflow ────────────────────────────────────────
echo ""
echo "=== [4/6] Deploying Cloud Workflow: ${WORKFLOW_NAME} ==="
echo "  Source: ${WORKFLOW_SRC}"
gcloud workflows deploy "${WORKFLOW_NAME}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --source="${WORKFLOW_SRC}" \
  --service-account="${SA}"
echo "  ✅ Workflow deployed."

# ── Step 5: Create the Eventarc GCS trigger ──────────────────────────────────
echo ""
echo "=== [5/6] Creating Eventarc trigger: ${TRIGGER_NAME} ==="
echo "  Listening for OBJECT_FINALIZE on bucket: ${BUCKET}"
gcloud eventarc triggers create "${TRIGGER_NAME}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --destination-workflow="${WORKFLOW_NAME}" \
  --destination-workflow-location="${REGION}" \
  --event-filters="type=google.cloud.storage.object.v1.finalized" \
  --event-filters="bucket=${BUCKET}" \
  --service-account="${SA}"
echo "  ✅ Eventarc trigger created."

# ── Step 6: Verify deployment ─────────────────────────────────────────────────
echo ""
echo "=== [6/6] Verifying deployment ==="

echo ""
echo "--- Cloud Workflow ---"
gcloud workflows describe "${WORKFLOW_NAME}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --format="table(name,state,updateTime)"

echo ""
echo "--- Eventarc Trigger ---"
gcloud eventarc triggers describe "${TRIGGER_NAME}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}"

echo ""
echo "======================================================"
echo "  ✅ Deployment complete!"
echo ""
echo "  Upload a CSV to trigger automatic GPU training:"
echo "  gs://${BUCKET}/${TRIGGER_PREFIX}/BNB_BTC_1m.csv"
echo ""
echo "  Required filename format : SYMBOL_PAIR*.csv"
echo "  Valid   : BNB_BTC_1m.csv, ETHUSDT_4h.csv"
echo "  Invalid : checkpoint.csv, raw-data.csv (job is skipped)"
echo ""
echo "  Manual launch_vertex_job.py is UNAFFECTED."
echo "  (It uploads to training-data/, NOT training-data-workflow/)"
echo "======================================================"
