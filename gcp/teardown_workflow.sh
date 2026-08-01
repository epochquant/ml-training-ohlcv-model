#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# gcp/teardown_workflow.sh
# Decommissions the event-driven training trigger (Cloud Workflows + Eventarc).
#
# This script ONLY removes:
#   - The Eventarc trigger: trigger-kronos-gcs-workflow-upload
#   - The Cloud Workflow : kronos-training-workflow
#
# It does NOT touch:
#   - launch_vertex_job.py  (manual launcher, fully preserved)
#   - launch_container_job.py
#   - The GCS bucket or any stored data
#   - Any running VMs or training jobs
#   - IAM bindings (retained for future redeployment)
#
# Run from repository root:
#   bash gcp/teardown_workflow.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_ID="dev-gemini-ai"
REGION="us-central1"
TRIGGER_NAME="trigger-kronos-gcs-workflow-upload"
WORKFLOW_NAME="kronos-training-workflow"

echo "======================================================"
echo "  Kronos Event-Driven Trigger — Teardown"
echo "  Project  : ${PROJECT_ID}"
echo "  Region   : ${REGION}"
echo "======================================================"
echo ""
echo "WARNING: This will permanently delete:"
echo "  - Eventarc trigger : ${TRIGGER_NAME}"
echo "  - Cloud Workflow   : ${WORKFLOW_NAME}"
echo ""
read -rp "Proceed? [y/N] " confirm
if [[ "${confirm}" != "y" && "${confirm}" != "Y" ]]; then
  echo "Teardown cancelled."
  exit 0
fi
echo ""

# ── Remove Eventarc trigger ───────────────────────────────────────────────────
echo "=== [1/2] Deleting Eventarc trigger: ${TRIGGER_NAME} ==="
if gcloud eventarc triggers describe "${TRIGGER_NAME}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" > /dev/null 2>&1; then
  gcloud eventarc triggers delete "${TRIGGER_NAME}" \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --quiet
  echo "  ✅ Eventarc trigger deleted."
else
  echo "  ⚠️  Trigger not found — already deleted or never deployed."
fi

# ── Remove Cloud Workflow ─────────────────────────────────────────────────────
echo ""
echo "=== [2/2] Deleting Cloud Workflow: ${WORKFLOW_NAME} ==="
if gcloud workflows describe "${WORKFLOW_NAME}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" > /dev/null 2>&1; then
  gcloud workflows delete "${WORKFLOW_NAME}" \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --quiet
  echo "  ✅ Cloud Workflow deleted."
else
  echo "  ⚠️  Workflow not found — already deleted or never deployed."
fi

echo ""
echo "======================================================"
echo "  ✅ Teardown complete."
echo ""
echo "  launch_vertex_job.py remains fully functional."
echo "  To redeploy, run: bash gcp/deploy_workflow.sh"
echo "======================================================"
