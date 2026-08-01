<#
.SYNOPSIS
    deploy_workflow.ps1 — One-shot deployment of Cloud Workflows + Eventarc GCS trigger.

.DESCRIPTION
    Reads all confidential configuration from the root-level .env file.
    No sensitive values are hardcoded in this script.

    What this script deploys:
      - Enables required GCP APIs
      - Grants 3 IAM role bindings to the service account
      - Creates gs://<BUCKET>/<TRIGGER_PREFIX>/ folder if it does not exist
      - Deploys gcp/workflow.yaml as a Cloud Workflow
      - Creates an Eventarc trigger that fires on OBJECT_FINALIZE

    Prerequisites:
      - gcloud CLI installed and authenticated (gcloud auth login or ADC)
      - Billing enabled on the GCP project
      - Bucket gs://<GCS_BUCKET_NAME> must already exist
      - .env file present at repository root with all required variables

.EXAMPLE
    # Run from repository root in PowerShell:
    .\gcp\deploy_workflow.ps1
#>

#Requires -Version 5.1
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ─── Helper: load .env file ────────────────────────────────────────────────────
function Import-DotEnv {
    param([string]$EnvPath)

    if (-not (Test-Path $EnvPath)) {
        Write-Error ".env file not found at: $EnvPath`nCopy .env.example to .env and fill in your values."
        exit 1
    }

    $count = 0
    Get-Content $EnvPath | ForEach-Object {
        $line = $_.Trim()
        # Skip blank lines and comments
        if ($line -and -not $line.StartsWith('#')) {
            $parts = $line -split '=', 2
            if ($parts.Count -eq 2) {
                $key   = $parts[0].Trim()
                $value = $parts[1].Trim() -replace '^[''"]|[''"]$', ''
                [System.Environment]::SetEnvironmentVariable($key, $value, 'Process')
                $count++
            }
        }
    }
    Write-Host "  Loaded $count variable(s) from .env" -ForegroundColor DarkGray
}

# ─── Helper: require an env variable, exit if missing ─────────────────────────
function Require-EnvVar {
    param([string]$Name)
    $val = [System.Environment]::GetEnvironmentVariable($Name, 'Process')
    if (-not $val) {
        Write-Error "Required variable '$Name' is missing or empty in .env. Please set it and retry."
        exit 1
    }
    return $val
}

# ─── Helper: run gcloud and throw on failure ──────────────────────────────────
function Invoke-Gcloud {
    param([string[]]$Args)
    & gcloud @Args
    if ($LASTEXITCODE -ne 0) {
        throw "gcloud command failed with exit code $LASTEXITCODE`: gcloud $Args"
    }
}

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

# Resolve repository root (parent of gcp/)
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot   = Split-Path -Parent $ScriptDir
$EnvFile    = Join-Path $RepoRoot ".env"
$WorkflowSrc = Join-Path $ScriptDir "workflow.yaml"

# Load .env
Write-Host ""
Write-Host "Loading configuration from: $EnvFile" -ForegroundColor DarkGray
Import-DotEnv -EnvPath $EnvFile

# Read required variables
$ProjectId    = Require-EnvVar 'GCP_PROJECT_ID'
$Region       = Require-EnvVar 'GCP_REGION'
$Sa           = Require-EnvVar 'GCP_SERVICE_ACCOUNT'
$Bucket       = Require-EnvVar 'GCS_BUCKET_NAME'
$TriggerPrefix = Require-EnvVar 'WORKFLOW_TRIGGER_PREFIX'
$WorkflowName = Require-EnvVar 'WORKFLOW_NAME'
$TriggerName  = Require-EnvVar 'EVENTARC_TRIGGER_NAME'

# Validate workflow.yaml exists
if (-not (Test-Path $WorkflowSrc)) {
    Write-Error "workflow.yaml not found at: $WorkflowSrc"
    exit 1
}

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  Deploying Kronos Event-Driven Training Trigger" -ForegroundColor Cyan
Write-Host "  Project  : $ProjectId"
Write-Host "  Region   : $Region"
Write-Host "  Bucket   : gs://$Bucket/$TriggerPrefix/"
Write-Host "  Workflow : $WorkflowName"
Write-Host "  Trigger  : $TriggerName"
Write-Host "  SA       : $Sa"
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Enable required GCP APIs ─────────────────────────────────────────
Write-Host "=== [1/6] Enabling required GCP APIs ===" -ForegroundColor Yellow
Invoke-Gcloud @(
    'services', 'enable',
    'workflows.googleapis.com',
    'eventarc.googleapis.com',
    'storage.googleapis.com',
    'compute.googleapis.com',
    'eventarcpublishing.googleapis.com',
    "--project=$ProjectId"
)
Write-Host "  OK APIs enabled." -ForegroundColor Green

# ── Step 2: Grant IAM roles ───────────────────────────────────────────────────
Write-Host ""
Write-Host "=== [2/6] Granting IAM roles to service account ===" -ForegroundColor Yellow
Write-Host "  SA: $Sa"
Write-Host ""

# [2a] roles/workflows.invoker — Eventarc invokes the workflow via this SA
Write-Host "  [2a] Granting roles/workflows.invoker ..."
Invoke-Gcloud @(
    'projects', 'add-iam-policy-binding', $ProjectId,
    "--member=serviceAccount:$Sa",
    '--role=roles/workflows.invoker',
    '--condition=None'
)
Write-Host "       OK roles/workflows.invoker granted." -ForegroundColor Green

# [2b] roles/compute.instanceAdmin.v1 — workflow creates GCE VM instances
Write-Host "  [2b] Granting roles/compute.instanceAdmin.v1 ..."
Invoke-Gcloud @(
    'projects', 'add-iam-policy-binding', $ProjectId,
    "--member=serviceAccount:$Sa",
    '--role=roles/compute.instanceAdmin.v1',
    '--condition=None'
)
Write-Host "       OK roles/compute.instanceAdmin.v1 granted." -ForegroundColor Green

# [2c] roles/pubsub.publisher on GCS SA — required for Eventarc GCS events
Write-Host "  [2c] Resolving GCS managed service account..."
$ProjectNumber = & gcloud projects describe $ProjectId --format="value(projectNumber)"
if ($LASTEXITCODE -ne 0) { throw "Failed to resolve project number for: $ProjectId" }
$GcsSa = "service-$ProjectNumber@gs-project-accounts.iam.gserviceaccount.com"
Write-Host "       GCS SA: $GcsSa"
Write-Host "  [2c] Granting roles/pubsub.publisher to GCS SA ..."
Invoke-Gcloud @(
    'projects', 'add-iam-policy-binding', $ProjectId,
    "--member=serviceAccount:$GcsSa",
    '--role=roles/pubsub.publisher',
    '--condition=None'
)
Write-Host "       OK roles/pubsub.publisher granted to GCS SA." -ForegroundColor Green

Write-Host ""
Write-Host "  OK All IAM roles granted." -ForegroundColor Green

# ── Step 3: Ensure GCS trigger folder exists ──────────────────────────────────
Write-Host ""
Write-Host "=== [3/6] Ensuring GCS trigger folder exists ===" -ForegroundColor Yellow
$PlaceholderObject = "gs://$Bucket/$TriggerPrefix/.keep"
Write-Host "  Target: $PlaceholderObject"

$lsOutput = & gcloud storage ls $PlaceholderObject 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  OK Folder already exists — no action needed." -ForegroundColor Green
} else {
    Write-Host "  Folder not found — creating placeholder object..."
    # Write an empty temp file and upload it as the folder placeholder
    $TempFile = [System.IO.Path]::GetTempFileName()
    try {
        Set-Content -Path $TempFile -Value "" -Encoding UTF8
        Invoke-Gcloud @('storage', 'cp', $TempFile, $PlaceholderObject)
    } finally {
        Remove-Item -Path $TempFile -Force -ErrorAction SilentlyContinue
    }
    Write-Host "  OK Folder gs://$Bucket/$TriggerPrefix/ created." -ForegroundColor Green
}

# ── Step 4: Deploy Cloud Workflow ─────────────────────────────────────────────
Write-Host ""
Write-Host "=== [4/6] Deploying Cloud Workflow: $WorkflowName ===" -ForegroundColor Yellow
Write-Host "  Source: $WorkflowSrc"
Invoke-Gcloud @(
    'workflows', 'deploy', $WorkflowName,
    "--project=$ProjectId",
    "--location=$Region",
    "--source=$WorkflowSrc",
    "--service-account=$Sa"
)
Write-Host "  OK Workflow deployed." -ForegroundColor Green

# ── Step 5: Create Eventarc trigger ──────────────────────────────────────────
Write-Host ""
Write-Host "=== [5/6] Creating Eventarc trigger: $TriggerName ===" -ForegroundColor Yellow
Write-Host "  Listening for OBJECT_FINALIZE on bucket: $Bucket"
Invoke-Gcloud @(
    'eventarc', 'triggers', 'create', $TriggerName,
    "--project=$ProjectId",
    "--location=$Region",
    "--destination-workflow=$WorkflowName",
    "--destination-workflow-location=$Region",
    '--event-filters=type=google.cloud.storage.object.v1.finalized',
    "--event-filters=bucket=$Bucket",
    "--service-account=$Sa"
)
Write-Host "  OK Eventarc trigger created." -ForegroundColor Green

# ── Step 6: Verify deployment ─────────────────────────────────────────────────
Write-Host ""
Write-Host "=== [6/6] Verifying deployment ===" -ForegroundColor Yellow

Write-Host ""
Write-Host "--- Cloud Workflow ---"
Invoke-Gcloud @(
    'workflows', 'describe', $WorkflowName,
    "--project=$ProjectId",
    "--location=$Region",
    '--format=table(name,state,updateTime)'
)

Write-Host ""
Write-Host "--- Eventarc Trigger ---"
Invoke-Gcloud @(
    'eventarc', 'triggers', 'describe', $TriggerName,
    "--project=$ProjectId",
    "--location=$Region"
)

Write-Host ""
Write-Host "======================================================" -ForegroundColor Green
Write-Host "  Deployment complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  Upload a CSV to trigger automatic GPU training:"
Write-Host "  gs://$Bucket/$TriggerPrefix/BNB_BTC_1m.csv"
Write-Host ""
Write-Host "  Required filename format : SYMBOL_PAIR*.csv"
Write-Host "  Valid   : BNB_BTC_1m.csv, ETHUSDT_4h.csv"
Write-Host "  Invalid : checkpoint.csv, raw-data.csv  (job is skipped)"
Write-Host ""
Write-Host "  launch_vertex_job.py is UNAFFECTED."
Write-Host "  (It uploads to $GCS_DESTINATION_PREFIX/, NOT $TriggerPrefix/)"
Write-Host "======================================================" -ForegroundColor Green
