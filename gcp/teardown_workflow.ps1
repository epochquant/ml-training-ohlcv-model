<#
.SYNOPSIS
    teardown_workflow.ps1 — Safely removes the Cloud Workflows + Eventarc event-driven trigger.

.DESCRIPTION
    Reads all confidential configuration from the root-level .env file.
    No sensitive values are hardcoded in this script.

    This script ONLY removes:
      - The Eventarc trigger (EVENTARC_TRIGGER_NAME from .env)
      - The Cloud Workflow  (WORKFLOW_NAME from .env)

    This script does NOT touch:
      - launch_vertex_job.py  (manual launcher, fully preserved)
      - launch_container_job.py
      - The GCS bucket or any stored data / models / logs
      - Any running VMs or active training jobs
      - IAM bindings (retained so redeployment requires no re-granting)

    Prerequisites:
      - gcloud CLI installed and authenticated
      - .env file present at repository root with all required variables

.EXAMPLE
    # Run from repository root in PowerShell:
    .\gcp\teardown_workflow.ps1
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

# ─── Helper: check if a gcloud resource exists (suppress output) ───────────────
function Test-GcloudResource {
    param([string[]]$Args)
    & gcloud @Args 2>&1 | Out-Null
    return ($LASTEXITCODE -eq 0)
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
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir
$EnvFile   = Join-Path $RepoRoot ".env"

# Load .env
Write-Host ""
Write-Host "Loading configuration from: $EnvFile" -ForegroundColor DarkGray
Import-DotEnv -EnvPath $EnvFile

# Read required variables
$ProjectId    = Require-EnvVar 'GCP_PROJECT_ID'
$Region       = Require-EnvVar 'GCP_REGION'
$WorkflowName = Require-EnvVar 'WORKFLOW_NAME'
$TriggerName  = Require-EnvVar 'EVENTARC_TRIGGER_NAME'

# ─── Banner ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  Kronos Event-Driven Trigger — Teardown" -ForegroundColor Cyan
Write-Host "  Project  : $ProjectId"
Write-Host "  Region   : $Region"
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "WARNING: This will permanently delete:" -ForegroundColor Yellow
Write-Host "  - Eventarc trigger : $TriggerName" -ForegroundColor Yellow
Write-Host "  - Cloud Workflow   : $WorkflowName" -ForegroundColor Yellow
Write-Host ""
Write-Host "This does NOT affect:" -ForegroundColor DarkGray
Write-Host "  - launch_vertex_job.py (manual launcher)" -ForegroundColor DarkGray
Write-Host "  - GCS bucket data, models, or logs" -ForegroundColor DarkGray
Write-Host "  - IAM role bindings" -ForegroundColor DarkGray
Write-Host ""

# ─── Confirmation prompt ──────────────────────────────────────────────────────
$confirm = Read-Host "Proceed? [y/N]"
if ($confirm -ne 'y' -and $confirm -ne 'Y') {
    Write-Host ""
    Write-Host "Teardown cancelled. No changes were made." -ForegroundColor DarkGray
    exit 0
}
Write-Host ""

# ── Step 1: Delete Eventarc trigger ──────────────────────────────────────────
Write-Host "=== [1/2] Deleting Eventarc trigger: $TriggerName ===" -ForegroundColor Yellow

$triggerExists = Test-GcloudResource @(
    'eventarc', 'triggers', 'describe', $TriggerName,
    "--project=$ProjectId",
    "--location=$Region"
)

if ($triggerExists) {
    Invoke-Gcloud @(
        'eventarc', 'triggers', 'delete', $TriggerName,
        "--project=$ProjectId",
        "--location=$Region",
        '--quiet'
    )
    Write-Host "  OK Eventarc trigger deleted." -ForegroundColor Green
} else {
    Write-Host "  WARN Trigger not found — already deleted or never deployed." -ForegroundColor DarkYellow
}

# ── Step 2: Delete Cloud Workflow ─────────────────────────────────────────────
Write-Host ""
Write-Host "=== [2/2] Deleting Cloud Workflow: $WorkflowName ===" -ForegroundColor Yellow

$workflowExists = Test-GcloudResource @(
    'workflows', 'describe', $WorkflowName,
    "--project=$ProjectId",
    "--location=$Region"
)

if ($workflowExists) {
    Invoke-Gcloud @(
        'workflows', 'delete', $WorkflowName,
        "--project=$ProjectId",
        "--location=$Region",
        '--quiet'
    )
    Write-Host "  OK Cloud Workflow deleted." -ForegroundColor Green
} else {
    Write-Host "  WARN Workflow not found — already deleted or never deployed." -ForegroundColor DarkYellow
}

# ─── Summary ──────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "======================================================" -ForegroundColor Green
Write-Host "  Teardown complete." -ForegroundColor Green
Write-Host ""
Write-Host "  launch_vertex_job.py remains fully functional."
Write-Host "  To redeploy, run: .\gcp\deploy_workflow.ps1"
Write-Host "======================================================" -ForegroundColor Green
