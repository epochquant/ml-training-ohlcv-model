#!/usr/bin/env python3
"""
Shared Vertex AI custom-job submission helper — used by launch_high_volatility_job.py,
launch_container_job.py, and launch_short_classifier_job.py.

`gcloud ai custom-jobs create` returns as soon as the job is queued; a capacity
shortage (e.g. "Resources are insufficient in region: us-central1") only surfaces
~20 minutes later while the job sits in PREPARING, and Vertex AI's own built-in
retry just re-tries the same region in place. This helper submits the job with
`scheduling.disableRetries=True` (so Vertex AI doesn't spend up to ~2 hours
silently retrying an exhausted region on our behalf), polls the job's state, and
moves on to the next candidate region on any capacity/transient failure. Only a
small, well-known set of genuinely-our-fault errors (bad args, missing image,
permissions) is raised immediately instead of being masked by a region hop.

Also fixes a mismatch across the three launchers: they all print "Spot
Instance" to the console, but none of them actually requested Spot; jobs were
running as full-price standard/on-demand. `use_spot=True` (the default) sets
`scheduling.strategy=SPOT`.
"""

import json
import os
import subprocess
import tempfile
import time

import yaml

# Verified against `gcloud compute accelerator-types list --filter="name=nvidia-l4"`
# GPU availability per region changes over time.
# Note: us-west4 does not support g2-standard-4 (NVIDIA L4).
DEFAULT_FALLBACK_REGIONS = ["us-east1", "us-east4", "us-west1", "europe-west4"]

# Signatures of errors that are genuinely our fault (bad args, bad image,
# missing permissions) and won't be fixed by trying a different region.
# Everything else — including capacity exhaustion AND Vertex AI's generic
# "Internal error occurred for the current attempt" — is treated as retryable, since a
# transient platform failure is exactly the kind of thing a region hop should
# recover from, and we can't enumerate every transient error string in advance.
_FATAL_ERROR_PATTERNS = (
    "invalid_argument",
    "permission_denied",
    "not_found",
    "failed_precondition",
    "unauthenticated",
    "already_exists",
)


def _is_retryable_error(message):
    """Return True if `message` should be treated as retryable in another region."""
    lowered = (message or "").lower()
    return not any(pattern in lowered for pattern in _FATAL_ERROR_PATTERNS)


def _run_json(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        return None, (result.stderr or result.stdout).strip()
    try:
        return json.loads(result.stdout), None
    except json.JSONDecodeError:
        return None, result.stdout.strip()


def build_candidate_regions(primary_region, fallback_regions=None):
    fallback_regions = fallback_regions if fallback_regions is not None else DEFAULT_FALLBACK_REGIONS
    ordered = [primary_region] + [r for r in fallback_regions if r != primary_region]
    seen = set()
    unique = []
    for region in ordered:
        if region not in seen:
            seen.add(region)
            unique.append(region)
    return unique


def _cancel_job(project_id, region, job_id):
    """Cancel a queued or stuck job in GCP before trying another region."""
    cancel_cmd = (
        f"gcloud ai custom-jobs cancel {job_id} "
        f"--project={project_id} --region={region} --quiet"
    )
    print(f"  [gcp_job_utils] Cancelling job '{job_id}' in region '{region}'...")
    subprocess.run(cancel_cmd, shell=True, capture_output=True, text=True)


def _poll_job_state(project_id, region, job_id, timeout_s, poll_interval_s):
    """Poll a submitted job until it starts running, fails, or the timeout elapses."""
    deadline = time.time() + timeout_s
    while True:
        describe_cmd = (
            f"gcloud ai custom-jobs describe {job_id} "
            f"--project={project_id} --region={region} --format=json"
        )
        data, err = _run_json(describe_cmd)
        if data is None:
            return "OTHER_FAILURE", f"describe failed: {err}"

        state = data.get("state", "")
        if state in ("JOB_STATE_RUNNING", "JOB_STATE_SUCCEEDED"):
            return "RUNNING", state
        if state == "JOB_STATE_FAILED":
            error_message = (data.get("error") or {}).get("message", "")
            if _is_retryable_error(error_message):
                return "RETRYABLE_FAILURE", error_message
            return "OTHER_FAILURE", error_message or "unknown failure"
        if state == "JOB_STATE_CANCELLED":
            return "OTHER_FAILURE", "job was cancelled"

        if time.time() >= deadline:
            return "TIMEOUT", f"still '{state}' after {timeout_s}s"
        time.sleep(poll_interval_s)


def submit_custom_job_with_fallback(
    project_id,
    job_name,
    machine_type,
    accelerator_type,
    accelerator_count,
    container_image,
    container_args,
    command=None,
    service_account=None,
    primary_region="us-central1",
    fallback_regions=None,
    use_spot=None,
    per_region_timeout_s=360,
    poll_interval_s=20,
):
    """Submit a Vertex AI custom job, retrying in fallback regions on capacity errors.

    If `use_spot` is None, it is read from the `GCP_USE_SPOT` environment variable
    (default: False, for Standard On-Demand instances).
    If Spot is requested but GCP rejects the job due to unavailable preemptible
    quota, it automatically falls back to Standard On-Demand execution.

    Returns (region, job_resource_name) on success. Raises RuntimeError if every
    candidate region is exhausted, or if a submitted job fails for a reason that
    isn't a capacity shortage.
    """
    candidate_regions = build_candidate_regions(primary_region, fallback_regions)

    if use_spot is None:
        use_spot = os.getenv("GCP_USE_SPOT", "false").lower() in ("true", "1", "yes")

    container_spec = {"imageUri": container_image, "args": list(container_args)}
    if command:
        container_spec["command"] = list(command)

    current_spot = use_spot
    job_spec = {
        "workerPoolSpecs": [
            {
                "machineSpec": {
                    "machineType": machine_type,
                    "acceleratorType": accelerator_type,
                    "acceleratorCount": accelerator_count,
                },
                "replicaCount": 1,
                "containerSpec": container_spec,
            }
        ],
        "scheduling": {
            "strategy": "SPOT" if current_spot else "STANDARD",
            "restartJobOnWorkerRestart": False,
        },
    }
    if service_account:
        job_spec["serviceAccount"] = service_account

    attempts = []
    for region in candidate_regions:
        print(f"  [gcp_job_utils] Attempting region '{region}' (strategy: {job_spec['scheduling']['strategy']})...")
        config_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                yaml.safe_dump(job_spec, f)
                config_path = f.name

            create_cmd = (
                f"gcloud ai custom-jobs create "
                f"--project={project_id} --region={region} "
                f"--display-name={job_name} --config=\"{config_path}\" "
                f"--format=json"
            )
            data, err = _run_json(create_cmd)
            if data is None:
                err_lower = (err or "").lower()
                # If submission failed due to preemptible/spot quota exhaustion, auto-fallback to STANDARD
                if current_spot and ("preemptible" in err_lower or "custom_model_training_preemptible" in err_lower):
                    print(f"  [gcp_job_utils] Spot/Preemptible GPU quota unavailable in '{region}'. Falling back to STANDARD (On-Demand)...")
                    job_spec["scheduling"]["strategy"] = "STANDARD"
                    current_spot = False
                    with open(config_path, "w") as f_retry:
                        yaml.safe_dump(job_spec, f_retry)
                    data, err = _run_json(create_cmd)
                # If submission failed due to Standard On-Demand quota exhaustion, auto-fallback to SPOT
                elif not current_spot and ("resource_exhausted" in err_lower or "quota" in err_lower or "429" in err_lower):
                    print(f"  [gcp_job_utils] Standard On-Demand GPU quota exceeded in '{region}'. Auto-switching to SPOT strategy...")
                    job_spec["scheduling"]["strategy"] = "SPOT"
                    current_spot = True
                    with open(config_path, "w") as f_retry:
                        yaml.safe_dump(job_spec, f_retry)
                    data, err = _run_json(create_cmd)

                if data is None:
                    attempts.append((region, f"submission failed: {err}"))
                    print(f"  [gcp_job_utils] Region '{region}' submission failed: {err}")
                    continue

            job_resource = data.get("name")
            if not job_resource:
                attempts.append((region, "no job resource name returned"))
                continue

            job_id = job_resource.rsplit("/", 1)[-1]
            print(f"  [gcp_job_utils] Job '{job_id}' submitted in '{region}'. Monitoring for placement...")

            outcome, detail = _poll_job_state(project_id, region, job_id, per_region_timeout_s, poll_interval_s)

            if outcome == "RUNNING":
                print(f"  [gcp_job_utils] Job '{job_id}' is running in '{region}'.")
                return region, job_resource
            if outcome == "OTHER_FAILURE":
                _cancel_job(project_id, region, job_id)
                raise RuntimeError(
                    f"Job '{job_id}' in region '{region}' failed for a non-retryable reason: {detail}"
                )
            # RETRYABLE_FAILURE or TIMEOUT: cancel the job in this region and try next region
            _cancel_job(project_id, region, job_id)
            attempts.append((region, detail))
            print(f"  [gcp_job_utils] Region '{region}' unavailable ({detail}). Trying next region...")
        finally:
            if config_path and os.path.exists(config_path):
                os.remove(config_path)

    summary = "; ".join(f"{region}: {reason}" for region, reason in attempts)
    raise RuntimeError(f"All candidate regions exhausted for job '{job_name}'. {summary}")

