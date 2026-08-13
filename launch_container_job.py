#!/usr/bin/env python3
"""
Launch GCP Serverless Containerized Training Job on Vertex AI
Uses pre-built Docker/Podman container images from Artifact Registry with Spot NVIDIA L4 GPUs.
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from gcp_job_utils import submit_custom_job_with_fallback

def get_env_var(key, default=""):
    return os.getenv(key, default)

def prompt(label, default=""):
    d_str = f" [{default}]" if default else ""
    val = input(f"{label}{d_str}: ").strip()
    return val if val else default

def parse_args():
    parser = argparse.ArgumentParser(description="Launch Serverless Container Job on GCP Vertex AI")
    parser.add_argument("--csv", type=str, help="Local CSV file path")
    parser.add_argument("--symbol", type=str, help="Trading Symbol (e.g., BNB_BTC)")
    parser.add_argument("--bucket", type=str, help="GCS Bucket URI (e.g., gs://epochquant-training)")
    parser.add_argument("--region", type=str, help="GCP Region (e.g., us-central1, us-east1)")
    parser.add_argument("--spot", action="store_true", default=False, help="Use Spot/Preemptible GPU instances (default: False, runs On-Demand)")
    parser.add_argument("--non-interactive", action="store_true", help="Run without interactive prompts")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load GCP defaults from environment (.env)
    project_id = get_env_var("GCP_PROJECT_ID", "dev-gemini-ai")
    repository_name = get_env_var("ARTIFACT_REGISTRY_IMAGE_PROJECT", "kronos-ml")
    region = args.region or get_env_var("GCP_REGION", "us-central1")
    service_account = get_env_var("GCP_SERVICE_ACCOUNT", "kronos-notebook-sa@dev-gemini-ai.iam.gserviceaccount.com")
    default_bucket = get_env_var("GCS_BUCKET_NAME", "gs://epochquant-training")
    container_image = get_env_var("ARTIFACT_REGISTRY_IMAGE", f"{region}-docker.pkg.dev/{project_id}/{repository_name}/ohlcv-model-training:latest")
    use_spot = args.spot or get_env_var("GCP_USE_SPOT", "false").lower() in ("true", "1", "yes")
    gpu_tier = "Spot" if use_spot else "On-Demand"

    print("=======================================================")
    print(f"  Launch GCP Serverless Container Training Job (L4 GPU - {gpu_tier})")
    print("=======================================================\n")

    if args.non_interactive:
        csv_file = args.csv
        symbol = args.symbol
        bucket = args.bucket or default_bucket
    else:
        csv_file = args.csv or prompt("1. Local CSV file path")
        if not csv_file or not os.path.exists(csv_file):
            print(f"Error: File '{csv_file}' not found.")
            sys.exit(1)

        bucket = args.bucket or prompt("2. GCS Bucket to store training CSV", default=default_bucket)
        symbol = args.symbol or prompt("3. Symbol (e.g., BNB_BTC)")

    bucket = bucket.rstrip("/")
    if not bucket.startswith("gs://"):
        bucket = f"gs://{bucket}"

    if not symbol:
        print("Error: Symbol cannot be empty.")
        sys.exit(1)

    symbol_upper = symbol.upper()
    timestamp = int(time.time())
    job_name = f"kronos-container-train-{symbol_upper.replace('_', '').lower()}-{timestamp}"
    csv_basename = os.path.basename(csv_file) if csv_file else f"{symbol_upper}.csv"
    gcs_dataset_uri = f"{bucket}/training-data/{csv_basename}"

    # Step 1: Upload dataset to GCS
    if csv_file and os.path.exists(csv_file):
        print(f"\n[1/3] Uploading dataset {csv_file} -> {gcs_dataset_uri}...")
        try:
            subprocess.run(f"gcloud storage cp \"{csv_file}\" \"{gcs_dataset_uri}\"", shell=True, check=True)
        except subprocess.CalledProcessError:
            print("Fallback: Uploading via gsutil...")
            subprocess.run(f"gsutil cp \"{csv_file}\" \"{gcs_dataset_uri}\"", shell=True, check=True)

    # Step 2: Submit Vertex AI Custom Training Job with Container Spec
    print(f"\n[2/3] Submitting Serverless Container Job '{job_name}' to Vertex AI...")
    print(f" -> Project: {project_id}")
    print(f" -> Primary Region: {region}")
    print(f" -> Container Image: {container_image}")
    print(f" -> GPU: 1x NVIDIA L4 (g2-standard-4 {gpu_tier} Instance)")

    try:
        placed_region, job_resource = submit_custom_job_with_fallback(
            project_id=project_id,
            job_name=job_name,
            machine_type="g2-standard-4",
            accelerator_type="NVIDIA_L4",
            accelerator_count=1,
            container_image=container_image,
            container_args=["--symbol", symbol_upper, "--dataset-gs-uri", gcs_dataset_uri, "--non-interactive"],
            primary_region=region,
            use_spot=use_spot,
        )
        print(f"\n=======================================================")
        print(f" Success! Container Job '{job_name}' submitted.")
        print(f" Placed in region: {placed_region}")
        print(f" Vertex AI is running container on {gpu_tier} NVIDIA L4 GPU.")
        print(f" Outputs saved to: {bucket}/models/")
        print(f"=======================================================\n")
    except RuntimeError as e:
        print(f"\nError launching Vertex AI Custom Job: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
