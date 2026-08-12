#!/usr/bin/env python3
"""
Launch GCP Serverless Containerized Training Job for the high-volatility-model
variant on Vertex AI. Mirrors launch_container_job.py / launch_short_classifier_job.py,
but targets the dedicated `ohlcv-model-training-hv` image and the
`gs://<bucket>/high-volatility/...` GCS namespace, so it never collides with the
base model's jobs, image, or storage prefix.
"""

import os
import sys
import argparse
import subprocess
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def get_env_var(key, default=""):
    return os.getenv(key, default)

def prompt(label, default=""):
    d_str = f" [{default}]" if default else ""
    val = input(f"{label}{d_str}: ").strip()
    return val if val else default

def parse_args():
    parser = argparse.ArgumentParser(description="Launch High-Volatility Container Job on GCP Vertex AI")
    parser.add_argument("--csv", type=str, help="Local CSV file path")
    parser.add_argument("--symbol", type=str, help="Trading Symbol (e.g., DOGEUSDT, PEPEUSDT)")
    parser.add_argument("--bucket", type=str, help="GCS Bucket URI (e.g., gs://epochquant-training)")
    parser.add_argument("--region", type=str, help="GCP Region (e.g., us-central1, us-east1)")
    parser.add_argument("--non-interactive", action="store_true", help="Run without interactive prompts")
    return parser.parse_args()

def main():
    args = parse_args()

    print("=======================================================")
    print("  Launch High-Volatility Container Training Job (L4 GPU)")
    print("=======================================================\n")

    # Load GCP defaults from environment (.env)
    project_id = get_env_var("GCP_PROJECT_ID", "dev-gemini-ai")
    repository_name = get_env_var("ARTIFACT_REGISTRY_IMAGE_PROJECT", "kronos-ml")
    region = args.region or get_env_var("GCP_REGION", "us-central1")
    default_bucket = get_env_var("GCS_BUCKET_NAME", "gs://epochquant-training")
    container_image = get_env_var(
        "ARTIFACT_REGISTRY_IMAGE_HV",
        f"{region}-docker.pkg.dev/{project_id}/{repository_name}/ohlcv-model-training-hv:latest",
    )

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
        symbol = args.symbol or prompt("3. Symbol (e.g., DOGEUSDT)")

    bucket = bucket.rstrip("/")
    if not bucket.startswith("gs://"):
        bucket = f"gs://{bucket}"

    if not symbol:
        print("Error: Symbol cannot be empty.")
        sys.exit(1)

    symbol_upper = symbol.upper()
    timestamp = int(time.time())
    job_name = f"kronos-hv-container-train-{symbol_upper.replace('_', '').lower()}-{timestamp}"
    csv_basename = os.path.basename(csv_file) if csv_file else f"{symbol_upper}.csv"
    gcs_dataset_uri = f"{bucket}/high-volatility/training-data/{csv_basename}"

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
    print(f" -> Region: {region}")
    print(f" -> Container Image: {container_image}")
    print(f" -> GPU: 1x NVIDIA L4 (g2-standard-4 Spot Instance)")

    gcloud_cmd = (
        f"gcloud ai custom-jobs create "
        f"--project={project_id} "
        f"--region={region} "
        f"--display-name={job_name} "
        f"--worker-pool-spec=machine-type=g2-standard-4,replica-count=1,"
        f"accelerator-type=NVIDIA_L4,accelerator-count=1,"
        f"container-image-uri={container_image} "
        f"--args=\"--symbol,{symbol_upper},--dataset-gs-uri,{gcs_dataset_uri},--non-interactive\""
    )

    try:
        subprocess.run(gcloud_cmd, shell=True, check=True)
        print(f"\n=======================================================")
        print(f" Success! Container Job '{job_name}' submitted.")
        print(f" Vertex AI is running container on Spot NVIDIA L4 GPU.")
        print(f" Outputs saved to: {bucket}/high-volatility/models/")
        print(f"=======================================================\n")
    except subprocess.CalledProcessError as e:
        print(f"\nError launching Vertex AI Custom Job: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
