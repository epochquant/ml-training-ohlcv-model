#!/usr/bin/env python3
"""
Launch GCP Serverless Containerized Training Job for the Short Classifier on Vertex AI.
Uses pre-built Docker/Podman container images from Artifact Registry with Spot NVIDIA L4 GPUs.
Overrides the default container command to run `train_short_classifier.py`.
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
    parser = argparse.ArgumentParser(description="Launch Short Classifier Container Job on GCP Vertex AI")
    parser.add_argument("--csv", type=str, help="Local CSV file path (labeled dataset)")
    parser.add_argument("--bucket", type=str, help="GCS Bucket URI (e.g., gs://epochquant-training)")
    parser.add_argument("--region", type=str, help="GCP Region (e.g., us-central1, us-east1)")
    parser.add_argument("--non-interactive", action="store_true", help="Run without interactive prompts")
    return parser.parse_args()

def main():
    args = parse_args()

    print("=======================================================")
    print("  Launch Short Classifier Training Job (Vertex AI / L4)")
    print("=======================================================\n")

    # Load GCP defaults from environment (.env)
    project_id = get_env_var("GCP_PROJECT_ID", "dev-gemini-ai")
    repository_name = get_env_var("ARTIFACT_REGISTRY_IMAGE_PROJECT", "kronos-ml")
    region = args.region or get_env_var("GCP_REGION", "us-central1")
    default_bucket = get_env_var("GCS_BUCKET_NAME", "gs://epochquant-training")
    container_image = get_env_var("ARTIFACT_REGISTRY_IMAGE", f"{region}-docker.pkg.dev/{project_id}/{repository_name}/ohlcv-model-training:latest")

    if args.non_interactive:
        csv_file = args.csv
        bucket = args.bucket or default_bucket
    else:
        csv_file = args.csv or prompt("1. Local Labeled CSV file path (e.g. ./dataset/master_short_labeled.csv)")
        if not csv_file or not os.path.exists(csv_file):
            print(f"Error: File '{csv_file}' not found.")
            sys.exit(1)

        bucket = args.bucket or prompt("2. GCS Bucket to store training data and models", default=default_bucket)

    bucket = bucket.rstrip("/")
    if not bucket.startswith("gs://"):
        bucket = f"gs://{bucket}"

    timestamp = int(time.time())
    job_name = f"kronos-short-classifier-train-{timestamp}"
    csv_basename = os.path.basename(csv_file) if csv_file else f"master_short_labeled_{timestamp}.csv"
    
    gcs_dataset_uri = f"{bucket}/training-data-short/{csv_basename}"
    gcs_output_dir = f"{bucket}/short-models/kronos_short_usdt_head_{timestamp}"

    # Default paths where the Kronos backbone is assumed to be stored from previous jobs
    pretrained_kronos_uri = f"{bucket}/models/kronos_short_usdt/output_models/predictor_finetuned/checkpoints/best_model"
    pretrained_tokenizer_uri = f"{bucket}/models/kronos_short_usdt/output_models/tokenizer_finetuned/checkpoints/best_model"

    # Step 1: Upload dataset to GCS
    if csv_file and os.path.exists(csv_file):
        print(f"\n[1/3] Uploading labeled dataset {csv_file} -> {gcs_dataset_uri}...")
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
    
    # Vertex AI Custom Job creation command
    # Overriding the command to explicitly call python with our short classifier script
    gcloud_cmd = (
        f"gcloud ai custom-jobs create "
        f"--project={project_id} "
        f"--region={region} "
        f"--display-name={job_name} "
        f"--worker-pool-spec=machine-type=g2-standard-4,replica-count=1,"
        f"accelerator-type=NVIDIA_L4,accelerator-count=1,"
        f"container-image-uri={container_image} "
        f"--command=\"python\" "
        f"--args=\"training/train_short_classifier.py,"
        f"--dataset,{gcs_dataset_uri},"
        f"--pretrained_kronos,{pretrained_kronos_uri},"
        f"--pretrained_tokenizer,{pretrained_tokenizer_uri},"
        f"--gcs_output_dir,{gcs_output_dir}\""
    )

    try:
        result = subprocess.run(gcloud_cmd, shell=True, check=True)
        print(f"\n=======================================================")
        print(f" Success! Container Job '{job_name}' submitted.")
        print(f" Vertex AI is running container on Spot NVIDIA L4 GPU.")
        print(f" Output Model will be saved to: {gcs_output_dir}/")
        print(f"=======================================================\n")
    except subprocess.CalledProcessError as e:
        print(f"\nError launching Vertex AI Custom Job: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
