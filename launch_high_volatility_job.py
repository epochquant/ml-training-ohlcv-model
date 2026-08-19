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

from gcp_job_utils import submit_custom_job_with_fallback

def get_env_var(key, default=""):
    return os.getenv(key, default)

def prompt(label, default=""):
    d_str = f" [{default}]" if default else ""
    val = input(f"{label}{d_str}: ").strip()
    return val if val else default

DEFAULT_FALLBACK_STR = "europe-west4,us-west1,northamerica-northeast1,us-central1,asia-east1"

def parse_args():
    parser = argparse.ArgumentParser(description="Launch High-Volatility Container Job on GCP Vertex AI")
    parser.add_argument("--csv", type=str, help="Local CSV file path")
    parser.add_argument("--symbol", type=str, help="Trading Symbol (e.g., DOGEUSDT, PEPEUSDT)")
    parser.add_argument("--bucket", type=str, help="GCS Bucket URI (e.g., gs://epochquant-training)")
    parser.add_argument("--region", type=str, help="GCP Region (e.g., europe-west4, us-central1, us-west1)")
    parser.add_argument("--gpu", type=str, choices=["t4", "l4", "T4", "L4"], default=None, help="GPU accelerator (t4: NVIDIA T4 [cheapest & high availability], l4: NVIDIA L4)")
    parser.add_argument("--spot", action="store_true", default=False, help="Use Spot/Preemptible GPU instances (default: False, runs On-Demand)")
    parser.add_argument("--timeout-per-region", type=int, default=600, help="Max timeout (in seconds) to wait for job placement per region before trying fallback (default: 600s)")
    parser.add_argument("--fallback-regions", type=str, default=DEFAULT_FALLBACK_STR, help=f"Comma-separated list of fallback regions (default: {DEFAULT_FALLBACK_STR})")
    parser.add_argument("--non-interactive", action="store_true", help="Run without interactive prompts")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load GCP defaults from environment (.env)
    project_id = get_env_var("GCP_PROJECT_ID", "dev-gemini-ai")
    repository_name = get_env_var("ARTIFACT_REGISTRY_IMAGE_PROJECT", "ohlcv-model-training")
    service_account = get_env_var("GCP_SERVICE_ACCOUNT", "kronos-notebook-sa@dev-gemini-ai.iam.gserviceaccount.com")
    default_bucket = get_env_var("GCS_BUCKET_NAME", "gs://epochquant-training")
    registry_region = get_env_var("ARTIFACT_REGISTRY_REGION", "us-central1")
    container_image = get_env_var(
        "ARTIFACT_REGISTRY_IMAGE_HV",
        f"{registry_region}-docker.pkg.dev/{project_id}/{repository_name}/ohlcv-model-training-hv:latest",
    )

    if args.non_interactive:
        region = args.region or get_env_var("GCP_REGION", "europe-west4")
        csv_file = args.csv
        symbol = args.symbol
        bucket = args.bucket or default_bucket
        use_spot = args.spot or get_env_var("GCP_USE_SPOT", "false").lower() in ("true", "1", "yes")
        gpu_choice = (args.gpu or get_env_var("GCP_ACCELERATOR", "t4")).strip().upper()
        fallback_input = args.fallback_regions or DEFAULT_FALLBACK_STR
        fallback_regions = [r.strip() for r in fallback_input.split(",") if r.strip()]
    else:
        print("=======================================================")
        print("  Launch High-Volatility Container Training Job (Vertex AI)")
        print("=======================================================\n")

        csv_file = args.csv or prompt("1. Local CSV file path")
        if not csv_file or not os.path.exists(csv_file):
            print(f"Error: File '{csv_file}' not found.")
            sys.exit(1)

        bucket = args.bucket or prompt("2. GCS Bucket to store training CSV", default=default_bucket)
        symbol = args.symbol or prompt("3. Symbol (e.g., DOGEUSDT)")
        region = args.region or prompt("4. Primary GCP Region (e.g., europe-west4, us-central1)", default=get_env_var("GCP_REGION", "europe-west4"))

        if args.spot:
            use_spot = True
        else:
            default_strat = "2" if get_env_var("GCP_USE_SPOT", "false").lower() in ("true", "1", "yes") else "1"
            strat_choice = prompt("5. Provisioning Strategy (1: On-Demand, 2: Spot [Lowest Cost])", default=default_strat)
            use_spot = strat_choice.strip().lower() in ("2", "spot", "yes", "true")

        if args.gpu:
            gpu_choice = args.gpu.strip().upper()
        else:
            default_gpu_env = get_env_var("GCP_ACCELERATOR", "t4").lower()
            default_gpu_opt = "2" if default_gpu_env == "l4" else "1"
            gpu_prompt_choice = prompt("6. GPU Accelerator (1: NVIDIA T4 [~$0.15/hr Spot - High Availability], 2: NVIDIA L4 [~$0.21/hr Spot])", default=default_gpu_opt)
            gpu_choice = "L4" if gpu_prompt_choice.strip().lower() in ("2", "l4") else "T4"

        fallback_input = prompt("7. Fallback Regions (Global Multi-Region Diversity)", default=args.fallback_regions or DEFAULT_FALLBACK_STR)
        fallback_regions = [r.strip() for r in fallback_input.split(",") if r.strip()]

    gpu_tier = "Spot" if use_spot else "On-Demand"
    if gpu_choice == "L4":
        machine_type = "g2-standard-4"
        accelerator_type = "NVIDIA_L4"
        gpu_display = "1x NVIDIA L4 (g2-standard-4)"
    else:
        machine_type = "n1-standard-4"
        accelerator_type = "NVIDIA_TESLA_T4"
        gpu_display = "1x NVIDIA T4 (n1-standard-4)"

    if args.non_interactive:
        print("=======================================================")
        print(f"  Launch High-Volatility Container Training Job ({gpu_display} - {gpu_tier})")
        print("=======================================================\n")

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
    print(f" -> Primary Region: {region}")
    print(f" -> Container Image: {container_image}")
    print(f" -> GPU: {gpu_display} ({gpu_tier} Instance)")
    if service_account:
        print(f" -> Service Account: {service_account}")
    print(f" -> Timeout per region: {args.timeout_per_region}s")

    try:
        placed_region, job_resource = submit_custom_job_with_fallback(
            project_id=project_id,
            job_name=job_name,
            machine_type=machine_type,
            accelerator_type=accelerator_type,
            accelerator_count=1,
            container_image=container_image,
            container_args=["--symbol", symbol_upper, "--dataset-gs-uri", gcs_dataset_uri, "--non-interactive"],
            command=None,
            service_account=service_account,
            primary_region=region,
            fallback_regions=fallback_regions,
            use_spot=use_spot,
            per_region_timeout_s=args.timeout_per_region,
        )
        print(f"\n=======================================================")
        print(f" Success! Container Job '{job_name}' submitted.")
        print(f" Placed in region: {placed_region}")
        print(f" Vertex AI is running container on {gpu_tier} {gpu_display}.")
        print(f" Outputs saved to: {bucket}/high-volatility/models/")
        print(f"=======================================================\n")
    except RuntimeError as e:
        # If L4 failed in all regions, offer attempt with T4 if running interactively or if auto fallback triggered
        if gpu_choice == "L4":
            print(f"\n[Warning] All regions exhausted for NVIDIA L4 ({e}). Trying fallback to high-availability NVIDIA T4...")
            try:
                placed_region, job_resource = submit_custom_job_with_fallback(
                    project_id=project_id,
                    job_name=f"{job_name}-t4-fallback",
                    machine_type="n1-standard-4",
                    accelerator_type="NVIDIA_TESLA_T4",
                    accelerator_count=1,
                    container_image=container_image,
                    container_args=["--symbol", symbol_upper, "--dataset-gs-uri", gcs_dataset_uri, "--non-interactive"],
                    command=None,
                    service_account=service_account,
                    primary_region=region,
                    fallback_regions=fallback_regions,
                    use_spot=use_spot,
                    per_region_timeout_s=args.timeout_per_region,
                )
                print(f"\n=======================================================")
                print(f" Success! Container Job '{job_name}-t4-fallback' submitted.")
                print(f" Placed in region: {placed_region}")
                print(f" Vertex AI is running container on {gpu_tier} 1x NVIDIA T4 (n1-standard-4).")
                print(f" Outputs saved to: {bucket}/high-volatility/models/")
                print(f"=======================================================\n")
                return
            except RuntimeError as err_t4:
                print(f"\nError launching Vertex AI Custom Job (T4 Fallback also failed): {err_t4}")
                sys.exit(1)
        print(f"\nError launching Vertex AI Custom Job: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
