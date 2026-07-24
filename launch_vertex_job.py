import os
import sys
import subprocess
import time
import tempfile
from pathlib import Path

def prompt(label, default=""):
    d_str = f" [{default}]" if default else ""
    val = input(f"{label}{d_str}: ").strip()
    return val if val else default

def main():
    print("===========================================")
    print("  Launch GCP Self-Deleting Training VM")
    print("===========================================\n")
    
    csv_file = prompt("1. Local CSV file path")
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found.")
        sys.exit(1)
        
    bucket = prompt("2. GCS Bucket to store the CSV", default="gs://epochquant-training")
    if bucket.endswith("/"):
        bucket = bucket[:-1]
        
    symbol = prompt("3. Symbol (e.g., BNB_BTC)")
    if not symbol:
        print("Error: Symbol cannot be empty.")
        sys.exit(1)
        
    symbol_upper = symbol.upper()
    timestamp = int(time.time())
    vm_name = f"kronos-train-{symbol_upper.replace('_', '').lower()}-{timestamp}"
    
    # Priority list of zones to search for available L4 GPUs
    candidate_zones = ["us-central1-b", "us-central1-a", "us-central1-c", "us-east1-b", "us-east1-c"]
    
    csv_basename = os.path.basename(csv_file)
    gcs_uri = f"{bucket}/training-data/{csv_basename}"
    
    print(f"\n[1/3] Uploading {csv_file} to {gcs_uri}...")
    try:
        subprocess.run(f"gsutil cp \"{csv_file}\" \"{gcs_uri}\"", shell=True, check=True)
    except subprocess.CalledProcessError:
        print("Error: Failed to upload file via gsutil. Please ensure gsutil is installed and authenticated.")
        sys.exit(1)
        
    print(f"\n[2/3] Preparing Compute Engine VM configuration...")
    
    provisioned_zone = None
    
    for zone in candidate_zones:
        print(f"\n[3/3] Attempting to provision GCE VM '{vm_name}' (1x NVIDIA L4 GPU) in zone {zone}...")
        
        startup_script = f"""#!/bin/bash
export PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH

# Tee output to both log file and serial console (/dev/ttyS0)
exec > >(tee -a /var/log/kronos_training.log /dev/ttyS0) 2>&1

cleanup() {{
    echo "Training finished or terminated. Shutting down VM..."
    gcloud compute instances delete {vm_name} --zone={zone} --quiet || sudo shutdown -h now
}}
trap cleanup EXIT

echo "=== Kronos Training Script Started ==="
echo "Cloning repository..."
git clone https://github.com/epochquant/ml-training-ohlcv-model.git /opt/kronos
cd /opt/kronos

echo "Installing OS dependencies..."
apt-get update && apt-get install -y zip

if [ -f /opt/conda/bin/python ]; then
    PYTHON_BIN="/opt/conda/bin/python"
else
    PYTHON_BIN=$(which python3 || which python)
fi

echo "Using Python executable: $PYTHON_BIN"

echo "Installing Python dependencies from requirements.txt..."
$PYTHON_BIN -m pip install -r requirements.txt

echo "Executing training pipeline for {symbol_upper}..."
$PYTHON_BIN run_training_pipeline.py --symbol {symbol_upper} --dataset-gs-uri {gcs_uri} --non-interactive

echo "Backing up execution log to GCS..."
gsutil cp /var/log/kronos_training.log {bucket}/logs/{vm_name}.log
echo "=== Kronos Training Script Completed ==="
"""

        # Write startup script with strict Unix line endings (\n) to a temporary file
        script_file_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', newline='\n', delete=False, suffix='.sh') as f:
                f.write(startup_script)
                script_file_path = f.name

            gcloud_cmd = (
                f"gcloud compute instances create {vm_name} "
                f"--project=dev-gemini-ai "
                f"--zone={zone} "
                f"--machine-type=g2-standard-4 "
                f"--accelerator=type=nvidia-l4,count=1 "
                f"--image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 "
                f"--image-project=deeplearning-platform-release "
                f"--maintenance-policy=TERMINATE "
                f"--service-account=kronos-notebook-sa@dev-gemini-ai.iam.gserviceaccount.com "
                f"--scopes=https://www.googleapis.com/auth/cloud-platform "
                f"--metadata-from-file=startup-script=\"{script_file_path}\""
            )
            
            result = subprocess.run(gcloud_cmd, shell=True)
            if result.returncode == 0:
                provisioned_zone = zone
                break
            else:
                print(f"--> Zone {zone} currently unavailable or out of GPU capacity. Trying next candidate zone...")
                time.sleep(2)
        finally:
            if script_file_path and os.path.exists(script_file_path):
                os.remove(script_file_path)
            
    if provisioned_zone:
        print(f"\n=======================================================")
        print(f" Success! VM '{vm_name}' provisioned in {provisioned_zone}.")
        print(f" Training is executing in the background on GCP.")
        print(f" Output models will be uploaded to: {bucket}/models/")
        print(f" Execution log will be saved to: {bucket}/logs/{vm_name}.log")
        print(f"=======================================================\n")
        
        print("Monitoring VM execution for automated deletion upon completion...")
        while True:
            try:
                res = subprocess.check_output(
                    f"gcloud compute instances describe {vm_name} --zone={provisioned_zone} --format=\"value(status)\"",
                    shell=True
                ).decode().strip()
                
                if res in ["TERMINATED", "STOPPED"]:
                    print(f"\n[Auto-Cleanup] Training finished! Deleting VM instance '{vm_name}'...")
                    subprocess.run(f"gcloud compute instances delete {vm_name} --zone={provisioned_zone} --quiet", shell=True)
                    print("[Auto-Cleanup] VM instance successfully deleted.")
                    break
                else:
                    print(".", end="", flush=True)
            except Exception:
                print("\n[Auto-Cleanup] VM instance is no longer active.")
                break
            time.sleep(20)
    else:
        print("\nError: All candidate zones currently lack L4 GPU capacity. Please try again later.")
        sys.exit(1)

if __name__ == "__main__":
    main()
