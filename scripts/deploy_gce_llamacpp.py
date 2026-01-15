#!/usr/bin/env python3
"""
Deploy MedGemma 27B GGUF on GCP Compute Engine with Ollama

Uses g2-standard-12 machine with 1x L4 GPU (24GB VRAM) and Ollama server.
Model: unsloth/medgemma-27b-it-GGUF (Q4_K_M quantization, ~16GB)

Usage:
    # Set environment variables
    export GOOGLE_CLOUD_PROJECT="your-project-id"

    # Deploy VM
    python deploy_gce_llamacpp.py deploy

    # Check status
    python deploy_gce_llamacpp.py status

    # Test endpoint
    python deploy_gce_llamacpp.py test

    # SSH into VM
    python deploy_gce_llamacpp.py ssh

    # Cleanup
    python deploy_gce_llamacpp.py cleanup
"""

import argparse
import os
import signal
import subprocess
import sys
import time

import requests

# Local port for IAP tunnel
LOCAL_TUNNEL_PORT = 18080


# Configuration
CONFIG = {
    "instance_name": "medgemma-llamacpp",
    "zone": "us-west1-a",
    "machine_type": "g2-standard-12",  # L4 GPU requires g2 machine series
    "accelerator_type": "nvidia-l4",
    "accelerator_count": 1,
    "boot_disk_size": "200GB",
    "model_repo": "unsloth/medgemma-27b-it-GGUF",
    "model_file": "medgemma-27b-it-Q4_K_M.gguf",  # Q4 fits in 24GB L4
    "model_size_gb": 16.5,
    "port": 8080,
}

# Startup script for the VM
STARTUP_SCRIPT = '''#!/bin/bash

LOG_FILE="/var/log/medgemma-setup.log"
exec > >(tee -a $LOG_FILE) 2>&1

echo "=== Starting MedGemma Setup with Ollama ==="
date

# Install Ollama
echo "Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# Wait for Ollama to be ready
sleep 5

# Install huggingface_hub (retry apt-get if it fails due to mirror sync)
echo "Installing huggingface_hub..."
for i in 1 2 3; do
    apt-get update && break
    echo "apt-get update failed, retrying in 30s..."
    sleep 30
done
apt-get install -y python3-pip || true
pip3 install huggingface_hub

# Download GGUF model using Python API (more reliable than CLI)
echo "Downloading model (~16GB, this may take a few minutes)..."
mkdir -p /opt/models
cd /opt/models

if [ ! -f "medgemma-27b-it-Q4_K_M.gguf" ]; then
    python3 << 'PYEOF'
from huggingface_hub import hf_hub_download
import os
os.chdir('/opt/models')
print('Starting download...')
path = hf_hub_download(
    repo_id='unsloth/medgemma-27b-it-GGUF',
    filename='medgemma-27b-it-Q4_K_M.gguf',
    local_dir='.',
    local_dir_use_symlinks=False
)
print(f'Downloaded to: {path}')
PYEOF
fi

# Verify model was downloaded
if [ ! -f "/opt/models/medgemma-27b-it-Q4_K_M.gguf" ]; then
    echo "ERROR: Model download failed!"
    exit 1
fi
echo "Model downloaded successfully"

# Create Modelfile for Ollama
echo "Creating Ollama Modelfile..."
cat > /opt/models/Modelfile << 'MODELEOF'
FROM /opt/models/medgemma-27b-it-Q4_K_M.gguf

TEMPLATE """{{ if .System }}<start_of_turn>system
{{ .System }}<end_of_turn>
{{ end }}<start_of_turn>user
{{ .Prompt }}<end_of_turn>
<start_of_turn>model
{{ .Response }}"""

PARAMETER temperature 0.7
PARAMETER num_ctx 4096
PARAMETER num_gpu 99
SYSTEM "You are a helpful medical assistant."
MODELEOF

# Start Ollama service if not running
echo "Starting Ollama service..."
systemctl start ollama || ollama serve &
sleep 5

# Create model in Ollama
echo "Creating Ollama model..."
ollama create medgemma -f /opt/models/Modelfile

# Configure Ollama to listen on all interfaces
echo "Configuring Ollama service for external access..."
mkdir -p /etc/systemd/system/ollama.service.d
cat > /etc/systemd/system/ollama.service.d/override.conf << 'OVERRIDEEOF'
[Service]
Environment="OLLAMA_HOST=0.0.0.0:8080"
OVERRIDEEOF

# Restart Ollama with new config
systemctl daemon-reload
systemctl restart ollama

echo "=== Setup Complete ==="
echo "Ollama server running on port 8080"
echo "Model: medgemma"
date
'''


def run_gcloud(args: list, capture_output: bool = False) -> subprocess.CompletedProcess:
    """Run gcloud command."""
    cmd = ["gcloud"] + args
    if capture_output:
        return subprocess.run(cmd, capture_output=True, text=True)
    return subprocess.run(cmd)


def deploy(project_id: str) -> str:
    """Deploy the VM with MedGemma."""
    print("=" * 60)
    print("Deploying MedGemma GGUF on GCE")
    print("=" * 60)
    print(f"Project: {project_id}")
    print(f"Instance: {CONFIG['instance_name']}")
    print(f"Zone: {CONFIG['zone']}")
    print(f"Machine: {CONFIG['machine_type']} + {CONFIG['accelerator_count']}x {CONFIG['accelerator_type']}")
    print(f"Model: {CONFIG['model_repo']}/{CONFIG['model_file']}")
    print(f"Cost: ~$1.08/hr")
    print("=" * 60)

    # Write startup script to temp file
    script_path = "/tmp/medgemma-startup.sh"
    with open(script_path, "w") as f:
        f.write(STARTUP_SCRIPT)

    # Create VM
    print("\n[Step 1/2] Creating VM...")
    # For g2 machines, GPU is pre-attached (no --accelerator needed)
    # For n1 machines, we need to add --accelerator
    gcloud_args = [
        "compute", "instances", "create", CONFIG["instance_name"],
        "--project", project_id,
        "--zone", CONFIG["zone"],
        "--machine-type", CONFIG["machine_type"],
        "--image-family", "common-cu128-ubuntu-2204-nvidia-570",
        "--image-project", "deeplearning-platform-release",
        "--boot-disk-size", CONFIG["boot_disk_size"],
        "--maintenance-policy", "TERMINATE",
        "--metadata-from-file", f"startup-script={script_path}",
        "--tags", "medgemma-server",
    ]

    # Add accelerator flag only for non-g2 machines
    if not CONFIG["machine_type"].startswith("g2-"):
        gcloud_args.insert(6, f"--accelerator=type={CONFIG['accelerator_type']},count={CONFIG['accelerator_count']}")

    result = run_gcloud(gcloud_args)

    if result.returncode != 0:
        print("Error creating VM. Check if GPU quota is available.")
        sys.exit(1)

    # Create IAP firewall rule (only allows access via IAP tunnel)
    print("\n[Step 2/2] Creating IAP firewall rule...")
    run_gcloud([
        "compute", "firewall-rules", "create", "allow-iap-tunnel",
        "--project", project_id,
        "--direction", "INGRESS",
        "--action", "ALLOW",
        "--rules", "tcp:8080,tcp:22",
        "--source-ranges", "35.235.240.0/20",
        "--target-tags", "medgemma-server",
        "--description", "Allow IAP tunnel access to MedGemma",
    ])

    # Get external IP
    result = run_gcloud([
        "compute", "instances", "describe", CONFIG["instance_name"],
        "--project", project_id,
        "--zone", CONFIG["zone"],
        "--format", "get(networkInterfaces[0].accessConfigs[0].natIP)",
    ], capture_output=True)

    external_ip = result.stdout.strip()

    print("\n" + "=" * 60)
    print("DEPLOYMENT INITIATED (IAP-secured)")
    print("=" * 60)
    print(f"Instance: {CONFIG['instance_name']}")
    print(f"Zone: {CONFIG['zone']}")
    print()
    print("The VM is now setting up. This takes 10-15 minutes:")
    print("  - Installing Ollama")
    print("  - Downloading model (~16 GB)")
    print("  - Creating model and starting server")
    print()
    print("Check progress with:")
    print(f"  python deploy_gce_llamacpp.py status")
    print()
    print("Start API tunnel with:")
    print(f"  python deploy_gce_llamacpp.py tunnel")
    print(f"  (Then access at http://localhost:{LOCAL_TUNNEL_PORT})")
    print()
    print("SSH into VM:")
    print(f"  python deploy_gce_llamacpp.py ssh")
    print("=" * 60)

    # Save IP to file
    with open(".medgemma_ip", "w") as f:
        f.write(external_ip)

    return external_ip


def get_external_ip(project_id: str) -> str:
    """Get external IP of the VM."""
    # Try to read from cache
    if os.path.exists(".medgemma_ip"):
        with open(".medgemma_ip") as f:
            return f.read().strip()

    result = run_gcloud([
        "compute", "instances", "describe", CONFIG["instance_name"],
        "--project", project_id,
        "--zone", CONFIG["zone"],
        "--format", "get(networkInterfaces[0].accessConfigs[0].natIP)",
    ], capture_output=True)

    return result.stdout.strip()


def start_iap_tunnel(project_id: str) -> subprocess.Popen:
    """Start IAP tunnel and return the process."""
    cmd = [
        "gcloud", "compute", "start-iap-tunnel",
        CONFIG["instance_name"], str(CONFIG["port"]),
        "--project", project_id,
        "--zone", CONFIG["zone"],
        "--local-host-port", f"localhost:{LOCAL_TUNNEL_PORT}",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Wait for tunnel to establish
    time.sleep(5)
    return proc


def check_status(project_id: str) -> None:
    """Check if the Ollama server is ready (via IAP tunnel)."""
    print("Starting IAP tunnel...")
    tunnel_proc = start_iap_tunnel(project_id)

    try:
        print(f"Checking Ollama server via IAP tunnel (localhost:{LOCAL_TUNNEL_PORT})...")

        # Check Ollama API
        response = requests.get(f"http://localhost:{LOCAL_TUNNEL_PORT}/api/tags", timeout=15)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print("✓ Ollama server is READY!")
            print(f"  Access via: python deploy_gce_llamacpp.py tunnel")
            if models:
                print(f"  Models: {[m['name'] for m in models]}")
            else:
                print("  Models: Still loading... (run 'status' again in a few minutes)")
        else:
            print(f"✗ Server returned status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("✗ Server not ready yet. Still setting up...")
        print("  Check setup progress:")
        print(f"    python deploy_gce_llamacpp.py --project-id {project_id} ssh")
        print("    sudo tail -f /var/log/medgemma-setup.log")
    except requests.exceptions.Timeout:
        print("✗ Connection timed out. Server may still be starting.")
    finally:
        tunnel_proc.terminate()
        tunnel_proc.wait()


def test_server(project_id: str, prompt: str) -> None:
    """Test the MedGemma server via IAP tunnel."""
    print("Starting IAP tunnel...")
    tunnel_proc = start_iap_tunnel(project_id)

    try:
        print(f"Testing Ollama server via IAP tunnel...")
        print(f"Prompt: {prompt}\n")

        # Ollama API endpoint
        response = requests.post(
            f"http://localhost:{LOCAL_TUNNEL_PORT}/api/chat",
            json={
                "model": "medgemma",
                "messages": [
                    {"role": "system", "content": "You are a helpful medical assistant."},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
            },
            timeout=120,
        )

        if response.status_code == 200:
            content = response.json()["message"]["content"]
            print("=" * 60)
            print("MedGemma Response:")
            print("=" * 60)
            print(content)
            print("=" * 60)
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect. Server may not be ready.")
        print("Check status: python deploy_gce_llamacpp.py status")
    except requests.exceptions.Timeout:
        print("Error: Request timed out. Model may still be loading.")
    finally:
        tunnel_proc.terminate()
        tunnel_proc.wait()


def ssh_to_vm(project_id: str) -> None:
    """SSH into the VM via IAP tunnel."""
    run_gcloud([
        "compute", "ssh", CONFIG["instance_name"],
        "--project", project_id,
        "--zone", CONFIG["zone"],
        "--tunnel-through-iap",
    ])


def run_tunnel(project_id: str) -> None:
    """Start a persistent IAP tunnel for API access."""
    print(f"Starting IAP tunnel to MedGemma server...")
    print(f"Local endpoint: http://localhost:{LOCAL_TUNNEL_PORT}")
    print(f"Press Ctrl+C to stop the tunnel.\n")

    cmd = [
        "gcloud", "compute", "start-iap-tunnel",
        CONFIG["instance_name"], str(CONFIG["port"]),
        "--project", project_id,
        "--zone", CONFIG["zone"],
        "--local-host-port", f"localhost:{LOCAL_TUNNEL_PORT}",
    ]

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nTunnel stopped.")


def cleanup(project_id: str) -> None:
    """Delete the VM and firewall rule."""
    print("Cleaning up resources...")

    # Delete VM
    print("Deleting VM...")
    run_gcloud([
        "compute", "instances", "delete", CONFIG["instance_name"],
        "--project", project_id,
        "--zone", CONFIG["zone"],
        "--quiet",
    ])

    # Delete firewall rule
    print("Deleting firewall rule...")
    run_gcloud([
        "compute", "firewall-rules", "delete", "allow-iap-tunnel",
        "--project", project_id,
        "--quiet",
    ])

    # Remove IP cache
    if os.path.exists(".medgemma_ip"):
        os.remove(".medgemma_ip")

    print("✓ Cleanup complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Deploy MedGemma GGUF on GCP with Ollama"
    )

    parser.add_argument(
        "--project-id",
        default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        help="GCP Project ID",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Deploy
    subparsers.add_parser("deploy", help="Deploy VM with MedGemma")

    # Status
    subparsers.add_parser("status", help="Check if server is ready")

    # Test
    test_parser = subparsers.add_parser("test", help="Test the server")
    test_parser.add_argument(
        "--prompt",
        default="What are the symptoms of pneumonia?",
        help="Test prompt",
    )

    # SSH
    subparsers.add_parser("ssh", help="SSH into the VM")

    # Tunnel
    subparsers.add_parser("tunnel", help="Start IAP tunnel for API access")

    # Cleanup
    subparsers.add_parser("cleanup", help="Delete VM and resources")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if not args.project_id:
        print("Error: --project-id or GOOGLE_CLOUD_PROJECT env var required")
        sys.exit(1)

    if args.command == "deploy":
        deploy(args.project_id)
    elif args.command == "status":
        check_status(args.project_id)
    elif args.command == "test":
        test_server(args.project_id, args.prompt)
    elif args.command == "ssh":
        ssh_to_vm(args.project_id)
    elif args.command == "tunnel":
        run_tunnel(args.project_id)
    elif args.command == "cleanup":
        cleanup(args.project_id)


if __name__ == "__main__":
    main()
