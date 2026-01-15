#!/bin/bash

# MedGemma API Server Startup Script

# GCP Configuration
export GOOGLE_CLOUD_PROJECT="chncaesar1"
export GOOGLE_CLOUD_REGION="us-central1"
export MEDGEMMA_ENDPOINT_ID="your-endpoint-id"

# Server Configuration (optional)
export HOST="0.0.0.0"
export PORT="8000"

# Activate conda environment and run
source ~/.zshrc 2>/dev/null || source ~/.bashrc 2>/dev/null
conda activate medgemma

echo "Starting MedGemma API Server..."
echo "Project: $GOOGLE_CLOUD_PROJECT"
echo "Region: $GOOGLE_CLOUD_REGION"
echo "Endpoint: $MEDGEMMA_ENDPOINT_ID"
echo "Server: http://$HOST:$PORT"
echo ""

python run.py
