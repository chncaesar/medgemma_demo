#!/usr/bin/env python3
"""
Simple test client for MedGemma on Vertex AI

Usage:
    python test_medgemma.py --endpoint-id YOUR_ENDPOINT_ID --prompt "What are symptoms of diabetes?"
"""

import argparse
import os

from google.cloud import aiplatform


def chat(endpoint_id: str, prompt: str, project_id: str, region: str) -> str:
    """Send a chat message to MedGemma."""
    aiplatform.init(project=project_id, location=region)

    endpoint = aiplatform.Endpoint(endpoint_id)

    response = endpoint.predict(
        instances=[{
            "@requestFormat": "chatCompletions",
            "messages": [
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0,
        }]
    )

    return response.predictions["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser(description="Test MedGemma endpoint")
    parser.add_argument("--endpoint-id", required=True, help="Vertex AI Endpoint ID")
    parser.add_argument("--prompt", required=True, help="Your question")
    parser.add_argument("--project-id", default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument("--region", default="us-central1")

    args = parser.parse_args()

    print(f"Question: {args.prompt}\n")
    print("Thinking...\n")

    response = chat(args.endpoint_id, args.prompt, args.project_id, args.region)

    print("=" * 60)
    print("MedGemma Response:")
    print("=" * 60)
    print(response)


if __name__ == "__main__":
    main()
