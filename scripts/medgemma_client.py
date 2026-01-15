#!/usr/bin/env python3
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MedGemma Client for Google Cloud Vertex AI

This script provides functionality to interact with MedGemma models deployed
on Google Cloud's Vertex AI platform. It supports:
- Online inference (synchronous) with text and images
- Streaming responses
- Batch inference (asynchronous)
- Model upload and management

Usage:
    # Online inference with image
    python medgemma_client.py online --endpoint-id YOUR_ENDPOINT_ID \
        --region us-central1 --prompt "Describe this X-ray" \
        --image-url "https://example.com/xray.png"

    # Online inference with text only
    python medgemma_client.py online --endpoint-id YOUR_ENDPOINT_ID \
        --region us-central1 --prompt "How do you differentiate bacterial from viral pneumonia?"

    # Streaming response
    python medgemma_client.py stream --endpoint-id YOUR_ENDPOINT_ID \
        --region us-central1 --prompt "Explain the symptoms of pneumonia"

    # Batch inference
    python medgemma_client.py batch --hf-token YOUR_HF_TOKEN \
        --input-file instances.jsonl --bucket gs://your-bucket
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import uuid
from typing import Any, Generator, Optional

import google.auth
import google.auth.transport.requests
import openai
import requests
from google.cloud import aiplatform, storage


class MedGemmaConfig:
    """Configuration for MedGemma client."""

    THINKING_VARIANTS = [
        "medgemma-1.5-4b-it",
        "medgemma-27b-it",
        "medgemma-27b-text-it",
    ]

    SERVE_DOCKER_URI = (
        "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/"
        "pytorch-vllm-serve:20250430_0916_RC00_maas"
    )

    # Hardware configurations for different model sizes
    HARDWARE_CONFIGS = {
        "4b": {
            "accelerator_type": "NVIDIA_L4",
            "machine_type": "g2-standard-24",
            "accelerator_count": 2,
        },
        "27b": {
            "accelerator_type": "NVIDIA_L4",
            "machine_type": "g2-standard-48",
            "accelerator_count": 4,
        },
    }


class MedGemmaClient:
    """Client for interacting with MedGemma models on Vertex AI."""

    def __init__(
        self,
        project_id: str,
        region: str,
        endpoint_id: Optional[str] = None,
        endpoint_region: Optional[str] = None,
        use_dedicated_endpoint: bool = True,
        is_thinking: bool = False,
    ):
        """
        Initialize the MedGemma client.

        Args:
            project_id: Google Cloud project ID
            region: Default region for Vertex AI operations
            endpoint_id: ID of deployed MedGemma endpoint (for online inference)
            endpoint_region: Region of the endpoint (defaults to region)
            use_dedicated_endpoint: Whether to use dedicated endpoint DNS
            is_thinking: Enable thinking mode for supported variants
        """
        self.project_id = project_id
        self.region = region
        self.endpoint_id = endpoint_id
        self.endpoint_region = endpoint_region or region
        self.use_dedicated_endpoint = use_dedicated_endpoint
        self.is_thinking = is_thinking

        self.endpoint = None
        self._credentials = None

        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=region)

        if endpoint_id:
            self._load_endpoint()

    def _load_endpoint(self) -> None:
        """Load the endpoint from Vertex AI."""
        self.endpoint = aiplatform.Endpoint(
            endpoint_name=self.endpoint_id,
            project=self.project_id,
            location=self.endpoint_region,
        )

        endpoint_name = self.endpoint.display_name

        # Validate endpoint is instruction-tuned
        if "pt" in endpoint_name:
            raise ValueError(
                "This client is intended for instruction-tuned variants. "
                "Please use an instruction-tuned model."
            )

        # Adjust thinking mode based on model variant
        if self.is_thinking and not any(
            variant in endpoint_name for variant in MedGemmaConfig.THINKING_VARIANTS
        ):
            self.is_thinking = False
            print(
                "Note: Thinking is not supported for this variant. "
                "Disabling thinking mode."
            )

    def _get_credentials(self) -> google.auth.credentials.Credentials:
        """Get and refresh Google Cloud credentials."""
        if self._credentials is None:
            self._credentials, _ = google.auth.default()

        auth_req = google.auth.transport.requests.Request()
        self._credentials.refresh(auth_req)
        return self._credentials

    def _get_base_url(self) -> str:
        """Get the base URL for API calls."""
        endpoint_resource_name = self.endpoint.resource_name

        if self.use_dedicated_endpoint:
            dedicated_dns = self.endpoint.gca_resource.dedicated_endpoint_dns
            return f"https://{dedicated_dns}/v1beta1/{endpoint_resource_name}"
        else:
            return (
                f"https://{self.endpoint_region}-aiplatform.googleapis.com/"
                f"v1beta1/{endpoint_resource_name}"
            )

    def _build_messages(
        self,
        prompt: str,
        system_instruction: str = "You are a helpful medical assistant.",
        image_url: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Build the messages payload for the API request.

        Args:
            prompt: User prompt text
            system_instruction: System instruction for the model
            image_url: Optional URL of an image to include

        Returns:
            List of message dictionaries
        """
        if self.is_thinking:
            system_instruction = (
                f"SYSTEM INSTRUCTION: think silently if needed. {system_instruction}"
            )

        if image_url:
            # Multimodal message with image
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_instruction}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ]
        else:
            # Text-only message
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ]

        return messages

    def _parse_thinking_response(self, response: str) -> tuple[str, str]:
        """
        Parse a thinking response to extract thought and final response.

        Args:
            response: Raw response from the model

        Returns:
            Tuple of (thought, response)
        """
        if "<unused95>" in response:
            thought, final_response = response.split("<unused95>")
            thought = thought.replace("<unused94>thought\n", "")
            return thought, final_response
        return "", response

    def predict(
        self,
        prompt: str,
        system_instruction: str = "You are a helpful medical assistant.",
        image_url: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0,
        use_openai_sdk: bool = False,
    ) -> dict[str, str]:
        """
        Generate a response from MedGemma (online inference).

        Args:
            prompt: User prompt text
            system_instruction: System instruction for the model
            image_url: Optional URL of an image to include
            max_tokens: Maximum tokens in response (auto-determined if None)
            temperature: Sampling temperature
            use_openai_sdk: Use OpenAI SDK instead of Vertex AI SDK

        Returns:
            Dictionary with 'response' and optionally 'thought' keys
        """
        if not self.endpoint:
            raise ValueError("No endpoint configured. Provide endpoint_id.")

        # Determine max tokens based on mode
        if max_tokens is None:
            if self.is_thinking:
                max_tokens = 2000 if not image_url else 1500
            else:
                max_tokens = 1000 if not image_url else 500

        messages = self._build_messages(prompt, system_instruction, image_url)

        if use_openai_sdk:
            response = self._predict_openai(messages, max_tokens, temperature)
        else:
            response = self._predict_vertex(messages, max_tokens, temperature)

        result = {"response": response}

        if self.is_thinking:
            thought, final_response = self._parse_thinking_response(response)
            result = {"thought": thought, "response": final_response}

        return result

    def _predict_vertex(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate response using Vertex AI SDK."""
        instances = [{
            "@requestFormat": "chatCompletions",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }]

        prediction = self.endpoint.predict(
            instances=instances,
            use_dedicated_endpoint=self.use_dedicated_endpoint,
        )

        return prediction.predictions["choices"][0]["message"]["content"]

    def _predict_openai(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate response using OpenAI SDK."""
        creds = self._get_credentials()
        base_url = self._get_base_url()

        client = openai.OpenAI(base_url=base_url, api_key=creds.token)

        response = client.chat.completions.create(
            model="",
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
        )

        return response.choices[0].message.content

    def stream(
        self,
        prompt: str,
        system_instruction: str = "You are a helpful medical assistant.",
        max_tokens: int = 1500,
        temperature: float = 0,
        use_openai_sdk: bool = False,
    ) -> Generator[str, None, None]:
        """
        Stream a response from MedGemma.

        Args:
            prompt: User prompt text
            system_instruction: System instruction for the model
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            use_openai_sdk: Use OpenAI SDK instead of raw HTTP

        Yields:
            Response chunks as they arrive
        """
        if not self.endpoint:
            raise ValueError("No endpoint configured. Provide endpoint_id.")

        messages = self._build_messages(prompt, system_instruction)
        creds = self._get_credentials()
        base_url = self._get_base_url()

        if use_openai_sdk:
            yield from self._stream_openai(
                messages, max_tokens, temperature, creds, base_url
            )
        else:
            yield from self._stream_vertex(
                messages, max_tokens, temperature, creds, base_url
            )

    def _stream_vertex(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        creds: google.auth.credentials.Credentials,
        base_url: str,
    ) -> Generator[str, None, None]:
        """Stream using raw HTTP requests."""
        endpoint_url = f"{base_url}/chat/completions"
        request_body = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        with requests.post(
            endpoint_url,
            headers={"Authorization": f"Bearer {creds.token}"},
            json=request_body,
            stream=True,
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_lines(chunk_size=8192):
                if chunk:
                    chunk_str = chunk.decode("utf-8").removeprefix("data:").strip()
                    if chunk_str == "[DONE]":
                        break
                    chunk_data = json.loads(chunk_str)
                    content = chunk_data["choices"][0]["delta"].get("content", "")
                    if content:
                        yield content

    def _stream_openai(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        creds: google.auth.credentials.Credentials,
        base_url: str,
    ) -> Generator[str, None, None]:
        """Stream using OpenAI SDK."""
        client = openai.OpenAI(base_url=base_url, api_key=creds.token)

        response = client.chat.completions.create(
            model="",
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content


class MedGemmaBatchClient:
    """Client for batch inference with MedGemma models."""

    def __init__(
        self,
        project_id: str,
        region: str,
        hf_token: str,
        bucket_uri: Optional[str] = None,
    ):
        """
        Initialize the batch inference client.

        Args:
            project_id: Google Cloud project ID
            region: Region for Vertex AI operations
            hf_token: Hugging Face token for model access
            bucket_uri: Cloud Storage bucket URI (created if not provided)
        """
        self.project_id = project_id
        self.region = region
        self.hf_token = hf_token
        self.bucket_uri = bucket_uri
        self.bucket_name = None
        self.service_account = None
        self.model = None

        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=region)

        # Setup cloud resources
        self._setup_bucket()
        self._setup_service_account()

    def _setup_bucket(self) -> None:
        """Setup Cloud Storage bucket."""
        if not self.bucket_uri:
            now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            self.bucket_uri = f"gs://{self.project_id}-tmp-{now}-{str(uuid.uuid4())[:4]}"
            self.bucket_name = self.bucket_uri

            # Create bucket
            subprocess.run(
                ["gcloud", "storage", "buckets", "create",
                 "--location", self.region, self.bucket_uri],
                check=True,
            )
            print(f"Created bucket: {self.bucket_uri}")
        else:
            if not self.bucket_uri.startswith("gs://"):
                raise ValueError("BUCKET_URI must start with 'gs://'")
            self.bucket_name = "/".join(self.bucket_uri.split("/")[:3])

        print(f"Using bucket: {self.bucket_uri}")

    def _setup_service_account(self) -> None:
        """Get the default compute service account."""
        result = subprocess.run(
            ["gcloud", "projects", "describe", self.project_id,
             "--format=value(projectNumber)"],
            capture_output=True,
            text=True,
            check=True,
        )
        project_number = result.stdout.strip()
        self.service_account = f"{project_number}-compute@developer.gserviceaccount.com"
        print(f"Using service account: {self.service_account}")

    def upload_model(
        self,
        model_variant: str = "4b-it",
        gpu_memory_utilization: float = 0.95,
        max_model_len: int = 32768,
        max_num_seqs: int = 16,
        max_images: int = 16,
    ) -> aiplatform.Model:
        """
        Upload a MedGemma model to Vertex AI Model Registry.

        Args:
            model_variant: Model variant (e.g., "4b-it", "27b-it")
            gpu_memory_utilization: GPU memory utilization ratio
            max_model_len: Maximum model context length
            max_num_seqs: Maximum number of sequences
            max_images: Maximum number of images per prompt

        Returns:
            Uploaded Model object
        """
        model_id = f"medgemma-{model_variant}"

        # Get hardware config
        size_key = "4b" if "4b" in model_id else "27b" if "27b" in model_id else None
        if not size_key:
            raise ValueError(f"Unknown model variant: {model_variant}")

        hw_config = MedGemmaConfig.HARDWARE_CONFIGS[size_key]
        accelerator_count = hw_config["accelerator_count"]

        # Build vLLM arguments
        vllm_args = [
            "python", "-m", "vllm.entrypoints.api_server",
            "--host=0.0.0.0", "--port=8080",
            f"--model=google/{model_id}",
            f"--tensor-parallel-size={accelerator_count}",
            "--swap-space=16",
            f"--gpu-memory-utilization={gpu_memory_utilization}",
            f"--max-model-len={max_model_len}",
            f"--max-num-seqs={max_num_seqs}",
            "--enable-chunked-prefill",
            "--disable-log-stats",
        ]

        # Add multimodal args for non-text variants
        if "text" not in model_id:
            vllm_args.extend([
                f"--limit_mm_per_prompt='image={max_images}'",
                "--mm-processor-kwargs='{\"do_pan_and_scan\": true}'"
            ])

        env_vars = {
            "MODEL_ID": model_id,
            "DEPLOY_SOURCE": "script",
            "VLLM_USE_V1": "0",
            "HF_TOKEN": self.hf_token,
        }

        model_name = f"{model_id}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

        self.model = aiplatform.Model.upload(
            display_name=model_name,
            serving_container_image_uri=MedGemmaConfig.SERVE_DOCKER_URI,
            serving_container_args=vllm_args,
            serving_container_ports=[8080],
            serving_container_predict_route="/generate",
            serving_container_health_route="/ping",
            serving_container_environment_variables=env_vars,
        )

        print(f"Model uploaded: {self.model.display_name}")
        return self.model

    def run_batch_prediction(
        self,
        instances: list[dict],
        model_variant: str = "4b-it",
        job_prefix: str = "batch-predict",
    ) -> aiplatform.BatchPredictionJob:
        """
        Run batch prediction job.

        Args:
            instances: List of prediction instances
            model_variant: Model variant for hardware config
            job_prefix: Prefix for the job name

        Returns:
            BatchPredictionJob object
        """
        if not self.model:
            raise ValueError("No model uploaded. Call upload_model() first.")

        # Get hardware config
        size_key = "4b" if "4b" in model_variant else "27b"
        hw_config = MedGemmaConfig.HARDWARE_CONFIGS[size_key]

        # Write instances to JSONL file
        os.makedirs("batch_predict_input", exist_ok=True)
        instances_filename = f"instances_{uuid.uuid4()[:8]}.jsonl"
        local_path = f"batch_predict_input/{instances_filename}"

        with open(local_path, "w") as f:
            for instance in instances:
                f.write(json.dumps(instance) + "\n")

        # Upload to Cloud Storage
        model_id = f"medgemma-{model_variant}"
        batch_prefix = f"batch-predict-{model_id}"
        gcs_input_path = f"{self.bucket_uri}/{batch_prefix}/input/{instances_filename}"

        subprocess.run(
            ["gcloud", "storage", "cp", local_path, gcs_input_path],
            check=True,
        )

        # Submit batch job
        job_name = f"{job_prefix}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

        batch_job = self.model.batch_predict(
            job_display_name=job_name,
            gcs_source=gcs_input_path,
            gcs_destination_prefix=f"{self.bucket_uri}/{batch_prefix}/output",
            machine_type=hw_config["machine_type"],
            accelerator_type=hw_config["accelerator_type"],
            accelerator_count=hw_config["accelerator_count"],
            service_account=self.service_account,
        )

        print(f"Batch job submitted: {batch_job.display_name}")
        print("Waiting for job to complete...")

        batch_job.wait()

        print(f"Job completed with state: {batch_job.state}")
        return batch_job

    def get_batch_results(
        self,
        batch_job: aiplatform.BatchPredictionJob,
    ) -> list[dict]:
        """
        Retrieve results from a completed batch job.

        Args:
            batch_job: Completed BatchPredictionJob

        Returns:
            List of prediction results
        """
        output_dir = batch_job.output_info.gcs_output_directory
        output_prefix = output_dir.replace(f"{self.bucket_name}/", "")

        results = []
        client = storage.Client()
        bucket = storage.bucket.Bucket.from_string(self.bucket_name, client)

        blobs = bucket.list_blobs(prefix=f"{output_prefix}/prediction.results")
        for blob in blobs:
            with blob.open("r") as f:
                for line in f:
                    results.append(json.loads(line))

        return results

    def cleanup(self, delete_bucket: bool = False) -> None:
        """
        Clean up resources.

        Args:
            delete_bucket: Whether to delete the Cloud Storage bucket
        """
        if self.model:
            print(f"Deleting model: {self.model.display_name}")
            self.model.delete()

        if delete_bucket and self.bucket_name:
            print(f"Deleting bucket: {self.bucket_name}")
            subprocess.run(
                ["gsutil", "-m", "rm", "-r", self.bucket_name],
                check=True,
            )


def create_multimodal_instance(
    prompt: str,
    image_url: str,
    system_instruction: str = "You are an expert radiologist.",
    max_tokens: int = 200,
) -> dict:
    """
    Create a multimodal batch prediction instance.

    Args:
        prompt: Text prompt
        image_url: URL of the image
        system_instruction: System instruction
        max_tokens: Maximum tokens in response

    Returns:
        Instance dictionary for batch prediction
    """
    return {
        "@requestFormat": "chatCompletions",
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_instruction}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ],
        "max_tokens": max_tokens,
    }


def create_text_instance(
    prompt: str,
    system_instruction: str = "You are a helpful medical assistant.",
    max_tokens: int = 200,
) -> dict:
    """
    Create a text-only batch prediction instance.

    Args:
        prompt: Text prompt
        system_instruction: System instruction
        max_tokens: Maximum tokens in response

    Returns:
        Instance dictionary for batch prediction
    """
    return {
        "@requestFormat": "chatCompletions",
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
    }


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="MedGemma Client for Google Cloud Vertex AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--project-id",
        default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        help="Google Cloud project ID (default: GOOGLE_CLOUD_PROJECT env var)",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("GOOGLE_CLOUD_REGION", "us-central1"),
        help="Google Cloud region (default: us-central1)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Online inference command
    online_parser = subparsers.add_parser(
        "online",
        help="Run online (synchronous) inference",
    )
    online_parser.add_argument(
        "--endpoint-id",
        required=True,
        help="Vertex AI endpoint ID",
    )
    online_parser.add_argument(
        "--endpoint-region",
        help="Endpoint region (default: same as --region)",
    )
    online_parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt for the model",
    )
    online_parser.add_argument(
        "--image-url",
        help="URL of image to include (optional)",
    )
    online_parser.add_argument(
        "--system-instruction",
        default="You are a helpful medical assistant.",
        help="System instruction for the model",
    )
    online_parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum tokens in response",
    )
    online_parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Sampling temperature (default: 0)",
    )
    online_parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable thinking mode (for supported variants)",
    )
    online_parser.add_argument(
        "--use-openai-sdk",
        action="store_true",
        help="Use OpenAI SDK instead of Vertex AI SDK",
    )
    online_parser.add_argument(
        "--no-dedicated-endpoint",
        action="store_true",
        help="Disable dedicated endpoint DNS",
    )

    # Streaming command
    stream_parser = subparsers.add_parser(
        "stream",
        help="Stream response from the model",
    )
    stream_parser.add_argument(
        "--endpoint-id",
        required=True,
        help="Vertex AI endpoint ID",
    )
    stream_parser.add_argument(
        "--endpoint-region",
        help="Endpoint region (default: same as --region)",
    )
    stream_parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt for the model",
    )
    stream_parser.add_argument(
        "--system-instruction",
        default="You are a helpful medical assistant.",
        help="System instruction for the model",
    )
    stream_parser.add_argument(
        "--max-tokens",
        type=int,
        default=1500,
        help="Maximum tokens in response (default: 1500)",
    )
    stream_parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Sampling temperature (default: 0)",
    )
    stream_parser.add_argument(
        "--use-openai-sdk",
        action="store_true",
        help="Use OpenAI SDK for streaming",
    )
    stream_parser.add_argument(
        "--no-dedicated-endpoint",
        action="store_true",
        help="Disable dedicated endpoint DNS",
    )

    # Batch inference command
    batch_parser = subparsers.add_parser(
        "batch",
        help="Run batch (asynchronous) inference",
    )
    batch_parser.add_argument(
        "--hf-token",
        required=True,
        help="Hugging Face token for model access",
    )
    batch_parser.add_argument(
        "--model-variant",
        default="4b-it",
        choices=["4b-it", "27b-it", "27b-text-it"],
        help="Model variant to use (default: 4b-it)",
    )
    batch_parser.add_argument(
        "--input-file",
        help="JSONL file with prediction instances",
    )
    batch_parser.add_argument(
        "--prompt",
        help="Single text prompt (alternative to --input-file)",
    )
    batch_parser.add_argument(
        "--image-url",
        help="Image URL for multimodal prompt",
    )
    batch_parser.add_argument(
        "--bucket",
        help="Cloud Storage bucket URI (created if not provided)",
    )
    batch_parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up resources after completion",
    )
    batch_parser.add_argument(
        "--delete-bucket",
        action="store_true",
        help="Delete bucket during cleanup",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if not args.project_id:
        print("Error: --project-id or GOOGLE_CLOUD_PROJECT environment variable required")
        sys.exit(1)

    if args.command == "online":
        client = MedGemmaClient(
            project_id=args.project_id,
            region=args.region,
            endpoint_id=args.endpoint_id,
            endpoint_region=args.endpoint_region,
            use_dedicated_endpoint=not args.no_dedicated_endpoint,
            is_thinking=args.thinking,
        )

        result = client.predict(
            prompt=args.prompt,
            system_instruction=args.system_instruction,
            image_url=args.image_url,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            use_openai_sdk=args.use_openai_sdk,
        )

        if "thought" in result and result["thought"]:
            print("\n--- MedGemma Thinking Trace ---")
            print(result["thought"])

        print("\n--- MedGemma Response ---")
        print(result["response"])

    elif args.command == "stream":
        client = MedGemmaClient(
            project_id=args.project_id,
            region=args.region,
            endpoint_id=args.endpoint_id,
            endpoint_region=args.endpoint_region,
            use_dedicated_endpoint=not args.no_dedicated_endpoint,
        )

        print("\n--- MedGemma Response (streaming) ---")
        for chunk in client.stream(
            prompt=args.prompt,
            system_instruction=args.system_instruction,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            use_openai_sdk=args.use_openai_sdk,
        ):
            print(chunk, end="", flush=True)
        print()

    elif args.command == "batch":
        batch_client = MedGemmaBatchClient(
            project_id=args.project_id,
            region=args.region,
            hf_token=args.hf_token,
            bucket_uri=args.bucket,
        )

        # Upload model
        batch_client.upload_model(model_variant=args.model_variant)

        # Prepare instances
        if args.input_file:
            with open(args.input_file, "r") as f:
                instances = [json.loads(line) for line in f]
        elif args.prompt:
            if args.image_url:
                instances = [create_multimodal_instance(
                    args.prompt, args.image_url
                )]
            else:
                instances = [create_text_instance(args.prompt)]
        else:
            # Default example
            instances = [create_text_instance(
                "How do you differentiate bacterial from viral pneumonia?"
            )]

        # Run batch prediction
        batch_job = batch_client.run_batch_prediction(
            instances=instances,
            model_variant=args.model_variant,
        )

        # Get results
        results = batch_client.get_batch_results(batch_job)

        print("\n--- Batch Prediction Results ---")
        for i, result in enumerate(results):
            prediction = result["prediction"]["predictions"]["choices"][0]["message"]["content"]
            print(f"\n[Result {i + 1}]")
            print(prediction)

        # Cleanup if requested
        if args.cleanup:
            batch_client.cleanup(delete_bucket=args.delete_bucket)


if __name__ == "__main__":
    main()
