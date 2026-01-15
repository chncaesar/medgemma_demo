"""MedGemma streaming client for Vertex AI."""

import json
from typing import AsyncGenerator

import google.auth
import google.auth.transport.requests
import requests
from google.cloud import aiplatform

from .config import settings


class MedGemmaClient:
    """Streaming client for MedGemma on Vertex AI."""

    SYSTEM_PROMPT = (
        "You are an expert orthopedic physician assistant. "
        "Provide thorough, evidence-based assessments of orthopedic conditions. "
        "Always recommend consulting with a healthcare provider for proper diagnosis and treatment. "
        "Be clear about the limitations of remote assessment."
    )

    def __init__(
        self,
        project_id: str | None = None,
        region: str | None = None,
        endpoint_id: str | None = None,
        use_dedicated_endpoint: bool | None = None,
    ):
        self.project_id = project_id or settings.project_id
        self.region = region or settings.region
        self.endpoint_id = endpoint_id or settings.endpoint_id
        self.use_dedicated_endpoint = (
            use_dedicated_endpoint
            if use_dedicated_endpoint is not None
            else settings.use_dedicated_endpoint
        )
        self._credentials = None
        self._endpoint = None

    @property
    def endpoint(self) -> aiplatform.Endpoint:
        """Lazy load the Vertex AI endpoint."""
        if self._endpoint is None:
            aiplatform.init(project=self.project_id, location=self.region)
            self._endpoint = aiplatform.Endpoint(
                endpoint_name=self.endpoint_id,
                project=self.project_id,
                location=self.region,
            )
        return self._endpoint

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
                f"https://{self.region}-aiplatform.googleapis.com/"
                f"v1beta1/{endpoint_resource_name}"
            )

    def _build_prompt(
        self,
        symptoms: str,
        patient_age: int | None = None,
        patient_gender: str | None = None,
        medical_history: str | None = None,
    ) -> str:
        """Build the diagnosis prompt with patient context."""
        parts = [f"Patient presents with the following orthopedic symptoms:\n{symptoms}"]

        if patient_age:
            parts.append(f"\nPatient age: {patient_age} years old")
        if patient_gender:
            parts.append(f"\nPatient gender: {patient_gender}")
        if medical_history:
            parts.append(f"\nRelevant medical history: {medical_history}")

        parts.append(
            "\n\nPlease provide:\n"
            "1. Possible diagnoses (differential diagnosis)\n"
            "2. Recommended physical examinations\n"
            "3. Suggested diagnostic tests or imaging\n"
            "4. Initial treatment recommendations\n"
            "5. Red flags that would require immediate attention"
        )

        return "".join(parts)

    async def stream_diagnosis(
        self,
        symptoms: str,
        patient_age: int | None = None,
        patient_gender: str | None = None,
        medical_history: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a diagnosis response from MedGemma.

        Args:
            symptoms: Patient's orthopedic symptoms
            patient_age: Patient's age (optional)
            patient_gender: Patient's gender (optional)
            medical_history: Relevant medical history (optional)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature

        Yields:
            Response chunks as they arrive
        """
        user_prompt = self._build_prompt(
            symptoms, patient_age, patient_gender, medical_history
        )

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        creds = self._get_credentials()
        base_url = self._get_base_url()
        endpoint_url = f"{base_url}/chat/completions"

        request_body = {
            "messages": messages,
            "max_tokens": max_tokens or settings.max_tokens,
            "temperature": temperature or settings.temperature,
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
                    try:
                        chunk_data = json.loads(chunk_str)
                        content = chunk_data["choices"][0]["delta"].get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue


# Singleton client instance
_client: MedGemmaClient | None = None


def get_client() -> MedGemmaClient:
    """Get or create the singleton MedGemma client."""
    global _client
    if _client is None:
        _client = MedGemmaClient()
    return _client
