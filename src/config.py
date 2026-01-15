"""Configuration settings for MedGemma API."""

import os
from dataclasses import dataclass


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    project_id: str = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    region: str = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
    endpoint_id: str = os.environ.get("MEDGEMMA_ENDPOINT_ID", "")
    use_dedicated_endpoint: bool = True

    # Server settings
    host: str = os.environ.get("HOST", "0.0.0.0")
    port: int = int(os.environ.get("PORT", "8000"))

    # Model settings
    max_tokens: int = 2000
    temperature: float = 0.1

    def validate(self) -> list[str]:
        """Validate required settings and return list of errors."""
        errors = []
        if not self.project_id:
            errors.append("GOOGLE_CLOUD_PROJECT environment variable not set")
        if not self.endpoint_id:
            errors.append("MEDGEMMA_ENDPOINT_ID environment variable not set")
        return errors


settings = Settings()
