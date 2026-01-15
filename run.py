#!/usr/bin/env python3
"""Entry point for running the MedGemma API server."""

import uvicorn

from src.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "src.app:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
