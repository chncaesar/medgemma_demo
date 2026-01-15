"""FastAPI application for orthopedic symptom diagnosis."""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .client import get_client
from .config import settings

app = FastAPI(
    title="MedGemma Orthopedic Diagnosis API",
    description="API for orthopedic symptom analysis using MedGemma",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SymptomRequest(BaseModel):
    """Request model for symptom analysis."""

    symptoms: str
    patient_age: int | None = None
    patient_gender: str | None = None
    medical_history: str | None = None


@app.post("/api/diagnose")
async def diagnose_symptoms(request: SymptomRequest) -> StreamingResponse:
    """
    Stream a diagnosis based on orthopedic symptoms.

    This endpoint accepts patient symptoms and streams back a diagnosis
    from the MedGemma model deployed on GCP Vertex AI.
    """
    if not request.symptoms or not request.symptoms.strip():
        raise HTTPException(status_code=400, detail="Symptoms are required")

    # Validate configuration
    errors = settings.validate()
    if errors:
        raise HTTPException(status_code=500, detail=errors[0])

    client = get_client()

    async def generate():
        async for chunk in client.stream_diagnosis(
            symptoms=request.symptoms,
            patient_age=request.patient_age,
            patient_gender=request.patient_gender,
            medical_history=request.medical_history,
        ):
            yield f"data: {json.dumps({'content': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


# Static files setup
STATIC_DIR = Path(__file__).parent.parent / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML page."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Frontend not found. Please create static/index.html"}
