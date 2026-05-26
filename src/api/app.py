"""FastAPI entry point.

Run locally:
    uvicorn src.api.app:app --reload --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

# Load .env into os.environ before any module reads it.
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from ..core.config import API_DESCRIPTION, API_TITLE, API_VERSION, ROOT_DIR
from . import service
from .report import build_report
from .schemas import (
    BatchPredictionResponse,
    ChatRequestSchema,
    DiseaseInfoResponse,
    HealthResponse,
    MutationRequest,
    PredictionResponse,
    SequenceRequest,
    StatsResponse,
    ValidationResponse,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Warm up models and indices once at startup."""
    service.warmup()
    yield


app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Frontend (static SPA) -------------------------------------------------
FRONTEND_DIR: Path = ROOT_DIR / "frontend"
FRONTEND_STATIC_DIR: Path = FRONTEND_DIR / "static"
FRONTEND_INDEX: Path = FRONTEND_DIR / "index.html"

if FRONTEND_STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def root():
    """Serve the frontend SPA, falling back to a JSON API map."""
    if FRONTEND_INDEX.exists():
        return FileResponse(FRONTEND_INDEX)
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "endpoints": [
            "/health",
            "/diseases",
            "/diseases/{name}",
            "/validate",
            "/stats",
            "/predict",
            "/mutation",
            "/batch",
            "/docs",
        ],
    }


@app.get("/api", tags=["meta"])
def api_index() -> dict:
    """JSON API index (kept available even when the SPA is served at /)."""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "endpoints": [
            "/health",
            "/diseases",
            "/diseases/{name}",
            "/validate",
            "/stats",
            "/predict",
            "/mutation",
            "/batch",
            "/docs",
        ],
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> dict:
    return service.health()


@app.get("/diseases", response_model=list[DiseaseInfoResponse], tags=["knowledge"])
def diseases() -> list[dict]:
    return service.diseases_payload()


@app.get("/diseases/{name}", response_model=DiseaseInfoResponse, tags=["knowledge"])
def disease_detail(name: str) -> dict:
    return service.disease_payload(name)


@app.post("/validate", response_model=ValidationResponse, tags=["sequence"])
def validate(req: SequenceRequest) -> dict:
    return service.validate_payload(req.sequence)


@app.post("/stats", response_model=StatsResponse, tags=["sequence"])
def stats(req: SequenceRequest) -> dict:
    return service.stats_payload(req.sequence)


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(req: SequenceRequest) -> dict:
    return service.predict_payload(
        req.sequence,
        include_explain=req.include_explain,
        include_ood=req.include_ood,
        include_similar=req.include_similar,
        top_k_similar=req.top_k_similar,
        model=req.model,
    )


@app.get("/models", tags=["meta"])
def models() -> dict:
    """Return availability of each prediction backbone."""
    return service.models_status()


@app.get("/chat/status", tags=["chat"])
def chat_status() -> dict:
    """Return availability of the RAG chat providers + KB stats."""
    from ..rag import chat as rag_chat

    return rag_chat.status()


@app.post("/chat", tags=["chat"])
def chat_endpoint(req: ChatRequestSchema) -> dict:
    """Answer a user question using retrieval-augmented generation."""
    from ..rag import chat as rag_chat
    from ..rag.chat import ChatRequest as RagChatRequest

    rag_req = RagChatRequest(
        message=req.message,
        provider=req.provider,
        top_k=req.top_k,
        history=req.history,
        prediction_context=req.prediction_context,
    )
    return rag_chat.chat(rag_req).to_dict()


@app.post("/mutation", tags=["analysis"])
def mutation(req: MutationRequest) -> dict:
    return service.mutation_payload(req.reference, req.variant)


@app.post("/batch", response_model=BatchPredictionResponse, tags=["prediction"])
async def batch(file: UploadFile = File(...)) -> dict:
    """Upload a FASTA file (or plain text with multiple sequences)."""
    if file.filename and file.filename.lower().endswith((".gz", ".zip")):
        raise HTTPException(status_code=400, detail="Compressed files are not supported.")
    content = (await file.read()).decode("utf-8", errors="ignore")
    if not content.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return service.batch_payload(content)


@app.post("/report/pdf", tags=["prediction"])
def report_pdf(req: SequenceRequest) -> Response:
    """Generate a PDF analysis report for a sequence."""
    payload = service.predict_payload(
        req.sequence,
        include_explain=req.include_explain,
        include_ood=req.include_ood,
        include_similar=req.include_similar,
        top_k_similar=req.top_k_similar,
        model=req.model,
    )
    pdf_bytes = build_report(payload, req.sequence)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="genomeiq-report.pdf"'},
    )
