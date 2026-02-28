"""
FastAPI REST Service for Enterprise LLM Platform with RAG
Project 303 - Reference Implementation

Endpoints:
  POST /v1/chat       - RAG query (retrieval + generation)
  POST /v1/documents  - Ingest documents into vector DB
  GET  /health        - Health check (Cloud Run / K8s readiness probe)
  GET  /metrics       - Prometheus metrics

Runs on:
  - Google Colab: uvicorn src.api.main:app --port 8080
  - GCP Cloud Run: PORT=8080 automatically
  - Local: uvicorn src.api.main:app --reload
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Prometheus Metrics (same metrics as original monitoring/prometheus/llm-rules.yaml)
# ─────────────────────────────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "rag_requests_total",
    "Total RAG requests",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "rag_request_duration_seconds",
    "RAG request duration in seconds",
    ["endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)
QUERY_LATENCY = Histogram(
    "rag_query_duration_seconds",
    "End-to-end RAG query latency",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
)
DOCUMENTS_INDEXED = Counter(
    "rag_documents_indexed_total",
    "Total documents indexed",
)
SAFETY_VIOLATIONS = Counter(
    "guardrails_violations_total",
    "Safety guardrails violations",
    ["violation_type", "risk_level"],
)
SAFETY_CHECKS = Counter(
    "guardrails_checks_total",
    "Safety guardrails checks performed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    """RAG query request"""
    query: str = Field(..., description="User question", min_length=1, max_length=4096)
    user_id: str = Field(default="anonymous", description="User identifier for rate limiting")
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata filters for retrieval (e.g. {'source': 'confluence'})",
    )
    return_sources: bool = Field(default=True, description="Include source documents in response")
    llm_backend: Optional[str] = Field(
        default=None,
        description="Override LLM backend: 'gemini' | 'vllm'. Defaults to LLM_BACKEND env var",
    )


class DocumentIngestRequest(BaseModel):
    """Document ingestion request"""
    documents: List[Dict[str, Any]] = Field(
        ...,
        description="List of documents with 'id', 'text', and 'metadata' fields",
        min_length=1,
    )
    chunk_size: int = Field(default=512, ge=64, le=2048)
    chunk_overlap: int = Field(default=50, ge=0, le=256)


class ChatResponse(BaseModel):
    """RAG query response"""
    answer: str
    model: str
    backend: str
    latency_ms: float
    num_documents_retrieved: int
    sources: Optional[List[Dict[str, Any]]] = None
    usage: Optional[Dict[str, int]] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    services: Dict[str, str]
    version: str = "1.0.0"
    model: str


# ─────────────────────────────────────────────────────────────────────────────
# App Lifespan (initialise pipeline + guardrails once at startup)
# ─────────────────────────────────────────────────────────────────────────────

# Global singletons
_pipeline = None
_guardrails = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle"""
    global _pipeline, _guardrails

    logger.info("Starting LLM RAG Platform...")

    # Initialise RAG pipeline
    from src.rag.pipeline import RAGPipeline, RAGConfig
    config = RAGConfig.from_env()
    logger.info(f"RAG config: embedding={config.embedding_model}, backend={config.llm_backend}")
    _pipeline = RAGPipeline(config)

    # Initialise safety guardrails
    from src.guardrails.safety import SafetyGuardrails, GuardrailsConfig
    guardrails_config = GuardrailsConfig(
        enable_pii_detection=os.getenv("ENABLE_PII_DETECTION", "true").lower() == "true",
        enable_content_moderation=os.getenv("ENABLE_CONTENT_MODERATION", "false").lower() == "true",
        enable_prompt_injection_detection=True,
        max_requests_per_minute=int(os.getenv("RATE_LIMIT_RPM", "60")),
    )
    _guardrails = SafetyGuardrails(guardrails_config)

    logger.info("LLM RAG Platform ready ✓")
    yield

    logger.info("Shutting down LLM RAG Platform...")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Enterprise LLM Platform with RAG",
    description=(
        "Project 303 - Reference Implementation. "
        "Supports Gemini Pro (GCP/Colab) and vLLM (self-hosted) backends."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS - allow all origins for Colab / dev; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Auth (simple API key - sufficient for initial deployment)
# ─────────────────────────────────────────────────────────────────────────────

API_KEY = os.getenv("API_KEY", "")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Verify API key if one is configured"""
    if not API_KEY:
        return "anonymous"  # No key configured - open access (dev/Colab)
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Set X-API-Key header.",
        )
    return api_key


# ─────────────────────────────────────────────────────────────────────────────
# Middleware: Request timing
# ─────────────────────────────────────────────────────────────────────────────

@app.middleware("http")
async def add_timing(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    response.headers["X-Request-Duration-Ms"] = f"{elapsed * 1000:.0f}"
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(elapsed)
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["Infrastructure"])
async def health_check():
    """
    Health check endpoint.
    Used by Cloud Run, GKE readiness probes, and monitoring.
    """
    services: Dict[str, str] = {}

    # Check Qdrant
    try:
        if _pipeline:
            _pipeline.vector_db.get_collections()
            services["qdrant"] = "ok"
        else:
            services["qdrant"] = "initializing"
    except Exception as e:
        services["qdrant"] = f"error: {str(e)[:50]}"

    # Check LLM Gateway availability
    if _pipeline:
        backends = _pipeline.llm_gateway.available_backends
        services["llm"] = f"ok ({', '.join(backends)})"
        model_name = _pipeline.config.gemini_model
    else:
        services["llm"] = "initializing"
        model_name = "unknown"

    overall = "ok" if all(v == "ok" or v.startswith("ok") for v in services.values()) else "degraded"

    return HealthResponse(
        status=overall,
        services=services,
        model=model_name,
    )


@app.get("/metrics", tags=["Infrastructure"])
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/chat", response_model=ChatResponse, tags=["RAG"])
async def chat(
    request: ChatRequest,
    _: str = Depends(verify_api_key),
):
    """
    RAG-powered chat endpoint.

    Retrieves relevant documents from the knowledge base and generates
    a grounded answer using the configured LLM (Gemini Pro or vLLM).
    """
    if not _pipeline or not _guardrails:
        raise HTTPException(status_code=503, detail="Service initializing, please retry in a moment")

    start_time = time.time()
    SAFETY_CHECKS.inc()

    # ── Safety: Input Validation ──────────────────────────────────────────────
    try:
        passed, safety_results = await _guardrails.validate_input(
            request.query, request.user_id
        )
    except Exception as e:
        logger.error(f"Safety check failed: {e}")
        safety_results = []
        passed = True  # Fail open on safety check errors

    # Collect PII info for routing decision
    contains_pii = False
    for result in safety_results:
        if not result.passed:
            violation_type = "unknown"
            if result.violations:
                violation_type = result.violations[0].split(":")[0].lower().replace(" ", "_")
            SAFETY_VIOLATIONS.labels(
                violation_type=violation_type,
                risk_level=result.risk_level.value,
            ).inc()
            # Check for PII violations
            if any("pii" in v.lower() for v in result.violations):
                contains_pii = True

    if not passed:
        REQUEST_COUNT.labels(endpoint="/v1/chat", status="rejected").inc()
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Request blocked by safety guardrails",
                "violations": [
                    v
                    for r in safety_results
                    if not r.passed
                    for v in r.violations
                ],
            },
        )

    # ── RAG Query ─────────────────────────────────────────────────────────────
    try:
        from src.llm.gateway import LLMBackend

        result = await _pipeline.query(
            query=request.query,
            filters=request.filters,
            return_sources=request.return_sources,
            contains_pii=contains_pii,
        )
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        REQUEST_COUNT.labels(endpoint="/v1/chat", status="error").inc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    # ── Safety: Output Validation ─────────────────────────────────────────────
    output_safety = await _guardrails.validate_output(result["answer"])
    if not output_safety.passed:
        logger.warning(f"Output failed safety check: {output_safety.violations}")
        REQUEST_COUNT.labels(endpoint="/v1/chat", status="output_filtered").inc()
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Generated response blocked by output safety filter",
                "violations": output_safety.violations,
            },
        )

    # ── Metrics & Response ────────────────────────────────────────────────────
    latency_ms = (time.time() - start_time) * 1000
    QUERY_LATENCY.observe(latency_ms / 1000)
    REQUEST_COUNT.labels(endpoint="/v1/chat", status="success").inc()

    return ChatResponse(
        answer=result["answer"],
        model=result.get("model", "unknown"),
        backend=result.get("backend", "unknown"),
        latency_ms=latency_ms,
        num_documents_retrieved=result.get("num_documents_retrieved", 0),
        sources=result.get("sources"),
        usage=result.get("usage"),
    )


@app.post("/v1/documents", tags=["RAG"])
async def ingest_documents(
    request: DocumentIngestRequest,
    _: str = Depends(verify_api_key),
):
    """
    Ingest documents into the vector knowledge base.

    Documents will be chunked, embedded using HuggingFace sentence-transformers,
    and stored in Qdrant for retrieval.
    """
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Service initializing")

    from src.rag.pipeline import Document, chunk_document

    all_chunks: List[Document] = []
    for raw_doc in request.documents:
        doc_id = raw_doc.get("id")
        text = raw_doc.get("text", "")
        metadata = raw_doc.get("metadata", {})

        if not doc_id or not text:
            raise HTTPException(
                status_code=422,
                detail="Each document must have 'id' and 'text' fields",
            )

        chunks = chunk_document(
            document_text=text,
            doc_id=doc_id,
            metadata=metadata,
            chunk_size=request.chunk_size,
            overlap=request.chunk_overlap,
        )
        all_chunks.extend(chunks)

    try:
        count = await _pipeline.add_documents(all_chunks)
        DOCUMENTS_INDEXED.inc(count)
        return {
            "status": "ok",
            "documents_received": len(request.documents),
            "chunks_indexed": count,
        }
    except Exception as e:
        logger.error(f"Document ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


class RetrieveRequest(BaseModel):
    """Vector retrieval-only request (no LLM generation)"""
    query: str = Field(..., description="Search query", min_length=1, max_length=4096)
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")


@app.post("/v1/retrieve", tags=["RAG"])
async def retrieve(
    request: RetrieveRequest,
    _: str = Depends(verify_api_key),
):
    """
    Retrieve relevant document chunks from the vector knowledge base without LLM generation.
    Useful for inspecting what the retriever finds for a given query.
    """
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Service initializing")

    try:
        documents = await _pipeline.retrieve_dense(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters,
        )
        documents = await _pipeline.rerank(query=request.query, documents=documents, top_k=request.top_k)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    return {
        "query": request.query,
        "num_results": len(documents),
        "results": [
            {
                "id": doc.id,
                "text": doc.text,
                "metadata": doc.metadata,
                "score": doc.score,
            }
            for doc in documents
        ],
    }

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("ENV", "production") == "development",
        log_level="info",
    )
