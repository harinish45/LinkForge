"""Main FastAPI application entry point for LinkForge."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import health_check, router as api_v1_router
from src.api.schemas import HealthResponse

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("linkforge.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager for startup and shutdown events."""
    logger.info("Initializing LinkForge ML Pipeline and API Services...")
    yield
    logger.info("Shutting down LinkForge API Services...")


def create_app() -> FastAPI:
    """FastAPI application factory."""
    app = FastAPI(
        title="LinkForge API",
        description="High-Performance ML-Powered Link & Domain Analysis Engine",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Configure CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes under /api/v1
    app.include_router(api_v1_router, prefix="/api/v1", tags=["Link Analysis"])

    # Root endpoint for basic discovery
    @app.get("/", tags=["General"])
    async def root() -> Dict[str, str]:
        return {
            "service": "LinkForge API",
            "version": "0.1.0",
            "docs_url": "/docs",
            "health_url": "/api/v1/health",
        }

    # Top-level /health alias for convenience
    @app.get("/health", response_model=HealthResponse, tags=["General"])
    async def top_health() -> HealthResponse:
        return await health_check()

    return app


app: FastAPI = create_app()

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
