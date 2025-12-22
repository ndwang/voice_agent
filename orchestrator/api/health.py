"""Health check endpoints."""
from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def root():
    """Service status endpoint."""
    return {"message": "Orchestrator Service is running"}


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Orchestrator"}

