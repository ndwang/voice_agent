"""
Core Server Factory

Standardized FastAPI application factory.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any
from .settings import get_settings
from .logging import setup_logging, get_logger

logger = get_logger(__name__)


def create_app(
    title: str,
    description: str = "",
    version: str = "0.1.0",
    lifespan: Optional[Any] = None,
    service_name: Optional[str] = None
) -> FastAPI:
    """
    Create a standardized FastAPI application.
    
    Args:
        title: API title.
        description: API description.
        version: API version.
        lifespan: Optional lifespan context manager.
        service_name: Service name for config lookup (e.g., "orchestrator", "stt", "tts", "ocr").
                     If None, defaults to "orchestrator" for backward compatibility.
        
    Returns:
        Configured FastAPI app.
    """
    # Initialize configuration
    if service_name is None:
        service_name = "orchestrator"

    settings = get_settings()
    service_settings = getattr(settings, service_name)
    log_level = service_settings.log_level
    log_file = service_settings.log_file
    setup_logging(level=log_level, log_file=log_file, service_name=service_name)
    
    app = FastAPI(
        title=title,
        description=description,
        version=version,
        lifespan=lifespan
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins for development
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add standard endpoints
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": title}
    
    logger.info(f"Initialized {title} service")
    
    return app


