"""
Core Server Factory

Standardized FastAPI application factory.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any
from .config import get_config
from .logging import setup_logging, get_logger

logger = get_logger(__name__)


def create_app(
    title: str,
    description: str = "",
    version: str = "0.1.0",
    lifespan: Optional[Any] = None
) -> FastAPI:
    """
    Create a standardized FastAPI application.
    
    Args:
        title: API title.
        description: API description.
        version: API version.
        lifespan: Optional lifespan context manager.
        
    Returns:
        Configured FastAPI app.
    """
    # Initialize configuration
    log_level = get_config("orchestrator", "log_level", default="INFO")
    setup_logging(level=log_level)
    
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


