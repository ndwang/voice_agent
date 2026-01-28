"""
Bilibili Live Chat Service - Entry point
Standalone service for managing Bilibili live stream chat integration.
"""

import uvicorn
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Ensure project root is in path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.logging import setup_logging, get_logger
from bilibili.settings import get_config
from bilibili.api import router, init_manager

# Initialize logging early
config = get_config()
setup_logging(level=config.service.log_level, log_file=None, service_name="bilibili")
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app):
    """Lifespan context manager for service startup/shutdown"""
    # Startup
    manager = init_manager(config)

    # Store manager in app state
    app.state.bilibili_manager = manager

    # Start manager
    await manager.start()
    logger.info("Bilibili service started")

    yield

    # Shutdown
    await manager.stop()
    logger.info("Bilibili service stopped")


def main():
    """Main entry point for Bilibili service"""
    # Config and logging already set up at module level
    HOST = config.service.host
    PORT = config.service.port

    # Create FastAPI app directly (not using core create_app since we have separate config)
    app = FastAPI(
        title="Bilibili Live Chat Service",
        description="Real-time Bilibili live chat integration with WebSocket streaming",
        version="1.0.0",
        lifespan=lifespan
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include router
    app.include_router(router)

    # Serve static files for dashboard
    # Mounted at root so /dashboard.html and /obs.html work directly
    from fastapi.staticfiles import StaticFiles
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=static_dir), name="static")
        logger.info(f"Serving static files from {static_dir}")

    logger.info(f"Starting Bilibili service on {HOST}:{PORT}...")
    logger.info(f"Dashboard: http://{HOST}:{PORT}/dashboard.html")
    logger.info(f"OBS Overlay: http://{HOST}:{PORT}/obs.html")
    logger.info(f"WebSocket: ws://{HOST}:{PORT}/ws/stream")

    # Start uvicorn with WebSocket ping configuration
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        ws_ping_interval=5.0,
        ws_ping_timeout=None
    )


if __name__ == "__main__":
    main()
