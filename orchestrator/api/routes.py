"""Main API router that combines all route modules."""
from fastapi import APIRouter
from . import health, ui, hotkeys, config, metrics, tools

router = APIRouter()

# Include all route modules
router.include_router(health.router, tags=["health"])
router.include_router(ui.router, tags=["ui"])
router.include_router(hotkeys.router, tags=["hotkeys"])
router.include_router(config.router, tags=["configuration"])
router.include_router(metrics.router, tags=["metrics"])
router.include_router(tools.router, tags=["tools"])

