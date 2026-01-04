from typing import Any, List, Dict, TYPE_CHECKING
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from core.settings import get_settings, update_settings, reload_settings
from orchestrator.api.dependencies import get_orchestrator

if TYPE_CHECKING:
    from orchestrator.server import OrchestratorServer

router = APIRouter(prefix="/api/config", tags=["configuration"])


class ConfigUpdateRequest(BaseModel):
    """Request body for config updates"""
    updates: dict[str, Any]


@router.get("/")
async def get_config():
    """
    Get current configuration.

    Returns:
        Full configuration as JSON
    """
    settings = get_settings()
    return settings.model_dump()


@router.patch("/")
async def update_config(request: ConfigUpdateRequest, orchestrator: "OrchestratorServer" = Depends(get_orchestrator)):
    """
    Update configuration values with hot-reload support.

    Request body example:
    {
        "updates": {
            "llm": {
                "provider": "gemini"
            },
            "orchestrator": {
                "enable_latency_tracking": true
            }
        }
    }

    Returns:
        Updated configuration with reload results
    """
    try:
        # Get reload coordinator from orchestrator
        reload_coordinator = orchestrator.reload_coordinator

        # Update settings with reload coordination
        new_settings, reload_results = update_settings(
            request.updates,
            persist=True,
            reload_coordinator=reload_coordinator
        )

        # Build response
        response = {
            "success": True,
            "message": "Configuration updated and saved",
            "config": new_settings.model_dump()
        }

        # Add reload results if available
        if reload_results:
            response["reload_results"] = [
                {
                    "handler": r.handler_name,
                    "success": r.success,
                    "changes_applied": r.changes_applied,
                    "restart_required": r.restart_required,
                    "errors": r.errors,
                    "warnings": r.warnings
                }
                for r in reload_results
            ]

            # Collect all restart-required items
            needs_restart = []
            for r in reload_results:
                needs_restart.extend(r.restart_required)
            response["needs_restart"] = needs_restart

        return response

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/reload")
async def reload_config():
    """
    Reload configuration from YAML file.
    Useful after manual edits to config.yaml.

    Returns:
        Reloaded configuration
    """
    try:
        settings = reload_settings()
        return {
            "success": True,
            "message": "Configuration reloaded from file",
            "config": settings.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schema")
async def get_config_schema():
    """
    Get JSON schema for configuration.
    Useful for validation and UI generation.

    Returns:
        Pydantic model schema
    """
    from core.settings import AppSettings
    return AppSettings.model_json_schema()


@router.get("/capabilities")
async def get_config_capabilities(orchestrator: "OrchestratorServer" = Depends(get_orchestrator)):
    """
    Get hot-reload capabilities for all config paths.

    Returns:
        Dict mapping config paths to hot-reload capability
    """
    return orchestrator.reload_coordinator.get_capabilities()
