from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, List, Dict
from core.settings import get_settings, update_settings, reload_settings

router = APIRouter(prefix="/api/config", tags=["configuration"])

# Global reference to orchestrator (injected in main)
orchestrator = None


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
async def update_config(request: ConfigUpdateRequest):
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
        # Get reload coordinator from orchestrator if available
        reload_coordinator = None
        if orchestrator and hasattr(orchestrator, 'reload_coordinator'):
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
async def get_config_capabilities():
    """
    Get hot-reload capabilities for all config paths.

    Returns:
        Dict mapping config paths to hot-reload capability
    """
    if orchestrator and hasattr(orchestrator, 'reload_coordinator'):
        return orchestrator.reload_coordinator.get_capabilities()
    else:
        # Fallback: return empty capabilities if orchestrator not initialized
        return {}
