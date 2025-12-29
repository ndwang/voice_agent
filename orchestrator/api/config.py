from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any
from core.settings import get_settings, update_settings, reload_settings

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
async def update_config(request: ConfigUpdateRequest):
    """
    Update configuration values.

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
        Updated configuration
    """
    try:
        new_settings = update_settings(request.updates, persist=True)

        return {
            "success": True,
            "message": "Configuration updated and saved",
            "config": new_settings.model_dump()
        }
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
