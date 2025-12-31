"""Tool management endpoints."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Global reference (injected by server.py)
tool_registry = None


class ToolToggleRequest(BaseModel):
    """Request model for toggling tool state."""
    name: str
    enabled: bool


@router.get("/ui/tools")
async def get_tools():
    """Get all tools with their current states."""
    return {"tools": tool_registry.get_all_tools_info()}


@router.post("/ui/tools/toggle")
async def toggle_tool(request: ToolToggleRequest):
    """Enable or disable a specific tool."""
    if request.enabled:
        success = tool_registry.enable_tool(request.name)
    else:
        success = tool_registry.disable_tool(request.name)

    if not success:
        raise HTTPException(status_code=404, detail=f"Tool '{request.name}' not found")

    return {
        "status": "success",
        "name": request.name,
        "enabled": request.enabled
    }
