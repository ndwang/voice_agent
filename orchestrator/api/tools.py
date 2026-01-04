"""Tool management endpoints."""
from typing import TYPE_CHECKING
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from core.logging import get_logger
from orchestrator.api.dependencies import get_tool_registry

if TYPE_CHECKING:
    from orchestrator.tools.registry import ToolRegistry

logger = get_logger(__name__)
router = APIRouter()


class ToolToggleRequest(BaseModel):
    """Request model for toggling tool state."""
    name: str
    enabled: bool


@router.get("/ui/tools")
async def get_tools(tool_registry: "ToolRegistry" = Depends(get_tool_registry)):
    """Get all tools with their current states."""
    return {"tools": tool_registry.get_all_tools_info()}


@router.post("/ui/tools/toggle")
async def toggle_tool(request: ToolToggleRequest, tool_registry: "ToolRegistry" = Depends(get_tool_registry)):
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
