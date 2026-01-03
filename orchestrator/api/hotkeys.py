"""Hotkey management endpoints."""
from typing import TYPE_CHECKING
from fastapi import APIRouter, HTTPException, Depends
from core.logging import get_logger
from orchestrator.api.models import HotkeyUpdateRequest
from orchestrator.api.dependencies import get_orchestrator

if TYPE_CHECKING:
    from orchestrator.server import OrchestratorServer

logger = get_logger(__name__)
router = APIRouter()


@router.get("/ui/hotkeys")
async def get_hotkeys(orchestrator: "OrchestratorServer" = Depends(get_orchestrator)):
    """Get all registered hotkeys."""
    return {"hotkeys": orchestrator.hotkey_manager.get_registered_hotkeys()}


@router.get("/ui/hotkeys/{hotkey_id}")
async def get_hotkey(hotkey_id: str, orchestrator: "OrchestratorServer" = Depends(get_orchestrator)):
    """Get specific hotkey configuration."""
    hotkey = orchestrator.hotkey_manager.get_hotkey(hotkey_id)
    if not hotkey:
        raise HTTPException(status_code=404, detail="Hotkey not found")
    return {"hotkey_id": hotkey_id, "hotkey": hotkey}


@router.post("/ui/hotkeys/{hotkey_id}")
async def update_hotkey(hotkey_id: str, request: HotkeyUpdateRequest, orchestrator: "OrchestratorServer" = Depends(get_orchestrator)):
    """Update existing hotkey."""
    if orchestrator.hotkey_manager.get_hotkey(hotkey_id):
        success = orchestrator.hotkey_manager.update_hotkey(hotkey_id, request.hotkey)
        if success:
            return {"status": "success", "hotkey": request.hotkey}
    raise HTTPException(status_code=400, detail="Failed to update hotkey")


@router.delete("/ui/hotkeys/{hotkey_id}")
async def delete_hotkey(hotkey_id: str, orchestrator: "OrchestratorServer" = Depends(get_orchestrator)):
    """Delete hotkey."""
    if hotkey_id == "toggle_listening":
        raise HTTPException(status_code=400, detail="Cannot delete core hotkey")
    success = orchestrator.hotkey_manager.unregister_hotkey(hotkey_id)
    if success:
        return {"status": "success"}
    raise HTTPException(status_code=400, detail="Failed to unregister hotkey")

