"""UI and interaction endpoints."""
import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from core.logging import get_logger
from core.config import get_full_config, save_config
from orchestrator.events import EventType
from orchestrator.core.constants import UI_ACTIVITY, UI_HISTORY_UPDATED, UI_LISTENING_STATE_CHANGED
from orchestrator.api.models import SystemPromptUpdate, ListeningSetRequest, ConfigUpdate
from orchestrator.utils.event_helpers import publish_history_updated

logger = get_logger(__name__)
router = APIRouter()

# Global reference (will be injected by server.py)
orchestrator = None


@router.get("/ui")
async def ui_page():
    """Serve simple control panel UI."""
    static_file = Path(__file__).parent.parent / "static" / "ui.html"
    if not static_file.exists():
        return {"error": "UI not found"}
    return FileResponse(static_file)


@router.get("/ui/history")
async def get_history():
    """Get conversation history."""
    return {"history": orchestrator.interaction_manager.context_manager.conversation_history}


@router.post("/ui/history/clear")
async def clear_history():
    """Clear conversation history."""
    orchestrator.interaction_manager.context_manager.clear_history()
    await publish_history_updated(orchestrator.event_bus)
    return {"status": "success", "message": "History cleared"}


@router.websocket("/ui/events")
async def ui_events(websocket: WebSocket):
    """Stream internal events to the browser UI."""
    await websocket.accept()
    queue = asyncio.Queue()
    
    async def forward_event(event):
        await queue.put(event)
    
    bus = orchestrator.event_bus
    topics = [
        EventType.TRANSCRIPT_FINAL.value, 
        EventType.TRANSCRIPT_INTERIM.value,
        EventType.LLM_TOKEN.value, 
        EventType.LLM_RESPONSE_DONE.value,
        EventType.SPEECH_START.value,
        EventType.LLM_CANCELLED.value,
        UI_LISTENING_STATE_CHANGED,
        UI_ACTIVITY,
        UI_HISTORY_UPDATED
    ]
    
    for topic in topics:
        bus.subscribe(topic, forward_event)
    
    # Send initial activity state
    initial_activity_state = orchestrator.interaction_manager.activity_state
    await websocket.send_json({
        "event": "listening_state_changed",
        "enabled": initial_activity_state.listening
    })
    await websocket.send_json({
        "event": "activity",
        "state": initial_activity_state.to_dict()
    })
    
    # Map internal event names to UI event names
    def transform_event(event):
        event_name = event.name
        data = event.data.copy() if event.data else {}
        
        # Transform event names to UI format
        if event_name == EventType.TRANSCRIPT_FINAL.value:
            return {"event": "stt", "stage": "final", "text": data.get("text", "")}
        elif event_name == EventType.TRANSCRIPT_INTERIM.value:
            return {"event": "stt", "stage": "interim", "text": data.get("text", "")}
        elif event_name == EventType.LLM_TOKEN.value:
            return {"event": "llm_token", "token": data.get("token", "")}
        elif event_name == EventType.LLM_RESPONSE_DONE.value:
            return {"event": "llm_done"}
        elif event_name == EventType.LLM_CANCELLED.value:
            return {"event": "llm_cancelled"}
        elif event_name == EventType.SPEECH_START.value:
            return {"event": "cancelled"}
        elif event_name == UI_LISTENING_STATE_CHANGED:
            return {"event": "listening_state_changed", "enabled": data.get("enabled", True)}
        elif event_name == UI_ACTIVITY:
            return {"event": "activity", "state": data.get("state", {})}
        elif event_name == UI_HISTORY_UPDATED:
            return {"event": "history_updated"}
        else:
            # For other events, pass through with original name
            return {"event": event_name, "data": data}
    
    try:
        while True:
            event = await queue.get()
            msg = transform_event(event)
            await websocket.send_json(msg)
    except WebSocketDisconnect:
        pass
    finally:
        for topic in topics:
            bus.unsubscribe(topic, forward_event)


@router.post("/ui/cancel")
async def cancel_interaction():
    """Cancel current interaction."""
    await orchestrator.cancel_interaction()
    return {"status": "cancelled"}


@router.get("/ui/system-prompt")
async def get_system_prompt():
    """Get current system prompt."""
    cm = orchestrator.interaction_manager.context_manager
    return {
        "prompt": cm.get_system_prompt(),
        "file_path": str(cm.get_system_prompt_file_path())
    }


@router.post("/ui/system-prompt")
async def update_system_prompt(request: SystemPromptUpdate):
    """Update system prompt."""
    success = orchestrator.interaction_manager.context_manager.set_system_prompt(request.prompt)
    if success:
        return {"status": "success", "message": "System prompt updated"}
    raise HTTPException(status_code=500, detail="Failed to save system prompt")


@router.get("/ui/listening/status")
async def get_listening_status():
    """Get listening state."""
    return {"enabled": orchestrator.interaction_manager.activity_state.listening}


@router.post("/ui/listening/toggle")
async def toggle_listening():
    """Toggle listening state."""
    enabled = await orchestrator.toggle_listening()
    return {"status": "success", "enabled": enabled}


@router.post("/ui/listening/set")
async def set_listening(request: ListeningSetRequest):
    """Set listening state."""
    await orchestrator.set_listening(request.enabled)
    return {"status": "success", "enabled": request.enabled}


@router.get("/ui/config")
async def get_ui_config():
    """Get full configuration."""
    return get_full_config()


@router.post("/ui/config")
async def update_ui_config(request: ConfigUpdate):
    """Update full configuration."""
    # Update the in-memory config
    config_dict = get_full_config()
    config_dict.clear()
    config_dict.update(request.config)
    
    # Save to file
    if save_config():
        return {"status": "success"}
    raise HTTPException(status_code=500, detail="Failed to save configuration")

