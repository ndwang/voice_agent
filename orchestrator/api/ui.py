"""UI and interaction endpoints."""
import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from core.logging import get_logger
from core.settings import get_settings, update_settings
from orchestrator.events import EventType
from orchestrator.core.constants import (
    UI_ACTIVITY,
    UI_HISTORY_UPDATED,
    UI_LISTENING_STATE_CHANGED,
    UI_BILIBILI_DANMAKU_STATE_CHANGED,
    UI_BILIBILI_SUPERCHAT_STATE_CHANGED
)
from orchestrator.core.activity_state import get_activity_state
from orchestrator.api.models import (
    SystemPromptUpdate,
    ListeningSetRequest,
    ConfigUpdate,
    BilibiliDanmakuSetRequest,
    BilibiliSuperChatSetRequest
)
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


@router.get("/ui/bilibili/chat")
async def get_bilibili_chat():
    """Get Bilibili chat snapshot."""
    if not orchestrator.bilibili_source.running:
        return {"enabled": False, "danmaku": [], "superchats": []}

    return {
        "enabled": True,
        "danmaku": orchestrator.bilibili_source.get_danmaku_snapshot(),
        "superchats": orchestrator.bilibili_source.get_superchat_snapshot()
    }


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
        EventType.BILIBILI_DANMAKU.value,
        EventType.BILIBILI_SUPERCHAT.value,
        UI_LISTENING_STATE_CHANGED,
        UI_BILIBILI_DANMAKU_STATE_CHANGED,
        UI_BILIBILI_SUPERCHAT_STATE_CHANGED,
        UI_ACTIVITY,
        UI_HISTORY_UPDATED
    ]
    
    for topic in topics:
        bus.subscribe(topic, forward_event)
    
    # Send initial activity state
    activity_state = get_activity_state()
    await websocket.send_json({
        "event": "listening_state_changed",
        "enabled": activity_state.state.listening
    })
    await websocket.send_json({
        "event": "bilibili_danmaku_state_changed",
        "enabled": activity_state.state.bilibili_danmaku_enabled
    })
    await websocket.send_json({
        "event": "bilibili_superchat_state_changed",
        "enabled": activity_state.state.bilibili_superchat_enabled
    })
    await websocket.send_json({
        "event": "activity",
        "state": activity_state.state.to_dict()
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
        elif event_name == EventType.BILIBILI_DANMAKU.value:
            return {"event": "bilibili_danmaku", "message": data}
        elif event_name == EventType.BILIBILI_SUPERCHAT.value:
            return {"event": "bilibili_superchat", "message": data}
        elif event_name == UI_BILIBILI_DANMAKU_STATE_CHANGED:
            return {"event": "bilibili_danmaku_state_changed", "enabled": data.get("enabled", True)}
        elif event_name == UI_BILIBILI_SUPERCHAT_STATE_CHANGED:
            return {"event": "bilibili_superchat_state_changed", "enabled": data.get("enabled", True)}
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
    return {"enabled": get_activity_state().state.listening}


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


@router.get("/ui/bilibili/danmaku/status")
async def get_bilibili_danmaku_status():
    """Get bilibili danmaku enabled state."""
    return {"enabled": get_activity_state().state.bilibili_danmaku_enabled}


@router.post("/ui/bilibili/danmaku/toggle")
async def toggle_bilibili_danmaku():
    """Toggle bilibili danmaku state."""
    current = get_activity_state().state.bilibili_danmaku_enabled
    new_state = not current
    await orchestrator.set_bilibili_danmaku(new_state)
    return {"status": "success", "enabled": new_state}


@router.post("/ui/bilibili/danmaku/set")
async def set_bilibili_danmaku(request: BilibiliDanmakuSetRequest):
    """Set bilibili danmaku state."""
    await orchestrator.set_bilibili_danmaku(request.enabled)
    return {"status": "success", "enabled": request.enabled}


@router.get("/ui/bilibili/superchat/status")
async def get_bilibili_superchat_status():
    """Get bilibili superchat enabled state."""
    return {"enabled": get_activity_state().state.bilibili_superchat_enabled}


@router.post("/ui/bilibili/superchat/toggle")
async def toggle_bilibili_superchat():
    """Toggle bilibili superchat state."""
    current = get_activity_state().state.bilibili_superchat_enabled
    new_state = not current
    await orchestrator.set_bilibili_superchat(new_state)
    return {"status": "success", "enabled": new_state}


@router.post("/ui/bilibili/superchat/set")
async def set_bilibili_superchat(request: BilibiliSuperChatSetRequest):
    """Set bilibili superchat state."""
    await orchestrator.set_bilibili_superchat(request.enabled)
    return {"status": "success", "enabled": request.enabled}


@router.get("/ui/config")
async def get_ui_config():
    """Get full configuration."""
    settings = get_settings()
    return settings.model_dump()


@router.post("/ui/config")
async def update_ui_config(request: ConfigUpdate):
    """Update full configuration with hot-reload support."""
    try:
        # Get reload coordinator from orchestrator if available
        reload_coordinator = None
        if orchestrator and hasattr(orchestrator, 'reload_coordinator'):
            reload_coordinator = orchestrator.reload_coordinator

        # Update settings with reload coordination
        new_settings, reload_results = update_settings(
            request.config,
            persist=True,
            reload_coordinator=reload_coordinator
        )

        # Build response
        response = {"status": "success"}

        # Add reload results if available
        if reload_results:
            # Collect all restart-required items
            needs_restart = []
            for r in reload_results:
                needs_restart.extend(r.restart_required)
            response["needs_restart"] = needs_restart

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save configuration: {str(e)}")

