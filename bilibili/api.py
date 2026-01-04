"""
FastAPI routes and WebSocket endpoints for Bilibili service.
"""

import json
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse
from pathlib import Path
from core.logging import get_logger
from bilibili.manager import BilibiliManager
from bilibili.settings import get_config, BilibiliServiceConfig
from bilibili import models

logger = get_logger(__name__)

# Router for all Bilibili service endpoints
router = APIRouter()

# Global manager instance (initialized in init_manager)
_manager: Optional[BilibiliManager] = None


def init_manager(config: Optional[BilibiliServiceConfig] = None) -> BilibiliManager:
    """Initialize the BilibiliManager singleton"""
    global _manager
    if _manager is None:
        if config is None:
            config = get_config()
        _manager = BilibiliManager(config)
    return _manager


def get_manager() -> BilibiliManager:
    """Get the initialized manager"""
    if _manager is None:
        raise RuntimeError("Manager not initialized. Call init_manager() first.")
    return _manager


# ============================================================================
# Health & Info Endpoints
# ============================================================================

@router.get("/", response_model=dict)
async def root():
    """Health check and service info"""
    manager = get_manager()
    state = manager.get_state()
    return {
        "service": "Bilibili Live Chat Service",
        "version": "1.0.0",
        "status": "running",
        "connected": state["connected"],
        "room_id": state["room_id"]
    }


@router.get("/health", response_model=models.HealthResponse)
async def health():
    """Detailed health status"""
    manager = get_manager()
    return manager.get_health()


# ============================================================================
# Chat Endpoints
# ============================================================================

@router.get("/chat", response_model=models.ChatSnapshot)
async def get_chat():
    """Get complete chat snapshot (danmaku + superchat)"""
    manager = get_manager()
    snapshot = manager.get_chat_snapshot()
    return snapshot


@router.get("/chat/danmaku", response_model=models.DanmakuSnapshot)
async def get_danmaku():
    """Get danmaku-only snapshot"""
    manager = get_manager()
    return {"danmaku": manager.get_danmaku_snapshot()}


@router.get("/chat/superchat", response_model=models.SuperChatSnapshot)
async def get_superchat():
    """Get superchat-only snapshot"""
    manager = get_manager()
    return {"superchat": manager.get_superchat_snapshot()}


# ============================================================================
# State Endpoints
# ============================================================================

@router.get("/state", response_model=models.ServiceState)
async def get_state():
    """Get service state"""
    manager = get_manager()
    return manager.get_state()


@router.post("/state/danmaku/enable", response_model=models.EnableStateResponse)
async def enable_danmaku():
    """Enable danmaku processing"""
    manager = get_manager()
    manager.set_danmaku_enabled(True)
    await manager._broadcast_state_change()
    return {
        "success": True,
        "danmaku_enabled": True
    }


@router.post("/state/danmaku/disable", response_model=models.EnableStateResponse)
async def disable_danmaku():
    """Disable danmaku processing"""
    manager = get_manager()
    manager.set_danmaku_enabled(False)
    await manager._broadcast_state_change()
    return {
        "success": True,
        "danmaku_enabled": False
    }


@router.post("/state/superchat/enable", response_model=models.EnableStateResponse)
async def enable_superchat():
    """Enable superchat processing"""
    manager = get_manager()
    manager.set_superchat_enabled(True)
    await manager._broadcast_state_change()
    return {
        "success": True,
        "superchat_enabled": True
    }


@router.post("/state/superchat/disable", response_model=models.EnableStateResponse)
async def disable_superchat():
    """Disable superchat processing"""
    manager = get_manager()
    manager.set_superchat_enabled(False)
    await manager._broadcast_state_change()
    return {
        "success": True,
        "superchat_enabled": False
    }


# ============================================================================
# Connection Endpoints
# ============================================================================

@router.post("/connect")
async def connect():
    """Start Bilibili client connection"""
    manager = get_manager()
    try:
        await manager.connect()
        return {"success": True, "connected": True}
    except Exception as e:
        logger.error(f"Failed to connect: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/disconnect")
async def disconnect():
    """Stop Bilibili client connection"""
    manager = get_manager()
    try:
        await manager.disconnect()
        return {"success": True, "connected": False}
    except Exception as e:
        logger.error(f"Failed to disconnect: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Statistics Endpoint
# ============================================================================

@router.get("/stats", response_model=models.ServiceStats)
async def get_stats():
    """Get service statistics"""
    manager = get_manager()
    return manager.get_stats()


# ============================================================================
# Configuration Endpoints
# ============================================================================

@router.get("/config")
async def get_config_endpoint():
    """Get current configuration"""
    config = get_config()
    return {
        "service": {
            "host": config.service.host,
            "port": config.service.port,
            "log_level": config.service.log_level
        },
        "bilibili": {
            "room_id": config.bilibili.room_id,
            "danmaku_ttl_seconds": config.bilibili.danmaku_ttl_seconds,
            "enabled": config.bilibili.enabled,
            "danmaku_enabled_default": config.bilibili.danmaku_enabled_default,
            "superchat_enabled_default": config.bilibili.superchat_enabled_default
        },
        "dashboard": {
            "default_theme": config.dashboard.default_theme,
            "default_max_messages": config.dashboard.default_max_messages,
            "default_font_size": config.dashboard.default_font_size
        }
    }


# ============================================================================
# WebSocket Streaming Endpoint
# ============================================================================

@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time message streaming.

    Protocol:
    Client → Server:
      - {"type": "subscribe", "channels": ["danmaku", "superchat"]}
      - {"type": "unsubscribe", "channels": ["danmaku"]}
      - {"type": "ping"}

    Server → Client:
      - {"type": "danmaku", "data": {...}}
      - {"type": "superchat", "data": {...}}
      - {"type": "state_changed", "data": {...}}
      - {"type": "pong"}
      - {"type": "error", "message": "..."}
    """
    manager = get_manager()
    await websocket.accept()
    manager.add_ws_client(websocket)
    logger.info("WebSocket client connected")

    try:
        # Subscribed channels (default: none, client must subscribe)
        subscribed_channels = set()

        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)
                msg_type = message.get("type")

                if msg_type == "subscribe":
                    # Subscribe to channels
                    channels = message.get("channels", [])
                    subscribed_channels.update(channels)
                    logger.debug(f"Client subscribed to: {channels}")

                elif msg_type == "unsubscribe":
                    # Unsubscribe from channels
                    channels = message.get("channels", [])
                    subscribed_channels -= set(channels)
                    logger.debug(f"Client unsubscribed from: {channels}")

                elif msg_type == "ping":
                    # Respond to ping with pong
                    await websocket.send_json({"type": "pong"})

                else:
                    logger.warning(f"Unknown message type: {msg_type}")

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from client: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format"
                })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.remove_ws_client(websocket)


# ============================================================================
# Static Dashboard Endpoints
# ============================================================================

@router.get("/dashboard.html", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the full dashboard"""
    static_dir = Path(__file__).parent / "static"
    dashboard_path = static_dir / "dashboard.html"

    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")

    with open(dashboard_path, encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@router.get("/obs.html", response_class=HTMLResponse)
async def serve_obs_overlay():
    """Serve the OBS overlay"""
    static_dir = Path(__file__).parent / "static"
    obs_path = static_dir / "obs.html"

    if not obs_path.exists():
        raise HTTPException(status_code=404, detail="OBS overlay not found")

    with open(obs_path, encoding="utf-8") as f:
        return HTMLResponse(content=f.read())
