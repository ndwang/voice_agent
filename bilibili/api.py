"""
FastAPI routes and WebSocket endpoints for Bilibili service.
"""

import json
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
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


@router.get("/chat/paid", response_model=models.PaidSnapshot)
async def get_paid():
    """Get paid messages snapshot (superchat + gift + guard)"""
    manager = get_manager()
    return {"paid": manager.get_paid_snapshot()}


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
    await manager.set_danmaku_enabled(True)
    return {
        "success": True,
        "danmaku_enabled": True
    }


@router.post("/state/danmaku/disable", response_model=models.EnableStateResponse)
async def disable_danmaku():
    """Disable danmaku processing"""
    manager = get_manager()
    await manager.set_danmaku_enabled(False)
    return {
        "success": True,
        "danmaku_enabled": False
    }


@router.post("/state/paid/enable", response_model=models.EnableStateResponse)
async def enable_paid():
    """Enable paid message processing (superchat/gift/guard)"""
    manager = get_manager()
    await manager.set_paid_enabled(True)
    return {
        "success": True,
        "paid_enabled": True
    }


@router.post("/state/paid/disable", response_model=models.EnableStateResponse)
async def disable_paid():
    """Disable paid message processing (superchat/gift/guard)"""
    manager = get_manager()
    await manager.set_paid_enabled(False)
    return {
        "success": True,
        "paid_enabled": False
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
    """Get current configuration (sessdata excluded)"""
    config = get_config()
    return config.model_dump(exclude={"bilibili": {"sessdata"}})


# ============================================================================
# WebSocket Streaming Endpoint
# ============================================================================

@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time message streaming.

    Protocol:
    Client → Server:
      - {"type": "subscribe", "channels": ["danmaku", "paid"]}
      - {"type": "unsubscribe", "channels": ["danmaku"]}
      - {"type": "ping"}

    Server → Client:
      - {"type": "danmaku", "data": {...}}
      - {"type": "paid", "data": {"paid_type": "superchat"|"gift"|"guard", ...}}
      - {"type": "superchat_delete", "data": {"ids": [...]}}
      - {"type": "state_changed", "data": {...}}
      - {"type": "pong"}
      - {"type": "error", "message": "..."}
    """
    manager = get_manager()
    await websocket.accept()
    manager.add_ws_client(websocket)
    try:
        client_host = getattr(websocket.client, "host", None)
        client_port = getattr(websocket.client, "port", None)
        ua = websocket.headers.get("user-agent", "")
        logger.info(f"WebSocket client connected ({client_host}:{client_port}) ua={ua!r}")
    except Exception:
        logger.info("WebSocket client connected")

    try:
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                msg_type = message.get("type")

                if msg_type == "subscribe":
                    channels = message.get("channels", [])
                    manager.subscribe_ws_client(websocket, channels)
                    logger.debug(f"Client subscribed to: {channels}")

                elif msg_type == "unsubscribe":
                    channels = message.get("channels", [])
                    manager.unsubscribe_ws_client(websocket, channels)
                    logger.debug(f"Client unsubscribed from: {channels}")

                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})

                else:
                    logger.warning(f"Unknown message type: {msg_type}")

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from client: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format"
                })

    except WebSocketDisconnect as e:
        # Close code reference:
        # - 1000: normal closure (often OBS deactivating/reloading browser source)
        # - 1001: going away (navigation/reload)
        # - 1006: abnormal closure (network drop / crash)
        code = getattr(e, "code", None)
        logger.info(f"WebSocket client disconnected (code={code})")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.remove_ws_client(websocket)
