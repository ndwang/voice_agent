import json
import asyncio
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from core.logging import get_logger
from .manager import STTManager

logger = get_logger(__name__)
router = APIRouter()

# Global manager instance
stt_manager = STTManager()

@router.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time STT.
    """
    await websocket.accept()
    await stt_manager.add_client(websocket)
    
    try:
        while True:
            try:
                # Use receive() to handle both bytes and text
                message = await asyncio.wait_for(websocket.receive(), timeout=1.0)
            except asyncio.TimeoutError:
                # Keep alive check
                continue
                
            if "bytes" in message:
                # Process audio - manager handles all state internally
                data = message["bytes"]
                await stt_manager.process_audio_chunk(websocket, data)
                
            elif "text" in message:
                # Process control messages
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "speech_start":
                        await stt_manager.broadcast({"type": "speech_start"})
                except Exception as e:
                    logger.warning(f"Invalid text message: {e}")
                    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"STT WebSocket error: {e}", exc_info=True)
    finally:
        await stt_manager.remove_client(websocket)


