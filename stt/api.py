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
            except RuntimeError as e:
                # Handle disconnect: "Cannot call 'receive' once a disconnect message has been received"
                if "disconnect" in str(e).lower():
                    logger.debug("WebSocket disconnect detected")
                    break
                raise
                
            # Check for disconnect message
            if message.get("type") == "websocket.disconnect":
                logger.debug("WebSocket disconnect message received")
                break
                
            if "bytes" in message:
                # Process audio - manager handles all state internally
                try:
                    data = message["bytes"]
                    await stt_manager.process_audio_chunk(websocket, data)
                except Exception as e:
                    # Log error but don't crash the connection
                    logger.error(f"Error processing audio chunk: {e}", exc_info=True)
                    # Continue to next message
                    continue
                
            elif "text" in message:
                # Process control messages
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "speech_start":
                        await stt_manager.broadcast({"type": "speech_start"})
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in text message: {e}")
                except Exception as e:
                    logger.warning(f"Error processing text message: {e}", exc_info=True)
                    
    except WebSocketDisconnect:
        logger.debug("WebSocket disconnect exception")
    except Exception as e:
        # Catch any other unexpected errors to prevent server crash
        logger.error(f"STT WebSocket error: {e}", exc_info=True)
    finally:
        try:
            await stt_manager.remove_client(websocket)
        except Exception as e:
            # Even cleanup errors shouldn't crash the server
            logger.error(f"Error removing client: {e}", exc_info=True)


