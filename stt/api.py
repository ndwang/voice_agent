import json
import asyncio
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from core.logging import get_logger
from .manager import STTManager

logger = get_logger(__name__)
router = APIRouter()

# Global manager instance
stt_manager = STTManager()

@router.get("/health/queue")
async def queue_health():
    """
    Get queue health status for all connected clients.

    Returns queue depth, dropped chunks, and worker status.
    Useful for monitoring and debugging.
    """
    health_data = []

    async with stt_manager.client_states_lock:
        for websocket, state in stt_manager.client_states.items():
            client_id = id(websocket)
            queue_size = state.audio_queue.qsize() if state.audio_queue else 0
            max_queue_size = stt_manager.AUDIO_QUEUE_SIZE
            worker_alive = state.worker_task and not state.worker_task.done() if state.worker_task else False

            health_data.append({
                "client_id": client_id,
                "queue_depth": queue_size,
                "queue_max": max_queue_size,
                "queue_utilization_pct": round((queue_size / max_queue_size) * 100, 1) if max_queue_size > 0 else 0,
                "chunks_dropped": state.chunks_dropped,
                "worker_alive": worker_alive,
                "is_speaking": state.is_speaking,
                "current_transcript_length": len(state.current_transcript)
            })

    return JSONResponse(content={
        "total_clients": len(health_data),
        "clients": health_data,
        "status": "healthy" if all(c["queue_utilization_pct"] < 80 for c in health_data) else "degraded"
    })

@router.websocket("/ws/audio")
async def audio_input_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for audio input only.

    Clients connecting here send audio chunks but do NOT receive transcripts.
    This prevents backpressure issues with clients that don't consume responses.

    Use this endpoint for:
    - Audio drivers that only send audio
    - Recording clients
    """
    await websocket.accept()
    await stt_manager.add_audio_client(websocket)

    try:
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except RuntimeError as e:
                if "disconnect" in str(e).lower():
                    logger.debug("Audio client disconnect detected")
                    break
                raise

            if message.get("type") == "websocket.disconnect":
                logger.debug("Audio client disconnect message received")
                break

            if "bytes" in message:
                try:
                    data = message["bytes"]
                    await stt_manager.process_audio_chunk(websocket, data)
                except Exception as e:
                    logger.error(f"Error processing audio chunk: {e}", exc_info=True)
                    continue

    except WebSocketDisconnect:
        logger.debug("Audio client disconnect exception")
    except Exception as e:
        logger.error(f"Audio input WebSocket error: {e}", exc_info=True)
    finally:
        try:
            await stt_manager.remove_client(websocket)
        except Exception as e:
            logger.error(f"Error removing audio client: {e}", exc_info=True)


@router.websocket("/ws/transcripts")
async def transcript_output_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for transcript output only.

    Clients connecting here receive transcripts but do NOT send audio.

    Use this endpoint for:
    - Orchestrator/STT source consuming transcripts
    - UI displaying live transcripts
    - Recording/logging services
    """
    await websocket.accept()
    await stt_manager.add_transcript_client(websocket)

    try:
        # Keep connection alive - transcripts are pushed via broadcast
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=10.0)

                # Check for disconnect
                if message.get("type") == "websocket.disconnect":
                    logger.debug("Transcript client disconnect")
                    break

            except asyncio.TimeoutError:
                # Send keepalive ping
                try:
                    await websocket.send_json({"type": "ping"})
                except:
                    break
            except RuntimeError as e:
                if "disconnect" in str(e).lower():
                    break
                raise

    except WebSocketDisconnect:
        logger.debug("Transcript client disconnect exception")
    except Exception as e:
        logger.error(f"Transcript output WebSocket error: {e}", exc_info=True)
    finally:
        try:
            await stt_manager.remove_transcript_client(websocket)
        except Exception as e:
            logger.error(f"Error removing transcript client: {e}", exc_info=True)


@router.websocket("/ws/transcribe")
async def legacy_websocket_endpoint(websocket: WebSocket):
    """
    DEPRECATED: Legacy combined endpoint.

    For backward compatibility. New clients should use:
    - /ws/audio for sending audio
    - /ws/transcripts for receiving transcripts
    """
    logger.warning("Client connected to deprecated /ws/transcribe endpoint. Use /ws/audio or /ws/transcripts instead.")
    await websocket.accept()
    await stt_manager.add_client(websocket)

    try:
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except RuntimeError as e:
                if "disconnect" in str(e).lower():
                    logger.debug("WebSocket disconnect detected")
                    break
                raise

            if message.get("type") == "websocket.disconnect":
                logger.debug("WebSocket disconnect message received")
                break

            if "bytes" in message:
                try:
                    data = message["bytes"]
                    await stt_manager.process_audio_chunk(websocket, data)
                except Exception as e:
                    logger.error(f"Error processing audio chunk: {e}", exc_info=True)
                    continue

            elif "text" in message:
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
        logger.error(f"STT WebSocket error: {e}", exc_info=True)
    finally:
        try:
            await stt_manager.remove_client(websocket)
        except Exception as e:
            logger.error(f"Error removing client: {e}", exc_info=True)


