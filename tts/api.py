import json
from typing import Optional
from fastapi import APIRouter, WebSocket, HTTPException, Response
from pydantic import BaseModel
from core.logging import get_logger
from .manager import TTSManager

logger = get_logger(__name__)
router = APIRouter()

# Global manager instance
tts_manager = TTSManager()

class SynthesizeRequest(BaseModel):
    """Request model for non-streaming TTS."""
    text: str
    voice: Optional[str] = None
    rate: Optional[str] = None
    pitch: Optional[str] = None
    speaker: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None

@router.get("/")
async def root():
    return {
        "message": "TTS Service is running",
        "provider": tts_manager.provider_name,
        "output_sample_rate": tts_manager.output_sample_rate
    }

@router.get("/voices")
async def list_voices():
    try:
        voices = await tts_manager.list_voices()
        return {"voices": voices}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/voices/find")
async def find_voices(
    gender: Optional[str] = None,
    language: Optional[str] = None,
    locale: Optional[str] = None
):
    try:
        filters = {k: v for k, v in locals().items() if v is not None}
        voices = await tts_manager.find_voices(**filters)
        return {"voices": voices}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/synthesize")
async def synthesize(request: SynthesizeRequest):
    try:
        # Filter None values
        params = request.model_dump(exclude_none=True, exclude={"text"})
        
        audio_bytes = await tts_manager.synthesize(request.text, **params)
        
        return Response(
            content=audio_bytes,
            media_type="audio/pcm",
            headers={
                "Content-Type": "audio/pcm",
                "Sample-Rate": str(tts_manager.output_sample_rate),
                "Channels": "1",
                "Format": "int16"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/synthesize/stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming TTS synthesis."""
    await tts_manager.handle_websocket_session(websocket)


