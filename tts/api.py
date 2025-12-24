import json
from typing import Optional
from fastapi import APIRouter, WebSocket, HTTPException, Response
from pydantic import BaseModel
from core.logging import get_logger
from .engine import TTSEngine

logger = get_logger(__name__)
router = APIRouter()

# Global TTS engine instance
tts_engine = TTSEngine()

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
        "provider": tts_engine.provider_name,
        "output_sample_rate": tts_engine.output_sample_rate
    }

@router.get("/voices")
async def list_voices():
    try:
        voices = await tts_engine.list_voices()
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
        voices = await tts_engine.find_voices(**filters)
        return {"voices": voices}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/synthesize")
async def synthesize(request: SynthesizeRequest):
    try:
        logger.info(f"[TTS] Non-streaming synthesis request ({len(request.text)} chars): {request.text!r}")
        # Filter None values
        params = request.model_dump(exclude_none=True, exclude={"text"})
        
        audio_bytes = await tts_engine.synthesize(request.text, **params)
        
        return Response(
            content=audio_bytes,
            media_type="audio/pcm",
            headers={
                "Content-Type": "audio/pcm",
                "Sample-Rate": str(tts_engine.output_sample_rate),
                "Channels": "1",
                "Format": "int16"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/synthesize/stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming TTS synthesis."""
    await tts_engine.handle_websocket_session(websocket)


