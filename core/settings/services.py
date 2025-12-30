from pydantic import BaseModel


class ServicesSettings(BaseModel):
    """Service URLs configuration"""
    orchestrator_base_url: str = "http://localhost:8000"
    stt_websocket_url: str = "ws://localhost:8001/ws/transcribe"  # DEPRECATED: use stt_audio_url or stt_transcript_url
    stt_audio_url: str = "ws://localhost:8001/ws/audio"  # For sending audio chunks only
    stt_transcript_url: str = "ws://localhost:8001/ws/transcripts"  # For receiving transcripts only
    tts_websocket_url: str = "ws://localhost:8003/synthesize/stream"
    ocr_websocket_url: str = "ws://localhost:8004/monitor/stream"
    ocr_base_url: str = "http://localhost:8004"

    @classmethod
    def from_dict(cls, data: dict) -> "ServicesSettings":
        """Create ServicesSettings from config dict"""
        return cls(**data)
