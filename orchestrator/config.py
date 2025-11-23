"""
Configuration Management

Centralized configuration for all service URLs and settings.
"""
import os
from typing import Optional


class Config:
    """Configuration for orchestrator service."""
    
    # STT Server
    STT_WEBSOCKET_URL = os.getenv("STT_WEBSOCKET_URL", "ws://localhost:8001/ws/transcribe")
    
    # LLM Service
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8002")
    LLM_STREAM_URL = f"{LLM_BASE_URL}/generate/stream"
    
    # TTS Service
    TTS_WEBSOCKET_URL = os.getenv("TTS_WEBSOCKET_URL", "ws://localhost:8003/synthesize/stream")
    
    # OCR Service
    OCR_WEBSOCKET_URL = os.getenv("OCR_WEBSOCKET_URL", "ws://localhost:8004/monitor/stream")
    OCR_BASE_URL = os.getenv("OCR_BASE_URL", "http://localhost:8004")
    
    # Orchestrator Server
    ORCHESTRATOR_HOST = os.getenv("ORCHESTRATOR_HOST", "0.0.0.0")
    ORCHESTRATOR_PORT = int(os.getenv("ORCHESTRATOR_PORT", "8000"))
    ORCHESTRATOR_STT_WEBSOCKET_PATH = "/ws/stt"
    
    # Audio Settings
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_CHANNELS = 1
    
    @classmethod
    def get_stt_websocket_url(cls) -> str:
        """Get STT WebSocket URL."""
        return cls.STT_WEBSOCKET_URL
    
    @classmethod
    def get_llm_stream_url(cls) -> str:
        """Get LLM streaming URL."""
        return cls.LLM_STREAM_URL
    
    @classmethod
    def get_tts_websocket_url(cls) -> str:
        """Get TTS WebSocket URL."""
        return cls.TTS_WEBSOCKET_URL
    
    @classmethod
    def get_ocr_websocket_url(cls) -> str:
        """Get OCR WebSocket URL."""
        return cls.OCR_WEBSOCKET_URL
    
    @classmethod
    def get_ocr_base_url(cls) -> str:
        """Get OCR base URL."""
        return cls.OCR_BASE_URL
    
    @classmethod
    def get_orchestrator_stt_url(cls) -> str:
        """Get orchestrator STT WebSocket URL."""
        return f"ws://{cls.ORCHESTRATOR_HOST}:{cls.ORCHESTRATOR_PORT}{cls.ORCHESTRATOR_STT_WEBSOCKET_PATH}"

