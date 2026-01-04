from pydantic import BaseModel, Field
from typing import Optional


class OrchestratorSettings(BaseModel):
    """Orchestrator service configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    stt_websocket_path: str = "/ws/stt"
    log_level: str = "INFO"
    log_file: Optional[str] = None
    system_prompt_file: Optional[str] = None
    queue_cooldown_seconds: float = 3.0  # Cooldown after TURN_ENDED before processing non-voice queue items
    interrupt_timeout_seconds: float = 10.0  # Timeout to clear pending voice interrupt if no transcript arrives
    hotkeys: dict = Field(default_factory=lambda: {
        "toggle_listening": "ctrl+shift+l",
        "cancel_speech": "ctrl+shift+c"
    })

    @classmethod
    def from_dict(cls, data: dict) -> "OrchestratorSettings":
        """Create from config dict"""
        return cls(**data)
