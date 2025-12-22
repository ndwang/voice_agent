from core.event_bus import EventBus, Event
from core.logging import get_logger
from core.config import get_config
from orchestrator.events import EventType
from orchestrator.managers.base import BaseManager
from orchestrator.clients.obs_client import OBSClient

logger = get_logger(__name__)

class SubtitleManager(BaseManager):
    """
    Manages OBS subtitles.
    Updates the text source in OBS as audio is played.
    """
    
    def __init__(self, event_bus: EventBus):
        self.source_name = get_config("obs", "subtitle_source", default="subtitle")
        self.obs_client = None
        super().__init__(event_bus)
        
        try:
            self.obs_client = OBSClient()
            # Note: OBSClient.connect() is async, but we can't await here
            # The connection will be attempted when methods are called
            logger.info("Subtitle Manager initialized with OBS client")
        except Exception as e:
            logger.warning(f"Failed to initialize OBS client: {e}")

    def _register_handlers(self):
        self.event_bus.subscribe(EventType.SPEECH_START.value, self.on_speech_start)
        self.event_bus.subscribe(EventType.TTS_REQUEST.value, self.on_tts_request)

    async def on_speech_start(self, event: Event):
        """Clear subtitles when user starts talking."""
        if self.obs_client:
            try:
                if not self.obs_client.ws:
                    await self.obs_client.connect()
                await self.obs_client.set_text(self.source_name, "")
            except Exception as e:
                self.logger.debug(f"OBS error: {e}")

    async def on_tts_request(self, event: Event):
        """Update subtitles when TTS starts speaking a sentence."""
        text = event.data.get("text", "")
        if self.obs_client and text:
            try:
                if not self.obs_client.ws:
                    await self.obs_client.connect()
                await self.obs_client.set_text(self.source_name, text)
            except Exception as e:
                self.logger.debug(f"OBS error: {e}")

