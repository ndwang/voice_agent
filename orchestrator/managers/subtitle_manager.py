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
    Accumulates sentences from one conversation round to show the full response.
    """
    
    def __init__(self, event_bus: EventBus):
        self.source_name = get_config("obs", "subtitle_source", default="subtitle")
        self.obs_client = None
        self.accumulated_text = ""  # Accumulates sentences from current conversation round
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
        self.event_bus.subscribe(EventType.SUBTITLE_REQUEST.value, self.on_subtitle_request)

    async def on_speech_start(self, event: Event):
        """Clear subtitles when user starts talking."""
        self.accumulated_text = ""
        if self.obs_client:
            try:
                if not self.obs_client.ws:
                    await self.obs_client.connect()
                await self.obs_client.set_text(self.source_name, "")
            except Exception as e:
                self.logger.debug(f"OBS error: {e}")

    async def on_subtitle_request(self, event: Event):
        """Update subtitles when Chinese content is available.
        Accumulates sentences from the same conversation round."""
        text = event.data.get("text", "")
        if self.obs_client and text:
            try:
                if not self.obs_client.ws:
                    await self.obs_client.connect()
                
                # Accumulate the new sentence
                self.accumulated_text += text
                
                # Update OBS with the accumulated text
                await self.obs_client.set_text(self.source_name, self.accumulated_text)
            except Exception as e:
                self.logger.debug(f"OBS error: {e}")

