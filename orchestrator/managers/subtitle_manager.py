import asyncio
from core.event_bus import EventBus, Event
from core.logging import get_logger
from core.settings import get_settings
from orchestrator.events import EventType
from orchestrator.managers.base import BaseManager
from orchestrator.clients.obs_client import OBSClient

logger = get_logger(__name__)

class SubtitleManager(BaseManager):
    """
    Manages OBS subtitles.
    Updates the text source in OBS as audio is played.
    Accumulates sentences from one conversation round to show the full response.
    Supports TTL (Time To Live) - clears subtitles after inactivity.
    """

    def __init__(self, event_bus: EventBus):
        settings = get_settings()
        self.source_name = settings.obs.subtitle_source
        self.ttl_seconds = settings.obs.subtitle_ttl_seconds
        self.visibility_source = settings.obs.visibility_source
        self.appear_filter_name = settings.obs.appear_filter_name
        self.clear_filter_name = settings.obs.clear_filter_name
        self.obs_client = None
        self.accumulated_text = ""  # Accumulates sentences from current conversation round
        self.clear_task = None  # Task for TTL-based clearing
        super().__init__(event_bus)
        
        try:
            self.obs_client = OBSClient()
            # Note: OBSClient.connect() is async, but we can't await here
            # The connection will be attempted when methods are called
            visibility_info = ""
            if self.visibility_source:
                filters = []
                if self.appear_filter_name:
                    filters.append(f"appear: {self.appear_filter_name}")
                if self.clear_filter_name:
                    filters.append(f"clear: {self.clear_filter_name}")
                if filters:
                    visibility_info = f", visibility filters ({self.visibility_source}): {', '.join(filters)}"
            logger.info(f"Subtitle Manager initialized with OBS client (TTL: {self.ttl_seconds}s{visibility_info})")
        except Exception as e:
            logger.warning(f"Failed to initialize OBS client: {e}")

    def _register_handlers(self):
        self.event_bus.subscribe(EventType.SPEECH_START.value, self.on_speech_start)
        self.event_bus.subscribe(EventType.SUBTITLE_REQUEST.value, self.on_subtitle_request)

    def _reset_ttl_timer(self):
        """Reset the TTL timer. Cancels existing timer and starts a new one if TTL is enabled."""
        # Cancel existing timer if it exists
        if self.clear_task and not self.clear_task.done():
            self.clear_task.cancel()
        
        # Start new timer if TTL is enabled (> 0)
        if self.ttl_seconds > 0:
            self.clear_task = asyncio.create_task(self._clear_after_ttl())
    
    async def _enable_filter(self, filter_name: str):
        """Enable a filter on the visibility source."""
        if not self.obs_client or not self.visibility_source or not filter_name:
            return
        
        try:
            if not self.obs_client.ws:
                await self.obs_client.connect()
            
            await self.obs_client.set_filter_visibility(
                self.visibility_source, 
                filter_name, 
                True
            )
            self.logger.debug(f"Enabled filter '{filter_name}' on source '{self.visibility_source}'")
        except Exception as e:
            self.logger.debug(f"OBS error while enabling filter: {e}")

    async def _clear_subtitles(self):
        """Clear subtitles text and enable clear filter."""
        self.accumulated_text = ""
        if self.obs_client:
            try:
                if not self.obs_client.ws:
                    await self.obs_client.connect()
                await self.obs_client.set_text(self.source_name, "")
                # Enable clear filter when subtitles are cleared
                await self._enable_filter(self.clear_filter_name)
            except Exception as e:
                self.logger.debug(f"OBS error while clearing: {e}")

    async def _clear_after_ttl(self):
        """Clear subtitles after TTL expires."""
        try:
            await asyncio.sleep(self.ttl_seconds)
            # Only clear if we still have accumulated text (no new subtitles arrived)
            if self.accumulated_text:
                await self._clear_subtitles()
                self.logger.debug(f"Cleared subtitles after {self.ttl_seconds}s TTL")
        except asyncio.CancelledError:
            # Timer was cancelled (new subtitle arrived or speech started)
            pass

    async def on_speech_start(self, event: Event):
        """Clear subtitles when user starts talking."""
        # Cancel any pending TTL timer
        if self.clear_task and not self.clear_task.done():
            self.clear_task.cancel()
        
        await self._clear_subtitles()

    async def on_subtitle_request(self, event: Event):
        """Update subtitles when Chinese content is available.
        Accumulates sentences from the same conversation round.
        Resets TTL timer on each update.
        Shows visibility source when subtitles first appear."""
        text = event.data.get("text", "")
        if self.obs_client and text:
            try:
                if not self.obs_client.ws:
                    await self.obs_client.connect()
                
                # Enable appear filter when subtitles first appear
                if not self.accumulated_text:
                    await self._enable_filter(self.appear_filter_name)

                # Accumulate the new sentence
                self.accumulated_text += text
                
                # Update OBS with the accumulated text
                await self.obs_client.set_text(self.source_name, self.accumulated_text)
                                
                # Reset TTL timer - new subtitle arrived, so extend the timer
                self._reset_ttl_timer()
            except Exception as e:
                self.logger.debug(f"OBS error: {e}")

