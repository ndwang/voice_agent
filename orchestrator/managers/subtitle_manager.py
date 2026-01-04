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
        self.new_round = False  # Track if user spoke since last subtitle
        self.character_on_screen = False  # Track if character is visible
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
        self.event_bus.subscribe(EventType.INPUT_RECEIVED.value, self.on_input_received)
        self.event_bus.subscribe(EventType.SUBTITLE_REQUEST.value, self.on_subtitle_request)
        self.event_bus.subscribe(EventType.TURN_ENDED.value, self.on_turn_ended)

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
            # OBS client handles connection automatically
            await self.obs_client.set_filter_visibility(
                self.visibility_source,
                filter_name,
                True
            )
            self.logger.debug(f"Enabled filter '{filter_name}' on source '{self.visibility_source}'")
        except Exception as e:
            self.logger.debug(f"OBS error while enabling filter: {e}")

    async def _clear_subtitles(self, hide_character=False):
        """Clear subtitles text and optionally hide character.

        Args:
            hide_character: If True, triggers clear_filter to remove character from screen
        """
        self.logger.info(f"Clearing subtitles, hide_character: {hide_character}")
        self.accumulated_text = ""
        if self.obs_client:
            try:
                # OBS client handles connection automatically
                await self.obs_client.set_text(self.source_name, "")
                # Only enable clear filter (hide character) when requested (e.g., on TTL expiry)
                if hide_character:
                    await self._enable_filter(self.clear_filter_name)
                    self.character_on_screen = False
            except Exception as e:
                self.logger.debug(f"OBS error while clearing: {e}")

    async def _clear_after_ttl(self):
        """Clear subtitles and hide character after TTL expires."""
        try:
            await asyncio.sleep(self.ttl_seconds)
            # Only clear if we still have accumulated text (no new subtitles arrived)
            if self.accumulated_text:
                await self._clear_subtitles(hide_character=True)
                self.logger.debug(f"Cleared subtitles and hid character after {self.ttl_seconds}s TTL")
        except asyncio.CancelledError:
            # Timer was cancelled (new subtitle arrived or speech started)
            pass

    async def on_speech_start(self, event: Event):
        """Cancel TTL timer when user starts speaking (voice input)"""
        # Cancel any pending TTL timer
        if self.clear_task and not self.clear_task.done():
            self.clear_task.cancel()

        # Mark that user spoke - next subtitle will be a new round
        self.new_round = True

    async def on_input_received(self, event: Event):
        """Cancel TTL timer when input received from any source (chat, voice, etc)"""
        # Cancel any pending TTL timer
        if self.clear_task and not self.clear_task.done():
            self.clear_task.cancel()

        # Mark new conversation round - next subtitle will clear old text
        self.new_round = True

    async def on_subtitle_request(self, event: Event):
        """Update subtitles when Chinese content is available.
        Clears old subtitles when new round begins.
        Shows character when subtitles first appear.
        Keeps character visible during conversation rounds."""
        text = event.data.get("text", "")
        if self.obs_client and text:
            try:
                # OBS client handles connection automatically
                self.logger.info(f"Subtitle request: {text[:10]}..., new round: {self.new_round}, character on screen: {self.character_on_screen}, accumulated text: {self.accumulated_text[:10]}...")
                # If new round, clear old subtitles (but keep character)
                if self.new_round and self.accumulated_text:
                    await self._clear_subtitles(hide_character=False)

                # Reset the new round flag
                self.new_round = False

                # Show character when subtitles first appear (character not on screen)
                if not self.character_on_screen:
                    await self._enable_filter(self.appear_filter_name)
                    self.character_on_screen = True

                # Accumulate the new sentence
                self.accumulated_text += text

                # Update OBS with the accumulated text
                await self.obs_client.set_text(self.source_name, self.accumulated_text)
            except Exception as e:
                self.logger.debug(f"OBS error: {e}")

    async def on_turn_ended(self, event: Event):
        """Start TTL countdown when turn ends."""
        # Start TTL timer - subtitles will be cleared after TTL expires
        self._reset_ttl_timer()

