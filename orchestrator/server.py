import uvicorn
import asyncio
import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.server import create_app
from core.config import get_config
from core.logging import get_logger
from core.event_bus import EventBus, Event
from orchestrator.api import router as orchestrator_router
from orchestrator.events import EventType
from orchestrator.core.constants import UI_LISTENING_STATE_CHANGED, UI_ACTIVITY
from orchestrator.utils.event_helpers import publish_activity, publish_listening_state_changed

# Managers
from orchestrator.managers.interaction_manager import InteractionManager
from orchestrator.managers.subtitle_manager import SubtitleManager
from orchestrator.managers.latency_manager import LatencyTracker
from orchestrator.managers.tts_manager import TTSManager
from orchestrator.sources.stt_source import STTSource
from orchestrator.sources.bilibili_source import BilibiliSource
from orchestrator.tools.registry import ToolRegistry
from orchestrator.hotkey_manager import HotkeyManager
from orchestrator.ocr_client import OCRClient
from audio.audio_driver import AudioDriver

logger = get_logger(__name__)

class OrchestratorServer:
    def __init__(self):
        self.event_bus = EventBus()
        self.tool_registry = ToolRegistry()
        self.ocr_client = OCRClient()
        
        # Sources
        self.stt_source = STTSource(self.event_bus)
        self.bilibili_source = BilibiliSource(self.event_bus)
        
        # Audio driver (captures microphone and streams to STT)
        self.audio_driver = AudioDriver(event_bus=self.event_bus)
        
        # Managers
        self.interaction_manager = InteractionManager(self.event_bus)
        self.subtitle_manager = SubtitleManager(self.event_bus)
        self.latency_tracker = LatencyTracker(self.event_bus)
        self.tts_manager = TTSManager(self.event_bus)
        
        # Hotkeys
        self.hotkey_manager = HotkeyManager()
        
    async def start(self):
        # Start sources
        await self.stt_source.start()
        if get_config("bilibili", "enabled", default=False):
            await self.bilibili_source.start()
        
        # Start audio driver
        await self.audio_driver.start()
        
        # Setup hotkeys
        toggle_key = get_config("orchestrator", "hotkeys", "toggle_listening", default="ctrl+shift+l")
        cancel_key = get_config("orchestrator", "hotkeys", "cancel_speech", default="ctrl+shift+c")
        
        # Set up toggle_listening callback with event loop
        event_loop = asyncio.get_event_loop()
        def toggle_cb():
            asyncio.run_coroutine_threadsafe(self.toggle_listening(), event_loop)
        
        # Set up cancel_speech callback with event loop
        def cancel_cb():
            asyncio.run_coroutine_threadsafe(self.cancel_interaction(), event_loop)
            
        self.hotkey_manager.register_hotkey(
            "toggle_listening", 
            toggle_key, 
            toggle_cb
        )
        self.hotkey_manager.register_hotkey(
            "cancel_speech", 
            cancel_key, 
            cancel_cb
        )
        self.hotkey_manager.start(event_loop)
        logger.info("Orchestrator Logic Started")

    async def stop(self):
        await self.audio_driver.stop()
        await self.stt_source.stop()
        await self.bilibili_source.stop()
        self.hotkey_manager.stop()
        logger.info("Orchestrator Logic Stopped")

    async def toggle_listening(self) -> bool:
        """Toggle listening state and return new state."""
        new_state = not self.interaction_manager.activity_state.listening
        logger.info(f"Listening {'enabled' if new_state else 'disabled'}")
        
        # Publish listening state change event (InteractionManager will update local state)
        await publish_listening_state_changed(self.event_bus, new_state)
        
        # Also publish activity update
        await publish_activity(self.event_bus, {"listening": new_state})
        
        return new_state

    async def set_listening(self, enabled: bool):
        """Set listening state explicitly."""
        logger.info(f"Listening set to: {enabled}")
        
        # Publish listening state change event
        await publish_listening_state_changed(self.event_bus, enabled)
        
        # Also publish activity update
        await publish_activity(self.event_bus, {"listening": enabled})

    async def cancel_interaction(self):
        # Fire cancel event
        await self.event_bus.publish(Event(EventType.LLM_CANCELLED.value))

def main():
    HOST = get_config("orchestrator", "host", default="0.0.0.0")
    PORT = get_config("orchestrator", "port", default=8000)
    
    # 1. Initialize Logic
    orch = OrchestratorServer()
    
    # 2. Inject into API router (Global variable hack for simplicity in refactor)
    import orchestrator.api.ui
    import orchestrator.api.hotkeys
    orchestrator.api.ui.orchestrator = orch
    orchestrator.api.hotkeys.orchestrator = orch

    # 3. Create Server
    app = create_app(
        title="Voice Agent Orchestrator",
        description="Event-driven Voice Agent",
        version="2.0.0",
        lifespan=lifespan_context(orch)
    )
    
    app.include_router(orchestrator_router)
    
    # Mount static files for UI
    from fastapi.staticfiles import StaticFiles
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/ui/assets", StaticFiles(directory=static_dir), name="static")
    
    logger.info(f"Starting Orchestrator on {HOST}:{PORT}...")
    uvicorn.run(app, host=HOST, port=PORT)

def lifespan_context(orch):
    from contextlib import asynccontextmanager
    @asynccontextmanager
    async def lifespan(app):
        await orch.start()
        yield
        await orch.stop()
    return lifespan

if __name__ == "__main__":
    main()
