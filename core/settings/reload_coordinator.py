"""
Configuration reload coordinator.

Manages reload handlers and provides capability metadata for the UI.
"""
from typing import List, Callable, Dict, Tuple
from core.logging import get_logger
from .reload_result import ReloadResult

logger = get_logger(__name__)


class ReloadCoordinator:
    """Coordinates configuration reload across all managers"""

    def __init__(self):
        self.handlers: List[Tuple[str, Callable[[dict], ReloadResult]]] = []
        self._init_capabilities()

    def _init_capabilities(self):
        """Initialize hot-reload capabilities mapping"""
        self.capabilities = {
            # LLM - Hot-reloadable
            "llm.provider": {"hot_reload": True},
            "llm.providers": {"hot_reload": True},

            # Orchestrator - Hot-reloadable
            "orchestrator.system_prompt_file": {"hot_reload": True},
            "orchestrator.queue_cooldown_seconds": {"hot_reload": True},
            "orchestrator.log_level": {"hot_reload": True},

            # Hotkeys - Hot-reloadable
            "orchestrator.hotkeys": {"hot_reload": True},

            # TTS - Partially hot-reloadable
            "tts.providers.edge-tts": {"hot_reload": True},
            "tts.providers.gpt-sovits.references": {"hot_reload": True},
            "tts.providers.gpt-sovits.default_reference": {"hot_reload": True},
            "tts.providers.gpt-sovits.temperature": {"hot_reload": True},
            "tts.providers.gpt-sovits.top_p": {"hot_reload": True},
            "tts.providers.gpt-sovits.top_k": {"hot_reload": True},
            "tts.provider": {"hot_reload": False},  # Restart required
            "tts.providers.gpt-sovits.server_url": {"hot_reload": False},  # Restart required

            # Audio - Output device is hot-reloadable, input is not
            "audio.output.device": {"hot_reload": True},
            "audio.input.device": {"hot_reload": False},
            "audio.input.sample_rate": {"hot_reload": False},
            "audio.input.channels": {"hot_reload": False},
            "audio.block_size_ms": {"hot_reload": False},

            # OBS - Hot-reloadable
            "obs.subtitle_source": {"hot_reload": True},
            "obs.subtitle_ttl_seconds": {"hot_reload": True},
            "obs.visibility_source": {"hot_reload": True},
            "obs.appear_filter_name": {"hot_reload": True},
            "obs.clear_filter_name": {"hot_reload": True},

            # Bilibili - Hot-reloadable
            "bilibili.room_id": {"hot_reload": True},
            "bilibili.enabled": {"hot_reload": True},
            "bilibili.danmaku_ttl_seconds": {"hot_reload": True},

            # STT - Restart required
            "stt.provider": {"hot_reload": False},
            "stt.providers": {"hot_reload": False},
            "stt.sample_rate": {"hot_reload": False},

            # Services - Restart required
            "services": {"hot_reload": False},

            # Server bindings - Restart required
            "orchestrator.host": {"hot_reload": False},
            "orchestrator.port": {"hot_reload": False},
            "stt.host": {"hot_reload": False},
            "stt.port": {"hot_reload": False},
            "tts.host": {"hot_reload": False},
            "tts.port": {"hot_reload": False},
        }

    def register_handler(self, name: str, handler: Callable[[dict], ReloadResult]):
        """
        Register a reload handler.

        Args:
            name: Handler name (e.g., "InteractionManager")
            handler: Callable that takes changes dict and returns ReloadResult
        """
        self.handlers.append((name, handler))
        logger.debug(f"Registered reload handler: {name}")

    def reload_config(self, changes: dict) -> List[ReloadResult]:
        """
        Call all handlers and collect results.

        Args:
            changes: Dict with changed config sections

        Returns:
            List of ReloadResult from all handlers
        """
        results = []

        for handler_name, handler in self.handlers:
            try:
                logger.debug(f"Calling reload handler: {handler_name}")
                result = handler(changes)
                results.append(result)

                if not result.success:
                    logger.warning(f"Handler {handler_name} reported failure: {result.errors}")
                elif result.changes_applied:
                    logger.info(f"Handler {handler_name} applied changes: {result.changes_applied}")

                if result.restart_required:
                    logger.info(f"Handler {handler_name} requires restart: {result.restart_required}")

            except Exception as e:
                logger.error(f"Error in reload handler {handler_name}: {e}", exc_info=True)
                results.append(ReloadResult(
                    handler_name=handler_name,
                    success=False,
                    errors=[f"Exception: {str(e)}"]
                ))

        return results

    def get_capabilities(self) -> Dict[str, Dict[str, bool]]:
        """
        Return hot-reload capabilities for all config paths.

        Returns:
            Dict mapping config path to {"hot_reload": bool}
        """
        return self.capabilities.copy()
