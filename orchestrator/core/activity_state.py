"""
Centralized Activity State Manager

Single source of truth for system activity state across all orchestrator components.
"""
import asyncio
from typing import Optional
from core.event_bus import EventBus, Event
from orchestrator.events import EventType
from orchestrator.core.models import SystemState
from orchestrator.core.constants import UI_ACTIVITY, UI_LISTENING_STATE_CHANGED


class ActivityState:
    """Centralized system state manager - single source of truth."""

    def __init__(self, event_bus: EventBus):
        """
        Initialize the activity state manager.

        Args:
            event_bus: Event bus for publishing state change events
        """
        self.event_bus = event_bus
        self.state = SystemState()  # Single instance
        self._lock = asyncio.Lock()

    async def update(self, changes: dict):
        """
        Update state fields and publish STATE_CHANGED event.

        Args:
            changes: Dictionary of field names and new values
        """
        async with self._lock:
            self.state.update(changes)
            # Publish internal system event
            await self.event_bus.publish(
                Event(EventType.STATE_CHANGED.value, {"state": changes})
            )
            # Publish UI compatibility event
            await self.event_bus.publish(
                Event(UI_ACTIVITY, {"state": changes})
            )

    async def set_listening(self, enabled: bool):
        """
        Update listening state with dedicated event.

        Args:
            enabled: Whether listening is enabled
        """
        async with self._lock:
            self.state.listening = enabled
            # Publish internal system event
            await self.event_bus.publish(
                Event(EventType.LISTENING_STATE_CHANGED.value, {"enabled": enabled})
            )
            # Publish UI compatibility event
            await self.event_bus.publish(
                Event(UI_LISTENING_STATE_CHANGED, {"enabled": enabled})
            )

    def is_busy(self) -> bool:
        """
        Check if agent is busy (responding/synthesizing/playing).

        Returns:
            True if any activity is in progress
        """
        return (
            self.state.responding or
            self.state.synthesizing or
            self.state.playing
        )


# Global instance (similar to get_settings pattern)
_activity_state: Optional[ActivityState] = None


def get_activity_state() -> ActivityState:
    """
    Get the global activity state instance.

    Returns:
        Global ActivityState instance

    Raises:
        RuntimeError: If ActivityState not initialized
    """
    if _activity_state is None:
        raise RuntimeError("ActivityState not initialized. Call init_activity_state() first.")
    return _activity_state


def init_activity_state(event_bus: EventBus) -> ActivityState:
    """
    Initialize the global activity state manager.

    Args:
        event_bus: Event bus for publishing state change events

    Returns:
        Initialized ActivityState instance
    """
    global _activity_state
    _activity_state = ActivityState(event_bus)
    return _activity_state
