"""Helper functions for common event publishing patterns."""
from core.event_bus import EventBus, Event
from orchestrator.events import EventType
from orchestrator.core.constants import UI_ACTIVITY, UI_HISTORY_UPDATED, UI_LISTENING_STATE_CHANGED


async def publish_activity(event_bus: EventBus, state: dict):
    """
    Publish system state change and UI activity update.
    
    Args:
        event_bus: Event bus instance
        state: Dictionary with activity state (e.g., {"transcribing": True, "responding": False})
    """
    # 1. Publish internal system event
    await event_bus.publish(Event(EventType.STATE_CHANGED.value, {"state": state}))
    
    # 2. Maintain UI compatibility
    await event_bus.publish(Event(UI_ACTIVITY, {"state": state}))


async def publish_ui_update(event_bus: EventBus, event_name: str, data: dict = None):
    """
    Publish UI update event.
    
    Args:
        event_bus: Event bus instance
        event_name: Event name (without "ui." prefix)
        data: Optional event data dictionary
    """
    if data is None:
        data = {}
    
    full_event_name = f"ui.{event_name}"
    await event_bus.publish(Event(full_event_name, data))


async def publish_history_updated(event_bus: EventBus):
    """Publish history updated event."""
    await event_bus.publish(Event(EventType.HISTORY_UPDATED.value))
    await event_bus.publish(Event(UI_HISTORY_UPDATED))


async def publish_listening_state_changed(event_bus: EventBus, enabled: bool):
    """
    Publish listening state changed event.
    
    Args:
        event_bus: Event bus instance
        enabled: Whether listening is enabled
    """
    await event_bus.publish(Event(EventType.LISTENING_STATE_CHANGED.value, {"enabled": enabled}))
    await event_bus.publish(Event(UI_LISTENING_STATE_CHANGED, {"enabled": enabled}))

