import asyncio
from typing import Optional, Protocol, Callable, Awaitable
from core.event_bus import EventBus, Event
from core.logging import get_logger
from orchestrator.events import EventType

logger = get_logger(__name__)

class InputSource(Protocol):
    """Protocol that all input sources must implement."""
    async def start(self): ...
    async def stop(self): ...

class BaseSource:
    """Base class for input sources."""
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.running = False

    async def publish(self, event_type: EventType, data: dict):
        """Publish an event to the bus."""
        event = Event(name=event_type.value, data=data)
        await self.event_bus.publish(event)


