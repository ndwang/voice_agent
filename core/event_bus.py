"""
Event Bus

Simple internal event bus for pub/sub communication between components.
"""
import asyncio
import threading
from typing import Dict, List, Callable, Any, Awaitable
from dataclasses import dataclass, field
from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class Event:
    """Base class for all events."""
    name: str
    data: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self):
        return f"Event(name='{self.name}', data={self.data})"


class EventBus:
    """
    Asynchronous event bus.
    
    Allows components to subscribe to specific event types and publish events
    to trigger callbacks.
    """
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Event], Awaitable[None]]]] = {}
        self._lock = threading.Lock()
    
    def subscribe(self, event_name: str, callback: Callable[[Event], Awaitable[None]]):
        """
        Subscribe to an event.
        
        Args:
            event_name: Name of the event to subscribe to.
            callback: Async function to call when event is published.
        """
        with self._lock:
            if event_name not in self._subscribers:
                self._subscribers[event_name] = []
            self._subscribers[event_name].append(callback)
            logger.debug(f"Subscribed to event '{event_name}'")
            
    def unsubscribe(self, event_name: str, callback: Callable[[Event], Awaitable[None]]):
        """
        Unsubscribe from an event.
        
        Args:
            event_name: Name of the event.
            callback: The callback function to remove.
        """
        with self._lock:
            if event_name in self._subscribers:
                if callback in self._subscribers[event_name]:
                    self._subscribers[event_name].remove(callback)
                    logger.debug(f"Unsubscribed from event '{event_name}'")
    
    async def publish(self, event: Event):
        """
        Publish an event to all subscribers.
        
        Args:
            event: The event object to publish.
        """
        with self._lock:
            callbacks = self._subscribers.get(event.name, [])[:]
            
        if not callbacks:
            return
            
        # Run callbacks concurrently
        tasks = []
        for callback in callbacks:
            tasks.append(asyncio.create_task(self._safe_callback(callback, event)))
            
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
    async def _safe_callback(self, callback: Callable[[Event], Awaitable[None]], event: Event):
        """Run callback safely, catching exceptions."""
        try:
            await callback(event)
        except Exception as e:
            logger.error(f"Error in event handler for '{event.name}': {e}", exc_info=True)

