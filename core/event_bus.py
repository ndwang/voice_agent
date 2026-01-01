"""
Event Bus

Simple internal event bus for pub/sub communication between components.
"""
import asyncio
import threading
from typing import Dict, List, Callable, Any, Awaitable, Optional
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

    Events are processed sequentially - each event's callbacks must complete
    before the next event begins processing. Callbacks within a single event
    run concurrently.
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Event], Awaitable[None]]]] = {}
        self._lock = threading.Lock()
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._processor_task: Optional[asyncio.Task] = None
        self._running = False
    
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
    
    async def start(self):
        """Start the event processor loop."""
        if self._running:
            return

        self._running = True
        self._processor_task = asyncio.create_task(self._process_events())
        logger.debug("Event bus processor started")

    async def stop(self):
        """Stop the event processor loop."""
        if not self._running:
            return

        self._running = False

        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        logger.debug("Event bus processor stopped")

    async def publish(self, event: Event):
        """
        Publish an event to all subscribers.

        Events are queued and processed sequentially. This method returns
        immediately after queueing the event.

        Args:
            event: The event object to publish.
        """
        await self._event_queue.put(event)

    async def _process_events(self):
        """
        Process events from the queue sequentially.

        Each event's callbacks are awaited to completion before the next
        event is processed. Callbacks within a single event run concurrently.
        """
        logger.debug("Event processor loop started")

        while self._running:
            try:
                # Wait for next event
                event = await self._event_queue.get()

                # Get callbacks for this event
                with self._lock:
                    callbacks = self._subscribers.get(event.name, [])[:]

                if not callbacks:
                    self._event_queue.task_done()
                    continue

                # Run all callbacks for this event concurrently
                tasks = []
                for callback in callbacks:
                    tasks.append(asyncio.create_task(self._safe_callback(callback, event)))

                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                self._event_queue.task_done()

            except asyncio.CancelledError:
                logger.debug("Event processor cancelled")
                break
            except Exception as e:
                logger.error(f"Error in event processor: {e}", exc_info=True)
            
    async def _safe_callback(self, callback: Callable[[Event], Awaitable[None]], event: Event):
        """Run callback safely, catching exceptions."""
        try:
            await callback(event)
        except Exception as e:
            logger.error(f"Error in event handler for '{event.name}': {e}", exc_info=True)

