import asyncio
import itertools
from dataclasses import dataclass
from typing import Callable, Optional, Any, Dict
from core.event_bus import EventBus, Event
from core.logging import get_logger

logger = get_logger(__name__)

@dataclass
class InputItem:
    priority: int  # Lower = higher priority
    source_type: str  # "voice", "bilibili_single", "bilibili_batch", "ocr", "game"
    get_data: Callable | dict  # Callable for fresh data, or static data
    metadata: Optional[Dict[str, Any]] = None

    def __lt__(self, other):
        return self.priority < other.priority

    def fetch_data(self):
        """Get the data - either call function or return static data."""
        if callable(self.get_data):
            return self.get_data()
        return self.get_data

class AsyncPriorityQueue:
    def __init__(self):
        self.queue = asyncio.PriorityQueue()
        self.counter = itertools.count()

    async def put(self, item, priority=0):
        await self.queue.put((priority, next(self.counter), item))

    def put_nowait(self, item, priority=0):
        self.queue.put_nowait((priority, next(self.counter), item))

    async def get(self):
        priority, counter, item = await self.queue.get()
        return item

    def get_nowait(self):
        priority, counter, item = self.queue.get_nowait()
        return item

    def task_done(self):
        """Indicate that a formerly enqueued task is complete"""
        self.queue.task_done()

    async def join(self):
        """Block until all items have been gotten and processed"""
        await self.queue.join()

    def qsize(self):
        return self.queue.qsize()

    def empty(self):
        return self.queue.empty()

    def __len__(self):
        return self.qsize()

    def clear(self, reset_counter=False):
        while not self.queue.empty():
            self.queue.get_nowait()
        if reset_counter:
            self.counter = itertools.count()

    def peek_priority(self):
        """
        Peek at priority of next item without removing it.
        Returns None if queue is empty.
        """
        if self.queue.empty():
            return None

        # Python's PriorityQueue uses heapq - peek at heap[0]
        priority, counter, item = self.queue._queue[0]
        return priority
