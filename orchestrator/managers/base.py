"""
Base Manager

Base class for all event-driven managers in the orchestrator.
"""
from abc import ABC, abstractmethod
from core.event_bus import EventBus
from core.logging import get_logger


class BaseManager(ABC):
    """Base class for all event-driven managers."""
    
    def __init__(self, event_bus: EventBus):
        """
        Initialize base manager.
        
        Args:
            event_bus: Event bus instance for pub/sub communication
        """
        self.event_bus = event_bus
        self.logger = get_logger(self.__class__.__name__)
        self._register_handlers()
    
    @abstractmethod
    def _register_handlers(self):
        """
        Register event handlers. Called during initialization.
        
        Subclasses should implement this to subscribe to relevant events.
        """
        pass

