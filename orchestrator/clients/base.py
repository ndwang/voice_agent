"""
Base Client

Base class for external service clients.
"""
from abc import ABC, abstractmethod
from core.logging import get_logger


class BaseClient(ABC):
    """Base class for external service clients."""
    
    def __init__(self):
        """Initialize base client."""
        self.logger = get_logger(self.__class__.__name__)
    
    @abstractmethod
    async def connect(self):
        """Establish connection to external service."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Close connection."""
        pass

