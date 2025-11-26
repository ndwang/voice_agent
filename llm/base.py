"""
LLM Provider Abstraction

Abstract base class for LLM providers with streaming support.
"""
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, Tuple


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a complete response (non-streaming).
        
        Args:
            prompt: User input text
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """
        Generate a streaming response (token by token).
        
        Args:
            prompt: User input text
            **kwargs: Additional provider-specific parameters
            
        Yields:
            Tokens as they are generated
        """
        pass
    
    def parse_error(self, exception: Exception) -> Tuple[int, str]:
        """
        Parse provider-specific errors and return appropriate HTTP status code and message.
        
        Args:
            exception: The exception raised by the provider
            
        Returns:
            Tuple of (status_code, error_message)
        """
        # Default implementation - can be overridden by providers
        return 500, f"Internal server error: {str(exception)}"

