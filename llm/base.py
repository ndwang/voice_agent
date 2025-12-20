"""
LLM Provider Abstraction

Abstract base class for LLM providers with streaming support.
"""
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, Tuple, List, Dict


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    Providers are stateless regarding conversation history. All history management
    is handled by the caller (typically ContextManager). Providers accept fully
    prepared messages and return responses without maintaining state.
    """
    
    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate a complete response (non-streaming).
        
        Args:
            messages: List of message dicts with "role" and "content" keys.
                     Must include system message (if any) and conversation history.
                     The last message should be the current user message.
            system_prompt: Optional system prompt. If messages already contains a system
                          message, this may be ignored.
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response (token by token).
        
        Args:
            messages: List of message dicts with "role" and "content" keys.
                     Must include system message (if any) and conversation history.
                     The last message should be the current user message.
            system_prompt: Optional system prompt. If messages already contains a system
                          message, this may be ignored.
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

