"""
LLM Provider Abstraction

Abstract base class for LLM providers with streaming support.
"""
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, Tuple, List, Dict, Union, Any


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    Providers are stateless regarding conversation history. All history management
    is handled by the caller (typically ContextManager). Providers accept fully
    prepared messages and return responses without maintaining state.
    """

    def __init__(self):
        """Initialize provider with token tracking."""
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
    
    @property
    def last_token_count(self) -> Dict[str, int]:
        """
        Get token counts from the last generation.
        
        Returns:
            Dictionary with prompt_tokens, completion_tokens, and total_tokens
        """
        return {
            "prompt_tokens": self.last_prompt_tokens,
            "completion_tokens": self.last_completion_tokens,
            "total_tokens": self.last_prompt_tokens + self.last_completion_tokens
        }
        
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
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """
        Generate a streaming response (token by token).

        Args:
            messages: List of message dicts with "role" and "content" keys.
                     Must include system message (if any) and conversation history.
                     The last message should be the current user message.
            system_prompt: Optional system prompt. If messages already contains a system
                          message, this may be ignored.
            tools: Optional list of tool schemas in OpenAI format:
                  [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]
                  Providers convert to their native format internally.
            **kwargs: Additional provider-specific parameters

        Yields:
            Either text tokens (str) OR tool call dicts:
            {"type": "tool_call", "id": "...", "name": "...", "arguments": {...}}
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

