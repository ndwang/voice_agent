"""
Google Gemini API Provider

Implementation of LLMProvider for Google Gemini API.
"""
from typing import AsyncIterator, Optional, List, Dict, Tuple
from google import genai

from llm.base import LLMProvider


class GeminiProvider(LLMProvider):
    """Google Gemini API provider implementation."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize Gemini provider.
        
        Args:
            model: Model name (e.g., "gemini-2.5-flash", "gemini-2.5-pro")
            api_key: API key for Gemini. If None, uses GEMINI_API_KEY from environment.
        """
        if model is None:
            self.model = "gemini-2.5-flash"
        else:
            self.model = model

        # If API key is provided, use it; otherwise Client will use environment variable
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client()
    
    def parse_error(self, exception: Exception) -> Tuple[int, str]:
        """
        Parse Gemini API errors and return appropriate HTTP status code and message.
        
        Expected exception structure: {'error': {'code': 503, 'message': '...', 'status': 'UNAVAILABLE'}}
        
        Error codes reference: https://ai.google.dev/gemini-api/docs/troubleshooting
        
        Args:
            exception: The exception raised by Gemini API
            
        Returns:
            Tuple of (status_code, error_message)
        """
        # Gemini API exceptions always have error dict structure
        if not hasattr(exception, 'error') or not isinstance(exception.error, dict):
            # Unexpected exception format - treat as internal server error
            return 500, f"Unexpected error format: {str(exception)}"
        
        error_dict = exception.error
        status_code = error_dict.get('code')
        error_message = error_dict.get('message')
        
        return status_code, error_message
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate a complete response using Gemini API.
        
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
        # Create a new stateless chat for this request
        chat = self.client.aio.chats.create(model=self.model)
        # Send all messages in sequence to build context
        # The last message should be the user message
        for i, msg in enumerate(messages):
            if msg["role"] == "system":
                # System messages need to be set on the chat, not sent
                # For Gemini, we'll skip it and let it be handled by the model
                continue
            elif msg["role"] == "user":
                if i == len(messages) - 1:
                    # Last message - this is the actual request
                    response = await chat.send_message(msg["content"])
                    return response.text
                else:
                    # Previous user messages - send to build context
                    await chat.send_message(msg["content"])
            elif msg["role"] == "assistant":
                # Assistant messages are responses, skip them as they're already in context
                continue
        
        # Fallback if no user message found (should not happen)
        raise ValueError("No user message found in messages")
    
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response using Gemini API.
        
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
        # Create a new stateless chat for this request
        chat = self.client.aio.chats.create(model=self.model)
        # Send all messages in sequence to build context
        # The last message should be the user message
        for i, msg in enumerate(messages):
            if msg["role"] == "system":
                # System messages need to be set on the chat, not sent
                # For Gemini, we'll skip it and let it be handled by the model
                continue
            elif msg["role"] == "user":
                if i == len(messages) - 1:
                    # Last message - this is the actual request, stream it
                    async for chunk in await chat.send_message_stream(msg["content"]):
                        yield chunk.text
                    return
                else:
                    # Previous user messages - send to build context
                    await chat.send_message(msg["content"])
            elif msg["role"] == "assistant":
                # Assistant messages are responses, skip them as they're already in context
                continue
        
        # Fallback if no user message found (should not happen)
        raise ValueError("No user message found in messages")

    def get_history(self) -> List[Dict[str, str]]:
        """
        Get the history of the conversation.
        
        Note: This provider is stateless. History is managed by the caller (ContextManager).
        This method returns an empty list.
        
        Returns:
            Empty list (history is managed externally)
        """
        return []
    
    def clear_history(self) -> None:
        """
        Clear the conversation history.
        
        Note: This provider is stateless. History is managed by the caller (ContextManager).
        This method is a no-op.
        """
        pass

