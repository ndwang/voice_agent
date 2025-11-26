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
        # chat handles multi-turn conversations
        self.chat = self.client.aio.chats.create(model=self.model)
    
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
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a complete response using Gemini API.
        
        Args:
            prompt: User input text
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated text response
        """
        response = await self.chat.send_message(prompt)
        return response.text
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """
        Generate a streaming response using Gemini API.
        
        Args:
            prompt: User input text
            **kwargs: Additional provider-specific parameters
            
        Yields:
            Tokens as they are generated
        """
        async for chunk in await self.chat.send_message_stream(prompt):
            yield chunk.text

    def get_history(self) -> List[Dict[str, str]]:
        """
        Get the history of the conversation.
        
        Returns:
            List of message dictionaries with "role" and "content" keys
        """
        history = self.chat.get_history()
        messages = []
        for content in history:
            text = "".join(part.text for part in content.parts)
            messages.append({"role": content.role, "content": text})
        return messages

