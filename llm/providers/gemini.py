"""
Google Gemini API Provider

Implementation of LLMProvider for Google Gemini API.
"""
from typing import AsyncIterator, Optional, List, Dict, Tuple
from google import genai
from google.genai import types

from llm.base import LLMProvider

# Access types through genai module



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
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
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
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated text response
        """
        # Extract system prompt from messages or use provided parameter
        final_system_prompt = system_prompt
        filtered_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                # Extract system prompt from messages if not provided
                if final_system_prompt is None:
                    final_system_prompt = msg["content"]
            else:
                # Keep non-system messages (user and assistant)
                filtered_messages.append(msg)
        
        # Convert messages to Gemini format (list of Content objects)
        contents = []
        for msg in filtered_messages:
            role = "user" if msg["role"] == "user" else "model"  # Gemini uses "model" not "assistant"
            contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
        
        # Prepare generation config
        config_kwargs = {}
        if final_system_prompt:
            config_kwargs["system_instruction"] = final_system_prompt
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        if top_p is not None:
            config_kwargs["top_p"] = top_p
        if top_k is not None:
            config_kwargs["top_k"] = top_k
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens
        
        # Merge any additional config from kwargs
        if "generation_config" in kwargs:
            config_kwargs.update(kwargs["generation_config"])
        
        # Use generate_content API with proper structure
        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None
        
        response = await self.client.aio.models.generate_content(
            model=self.model,
            config=config,
            contents=contents
        )
        return response.text
    
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
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
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Yields:
            Tokens as they are generated
        """
        # Extract system prompt from messages or use provided parameter
        final_system_prompt = system_prompt
        filtered_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                # Extract system prompt from messages if not provided
                if final_system_prompt is None:
                    final_system_prompt = msg["content"]
            else:
                # Keep non-system messages (user and assistant)
                filtered_messages.append(msg)
        
        # Convert messages to Gemini format (list of Content objects)
        contents = []
        for msg in filtered_messages:
            role = "user" if msg["role"] == "user" else "model"  # Gemini uses "model" not "assistant"
            contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
        
        # Prepare generation config
        config_kwargs = {}
        if final_system_prompt:
            config_kwargs["system_instruction"] = final_system_prompt
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        if top_p is not None:
            config_kwargs["top_p"] = top_p
        if top_k is not None:
            config_kwargs["top_k"] = top_k
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens
        
        # Merge any additional config from kwargs
        if "generation_config" in kwargs:
            config_kwargs.update(kwargs["generation_config"])
        
        # Use generate_content_stream API with proper structure
        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None
        
        stream = await self.client.aio.models.generate_content_stream(
            model=self.model,
            config=config,
            contents=contents
        )
        async for chunk in stream:
            if hasattr(chunk, 'text') and chunk.text:
                yield chunk.text

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

