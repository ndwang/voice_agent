"""
Google Gemini API Provider

Implementation of LLMProvider for Google Gemini API.
"""
from typing import AsyncIterator, Optional, List, Dict, Tuple, Union, Any
import logging
from google import genai
from google.genai import types

from llm.base import LLMProvider

# Access types through genai module

logger = logging.getLogger(__name__)



class GeminiProvider(LLMProvider):
    """Google Gemini API provider implementation."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        generation_config: Optional[Dict] = None
    ):
        """
        Initialize Gemini provider.
        
        Args:
            model: Model name (e.g., "gemini-2.5-flash", "gemini-2.5-pro")
            api_key: API key for Gemini. If None, uses GEMINI_API_KEY from environment.
            generation_config: Dictionary of default generation parameters (temperature, top_p, top_k, max_output_tokens, etc.)
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
        
        # Store default generation config
        self.default_generation_config = generation_config or {}

        super().__init__()
    
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

    def _create_parts_from_message(self, msg: Dict[str, Any]) -> List[types.Part]:
        """
        Convert message to list of Gemini Part objects.

        Args:
            msg: Message dict with "content" and optional "images"

        Returns:
            List of Part objects (text and/or images)
        """
        parts = []

        # Add text content
        if msg.get("content"):
            parts.append(types.Part(text=msg["content"]))

        # Add images if present
        if "images" in msg and msg["images"]:
            from llm.utils.image_utils import read_image_file, get_mime_type, validate_image_path

            for image_path in msg["images"]:
                try:
                    if not validate_image_path(image_path):
                        logger.warning(f"Skipping invalid image: {image_path}")
                        continue

                    # Read image bytes
                    image_bytes = read_image_file(image_path)
                    mime_type = get_mime_type(image_path)

                    # Create Part from bytes
                    parts.append(types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=mime_type
                    ))
                    logger.debug(f"Added image to message: {image_path} ({mime_type})")

                except Exception as e:
                    logger.error(f"Failed to load image {image_path}: {e}")
                    continue

        return parts

    async def generate(
        self,
        messages: List[Dict[str, Any]],
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
                     User messages may optionally include "images" field with List[str] of file paths.
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

            # Create parts from message (handles text + images)
            parts = self._create_parts_from_message(msg)

            contents.append(types.Content(role=role, parts=parts))
        
        # Prepare generation config - start with defaults from config
        config_kwargs = self.default_generation_config.copy()
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
        
        # Merge any additional config from kwargs (overrides defaults and explicit params)
        if "generation_config" in kwargs:
            config_kwargs.update(kwargs["generation_config"])
        
        # Extract thinking config if present (thinking_level takes priority over thinking_budget)
        thinking_config = None
        if "thinking_level" in config_kwargs:
            thinking_level = config_kwargs.pop("thinking_level")
            thinking_config = types.ThinkingConfig(thinking_level=thinking_level)
            # Remove thinking_budget if present (thinking_level takes priority)
            config_kwargs.pop("thinking_budget", None)
        elif "thinking_budget" in config_kwargs:
            thinking_budget = config_kwargs.pop("thinking_budget")
            thinking_config = types.ThinkingConfig(thinking_budget=thinking_budget)
        
        # Use generate_content API with proper structure
        if thinking_config:
            config_kwargs["thinking_config"] = thinking_config
        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None
        
        response = await self.client.aio.models.generate_content(
            model=self.model,
            config=config,
            contents=contents
        )

        self.last_prompt_tokens = response.usage_metadata.prompt_token_count
        self.last_completion_tokens = response.usage_metadata.candidates_token_count

        return response.text
    
    async def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """
        Generate a streaming response using Gemini API.

        Args:
            messages: List of message dicts with "role" and "content" keys.
                     User messages may optionally include "images" field with List[str] of file paths.
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

            # Create parts from message (handles text + images)
            parts = self._create_parts_from_message(msg)

            contents.append(types.Content(role=role, parts=parts))
        
        # Prepare generation config - start with defaults from config
        config_kwargs = self.default_generation_config.copy()
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
        
        # Merge any additional config from kwargs (overrides defaults and explicit params)
        if "generation_config" in kwargs:
            config_kwargs.update(kwargs["generation_config"])
        
        # Convert tools to Gemini format if provided
        if tools and len(tools) > 0:
            function_declarations = []
            for tool in tools:
                func = tool["function"]
                function_declarations.append(
                    types.FunctionDeclaration(
                        name=func["name"],
                        description=func["description"],
                        parameters=func["parameters"]
                    )
                )
            config_kwargs["tools"] = [types.Tool(function_declarations=function_declarations)]

        # Extract thinking config if present (thinking_level takes priority over thinking_budget)
        thinking_config = None
        if "thinking_level" in config_kwargs:
            thinking_level = config_kwargs.pop("thinking_level")
            thinking_config = types.ThinkingConfig(thinking_level=thinking_level)
            # Remove thinking_budget if present (thinking_level takes priority)
            config_kwargs.pop("thinking_budget", None)
        elif "thinking_budget" in config_kwargs:
            thinking_budget = config_kwargs.pop("thinking_budget")
            thinking_config = types.ThinkingConfig(thinking_budget=thinking_budget)

        # Use generate_content_stream API with proper structure
        if thinking_config:
            config_kwargs["thinking_config"] = thinking_config
        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None
        
        stream = await self.client.aio.models.generate_content_stream(
            model=self.model,
            config=config,
            contents=contents
        )
        async for chunk in stream:
            # Check for function calls in the chunk
            if hasattr(chunk, 'candidates') and chunk.candidates:
                parts = chunk.candidates[0].content.parts
                for part in parts:
                    if hasattr(part, 'function_call'):
                        # Yield tool call dict
                        func_call = part.function_call
                        yield {
                            "type": "tool_call",
                            "id": f"call_{hash(func_call.name)}_{id(func_call)}",
                            "name": func_call.name,
                            "arguments": dict(func_call.args) if func_call.args else {}
                        }
                    elif hasattr(part, 'text') and part.text:
                        yield part.text
            elif hasattr(chunk, 'text') and chunk.text:
                # Fallback for text-only chunks
                yield chunk.text

            if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                self.last_prompt_tokens = chunk.usage_metadata.prompt_token_count
                self.last_completion_tokens = chunk.usage_metadata.candidates_token_count