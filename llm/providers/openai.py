"""
OpenAI API Provider

Implementation of LLMProvider for OpenAI API and OpenAI-compatible servers.
"""
import json
import os
from typing import AsyncIterator, Optional, List, Dict, Tuple, Union, Any

try:
    from openai import AsyncOpenAI
    from openai import (
        AuthenticationError,
        PermissionDeniedError,
        NotFoundError,
        RateLimitError,
        InternalServerError,
        APIConnectionError,
        APITimeoutError
    )
except ImportError:
    AsyncOpenAI = None
    AuthenticationError = None
    PermissionDeniedError = None
    NotFoundError = None
    RateLimitError = None
    InternalServerError = None
    APIConnectionError = None
    APITimeoutError = None

from llm.base import LLMProvider
from core.logging import get_logger

logger = get_logger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        generation_config: Optional[Dict] = None,
        **kwargs
    ):
        """
        Initialize OpenAI provider.

        Args:
            model: Model name (e.g., "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo")
            api_key: API key for OpenAI. If None, uses OPENAI_API_KEY environment variable
            base_url: Base URL for OpenAI-compatible servers. If None, uses official OpenAI API
            organization: Organization ID for multi-org accounts
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts for failed requests
            generation_config: Default generation parameters (temperature, top_p, max_tokens, etc.)
            **kwargs: Additional parameters
        """
        if AsyncOpenAI is None:
            raise ImportError(
                "openai is not installed. "
                "Install it with: pip install openai"
            )

        self.model = model
        self.default_generation_config = generation_config or {}

        # Initialize async client
        client_kwargs = {
            "timeout": timeout,
            "max_retries": max_retries
        }

        if api_key:
            client_kwargs["api_key"] = api_key
        elif os.getenv("OPENAI_API_KEY"):
            client_kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
        else:
            logger.warning("No API key provided for OpenAI - will fail unless base_url server doesn't require auth")

        if base_url:
            client_kwargs["base_url"] = base_url

        if organization:
            client_kwargs["organization"] = organization

        self.client = AsyncOpenAI(**client_kwargs)

        super().__init__()

        logger.info(f"Initialized OpenAI provider: model={model}, base_url={base_url or 'api.openai.com'}")

    def parse_error(self, exception: Exception) -> Tuple[int, str]:
        """
        Parse OpenAI API errors and return appropriate HTTP status code and message.

        Args:
            exception: The exception raised by OpenAI API

        Returns:
            Tuple of (status_code, error_message)
        """
        if AsyncOpenAI is None:
            return 500, str(exception)

        # Map OpenAI exceptions to HTTP status codes
        if isinstance(exception, AuthenticationError):
            return 401, f"Authentication failed: {str(exception)}"
        elif isinstance(exception, PermissionDeniedError):
            return 403, f"Permission denied: {str(exception)}"
        elif isinstance(exception, NotFoundError):
            return 404, f"Resource not found: {str(exception)}"
        elif isinstance(exception, RateLimitError):
            return 429, f"Rate limit exceeded: {str(exception)}"
        elif isinstance(exception, InternalServerError):
            return 500, f"OpenAI internal error: {str(exception)}"
        elif isinstance(exception, APIConnectionError):
            return 503, f"API connection failed: {str(exception)}"
        elif isinstance(exception, APITimeoutError):
            return 504, f"Request timeout: {str(exception)}"
        else:
            return 500, f"OpenAI error: {str(exception)}"

    def _convert_messages_to_openai_format(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Convert internal message format to OpenAI API format.

        Handles:
        - System prompt injection
        - Image conversion to base64 data URLs
        - Multi-modal content structuring

        Args:
            messages: Internal format messages
            system_prompt: Optional system prompt to prepend

        Returns:
            OpenAI-compatible messages list
        """
        openai_messages = []

        # Add system prompt if provided and no system message exists
        has_system = any(msg.get("role") == "system" for msg in messages)
        if system_prompt and not has_system:
            openai_messages.append({
                "role": "system",
                "content": system_prompt
            })

        for msg in messages:
            role = msg["role"]

            # Handle system messages
            if role == "system":
                openai_messages.append({
                    "role": "system",
                    "content": msg["content"]
                })
                continue

            # Handle messages with images (multimodal)
            if "images" in msg and msg["images"]:
                from llm.utils.image_utils import validate_image_paths, encode_image_to_base64

                # Validate images
                valid_images = validate_image_paths(msg["images"])

                # Build content array
                content = []

                # Add text content if present
                if msg.get("content"):
                    content.append({
                        "type": "text",
                        "text": msg["content"]
                    })

                # Add images
                for image_path in valid_images:
                    try:
                        data_url = encode_image_to_base64(image_path)
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": data_url
                            }
                        })
                        logger.debug(f"Added image to message: {image_path}")
                    except Exception as e:
                        logger.error(f"Failed to encode image {image_path}: {e}")
                        continue

                openai_messages.append({
                    "role": role,
                    "content": content
                })
            else:
                # Text-only message
                openai_messages.append({
                    "role": role,
                    "content": msg["content"]
                })

        return openai_messages

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate a complete response using OpenAI API.

        Args:
            messages: List of message dicts with "role" and "content" keys.
                     User messages may optionally include "images" field with List[str] of file paths.
                     Must include system message (if any) and conversation history.
                     The last message should be the current user message.
            system_prompt: Optional system prompt. If messages already contains a system
                          message, this may be ignored.
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Generated text response
        """
        # Convert to OpenAI format
        openai_messages = self._convert_messages_to_openai_format(messages, system_prompt)

        # Build generation parameters
        api_params = {"model": self.model, "messages": openai_messages}

        # Apply generation config defaults
        config = self.default_generation_config.copy()
        if temperature is not None:
            config["temperature"] = temperature
        if top_p is not None:
            config["top_p"] = top_p
        if max_tokens is not None:
            config["max_tokens"] = max_tokens

        # Merge with kwargs
        config.update(kwargs.get("generation_config", {}))

        # Apply config to API params
        for key in ["temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "stop"]:
            if key in config:
                api_params[key] = config[key]

        # Generate
        response = await self.client.chat.completions.create(**api_params)

        # Track tokens
        if response.usage:
            self.last_prompt_tokens = response.usage.prompt_tokens
            self.last_completion_tokens = response.usage.completion_tokens

        return response.choices[0].message.content

    async def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """
        Generate a streaming response using OpenAI API.

        Args:
            messages: List of message dicts with "role" and "content" keys.
                     User messages may optionally include "images" field with List[str] of file paths.
                     Must include system message (if any) and conversation history.
                     The last message should be the current user message.
            system_prompt: Optional system prompt. If messages already contains a system
                          message, this may be ignored.
            tools: Optional tool schemas in OpenAI format:
                  [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Yields:
            Text tokens (str) or tool call dicts:
            {"type": "tool_call", "id": "...", "name": "...", "arguments": {...}}
        """
        # Convert to OpenAI format
        openai_messages = self._convert_messages_to_openai_format(messages, system_prompt)

        # Build generation parameters
        api_params = {
            "model": self.model,
            "messages": openai_messages,
            "stream": True,
            "stream_options": {"include_usage": True}  # Get token counts in stream
        }

        # Apply generation config defaults
        config = self.default_generation_config.copy()
        if temperature is not None:
            config["temperature"] = temperature
        if top_p is not None:
            config["top_p"] = top_p
        if max_tokens is not None:
            config["max_tokens"] = max_tokens

        # Merge with kwargs
        config.update(kwargs.get("generation_config", {}))

        # Apply config to API params
        for key in ["temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "stop"]:
            if key in config:
                api_params[key] = config[key]

        # Add tools if provided
        if tools:
            api_params["tools"] = tools

        # Tool call accumulation buffer
        tool_call_buffer = {}  # {index: {id, name, arguments}}

        # Stream response
        stream = await self.client.chat.completions.create(**api_params)

        async for chunk in stream:
            # Handle usage metadata (comes in final chunk)
            if chunk.usage:
                self.last_prompt_tokens = chunk.usage.prompt_tokens
                self.last_completion_tokens = chunk.usage.completion_tokens

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Handle text content
            if delta.content:
                yield delta.content

            # Handle tool calls (streaming accumulation)
            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    idx = tool_call.index

                    # Initialize buffer for this tool call if needed
                    if idx not in tool_call_buffer:
                        tool_call_buffer[idx] = {
                            "id": "",
                            "name": "",
                            "arguments": ""
                        }

                    # Accumulate fields
                    if tool_call.id:
                        tool_call_buffer[idx]["id"] = tool_call.id
                    if tool_call.function and tool_call.function.name:
                        tool_call_buffer[idx]["name"] = tool_call.function.name
                    if tool_call.function and tool_call.function.arguments:
                        tool_call_buffer[idx]["arguments"] += tool_call.function.arguments

            # When finish_reason is "tool_calls", yield all accumulated tool calls
            if chunk.choices[0].finish_reason == "tool_calls":
                for tc_data in tool_call_buffer.values():
                    try:
                        yield {
                            "type": "tool_call",
                            "id": tc_data["id"],
                            "name": tc_data["name"],
                            "arguments": json.loads(tc_data["arguments"])
                        }
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse tool call arguments: {e}")
                        logger.error(f"Arguments string: {tc_data['arguments']}")
                        continue
                tool_call_buffer.clear()
