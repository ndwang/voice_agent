"""
Ollama Provider

Implementation of LLMProvider for Ollama using the ollama Python library.
"""
import logging
import re
from typing import AsyncIterator, Optional, List, Dict, Tuple, Union, Any

try:
    from ollama import AsyncClient
except ImportError:
    AsyncClient = None

from llm.base import LLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Ollama provider implementation using the ollama Python library."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 300.0,
        disable_thinking: bool = False,
        generation_config: Optional[Dict] = None,
        **kwargs
    ):
        """
        Initialize Ollama provider.
        
        Args:
            model: Model name (e.g., "llama3", "qwen2.5", "mistral")
            base_url: Base URL for Ollama API (default: "http://localhost:11434")
            timeout: Request timeout in seconds (default: 300.0)
            disable_thinking: If True, disables thinking/reasoning mode for models that support it
            generation_config: Dictionary of default generation parameters (temperature, top_p, top_k, num_predict, etc.)
            **kwargs: Additional parameters
        """
        if AsyncClient is None:
            raise ImportError(
                "ollama is not installed. "
                "Install it with: pip install ollama"
            )
        
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.disable_thinking = disable_thinking
        
        # Store default generation config
        self.default_generation_config = generation_config or {}
        
        # Initialize the async client
        self.client = AsyncClient(host=base_url, timeout=timeout)

        super().__init__()
        
        logger.info(f"Initialized Ollama provider with model: {model}, base_url: {base_url}, disable_thinking: {disable_thinking}")
    
    def _filter_thinking_markers(self, text: str) -> str:
        """
        Filter out thinking markers like '<think>...</think>' from text.
        
        Args:
            text: Text that may contain thinking markers
            
        Returns:
            Text with thinking markers removed
        """
        # Pattern to match <think>...</think> tags
        # Uses non-greedy matching to handle multiple tags in the same text
        # Also handles whitespace around tags
        thinking_pattern = re.compile(r'<think>.*?</think>', re.DOTALL)
        filtered_text = thinking_pattern.sub('', text)
        # Clean up any double spaces or newlines that might result
        filtered_text = re.sub(r'\s+', ' ', filtered_text)
        return filtered_text.strip()
    
    def parse_error(self, exception: Exception) -> Tuple[int, str]:
        """
        Parse Ollama API errors and return appropriate HTTP status code and message.
        
        Args:
            exception: The exception raised by Ollama API
            
        Returns:
            Tuple of (status_code, error_message)
        """
        error_msg = str(exception)
        
        # Check for common error types
        if "connection" in error_msg.lower() or "refused" in error_msg.lower():
            return 503, f"Ollama service unavailable: {error_msg}"
        elif "model" in error_msg.lower() and "not found" in error_msg.lower():
            return 404, f"Model not found: {error_msg}"
        elif "timeout" in error_msg.lower():
            return 504, f"Request timeout: {error_msg}"
        else:
            return 500, f"Ollama error: {error_msg}"
    
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
        Generate a complete response using Ollama.

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
            **kwargs: Additional parameters

        Returns:
            Generated text response
        """
        # Process messages - handle images field
        final_messages = []
        for msg in messages:
            processed_msg = {
                "role": msg["role"],
                "content": msg["content"]
            }

            # Handle images if present (Ollama expects file paths directly)
            if "images" in msg and msg["images"]:
                from llm.utils.image_utils import validate_image_paths
                valid_images = validate_image_paths(msg["images"])
                if valid_images:
                    processed_msg["images"] = valid_images

            final_messages.append(processed_msg)

        # Apply /no_think to the last user message if thinking is disabled
        if self.disable_thinking:
            # Find the last user message and append /no_think
            for i in range(len(final_messages) - 1, -1, -1):
                if final_messages[i]["role"] == "user":
                    # Preserve images if present
                    updated_msg = {
                        "role": "user",
                        "content": final_messages[i]["content"] + " /no_think"
                    }
                    if "images" in final_messages[i]:
                        updated_msg["images"] = final_messages[i]["images"]
                    final_messages[i] = updated_msg
                    break
        
        # Prepare generation options - start with defaults from config
        options = self.default_generation_config.copy()
        if temperature is not None:
            options["temperature"] = temperature
        if top_p is not None:
            options["top_p"] = top_p
        if top_k is not None:
            options["top_k"] = top_k
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        
        # Merge any additional options from kwargs (overrides defaults and explicit params)
        options.update(kwargs.get("options", {}))
        
        # Generate response
        response = await self.client.chat(
            model=self.model,
            messages=final_messages,
            options=options if options else None
        )

        self.last_prompt_tokens = response.get("prompt_eval_count", 0)
        self.last_completion_tokens = response.get("eval_count", 0)
        
        # Extract text from response and filter thinking markers
        text = response["message"]["content"]
        if self.disable_thinking:
            text = self._filter_thinking_markers(text)
        
        return text
    
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
        Generate a streaming response using Ollama.

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
            **kwargs: Additional parameters

        Yields:
            Tokens as they are generated
        """
        # Process messages - handle images field
        final_messages = []
        for msg in messages:
            processed_msg = {
                "role": msg["role"],
                "content": msg["content"]
            }

            # Handle images if present (Ollama expects file paths directly)
            if "images" in msg and msg["images"]:
                from llm.utils.image_utils import validate_image_paths
                valid_images = validate_image_paths(msg["images"])
                if valid_images:
                    processed_msg["images"] = valid_images

            final_messages.append(processed_msg)

        # Apply /no_think to the last user message if thinking is disabled
        if self.disable_thinking:
            # Find the last user message and append /no_think
            for i in range(len(final_messages) - 1, -1, -1):
                if final_messages[i]["role"] == "user":
                    # Preserve images if present
                    updated_msg = {
                        "role": "user",
                        "content": final_messages[i]["content"] + " /no_think"
                    }
                    if "images" in final_messages[i]:
                        updated_msg["images"] = final_messages[i]["images"]
                    final_messages[i] = updated_msg
                    break
        
        # Prepare generation options - start with defaults from config
        options = self.default_generation_config.copy()
        if temperature is not None:
            options["temperature"] = temperature
        if top_p is not None:
            options["top_p"] = top_p
        if top_k is not None:
            options["top_k"] = top_k
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        
        # Merge any additional options from kwargs (overrides defaults and explicit params)
        options.update(kwargs.get("options", {}))

        # Stream response - await the coroutine to get the async iterator
        stream = await self.client.chat(
            model=self.model,
            messages=final_messages,
            tools=tools,  # Ollama accepts OpenAI format directly
            options=options if options else None,
            stream=True
        )
        async for chunk in stream:
            # Check for tool calls
            if "message" in chunk and "tool_calls" in chunk["message"]:
                for tool_call in chunk["message"]["tool_calls"]:
                    # Yield tool call in internal format for StreamProcessor
                    # Ollama returns OpenAI format, convert to internal format
                    yield {
                        "type": "tool_call",
                        "id": tool_call.get("id", f"call_{tool_call['function']['name']}"),
                        "name": tool_call["function"]["name"],
                        "arguments": tool_call["function"]["arguments"]
                    }
            elif "message" in chunk and "content" in chunk["message"]:
                content = chunk["message"]["content"]
                if content:
                    yield content

            if chunk.get("done"):
                self.last_prompt_tokens = chunk.get("prompt_eval_count", 0)
                self.last_completion_tokens = chunk.get("eval_count", 0)
