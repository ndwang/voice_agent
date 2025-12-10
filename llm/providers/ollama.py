"""
Ollama Provider

Implementation of LLMProvider for Ollama using the ollama Python library.
"""
import logging
import re
from typing import AsyncIterator, Optional, List, Dict, Tuple

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
        **kwargs
    ):
        """
        Initialize Ollama provider.
        
        Args:
            model: Model name (e.g., "llama3", "qwen2.5", "mistral")
            base_url: Base URL for Ollama API (default: "http://localhost:11434")
            timeout: Request timeout in seconds (default: 300.0)
            disable_thinking: If True, disables thinking/reasoning mode for models that support it
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
        
        # Initialize the async client
        self.client = AsyncClient(host=base_url, timeout=timeout)
        
        # Conversation history management
        self.conversation_history: List[Dict[str, str]] = []
        
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
        prompt: str,
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
            prompt: User input text
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        # Build messages from conversation history
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add current user prompt, appending '/no_think' if thinking is disabled
        user_prompt = prompt
        if self.disable_thinking:
            user_prompt = prompt + " /no_think"
        messages.append({"role": "user", "content": user_prompt})
        
        # Prepare generation options
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if top_p is not None:
            options["top_p"] = top_p
        if top_k is not None:
            options["top_k"] = top_k
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        
        # Merge any additional options from kwargs
        options.update(kwargs.get("options", {}))
        
        # Generate response
        response = await self.client.chat(
            model=self.model,
            messages=messages,
            options=options if options else None
        )
        
        # Extract text from response and filter thinking markers
        text = response["message"]["content"]
        if self.disable_thinking:
            text = self._filter_thinking_markers(text)
        
        # Update conversation history (store original prompt, not the modified one)
        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": text})
        
        return text
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response using Ollama.
        
        Args:
            prompt: User input text
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Yields:
            Tokens as they are generated
        """
        # Build messages from conversation history
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add current user prompt, appending '/no_think' if thinking is disabled
        user_prompt = prompt
        if self.disable_thinking:
            user_prompt = prompt + " /no_think"
        messages.append({"role": "user", "content": user_prompt})
        
        # Prepare generation options
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if top_p is not None:
            options["top_p"] = top_p
        if top_k is not None:
            options["top_k"] = top_k
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        
        # Merge any additional options from kwargs
        options.update(kwargs.get("options", {}))
        
        # Collect full response for history
        full_response = ""
        
        # Stream response - await the coroutine to get the async iterator
        stream = await self.client.chat(
            model=self.model,
            messages=messages,
            options=options if options else None,
            stream=True
        )
        async for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                content = chunk["message"]["content"]
                if content:
                    full_response += content
                    yield content
        
        # Filter the full response for history
        if self.disable_thinking:
            full_response = self._filter_thinking_markers(full_response)
        
        # Update conversation history after streaming completes (store original prompt)
        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": full_response})
    
    def get_history(self) -> List[Dict[str, str]]:
        """
        Get the history of the conversation.
        
        Returns:
            List of message dictionaries with "role" and "content" keys
        """
        return self.conversation_history.copy()
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history.clear()

