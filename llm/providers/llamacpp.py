"""
Llama.cpp Provider

Implementation of LLMProvider for llama.cpp using llama-cpp-python.
"""
import asyncio
import logging
from typing import AsyncIterator, Optional, List, Dict, Tuple
from pathlib import Path

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

from llm.base import LLMProvider

logger = logging.getLogger(__name__)


class LlamaCppProvider(LLMProvider):
    """Llama.cpp provider implementation using llama-cpp-python."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = -1,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize Llama.cpp provider.
        
        Args:
            model_path: Path to the GGUF model file
            n_ctx: Context window size (default: 4096)
            n_threads: Number of threads to use (None = auto)
            n_gpu_layers: Number of layers to offload to GPU (-1 = all layers)
            verbose: Enable verbose logging
            **kwargs: Additional parameters passed to Llama constructor
        """
        if Llama is None:
            raise ImportError(
                "llama-cpp-python is not installed. "
                "Install it with: pip install llama-cpp-python"
            )
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        
        # Initialize the model
        logger.info(f"Loading Llama.cpp model from {model_path}...")
        try:
            self.llama = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose,
                **kwargs
            )
            logger.info("Llama.cpp model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Llama.cpp model: {e}", exc_info=True)
            raise
        
        # Conversation history management
        self.conversation_history: List[Dict[str, str]] = []
        
        # Default generation parameters
        self.default_temperature = 0.8
        self.default_top_p = 0.95
        self.default_top_k = 40
        self.default_repeat_penalty = 1.1
    
    def _format_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Format prompt with conversation history.
        
        Args:
            prompt: Current user prompt
            system_prompt: Optional system prompt
            
        Returns:
            Formatted prompt string
        """
        # Build the full prompt with history
        full_prompt_parts = []
        
        if system_prompt:
            full_prompt_parts.append(f"System: {system_prompt}\n")
        
        # Add conversation history
        for msg in self.conversation_history:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                full_prompt_parts.append(f"User: {content}\n")
            elif role == "assistant":
                full_prompt_parts.append(f"Assistant: {content}\n")
        
        # Add current prompt
        full_prompt_parts.append(f"User: {prompt}\nAssistant:")
        
        return "".join(full_prompt_parts)
    
    def parse_error(self, exception: Exception) -> Tuple[int, str]:
        """
        Parse llama.cpp errors and return appropriate HTTP status code and message.
        
        Args:
            exception: The exception raised by llama.cpp
            
        Returns:
            Tuple of (status_code, error_message)
        """
        error_msg = str(exception)
        
        # Check for common error types
        if "model" in error_msg.lower() or "file" in error_msg.lower():
            return 400, f"Model error: {error_msg}"
        elif "memory" in error_msg.lower() or "out of" in error_msg.lower():
            return 507, f"Insufficient resources: {error_msg}"
        else:
            return 500, f"Llama.cpp error: {error_msg}"
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Generate a complete response using llama.cpp.
        
        Args:
            prompt: User input text
            system_prompt: Optional system prompt
            temperature: Sampling temperature (default: 0.7)
            top_p: Top-p sampling parameter (default: 0.9)
            top_k: Top-k sampling parameter (default: 40)
            max_tokens: Maximum tokens to generate (default: None = use model default)
            stop: List of stop sequences
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        # Format prompt with history
        formatted_prompt = self._format_prompt(prompt, system_prompt)
        
        # Prepare generation parameters
        gen_params = {
            "prompt": formatted_prompt,
            "temperature": temperature if temperature is not None else self.default_temperature,
            "top_p": top_p if top_p is not None else self.default_top_p,
            "top_k": top_k if top_k is not None else self.default_top_k,
            "repeat_penalty": kwargs.get("repeat_penalty", self.default_repeat_penalty),
        }
        
        if max_tokens is not None:
            gen_params["max_tokens"] = max_tokens
        
        if stop is not None:
            gen_params["stop"] = stop
        
        # Run generation in executor to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.llama(**gen_params, echo=False)
        )
        
        # Extract text from response
        text = response["choices"][0]["text"].strip()
        
        # Update conversation history
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
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response using llama.cpp.
        
        Args:
            prompt: User input text
            system_prompt: Optional system prompt
            temperature: Sampling temperature (default: 0.7)
            top_p: Top-p sampling parameter (default: 0.9)
            top_k: Top-k sampling parameter (default: 40)
            max_tokens: Maximum tokens to generate (default: None = use model default)
            stop: List of stop sequences
            **kwargs: Additional parameters
            
        Yields:
            Tokens as they are generated
        """
        # Format prompt with history
        formatted_prompt = self._format_prompt(prompt, system_prompt)
        
        # Prepare generation parameters
        gen_params = {
            "prompt": formatted_prompt,
            "temperature": temperature if temperature is not None else self.default_temperature,
            "top_p": top_p if top_p is not None else self.default_top_p,
            "top_k": top_k if top_k is not None else self.default_top_k,
            "repeat_penalty": kwargs.get("repeat_penalty", self.default_repeat_penalty),
            "stream": True,
        }
        
        if max_tokens is not None:
            gen_params["max_tokens"] = max_tokens
        
        if stop is not None:
            gen_params["stop"] = stop
        
        # Collect full response for history
        full_response = ""
        
        # Use a queue to bridge synchronous generator to async
        queue = asyncio.Queue()
        loop = asyncio.get_event_loop()
        done = False
        
        def _stream_generator():
            """Run the synchronous stream generator in a thread."""
            nonlocal done
            try:
                stream = self.llama(**gen_params, echo=False)
                for chunk in stream:
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("text", "")
                        if delta:
                            # Put tokens in queue
                            loop.call_soon_threadsafe(queue.put_nowait, delta)
                loop.call_soon_threadsafe(queue.put_nowait, None)  # Signal completion
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                done = True
        
        # Start the generator in a thread
        await loop.run_in_executor(None, _stream_generator)
        
        # Yield tokens from the queue
        while True:
            item = await queue.get()
            if item is None:  # Completion signal
                break
            if isinstance(item, Exception):
                raise item
            full_response += item
            yield item
        
        # Update conversation history after streaming completes
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

