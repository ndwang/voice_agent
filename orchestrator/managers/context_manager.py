"""
Context Manager

Manages conversation history and OCR context for LLM prompts.
"""
from typing import List, Optional, Dict
from datetime import datetime
from pathlib import Path
import os
import asyncio
import logging

logger = logging.getLogger(__name__)


class ContextManager:
    """Manages conversation history and OCR context."""

    def __init__(
        self,
        max_history: int = 10,
        system_prompt_file: Optional[str] = None
    ):
        """
        Initialize context manager.

        Args:
            max_history: Maximum number of conversation turns to keep
            system_prompt_file: Path to system prompt file. If None, uses default location.
        """
        self.max_history = max_history
        self.conversation_history: List[Dict[str, str]] = []
        self.ocr_text: Optional[str] = None
        self.ocr_timestamp: Optional[float] = None
        
        # System prompt file management
        project_root = Path(__file__).parent.parent.parent
        if system_prompt_file is None:
            # Default to orchestrator/system_prompt.txt
            self.system_prompt_file = project_root / "orchestrator" / "system_prompt.txt"
        else:
            # Resolve relative paths relative to project root
            prompt_path = Path(system_prompt_file)
            if not prompt_path.is_absolute():
                self.system_prompt_file = project_root / prompt_path
            else:
                self.system_prompt_file = prompt_path
        self._system_prompt: Optional[str] = None
        self._system_prompt_mtime: Optional[float] = None
        self._hot_reload_task: Optional[asyncio.Task] = None
        self._hot_reload_enabled = False
        
        # Load initial system prompt
        self._load_system_prompt()
    
    def _load_system_prompt(self) -> str:
        """
        Load system prompt from file.
        
        Returns:
            System prompt text
        """
        try:
            if self.system_prompt_file.exists():
                mtime = self.system_prompt_file.stat().st_mtime
                # Only reload if file has changed
                if mtime != self._system_prompt_mtime:
                    with open(self.system_prompt_file, 'r', encoding='utf-8') as f:
                        self._system_prompt = f.read().strip()
                    self._system_prompt_mtime = mtime
                    logger.info(f"System prompt loaded from {self.system_prompt_file}")
            else:
                # Create default system prompt file if it doesn't exist
                default_prompt = "You are a helpful assistant."
                self._system_prompt = default_prompt
                self.system_prompt_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.system_prompt_file, 'w', encoding='utf-8') as f:
                    f.write(default_prompt)
                self._system_prompt_mtime = self.system_prompt_file.stat().st_mtime
                logger.info(f"Created default system prompt file at {self.system_prompt_file}")
        except Exception as e:
            logger.error(f"Error loading system prompt: {e}", exc_info=True)
            # Fallback to default
            if self._system_prompt is None:
                self._system_prompt = "You are a helpful assistant."
        
        return self._system_prompt or "You are a helpful assistant."
    
    def get_system_prompt(self) -> str:
        """Get current system prompt (checks file for updates)."""
        self._load_system_prompt()
        return self._system_prompt or "You are a helpful assistant."

    def reload_system_prompt(self) -> str:
        """
        Force reload system prompt from file.

        Returns:
            Reloaded system prompt text
        """
        # Reset mtime to force reload
        self._system_prompt_mtime = None
        return self._load_system_prompt()

    def set_system_prompt(self, prompt: str) -> bool:
        """
        Update system prompt and save to file.
        
        Args:
            prompt: New system prompt text
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.system_prompt_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.system_prompt_file, 'w', encoding='utf-8') as f:
                f.write(prompt)
            # Reload to update mtime
            self._load_system_prompt()
            logger.info(f"System prompt updated and saved to {self.system_prompt_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving system prompt: {e}", exc_info=True)
            return False
    
    def get_system_prompt_file_path(self) -> str:
        """Get the path to the system prompt file."""
        return str(self.system_prompt_file)
    
    async def _hot_reload_loop(self, check_interval: float = 1.0):
        """
        Background task to periodically check for system prompt file changes.
        
        Args:
            check_interval: Seconds between file checks
        """
        logger.info(f"System prompt hot-reload started (checking every {check_interval}s)")
        try:
            while self._hot_reload_enabled:
                try:
                    await asyncio.sleep(check_interval)
                    old_prompt = self._system_prompt
                    self._load_system_prompt()
                    if old_prompt != self._system_prompt:
                        logger.info("System prompt reloaded from file")
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Error in hot-reload loop: {e}", exc_info=True)
        except asyncio.CancelledError:
            logger.info("System prompt hot-reload stopped")
            raise
    
    def enable_hot_reload(self, check_interval: float = 1.0):
        """
        Enable hot-reloading of system prompt from file.
        
        Args:
            check_interval: Seconds between file checks
        """
        if not self._hot_reload_enabled:
            self._hot_reload_enabled = True
            self._hot_reload_task = asyncio.create_task(self._hot_reload_loop(check_interval))
            logger.info("System prompt hot-reload enabled")
    
    def disable_hot_reload(self):
        """Disable hot-reloading of system prompt."""
        if self._hot_reload_enabled:
            self._hot_reload_enabled = False
            if self._hot_reload_task:
                self._hot_reload_task.cancel()
                self._hot_reload_task = None
            logger.info("System prompt hot-reload disabled")
    
    def add_user_message(self, text: str, images: Optional[List[str]] = None):
        """
        Add user message with optional images to conversation history.

        Args:
            text: User message text
            images: Optional list of image file paths
        """
        message = {
            "role": "user",
            "content": text,
            "timestamp": datetime.now().isoformat()
        }

        # Only add images field if provided and valid
        if images:
            from llm.utils.image_utils import validate_image_paths
            valid_images = validate_image_paths(images)
            if valid_images:
                message["images"] = valid_images

        self.conversation_history.append(message)
        # Keep only last max_history messages
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-self.max_history:]
    
    def add_assistant_message(self, text: str):
        """Add assistant message to conversation history."""
        self.conversation_history.append({
            "role": "assistant",
            "content": text,
            "timestamp": datetime.now().isoformat()
        })

    def add_assistant_message_with_tool_calls(self, tool_calls: List[Dict]):
        """
        Add assistant message with tool calls to conversation history.

        Args:
            tool_calls: List of tool call dicts with "id", "name", "arguments"
        """
        self.conversation_history.append({
            "role": "assistant",
            "content": "",  # Empty content when tool calls are present
            "tool_calls": tool_calls,
            "timestamp": datetime.now().isoformat()
        })

    def add_tool_result_messages(self, tool_calls: List[Dict], results: List):
        """
        Add tool result messages to conversation history.

        Args:
            tool_calls: List of tool call dicts with "id", "name", "arguments"
            results: List of tool results (can be any type, will be stringified)
        """
        for tool_call, result in zip(tool_calls, results):
            # Format result as string
            if isinstance(result, Exception):
                result_str = f"Error: {str(result)}"
            else:
                result_str = str(result)

            self.conversation_history.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": tool_call["name"],
                "content": result_str,
                "timestamp": datetime.now().isoformat()
            })

    def get_messages_with_tool_results(
        self,
        tool_calls: List[Dict],
        results: List
    ) -> List[Dict[str, str]]:
        """
        Get messages including tool calls and results for interpretation request.

        Args:
            tool_calls: List of tool call dicts
            results: List of tool results

        Returns:
            List of message dicts formatted for LLM provider
        """
        # Add tool calls to history
        self.add_assistant_message_with_tool_calls(tool_calls)

        # Add tool results
        self.add_tool_result_messages(tool_calls, results)

        # Build message list with system prompt and history
        messages = []
        system_prompt = self.get_system_prompt()

        # Add OCR context to system prompt if available
        if self.ocr_text:
            system_prompt = f"{system_prompt}\n\nCurrent story content:\n{self.ocr_text}"

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add conversation history (including tool calls and results)
        for msg in self.conversation_history[-self.max_history * 2:]:
            if msg["role"] == "tool":
                # Tool result message
                messages.append({
                    "role": "tool",
                    "tool_call_id": msg["tool_call_id"],
                    "content": msg["content"]
                })
            elif "tool_calls" in msg:
                # Assistant message with tool calls
                # Convert internal format to OpenAI format
                openai_tool_calls = []
                for tc in msg["tool_calls"]:
                    openai_tool_calls.append({
                        "type": "function",
                        "id": tc["id"],
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"]  # Keep as dict
                        }
                    })
                messages.append({
                    "role": "assistant",
                    "content": msg.get("content", ""),
                    "tool_calls": openai_tool_calls
                })
            else:
                # Regular user/assistant message
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        return messages

    def update_last_message(self, text: str, role: Optional[str] = None):
        """Update the content of the last message in history."""
        if not self.conversation_history:
            return
            
        last_msg = self.conversation_history[-1]
        if role and last_msg["role"] != role:
            return
            
        last_msg["content"] = text
        last_msg["timestamp"] = datetime.now().isoformat()

    def get_last_message(self) -> Optional[Dict[str, str]]:
        """Get the last message in history."""
        if not self.conversation_history:
            return None
        return self.conversation_history[-1]

    def update_ocr_context(self, text: str):
        """Update OCR context with latest text."""
        import time
        self.ocr_text = text
        self.ocr_timestamp = time.time()
    
    def get_ocr_context(self) -> Optional[str]:
        """Get current OCR context."""
        return self.ocr_text
    
    def format_context_for_llm(self, user_message: str, images: Optional[List[str]] = None) -> Dict[str, any]:
        """
        Format context for LLM request with optional images.

        Args:
            user_message: Current user message
            images: Optional list of image file paths for current message

        Returns:
            Dictionary with prompt and context for LLM
        """
        # Get base system prompt from file
        system_parts = [self.get_system_prompt()]

        # Append OCR context if available
        if self.ocr_text:
            system_parts.append(f"\n\nCurrent story content:\n{self.ocr_text}")

        system_message = "\n".join(system_parts)

        # Build conversation history
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})

        # Add conversation history (excluding the current message if it's already the last one)
        history_to_include = self.conversation_history[-self.max_history * 2:]

        # Check if the last message in history matches the current user message
        # If so, don't include it in history (it will be added separately)
        if (history_to_include and
            history_to_include[-1]["role"] == "user" and
            history_to_include[-1]["content"] == user_message):
            history_to_include = history_to_include[:-1]

        for msg in history_to_include:
            if msg["role"] == "tool":
                # Tool result message
                messages.append({
                    "role": "tool",
                    "tool_call_id": msg["tool_call_id"],
                    "content": msg["content"]
                })
            elif "tool_calls" in msg:
                # Assistant message with tool calls
                # Convert internal format to OpenAI format
                openai_tool_calls = []
                for tc in msg["tool_calls"]:
                    openai_tool_calls.append({
                        "type": "function",
                        "id": tc["id"],
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"]  # Keep as dict
                        }
                    })
                messages.append({
                    "role": "assistant",
                    "content": msg.get("content", ""),
                    "tool_calls": openai_tool_calls
                })
            else:
                # Regular user/assistant message
                formatted_msg = {
                    "role": msg["role"],
                    "content": msg["content"]
                }
                # Include images if present in history
                if "images" in msg:
                    formatted_msg["images"] = msg["images"]
                messages.append(formatted_msg)

        # Add current user message with images
        current_msg = {"role": "user", "content": user_message}
        if images:
            from llm.utils.image_utils import validate_image_paths
            valid_images = validate_image_paths(images)
            if valid_images:
                current_msg["images"] = valid_images

        messages.append(current_msg)

        return {
            "messages": messages,
            "system_prompt": system_message,
            "context": self.ocr_text or ""
        }
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

    def clear_ocr_context(self):
        """Clear OCR context."""
        self.ocr_text = None
        self.ocr_timestamp = None

    # ========================================================================
    # Source-specific format functions
    # ========================================================================

    def format_voice_input(self, raw_text: str, was_interrupted: bool = False) -> str:
        """
        Format voice input with interruption concatenation logic.

        When a user interrupts the assistant mid-response and speaks again,
        this concatenates the new speech with the previous interrupted message.

        Args:
            raw_text: Raw voice input text
            was_interrupted: Whether the previous turn was interrupted before completion

        Returns:
            Formatted text, potentially concatenated with previous interrupted message
        """
        # Check if previous turn was interrupted before LLM finished
        if was_interrupted:
            last_msg = self.get_last_message()
            if last_msg and last_msg["role"] == "user":
                # Concatenate with previous interrupted message
                logger.info(f"Concatenating interrupted message: '{last_msg['content']}' + '{raw_text}'")
                formatted_text = last_msg['content'] + raw_text
                self.update_last_message(formatted_text, role="user")
                return formatted_text

        # No interruption or no previous user message, add normally
        self.add_user_message(raw_text)
        return raw_text

    def format_chat_input(self, raw_text: str, images: Optional[List[str]] = None) -> str:
        """
        Format chat input with optional images.

        TODO: Implement chat-specific formatting logic.

        Args:
            raw_text: Raw chat input text
            images: Optional list of image file paths

        Returns:
            Formatted text
        """
        # TBD: Add chat-specific formatting logic here
        self.add_user_message(raw_text, images=images)
        return raw_text

    def format_ocr_input(self, raw_text: str) -> str:
        """
        Format OCR input.

        TODO: Implement OCR-specific formatting logic.

        Args:
            raw_text: Raw OCR input text

        Returns:
            Formatted text
        """
        # TBD: Add OCR-specific formatting logic here
        self.add_user_message(raw_text)
        return raw_text

    def format_input(self, raw_text: str, source: str, images: Optional[List[str]] = None, was_interrupted: bool = False) -> str:
        """
        Format input based on source type with optional images.

        Args:
            raw_text: Raw input text
            source: Input source ('voice', 'chat', 'ocr', etc.)
            images: Optional list of image file paths
            was_interrupted: Whether the previous turn was interrupted (used for voice inputs)

        Returns:
            Formatted text
        """
        if source == "voice":
            # Voice input doesn't support images
            return self.format_voice_input(raw_text, was_interrupted)
        elif source == "bilibili_single":
            return self.format_chat_input(raw_text, images=images)
        elif source == "ocr":
            # OCR input doesn't support images
            return self.format_ocr_input(raw_text)
        else:
            # Unknown source - just add normally
            logger.warning(f"Unknown input source: {source}")
            self.add_user_message(raw_text, images=images)
            return raw_text

