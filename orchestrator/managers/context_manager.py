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
    
    def __init__(self, max_history: int = 10, system_prompt_file: Optional[str] = None):
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
            system_prompt_file = str(project_root / "orchestrator" / "system_prompt.txt")
        else:
            # Resolve relative paths relative to project root
            prompt_path = Path(system_prompt_file)
            if not prompt_path.is_absolute():
                system_prompt_file = str(project_root / prompt_path)
        
        self.system_prompt_file = Path(system_prompt_file)
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
    
    def add_user_message(self, text: str):
        """Add user message to conversation history."""
        self.conversation_history.append({
            "role": "user",
            "content": text,
            "timestamp": datetime.now().isoformat()
        })
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
    
    def update_ocr_context(self, text: str):
        """Update OCR context with latest text."""
        import time
        self.ocr_text = text
        self.ocr_timestamp = time.time()
    
    def get_ocr_context(self) -> Optional[str]:
        """Get current OCR context."""
        return self.ocr_text
    
    def format_context_for_llm(self, user_message: str) -> Dict[str, any]:
        """
        Format context for LLM request.
        
        Args:
            user_message: Current user message
            
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
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
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

