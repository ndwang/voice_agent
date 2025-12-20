"""
Context Manager

Manages conversation history and OCR context for LLM prompts.
"""
from typing import List, Optional, Dict
from datetime import datetime


class ContextManager:
    """Manages conversation history and OCR context."""
    
    def __init__(self, max_history: int = 10):
        """
        Initialize context manager.
        
        Args:
            max_history: Maximum number of conversation turns to keep
        """
        self.max_history = max_history
        self.conversation_history: List[Dict[str, str]] = []
        self.ocr_text: Optional[str] = None
        self.ocr_timestamp: Optional[float] = None
    
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
        # Build system message with OCR context
        system_parts = []
        if self.ocr_text:
            system_parts.append(f"You are reading a story with the user. Current story content:\n{self.ocr_text}")
        else:
            system_parts.append("You are a helpful assistant.")
        
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

