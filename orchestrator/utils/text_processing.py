"""Text processing utilities for LLM output."""
import re
from typing import List, Tuple, Optional, Callable, Dict
from orchestrator.core.constants import (
    SENTENCE_END_PATTERN,
    THINKING_TAG_COMPLETE_PATTERN,
    THINKING_TAG_INCOMPLETE_PATTERN,
    THINKING_TAG_OPEN,
    THINKING_TAG_CLOSE,
)


def filter_thinking_tags(text: str, buffer: str = "") -> Tuple[str, str]:
    """
    Filter thinking/reasoning tags from text, handling incomplete tags.
    
    This function handles streaming text where thinking tags may be incomplete.
    It buffers incomplete tags and only returns safe text that doesn't contain
    thinking content.
    
    Args:
        text: New text chunk to process
        buffer: Previous buffer containing potentially incomplete tags
        
    Returns:
        Tuple of (safe_text, new_buffer)
        - safe_text: Text safe to process (thinking tags removed)
        - new_buffer: Buffer for next call (contains incomplete tags if any)
    """
    import re
    
    # Add new text to buffer
    buffer += text
    
    # Pattern to find and remove complete <think>...</think> tags
    thinking_pattern = re.compile(r'<think>.*?</think>', re.DOTALL)
    
    # Remove complete tags
    filtered_buffer = thinking_pattern.sub('', buffer)
    
    # Check if we're inside an incomplete tag (only check for full tag start)
    last_open = filtered_buffer.rfind('<think>')
    last_close = filtered_buffer.rfind('</think>')
    
    if last_open > last_close:
        # We are inside a thinking block
        safe_text = filtered_buffer[:last_open]
        new_buffer = filtered_buffer[last_open:]
    else:
        # No incomplete tags - process everything
        safe_text = filtered_buffer
        new_buffer = ""
    
    return safe_text, new_buffer


def filter_thinking_tags_final(buffer: str) -> str:
    """
    Final pass to filter any remaining thinking tags from buffer.
    
    Used when stream is complete to clean up any remaining tags.
    
    Args:
        buffer: Final buffer to clean
        
    Returns:
        Cleaned text with all thinking tags removed
    """
    import re
    # Remove complete <think> tags
    filtered = re.sub(r'<think>.*?</think>', '', buffer, flags=re.DOTALL)
    # Remove any trailing incomplete <think> tag
    filtered = re.sub(r'<think>.*$', '', filtered, flags=re.DOTALL)
    return filtered.strip()


def detect_sentence_end(text: str) -> int:
    """
    Detect if text ends with a sentence-ending punctuation.
    
    Args:
        text: Text to check
        
    Returns:
        Position of sentence end, or -1 if no sentence end found
    """
    match = SENTENCE_END_PATTERN.search(text)
    return match.end() if match else -1


def extract_sentences(text: str) -> List[str]:
    """
    Extract complete sentences from text.
    
    Args:
        text: Text to extract sentences from
        
    Returns:
        List of complete sentences
    """
    sentences = []
    remaining = text
    
    while True:
        match = SENTENCE_END_PATTERN.search(remaining)
        if not match:
            break
        
        end_pos = match.end()
        sentence = remaining[:end_pos].strip()
        if sentence:
            sentences.append(sentence)
        
        remaining = remaining[end_pos:]
    
    return sentences


class LLMStreamParser:
    """
    State-machine-based streaming parser for LLM responses with tagged content.
    
    Processes tokens incrementally, extracts content from tags, and fires callbacks
    immediately when complete sentences are detected. Designed for lowest latency.
    """
    
    def __init__(self, tag_configs: List[Dict[str, any]], default_callback: Optional[Callable[[str], None]] = None):
        """
        Initialize the parser.
        
        Args:
            tag_configs: List of tag configurations. Each config is a dict with:
                - "name": str - Tag name without brackets (e.g., "jp", "zh", "redacted_reasoning")
                - "callback": Optional[Callable] - Async callback function(text: str) -> None
                  If None, content is skipped. If provided, content is extracted and sent via callback.
            default_callback: Optional async callback for content outside any tags.
                If None, untagged content is skipped.
        """
        self.tag_configs = tag_configs
        self.default_callback = default_callback
        
        # Build tag lookup dictionary
        self.tag_map: Dict[str, Dict] = {}
        for config in tag_configs:
            tag_name = config["name"]
            self.tag_map[f"<{tag_name}>"] = config
            self.tag_map[f"</{tag_name}>"] = config
        
        # State management
        self.current_state: Optional[str] = None  # None = untagged, otherwise tag name
        self.buffer = ""  # Accumulates text for tag detection
        self.content_buffer = ""  # Accumulates text for sentence extraction
        
    async def process_token(self, token: str):
        """
        Process a new token from the stream.
        
        Args:
            token: New text chunk to process
        """
        self.buffer += token
        
        while self.buffer:
            # 1. State Switching Logic (Detect Tags)
            if self.current_state is None:
                # First, check if we see a closing tag without an opening tag
                # This can happen if the opening tag was split and we missed it
                for config in self.tag_configs:
                    tag_name = config["name"]
                    closing_tag = f"</{tag_name}>"
                    if closing_tag in self.buffer:
                        # Found closing tag but we're not in that state - skip it
                        # This means the opening tag was likely split and we missed it
                        parts = self.buffer.split(closing_tag, 1)
                        # Discard content before closing tag (it was part of the missed tag)
                        self.buffer = parts[1] if len(parts) > 1 else ""
                        # Continue to look for opening tags
                        continue
                
                # Look for opening tags in configuration order
                found_tag = False
                for config in self.tag_configs:
                    tag_name = config["name"]
                    opening_tag = f"<{tag_name}>"
                    
                    if opening_tag in self.buffer:
                        # Found opening tag - switch state
                        # First, flush any remaining untagged content in content_buffer
                        if self.content_buffer and self.default_callback:
                            await self._extract_and_send_sentences(is_final=True)
                        
                        self.current_state = tag_name
                        # Remove tag from buffer and continue processing
                        parts = self.buffer.split(opening_tag, 1)
                        # Process any content before the tag (untagged content)
                        if parts[0]:
                            await self._process_untagged_content(parts[0])
                        self.buffer = parts[1] if len(parts) > 1 else ""
                        found_tag = True
                        break
                
                if not found_tag:
                    # No tag found yet - process as untagged content
                    # But protect against partial tags at the end
                    # Check if buffer might contain start of any opening tag
                    max_tag_len = max(len(f"<{cfg['name']}>") for cfg in self.tag_configs) if self.tag_configs else 0
                    
                    # Check if buffer ends with a potential partial opening tag
                    # (e.g., if buffer ends with "<redacted_reason" we should wait)
                    potential_partial_tag = False
                    if self.buffer:
                        for config in self.tag_configs:
                            tag_name = config["name"]
                            opening_tag = f"<{tag_name}>"
                            # Check if buffer ends with a prefix of any opening tag (but not the full tag)
                            for i in range(1, len(opening_tag)):
                                if self.buffer.endswith(opening_tag[:i]):
                                    potential_partial_tag = True
                                    break
                            if potential_partial_tag:
                                break
                    
                    safe_len = len(self.buffer) - max_tag_len
                    
                    if safe_len > 0 and not potential_partial_tag:
                        to_process = self.buffer[:safe_len]
                        self.buffer = self.buffer[safe_len:]
                        await self._process_untagged_content(to_process)
                    else:
                        # Buffer too short or contains potential partial tag, wait for more tokens
                        break
            
            # 2. Inside a tag: Look for closing tag or process content
            else:
                closing_tag = f"</{self.current_state}>"
                config = self.tag_configs[next(i for i, cfg in enumerate(self.tag_configs) if cfg["name"] == self.current_state)]
                callback = config.get("callback")
                
                if closing_tag in self.buffer:
                    # Found the end of the section
                    parts = self.buffer.split(closing_tag, 1)
                    content = parts[0]
                    remaining = parts[1] if len(parts) > 1 else ""
                    
                    if callback:
                        await self._process_content(content, is_final=True)
                    
                    # Reset state and continue processing
                    self.current_state = None
                    self.buffer = remaining
                    self.content_buffer = ""
                else:
                    # Still inside the tag, process what we have for sentences
                    # But leave the end of the buffer in case a tag is partially formed
                    process_limit = len(self.buffer) - len(closing_tag)
                    if process_limit > 0:
                        to_process = self.buffer[:process_limit]
                        self.buffer = self.buffer[process_limit:]
                        
                        if callback:
                            await self._process_content(to_process, is_final=False)
                        else:
                            # No callback - just discard content
                            pass
                    break
    
    async def _process_untagged_content(self, text: str):
        """Process untagged content for sentence extraction."""
        if not self.default_callback:
            return  # Skip untagged content if no callback
        
        self.content_buffer += text
        await self._extract_and_send_sentences(is_final=False)
    
    async def _process_content(self, text: str, is_final: bool):
        """Process content within a tag for sentence extraction."""
        self.content_buffer += text
        await self._extract_and_send_sentences(is_final=is_final)
    
    async def _extract_and_send_sentences(self, is_final: bool):
        """Extract sentences from content_buffer and fire callbacks."""
        # Use regex to find sentence endings
        # SENTENCE_END_PATTERN matches: [!?。！？](?=[\s\n]|$)
        # We need to split while keeping delimiters
        
        # Find all sentence endings
        sentences = []
        remaining = self.content_buffer
        
        while True:
            match = SENTENCE_END_PATTERN.search(remaining)
            if not match:
                break
            
            end_pos = match.end()
            sentence = remaining[:end_pos].strip()
            if sentence:
                sentences.append(sentence)
            
            remaining = remaining[end_pos:]
        
        # Send complete sentences via callback
        callback = self._get_current_callback()
        if callback:
            for sentence in sentences:
                await callback(sentence)
        
        # Update content_buffer
        if is_final:
            # Send any remaining content as final sentence
            if remaining.strip() and callback:
                await callback(remaining.strip())
            self.content_buffer = ""
        else:
            # Keep incomplete sentence tail
            self.content_buffer = remaining
    
    def _get_current_callback(self) -> Optional[Callable]:
        """Get the appropriate callback for current state."""
        if self.current_state is None:
            return self.default_callback
        else:
            config = self.tag_configs[next(i for i, cfg in enumerate(self.tag_configs) if cfg["name"] == self.current_state)]
            return config.get("callback")
    
    async def finalize(self):
        """Flush any remaining content in buffers after stream ends."""
        # Process any remaining content in buffer
        if self.buffer:
            if self.current_state is None:
                # Untagged content remaining
                if self.default_callback:
                    await self._process_untagged_content(self.buffer)
            else:
                # Tag content remaining - treat as final
                config = self.tag_configs[next(i for i, cfg in enumerate(self.tag_configs) if cfg["name"] == self.current_state)]
                callback = config.get("callback")
                if callback:
                    await self._process_content(self.buffer, is_final=True)
        
        # Final flush of content_buffer
        if self.content_buffer:
            callback = self._get_current_callback()
            if callback and self.content_buffer.strip():
                await callback(self.content_buffer.strip())
        
        # Reset state
        self.current_state = None
        self.buffer = ""
        self.content_buffer = ""

