"""Text processing utilities for LLM output."""
import re
from typing import List, Tuple
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

