"""
Pydantic models for Chat Summarizer API.
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class SummarizeRequest(BaseModel):
    """Request to summarize recent chat messages."""
    max_messages: Optional[int] = Field(default=None, ge=1, le=500, description="Maximum number of recent messages to analyze (None = all messages)")
    time_window_seconds: Optional[int] = Field(default=None, ge=1, description="Only analyze messages from last N seconds")


class MessageInfo(BaseModel):
    """Information about a single chat message."""
    id: str
    type: str  # "danmaku" or "superchat"
    user: str
    content: str
    timestamp: float
    amount: Optional[float] = None  # For superchat


class SummarizeResponse(BaseModel):
    """Response containing chat summary and analysis."""
    overall_sentiment: str = Field(description="LLM's analysis of overall chat sentiment and topics")
    most_interesting_message: Optional[MessageInfo] = Field(default=None, description="The most interesting message to respond to (if any)")
    reasoning: str = Field(description="LLM's reasoning for selecting this message")
    messages_analyzed: int = Field(description="Number of messages analyzed")
    timestamp: datetime = Field(description="When this summary was generated")


class BufferStatsResponse(BaseModel):
    """Response containing buffer statistics."""
    danmaku_count: int
    superchat_count: int
    total_count: int
    oldest_timestamp: Optional[float] = None
    newest_timestamp: Optional[float] = None
    sample_messages: list[dict] = Field(default_factory=list, description="Sample of recent messages")
