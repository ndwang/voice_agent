"""
Data models for visual novel commentary system.
"""
from typing import Optional, Literal
from pydantic import BaseModel, Field


class Dialogue(BaseModel):
    """Represents a single dialogue line from a visual novel."""
    dialogue_id: str
    speaker: str
    japanese_text: str
    chinese_text: str

    def format_for_llm(self) -> str:
        """Format dialogue for LLM consumption."""
        return f"{self.speaker}: {self.chinese_text} ({self.japanese_text})"


class CommentaryDecision(BaseModel):
    """Structured output from LLM deciding whether to comment."""
    action: Literal["silent", "react"] = Field(
        description="Whether to stay silent or react to this dialogue"
    )
    reaction: Optional[str] = Field(
        default=None,
        description="The commentary/reaction text if action is 'react', None if silent"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief reasoning for the decision (for debugging/logging)"
    )


class CommentaryResult(BaseModel):
    """Result of processing a dialogue line."""
    dialogue: Dialogue
    decision: CommentaryDecision
