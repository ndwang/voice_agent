"""
Data models for visual novel commentary system.
"""
from typing import Optional, Literal
from pydantic import BaseModel, Field


class Dialogue(BaseModel):
    """Represents a single dialogue line from a visual novel."""
    dialogue_id: str
    speaker: str
    chinese_text: str
    japanese_text: Optional[str] = None

    def format_for_llm(self) -> str:
        """Format dialogue for LLM consumption (Chinese only)."""
        return f"{self.speaker}: {self.chinese_text}"


class CommentaryDecision(BaseModel):
    """Structured output from LLM deciding whether to comment."""
    action: Literal["silent", "react"] = Field(
        description="Whether to stay silent or react to this dialogue"
    )
    mode: Optional[Literal["inner_monologue", "spoken", "streamer_aside"]] = Field(
        default=None,
        description="How the reaction should be delivered."
    )
    emotion: Optional[str] = Field(
        default=None,
        description="Core emotion driving the reaction (e.g. shock, confusion, fear)."
    )
    intensity: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Emotional intensity of the reaction, from 0.0 (very mild) to 1.0 (emotional spike)."
    )
    instruction: Optional[str] = Field(
        default=None,
        description="Instruction for the speaker model (silent时为null)."
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief reasoning for the decision (for debugging/logging)."
    )


class CommentaryResult(BaseModel):
    """Result of processing a dialogue line."""
    dialogue: Dialogue
    decision: CommentaryDecision
