"""
Commentary analyzer using LLM to decide whether to react to dialogues.
"""
from typing import Optional, List
import json

from core.logging import get_logger
from llm.base import LLMProvider
from llm.providers.gemini import GeminiProvider
from vn_commentary.models import Dialogue, CommentaryDecision, CommentaryResult
from vn_commentary.context_manager import ContextManager

logger = get_logger(__name__)


class CommentaryAnalyzer:
    """
    Analyzes visual novel dialogues and generates contextual commentary.

    Uses LLM with structured output to decide whether to react to each
    dialogue line based on context and character personality.
    """

    DEFAULT_SYSTEM_PROMPT = """You are a visual novel commentary assistant observing a story unfold in real-time.

Your role is to decide whether a dialogue line is worth commenting on, and if so, provide a brief, natural reaction.

Guidelines:
- Stay silent for mundane, transitional, or uninteresting lines
- React to important plot developments, emotional moments, surprising revelations, or humorous situations
- Keep reactions brief and natural (1-2 sentences max)
- You see the story as it unfolds - you don't know what happens next
- React as an engaged observer, not a narrator

Pacing Guidelines:
- Avoid reacting to consecutive lines unless there's a major development
- If you haven't reacted in 8+ lines, consider reacting to keep engagement
- Balance is key: don't over-react, but don't stay silent too long

You will receive:
1. Chapter context so far (all dialogues from start of chapter up to current line)
2. How many lines since your last reaction
3. Current dialogue line to analyze

Respond with structured output indicating:
- action: "silent" or "react"
- reaction: Your commentary text (only if reacting)
- reasoning: Brief explanation of your decision (for debugging)
"""

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        context_manager: Optional[ContextManager] = None,
        system_prompt: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None
    ):
        """
        Initialize commentary analyzer.

        Args:
            llm_provider: LLM provider instance (if None, creates GeminiProvider)
            context_manager: Context manager instance (if None, creates default)
            system_prompt: Custom system prompt (if None, uses default)
            model: Model name for default Gemini provider
            api_key: API key for default Gemini provider
        """
        self.llm_provider = llm_provider or GeminiProvider(model=model, api_key=api_key)
        self.context_manager = context_manager or ContextManager()
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        # Chapter state
        self.current_chapter: List[Dialogue] = []
        self.current_index: int = 0
        self.last_reaction_index: Optional[int] = None

        logger.info(f"Initialized CommentaryAnalyzer with model: {model}")

    def set_chapter(self, dialogues: List[Dialogue]):
        """
        Set the current chapter dialogues for full context.

        Args:
            dialogues: List of all dialogues in this chapter
        """
        self.current_chapter = dialogues
        self.current_index = 0
        self.last_reaction_index = None
        logger.info(f"Loaded chapter with {len(dialogues)} dialogues")

    def end_chapter(self):
        """Signal end of current chapter and reset state."""
        logger.info(f"Chapter ended. Processed {self.current_index} dialogues, "
                   f"reactions at indices: {self._get_reaction_indices()}")
        self.current_chapter = []
        self.current_index = 0
        self.last_reaction_index = None

    def _get_reaction_indices(self) -> List[int]:
        """Get all indices where reactions occurred (for debugging)."""
        # This would require tracking all reactions, simplified for now
        return [self.last_reaction_index] if self.last_reaction_index is not None else []

    def _format_chapter_context(self) -> str:
        """Format chapter context up to current line (no spoilers)."""
        if not self.current_chapter:
            return "No chapter context yet."

        lines = ["Chapter context so far:"]
        # Only show dialogues up to (not including) current index
        for i in range(self.current_index):
            dialogue = self.current_chapter[i]
            lines.append(f"{i+1}. {dialogue.format_for_llm()}")

        if self.current_index == 0:
            lines.append("(This is the first line of the chapter)")

        return "\n".join(lines)

    def _calculate_lines_since_last_reaction(self) -> int:
        """Calculate number of lines since last reaction."""
        if self.last_reaction_index is None:
            return self.current_index
        return self.current_index - self.last_reaction_index

    async def analyze_dialogue(self, dialogue: Dialogue) -> CommentaryResult:
        """
        Analyze a dialogue line and decide whether to comment.

        Args:
            dialogue: Dialogue to analyze

        Returns:
            CommentaryResult with decision and optional reaction
        """
        # Build prompt with full chapter context
        chapter_context = self._format_chapter_context()
        current_line = dialogue.format_for_llm()
        lines_since_reaction = self._calculate_lines_since_last_reaction()

        # Build pacing info
        pacing_info = f"Lines since last reaction: {lines_since_reaction}"
        if lines_since_reaction == 0:
            pacing_info += " (just reacted - avoid consecutive reactions unless crucial)"
        elif lines_since_reaction >= 8:
            pacing_info += " (consider reacting to maintain engagement)"

        user_prompt = f"""{chapter_context}

{pacing_info}

Current dialogue to analyze:
{current_line}

Analyze this dialogue line and decide whether to react. Provide your response as JSON with this structure:
{{
    "action": "silent" or "react",
    "reaction": "your commentary here (only if reacting, null if silent)",
    "reasoning": "brief explanation of your decision"
}}"""

        # Call LLM with structured output request
        messages = [
            {"role": "user", "content": user_prompt}
        ]

        try:
            # Use generate (non-streaming) for structured output
            response = await self.llm_provider.generate(
                messages=messages,
                system_prompt=self.system_prompt,
                temperature=0.7
            )

            # Parse JSON response
            decision = self._parse_llm_response(response)

            # Update state
            if decision.action == "react":
                self.last_reaction_index = self.current_index
                logger.info(f"[{dialogue.dialogue_id}] REACT: {decision.reaction}")
            else:
                logger.debug(f"[{dialogue.dialogue_id}] SILENT: {decision.reasoning}")

            self.current_index += 1

            result = CommentaryResult(dialogue=dialogue, decision=decision)
            return result

        except Exception as e:
            logger.error(f"Failed to analyze dialogue {dialogue.dialogue_id}: {e}")
            self.current_index += 1
            # Return silent decision on error
            return CommentaryResult(
                dialogue=dialogue,
                decision=CommentaryDecision(
                    action="silent",
                    reasoning=f"Error during analysis: {str(e)}"
                )
            )

    def _parse_llm_response(self, response: str) -> CommentaryDecision:
        """
        Parse LLM response into CommentaryDecision.

        Args:
            response: Raw LLM response (should be JSON)

        Returns:
            Parsed CommentaryDecision
        """
        # Try to extract JSON from response (handle markdown code blocks)
        response = response.strip()

        # Remove markdown code blocks if present
        if response.startswith("```"):
            lines = response.split("\n")
            # Remove first and last lines (```json and ```)
            if len(lines) > 2:
                response = "\n".join(lines[1:-1])
            # Handle single-line code blocks
            else:
                response = response.replace("```json", "").replace("```", "").strip()

        try:
            data = json.loads(response)
            return CommentaryDecision(**data)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}\nResponse: {response}")
            # Try to extract action at minimum
            if "silent" in response.lower():
                return CommentaryDecision(action="silent", reasoning="Parsed from non-JSON response")
            elif "react" in response.lower():
                # Try to extract reaction text
                return CommentaryDecision(
                    action="react",
                    reaction=response,
                    reasoning="Parsed from non-JSON response"
                )
            else:
                # Default to silent
                return CommentaryDecision(
                    action="silent",
                    reasoning="Could not parse LLM response"
                )

    def reset_context(self):
        """Reset context (deprecated - use end_chapter instead)."""
        self.end_chapter()
        self.context_manager.clear()
        logger.info("Commentary analyzer context reset")
