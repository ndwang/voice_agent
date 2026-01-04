"""
Commentary analyzer using LLM to decide whether to react to dialogues.
"""
from typing import Optional
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

    DEFAULT_SYSTEM_PROMPT = """You are a visual novel commentary assistant observing a story unfold.

Your role is to decide whether a dialogue line is worth commenting on, and if so, provide a brief, natural reaction.

Guidelines:
- Stay silent for mundane, transitional, or uninteresting lines
- React to important plot developments, emotional moments, surprising revelations, or humorous situations
- Keep reactions brief and natural (1-2 sentences max)
- Maintain awareness of the story context from previous dialogues
- React as an engaged observer, not a narrator

You will receive:
1. Recent dialogue context (previous lines)
2. Current dialogue line to analyze

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

        logger.info(f"Initialized CommentaryAnalyzer with model: {model}")

    async def analyze_dialogue(self, dialogue: Dialogue) -> CommentaryResult:
        """
        Analyze a dialogue line and decide whether to comment.

        Args:
            dialogue: Dialogue to analyze

        Returns:
            CommentaryResult with decision and optional reaction
        """
        # Build prompt with context
        context_str = self.context_manager.format_context_for_llm()
        current_line = dialogue.format_for_llm()

        user_prompt = f"""{context_str}

Current dialogue:
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

            # Add dialogue to context for future analysis
            self.context_manager.add_dialogue(dialogue)

            result = CommentaryResult(dialogue=dialogue, decision=decision)

            if decision.action == "react":
                logger.info(f"[{dialogue.dialogue_id}] REACT: {decision.reaction}")
            else:
                logger.debug(f"[{dialogue.dialogue_id}] SILENT: {decision.reasoning}")

            return result

        except Exception as e:
            logger.error(f"Failed to analyze dialogue {dialogue.dialogue_id}: {e}")
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
        """Reset context (e.g., when starting a new chapter)."""
        self.context_manager.clear()
        logger.info("Commentary analyzer context reset")
