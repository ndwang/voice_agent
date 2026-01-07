"""
Commentary analyzer using LLM to decide whether to react to dialogues.
"""
from typing import Optional, List, Dict
from collections import deque
import json

from core.logging import get_logger
from vn_commentary.models import Dialogue, CommentaryDecision, CommentaryResult
from core.settings import LLMSettings
from orchestrator.utils.llm_factory import create_provider

logger = get_logger(__name__)


class CommentaryAnalyzer:
    """
    Analyzes visual novel dialogues and generates contextual commentary.

    Uses LLM with structured output to decide whether to react to each
    dialogue line based on context and character personality.
    """

    def __init__(
        self,
        llm_settings: LLMSettings,
        system_prompt: str,
        user_prompt_template: Optional[str] = None,
        max_recent_reactions: int = 5
    ):
        """
        Initialize commentary analyzer.

        Args:
            llm_settings: LLM settings instance (if None, creates GeminiProvider)
            system_prompt: Custom system prompt
            user_prompt_template: User prompt template with placeholders (optional)
            max_recent_reactions: Maximum number of recent reactions to track
        """
        self.llm_provider = create_provider(llm_settings)
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.max_recent_reactions = max_recent_reactions

        # Chapter state
        self.current_chapter: List[Dialogue] = []
        self.current_index: int = 0
        self.recent_reactions: deque = deque(maxlen=max_recent_reactions)
        self.last_reaction_index: Optional[int] = None  # Keep for backward compatibility

        # Token usage tracking (latest call only)
        self.last_prompt_tokens: int = 0
        self.last_completion_tokens: int = 0

        logger.info(f"Initialized CommentaryAnalyzer with model: {llm_settings.get_provider_config().model}")

    @property
    def last_token_usage(self) -> Dict:
        """
        Get token usage from the most recent LLM call.

        Returns:
            Dictionary with prompt_tokens, completion_tokens, and total_tokens
        """
        return {
            "prompt_tokens": self.last_prompt_tokens,
            "completion_tokens": self.last_completion_tokens,
            "total_tokens": self.last_prompt_tokens + self.last_completion_tokens
        }

    def set_chapter(self, dialogues: List[Dialogue]):
        """
        Set the current chapter dialogues for full context.

        Args:
            dialogues: List of all dialogues in this chapter
        """
        self.current_chapter = dialogues
        self.current_index = 0
        self.last_reaction_index = None
        self.recent_reactions.clear()
        logger.info(f"Loaded chapter with {len(dialogues)} dialogues")

    def end_chapter(self) -> Dict[str, any]:
        """
        Signal end of current chapter and return chapter data for summary.

        Returns:
            Dict containing chapter data: dialogues, reactions, counts
        """
        logger.info(f"Chapter ended. Processed {self.current_index} dialogues, "
                   f"reactions at indices: {self._get_reaction_indices()}")

        # Prepare chapter data for summary generation
        chapter_data = {
            "dialogues": self.current_chapter.copy(),
            "reactions": list(self.recent_reactions),  # Convert deque to list
            "dialogue_count": self.current_index,
            "reaction_count": len(self.recent_reactions)
        }

        # Reset state
        self.current_chapter = []
        self.current_index = 0
        self.last_reaction_index = None
        self.recent_reactions.clear()

        return chapter_data

    def _get_reaction_indices(self) -> List[int]:
        """Get all indices where reactions occurred (for debugging)."""
        return [r["index"] for r in self.recent_reactions]

    def _format_chapter_context(self) -> str:
        """Format chapter context up to current line (no spoilers)."""
        if not self.current_chapter:
            return "暂无章节上下文。"

        lines = ["目前为止的章节内容："]
        # Only show dialogues up to (not including) current index
        for i in range(self.current_index):
            dialogue = self.current_chapter[i]
            lines.append(f"{i+1}. {dialogue.format_for_llm()}")

        if self.current_index == 0:
            lines.append("（这是本章的第一行）")

        return "\n".join(lines)

    def _calculate_lines_since_last_reaction(self) -> int:
        """Calculate number of lines since last reaction."""
        if self.last_reaction_index is None:
            return self.current_index
        return self.current_index - self.last_reaction_index

    def _format_recent_reactions(self) -> str:
        """
        Format recent reactions for LLM prompt (KV cache friendly).

        Returns a fixed-format section showing recent reaction history to help
        the LLM maintain appropriate pacing.
        """
        if not self.recent_reactions:
            return "=== 最近的反应记录 ===\n（暂无反应记录）\n"

        lines = ["=== 最近的反应记录 ==="]
        for reaction in self.recent_reactions:
            idx = reaction["index"]
            mode = reaction.get("mode", "未知")
            emotion = reaction.get("emotion", "未知")
            intensity = reaction.get("intensity", "未知")
            instruction = reaction.get("instruction", "未知")

            lines.append(f"行{idx+1}: [{mode}] {emotion} {intensity} - {instruction}")

        return "\n".join(lines) + "\n"

    async def analyze_dialogue(self, dialogue: Dialogue) -> CommentaryResult:
        """
        Analyze a dialogue line and decide whether to comment.

        Args:
            dialogue: Dialogue to analyze

        Returns:
            CommentaryResult with decision and optional reaction
        """
        # Build prompt with optimal KV cache structure: static → semi-static → dynamic
        recent_reactions = self._format_recent_reactions()
        chapter_context = self._format_chapter_context()
        current_line = dialogue.format_for_llm()
        lines_since_reaction = self._calculate_lines_since_last_reaction()

        # Build pacing info (just the number, let LLM decide)
        pacing_info = f"距离上次发言过去了{lines_since_reaction}行台词。"

        # Use user prompt template if provided, otherwise use default
        if self.user_prompt_template:
            user_prompt = self.user_prompt_template.format(
                chapter_context=chapter_context,
                recent_reactions=recent_reactions,
                pacing_info=pacing_info,
                current_line=current_line
            )
        else:
            # Fallback to original hardcoded prompt
            user_prompt = f"""艾玛，现在的气氛适合你开口吗？考虑到直播间的观感和你对这段剧情的感触，以及回复的频率。
请输出 JSON：
{{
    "reasoning": "限20字内。分析该行是否触动了艾玛的怕寂寞性格、推理直觉或直播效果。",
    "action": "silent" 或 "react",
    "mode": "spoken" | "streamer_aside" (silent时为null),
    "emotion": "情绪关键词" (silent时为null),
    "intensity": 0.0-1.0 (情绪波动强度),
    "instruction": "给下游模型的具体演说指导。说明侧重点、潜台词或互动方向(silent时为null)"
}}

{chapter_context}

{recent_reactions}

{pacing_info}

当前需要分析的对话：
{current_line}"""

        # Call LLM with structured output request
        messages = [
            {"role": "user", "content": user_prompt}
        ]

        try:
            logger.debug(f"Analyzing dialogue {dialogue.dialogue_id}")
            logger.debug(f"Pacing info: {pacing_info}")
            logger.debug(f"User prompt: {user_prompt}")
            # Use generate (non-streaming) for structured output
            response = await self.llm_provider.generate(
                messages=messages,
                system_prompt=self.system_prompt,
                temperature=0.7,
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": CommentaryDecision.model_json_schema()
                }
            )

            # Capture token usage from provider
            self.last_prompt_tokens = self.llm_provider.last_prompt_tokens
            self.last_completion_tokens = self.llm_provider.last_completion_tokens
            logger.debug(f"Token usage - Prompt: {self.last_prompt_tokens}, "
                        f"Completion: {self.last_completion_tokens}, "
                        f"Total: {self.last_prompt_tokens + self.last_completion_tokens}")

            # Parse JSON response
            decision = self._parse_llm_response(response)

            # Update state
            if decision.action == "react":
                self.last_reaction_index = self.current_index

                # Store reaction details for future context
                reaction_record = {
                    "index": self.current_index,
                    "reasoning": decision.reasoning or "未提供原因",
                    "mode": decision.mode,
                    "emotion": decision.emotion,
                    "intensity": decision.intensity,
                    "instruction": decision.instruction
                }
                self.recent_reactions.append(reaction_record)

                logger.info(f"[{dialogue.dialogue_id}] REACT: {decision.reasoning}")
            else:
                logger.info(f"[{dialogue.dialogue_id}] SILENT: {decision.reasoning}")

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

