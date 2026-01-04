"""
Commentary analyzer using LLM to decide whether to react to dialogues.
"""
from typing import Optional, List
import json

from core.logging import get_logger
from llm.base import LLMProvider
from llm.providers.gemini import GeminiProvider
from vn_commentary.models import Dialogue, CommentaryDecision, CommentaryResult

logger = get_logger(__name__)


class CommentaryAnalyzer:
    """
    Analyzes visual novel dialogues and generates contextual commentary.

    Uses LLM with structured output to decide whether to react to each
    dialogue line based on context and character personality.
    """

    DEFAULT_SYSTEM_PROMPT = """你是一个视觉小说解说助手，实时观察故事展开。

你的任务是判断某一行对话是否值得评论，如果值得，提供简短自然的反应。

指导原则：
- 对于平淡、过渡性或无趣的台词保持沉默
- 对重要的剧情发展、情感时刻、意外揭示或幽默情况做出反应
- 反应要简短自然（最多1-2句话）
- 你是实时看到故事的——你不知道接下来会发生什么
- 作为一个投入的观察者来反应，而不是旁白

节奏指导：
- 保持自然的反应节奏，在重要时刻做出评论
- 避免对每一行都评论，也不要长时间保持沉默
- 关键在于平衡：根据内容的重要性和情感强度来判断

你将收到：
1. 目前为止的章节上下文（从章节开始到当前行的所有对话）
2. 距离上次反应有多少行
3. 当前需要分析的对话行

以结构化输出回应：
- action: "silent" 或 "react"
- reaction: 你的评论文本（仅在反应时提供）
- reasoning: 简要解释你的决定（用于调试）
"""

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        system_prompt: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None
    ):
        """
        Initialize commentary analyzer.

        Args:
            llm_provider: LLM provider instance (if None, creates GeminiProvider)
            system_prompt: Custom system prompt (if None, uses default)
            model: Model name for default Gemini provider
            api_key: API key for default Gemini provider
        """
        self.llm_provider = llm_provider or GeminiProvider(model=model, api_key=api_key)
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
            return "暂无章节上下文。"

        lines = ["目前为止的章节上下文："]
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

        # Build pacing info (just the number, let LLM decide)
        pacing_info = f"距离上次反应的行数：{lines_since_reaction}"

        user_prompt = f"""{chapter_context}

{pacing_info}

当前需要分析的对话：
{current_line}

分析这行对话并决定是否反应。用以下JSON结构回应：
{{
    "action": "silent" 或 "react",
    "reaction": "你的评论（仅在反应时提供，沉默时为null）",
    "reasoning": "简要解释你的决定"
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

