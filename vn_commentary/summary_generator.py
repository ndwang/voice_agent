"""
Summary generator for visual novel chapters.
"""
from typing import List, Dict, Optional
from pathlib import Path

from core.logging import get_logger
from vn_commentary.models import Dialogue
from core.settings import LLMSettings
from orchestrator.utils.llm_factory import create_provider

logger = get_logger(__name__)


class SummaryGenerator:
    """Generates neutral plot summaries for visual novel chapters."""

    def __init__(
        self,
        llm_settings: LLMSettings,
        system_prompt: str,
        user_prompt_template: str,
        max_words: int = 300,
        include_reactions: bool = True
    ):
        """
        Initialize summary generator.

        Args:
            llm_settings: LLM settings for API access
            system_prompt: System prompt for summarizer
            user_prompt_template: User prompt template with placeholders
            max_words: Maximum word count for summary
            include_reactions: Whether to include reaction metadata in context
        """
        self.llm_provider = create_provider(llm_settings)
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.max_words = max_words
        self.include_reactions = include_reactions

        logger.info(f"Initialized SummaryGenerator with model: {llm_settings.get_provider_config().model}")

    def _format_dialogues(self, dialogues: List[Dialogue]) -> str:
        """Format dialogues for summary prompt."""
        lines = []
        for i, dialogue in enumerate(dialogues, 1):
            lines.append(f"{i}. {dialogue.speaker}: {dialogue.chinese_text}")
        return "\n".join(lines)

    def _format_reactions_context(self, reactions: List[Dict]) -> str:
        """Format reaction metadata for context (optional)."""
        if not self.include_reactions or not reactions:
            return ""

        lines = ["反应发生位置："]
        for r in reactions:
            idx = r["index"]
            emotion = r.get("emotion", "未知")
            intensity = r.get("intensity", 0.0)
            lines.append(f"- 第{idx+1}行：{emotion}（强度：{intensity}）")

        return "\n".join(lines)

    async def generate_summary(
        self,
        chapter_name: str,
        dialogues: List[Dialogue],
        reactions: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate neutral plot summary for a chapter.

        Args:
            chapter_name: Name/identifier of the chapter
            dialogues: All dialogues in the chapter
            reactions: Optional list of reaction metadata

        Returns:
            Generated summary text
        """
        if not dialogues:
            return "空章节 - 无对话内容可摘要。"

        # Format context
        dialogues_text = self._format_dialogues(dialogues)
        reactions_context = self._format_reactions_context(reactions or [])

        # Fill user prompt template
        user_prompt = self.user_prompt_template.format(
            chapter_name=chapter_name,
            dialogue_count=len(dialogues),
            reaction_count=len(reactions) if reactions else 0,
            dialogues_text=dialogues_text,
            reactions_context=reactions_context,
            max_words=self.max_words
        )

        # Format system prompt with max_words
        system_prompt = self.system_prompt.format(max_words=self.max_words)

        messages = [{"role": "user", "content": user_prompt}]

        try:
            logger.debug(f"Generating summary for chapter: {chapter_name}")
            summary = await self.llm_provider.generate(
                messages=messages,
                system_prompt=system_prompt,
                temperature=0.5  # Lower temperature for more factual summaries
            )

            logger.info(f"Generated summary for {chapter_name} ({len(summary)} chars)")
            return summary.strip()

        except Exception as e:
            logger.error(f"Failed to generate summary for {chapter_name}: {e}")
            return f"生成摘要时出错：{str(e)}"
