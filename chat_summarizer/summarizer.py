"""
LLM-based chat summarization logic.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from llm.base import LLMProvider
from core.logging import get_logger

logger = get_logger(__name__)

# Set up file logging for this module - use absolute path to ensure it's in project root
_project_root = Path(__file__).parent.parent
_log_file_handler = logging.FileHandler(_project_root / 'chat_summarizer_llm.log', encoding='utf-8')
_log_file_handler.setLevel(logging.DEBUG)
_log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(_log_file_handler)
logger.setLevel(logging.DEBUG)

logger.info("=" * 60)
logger.info("ChatSummarizer module loaded - logging to chat_summarizer_llm.log")
logger.info("=" * 60)


class ChatSummarizer:
    """
    Uses LLM to analyze chat messages and extract sentiment and interesting messages.
    """

    def __init__(self, llm_provider: LLMProvider):
        """
        Initialize summarizer.

        Args:
            llm_provider: LLM provider instance (Gemini, Ollama, etc.)
        """
        self.llm_provider = llm_provider

    async def summarize(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze messages using LLM.

        Args:
            messages: List of message dicts with keys: id, type, user, content, timestamp, amount (optional)

        Returns:
            Dict with keys:
                - overall_sentiment: str
                - most_interesting_message_id: str or None
                - reasoning: str
        """
        if not messages:
            return {
                "overall_sentiment": "No messages to analyze.",
                "most_interesting_message_id": None,
                "reasoning": "Message buffer is empty."
            }

        # Build prompt
        chat_text = self._format_messages_for_prompt(messages)
        message_ids = [m["id"] for m in messages]

        system_prompt = """你正在分析 Bilibili 的直播弹幕。你的任务是：
1. 总结整体情绪和讨论的主要话题
2. 识别出最值得回复的一条有趣消息（如果有的话）

“最有趣的消息”可以是：
- 一个有深度的提问
- 见解深刻的评论
- 幽默有趣的内容
- 对讨论有意义的贡献
- 值得感谢的 Bilibili 醒目留言（Superchat）

你必须仅返回一个有效的 JSON 对象。不要包含任何解释，不要使用 Markdown 格式，只需输出纯 JSON。

要求格式：
{
  "overall_sentiment": "用 1-2 句话总结氛围和话题",
  "most_interesting_message_id": "消息 ID 或 null",
  "reasoning": "简要说明选择该消息的原因"
}"""

        user_prompt = f"""以下是最近的弹幕消息（格式：[类型] [用户]: 消息）：

{chat_text}

可选的消息 ID 列表：{', '.join(message_ids)}

请仅返回一个 JSON 对象，不要包含任何其他文字。"""

        # Call LLM
        try:
            llm_messages = [
                {"role": "user", "content": user_prompt}
            ]

            response_text = await self.llm_provider.generate(
                messages=llm_messages,
                system_prompt=system_prompt,
                temperature=0.3,  # Lower temperature for more consistent JSON formatting
                max_tokens=500
            )

            logger.debug(f"LLM response: {response_text}")

            # Parse JSON from response
            result = self._extract_json(response_text)

            # Validate the result
            if not isinstance(result, dict):
                raise ValueError("LLM response is not a dictionary")

            # Ensure required fields exist
            result.setdefault("overall_sentiment", "Unable to analyze sentiment")
            result.setdefault("most_interesting_message_id", None)
            result.setdefault("reasoning", "No specific reasoning provided")

            return result

        except ValueError as e:
            # JSON parsing failed - try to extract info manually
            logger.warning(f"JSON parsing failed: {e}")
            logger.warning("Attempting manual extraction from LLM response...")

            try:
                # Use a simple text-based fallback
                return {
                    "overall_sentiment": response_text[:200] if len(response_text) > 200 else response_text,
                    "most_interesting_message_id": None,
                    "reasoning": "JSON parsing failed, showing raw LLM response"
                }
            except:
                return {
                    "overall_sentiment": f"Analysis error: {str(e)}",
                    "most_interesting_message_id": None,
                    "reasoning": "Failed to complete analysis"
                }

        except Exception as e:
            logger.error(f"Summarization failed: {e}", exc_info=True)
            return {
                "overall_sentiment": f"Analysis error: {str(e)}",
                "most_interesting_message_id": None,
                "reasoning": "Failed to complete analysis"
            }

    def _format_messages_for_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages as text for LLM prompt."""
        lines = []
        for msg in messages:
            msg_type = msg.get("type", "unknown")
            user = msg.get("user", "Anonymous")
            content = msg.get("content", "")
            msg_id = msg.get("id", "")

            # Add type indicator and amount for superchats
            if msg_type == "superchat" and msg.get("amount"):
                prefix = f"[SuperChat ¥{msg['amount']}]"
            else:
                prefix = f"[{msg_type}]"

            lines.append(f"{prefix} {user}: {content} (id: {msg_id})")

        return "\n".join(lines)

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON from LLM response.

        Handles responses that may have markdown code blocks or extra text.
        """
        # Try to find JSON in code blocks first
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            json_text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            json_text = text[start:end].strip()
        else:
            # Try to find JSON by looking for { and }
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                json_text = text[start:end]
            else:
                json_text = text.strip()

        try:
            result = json.loads(json_text)
            return result
        except json.JSONDecodeError as e:
            # Log the full response for debugging
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Full LLM response:\n{text}")
            logger.error(f"Extracted JSON text:\n{json_text}")

            # Try to salvage with a more aggressive extraction
            # Look for content between outermost braces
            import re
            match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
            if match:
                try:
                    logger.info("Trying alternative JSON extraction...")
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

            raise ValueError(f"Invalid JSON in LLM response: {e}")
