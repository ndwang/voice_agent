"""
Memory Tools

Tools for storing and retrieving semantic memories across conversations.
"""
from typing import Dict, Any
from orchestrator.tools.registry import tool
from orchestrator.memory.storage import get_memory_storage
from core.logging import get_logger

logger = get_logger(__name__)


@tool(
    name="write_memory",
    description="【储存长期记忆】将当前对话中的关键事实、用户偏好或背景信息提取并持久化存储。当用户提到未来可能需要引用的新信息（如职业变化、习惯、特定项目背景）时，应调用此工具。请确保储存的内容具有独立性，且包含完整的语境，避免存储模糊的代词（如'他'、'那个'）。",
    schema={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "要记忆的具体内容。应包含主题和背景。例如：'用户 Alex 是一名软件工程师，偏好 Python 语言，目前在从事机器学习项目。'"
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "（可选）用于对该记忆进行分类的标签。例如：['用户信息', '偏好'], ['工作', '项目'], ['重要']"
            }
        },
        "required": ["content"]
    }
)
async def write_memory(params: Dict[str, Any]) -> str:
    """
    Store a memory with optional tags.

    Args:
        params: Dict with "content" (required) and "tags" (optional)

    Returns:
        Confirmation message with memory ID
    """
    content = params.get("content")
    tags = params.get("tags", [])

    if not content:
        return "Error: No content provided for memory"

    try:
        storage = get_memory_storage()
        memory_id = await storage.add_memory(content=content, tags=tags)

        tags_str = f" with tags: {', '.join(tags)}" if tags else ""
        return f"Memory stored successfully (ID: {memory_id[:8]}...){tags_str}"
    except Exception as e:
        logger.error(f"Error storing memory: {e}", exc_info=True)
        return f"Error storing memory: {str(e)}"


@tool(
    name="search_memories",
    description="通过语义匹配检索之前存储的记忆。当用户询问‘我之前说过什么’、‘我的偏好是什么’或提到模型不具备的特定上下文时，必须调用此工具。支持模糊匹配，因此查询词应侧重于核心实体名或概念词。",
    schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "语义搜索关键词或自然语言问题。例如：'用户的职业背景是什么？' 或 '项目A的截止日期'"
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "（可选）缩小搜索范围的标签过滤器。如果已知信息类别，应提供标签以提高检索准确度。"
            },
            "limit": {
                "type": "integer",
                "description": "（可选）返回的最相关记忆条数。对于宽泛的查询建议设为 10，对于精确查询设为 3-5。",
                "default": 5
            }
        },
        "required": ["query"]
    }
)
async def search_memories(params: Dict[str, Any]) -> str:
    """
    Search memories semantically.

    Args:
        params: Dict with "query" (required), "tags" (optional), "limit" (optional)

    Returns:
        Formatted list of matching memories with relevance
    """
    query = params.get("query")
    tags = params.get("tags")
    limit = params.get("limit", 5)

    if not query:
        return "Error: No search query provided"

    try:
        storage = get_memory_storage()
        memories = await storage.search_memories(query=query, limit=limit, tags=tags)

        if not memories:
            return f"No memories found matching: {query}"

        # Format results
        result_lines = [f"Found {len(memories)} relevant memor{'ies' if len(memories) > 1 else 'y'}:\n"]

        for i, mem in enumerate(memories, 1):
            tags_str = f" [{', '.join(mem['tags'])}]" if mem['tags'] else ""
            result_lines.append(f"{i}. {mem['content']}{tags_str}")

        return "\n".join(result_lines)
    except Exception as e:
        logger.error(f"Error searching memories: {e}", exc_info=True)
        return f"Error searching memories: {str(e)}"


@tool(
    name="list_memory_tags",
    description="获取当前已存储记忆的所有标签分类。当你需要了解目前掌握了用户哪方面的信息，或者在执行搜索前需要确定过滤范围时，调用此工具。这通常是复杂检索任务的第一步，用于确定后续的过滤范围。",
    schema={
        "type": "object",
        "properties": {},
        "required": []
    }
)
async def list_memory_tags(params: Dict[str, Any]) -> str:
    """
    List all unique tags with counts.

    Args:
        params: Empty dict (no parameters required)

    Returns:
        Formatted list of tags and their counts
    """
    try:
        storage = get_memory_storage()
        tag_counts = await storage.list_all_tags()

        if not tag_counts:
            return "No tags found in stored memories"

        # Format results (sort by count, descending)
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        result_lines = [f"Found {len(sorted_tags)} unique tag{'s' if len(sorted_tags) > 1 else ''}:\n"]

        for tag, count in sorted_tags:
            result_lines.append(f"- {tag}: {count} memor{'ies' if count > 1 else 'y'}")

        return "\n".join(result_lines)
    except Exception as e:
        logger.error(f"Error listing tags: {e}", exc_info=True)
        return f"Error listing tags: {str(e)}"
