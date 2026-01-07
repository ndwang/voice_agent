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
    description="Store a memory for future retrieval. Use this to remember important information about the user, conversation context, or facts that should persist across interactions. Be specific and include context in the content.",
    schema={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The text content to remember. Be descriptive and include context. Example: 'User Alex is a software engineer who prefers Python and works on ML projects.'"
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags to categorize this memory. Examples: ['user_info', 'preferences'], ['work', 'projects'], ['important']"
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
    description="Search stored memories using semantic search. Returns the most relevant memories matching your query. Use this to recall information from previous conversations.",
    schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find relevant memories. Examples: 'What does the user do for work?', 'user preferences about coffee'"
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional: filter results to only memories with these tags. Example: ['user_info']"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 5, max: 20)",
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
    description="Get a list of all unique tags used in stored memories. Useful for discovering what categories of information are available.",
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
