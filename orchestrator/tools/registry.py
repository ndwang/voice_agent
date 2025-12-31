import asyncio
from typing import Dict, Any, Callable, Awaitable, Optional
from dataclasses import dataclass
from functools import wraps
from core.logging import get_logger

logger = get_logger(__name__)

@dataclass
class Tool:
    name: str
    description: str
    handler: Callable[[Dict[str, Any]], Awaitable[Any]]
    schema: Dict[str, Any]


# Global list to collect decorated tools
_registered_tools = []


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    schema: Optional[Dict[str, Any]] = None
):
    """
    Decorator to register a function as a tool.

    Usage:
        @tool(
            name="my_tool",
            description="Does something useful",
            schema={
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "First param"}
                },
                "required": ["param1"]
            }
        )
        async def my_tool(params: Dict[str, Any]) -> str:
            return f"Result: {params['param1']}"

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to function docstring)
        schema: JSON Schema for parameters
    """
    def decorator(func: Callable[[Dict[str, Any]], Awaitable[Any]]):
        # Use function name if name not provided
        tool_name = name or func.__name__

        # Use docstring if description not provided
        tool_description = description or (func.__doc__ or "").strip()

        # Default schema if not provided
        tool_schema = schema or {
            "type": "object",
            "properties": {},
            "required": []
        }

        # Create Tool object
        tool_obj = Tool(
            name=tool_name,
            description=tool_description,
            handler=func,
            schema=tool_schema
        )

        # Add to global registry
        _registered_tools.append(tool_obj)

        # Return original function unchanged
        return func

    return decorator


def get_registered_tools():
    """Get all tools registered via @tool decorator."""
    return _registered_tools.copy()


def discover_and_load_tools(tools_dir: str = None):
    """
    Automatically discover and import all tool modules in the tools directory.

    This scans the tools directory for Python files and imports them,
    which triggers the @tool decorator to register them.

    Args:
        tools_dir: Path to tools directory (defaults to orchestrator/tools)
    """
    import os
    import importlib.util
    from pathlib import Path

    # Default to orchestrator/tools directory
    if tools_dir is None:
        current_file = Path(__file__)
        tools_dir = current_file.parent
    else:
        tools_dir = Path(tools_dir)

    if not tools_dir.exists():
        logger.warning(f"Tools directory not found: {tools_dir}")
        return

    # Find all .py files in the directory
    tool_files = tools_dir.glob("*.py")

    for tool_file in tool_files:
        # Skip special files
        if tool_file.name in ["__init__.py", "registry.py"]:
            continue

        # Import the module
        module_name = f"orchestrator.tools.{tool_file.stem}"
        try:
            importlib.import_module(module_name)
            logger.info(f"Loaded tool module: {module_name}")
        except Exception as e:
            logger.error(f"Failed to load tool module {module_name}: {e}", exc_info=True)

class ToolRegistry:
    """
    Registry for tools that can be invoked by the LLM.
    """
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._enabled_tools: Dict[str, bool] = {}
        
    def register(self, tool: Tool):
        self._tools[tool.name] = tool
        self._enabled_tools[tool.name] = True  # Default enabled
        logger.info(f"Registered tool: {tool.name}")
        
    def get_tool(self, name: str) -> Tool:
        return self._tools.get(name)
        
    def get_tools_schema(self) -> list:
        """Get schema list for LLM function calling (filtered by enabled state)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.schema
                }
            }
            for tool in self._tools.values()
            if self._enabled_tools.get(tool.name, True)  # Filter by enabled state
        ]
        
    async def execute(self, name: str, params: Dict[str, Any]) -> Any:
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool {name} not found")

        try:
            return await tool.handler(params)
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}", exc_info=True)
            raise

    def get_all_tools_info(self) -> list:
        """Get all tools with metadata for UI."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "enabled": self._enabled_tools.get(tool.name, True)
            }
            for tool in self._tools.values()
        ]

    def enable_tool(self, name: str) -> bool:
        """Enable a tool. Returns True if successful."""
        if name not in self._tools:
            return False
        self._enabled_tools[name] = True
        logger.info(f"Tool enabled: {name}")
        return True

    def disable_tool(self, name: str) -> bool:
        """Disable a tool. Returns True if successful."""
        if name not in self._tools:
            return False
        self._enabled_tools[name] = False
        logger.info(f"Tool disabled: {name}")
        return True

    def is_tool_enabled(self, name: str) -> bool:
        """Check if tool is enabled."""
        return self._enabled_tools.get(name, True)


