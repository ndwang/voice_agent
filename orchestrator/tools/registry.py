import asyncio
from typing import Dict, Any, Callable, Awaitable
from dataclasses import dataclass
from core.logging import get_logger

logger = get_logger(__name__)

@dataclass
class Tool:
    name: str
    description: str
    handler: Callable[[Dict[str, Any]], Awaitable[Any]]
    schema: Dict[str, Any]

class ToolRegistry:
    """
    Registry for tools that can be invoked by the LLM.
    """
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        
    def register(self, tool: Tool):
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
        
    def get_tool(self, name: str) -> Tool:
        return self._tools.get(name)
        
    def get_tools_schema(self) -> list:
        """Get schema list for LLM function calling."""
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


