"""
Tool package initializer.

This file makes `orchestrator.tools` a proper Python package so that
`discover_and_load_tools` can import modules like `orchestrator.tools.memory_tools`
and register their `@tool`-decorated functions.
"""

