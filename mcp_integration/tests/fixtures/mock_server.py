"""
Mock MCP Server for testing.

A simple MCP server that exposes test tools:
- echo: Returns input text unchanged
- add: Adds two numbers
- greet: Returns a greeting message
- fail: Always returns an error (for testing error handling)

Run directly: python -m mcp_integration.tests.fixtures.mock_server
"""

import asyncio
import json
import logging
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    TextContent,
    Tool,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create server instance
server = Server("mock-test-server")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return list of available tools."""
    return [
        Tool(
            name="echo",
            description="Returns the input text unchanged",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to echo back",
                    }
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="add",
            description="Adds two numbers together",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "First number",
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number",
                    },
                },
                "required": ["a", "b"],
            },
        ),
        Tool(
            name="greet",
            description="Returns a greeting message",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name to greet",
                    }
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="fail",
            description="Always returns an error (for testing)",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Error message to return",
                    }
                },
                "required": [],
            },
        ),
        Tool(
            name="get_info",
            description="Returns server information",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Handle tool calls."""
    logger.info(f"Tool called: {name} with arguments: {arguments}")

    if name == "echo":
        text = arguments.get("text", "")
        return [TextContent(type="text", text=text)]

    elif name == "add":
        a = arguments.get("a", 0)
        b = arguments.get("b", 0)
        result = a + b
        return [TextContent(type="text", text=str(result))]

    elif name == "greet":
        name_arg = arguments.get("name", "World")
        greeting = f"Hello, {name_arg}!"
        return [TextContent(type="text", text=greeting)]

    elif name == "fail":
        message = arguments.get("message", "This tool always fails")
        # For errors, raise an exception instead of returning error result
        raise RuntimeError(message)

    elif name == "get_info":
        info = {
            "server_name": "mock-test-server",
            "version": "1.0.0",
            "tools_count": 5,
        }
        return [TextContent(type="text", text=json.dumps(info))]

    else:
        raise RuntimeError(f"Unknown tool: {name}")


async def main():
    """Run the MCP server."""
    logger.info("Starting mock MCP server...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
