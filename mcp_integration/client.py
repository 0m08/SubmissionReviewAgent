"""
MCP Client wrapper for connecting to individual MCP servers.

Provides a clean interface for managing MCP server connections,
discovering tools/resources, and invoking operations.
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import (
    CallToolResult,
    ListToolsResult,
    ListResourcesResult,
    ReadResourceResult,
    Tool,
    Resource,
)

from mcp_integration.config import ServerConfig

if TYPE_CHECKING:
    from mcp.client.session import ClientSession

logger = logging.getLogger(__name__)


class MCPClientError(Exception):
    """Base exception for MCP client errors."""

    pass


class MCPConnectionError(MCPClientError):
    """Raised when connection to MCP server fails."""

    pass


class MCPNotConnectedError(MCPClientError):
    """Raised when trying to use client before connecting."""

    pass


class MCPClient:
    """
    Wrapper for connecting to and interacting with a single MCP server.

    Handles connection lifecycle, tool discovery, and tool invocation
    for one MCP server connection.

    Usage:
        client = MCPClient("my-server", config)
        await client.connect()
        tools = await client.list_tools()
        result = await client.call_tool("tool_name", {"arg": "value"})
        await client.disconnect()

    Or as async context manager:
        async with MCPClient("my-server", config) as client:
            tools = await client.list_tools()
    """

    def __init__(self, name: str, config: ServerConfig):
        """
        Initialize MCP client.

        Args:
            name: Identifier for this server connection
            config: Server configuration including transport details
        """
        self.name = name
        self.config = config
        self._session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if client is currently connected."""
        return self._connected and self._session is not None

    async def connect(self) -> None:
        """
        Establish connection to the MCP server.

        Raises:
            MCPConnectionError: If connection fails
            ValueError: If transport type is not supported
        """
        if self._connected:
            logger.warning(f"Client '{self.name}' is already connected")
            return

        self._exit_stack = AsyncExitStack()

        try:
            if self.config.transport == "stdio":
                await self._connect_stdio()
            elif self.config.transport == "sse":
                await self._connect_sse()
            elif self.config.transport == "streamable_http":
                await self._connect_streamable_http()
            else:
                raise ValueError(f"Unsupported transport: {self.config.transport}")

            # Initialize the session (required MCP handshake)
            await self._session.initialize()
            self._connected = True
            logger.info(f"Connected to MCP server '{self.name}'")

        except Exception as e:
            # Clean up on failure
            if self._exit_stack:
                await self._exit_stack.aclose()
                self._exit_stack = None
            self._session = None
            self._connected = False
            raise MCPConnectionError(
                f"Failed to connect to server '{self.name}': {e}"
            ) from e

    async def _connect_stdio(self) -> None:
        """Connect using stdio transport."""
        server_params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
            env={**self.config.env} if self.config.env else None,
        )

        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read_stream, write_stream = stdio_transport

        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

    async def _connect_sse(self) -> None:
        """Connect using SSE transport."""
        # Import here to avoid dependency if not used
        from mcp.client.sse import sse_client

        sse_transport = await self._exit_stack.enter_async_context(
            sse_client(self.config.url)
        )
        read_stream, write_stream = sse_transport

        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

    async def _connect_streamable_http(self) -> None:
        """Connect using Streamable HTTP transport."""
        # Import here to avoid dependency if not used
        from mcp.client.streamable_http import streamablehttp_client

        http_transport = await self._exit_stack.enter_async_context(
            streamablehttp_client(self.config.url)
        )
        read_stream, write_stream, _ = http_transport

        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

    async def disconnect(self) -> None:
        """
        Close connection to the MCP server.

        Safe to call multiple times.
        """
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
        self._session = None
        self._connected = False
        logger.info(f"Disconnected from MCP server '{self.name}'")

    def _ensure_connected(self) -> None:
        """Raise if not connected."""
        if not self.is_connected:
            raise MCPNotConnectedError(
                f"Client '{self.name}' is not connected. Call connect() first."
            )

    async def list_tools(self) -> list[Tool]:
        """
        Get list of available tools from the server.

        Returns:
            List of Tool objects with name, description, and input schema

        Raises:
            MCPNotConnectedError: If not connected
        """
        self._ensure_connected()
        result: ListToolsResult = await self._session.list_tools()
        return list(result.tools)

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> CallToolResult:
        """
        Execute a tool on the MCP server.

        Args:
            name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            CallToolResult containing the tool's output or error

        Raises:
            MCPNotConnectedError: If not connected
        """
        self._ensure_connected()
        result = await self._session.call_tool(name, arguments or {})
        return result

    async def list_resources(self) -> list[Resource]:
        """
        Get list of available resources from the server.

        Returns:
            List of Resource objects with URI, name, and metadata

        Raises:
            MCPNotConnectedError: If not connected
        """
        self._ensure_connected()
        result: ListResourcesResult = await self._session.list_resources()
        return list(result.resources)

    async def read_resource(self, uri: str) -> ReadResourceResult:
        """
        Read a specific resource by URI.

        Args:
            uri: Resource URI to read

        Returns:
            ReadResourceResult containing the resource content

        Raises:
            MCPNotConnectedError: If not connected
        """
        self._ensure_connected()
        result = await self._session.read_resource(uri)
        return result

    async def __aenter__(self) -> "MCPClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()

    def __repr__(self) -> str:
        status = "connected" if self.is_connected else "disconnected"
        return f"MCPClient(name={self.name!r}, transport={self.config.transport!r}, status={status})"
