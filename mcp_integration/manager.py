"""
MCP Manager for orchestrating multiple MCP server connections.

Provides a unified interface for managing multiple MCP servers,
discovering tools across all servers, and routing tool calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from mcp.types import CallToolResult, Tool, Resource

from mcp_integration.client import MCPClient, MCPClientError, MCPConnectionError
from mcp_integration.config import ServerConfig, get_enabled_servers

logger = logging.getLogger(__name__)


class ServerStatus(Enum):
    """Status of an MCP server connection."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class ServerInfo:
    """Information about a connected MCP server."""

    name: str
    config: ServerConfig
    status: ServerStatus
    tools: list[Tool] = field(default_factory=list)
    resources: list[Resource] = field(default_factory=list)
    error: str | None = None


class MCPManagerError(Exception):
    """Base exception for MCP manager errors."""

    pass


class ToolNotFoundError(MCPManagerError):
    """Raised when a requested tool is not found."""

    pass


class MCPManager:
    """
    Orchestrates multiple MCP server connections.

    Manages the lifecycle of multiple MCP servers, provides unified
    tool discovery, and routes tool calls to the appropriate server.

    Usage:
        manager = MCPManager("config/servers.yaml")
        await manager.start()

        # Get all tools
        tools = await manager.get_all_tools()

        # Call a tool (auto-routed to correct server)
        result = await manager.call_tool("read_file", {"path": "/tmp/test.txt"})

        await manager.shutdown()

    Or as async context manager:
        async with MCPManager("config/servers.yaml") as manager:
            tools = await manager.get_all_tools()
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        configs: dict[str, ServerConfig] | None = None,
    ):
        """
        Initialize MCP Manager.

        Args:
            config_path: Path to servers.yaml configuration file
            configs: Direct server configurations (alternative to config_path)

        Either config_path or configs must be provided.
        """
        self._config_path = Path(config_path) if config_path else None
        self._provided_configs = configs
        self._clients: dict[str, MCPClient] = {}
        self._server_info: dict[str, ServerInfo] = {}
        self._tool_index: dict[str, str] = {}  # tool_name -> server_name
        self._started = False

    @property
    def is_started(self) -> bool:
        """Check if manager has been started."""
        return self._started

    async def start(self) -> None:
        """
        Load configuration and connect to all enabled servers.

        Connects to each enabled server in parallel (where possible).
        Servers that fail to connect are marked with error status but
        don't prevent other servers from connecting.
        """
        if self._started:
            logger.warning("MCPManager is already started")
            return

        # Load configurations
        if self._provided_configs is not None:
            configs = self._provided_configs
        elif self._config_path:
            configs = get_enabled_servers(self._config_path)
        else:
            raise MCPManagerError(
                "No configuration provided. Pass config_path or configs."
            )

        if not configs:
            logger.warning("No enabled servers found in configuration")
            self._started = True
            return

        # Connect to each server
        for name, config in configs.items():
            await self._connect_server(name, config)

        # Build tool index from connected servers
        await self._build_tool_index()

        self._started = True
        connected_count = sum(
            1
            for info in self._server_info.values()
            if info.status == ServerStatus.CONNECTED
        )
        logger.info(
            f"MCPManager started: {connected_count}/{len(configs)} servers connected"
        )

    async def _connect_server(self, name: str, config: ServerConfig) -> None:
        """Connect to a single server, handling errors gracefully."""
        info = ServerInfo(
            name=name,
            config=config,
            status=ServerStatus.CONNECTING,
        )
        self._server_info[name] = info

        try:
            client = MCPClient(name, config)
            await client.connect()
            self._clients[name] = client

            # Fetch tools and resources
            info.tools = await client.list_tools()
            try:
                info.resources = await client.list_resources()
            except Exception:
                # Some servers may not support resources
                info.resources = []

            info.status = ServerStatus.CONNECTED
            logger.info(
                f"Server '{name}' connected with {len(info.tools)} tools, "
                f"{len(info.resources)} resources"
            )

        except MCPConnectionError as e:
            info.status = ServerStatus.ERROR
            info.error = str(e)
            logger.error(f"Failed to connect to server '{name}': {e}")

        except Exception as e:
            info.status = ServerStatus.ERROR
            info.error = str(e)
            logger.error(f"Unexpected error connecting to server '{name}': {e}")

    async def _build_tool_index(self) -> None:
        """Build index mapping tool names to server names."""
        self._tool_index.clear()

        for name, info in self._server_info.items():
            if info.status != ServerStatus.CONNECTED:
                continue

            for tool in info.tools:
                if tool.name in self._tool_index:
                    existing_server = self._tool_index[tool.name]
                    logger.warning(
                        f"Tool '{tool.name}' exists in multiple servers: "
                        f"'{existing_server}' and '{name}'. Using '{name}'."
                    )
                self._tool_index[tool.name] = name

    async def shutdown(self) -> None:
        """
        Disconnect from all servers and clean up resources.

        Safe to call multiple times.
        """
        for name, client in list(self._clients.items()):
            try:
                await client.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting from server '{name}': {e}")

        self._clients.clear()
        self._tool_index.clear()

        for info in self._server_info.values():
            info.status = ServerStatus.DISCONNECTED

        self._started = False
        logger.info("MCPManager shutdown complete")

    def get_server_status(self) -> dict[str, ServerInfo]:
        """
        Get status information for all configured servers.

        Returns:
            Dictionary mapping server names to their ServerInfo
        """
        return dict(self._server_info)

    async def get_all_tools(self) -> dict[str, list[Tool]]:
        """
        Get tools from all connected servers.

        Returns:
            Dictionary mapping server names to their list of tools
        """
        result = {}
        for name, info in self._server_info.items():
            if info.status == ServerStatus.CONNECTED:
                result[name] = info.tools
        return result

    def get_tool_list(self) -> list[tuple[str, str, Tool]]:
        """
        Get flat list of all tools with their server names.

        Returns:
            List of (server_name, tool_name, Tool) tuples
        """
        result = []
        for name, info in self._server_info.items():
            if info.status == ServerStatus.CONNECTED:
                for tool in info.tools:
                    result.append((name, tool.name, tool))
        return result

    def find_tool_server(self, tool_name: str) -> str | None:
        """
        Find which server provides a given tool.

        Args:
            tool_name: Name of the tool to find

        Returns:
            Server name or None if tool not found
        """
        return self._tool_index.get(tool_name)

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        server_name: str | None = None,
    ) -> CallToolResult:
        """
        Execute a tool, automatically routing to the correct server.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            server_name: Optionally specify which server to use

        Returns:
            CallToolResult from the tool execution

        Raises:
            ToolNotFoundError: If tool is not found on any server
            MCPClientError: If tool execution fails
        """
        # Determine which server to use
        if server_name:
            target_server = server_name
        else:
            target_server = self._tool_index.get(tool_name)

        if not target_server:
            available = list(self._tool_index.keys())
            raise ToolNotFoundError(
                f"Tool '{tool_name}' not found. Available tools: {available}"
            )

        if target_server not in self._clients:
            raise MCPClientError(f"Server '{target_server}' is not connected")

        client = self._clients[target_server]
        logger.debug(f"Calling tool '{tool_name}' on server '{target_server}'")

        return await client.call_tool(tool_name, arguments)

    async def call_tool_on_server(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> CallToolResult:
        """
        Execute a tool on a specific server.

        Args:
            server_name: Name of the server to use
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            CallToolResult from the tool execution

        Raises:
            MCPClientError: If server not connected or tool execution fails
        """
        if server_name not in self._clients:
            raise MCPClientError(f"Server '{server_name}' is not connected")

        client = self._clients[server_name]
        return await client.call_tool(tool_name, arguments)

    async def get_all_resources(self) -> dict[str, list[Resource]]:
        """
        Get resources from all connected servers.

        Returns:
            Dictionary mapping server names to their list of resources
        """
        result = {}
        for name, info in self._server_info.items():
            if info.status == ServerStatus.CONNECTED:
                result[name] = info.resources
        return result

    async def read_resource(
        self, uri: str, server_name: str | None = None
    ) -> Any:
        """
        Read a resource by URI.

        Args:
            uri: Resource URI to read
            server_name: Server to read from (required if URI doesn't identify server)

        Returns:
            Resource content

        Raises:
            MCPClientError: If server not specified and can't be determined
        """
        if not server_name:
            # Try to find server with this resource
            for name, info in self._server_info.items():
                if info.status == ServerStatus.CONNECTED:
                    for resource in info.resources:
                        if resource.uri == uri:
                            server_name = name
                            break
                if server_name:
                    break

        if not server_name:
            raise MCPClientError(
                f"Cannot determine server for resource '{uri}'. Specify server_name."
            )

        if server_name not in self._clients:
            raise MCPClientError(f"Server '{server_name}' is not connected")

        client = self._clients[server_name]
        return await client.read_resource(uri)

    async def refresh_tools(self) -> None:
        """
        Refresh tool lists from all connected servers.

        Useful if servers have added/removed tools dynamically.
        """
        for name, client in self._clients.items():
            try:
                info = self._server_info[name]
                info.tools = await client.list_tools()
                logger.debug(f"Refreshed tools for server '{name}'")
            except Exception as e:
                logger.error(f"Failed to refresh tools for server '{name}': {e}")

        await self._build_tool_index()

    async def __aenter__(self) -> "MCPManager":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.shutdown()

    def __repr__(self) -> str:
        if not self._started:
            return "MCPManager(not started)"
        connected = sum(
            1
            for info in self._server_info.values()
            if info.status == ServerStatus.CONNECTED
        )
        total = len(self._server_info)
        return f"MCPManager(servers={connected}/{total}, tools={len(self._tool_index)})"
