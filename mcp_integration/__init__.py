"""
MCP Integration Module

A standalone module for integrating Model Context Protocol (MCP) servers
into Python applications. Provides configuration-based multi-server management.
"""

from mcp_integration.config import load_server_configs, ServerConfig
from mcp_integration.client import MCPClient
from mcp_integration.manager import MCPManager

__all__ = [
    "MCPManager",
    "MCPClient",
    "ServerConfig",
    "load_server_configs",
]
