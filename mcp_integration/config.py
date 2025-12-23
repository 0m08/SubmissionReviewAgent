"""
Configuration module for MCP server definitions.

Handles loading, parsing, and validating MCP server configurations from YAML files.
Supports environment variable interpolation using ${VAR} syntax.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ServerConfig(BaseModel):
    """Configuration for a single MCP server."""

    name: str = Field(..., description="Unique identifier for this server")
    enabled: bool = Field(default=True, description="Whether this server is active")
    transport: Literal["stdio", "sse", "streamable_http"] = Field(
        ..., description="Transport mechanism to use"
    )

    # Stdio transport fields
    command: str | None = Field(
        default=None, description="Command to run for stdio transport"
    )
    args: list[str] = Field(
        default_factory=list, description="Arguments for the command"
    )

    # HTTP transport fields (SSE and Streamable HTTP)
    url: str | None = Field(default=None, description="URL for HTTP-based transports")

    # Common fields
    env: dict[str, str] = Field(
        default_factory=dict, description="Environment variables for the server"
    )
    description: str = Field(
        default="", description="Human-readable description of this server"
    )

    @model_validator(mode="after")
    def validate_transport_requirements(self) -> "ServerConfig":
        """Ensure required fields are present based on transport type."""
        if self.transport == "stdio":
            if not self.command:
                raise ValueError(
                    f"Server '{self.name}': 'command' is required for stdio transport"
                )
        elif self.transport in ("sse", "streamable_http"):
            if not self.url:
                raise ValueError(
                    f"Server '{self.name}': 'url' is required for {self.transport} transport"
                )
        return self


class MCPConfig(BaseModel):
    """Root configuration containing all MCP servers."""

    servers: dict[str, ServerConfig] = Field(
        default_factory=dict, description="Map of server name to configuration"
    )


def interpolate_env_vars(value: str) -> str:
    """
    Replace ${VAR} patterns with environment variable values.

    Args:
        value: String potentially containing ${VAR} patterns

    Returns:
        String with environment variables interpolated

    Raises:
        ValueError: If an environment variable is not set and has no default
    """
    pattern = r"\$\{([^}]+)\}"

    def replace(match: re.Match) -> str:
        var_expr = match.group(1)
        # Support ${VAR:-default} syntax
        if ":-" in var_expr:
            var_name, default = var_expr.split(":-", 1)
            return os.environ.get(var_name, default)
        else:
            var_name = var_expr
            value = os.environ.get(var_name)
            if value is None:
                raise ValueError(
                    f"Environment variable '{var_name}' is not set and has no default"
                )
            return value

    return re.sub(pattern, replace, value)


def interpolate_config_dict(data: dict | list | str) -> dict | list | str:
    """
    Recursively interpolate environment variables in a configuration structure.

    Args:
        data: Configuration data (dict, list, or string)

    Returns:
        Configuration with environment variables interpolated
    """
    if isinstance(data, dict):
        return {k: interpolate_config_dict(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [interpolate_config_dict(item) for item in data]
    elif isinstance(data, str):
        return interpolate_env_vars(data)
    else:
        return data


def load_server_configs(
    config_path: str | Path,
    interpolate_env: bool = True,
) -> dict[str, ServerConfig]:
    """
    Load MCP server configurations from a YAML file.

    Args:
        config_path: Path to the YAML configuration file
        interpolate_env: Whether to interpolate ${VAR} environment variables

    Returns:
        Dictionary mapping server names to their configurations

    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        ValueError: If the configuration is invalid
        yaml.YAMLError: If the YAML is malformed
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    if raw_config is None:
        return {}

    # Interpolate environment variables if requested
    if interpolate_env:
        raw_config = interpolate_config_dict(raw_config)

    # Handle both formats: {servers: {name: config}} and {servers: [{name: x, ...}]}
    servers_data = raw_config.get("servers", {})

    result: dict[str, ServerConfig] = {}

    if isinstance(servers_data, dict):
        # Format: servers: { server_name: { config } }
        for name, config in servers_data.items():
            if config is None:
                config = {}
            config["name"] = name
            result[name] = ServerConfig(**config)
    elif isinstance(servers_data, list):
        # Format: servers: [ { name: x, config } ]
        for config in servers_data:
            if "name" not in config:
                raise ValueError("Each server in list format must have a 'name' field")
            name = config["name"]
            result[name] = ServerConfig(**config)
    else:
        raise ValueError(
            f"'servers' must be a dict or list, got {type(servers_data).__name__}"
        )

    return result


def get_enabled_servers(
    config_path: str | Path,
    interpolate_env: bool = True,
) -> dict[str, ServerConfig]:
    """
    Load only enabled MCP server configurations.

    Convenience wrapper around load_server_configs that filters out disabled servers.

    Args:
        config_path: Path to the YAML configuration file
        interpolate_env: Whether to interpolate ${VAR} environment variables

    Returns:
        Dictionary mapping server names to their configurations (enabled only)
    """
    all_configs = load_server_configs(config_path, interpolate_env)
    return {name: config for name, config in all_configs.items() if config.enabled}
