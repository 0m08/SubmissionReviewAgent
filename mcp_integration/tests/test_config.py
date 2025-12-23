"""
Unit tests for the configuration module.
"""

import os
from pathlib import Path

import pytest
import yaml

from mcp_integration.config import (
    ServerConfig,
    MCPConfig,
    interpolate_env_vars,
    interpolate_config_dict,
    load_server_configs,
    get_enabled_servers,
)


class TestServerConfig:
    """Tests for ServerConfig validation."""

    def test_valid_stdio_config(self):
        """Test creating valid stdio server config."""
        config = ServerConfig(
            name="test",
            transport="stdio",
            command="python",
            args=["-m", "some_module"],
        )
        assert config.name == "test"
        assert config.transport == "stdio"
        assert config.command == "python"
        assert config.enabled is True  # default

    def test_valid_http_config(self):
        """Test creating valid HTTP server config."""
        config = ServerConfig(
            name="test-http",
            transport="streamable_http",
            url="http://localhost:8080/mcp",
        )
        assert config.name == "test-http"
        assert config.transport == "streamable_http"
        assert config.url == "http://localhost:8080/mcp"

    def test_valid_sse_config(self):
        """Test creating valid SSE server config."""
        config = ServerConfig(
            name="test-sse",
            transport="sse",
            url="http://localhost:8080/sse",
        )
        assert config.transport == "sse"

    def test_stdio_missing_command_raises(self):
        """Test that stdio transport requires command."""
        with pytest.raises(ValueError, match="'command' is required"):
            ServerConfig(
                name="bad",
                transport="stdio",
            )

    def test_http_missing_url_raises(self):
        """Test that HTTP transport requires url."""
        with pytest.raises(ValueError, match="'url' is required"):
            ServerConfig(
                name="bad",
                transport="streamable_http",
            )

    def test_sse_missing_url_raises(self):
        """Test that SSE transport requires url."""
        with pytest.raises(ValueError, match="'url' is required"):
            ServerConfig(
                name="bad",
                transport="sse",
            )

    def test_invalid_transport_rejected(self):
        """Test that invalid transport type is rejected."""
        with pytest.raises(ValueError):
            ServerConfig(
                name="bad",
                transport="invalid_transport",  # type: ignore
                command="echo",
            )

    def test_disabled_server(self):
        """Test creating disabled server config."""
        config = ServerConfig(
            name="disabled",
            transport="stdio",
            command="echo",
            enabled=False,
        )
        assert config.enabled is False

    def test_with_env_vars(self):
        """Test config with environment variables."""
        config = ServerConfig(
            name="with-env",
            transport="stdio",
            command="python",
            env={"API_KEY": "secret123", "DEBUG": "true"},
        )
        assert config.env["API_KEY"] == "secret123"

    def test_with_description(self):
        """Test config with description."""
        config = ServerConfig(
            name="described",
            transport="stdio",
            command="echo",
            description="A test server",
        )
        assert config.description == "A test server"


class TestInterpolateEnvVars:
    """Tests for environment variable interpolation."""

    def test_simple_interpolation(self, monkeypatch):
        """Test basic ${VAR} interpolation."""
        monkeypatch.setenv("TEST_VAR", "test_value")
        result = interpolate_env_vars("prefix_${TEST_VAR}_suffix")
        assert result == "prefix_test_value_suffix"

    def test_multiple_vars(self, monkeypatch):
        """Test multiple variables in one string."""
        monkeypatch.setenv("VAR1", "one")
        monkeypatch.setenv("VAR2", "two")
        result = interpolate_env_vars("${VAR1} and ${VAR2}")
        assert result == "one and two"

    def test_default_value(self, monkeypatch):
        """Test ${VAR:-default} syntax."""
        monkeypatch.delenv("UNSET_VAR", raising=False)
        result = interpolate_env_vars("${UNSET_VAR:-default_value}")
        assert result == "default_value"

    def test_default_when_var_set(self, monkeypatch):
        """Test that default is not used when var is set."""
        monkeypatch.setenv("SET_VAR", "actual_value")
        result = interpolate_env_vars("${SET_VAR:-default}")
        assert result == "actual_value"

    def test_missing_var_raises(self, monkeypatch):
        """Test that missing var without default raises."""
        monkeypatch.delenv("MISSING_VAR", raising=False)
        with pytest.raises(ValueError, match="not set"):
            interpolate_env_vars("${MISSING_VAR}")

    def test_no_interpolation_needed(self):
        """Test string without variables."""
        result = interpolate_env_vars("plain string")
        assert result == "plain string"

    def test_empty_default(self, monkeypatch):
        """Test empty default value."""
        monkeypatch.delenv("UNSET", raising=False)
        result = interpolate_env_vars("${UNSET:-}")
        assert result == ""


class TestInterpolateConfigDict:
    """Tests for recursive config interpolation."""

    def test_dict_interpolation(self, monkeypatch):
        """Test interpolation in nested dict."""
        monkeypatch.setenv("VAR", "value")
        data = {"key": "${VAR}", "nested": {"inner": "${VAR}"}}
        result = interpolate_config_dict(data)
        assert result["key"] == "value"
        assert result["nested"]["inner"] == "value"

    def test_list_interpolation(self, monkeypatch):
        """Test interpolation in lists."""
        monkeypatch.setenv("VAR", "value")
        data = ["${VAR}", "plain", "${VAR}"]
        result = interpolate_config_dict(data)
        assert result == ["value", "plain", "value"]

    def test_mixed_types(self, monkeypatch):
        """Test non-string values pass through."""
        monkeypatch.setenv("VAR", "value")
        data = {
            "string": "${VAR}",
            "number": 42,
            "bool": True,
            "none": None,
        }
        result = interpolate_config_dict(data)
        assert result["string"] == "value"
        assert result["number"] == 42
        assert result["bool"] is True
        assert result["none"] is None


class TestLoadServerConfigs:
    """Tests for loading configurations from YAML files."""

    def test_load_valid_config(self, test_config_path):
        """Test loading a valid configuration file."""
        configs = load_server_configs(test_config_path)
        assert "mock-server" in configs
        assert configs["mock-server"].transport == "stdio"
        assert configs["mock-server"].enabled is True

    def test_load_includes_disabled(self, test_config_path):
        """Test that disabled servers are included in load."""
        configs = load_server_configs(test_config_path)
        assert "disabled-server" in configs
        assert configs["disabled-server"].enabled is False

    def test_missing_file_raises(self, nonexistent_config_path):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_server_configs(nonexistent_config_path)

    def test_empty_file_returns_empty(self, empty_config_file):
        """Test that empty file returns empty dict."""
        configs = load_server_configs(empty_config_file)
        assert configs == {}

    def test_invalid_yaml_raises(self, invalid_yaml_file):
        """Test that invalid YAML raises error."""
        with pytest.raises(yaml.YAMLError):
            load_server_configs(invalid_yaml_file)

    def test_missing_command_raises(self, config_missing_command):
        """Test that missing required field raises."""
        with pytest.raises(ValueError, match="'command' is required"):
            load_server_configs(config_missing_command)

    def test_env_interpolation(self, config_with_env_vars, monkeypatch):
        """Test environment variable interpolation during load."""
        monkeypatch.setenv("TEST_COMMAND", "python")
        monkeypatch.setenv("TEST_ARG", "my_arg")
        monkeypatch.setenv("CUSTOM_VALUE", "custom")

        configs = load_server_configs(config_with_env_vars)
        assert configs["env-server"].command == "python"
        assert configs["env-server"].args == ["my_arg"]
        assert configs["env-server"].env["MY_VAR"] == "custom"

    def test_skip_interpolation(self, config_with_env_vars, monkeypatch):
        """Test skipping interpolation preserves ${VAR} syntax."""
        monkeypatch.setenv("TEST_COMMAND", "python")
        monkeypatch.setenv("TEST_ARG", "arg")

        configs = load_server_configs(config_with_env_vars, interpolate_env=False)
        assert configs["env-server"].command == "${TEST_COMMAND}"

    def test_dict_format(self, tmp_path):
        """Test loading dict-format configuration."""
        config_content = """
servers:
  server1:
    transport: stdio
    command: echo
    args: ["test"]
  server2:
    transport: streamable_http
    url: http://localhost:8080
"""
        config_path = tmp_path / "dict_format.yaml"
        config_path.write_text(config_content)

        configs = load_server_configs(config_path)
        assert "server1" in configs
        assert "server2" in configs
        assert configs["server1"].name == "server1"

    def test_list_format(self, tmp_path):
        """Test loading list-format configuration."""
        config_content = """
servers:
  - name: server1
    transport: stdio
    command: echo
  - name: server2
    transport: streamable_http
    url: http://localhost:8080
"""
        config_path = tmp_path / "list_format.yaml"
        config_path.write_text(config_content)

        configs = load_server_configs(config_path)
        assert "server1" in configs
        assert "server2" in configs

    def test_list_format_missing_name_raises(self, tmp_path):
        """Test that list format requires name field."""
        config_content = """
servers:
  - transport: stdio
    command: echo
"""
        config_path = tmp_path / "no_name.yaml"
        config_path.write_text(config_content)

        with pytest.raises(ValueError, match="must have a 'name' field"):
            load_server_configs(config_path)


class TestGetEnabledServers:
    """Tests for get_enabled_servers function."""

    def test_filters_disabled(self, test_config_path):
        """Test that disabled servers are filtered out."""
        configs = get_enabled_servers(test_config_path)
        assert "mock-server" in configs
        assert "disabled-server" not in configs

    def test_empty_when_all_disabled(self, tmp_path):
        """Test returns empty when all servers disabled."""
        config_content = """
servers:
  server1:
    enabled: false
    transport: stdio
    command: echo
"""
        config_path = tmp_path / "all_disabled.yaml"
        config_path.write_text(config_content)

        configs = get_enabled_servers(config_path)
        assert configs == {}
