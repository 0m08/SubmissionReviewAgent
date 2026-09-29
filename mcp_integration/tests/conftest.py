"""
Pytest configuration and shared fixtures for MCP integration tests.
"""

import os
import sys
from pathlib import Path
from typing import Generator
from unittest.mock import AsyncMock, MagicMock

import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mcp_integration.config import ServerConfig


# =============================================================================
# Path Fixtures
# =============================================================================


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to the fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def test_config_path(fixtures_dir: Path) -> Path:
    """Path to test server configuration file."""
    return fixtures_dir / "test_servers.yaml"


@pytest.fixture
def nonexistent_config_path(tmp_path: Path) -> Path:
    """Path to a configuration file that doesn't exist."""
    return tmp_path / "does_not_exist.yaml"


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def sample_stdio_config() -> ServerConfig:
    """Sample stdio server configuration."""
    return ServerConfig(
        name="test-stdio",
        transport="stdio",
        command="python",
        args=["-m", "mcp_integration.tests.fixtures.mock_server"],
        enabled=True,
        description="Test stdio server",
    )


@pytest.fixture
def sample_http_config() -> ServerConfig:
    """Sample HTTP server configuration."""
    return ServerConfig(
        name="test-http",
        transport="streamable_http",
        url="http://localhost:8080/mcp",
        enabled=True,
        description="Test HTTP server",
    )


@pytest.fixture
def sample_disabled_config() -> ServerConfig:
    """Sample disabled server configuration."""
    return ServerConfig(
        name="test-disabled",
        transport="stdio",
        command="echo",
        args=["disabled"],
        enabled=False,
        description="Disabled test server",
    )


@pytest.fixture
def mock_server_config() -> ServerConfig:
    """Configuration for the actual mock server."""
    return ServerConfig(
        name="mock-server",
        transport="stdio",
        command="python",
        args=["-m", "mcp_integration.tests.fixtures.mock_server"],
        enabled=True,
        description="Mock MCP server for testing",
    )


# =============================================================================
# Environment Variable Fixtures
# =============================================================================


@pytest.fixture
def env_with_test_vars(monkeypatch) -> Generator[None, None, None]:
    """Set up environment variables for testing."""
    monkeypatch.setenv("TEST_COMMAND", "python")
    monkeypatch.setenv("TEST_ARG", "test_arg_value")
    monkeypatch.setenv("CUSTOM_VALUE", "custom_env_value")
    yield


@pytest.fixture
def clean_env(monkeypatch) -> Generator[None, None, None]:
    """Ensure certain env vars are not set."""
    monkeypatch.delenv("UNSET_VAR", raising=False)
    yield


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_client_session() -> MagicMock:
    """Create a mock MCP ClientSession."""
    session = MagicMock()
    session.initialize = AsyncMock()
    session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))
    session.list_resources = AsyncMock(return_value=MagicMock(resources=[]))
    session.call_tool = AsyncMock()
    session.read_resource = AsyncMock()
    return session


@pytest.fixture
def mock_tool():
    """Create a mock MCP Tool."""
    from mcp.types import Tool

    return Tool(
        name="mock_tool",
        description="A mock tool for testing",
        inputSchema={
            "type": "object",
            "properties": {
                "input": {"type": "string"},
            },
            "required": ["input"],
        },
    )


@pytest.fixture
def mock_tool_result():
    """Create a mock CallToolResult."""
    from mcp.types import CallToolResult, TextContent

    return CallToolResult(
        content=[TextContent(type="text", text="mock result")],
        isError=False,
    )


@pytest.fixture
def mock_tool_error():
    """Create a mock error CallToolResult."""
    from mcp.types import CallToolResult, TextContent

    return CallToolResult(
        content=[TextContent(type="text", text="mock error")],
        isError=True,
    )


# =============================================================================
# Temporary File Fixtures
# =============================================================================


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """Create a temporary configuration file."""
    config_content = """
servers:
  temp-server:
    enabled: true
    transport: stdio
    command: echo
    args: ["temp"]
    description: "Temporary test server"
"""
    config_path = tmp_path / "temp_servers.yaml"
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def invalid_yaml_file(tmp_path: Path) -> Path:
    """Create an invalid YAML file."""
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("{ invalid yaml: [")
    return config_path


@pytest.fixture
def empty_config_file(tmp_path: Path) -> Path:
    """Create an empty configuration file."""
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("")
    return config_path


@pytest.fixture
def config_missing_command(tmp_path: Path) -> Path:
    """Create config with missing required field."""
    config_content = """
servers:
  bad-server:
    enabled: true
    transport: stdio
    # command is missing
    args: ["test"]
"""
    config_path = tmp_path / "missing_command.yaml"
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def config_with_env_vars(tmp_path: Path) -> Path:
    """Create config with environment variable references."""
    config_content = """
servers:
  env-server:
    enabled: true
    transport: stdio
    command: ${TEST_COMMAND}
    args: ["${TEST_ARG}"]
    env:
      MY_VAR: ${CUSTOM_VALUE:-default_value}
"""
    config_path = tmp_path / "env_vars.yaml"
    config_path.write_text(config_content)
    return config_path


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (may be slow)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
