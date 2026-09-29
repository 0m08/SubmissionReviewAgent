"""
Unit tests for the MCP client module.

These tests mock the MCP SDK to test client logic without real connections.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_integration.client import (
    MCPClient,
    MCPClientError,
    MCPConnectionError,
    MCPNotConnectedError,
)
from mcp_integration.config import ServerConfig


class TestMCPClientInit:
    """Tests for MCPClient initialization."""

    def test_init_with_config(self, sample_stdio_config):
        """Test client initialization with config."""
        client = MCPClient("test", sample_stdio_config)
        assert client.name == "test"
        assert client.config == sample_stdio_config
        assert client.is_connected is False

    def test_repr_disconnected(self, sample_stdio_config):
        """Test string representation when disconnected."""
        client = MCPClient("test", sample_stdio_config)
        repr_str = repr(client)
        assert "test" in repr_str
        assert "disconnected" in repr_str

    def test_is_connected_false_initially(self, sample_stdio_config):
        """Test is_connected is False before connect."""
        client = MCPClient("test", sample_stdio_config)
        assert client.is_connected is False


class TestMCPClientConnect:
    """Tests for MCPClient connection logic."""

    @pytest.mark.asyncio
    async def test_connect_stdio_success(self, sample_stdio_config):
        """Test successful stdio connection."""
        client = MCPClient("test", sample_stdio_config)

        # Mock the stdio_client and ClientSession
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()

        with patch("mcp_integration.client.stdio_client") as mock_stdio:
            with patch("mcp_integration.client.ClientSession") as mock_session_cls:
                # Set up context manager mocks
                mock_stdio.return_value.__aenter__ = AsyncMock(
                    return_value=(MagicMock(), MagicMock())
                )
                mock_stdio.return_value.__aexit__ = AsyncMock()

                mock_session_cls.return_value.__aenter__ = AsyncMock(
                    return_value=mock_session
                )
                mock_session_cls.return_value.__aexit__ = AsyncMock()

                await client.connect()

                assert client.is_connected is True
                mock_session.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_already_connected_warns(self, sample_stdio_config, caplog):
        """Test connecting when already connected logs warning."""
        client = MCPClient("test", sample_stdio_config)
        client._connected = True  # Simulate connected state

        await client.connect()

        assert "already connected" in caplog.text

    @pytest.mark.asyncio
    async def test_connect_failure_raises(self, sample_stdio_config):
        """Test connection failure raises MCPConnectionError."""
        client = MCPClient("test", sample_stdio_config)

        with patch("mcp_integration.client.stdio_client") as mock_stdio:
            mock_stdio.side_effect = Exception("Connection failed")

            with pytest.raises(MCPConnectionError, match="Connection failed"):
                await client.connect()

            assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_connect_unsupported_transport_raises(self):
        """Test unsupported transport raises ValueError."""
        # Create config with mocked invalid transport
        config = ServerConfig(
            name="test",
            transport="stdio",  # Valid for creation
            command="echo",
        )
        client = MCPClient("test", config)
        # Manually override transport to invalid value for test
        client.config = MagicMock()
        client.config.transport = "invalid"

        with pytest.raises(MCPConnectionError, match="Unsupported transport"):
            await client.connect()


class TestMCPClientDisconnect:
    """Tests for MCPClient disconnection."""

    @pytest.mark.asyncio
    async def test_disconnect_cleans_up(self, sample_stdio_config):
        """Test disconnect cleans up resources."""
        client = MCPClient("test", sample_stdio_config)
        client._connected = True
        client._session = MagicMock()
        mock_exit_stack = AsyncMock()
        mock_exit_stack.aclose = AsyncMock()
        client._exit_stack = mock_exit_stack

        await client.disconnect()

        assert client.is_connected is False
        assert client._session is None
        mock_exit_stack.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_safe_when_not_connected(self, sample_stdio_config):
        """Test disconnect is safe when not connected."""
        client = MCPClient("test", sample_stdio_config)

        # Should not raise
        await client.disconnect()
        assert client.is_connected is False


class TestMCPClientOperations:
    """Tests for MCPClient operations (list_tools, call_tool, etc.)."""

    @pytest.mark.asyncio
    async def test_list_tools_when_connected(self, sample_stdio_config, mock_tool):
        """Test list_tools when connected."""
        client = MCPClient("test", sample_stdio_config)
        client._connected = True

        mock_session = AsyncMock()
        mock_session.list_tools = AsyncMock(
            return_value=MagicMock(tools=[mock_tool])
        )
        client._session = mock_session

        tools = await client.list_tools()

        assert len(tools) == 1
        assert tools[0].name == "mock_tool"

    @pytest.mark.asyncio
    async def test_list_tools_when_not_connected_raises(self, sample_stdio_config):
        """Test list_tools raises when not connected."""
        client = MCPClient("test", sample_stdio_config)

        with pytest.raises(MCPNotConnectedError):
            await client.list_tools()

    @pytest.mark.asyncio
    async def test_call_tool_when_connected(
        self, sample_stdio_config, mock_tool_result
    ):
        """Test call_tool when connected."""
        client = MCPClient("test", sample_stdio_config)
        client._connected = True

        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=mock_tool_result)
        client._session = mock_session

        result = await client.call_tool("test_tool", {"arg": "value"})

        assert result == mock_tool_result
        mock_session.call_tool.assert_called_once_with("test_tool", {"arg": "value"})

    @pytest.mark.asyncio
    async def test_call_tool_with_no_args(self, sample_stdio_config, mock_tool_result):
        """Test call_tool with no arguments."""
        client = MCPClient("test", sample_stdio_config)
        client._connected = True

        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=mock_tool_result)
        client._session = mock_session

        await client.call_tool("test_tool")

        mock_session.call_tool.assert_called_once_with("test_tool", {})

    @pytest.mark.asyncio
    async def test_call_tool_when_not_connected_raises(self, sample_stdio_config):
        """Test call_tool raises when not connected."""
        client = MCPClient("test", sample_stdio_config)

        with pytest.raises(MCPNotConnectedError):
            await client.call_tool("test_tool", {})

    @pytest.mark.asyncio
    async def test_list_resources_when_connected(self, sample_stdio_config):
        """Test list_resources when connected."""
        from mcp.types import Resource

        mock_resource = Resource(
            uri="file:///test.txt",
            name="test.txt",
        )

        client = MCPClient("test", sample_stdio_config)
        client._connected = True

        mock_session = AsyncMock()
        mock_session.list_resources = AsyncMock(
            return_value=MagicMock(resources=[mock_resource])
        )
        client._session = mock_session

        resources = await client.list_resources()

        assert len(resources) == 1
        assert str(resources[0].uri) == "file:///test.txt"

    @pytest.mark.asyncio
    async def test_list_resources_when_not_connected_raises(self, sample_stdio_config):
        """Test list_resources raises when not connected."""
        client = MCPClient("test", sample_stdio_config)

        with pytest.raises(MCPNotConnectedError):
            await client.list_resources()

    @pytest.mark.asyncio
    async def test_read_resource_when_connected(self, sample_stdio_config):
        """Test read_resource when connected."""
        mock_result = MagicMock()

        client = MCPClient("test", sample_stdio_config)
        client._connected = True

        mock_session = AsyncMock()
        mock_session.read_resource = AsyncMock(return_value=mock_result)
        client._session = mock_session

        result = await client.read_resource("file:///test.txt")

        assert result == mock_result
        mock_session.read_resource.assert_called_once_with("file:///test.txt")


class TestMCPClientContextManager:
    """Tests for MCPClient as async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_connects_and_disconnects(self, sample_stdio_config):
        """Test context manager handles connect/disconnect."""
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()

        with patch("mcp_integration.client.stdio_client") as mock_stdio:
            with patch("mcp_integration.client.ClientSession") as mock_session_cls:
                mock_stdio.return_value.__aenter__ = AsyncMock(
                    return_value=(MagicMock(), MagicMock())
                )
                mock_stdio.return_value.__aexit__ = AsyncMock()

                mock_session_cls.return_value.__aenter__ = AsyncMock(
                    return_value=mock_session
                )
                mock_session_cls.return_value.__aexit__ = AsyncMock()

                async with MCPClient("test", sample_stdio_config) as client:
                    assert client.is_connected is True

                assert client.is_connected is False
