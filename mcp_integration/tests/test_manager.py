"""
Unit tests for the MCP Manager module.

These tests mock the MCPClient to test manager logic without real connections.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp.types import Tool, CallToolResult, TextContent

from mcp_integration.manager import (
    MCPManager,
    MCPManagerError,
    ToolNotFoundError,
    ServerStatus,
    ServerInfo,
)
from mcp_integration.config import ServerConfig


@pytest.fixture
def mock_tools():
    """Create mock tools for testing."""
    return [
        Tool(
            name="tool1",
            description="First tool",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="tool2",
            description="Second tool",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@pytest.fixture
def mock_call_result():
    """Create mock tool call result."""
    return CallToolResult(
        content=[TextContent(type="text", text="result")],
        isError=False,
    )


@pytest.fixture
def sample_configs():
    """Create sample server configurations."""
    return {
        "server1": ServerConfig(
            name="server1",
            transport="stdio",
            command="echo",
            args=["test"],
            enabled=True,
        ),
        "server2": ServerConfig(
            name="server2",
            transport="stdio",
            command="echo",
            args=["test2"],
            enabled=True,
        ),
    }


class TestMCPManagerInit:
    """Tests for MCPManager initialization."""

    def test_init_with_config_path(self, test_config_path):
        """Test initialization with config path."""
        manager = MCPManager(config_path=test_config_path)
        assert manager._config_path == test_config_path
        assert manager.is_started is False

    def test_init_with_configs(self, sample_configs):
        """Test initialization with direct configs."""
        manager = MCPManager(configs=sample_configs)
        assert manager._provided_configs == sample_configs
        assert manager.is_started is False

    def test_repr_not_started(self):
        """Test repr when not started."""
        manager = MCPManager(configs={})
        assert "not started" in repr(manager)


class TestMCPManagerStart:
    """Tests for MCPManager start logic."""

    @pytest.mark.asyncio
    async def test_start_with_no_config_raises(self):
        """Test start without config raises error."""
        manager = MCPManager()

        with pytest.raises(MCPManagerError, match="No configuration"):
            await manager.start()

    @pytest.mark.asyncio
    async def test_start_already_started_warns(self, sample_configs, caplog):
        """Test starting when already started logs warning."""
        manager = MCPManager(configs=sample_configs)
        manager._started = True

        await manager.start()

        assert "already started" in caplog.text

    @pytest.mark.asyncio
    async def test_start_empty_config(self, caplog):
        """Test start with empty config."""
        manager = MCPManager(configs={})

        await manager.start()

        assert manager.is_started is True
        assert "No enabled servers" in caplog.text

    @pytest.mark.asyncio
    async def test_start_connects_servers(self, sample_configs, mock_tools):
        """Test start connects to all enabled servers."""
        manager = MCPManager(configs=sample_configs)

        with patch("mcp_integration.manager.MCPClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.list_tools = AsyncMock(return_value=mock_tools)
            mock_client.list_resources = AsyncMock(return_value=[])
            mock_client_cls.return_value = mock_client

            await manager.start()

            assert manager.is_started is True
            assert mock_client_cls.call_count == 2  # Two servers
            assert mock_client.connect.call_count == 2

    @pytest.mark.asyncio
    async def test_start_handles_connection_failure(self, sample_configs):
        """Test start handles individual server failures."""
        manager = MCPManager(configs=sample_configs)

        with patch("mcp_integration.manager.MCPClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock(side_effect=Exception("Connection failed"))
            mock_client_cls.return_value = mock_client

            await manager.start()

            # Manager should still be started
            assert manager.is_started is True
            # Servers should be marked as error
            for info in manager._server_info.values():
                assert info.status == ServerStatus.ERROR

    @pytest.mark.asyncio
    async def test_start_builds_tool_index(self, mock_tools):
        """Test start builds tool index correctly."""
        configs = {
            "server1": ServerConfig(
                name="server1",
                transport="stdio",
                command="echo",
                enabled=True,
            ),
        }
        manager = MCPManager(configs=configs)

        with patch("mcp_integration.manager.MCPClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_client.list_tools = AsyncMock(return_value=mock_tools)
            mock_client.list_resources = AsyncMock(return_value=[])
            mock_client_cls.return_value = mock_client

            await manager.start()

            assert "tool1" in manager._tool_index
            assert "tool2" in manager._tool_index
            assert manager._tool_index["tool1"] == "server1"


class TestMCPManagerShutdown:
    """Tests for MCPManager shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_disconnects_all(self, sample_configs):
        """Test shutdown disconnects all clients."""
        manager = MCPManager(configs=sample_configs)

        mock_client1 = AsyncMock()
        mock_client2 = AsyncMock()
        manager._clients = {"server1": mock_client1, "server2": mock_client2}
        manager._started = True

        await manager.shutdown()

        mock_client1.disconnect.assert_called_once()
        mock_client2.disconnect.assert_called_once()
        assert manager.is_started is False
        assert len(manager._clients) == 0

    @pytest.mark.asyncio
    async def test_shutdown_handles_disconnect_error(self, sample_configs, caplog):
        """Test shutdown handles disconnect errors gracefully."""
        manager = MCPManager(configs=sample_configs)

        mock_client = AsyncMock()
        mock_client.disconnect = AsyncMock(side_effect=Exception("Disconnect error"))
        manager._clients = {"server1": mock_client}
        manager._started = True

        await manager.shutdown()

        assert "Error disconnecting" in caplog.text
        assert manager.is_started is False


class TestMCPManagerToolOperations:
    """Tests for MCPManager tool operations."""

    @pytest.mark.asyncio
    async def test_get_all_tools(self, mock_tools):
        """Test get_all_tools returns tools from all servers."""
        manager = MCPManager(configs={})
        manager._server_info = {
            "server1": ServerInfo(
                name="server1",
                config=MagicMock(),
                status=ServerStatus.CONNECTED,
                tools=mock_tools,
            ),
            "server2": ServerInfo(
                name="server2",
                config=MagicMock(),
                status=ServerStatus.DISCONNECTED,
                tools=[],  # Disconnected, should be excluded
            ),
        }

        tools = await manager.get_all_tools()

        assert "server1" in tools
        assert "server2" not in tools  # Disconnected
        assert len(tools["server1"]) == 2

    def test_get_tool_list(self, mock_tools):
        """Test get_tool_list returns flat list."""
        manager = MCPManager(configs={})
        manager._server_info = {
            "server1": ServerInfo(
                name="server1",
                config=MagicMock(),
                status=ServerStatus.CONNECTED,
                tools=mock_tools,
            ),
        }

        tool_list = manager.get_tool_list()

        assert len(tool_list) == 2
        assert tool_list[0][0] == "server1"  # server name
        assert tool_list[0][1] == "tool1"  # tool name

    def test_find_tool_server(self):
        """Test find_tool_server returns correct server."""
        manager = MCPManager(configs={})
        manager._tool_index = {"tool1": "server1", "tool2": "server2"}

        assert manager.find_tool_server("tool1") == "server1"
        assert manager.find_tool_server("tool2") == "server2"
        assert manager.find_tool_server("unknown") is None

    @pytest.mark.asyncio
    async def test_call_tool_routes_correctly(self, mock_call_result):
        """Test call_tool routes to correct server."""
        manager = MCPManager(configs={})
        manager._tool_index = {"my_tool": "server1"}

        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value=mock_call_result)
        manager._clients = {"server1": mock_client}

        result = await manager.call_tool("my_tool", {"arg": "value"})

        assert result == mock_call_result
        mock_client.call_tool.assert_called_once_with("my_tool", {"arg": "value"})

    @pytest.mark.asyncio
    async def test_call_tool_unknown_tool_raises(self):
        """Test call_tool raises for unknown tool."""
        manager = MCPManager(configs={})
        manager._tool_index = {}

        with pytest.raises(ToolNotFoundError, match="not found"):
            await manager.call_tool("unknown_tool", {})

    @pytest.mark.asyncio
    async def test_call_tool_with_explicit_server(self, mock_call_result):
        """Test call_tool with explicit server name."""
        manager = MCPManager(configs={})
        manager._tool_index = {}  # Empty index

        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value=mock_call_result)
        manager._clients = {"server1": mock_client}

        result = await manager.call_tool(
            "any_tool", {"arg": "value"}, server_name="server1"
        )

        assert result == mock_call_result

    @pytest.mark.asyncio
    async def test_call_tool_on_server(self, mock_call_result):
        """Test call_tool_on_server."""
        manager = MCPManager(configs={})

        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value=mock_call_result)
        manager._clients = {"server1": mock_client}

        result = await manager.call_tool_on_server(
            "server1", "tool_name", {"arg": "value"}
        )

        assert result == mock_call_result

    @pytest.mark.asyncio
    async def test_call_tool_on_server_not_connected_raises(self):
        """Test call_tool_on_server raises for disconnected server."""
        manager = MCPManager(configs={})
        manager._clients = {}

        with pytest.raises(Exception, match="not connected"):
            await manager.call_tool_on_server("server1", "tool", {})


class TestMCPManagerResourceOperations:
    """Tests for MCPManager resource operations."""

    @pytest.mark.asyncio
    async def test_get_all_resources(self):
        """Test get_all_resources returns resources from connected servers."""
        from mcp.types import Resource

        mock_resource = Resource(uri="file:///test.txt", name="test.txt")

        manager = MCPManager(configs={})
        manager._server_info = {
            "server1": ServerInfo(
                name="server1",
                config=MagicMock(),
                status=ServerStatus.CONNECTED,
                resources=[mock_resource],
            ),
        }

        resources = await manager.get_all_resources()

        assert "server1" in resources
        assert len(resources["server1"]) == 1


class TestMCPManagerContextManager:
    """Tests for MCPManager as async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test context manager starts and shuts down."""
        manager = MCPManager(configs={})

        async with manager:
            assert manager.is_started is True

        assert manager.is_started is False


class TestMCPManagerDuplicateTools:
    """Tests for handling duplicate tool names across servers."""

    @pytest.mark.asyncio
    async def test_duplicate_tool_warns(self, caplog):
        """Test duplicate tool names log warning."""
        tool = Tool(
            name="shared_tool",
            description="Shared",
            inputSchema={"type": "object"},
        )

        manager = MCPManager(configs={})
        manager._server_info = {
            "server1": ServerInfo(
                name="server1",
                config=MagicMock(),
                status=ServerStatus.CONNECTED,
                tools=[tool],
            ),
            "server2": ServerInfo(
                name="server2",
                config=MagicMock(),
                status=ServerStatus.CONNECTED,
                tools=[tool],
            ),
        }

        await manager._build_tool_index()

        assert "exists in multiple servers" in caplog.text


class TestMCPManagerRefreshTools:
    """Tests for refreshing tools."""

    @pytest.mark.asyncio
    async def test_refresh_tools(self, mock_tools):
        """Test refresh_tools updates tool list."""
        manager = MCPManager(configs={})

        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=mock_tools)
        manager._clients = {"server1": mock_client}
        manager._server_info = {
            "server1": ServerInfo(
                name="server1",
                config=MagicMock(),
                status=ServerStatus.CONNECTED,
                tools=[],
            ),
        }

        await manager.refresh_tools()

        assert len(manager._server_info["server1"].tools) == 2
        mock_client.list_tools.assert_called_once()
