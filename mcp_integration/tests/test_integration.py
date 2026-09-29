"""
Integration tests for MCP client and manager.

These tests actually connect to the mock MCP server to verify
end-to-end functionality. They are marked with @pytest.mark.integration
and can be skipped in fast test runs.
"""

import pytest

from mcp_integration.client import MCPClient
from mcp_integration.manager import MCPManager, ServerStatus
from mcp_integration.config import ServerConfig


# Skip all tests in this module if MCP package is not available
pytest.importorskip("mcp")


@pytest.fixture
def mock_server_config() -> ServerConfig:
    """Configuration for the mock MCP server."""
    return ServerConfig(
        name="mock-server",
        transport="stdio",
        command="python",
        args=["-m", "mcp_integration.tests.fixtures.mock_server"],
        enabled=True,
        description="Mock MCP server for integration testing",
    )


@pytest.mark.integration
class TestMCPClientIntegration:
    """Integration tests for MCPClient with real mock server."""

    @pytest.mark.asyncio
    async def test_connect_to_mock_server(self, mock_server_config):
        """Test connecting to the mock MCP server."""
        client = MCPClient("mock", mock_server_config)

        try:
            await client.connect()
            assert client.is_connected is True
        finally:
            await client.disconnect()

        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_list_tools_from_mock_server(self, mock_server_config):
        """Test listing tools from the mock server."""
        async with MCPClient("mock", mock_server_config) as client:
            tools = await client.list_tools()

            # Mock server exposes: echo, add, greet, fail, get_info
            tool_names = [t.name for t in tools]
            assert "echo" in tool_names
            assert "add" in tool_names
            assert "greet" in tool_names
            assert "fail" in tool_names
            assert "get_info" in tool_names

    @pytest.mark.asyncio
    async def test_call_echo_tool(self, mock_server_config):
        """Test calling the echo tool."""
        async with MCPClient("mock", mock_server_config) as client:
            result = await client.call_tool("echo", {"text": "Hello, World!"})

            assert result.isError is False
            assert len(result.content) == 1
            assert result.content[0].text == "Hello, World!"

    @pytest.mark.asyncio
    async def test_call_add_tool(self, mock_server_config):
        """Test calling the add tool with numeric arguments."""
        async with MCPClient("mock", mock_server_config) as client:
            result = await client.call_tool("add", {"a": 5, "b": 3})

            assert result.isError is False
            assert result.content[0].text == "8"

    @pytest.mark.asyncio
    async def test_call_greet_tool(self, mock_server_config):
        """Test calling the greet tool."""
        async with MCPClient("mock", mock_server_config) as client:
            result = await client.call_tool("greet", {"name": "Tester"})

            assert result.isError is False
            assert result.content[0].text == "Hello, Tester!"

    @pytest.mark.asyncio
    async def test_call_fail_tool(self, mock_server_config):
        """Test calling the fail tool returns error."""
        async with MCPClient("mock", mock_server_config) as client:
            result = await client.call_tool("fail", {"message": "Expected error"})

            assert result.isError is True
            assert result.content[0].text == "Expected error"

    @pytest.mark.asyncio
    async def test_call_get_info_tool(self, mock_server_config):
        """Test calling the get_info tool."""
        import json

        async with MCPClient("mock", mock_server_config) as client:
            result = await client.call_tool("get_info", {})

            assert result.isError is False
            info = json.loads(result.content[0].text)
            assert info["server_name"] == "mock-test-server"
            assert info["version"] == "1.0.0"

    @pytest.mark.asyncio
    async def test_tool_with_missing_required_arg(self, mock_server_config):
        """Test calling tool without required argument."""
        async with MCPClient("mock", mock_server_config) as client:
            # Echo requires "text" argument
            # MCP SDK validates input schema and returns validation error
            result = await client.call_tool("echo", {})
            # Server returns validation error for missing required field
            assert "required" in result.content[0].text.lower()


@pytest.mark.integration
class TestMCPManagerIntegration:
    """Integration tests for MCPManager with real mock server."""

    @pytest.mark.asyncio
    async def test_manager_with_mock_server(self, mock_server_config):
        """Test manager connects and discovers tools."""
        configs = {"mock": mock_server_config}
        manager = MCPManager(configs=configs)

        try:
            await manager.start()

            assert manager.is_started is True

            # Check server status
            status = manager.get_server_status()
            assert "mock" in status
            assert status["mock"].status == ServerStatus.CONNECTED

            # Check tools discovered
            tools = await manager.get_all_tools()
            assert "mock" in tools
            assert len(tools["mock"]) >= 5  # echo, add, greet, fail, get_info

        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_manager_call_tool(self, mock_server_config):
        """Test calling tools through manager."""
        configs = {"mock": mock_server_config}

        async with MCPManager(configs=configs) as manager:
            # Call echo tool
            result = await manager.call_tool("echo", {"text": "Manager test"})
            assert result.content[0].text == "Manager test"

            # Call add tool
            result = await manager.call_tool("add", {"a": 10, "b": 20})
            assert result.content[0].text == "30"

    @pytest.mark.asyncio
    async def test_manager_tool_routing(self, mock_server_config):
        """Test manager routes tools to correct server."""
        configs = {"mock": mock_server_config}

        async with MCPManager(configs=configs) as manager:
            # Verify tool index was built
            assert manager.find_tool_server("echo") == "mock"
            assert manager.find_tool_server("add") == "mock"
            assert manager.find_tool_server("nonexistent") is None

    @pytest.mark.asyncio
    async def test_manager_get_tool_list(self, mock_server_config):
        """Test manager returns flat tool list."""
        configs = {"mock": mock_server_config}

        async with MCPManager(configs=configs) as manager:
            tool_list = manager.get_tool_list()

            # Each entry is (server_name, tool_name, Tool)
            server_names = [t[0] for t in tool_list]
            tool_names = [t[1] for t in tool_list]

            assert all(name == "mock" for name in server_names)
            assert "echo" in tool_names
            assert "add" in tool_names

    @pytest.mark.asyncio
    async def test_manager_context_manager(self, mock_server_config):
        """Test manager works as context manager."""
        configs = {"mock": mock_server_config}

        async with MCPManager(configs=configs) as manager:
            assert manager.is_started is True
            tools = await manager.get_all_tools()
            assert len(tools) > 0

        assert manager.is_started is False

    @pytest.mark.asyncio
    async def test_manager_refresh_tools(self, mock_server_config):
        """Test refreshing tools from servers."""
        configs = {"mock": mock_server_config}

        async with MCPManager(configs=configs) as manager:
            original_tools = await manager.get_all_tools()

            # Refresh tools
            await manager.refresh_tools()

            new_tools = await manager.get_all_tools()

            # Should have same tools after refresh
            assert len(new_tools["mock"]) == len(original_tools["mock"])


@pytest.mark.integration
class TestFullWorkflow:
    """End-to-end workflow tests."""

    @pytest.mark.asyncio
    async def test_complete_workflow(self, mock_server_config):
        """Test complete workflow: connect, discover, call, disconnect."""
        configs = {"mock": mock_server_config}

        # 1. Create and start manager
        manager = MCPManager(configs=configs)
        await manager.start()

        try:
            # 2. Verify connection
            assert manager.is_started
            status = manager.get_server_status()
            assert status["mock"].status == ServerStatus.CONNECTED

            # 3. Discover tools
            tools = await manager.get_all_tools()
            tool_names = [t.name for t in tools["mock"]]
            assert "echo" in tool_names

            # 4. Call multiple tools
            echo_result = await manager.call_tool("echo", {"text": "test"})
            assert echo_result.content[0].text == "test"

            add_result = await manager.call_tool("add", {"a": 1, "b": 2})
            assert add_result.content[0].text == "3"

            greet_result = await manager.call_tool("greet", {"name": "World"})
            assert greet_result.content[0].text == "Hello, World!"

            # 5. Test error handling
            fail_result = await manager.call_tool("fail", {"message": "test error"})
            assert fail_result.isError is True

        finally:
            # 6. Shutdown
            await manager.shutdown()

        assert manager.is_started is False

    @pytest.mark.asyncio
    async def test_multiple_sequential_operations(self, mock_server_config):
        """Test multiple sequential tool calls."""
        configs = {"mock": mock_server_config}

        async with MCPManager(configs=configs) as manager:
            # Perform 10 sequential calls
            for i in range(10):
                result = await manager.call_tool("add", {"a": i, "b": i})
                expected = str(i + i)
                assert result.content[0].text == expected
