"""
Example usage of the MCP Integration module.

This example demonstrates how to:
1. Use MCPClient for single server connections
2. Use MCPManager for multi-server orchestration
3. Discover and call tools
"""

import asyncio
from pathlib import Path

# Add parent to path for imports when running directly
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp_integration import MCPManager, MCPClient, ServerConfig


async def single_client_example():
    """Example: Using MCPClient for a single server."""
    print("\n" + "=" * 60)
    print("Single Client Example")
    print("=" * 60)

    # Create configuration
    config = ServerConfig(
        name="mock-server",
        transport="stdio",
        command="python",
        args=["-m", "mcp_integration.tests.fixtures.mock_server"],
        enabled=True,
    )

    # Use as context manager for automatic cleanup
    async with MCPClient("mock", config) as client:
        # Discover available tools
        tools = await client.list_tools()
        print(f"\nDiscovered {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")

        # Call a tool
        print("\nCalling 'echo' tool:")
        result = await client.call_tool("echo", {"text": "Hello from MCPClient!"})
        print(f"  Result: {result.content[0].text}")

        # Call another tool
        print("\nCalling 'add' tool:")
        result = await client.call_tool("add", {"a": 5, "b": 7})
        print(f"  Result: 5 + 7 = {result.content[0].text}")


async def manager_example():
    """Example: Using MCPManager for multiple servers."""
    print("\n" + "=" * 60)
    print("Manager Example")
    print("=" * 60)

    # Create configurations for multiple servers (using same mock server twice)
    configs = {
        "server1": ServerConfig(
            name="server1",
            transport="stdio",
            command="python",
            args=["-m", "mcp_integration.tests.fixtures.mock_server"],
            enabled=True,
        ),
        # You can add more servers here
        # "server2": ServerConfig(...)
    }

    # Use manager as context manager
    async with MCPManager(configs=configs) as manager:
        # Get all tools from all servers
        all_tools = await manager.get_all_tools()
        print("\nTools by server:")
        for server_name, tools in all_tools.items():
            print(f"\n  {server_name}:")
            for tool in tools:
                print(f"    - {tool.name}")

        # Call tools (manager routes to correct server)
        print("\nCalling tools through manager:")

        result = await manager.call_tool("greet", {"name": "MCPManager"})
        print(f"  greet: {result.content[0].text}")

        result = await manager.call_tool("add", {"a": 100, "b": 200})
        print(f"  add: 100 + 200 = {result.content[0].text}")

        # Check which server handles each tool
        print("\nTool routing:")
        for tool_name in ["echo", "add", "greet"]:
            server = manager.find_tool_server(tool_name)
            print(f"  {tool_name} -> {server}")


async def config_file_example():
    """Example: Loading configuration from YAML file."""
    print("\n" + "=" * 60)
    print("Config File Example")
    print("=" * 60)

    # Path to test configuration
    config_path = Path(__file__).parent.parent / "tests" / "fixtures" / "test_servers.yaml"

    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return

    print(f"\nLoading config from: {config_path}")

    # Manager loads enabled servers from config
    async with MCPManager(config_path=config_path) as manager:
        status = manager.get_server_status()
        print("\nServer status:")
        for name, info in status.items():
            print(f"  {name}: {info.status.value}")
            if info.tools:
                print(f"    Tools: {[t.name for t in info.tools]}")


async def error_handling_example():
    """Example: Handling tool errors."""
    print("\n" + "=" * 60)
    print("Error Handling Example")
    print("=" * 60)

    config = ServerConfig(
        name="mock",
        transport="stdio",
        command="python",
        args=["-m", "mcp_integration.tests.fixtures.mock_server"],
        enabled=True,
    )

    async with MCPClient("mock", config) as client:
        # Call a tool that always returns an error
        print("\nCalling 'fail' tool (expected to return error):")
        result = await client.call_tool("fail", {"message": "Intentional error"})

        if result.isError:
            print(f"  Tool returned error: {result.content[0].text}")
        else:
            print(f"  Tool succeeded: {result.content[0].text}")


async def main():
    """Run all examples."""
    print("MCP Integration Examples")
    print("=" * 60)

    try:
        await single_client_example()
        await manager_example()
        await config_file_example()
        await error_handling_example()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
