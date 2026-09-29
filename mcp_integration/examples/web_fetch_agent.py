"""
Web Tools Agent - Proof of Concept

This demo agent uses the Web Tools MCP server to retrieve and analyze web content.
It demonstrates how MCP tools can be integrated into a workflow.

The Web Tools MCP provides:
- fetch_url: Fetch web pages and convert to markdown
- fetch_json: Fetch JSON from APIs
- check_url: Check if URLs are accessible
- extract_links: Extract all links from a page

Usage:
    python -m mcp_integration.examples.web_fetch_agent

Requirements:
    - mcp_integration package
    - httpx package (for the web tools server)
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path for imports when running directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp_integration import MCPManager, ServerConfig


async def demo_web_tools_mcp():
    """
    Demonstrate the Web Tools MCP server capabilities.

    This shows:
    1. Connecting to a real MCP server (Web Tools)
    2. Discovering available tools
    3. Using tools to fetch and analyze web content
    """
    print("=" * 70)
    print("Web Tools Agent - MCP Integration Proof of Concept")
    print("=" * 70)

    # Configure the Web Tools MCP server (Python-based, no npm needed)
    config = {
        "web-tools": ServerConfig(
            name="web-tools",
            transport="stdio",
            command="python",
            args=["-m", "mcp_integration.servers.web_tools_server"],
            enabled=True,
            description="Web utilities - fetch pages, check URLs, extract links",
        )
    }

    print("\n[1] Connecting to Web Tools MCP server...")

    async with MCPManager(configs=config) as manager:
        # Check connection status
        status = manager.get_server_status()
        for name, info in status.items():
            print(f"\n    Server '{name}': {info.status.value}")
            if info.error:
                print(f"    Error: {info.error}")
                return

        # Discover tools
        print("\n[2] Discovering available tools...")
        tools = await manager.get_all_tools()

        for server_name, tool_list in tools.items():
            print(f"\n    Tools from '{server_name}':")
            for tool in tool_list:
                desc = tool.description[:60] + "..." if len(tool.description) > 60 else tool.description
                print(f"      - {tool.name}: {desc}")

        # Demo 1: Fetch a web page
        print("\n" + "=" * 70)
        print("[3] Demo: Fetching a web page (fetch_url)")
        print("=" * 70)

        url = "https://example.com"
        print(f"\n    Fetching: {url}")

        try:
            result = await manager.call_tool("fetch_url", {"url": url})

            if result.isError:
                print(f"    Error: {result.content[0].text}")
            else:
                content = result.content[0].text
                # Show first 500 chars of content
                preview = content[:500] + "..." if len(content) > 500 else content
                print(f"\n    Content (converted to markdown):\n    {'-' * 50}")
                for line in preview.split('\n')[:15]:
                    print(f"    {line}")
                print(f"    {'-' * 50}")
                print(f"\n    Total content length: {len(content)} characters")
        except Exception as e:
            print(f"    Error fetching page: {e}")

        # Demo 2: Check URL accessibility
        print("\n" + "=" * 70)
        print("[4] Demo: Checking URL status (check_url)")
        print("=" * 70)

        urls_to_check = [
            "https://example.com",
            "https://httpbin.org/status/200",
        ]

        for check_url in urls_to_check:
            print(f"\n    Checking: {check_url}")
            try:
                result = await manager.call_tool("check_url", {"url": check_url})
                if not result.isError:
                    print(f"    {result.content[0].text}")
            except Exception as e:
                print(f"    Error: {e}")

        # Demo 3: Fetch JSON from an API
        print("\n" + "=" * 70)
        print("[5] Demo: Fetching JSON from API (fetch_json)")
        print("=" * 70)

        api_url = "https://httpbin.org/json"
        print(f"\n    Fetching: {api_url}")

        try:
            result = await manager.call_tool("fetch_json", {"url": api_url})

            if result.isError:
                print(f"    Error: {result.content[0].text}")
            else:
                content = result.content[0].text
                print(f"\n    Response:\n    {'-' * 50}")
                for line in content.split('\n')[:20]:
                    print(f"    {line}")
                print(f"    {'-' * 50}")
        except Exception as e:
            print(f"    Error fetching API: {e}")

        # Demo 4: Extract links from a page
        print("\n" + "=" * 70)
        print("[6] Demo: Extracting links (extract_links)")
        print("=" * 70)

        link_url = "https://example.com"
        print(f"\n    Extracting links from: {link_url}")

        try:
            result = await manager.call_tool("extract_links", {"url": link_url})

            if result.isError:
                print(f"    Error: {result.content[0].text}")
            else:
                print(f"\n    {result.content[0].text}")
        except Exception as e:
            print(f"    Error extracting links: {e}")

        # Show workflow integration example
        print("\n" + "=" * 70)
        print("[7] Integration Example: Research Workflow")
        print("=" * 70)

        print("""
    The Web Tools MCP can be integrated into your content generation workflow:

    1. Research Phase:
       - fetch_url: Get documentation for technical accuracy
       - fetch_json: Pull data from APIs for statistics
       - extract_links: Find related resources

    2. Quality Assurance:
       - check_url: Verify all external links work
       - Validate sources before publishing

    Example workflow integration:

        from mcp_integration import MCPManager

        async def research_topic(topic_url: str):
            async with MCPManager(config_path="config/servers.yaml") as manager:
                # Fetch the content
                result = await manager.call_tool("fetch_url", {"url": topic_url})
                content = result.content[0].text

                # Extract related links for further research
                links = await manager.call_tool("extract_links", {"url": topic_url})

                # Pass to LLM for summarization
                summary = await llm.summarize(content)

                return summary, links

        async def validate_course_links(urls: list[str]):
            async with MCPManager(config_path="config/servers.yaml") as manager:
                results = []
                for url in urls:
                    status = await manager.call_tool("check_url", {"url": url})
                    results.append(status)
                return results
        """)

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nThe Web Tools MCP server has been successfully integrated.")
    print("You can now use it in your agents and workflows.")
    print("\nTo use from config file:")
    print("    async with MCPManager('mcp_integration/config/servers.yaml') as manager:")
    print("        result = await manager.call_tool('fetch_url', {'url': '...'})")


async def quick_example():
    """
    Minimal example showing how to use Web Tools MCP.
    Copy this pattern for quick integration.
    """
    from mcp_integration import MCPManager

    # Use the config file
    config_path = Path(__file__).parent.parent / "config" / "servers.yaml"

    async with MCPManager(config_path=config_path) as manager:
        # Fetch a webpage
        result = await manager.call_tool("fetch_url", {
            "url": "https://example.com"
        })
        print("Fetched content:")
        print(result.content[0].text[:300])

        # Check a URL
        status = await manager.call_tool("check_url", {
            "url": "https://example.com"
        })
        print("\nURL status:")
        print(status.content[0].text)


if __name__ == "__main__":
    print("\nStarting Web Tools Agent Demo...")
    print("(Using Python-based MCP server - no npm required)\n")

    try:
        asyncio.run(demo_web_tools_mcp())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nError running demo: {e}")
        import traceback
        traceback.print_exc()
