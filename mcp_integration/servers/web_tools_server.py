"""
Web Tools MCP Server

A Python-based MCP server providing web utilities:
- fetch_url: Fetch and parse web pages to markdown
- fetch_json: Fetch JSON from APIs
- check_url: Check if a URL is accessible

This server can be used for research and content validation
in the content generation workflow.

Run as MCP server:
    python -m mcp_integration.servers.web_tools_server
"""

import asyncio
import json
import logging
import re
import sys
from urllib.parse import urlparse

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    TextContent,
    Tool,
)

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

# Create server instance
server = Server("web-tools-server")

# HTTP client with reasonable defaults
HTTP_TIMEOUT = 30.0
USER_AGENT = "MCP-WebTools/1.0 (Content-Generation-Workflow)"


def html_to_markdown(html: str) -> str:
    """
    Simple HTML to markdown conversion.
    Strips tags and preserves basic structure.
    """
    # Remove script and style elements
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)

    # Convert headers
    for i in range(1, 7):
        html = re.sub(rf'<h{i}[^>]*>(.*?)</h{i}>', rf'\n{"#" * i} \1\n', html, flags=re.DOTALL | re.IGNORECASE)

    # Convert paragraphs
    html = re.sub(r'<p[^>]*>(.*?)</p>', r'\n\1\n', html, flags=re.DOTALL | re.IGNORECASE)

    # Convert links
    html = re.sub(r'<a[^>]*href=["\']([^"\']*)["\'][^>]*>(.*?)</a>', r'[\2](\1)', html, flags=re.DOTALL | re.IGNORECASE)

    # Convert bold/strong
    html = re.sub(r'<(b|strong)[^>]*>(.*?)</\1>', r'**\2**', html, flags=re.DOTALL | re.IGNORECASE)

    # Convert italic/em
    html = re.sub(r'<(i|em)[^>]*>(.*?)</\1>', r'*\2*', html, flags=re.DOTALL | re.IGNORECASE)

    # Convert list items
    html = re.sub(r'<li[^>]*>(.*?)</li>', r'\n- \1', html, flags=re.DOTALL | re.IGNORECASE)

    # Convert line breaks
    html = re.sub(r'<br\s*/?>', '\n', html, flags=re.IGNORECASE)

    # Remove remaining tags
    html = re.sub(r'<[^>]+>', '', html)

    # Clean up whitespace
    html = re.sub(r'\n\s*\n', '\n\n', html)
    html = re.sub(r'  +', ' ', html)

    # Decode HTML entities
    html = html.replace('&nbsp;', ' ')
    html = html.replace('&amp;', '&')
    html = html.replace('&lt;', '<')
    html = html.replace('&gt;', '>')
    html = html.replace('&quot;', '"')
    html = html.replace('&#39;', "'")

    return html.strip()


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return list of available tools."""
    return [
        Tool(
            name="fetch_url",
            description="Fetch a web page and convert it to markdown. Returns the text content of the page.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch",
                    },
                    "max_length": {
                        "type": "integer",
                        "description": "Maximum length of returned content (default: 50000)",
                        "default": 50000,
                    },
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="fetch_json",
            description="Fetch JSON data from an API endpoint. Returns parsed JSON as formatted text.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The API URL to fetch",
                    },
                    "headers": {
                        "type": "object",
                        "description": "Optional HTTP headers to include",
                        "additionalProperties": {"type": "string"},
                    },
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="check_url",
            description="Check if a URL is accessible and return status information.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to check",
                    },
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="extract_links",
            description="Fetch a web page and extract all links from it.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch and extract links from",
                    },
                    "filter_domain": {
                        "type": "string",
                        "description": "Only return links matching this domain (optional)",
                    },
                },
                "required": ["url"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Handle tool calls."""
    logger.info(f"Tool called: {name} with arguments: {arguments}")

    try:
        if name == "fetch_url":
            return await handle_fetch_url(arguments)
        elif name == "fetch_json":
            return await handle_fetch_json(arguments)
        elif name == "check_url":
            return await handle_check_url(arguments)
        elif name == "extract_links":
            return await handle_extract_links(arguments)
        else:
            raise RuntimeError(f"Unknown tool: {name}")
    except Exception as e:
        logger.error(f"Error in {name}: {e}")
        raise


async def handle_fetch_url(arguments: dict):
    """Fetch a URL and convert to markdown."""
    url = arguments.get("url")
    max_length = arguments.get("max_length", 50000)

    if not url:
        raise ValueError("URL is required")

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        response = await client.get(url, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")

        if "text/html" in content_type:
            # Convert HTML to markdown
            content = html_to_markdown(response.text)
        else:
            # Return raw text
            content = response.text

        # Truncate if needed
        if len(content) > max_length:
            content = content[:max_length] + f"\n\n[Content truncated at {max_length} characters]"

        return [TextContent(type="text", text=content)]


async def handle_fetch_json(arguments: dict):
    """Fetch JSON from an API."""
    url = arguments.get("url")
    headers = arguments.get("headers", {})

    if not url:
        raise ValueError("URL is required")

    request_headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    request_headers.update(headers)

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        response = await client.get(url, headers=request_headers)
        response.raise_for_status()

        try:
            data = response.json()
            formatted = json.dumps(data, indent=2)
            return [TextContent(type="text", text=formatted)]
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON: {e}")


async def handle_check_url(arguments: dict):
    """Check if a URL is accessible."""
    url = arguments.get("url")

    if not url:
        raise ValueError("URL is required")

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
            response = await client.head(url, headers={"User-Agent": USER_AGENT})

            result = {
                "url": url,
                "status": "accessible",
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type", "unknown"),
                "content_length": response.headers.get("content-length", "unknown"),
            }

            if response.status_code >= 400:
                result["status"] = "error"

            return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except httpx.TimeoutException:
        return [TextContent(type="text", text=json.dumps({
            "url": url,
            "status": "timeout",
            "error": "Request timed out"
        }, indent=2))]
    except httpx.ConnectError as e:
        return [TextContent(type="text", text=json.dumps({
            "url": url,
            "status": "unreachable",
            "error": str(e)
        }, indent=2))]


async def handle_extract_links(arguments: dict):
    """Extract links from a web page."""
    url = arguments.get("url")
    filter_domain = arguments.get("filter_domain")

    if not url:
        raise ValueError("URL is required")

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        response = await client.get(url, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()

        # Extract all href attributes
        links = re.findall(r'href=["\']([^"\']+)["\']', response.text)

        # Resolve relative URLs
        base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        resolved_links = []
        for link in links:
            if link.startswith("//"):
                link = f"https:{link}"
            elif link.startswith("/"):
                link = f"{base_url}{link}"
            elif not link.startswith("http"):
                continue  # Skip non-http links

            # Apply domain filter if specified
            if filter_domain:
                if filter_domain not in urlparse(link).netloc:
                    continue

            if link not in resolved_links:
                resolved_links.append(link)

        result = {
            "source_url": url,
            "total_links": len(resolved_links),
            "links": resolved_links[:100],  # Limit to first 100
        }

        if len(resolved_links) > 100:
            result["note"] = f"Showing first 100 of {len(resolved_links)} links"

        return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def main():
    """Run the MCP server."""
    logger.info("Starting Web Tools MCP server...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
