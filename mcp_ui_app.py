"""
MCP Server Manager UI

A Streamlit-based web interface for:
1. Adding and managing MCP servers dynamically
2. Interacting with an agent that can use tools from connected MCP servers

Usage:
    streamlit run mcp_ui_app.py
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import streamlit as st

from mcp_integration import MCPManager, ServerConfig


# Initialize session state
def init_session_state():
    """Initialize Streamlit session state."""
    if "servers" not in st.session_state:
        st.session_state.servers = {}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "manager" not in st.session_state:
        st.session_state.manager = None
    if "available_tools" not in st.session_state:
        st.session_state.available_tools = {}

    # Initialize persistent async runner
    if "async_runner" not in st.session_state:
        from mcp_integration.streamlit_async import StreamlitAsyncRunner
        st.session_state.async_runner = StreamlitAsyncRunner()
        st.session_state.async_runner.start()


def add_server(name: str, transport: str, command: str = None, args: list = None, url: str = None, description: str = ""):
    """Add a new MCP server to the configuration."""
    config_data = {
        "name": name,
        "transport": transport,
        "enabled": True,
        "description": description,
    }

    if transport == "stdio":
        if not command:
            st.error("Command is required for stdio transport")
            return False
        config_data["command"] = command
        config_data["args"] = args if args else []
    elif transport in ["sse", "streamable_http"]:
        if not url:
            st.error(f"URL is required for {transport} transport")
            return False
        config_data["url"] = url

    try:
        server_config = ServerConfig(**config_data)
        st.session_state.servers[name] = server_config
        return True
    except Exception as e:
        st.error(f"Error creating server config: {e}")
        return False


def remove_server(name: str):
    """Remove a server from the configuration."""
    if name in st.session_state.servers:
        del st.session_state.servers[name]
        return True
    return False


async def connect_to_servers(servers: dict, existing_manager=None):
    """Connect to all configured MCP servers.

    Args:
        servers: Dictionary of server configurations
        existing_manager: Existing manager to shutdown before creating new one

    Returns:
        Tuple of (manager, error_message, tools)
    """
    if not servers:
        return None, "No servers configured", {}

    try:
        # Close existing manager if any
        if existing_manager:
            try:
                await existing_manager.shutdown()
            except:
                pass

        # Create new manager with current servers
        manager = MCPManager(configs=servers)
        await manager.start()

        # Get server status
        status = manager.get_server_status()

        # Check for errors
        errors = []
        for name, info in status.items():
            if info.error:
                errors.append(f"{name}: {info.error}")

        if errors:
            await manager.shutdown()
            return None, "\n".join(errors), {}

        # Get available tools
        tools = await manager.get_all_tools()

        return manager, None, tools

    except Exception as e:
        return None, str(e), {}


async def call_agent_with_tool(manager, tool_name: str, tool_args: dict):
    """Call a tool through the agent.

    Args:
        manager: The MCP manager instance
        tool_name: Name of the tool to call
        tool_args: Arguments to pass to the tool

    Returns:
        Dictionary with success/error/warning keys
    """
    if not manager:
        return {"error": "Not connected to any servers"}

    result = None
    success_content = None

    try:
        result = await manager.call_tool(tool_name, tool_args)

        if result.isError:
            # Extract error message safely
            error_text = "Unknown error"
            if result.content and len(result.content) > 0:
                if hasattr(result.content[0], 'text') and result.content[0].text:
                    error_text = result.content[0].text
                else:
                    # Fallback: try to get string representation
                    error_text = str(result.content[0])

            return {"error": error_text}
        else:
            # Extract success content safely
            if result.content and len(result.content) > 0:
                if hasattr(result.content[0], 'text'):
                    success_content = result.content[0].text or ""
                else:
                    success_content = str(result.content[0])
            else:
                success_content = ""

            if not success_content:
                return {"error": "Tool returned empty response"}

            return {"success": success_content}

    except (RuntimeError, asyncio.exceptions.CancelledError) as e:
        # These errors should not occur with persistent event loop
        # If they do, it indicates a problem that needs investigation
        import logging
        logger = logging.getLogger(__name__)

        error_msg = str(e).lower()
        if "cancel scope" in error_msg or "cancelled" in error_msg:
            logger.warning(
                f"Event loop cancellation error occurred (should be rare with persistent loop): {e}"
            )

            # Try to recover if we got results
            if success_content:
                return {
                    "success": success_content,
                    "warning": "⚠️ Unexpected event loop error occurred, but results appear valid."
                }
            elif result and not result.isError:
                try:
                    content = result.content[0].text if result.content else None
                    if content:
                        return {
                            "success": content,
                            "warning": "⚠️ Unexpected event loop error occurred, but results appear valid."
                        }
                except:
                    pass

            # Error occurred before we could get results
            return {
                "error": "Event loop error. Please reconnect and try again.",
                "technical": str(e)
            }
        return {"error": f"Runtime error: {str(e)}"}

    except Exception as e:
        if success_content:
            # We got results but some other error occurred
            return {
                "success": success_content,
                "warning": f"⚠️ Warning: {str(e)}"
            }

        # Provide detailed error message
        error_msg = str(e) if str(e) else f"Unknown error of type {type(e).__name__}"
        return {"error": error_msg, "technical": f"{type(e).__name__}: {e}"}


def mcp_ui_page():
    """MCP UI page for admin users."""
    init_session_state()

    st.title("🔧 MCP Server Manager")
    st.markdown("Manage MCP servers and interact with an AI agent that can use their tools")

    # Create two columns for layout
    col1, col2 = st.columns([1, 2])

    # Left column: Server Management
    with col1:
        st.header("📡 MCP Servers")

        # Add server section
        with st.expander("➕ Add New Server", expanded=False):
            # Select transport type first (outside form for reactivity)
            transport = st.selectbox(
                "Transport Type",
                ["stdio", "streamable_http", "sse"],
                help="Select how to connect to the MCP server"
            )

            # Show appropriate help text
            if transport == "stdio":
                st.info("📋 **Local Server**: Runs as a subprocess on your machine (Python, Node.js, etc.)")
            else:
                st.info("🌐 **Remote Server**: Connects to an HTTP/HTTPS endpoint")

            st.markdown("---")

            # Now create the form with appropriate fields
            with st.form("add_server_form", clear_on_submit=True):
                server_name = st.text_input(
                    "Server Name*",
                    placeholder="my-server",
                    help="Unique identifier for this server"
                )

                # STDIO fields (only show if stdio selected)
                command = None
                args_input = None
                url = None

                if transport == "stdio":
                    st.markdown("**Local Server Configuration:**")
                    command = st.text_input(
                        "Command*",
                        placeholder="python",
                        help="Executable to run (e.g., python, node, npx)"
                    )
                    args_input = st.text_input(
                        "Arguments*",
                        placeholder="-m, mcp_integration.servers.web_tools_server",
                        help="Comma-separated command arguments"
                    )

                    # Example
                    st.caption("💡 Example: `python` with args `-m, mcp_integration.servers.web_tools_server`")

                # HTTP fields (only show if http/sse selected)
                elif transport in ["streamable_http", "sse"]:
                    st.markdown("**Remote Server Configuration:**")
                    url = st.text_input(
                        "URL*",
                        placeholder="https://remote.mcpservers.org/fetch/mcp",
                        help="Full HTTP/HTTPS endpoint URL"
                    )

                    # Example
                    st.caption("💡 Example: `https://remote.mcpservers.org/fetch/mcp`")

                description = st.text_input(
                    "Description",
                    placeholder="What does this server do?",
                    help="Optional description"
                )

                col1_btn, col2_btn = st.columns(2)
                with col1_btn:
                    submitted = st.form_submit_button("✅ Add Server", type="primary", use_container_width=True)
                with col2_btn:
                    cancelled = st.form_submit_button("❌ Cancel", use_container_width=True)

                if submitted and server_name:
                    # Validate inputs
                    if transport == "stdio" and (not command or not args_input):
                        st.error("Command and Arguments are required for stdio transport")
                    elif transport in ["streamable_http", "sse"] and not url:
                        st.error("URL is required for HTTP-based transports")
                    else:
                        # Warn if URL looks like a documentation site
                        if transport in ["streamable_http", "sse"] and url:
                            if any(pattern in url.lower() for pattern in ["docs.", "/docs", "documentation"]):
                                st.warning(
                                    "⚠️ This URL looks like a documentation site, not an MCP server. "
                                    "MCP servers are special endpoints that implement the Model Context Protocol. "
                                    "Regular websites won't work as MCP servers."
                                )

                        args = [arg.strip() for arg in args_input.split(",")] if args_input else []
                        if add_server(server_name, transport, command, args, url, description):
                            st.success(f"✅ Added server: {server_name}")
                            st.rerun()

        # Display configured servers
        st.subheader("Configured Servers")

        if st.session_state.servers:
            for name, config in st.session_state.servers.items():
                with st.container():
                    col_info, col_btn = st.columns([3, 1])
                    with col_info:
                        st.markdown(f"**{name}**")
                        st.caption(f"{config.transport} - {config.description or 'No description'}")
                    with col_btn:
                        if st.button("🗑️", key=f"remove_{name}", help="Remove server"):
                            remove_server(name)
                            st.rerun()
                    st.divider()
        else:
            st.info("No servers configured yet")

        # Connection controls
        st.subheader("Connection")

        col_conn, col_disc = st.columns(2)

        with col_conn:
            if st.button("🔌 Connect", use_container_width=True, type="primary"):
                with st.spinner("Connecting to servers..."):
                    manager, error, tools = st.session_state.async_runner.run_coroutine(
                        connect_to_servers(
                            st.session_state.servers,
                            st.session_state.manager
                        )
                    )
                    if error:
                        st.error(f"Connection failed: {error}")
                    else:
                        st.session_state.manager = manager
                        st.session_state.available_tools = tools
                        st.success("Connected successfully!")
                        st.rerun()

        with col_disc:
            if st.button("🔌 Disconnect", use_container_width=True):
                if st.session_state.manager:
                    try:
                        st.session_state.async_runner.run_coroutine(st.session_state.manager.shutdown())
                    except:
                        pass
                    st.session_state.manager = None
                    st.session_state.available_tools = {}
                    st.info("Disconnected")
                    st.rerun()

        # Connection status
        if st.session_state.manager:
            st.success("✅ Connected")

            # Show available tools
            with st.expander("🔧 Available Tools"):
                for server_name, tools in st.session_state.available_tools.items():
                    st.markdown(f"**{server_name}:**")
                    for tool in tools:
                        st.markdown(f"- `{tool.name}`: {tool.description or 'No description'}")
        else:
            st.warning("⚠️ Not connected")

    # Right column: Agent Interaction
    with col2:
        st.header("🤖 Agent Chat")

        if not st.session_state.manager:
            st.info("👈 Connect to MCP servers first to enable the agent")
        else:
            # Tool selection and usage
            st.subheader("Call Tools")

            # Get all tools
            all_tools = []
            tool_map = {}
            for server_name, tools in st.session_state.available_tools.items():
                for tool in tools:
                    tool_key = f"{tool.name} ({server_name})"
                    all_tools.append(tool_key)
                    tool_map[tool_key] = tool

            if all_tools:
                selected_tool_key = st.selectbox("Select a tool", all_tools)
                selected_tool = tool_map[selected_tool_key]

                st.markdown(f"**Description:** {selected_tool.description or 'No description'}")

                # Show input schema
                schema = selected_tool.inputSchema
                required_params = schema.get("required", [])
                properties = schema.get("properties", {})

                # Create form for tool parameters
                with st.form("tool_form"):
                    st.markdown("**Parameters:**")

                    param_values = {}
                    for param_name, param_info in properties.items():
                        param_type = param_info.get("type", "string")
                        param_desc = param_info.get("description", "")
                        is_required = param_name in required_params

                        label = f"{param_name}{'*' if is_required else ''}"

                        if param_type == "string":
                            value = st.text_input(label, help=param_desc)
                            if value:
                                param_values[param_name] = value
                        elif param_type == "integer":
                            value = st.number_input(label, help=param_desc, step=1)
                            param_values[param_name] = int(value)
                        elif param_type == "boolean":
                            value = st.checkbox(label, help=param_desc)
                            param_values[param_name] = value
                        elif param_type == "object":
                            value = st.text_area(label, help=f"{param_desc} (JSON format)")
                            if value:
                                try:
                                    param_values[param_name] = json.loads(value)
                                except:
                                    st.error(f"Invalid JSON for {param_name}")

                    call_tool = st.form_submit_button("🚀 Call Tool", type="primary")

                    if call_tool:
                        # Validate required parameters
                        missing = [p for p in required_params if p not in param_values]
                        if missing:
                            st.error(f"Missing required parameters: {', '.join(missing)}")
                        else:
                            with st.spinner("Calling tool..."):
                                result = st.session_state.async_runner.run_coroutine(
                                    call_agent_with_tool(
                                        st.session_state.manager,
                                        selected_tool.name,
                                        param_values
                                    )
                                )

                                if "error" in result and "success" not in result:
                                    st.error(f"**Error:** {result['error']}")
                                    if "technical" in result:
                                        with st.expander("Technical Details"):
                                            st.code(result['technical'])
                                elif "success" in result:
                                    st.success("✅ Tool executed successfully!")

                                    # Show warning if present
                                    if "warning" in result:
                                        st.warning(result['warning'])

                                    # Display full result
                                    content = result['success']

                                    # Calculate appropriate height based on content
                                    lines = content.count('\n') + 1
                                    height = min(max(150, lines * 25), 600)  # Between 150 and 600 pixels

                                    st.text_area(
                                        f"Result ({len(content)} characters, {lines} lines)",
                                        content,
                                        height=height,
                                        disabled=True
                                    )

                                    # Add copy button helper text
                                    st.caption("💡 Tip: Click in the text area and use Ctrl+A then Ctrl+C to copy the full result")

                                    # Add to chat history
                                    st.session_state.chat_history.append({
                                        "tool": selected_tool.name,
                                        "params": param_values,
                                        "result": result['success']
                                    })

                                    # Don't rerun - let user see the results
            else:
                st.warning("No tools available from connected servers")

            # Chat history
            st.divider()
            st.subheader("📜 Execution History")

            if st.session_state.chat_history:
                for idx, entry in enumerate(reversed(st.session_state.chat_history)):
                    call_num = len(st.session_state.chat_history) - idx
                    result_preview = entry['result'][:100] + "..." if len(entry['result']) > 100 else entry['result']

                    with st.expander(f"🔧 {entry['tool']} - Call #{call_num} - {result_preview}"):
                        st.markdown("**Parameters:**")
                        st.json(entry['params'])

                        st.markdown("**Result:**")
                        result_length = len(entry['result'])
                        result_lines = entry['result'].count('\n') + 1

                        # Show toggle for large results
                        if result_length > 500:
                            show_full = st.checkbox(
                                f"📄 Show Full Result ({result_length} characters, {result_lines} lines)",
                                value=False,
                                key=f"show_full_{call_num}"
                            )

                            if show_full:
                                st.text_area(
                                    "Full Output",
                                    entry['result'],
                                    height=400,
                                    disabled=True,
                                    key=f"history_result_{call_num}"
                                )
                            else:
                                # Show preview
                                st.text_area(
                                    "Preview (first 500 characters)",
                                    entry['result'][:500] + "\n\n... (check box above to see full result)",
                                    height=150,
                                    disabled=True,
                                    key=f"history_preview_{call_num}"
                                )
                        else:
                            # For smaller results, show directly
                            st.text_area(
                                f"Output ({result_length} characters)",
                                entry['result'],
                                height=min(max(100, result_lines * 20), 300),
                                disabled=True,
                                key=f"history_result_{call_num}"
                            )
            else:
                st.info("No execution history yet")

            # Clear history button
            if st.session_state.chat_history:
                if st.button("🗑️ Clear History"):
                    st.session_state.chat_history = []
                    st.rerun()

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <small>MCP Server Manager - Connect to MCP servers and interact with AI agents</small>
    </div>
    """, unsafe_allow_html=True)


# Cleanup handler for session end
import atexit


def cleanup_async_runner():
    """Clean up the async runner when session ends."""
    try:
        # Check if session_state exists and has the required attributes
        if hasattr(st, 'session_state') and "async_runner" in st.session_state:
            try:
                # Disconnect manager first
                if st.session_state.manager:
                    st.session_state.async_runner.run_coroutine(
                        st.session_state.manager.shutdown()
                    )
            except Exception as e:
                # Silently ignore errors during cleanup
                pass

            # Stop the event loop
            try:
                st.session_state.async_runner.stop()
            except Exception as e:
                # Silently ignore errors during cleanup
                pass
    except:
        # Silently ignore any errors during cleanup
        pass


# Register cleanup (runs when Python process exits)
atexit.register(cleanup_async_runner)


def main():
    """Main Streamlit app for standalone mode."""
    st.set_page_config(
        page_title="MCP Server Manager",
        page_icon="🔧",
        layout="wide"
    )
    mcp_ui_page()


if __name__ == "__main__":
    main()
