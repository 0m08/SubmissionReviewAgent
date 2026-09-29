# about_agents.py
import streamlit as st
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

st.set_page_config(page_title="About Agents", page_icon="📘", layout="wide")
MD_PATH = Path("about_agents.md")  # keep your filename

# -----------------------------
# Markdown parser (Agents + Tools)
# -----------------------------
def parse_md(md_text: str) -> Dict[str, Any]:
    """
    Supported structure (agents):
      # Title
      ## <Section>
      **Overview**:
      **Key Outputs**:
        - ...
      ### <Agent>
      **Description**:
      **How to Use**:
        1. ...
      **Tools Used**:
        - ...

    Supported structure (tools):
      ## Available Tools
      ### <Category>
      - **Tool Name**: description...
    """
    lines = md_text.splitlines()

    data: Dict[str, Any] = {
        "_title": None,
        "sections": {},               # {section: {overview, key_outputs[], agents{ name: {desc, how_to_use[], tools_used[]} } } }
        "available_tools": {          # {categories: [{name, tools: [(tool_name, description), ...]}], index: {tool_name_lower: (cat_idx, tool_idx)} }
            "categories": [],
            "index": {}               # quick lookup by tool name (lower)
        }
    }

    # state
    current_section = None
    current_agent = None
    mode = None  # "overview" | "keyoutputs" | "desc" | "how" | "tools" | "in_tools_section" | "tool_category"

    # helpers
    def ensure_section(name: str):
        if name not in data["sections"]:
            data["sections"][name] = {"overview": "", "key_outputs": [], "agents": {}}

    def ensure_agent(section: str, name: str):
        if name not in data["sections"][section]["agents"]:
            data["sections"][section]["agents"][name] = {
                "description": "",
                "how_to_use": [],
                "tools_used": [],
            }

    def start_tool_category(cat_name: str):
        data["available_tools"]["categories"].append({"name": cat_name, "tools": []})

    def append_tool(cat_name: str, tool_name: str, desc: str):
        # find category (it's always the last one we opened)
        if not data["available_tools"]["categories"] or data["available_tools"]["categories"][-1]["name"] != cat_name:
            start_tool_category(cat_name)
        cat_idx = len(data["available_tools"]["categories"]) - 1
        data["available_tools"]["categories"][cat_idx]["tools"].append((tool_name, desc))
        tool_key = tool_name.strip().lower()
        data["available_tools"]["index"][tool_key] = (cat_idx, len(data["available_tools"]["categories"][cat_idx]["tools"]) - 1)

    # pass 1: parse
    tools_current_category = None

    for raw in lines:
        line = raw.rstrip()

        # title
        if line.startswith("# "):
            data["_title"] = line[2:].strip()
            continue

        # section headers
        if line.startswith("## "):
            h2 = line[3:].strip()
            current_agent = None
            mode = None

            if re.match(r"^available tools$", h2, flags=re.I):
                mode = "in_tools_section"
                tools_current_category = None
            else:
                current_section = h2
                ensure_section(current_section)
            continue

        # agent headers
        if line.startswith("### "):
            h3 = line[4:].strip()

            if mode == "in_tools_section":
                # tool category
                tools_current_category = h3
                start_tool_category(tools_current_category)
                mode = "tool_category"
            else:
                # agent under a course section
                if current_section is None:
                    current_section = "General"
                    ensure_section(current_section)
                current_agent = h3
                ensure_agent(current_section, current_agent)
                mode = None
            continue

        # within Agents sections
        if mode not in ("in_tools_section", "tool_category"):
            if re.match(r"^\*\*Overview\*\*:", line, flags=re.I):
                mode = "overview"
                content = line.split(":", 1)[1].strip()
                if content:
                    data["sections"][current_section]["overview"] += content + "\n"
                continue

            if re.match(r"^\*\*Key Outputs\*\*:", line, flags=re.I):
                mode = "keyoutputs"
                continue

            if re.match(r"^\*\*Description\*\*:", line, flags=re.I):
                mode = "desc"
                content = line.split(":", 1)[1].strip()
                if current_agent:
                    data["sections"][current_section]["agents"][current_agent]["description"] += content + "\n"
                continue

            if re.match(r"^\*\*How to Use\*\*:", line, flags=re.I):
                mode = "how"
                continue

            if re.match(r"^\*\*Tools Used\*\*:", line, flags=re.I):
                mode = "tools"
                continue

            # list collectors
            if mode == "keyoutputs" and line.strip().startswith("- "):
                data["sections"][current_section]["key_outputs"].append(line.strip()[2:].strip())
                continue

            if mode == "tools" and current_agent and line.strip().startswith("- "):
                data["sections"][current_section]["agents"][current_agent]["tools_used"].append(
                    line.strip()[2:].strip()
                )
                continue

            if mode == "how" and current_agent and re.match(r"^\s*\d+\.\s+", line):
                step = re.sub(r"^\s*\d+\.\s+", "", line).strip()
                if step:
                    data["sections"][current_section]["agents"][current_agent]["how_to_use"].append(step)
                continue

            # multiline text
            if mode == "overview" and current_section and line.strip():
                data["sections"][current_section]["overview"] += line.strip() + "\n"
                continue

            if mode == "desc" and current_agent and line.strip():
                data["sections"][current_section]["agents"][current_agent]["description"] += line.strip() + "\n"
                continue

        # within Tools section
        if mode in ("in_tools_section", "tool_category"):
            # bullets like: - **Tool Name**: description...
            m = re.match(r"^\s*-\s*\*\*(.+?)\*\*\s*:\s*(.+)$", line)
            if m and tools_current_category:
                tool_name = m.group(1).strip()
                desc = m.group(2).strip()
                append_tool(tools_current_category, tool_name, desc)
            continue

    # tidy
    for sec in data["sections"].values():
        sec["overview"] = sec["overview"].strip()
        for ag in sec["agents"].values():
            ag["description"] = ag["description"].strip()

    return data

# -----------------------------
# Load MD
# -----------------------------
if not MD_PATH.exists():
    st.error(f"Markdown file not found at: {MD_PATH.resolve()}")
    st.stop()

data = parse_md(MD_PATH.read_text(encoding="utf-8"))

# -----------------------------
# UI helpers
# -----------------------------
def render_tools_tab(tools_data: Dict[str, Any], spotlight: str | None = None):
    """
    Render the Available Tools tab.
    If `spotlight` is provided (tool name, case-insensitive), that tool is highlighted.
    """
    spot_key = (spotlight or "").strip().lower()
    st.header("Available Tools")

    for cat in tools_data["categories"]:
        with st.expander(f"🧰 {cat['name']}", expanded=False):
            for (tool_name, desc) in cat["tools"]:
                with st.container(border=True):
                    if tool_name.strip().lower() == spot_key:
                        st.markdown("✅ **Highlighted**")
                    st.markdown(f"### {tool_name}")
                    st.write(desc)

def render_agent_card(section_name: str, agent_name: str, agent: Dict[str, Any]):
    key_base = f"agent_{section_name}_{agent_name}"
    with st.container(border=False):
        left, right = st.columns([0.08, 0.92])
        with left:
            opened = st.toggle("", key=f"{key_base}_toggle", value=False)
        with right:
            st.markdown(f"### {agent_name}")
            # how_n = len(agent.get("how_to_use", []))
            # tool_n = len(agent.get("tools_used", []))
            # st.caption(f"{how_n} step{'s' if how_n!=1 else ''} • {tool_n} tool{'s' if tool_n!=1 else ''}")

        if opened:
            if agent.get("description"):
                st.markdown("**Description**")
                st.write(agent["description"])

            if agent.get("how_to_use"):
                st.markdown("**How to Use**")
                for i, step in enumerate(agent["how_to_use"], start=1):
                    st.markdown(f"{i}. {step}")

            if agent.get("tools_used"):
                st.markdown("**Tools Used**")
                # Render as buttons that jump to the Tools tab and spotlight the selected tool
                cols = st.columns(min(4, max(1, len(agent["tools_used"]))))
                for i, tool in enumerate(agent["tools_used"]):
                    with cols[i % len(cols)]:
                        if st.button(tool, key=f"{key_base}_tool_{i}"):
                            st.session_state["active_tab"] = "Tools"
                            st.session_state["spotlight_tool"] = tool

    # subtle spacing between agents
    # st.write("")


# -----------------------------
# Page chrome
# -----------------------------
st.title(data.get("_title") or "Abouts Agents")
st.caption("Browse sections & agents, or switch to the other tabs to explore.")

# Tabs: Agents | Tools
tabs = st.tabs(["Course Outline Agents", "More Agents..."])

# session state for cross-tab spotlighting
st.session_state.setdefault("active_tab", "Agents")
st.session_state.setdefault("spotlight_tool", "")

# -----------------------------
# Agents tab
# -----------------------------
with tabs[0]:
    # query = st.text_input("🔎 Filter agents by name (optional)").strip().lower()
    for section_name, section in data["sections"].items():
        with st.expander(f"{section_name}", expanded=False):

            with st.container(border=True):
                st.markdown(f"## {section_name}")
                cols = st.columns([0.65, 0.35], vertical_alignment="top")
                with cols[0]:
                    if section.get("overview"):
                        st.markdown("**Overview**")
                        st.write(section["overview"])
                with cols[1]:
                    if section.get("key_outputs"):
                        st.markdown("**Key Outputs**")
                        for item in section["key_outputs"]:
                            st.markdown(f"- {item}")

            st.divider()
            if section["agents"]:
                st.subheader("Agents")
                # Expand / Collapse all
                c1, c2, _ = st.columns([0.18, 0.18, 0.64])
                with c1:
                    if st.button("Expand all", key=f"expand_{section_name}"):
                        for agent_name in section["agents"].keys():
                            st.session_state[f"agent_{section_name}_{agent_name}_toggle"] = True
                with c2:
                    if st.button("Collapse all", key=f"collapse_{section_name}"):
                        for agent_name in section["agents"].keys():
                            st.session_state[f"agent_{section_name}_{agent_name}_toggle"] = False

            for agent_name, agent in section["agents"].items():
                # if query and query not in agent_name.lower():
                #     continue
                render_agent_card(section_name, agent_name, agent)
                st.write("")

with tabs[1]:
    st.info("Documentation of other agents to be added here...")
# -----------------------------
# Tools tab
# -----------------------------
# with tabs[1]:
#     spotlight = st.session_state.get("spotlight_tool", "")
#     render_tools_tab(data["available_tools"], spotlight=spotlight)
#     # Reset spotlight button
#     if spotlight:
#         st.button("Clear highlight", key="clear_spotlight", on_click=lambda: st.session_state.update(spotlight_tool=""))

# If any agent button asked to jump to Tools, switch tabs (visual cue)
# if st.session_state.get("active_tab") == "Tools":
#     st.toast(f"Jumped to Tools • Highlighting: {st.session_state.get('spotlight_tool', '')}")
