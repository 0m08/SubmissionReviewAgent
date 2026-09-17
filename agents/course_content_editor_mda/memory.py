"""Durable memory for this deployment."""

from managed_deepagents import define_memory

# Memory is opt-in: without this file the agent keeps nothing between runs.
#
# `scope="agent"` mounts one deployment-shared tree at `/memories/agent/`,
# read/write. `/memories/agent/AGENTS.md` is hot memory — loaded into every run —
# and other files under that path are read on demand. Every caller of this
# deployment shares it, so keep procedural knowledge there and never per-person
# facts, API keys, or tokens. `scope="none"` keeps no durable memory.
#
# The shared slice is a trust boundary. Because runs can write hot memory and hot
# memory is injected into every later run, whatever one caller gets the agent to
# save is read by everyone after them — instructions included. Treat memory as
# untrusted input, keep authority (tool access, approvals) in agent.py rather than
# in memory, and use "none" if your callers should not influence each other.
#
# Per-person memory rides on top of this, without a second mount. MDA has no
# private per-caller slice yet — `scope="user"` raises "not part of the launch
# contract" — so one file per editor lives *inside* this shared tree at
# `editors/<editor_id>.md`, and the agent is told which id it is working for by
# `middleware/editor_identity.py`. That is an honour system: the id is a label,
# not a key, and any caller can read any editor's file. It is acceptable only
# because what goes in those files is working preference. Re-scope here when
# `scope="user"` ships.
#
# The content lives in LangSmith Context Hub, not in this file: deploys never
# overwrite what earlier runs learned.
memory = define_memory(scope="agent")
