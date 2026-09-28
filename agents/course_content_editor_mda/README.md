# Course Content Editor — Managed Deep Agents build

The same product as [`agents/course_content_editor/`](../course_content_editor/)
(Anthropic Managed Agents) and
[`agents/course_content_editor_deepagents/`](../course_content_editor_deepagents/)
(LangChain Deep Agents, run locally), rebuilt on
[Managed Deep Agents](https://docs.langchain.com/langsmith/python/managed-deep-agents-overview) —
LangSmith's hosted harness, driven by the `mda` CLI.

**The one architectural difference: there is no shell.** That is not a
limitation worked around; it is the point. See [Why no sandbox](#why-no-sandbox).

## Status

**Deployed and passing.** Rebuilt on a sandbox with the auth proxy, redeployed,
and verified end to end against a real course sheet: the agent finds the baked
scripts, edits, presents, withholds the commit until asked, then writes a new
tab with every original column intact — and the committed notes are the *edited*
ones (4407 -> 3696 words), not a re-pull of the source.

Suites: `selftest.py` (offline), `sandboxprobe.py` (the auth proxy),
`scriptstest.py` (the scripts in a real sandbox), `deployedtest.py` (the agent's
behaviour on the deployment), `disconnecttest.py` (survival when the client
goes away).

## Layout

```
agents/course_content_editor_mda/
├── agent.py               # define_deep_agent: model, tools, middleware, subagent, permissions, interrupt_on
├── instructions.md        # the system prompt (Context Hub on deploy)
├── memory.py              # define_memory(scope="agent")
├── identity.py            # define_identity(auth=auth.langsmith_api_key())
├── skills/
│   ├── working-with-google-sheets/   # sheet I/O workflow over the tools
│   ├── editing-slide-chunks/         # + references/writing-styles/*
│   └── editing-research-notes/
├── tools/
│   ├── sheets.py          # open_course, commit_*, list_course_tabs — in-process
│   ├── presentation.py    # present_files, measure_file
│   └── _sheetlib/         # byte-identical copies of the sheet scripts
├── middleware/audit.py    # async commit guard
├── selftest.py            # 45 offline checks, no API key
├── livetest.py            # real model + real sheet, incl. the approval gate
└── pyproject.toml
```

No `sandbox/`, no `channels/`, no `schedules/`, no `connectors/` — MDA turns
those on by their presence, and an unused directory is a capability you pay
attention to for nothing.

## Why a sandbox, and how the credential stays out of it

An earlier version of this build declined MDA's sandbox. The reasoning looked
sound: a running per-thread sandbox does not inherit deployment secrets, and
baking a key into the snapshot puts it in every thread's image, readable by an
agent that an injected instruction in a course sheet could redirect. No shell
meant no way to run a script, so sheet access became four in-process tools.

That reasoning had a hole. It assumed the credential had to *reach* the sandbox.

The **sandbox auth proxy** intercepts egress and injects credentials resolved
from LangSmith workspace secrets, outside the box. Scripts send a bare,
unauthenticated request; the header is filled in on the way out. So the box
holds no key, and there is nothing for an injected instruction to read — a
stronger position than the Managed Agents build, which mounts
`service_account.json` at `/uploads/` where the model can read it.

`sandbox/__init__.py` declares it. Verified against the live API by
`sandboxprobe.py`: no credential visible inside, a bogus sheet id refused, and a
bare `gspread` read and write both succeeding.

What this costs, stated plainly:

- **Filesystem permissions are gone.** MDA sets `permissions = []` whenever a
  sandbox is declared — silently, not as an error — because deepagents cannot
  apply them to a backend that also executes. This build therefore declares
  none, rather than declaring rules that read as protection and do nothing.
- **The editor subagent's boundary is prose.** It shares the sandbox and shell,
  so nothing mechanically stops it running a commit script. Its prompt says so
  in as many words, and `selftest.py` asserts that sentence stays there.
- **No approval interrupt.** Both commit scripts write to a *new* tab and never
  modify a source tab, so an unwanted commit costs a tab someone deletes. The
  Managed Agents build has no hard gate either. The rule that a commit needs the
  user's own words lives in `instructions.md`, as judgment rather than a lock.

## What each MDA capability is used for

| Capability | File | Used for |
|---|---|---|
| Agent definition | `agent.py` | model, tools, subagent, permissions, `interrupt_on` |
| Instructions | `instructions.md` | the coordinator's system prompt |
| Skills | `skills/` | sheet I/O workflow + the two editing skills, loaded on demand |
| Tools | `tools/` | sheet I/O, `present_files`, `measure_file` |
| Middleware | `middleware/audit.py` | async guard refusing a commit with no open course |
| Memory | `memory.py` | `scope="agent"` — shared editorial standards |
| Identity | `identity.py` | LangSmith API key auth |
| HITL | `agent.py` | both commit tools pause for approval |
| Sandbox | — | **deliberately absent** |
| Channels / schedules / connectors | — | not needed; MCP connectors were removed from MDA |

## Changing the model

`CCE_MODEL` in `.env`, no code change:

```bash
CCE_MODEL=google_genai:gemini-3.8-flash   # default
CCE_MODEL=openai:gpt-5.6-luna
CCE_MODEL=anthropic:claude-sonnet-5
```

Use the provider-prefixed form. Python's Google slug is `google_genai:`
(TypeScript uses `google-genai:`). Whichever you pick, that provider's API key
must be in `.env` too or `mda deploy` fails preflight.

## Running it

```bash
# from the repo root, with the deepagents venv on PATH
mda build .                 # compile to .mda/build
mda dev .                   # local LangGraph dev server + Studio (needs uv)
mda deploy .                # needs MDA beta access on the workspace
```

`.env` needs `LANGSMITH_API_KEY` (to deploy), the provider key for `CCE_MODEL`,
and `GDRIVE_SA_B64` (the base64 service account the sheet tools use). The file
is gitignored and never enters the build archive. **Fill in the values
yourself** — they are not written here.

## Tests

```bash
python selftest.py     # 45 checks, offline, no API key, no sheet
python livetest.py --sheet-url "<a course sheet>" [--all-topics]
```

`selftest.py` covers what fails silently: the tools' reading and writing of the
virtual filesystem, block counting and diff selection, the commit guard, the
shape of the agent definition (editor has no sheet tools, both commits are
interrupt-gated, baseline is write-protected), and a byte-for-byte check that
`tools/_sheetlib/` still matches the Deep Agents build's copies — that diff is
the only thing keeping the block grammar from drifting between builds.

`livetest.py` drives the *compiled* graph with a real model against a real
sheet, in three scripted turns: open and edit, then a **rejected** commit
(nothing may reach the sheet), then an **approved** one (the tab must appear
with the edited text in it). It creates one tab, reads it back, and deletes it.
`--all-topics` exercises delegation.

One harness-only liberty: the managed runtime leaves the checkpointer to the
LangGraph server, so a directly-imported graph has none and cannot interrupt.
`livetest.py` attaches an `InMemorySaver` to stand in. Everything else is what
`mda deploy` would run.

## Verification

**Offline — `selftest.py`, 45 checks, passing.**

**Live, against the real course sheet** (*AI test Copy of Course - Plumbing
System Fundamentals*), driving the compiled graph:

| | Gemini 3.8 Flash (default) | GPT-5.6 Luna |
|---|---|---|
| single topic | **16/16** | **16/16** |
| all topics, with delegation | **22/22** | 21/22 |
| wall clock (single / all) | 91s / 260s | 81s / 202s |
| multi-topic trim | 1504→894, 1432→694, 1471→983 words | 1504→253 words |
| editors delegated | 3, one per topic | 3, one per topic |

Both: `open_course` routed to the research-notes-only path without being told,
the baseline snapshot landed alongside the pulled files, every `###LO ID:` block
survived, `present_files` was called unprompted, the **rejected** commit left
the tab count unchanged at 31, and the **approved** commit produced one new tab
with 9 rows and all 17 original columns, the edited text present and the
untouched rows' notes intact. Scratch tabs deleted after each run.

Luna's one failure on the multi-topic run is `every topic file was edited`
(2/3). All three editors ran and all three called `edit_file`; one file ended
byte-identical to its baseline anyway. That is a model-behaviour difference, not
a build defect — Gemini gets 3/3 on the identical code — and it is one reason
the default model is Gemini.

### A real bug this testing caught

The editor subagent's permissions originally ended in a catch-all
`deny /**` with only `/workspace/**` and `/skills/**` allowed. That silently
denied reads of `/memories/agent/AGENTS.md` — which the editing skills instruct
every editor to read before its first edit. The visible symptom was subtle:
delegation "worked", but only **1 of 3** editors ever called `edit_file`, and
the coordinator quietly did the rest itself. Allowing `/memories/**` (and the
runtime's own `/large_tool_results/**`, `/conversation_history/**`) took it to
3 of 3. A permission that blocks an instruction the agent was given produces
under-work, not an error.

## Three bugs the deployment found that local testing could not

Each of these passed every offline check and every in-process live run, and
failed only once hosted. They are the reason this build was worth deploying
rather than reasoning about.

**1. `ToolRuntime` is not the runtime MDA injects.** Annotating a tool
parameter `runtime: ToolRuntime` fails pydantic validation on MDA, which
injects its own `_ManagedRuntime`. Every tool that asked for one died;
`list_course_tabs`, the only tool without one, was untouched. LangGraph's
ToolNode converts the failure into "Error invoking tool ... Please fix the
error and try again" with the message stripped and nothing logged, so the
symptom was tools that silently didn't work.

MDA does export `ManagedDeepAgentRuntime` for this, but it has no
`tool_call_id`, which the `Command` + `ToolMessage` construction needed. The
fix was to stop asking for a runtime at all and reach state through
`StateBackend()`, which resolves from ambient graph config the way the
framework's own `read_file` / `write_file` do — no annotation, works under
either runtime.

**2. `ReadResult` has no `.content`.** The bytes live in `.file_data`, a
`FileData` mapping. Reading the wrong attribute returned `None` for every file
that existed, so `commit_research_notes` reported "no `/workspace/manifest.json`
— open_course has not run" while `ls` listed the manifest right there.
`GlobResult.matches` holds `FileInfo` mappings, not path strings — the same
class of guess.

**3. The test fake agreed with the bug.** `FakeBackend.read` returned an
invented `.content`, matching the broken implementation, so 45 offline checks
passed against a build that could not read a single file. The fake now mirrors
the real result shapes, and `test_backend_contract` asserts that it does —
`ReadResult` really has `file_data` and really has no `content`. A fake that
agrees with a wrong implementation tests nothing, and this one proved it.

## The skill files are shared, not copied

`skills/working-with-google-sheets/scripts/` holds the same eight files as the
Deep Agents build, byte for byte, and `selftest.py` fails if any of them drift.
`_common.py` carries the one change that makes this possible: `authorize()`
prefers a real service account key where one is reachable, and falls back to
`AnonymousCredentials` only when none is. Absence of a key is the sandbox case,
not an error — so a single file serves the local build, the Managed Agents
container, and this one.

`present.py` replaces the `present_files` and `measure_file` tools. Same
contract: the agent names paths, the script reads the real bytes and prints
them with a `.baseline/` diff and counted facts. `--measure` gives counts alone.

## Known gaps

- **No per-user memory.** MDA's `define_memory` takes only `scope="agent"` or
  `"none"` — one tree shared by every caller, confirmed in the installed
  package source ("A privileged-writer policy that closes this is future
  work"). The other two builds have a per-user memory file; here that scope
  does not exist, so `instructions.md` tells the agent to keep only team-wide
  standards and never per-person facts.
- **Slide chunks untested against a real sheet.** `commit_workspace.py` and the
  slide-chunks path of `prepare_workspace.py` are exercised by the Deep Agents
  build's suite but have never run here against a live sheet. Every live run so
  far has been research notes, the lower-blast-radius half.
- **Nothing prevents a destructive re-prepare.** A prepare script overwrites
  every workspace file with what is currently on the sheet, discarding uncommitted
  edits. The skill says so twice and the agent has been observed running prepare a
  second time mid-session anyway. It did no harm on that run, and `deployedtest.py`
  now asserts the committed notes differ from the source so a lost edit fails
  loudly — but the guard is prose, not code. A `--force` requirement in
  `prepare_*.py` when a non-empty `.baseline/` exists would close it properly.
- **`disconnecttest.py` still assumes the removed interrupt.** Its Phase 1
  (a run surviving a dead client) is unaffected and remains valid; Phases 2-3
  test an approval gate this build no longer has.
- **Programmatic invocation works, but is undocumented.** The docs call it
  undocumented during public beta, which is not the same as absent: the Agent
  Server is a LangGraph server and `langgraph_sdk` drives it with the workspace
  API key `identity.py` declares. `deployedtest.py` does exactly that, so a
  Streamlit page could too. Undocumented means it can change without notice —
  treat it as a finding, not a contract.
- **`livetest.py` does not capture assistant text** when a provider returns
  content as block lists rather than a string, so transcripts show tool calls
  but not the model's prose. Cosmetic, but it makes a failed run harder to read.
