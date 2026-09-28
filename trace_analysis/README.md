# trace_analysis

Readable session logs for the `course-content-editor-mda` deployment, and a
place to keep what reading them turns up.

## Why this exists

LangSmith renders a trace as a run *tree*. That is the wrong shape for judging
an agent: a subagent is a node you open separately, so the one thing you want —
what the whole system did, in the order it did it — is the thing the tree hides.
A single large session here is ~1,000 runs, of which roughly 90% are middleware
`chain` wrappers carrying no conversation at all.

This renders the same trace as a transcript: chronological, subagents interleaved
where they actually ran, JSON scaffolding stripped.

## Use

```bash
.venv/Scripts/python.exe -m trace_analysis list --limit 25   # what sessions exist
.venv/Scripts/python.exe -m trace_analysis pull --last 10    # write their logs
.venv/Scripts/python.exe -m trace_analysis spans             # step spread per fan-out
```

`pull` also takes `--trace <id>`, `--since 2026-09-01`, and `--refresh` (ignore
the cache and re-fetch).

Then invoke the **`analyze-mda-sessions`** skill to read them and record findings.

## Layout

| path | what |
|---|---|
| `sessions/*.md` | one log per session, named so filename order is chronological |
| `findings/FINDINGS.md` | the durable ledger of what to change |
| `.cache/*.json` | raw runs, gitignored — re-rendering never re-hits the API |

## How it works

`dotted_order` is LangSmith's hierarchical-chronological key, and a child's value
is prefixed by its parent's. Sorting by it interleaves subagent work in place;
prefix-matching a `task` tool run's value finds exactly the runs performed under
that brief, which is what makes the per-span step counts trustworthy.

Two things worth knowing about the data:

- **`run["status"]` under-reports failure.** A script that fails inside the
  sandbox shell exits cleanly having printed its error, so the run is `success`
  and the only evidence is in the output text. Both channels are merged; `grep`
  exiting 1 on no-match is excluded, since that is how you ask whether something
  is absent.
- **Truncation is always announced.** Every elision states what it removed, so a
  cut is visible rather than silent. Commands, paths and patterns are never
  trimmed — an argument you cannot read is a step you cannot judge.

## Scope

Read-only, and deliberately outside `agents/`: this analyses the deployment, it
is not part of it and must never ship inside the build. The LangSmith key is read
from the environment or `agents/course_content_editor_mda/.env`, and is never
logged.
