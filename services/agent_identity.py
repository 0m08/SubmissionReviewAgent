"""Who the agent is working for, in two forms that go to two different places.

The Managed Deep Agent has one shared memory tree. `define_memory(scope="user")`
— private per-caller memory — raises today:

    `scope="user"` is not supported yet. Private per-caller memory is not part
    of the launch contract; declare `scope="agent"` for deployment-shared
    memory.                       (managed_deepagents/_memory/slices.py:83)

So per-person memory is files inside the shared tree, addressed by an id the
agent is handed and trusted to use — an honour system, not an access boundary.
See the Memory section of the agent's `instructions.md` for the rules that ride
on that, and the note at the bottom of this file for what it does not protect.

## Why the email does not go to the agent

Two destinations, two different stores:

- **`run_metadata()` → LangSmith traces.** The email belongs here. Traces live
  in the LangSmith workspace behind its own access control, and being able to
  ask "what did this person's runs do" is the reason to put it there at all.
- **`run_context()` → the agent, and from there a path in shared memory.** The
  email must not go here. Every caller of the deployment can read that tree, so
  a directory of `firstname@company.com.md` next to eight colleagues is exactly
  the "never store personal data in memory" case the MDA docs warn about. The
  opaque id carries the same *distinctness* with none of the identity, and the
  email→id mapping stays in this process.
"""

from __future__ import annotations

import hashlib

__all__ = ["editor_id", "run_context", "run_metadata"]

_PREFIX = "ed_"
_WIDTH = 12


def editor_id(email: str | None) -> str | None:
    """A stable, opaque id for one logged-in person, or None if not signed in.

    Stable across sessions and machines for the same address, so the memory
    file written on Monday is the one found on Friday. Case and surrounding
    whitespace are normalised first, because `Dilip@…` and `dilip@…` are one
    person and would otherwise be two.

    Pure and unconfigurable on purpose. The id is the name of a file people
    accumulate work in, so anything that could change how it is derived — a
    salt, a setting, a version prefix — is a way to silently orphan all of
    them at once. If this ever has to change, migrate the files deliberately.
    """
    if not email or not email.strip():
        return None
    digest = hashlib.sha256(email.strip().lower().encode()).hexdigest()
    return f"{_PREFIX}{digest[:_WIDTH]}"


def run_context(email: str | None) -> dict[str, str]:
    """Runtime context for a run — what the *agent* is allowed to know.

    Goes to `client.runs.stream(context=...)`, arrives as `runtime.context` in
    the deployment, and is read there by the `editor_identity` middleware.
    Deliberately the id alone: anything added here can end up written into
    shared memory by a model that thought it was being helpful.
    """
    resolved = editor_id(email)
    return {"editor_id": resolved} if resolved else {}


def run_metadata(email: str | None) -> dict[str, str]:
    """Trace metadata for a run — what *we* need to debug and attribute.

    Goes to `client.runs.stream(metadata=...)` and lands on the LangSmith run,
    where it is filterable. The email is here and only here.
    """
    meta: dict[str, str] = {"app": "streamlit", "surface": "course-content-editor-mda"}
    if email and email.strip():
        meta["user_email"] = email.strip()
    resolved = editor_id(email)
    if resolved:
        meta["editor_id"] = resolved
    return meta


# ---------------------------------------------------------------------------
# What this does not protect against
# ---------------------------------------------------------------------------
#
# 1. The id is a *label*, not a key. Any caller can read or write any editor's
#    memory file; nothing in the deployment enforces the boundary. It holds
#    because the agent follows its instructions, which is why the only things
#    that belong in those files are working preferences a colleague reading
#    them would find boring.
# 2. An id is recoverable. Anyone holding both the memory tree and the team
#    roster can hash the roster and match — 44 addresses is under a millisecond
#    of work. Salting the hash would close that, and was deliberately left out:
#    reading the tree requires a workspace key, which means a colleague, who
#    already has the roster. It would defend the team's addresses from the
#    team, at the cost of a secret whose loss silently orphans every file. What
#    the hash is actually for is keeping raw addresses out of the shared store.
# 3. A user can claim in chat to be someone else. The middleware states that
#    identity comes from context and that conversation text does not change it,
#    which handles the honest case and not a determined one.
#
# The day something belongs in an editor's file that its owner would not say in
# a team channel, this design is the wrong one and the answer is to wait for
# `scope="user"` rather than to harden this.
