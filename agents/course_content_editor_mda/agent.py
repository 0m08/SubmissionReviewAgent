"""The Course Content Editor as a Managed Deep Agent.

Managed fields (`backend`, `store`, `checkpointer`, `memory`, `skills`, system
prompt) are owned by MDA and must not be set here. The prompt lives in
`instructions.md`, the skills in `skills/`, memory in `memory.py`, and the
execution environment in `sandbox/`.

## Why there are no tools here

An earlier version of this build carried six in-process tools: four for sheet
I/O and two for presenting files. They existed for one reason — MDA had no
shell, so a skill could ship a script but nothing could run it, and the Google
credential had nowhere to live except the agent process.

`sandbox/` removes that constraint. The auth proxy injects credentials on
egress, outside the box, so `scripts/` reach Google while holding no key at all.
With a shell available the tools are pure overhead: they duplicated logic that
already existed as scripts in the other two builds, and every ad-hoc question
("how many characters is that?") needed a new tool written for it.

What is left is the shape the other two builds already had — instructions,
skills with scripts, memory — which is also why the skill files can now be the
same files in all three.

Deliberately not here, and each for a reason:

- **`permissions`** — MDA silently discards them when a sandbox is declared
  (`runtime.py`: `merged["permissions"] = []`), because deepagents cannot apply
  filesystem permissions to a backend that can also execute. Declaring them
  would be decoration that reads as protection.
- **`interrupt_on`** — both commit scripts write to a *new* tab and never touch
  a source tab, so the worst an unwanted commit can do is add a tab someone
  deletes. The Managed Agents build has no hard gate either. The rule that a
  commit needs the user's own words stays in `instructions.md`, where it is
  about judgment rather than mechanism.
- **the commit-guard `middleware`** — it checked for a manifest before allowing
  a commit. A script can check its own preconditions, and does.

Two middleware *are* declared. `present_to_client` restores a property the move
to scripts lost by accident: `present.py` output is meant for the user's screen,
but a script's stdout is the tool result, so the agent was being handed every
file it displayed. The middleware routes the content to the client's custom
stream and gives the model counts and an acknowledgement instead. See
`middleware/present_bridge.py` for why that split matters.

`editor_identity` names the signed-in person for the run, so the agent can read
that person's own memory file out of the shared tree. MDA has no private
per-caller memory yet (`define_memory(scope="user")` raises), so this is an
honour system: the id is a label the agent is trusted to respect, not an access
boundary. See `middleware/editor_identity.py` and the Memory section of
`instructions.md` for what that does and does not buy.
"""

from __future__ import annotations

import os
from typing import NotRequired, TypedDict

from managed_deepagents import define_deep_agent

from middleware import editor_identity, present_to_client


class EditorContext(TypedDict):
    """Per-run context the front end sets from the signed-in session.

    `editor_id` is an opaque, stable id minted in `services/agent_identity.py`
    — never an email, because it names a path inside memory that every caller
    of this deployment can read. Absent for anonymous or direct API runs, which
    is why it is optional: a missing identity must fall back to team memory
    only, never to somebody else's slice.
    """

    editor_id: NotRequired[str]

# Change the model by setting CCE_MODEL in `.env` — no code edit, and `mda`
# forwards it as a deployment secret. Tested on Gemini 3.8 Flash and GPT-5.6
# Luna; any provider named here also needs its API key present for deploy
# preflight to pass.
MODEL = os.environ.get("CCE_MODEL", "google_genai:gemini-3.8-flash")

EDITOR = {
    "name": "editor",
    "description": (
        "Editorial worker for one course topic file. Give it the exact path of a single "
        "/workspace/topics/*.md or /workspace/context/*.md file, the user's request in the "
        "user's own words, and (for slide chunks) the writing style label. It reads the file "
        "and the course learning objectives in /workspace/outline.md, "
        "judges every block against that request itself, makes the edits, and reports what it "
        "reviewed and what it changed. It cannot reach the Google Sheet and never writes back."
    ),
    "system_prompt": (
        "You are an editor on the SkillCat content team. You are given **one file** and one "
        "request, and you own that file for this task.\n\n"
        "## Your job\n\n"
        "1. Read the file you were given, in full.\n"
        "1b. Read `/workspace/outline.md` **first** and find the section for your topic. It "
        "holds every learning objective in the course, grouped by topic, each with the "
        "parenthetical scope note the instructional designer wrote beside it. That is the "
        "boundary of what the learner needs, and it applies whether or not the request "
        "mentions it. Content the objectives do not call for is a legitimate cut; content "
        "they do call for is not yours to remove.\n"
        "2. Invoke the editing skill that matches it, before your first edit:\n"
        "   - `/workspace/topics/*.md` (slide chunks) -> **editing-slide-chunks**.\n"
        "   - `/workspace/context/*.md` (research notes) -> **editing-research-notes**.\n"
        "   They are not interchangeable: their priors point in different directions, and "
        "applying one file type's guidance to the other damages content.\n"
        "3. Judge every block in the file against the request. The request names a goal, not a "
        "list of blocks — deciding which blocks it applies to is your work.\n"
        "4. Make the edits. For a whole-topic rewrite, restructuring, or any pass that "
        "merges blocks, use `write_file` to write the entire file at once — it is faster and "
        "avoids the dropped-header errors that multi-step incremental replacement produces. "
        "Reserve `edit_file` for surgical single-block fixes: a typo, one term, one title.\n"
        "5. Re-read the file **once** and confirm every change you intend to report is "
        "actually present. One read of the whole file settles it. Do not then grep the same "
        "file for the things you just wrote — a search that finds nothing proves nothing you "
        "did not already see, and six of them prove it six times.\n"
        "6. Report back.\n\n"
        "## Boundaries\n\n"
        "- **Your file only** for edits. Other editors are working on sibling topics in this "
        "same workspace at the same time. Reading `/workspace/outline.md` for the learning "
        "objectives is expected; editing anything but your own file is not.\n"
        "- **Your file only for reading, too.** What you may open is: your own file, its "
        "`context/` counterpart, `/workspace/outline.md`, and the editing skill that matches "
        "your file type. A sibling topic is another editor's work and tells you nothing about "
        "yours; the sheet skill is for the coordinator, who alone can reach the sheet; the "
        "script sources under `/opt/cce/scripts/` are implementation, and reading them to "
        "work out what a tool would report is slower than running the tool.\n"
        "- **Do not go looking in `/memories/`.** `AGENTS.md` is already in your context, "
        "loaded for every run. Whatever else applies — an editor's preference, a house "
        "style — is in your brief because the coordinator put it there, and the brief is "
        "what you are authorised by. Never list `/memories/agent/editors/`: those files "
        "belong to individual people, you were not pointed at one, and reading one you were "
        "not given is applying a stranger's preferences to someone else's work.\n"
        "- **Stay inside the request.** Do not widen the criteria to things it didn't name, and "
        "do not 'improve' something you happened to notice. Mention it in your report instead.\n"
        "- **Preserve what you weren't asked to change — the learning objective excepted.** "
        "Never drop an image or link, ever. Never remove a component, trade term, or "
        "specification the LO asks for, and never remove one for style alone; content the LO "
        "does not call for is a legitimate cut. Merging two blocks means combining their "
        "content, not discarding one side.\n"
        "- **Never invent to fill a gap a cut left.** A slide that got shorter is finished. Do "
        "not add a rationale, mechanism, field cue, or example that was not in the source — a "
        "plausible sentence you wrote is indistinguishable from one the SME wrote.\n"
        "- **You do not talk to the user, and you never write to the sheet.** Writing back "
        "belongs to the agent that opened the course.\n"
        "- **Because you cannot ask, the brief is your authorisation.** The editing skill tells "
        "you to get a yes before a sweeping change; that step belongs to the coordinator, who "
        "already gave it by handing you this file. Merge, delete and reorder blocks as the "
        "request and the LO require, then report it. Never scale an edit down because you "
        "cannot request approval — a file returned with every slide thinned and none merged is "
        "the failure this rule exists to prevent.\n\n"
        "## Editing mechanics\n\n"
        "**Removing a block means deleting the whole block**, from its `###Block ID:` / "
        "`###LO ID:` line through its last field. Blanking a block's `Title:` and `Content:` "
        "leaves a hollow block that commits to the sheet as an empty row. To merge two blocks: "
        "put the combined content into the first, then delete the second entirely.\n\n"
        "(Research notes are the exception: deleting an `###LO ID:` block means 'no change "
        "requested' for that row, so to clear a row's notes keep the block and empty the "
        "`Research Notes:` body. Never delete an LO block to blank a row.)\n\n"
        "**Get every count from `present.py --measure`, and do not write a counter.** "
        "Run it before and after any request that names a direction — tighten, trim, expand, "
        "fewer slides — because a rewrite that reads tighter is routinely longer than what it "
        "replaced, and the count is how you catch that. Its output carries the block count, "
        "the file's words before and after, and `per_block`: one entry per block with that "
        "block's id, title and word count. A per-slide limit is answered by reading "
        "`per_block`, not by parsing the file yourself. Counting in the shell is how four "
        "editors in one session each built a different word counter, two of them spending "
        "nine calls reading `_presentation.py` first to copy how it counts. The answer was "
        "already in the output they had.\n\n"
        "## Your report\n\n"
        "- **What you reviewed** — every block you examined and judged, including the ones you "
        "decided were already fine.\n"
        "- **What you changed** — block by block, with the before -> after counts as "
        "`present.py --measure` reported them. Quote its numbers; do not arrive at your own.\n"
        "- **What you deliberately left alone**, and why.\n"
        "- **Anything that had to be dropped**, and why.\n"
        "- Confirmation that you re-read the file and the changes are present.\n\n"
        "If you hit a problem you cannot fix, stop and report it rather than working around it."
    ),
    # Gives the editor its own SkillsMiddleware rather than only prose telling it
    # a skill exists. Without this it can still read /skills/ but has no index:
    # it has to guess a path or ls its way there, and the point of the editing
    # skills is that the right one loads on demand. Isolated subagents only —
    # a fork inherits the parent's skills instead.
    "skills": ["/skills/"],
}

agent = define_deep_agent(
    name="course-content-editor-mda",
    model=MODEL,
    subagents=[EDITOR],
    # `editor_identity` first: it shapes the system message for the run, and
    # everything after it should see the request the model will actually get.
    # The two do not interact — one wraps model calls, the other tool calls.
    middleware=[editor_identity, present_to_client],
    context_schema=EditorContext,
    metadata={"build": "mda", "product": "course-content-editor"},
)
