"""One-time setup for the Course Content Editor Managed Agent.

Uploads the two skills, creates the environment, and creates (or updates) the
agent. IDs are persisted to deploy_state.json so re-runs reuse existing
resources instead of orphaning them — the mandatory Managed Agents pattern:
create the agent once, reference it by ID from every session.

Forked from agents/slide_chunks_skills/deploy.py (the trusted, already-live
checklist agent's deploy script). This agent is deliberately separate: its
own skills/ directory, its own environment, its own deploy_state.json. It
does not share resources with the checklist agent.

Run this when you first deploy, or after editing a skill / the agent config:

    python agents/course_content_editor/deploy.py

Reads the agent's name / model / system prompt / tools from
managed_agent_config.yaml so that file stays the single source of truth.
Expects ANTHROPIC_API_KEY in the environment (or a repo-root .env).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    _here = Path(__file__).resolve()
    for _parent in _here.parents:
        if (_parent / ".env").exists():
            load_dotenv(_parent / ".env")
            break
except ImportError:
    pass

import yaml
import anthropic
from anthropic.lib import files_from_dir


SKILL_DIR = Path(__file__).parent / "skills"
CONFIG_PATH = Path(__file__).parent / "managed_agent_config.yaml"
EDITOR_CONFIG_PATH = Path(__file__).parent / "editor_agent_config.yaml"
STATE_PATH = Path(__file__).parent / "deploy_state.json"

# dir name under skills/  ->  human-readable title shown in the console.
SKILLS = {
    "working-with-google-sheets": "Working with Google Sheets (Course Content Editor)",
    "editing-slide-chunks": "Editing Slide Chunks",
    "editing-research-notes": "Editing Research Notes",
}

# The editor worker gets only the editing skills — it has no bash tool, so the
# sheets scripts would be unusable, and omitting them keeps its context narrow.
EDITOR_SKILLS = ("editing-slide-chunks", "editing-research-notes")

ENVIRONMENT_NAME = "course-content-editor-env"

_HASH_EXCLUDE_PARTS = {"__pycache__", ".DS_Store", ".git", ".gitignore"}


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"skills": {}}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"  saved state -> {STATE_PATH.name}")


def _hash_skill_dir(path: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(path.rglob("*")):
        if not f.is_file():
            continue
        if any(part in _HASH_EXCLUDE_PARTS for part in f.parts):
            continue
        rel = f.relative_to(path).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def upload_skills(client: anthropic.Anthropic, state: dict, force_reupload: bool = False) -> list[dict]:
    skill_refs = []
    for dir_name, title in SKILLS.items():
        path = SKILL_DIR / dir_name
        if not path.exists():
            print(f"ERROR: skill directory {path} not found.", file=sys.stderr)
            sys.exit(2)
        current_hash = _hash_skill_dir(path)
        existing = state["skills"].get(dir_name)

        if existing and existing.get("hash") == current_hash and not force_reupload:
            print(f"  {dir_name}: unchanged, skipping")
        elif existing:
            version = client.beta.skills.versions.create(
                skill_id=existing["skill_id"], files=files_from_dir(str(path)),
            )
            existing["version"] = version.version
            existing["hash"] = current_hash
            print(f"  {dir_name}: new version {version.version}")
        else:
            skill = client.beta.skills.create(
                display_title=title, files=files_from_dir(str(path)),
            )
            state["skills"][dir_name] = {
                "skill_id": skill.id,
                "version": skill.latest_version,
                "hash": current_hash,
            }
            print(f"  {dir_name}: created {skill.id}")
        ref = state["skills"][dir_name]
        skill_refs.append({"type": "custom", "skill_id": ref["skill_id"], "version": "latest"})
    return skill_refs


def ensure_sa_uploaded(client: anthropic.Anthropic, state: dict, force: bool = False) -> str | None:
    """Optional: the agent is unusable against Google Sheets without a service
    account, but deployment shouldn't hard-fail if one isn't configured yet —
    lets you deploy and inspect the agent before wiring credentials."""
    if state.get("sa_file_id") and not force:
        print(f"  service account: reusing {state['sa_file_id']}")
        return state["sa_file_id"]

    sa_json = os.environ.get("GDRIVE_SA_JSON")
    sa_b64 = os.environ.get("GDRIVE_SA_B64")
    if not sa_json and not sa_b64:
        print("  service account: not configured (set GDRIVE_SA_JSON or GDRIVE_SA_B64) "
              "— deploying anyway, but chat.py will need a credential to read/write sheets")
        return None

    sa_bytes = sa_json.encode("utf-8") if sa_json else base64.b64decode(sa_b64, validate=True)
    try:
        sa = json.loads(sa_bytes)
    except json.JSONDecodeError as exc:
        print(f"ERROR: SA payload isn't valid JSON: {exc}", file=sys.stderr)
        sys.exit(2)

    uploaded = client.beta.files.upload(
        file=("service_account.json", io.BytesIO(sa_bytes), "application/json"),
    )
    state["sa_file_id"] = uploaded.id
    state["sa_client_email"] = sa.get("client_email", "<unknown>")
    print(f"  service account: uploaded {uploaded.id}")
    print(f"    client_email: {state['sa_client_email']}")
    print("    (share each target Google Sheet with that address as Editor)")
    return uploaded.id


SHARED_MEMORY_STORE_NAME = "cce-shared-editorial-standards"


def ensure_shared_memory_store(client: anthropic.Anthropic, state: dict) -> str:
    """The one workspace-wide memory store every session attaches read-write.

    Created once here (like the SA file) rather than lazily from the app —
    it's a singleton shared across every user, so creating it lazily from
    Streamlit would risk two concurrent first-time users racing to create
    two stores with the same intended name. Per-user stores are still
    created lazily from the app itself (course_content_editor.py), since
    those are genuinely per-user and don't have that race.
    """
    if state.get("shared_memory_store_id"):
        print(f"  shared memory store: reusing {state['shared_memory_store_id']}")
        return state["shared_memory_store_id"]
    store = client.beta.memory_stores.create(
        name=SHARED_MEMORY_STORE_NAME,
        description=(
            "Team-wide editorial standards for the Course Content Editor: durable, "
            "recurring preferences about how slide chunks and research notes should "
            "be written, that apply across all users and courses (not one-off "
            "corrections, and not any single course's writing style — those live in "
            "the editing-slide-chunks skill's writing-styles reference instead)."
        ),
    )
    state["shared_memory_store_id"] = store.id
    print(f"  shared memory store: created {store.id}")
    return store.id


def ensure_environment(client: anthropic.Anthropic, state: dict) -> str:
    if state.get("environment_id"):
        print(f"  environment: reusing {state['environment_id']}")
        return state["environment_id"]
    env = client.beta.environments.create(
        name=ENVIRONMENT_NAME,
        config={
            "type": "cloud",
            "networking": {
                "type": "limited",
                "allowed_hosts": ["sheets.googleapis.com", "oauth2.googleapis.com"],
                "allow_package_managers": True,
            },
            "packages": {"pip": ["gspread", "pandas"]},
        },
    )
    state["environment_id"] = env.id
    print(f"  environment: created {env.id}")
    return env.id


def preflight(state: dict) -> None:
    """Refuse to deploy from a state file that would overwrite the wrong agent.

    Both agents run the same model, so nothing but the name distinguishes them
    in the console, and a crossed id is invisible until sessions stop starting.
    Two conditions make that possible, and both are cheap to rule out here
    rather than after the API call has already rewritten a live agent.
    """
    coordinator, editor = state.get("agent_id"), state.get("editor_agent_id")
    if coordinator and editor and coordinator == editor:
        print(f"ERROR: deploy_state.json points agent_id and editor_agent_id at the "
              f"same agent ({coordinator}). Deploying would overwrite the "
              f"coordinator with the editor's config. Fix the state file: the "
              f"coordinator is the one whose multiagent roster is non-empty.",
              file=sys.stderr)
        sys.exit(2)

    # The coordinator rosters *itself* alongside the editor, so the two configs
    # sharing a name is not a cosmetic clash: callable agents are resolved by
    # name, and the API rejects the session outright with
    #   "Failed to resolve callable agents: duplicate callable agent name"
    coord_name = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["name"]
    editor_name = yaml.safe_load(EDITOR_CONFIG_PATH.read_text(encoding="utf-8"))["name"]
    if coord_name == editor_name:
        print(f"ERROR: both configs are named {coord_name!r}. The coordinator "
              f"rosters itself next to the editor, so identical names make every "
              f"session fail to resolve its callable agents. Give them distinct "
              f"names in managed_agent_config.yaml / editor_agent_config.yaml.",
              file=sys.stderr)
        sys.exit(2)


def _upsert_agent(client: anthropic.Anthropic, state: dict, cfg: dict,
                  skill_refs: list[dict], id_key: str, version_key: str,
                  label: str, multiagent: dict | None = None) -> str:
    """Create or update one agent, tracking its id/version under the given keys."""
    agent_kwargs = dict(
        name=cfg["name"],
        model=cfg["model"],
        description=cfg.get("description", ""),
        system=cfg["system"],
        tools=cfg.get("tools", []),
        skills=skill_refs,
    )
    # multiagent is optional — only set it when there is a roster, so an agent
    # without one doesn't send a spurious null. The editor must never get one:
    # roster members cannot themselves delegate.
    if multiagent:
        agent_kwargs["multiagent"] = multiagent
    if state.get(id_key):
        # update() uses optimistic concurrency — it requires the version being
        # updated FROM, not just the new field values.
        #
        # A stale version in deploy_state.json is normal and harmless: someone
        # else deployed since this checkout last did. Retry from the live
        # version rather than failing, because the manual workaround — editing
        # the version number in deploy_state.json by hand — puts a human one
        # typo away from pointing an id_key at the wrong agent, which is
        # exactly how the coordinator once got overwritten with the editor's
        # config (both then answered to the same callable name, and every
        # session creation failed with "duplicate callable agent name").
        try:
            agent = client.beta.agents.update(
                state[id_key], version=state[version_key], **agent_kwargs
            )
        except anthropic.APIStatusError as exc:
            live = client.beta.agents.retrieve(state[id_key])
            if live.version == state.get(version_key):
                raise  # not version drift — a real failure
            print(f"  {label}: state had version {state.get(version_key)}, "
                  f"live is {live.version} (someone else deployed) — retrying")
            agent = client.beta.agents.update(
                state[id_key], version=live.version, **agent_kwargs
            )
        print(f"  {label}: updated {agent.id} -> version {agent.version}")
        # The id under this key must be the agent this config describes. If a
        # bad merge or hand-edit crossed the two ids, the update silently
        # rewrites the wrong agent — caught here, after the fact, because the
        # name is the only field that distinguishes them (both are haiku).
        if agent.name != cfg["name"]:
            print(f"ERROR: {label} {agent.id} is named {agent.name!r} after an "
                  f"update that set {cfg['name']!r}.", file=sys.stderr)
            sys.exit(2)
    else:
        agent = client.beta.agents.create(**agent_kwargs)
        state[id_key] = agent.id
        print(f"  {label}: created {agent.id}")
    state[version_key] = agent.version
    return agent.id


def ensure_editor_agent(client: anthropic.Anthropic, state: dict,
                        skill_refs: list[dict]) -> str:
    """The editorial worker rostered by the coordinator.

    Narrowed in two real ways: it gets only the editing skills (no sheets
    skill, so it has no commit scripts or credentials to reach the Google
    Sheet with), and it has no multiagent block, so it cannot delegate
    further. All sheet I/O stays with the coordinator.

    It is NOT tool-restricted, despite what this docstring claimed until
    2026-09-04. editor_agent_config.yaml has enabled read, edit, glob, grep,
    bash AND write since the config was first committed (0e23763) — the
    claim that bash and write were withheld was aspirational and never true,
    so nothing should be relied on as if it were. A delegated editor can
    therefore still hand-roll a rewrite or replace a topic file wholesale.
    Withholding those two members is a one-line change to the toolset
    `configs` block if that restriction is actually wanted.
    """
    cfg = yaml.safe_load(EDITOR_CONFIG_PATH.read_text(encoding="utf-8"))
    by_name = dict(zip(SKILLS, skill_refs))  # upload_skills builds refs in SKILLS order
    editor_refs = [by_name[name] for name in EDITOR_SKILLS]
    return _upsert_agent(client, state, cfg, editor_refs,
                         "editor_agent_id", "editor_agent_version", "editor agent")


def ensure_agent(client: anthropic.Anthropic, state: dict, skill_refs: list[dict],
                 editor_agent_id: str) -> str:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    multiagent = cfg.get("multiagent")
    if multiagent:
        # Resolve the symbolic roster entry "editor" to the agent id built from
        # editor_agent_config.yaml. Every other entry passes through untouched,
        # so the YAML stays the single readable source of truth for the roster.
        resolved = []
        for entry in multiagent.get("agents", []):
            resolved.append(editor_agent_id if entry == "editor" else entry)
        if editor_agent_id not in resolved:
            print("ERROR: managed_agent_config.yaml multiagent.agents must include "
                  "'editor'; the coordinator would spawn full-powered copies of "
                  "itself instead of the restricted editor.", file=sys.stderr)
            sys.exit(2)
        multiagent = dict(multiagent, agents=resolved)
    agent_id = _upsert_agent(client, state, cfg, skill_refs,
                             "agent_id", "agent_version", "agent",
                             multiagent=multiagent)
    # Verify the roster actually landed. A coordinator whose roster silently
    # degrades to self-only still runs — it just spawns full-powered copies of
    # itself instead of the narrow editor, which looks like working delegation
    # in the logs while bypassing every capability restriction the editor has.
    if multiagent:
        entries = client.beta.agents.retrieve(agent_id).multiagent.agents
        # Advisor entries carry {type, model} and no id — skip them here.
        rostered = {getattr(a, "id", None) for a in entries} - {None}
        advisors = [getattr(a, "model", None) for a in entries
                    if getattr(a, "type", None) == "advisor"]
        if editor_agent_id not in rostered:
            print(f"ERROR: editor {editor_agent_id} missing from coordinator roster "
                   f"after update (roster={sorted(rostered)}).", file=sys.stderr)
            sys.exit(2)
        # Names, as they actually stand on the live agents. preflight() checks
        # the configs, but a roster member can be renamed out from under this
        # checkout by anyone else's deploy; this is the check that would have
        # caught the clobber at deploy time instead of at session creation.
        names = [client.beta.agents.retrieve(aid).name for aid in sorted(rostered)]
        if len(set(names)) != len(names):
            print(f"ERROR: two rostered agents share a callable name ({names}). "
                  f"Sessions will fail with 'duplicate callable agent name'. One of "
                  f"these agents was deployed with the wrong config.", file=sys.stderr)
            sys.exit(2)
        print(f"  roster verified: editor + self ({len(rostered)} agents"
              + (f", advisor={advisors[0]}" if advisors else ", no advisor") + ")")
    return agent_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reupload-sa", action="store_true",
                        help="Force re-uploading the service account JSON (after key rotation).")
    parser.add_argument("--force-reupload-skills", action="store_true",
                        help="Push a new skill version even if the content hash hasn't changed.")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set (env or .env).", file=sys.stderr)
        sys.exit(2)

    client = anthropic.Anthropic()
    state = load_state()
    preflight(state)

    print("Uploading skills...")
    skill_refs = upload_skills(client, state, force_reupload=args.force_reupload_skills)
    save_state(state)

    print("Ensuring service account is uploaded...")
    ensure_sa_uploaded(client, state, force=args.reupload_sa)
    save_state(state)

    print("Ensuring environment...")
    ensure_environment(client, state)
    save_state(state)

    print("Ensuring shared memory store...")
    ensure_shared_memory_store(client, state)
    save_state(state)

    print("Creating/updating editor agent...")
    editor_agent_id = ensure_editor_agent(client, state, skill_refs)
    save_state(state)

    print("Creating/updating coordinator agent...")
    ensure_agent(client, state, skill_refs, editor_agent_id)
    save_state(state)

    print("\nDone. Resource IDs:")
    print(f"  agent_id        = {state['agent_id']}")
    print(f"  editor_agent_id = {state.get('editor_agent_id')}")
    print(f"  environment_id  = {state['environment_id']}")
    print(f"  sa_file_id      = {state.get('sa_file_id') or '(none — set GDRIVE_SA_B64 and re-run)'}")
    print(f"  shared_memory_store_id = {state.get('shared_memory_store_id')}")
    print("\nStart a chat with:")
    print("  python agents/course_content_editor/chat.py [--sheet-url <URL>]")


if __name__ == "__main__":
    main()
