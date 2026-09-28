"""Offline checks. No network, no model, no sandbox.

    python selftest.py

The tool-based build's selftest exercised six in-process tools. Those are gone,
and with them most of what there was to unit-test here: the sheet logic now
lives in `skills/working-with-google-sheets/scripts/`, which is the *same code*
the Deep Agents build runs and which that build's selftest already covers in
depth. Re-testing it here would be a second, silently-diverging copy of those
assertions — so instead this asserts the files are byte-identical and lets one
suite own the behaviour.

What remains is specific to this build:

- the agent definition really is as small as it claims, and nothing that a
  sandbox silently discards is being declared as though it protected something
- the sandbox is wired with the auth proxy and carries no credential
- the skill ships runnable scripts, and the shared ones have not drifted
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

SCRIPTS = ROOT / "skills" / "working-with-google-sheets" / "scripts"
DEEPAGENTS = ROOT.parent / "course_content_editor_deepagents"
SHARED_WITH_DEEPAGENTS = {
    "_common.py": DEEPAGENTS / "skills" / "sheets" / "working-with-google-sheets" / "scripts",
    "check_auth.py": DEEPAGENTS / "skills" / "sheets" / "working-with-google-sheets" / "scripts",
    "commit_context.py": DEEPAGENTS / "skills" / "sheets" / "working-with-google-sheets" / "scripts",
    "commit_workspace.py": DEEPAGENTS / "skills" / "sheets" / "working-with-google-sheets" / "scripts",
    "list_tabs.py": DEEPAGENTS / "skills" / "sheets" / "working-with-google-sheets" / "scripts",
    "prepare_context_workspace.py": DEEPAGENTS / "skills" / "sheets" / "working-with-google-sheets" / "scripts",
    "prepare_workspace.py": DEEPAGENTS / "skills" / "sheets" / "working-with-google-sheets" / "scripts",
    "_presentation.py": DEEPAGENTS,
}

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(label)


def section(name: str) -> None:
    print(name)


# ---------------------------------------------------------------------------


def test_definition() -> None:
    section("agent definition")
    import agent as agent_module

    config = agent_module.agent.config
    check("name is set", config.get("name") == "course-content-editor-mda")
    check("model is overridable via CCE_MODEL",
          "CCE_MODEL" in (ROOT / "agent.py").read_text(encoding="utf-8"))
    check("default model is gemini 3.8 flash",
          agent_module.MODEL.endswith("gemini-3.8-flash") or "CCE_MODEL" in __import__("os").environ,
          agent_module.MODEL)

    # Each of these is discarded or pointless under a sandbox. Declaring one
    # would read as protection while doing nothing, which is worse than its
    # absence — hence asserting they stay absent rather than merely unused.
    for field in ("tools", "permissions", "interrupt_on"):
        check(f"no {field} declared", field not in config, str(config.get(field)))

    # The one exception, and it earns its place: without it `present.py` output
    # goes into the agent's context instead of onto the user's screen, which is
    # the failure the whole present/ack split exists to prevent.
    names = [getattr(m, "name", type(m).__name__) for m in (config.get("middleware") or [])]
    check("present_to_client middleware is wired",
          any("present" in str(n) for n in names), str(names))

    subagents = config.get("subagents") or []
    check("one editor subagent", len(subagents) == 1 and subagents[0]["name"] == "editor")
    editor = subagents[0]
    check("editor carries its own skills index", editor.get("skills") == ["/skills/"],
          repr(editor.get("skills")))
    # A picked model must reach both, or the editors — who do most of the
    # writing — stay on the deploy-time model.
    editor_names = [getattr(m, "name", type(m).__name__) for m in (editor.get("middleware") or [])]
    check("select_model on coordinator and editor",
          any("select_model" in str(n) for n in names)
          and any("select_model" in str(n) for n in editor_names),
          f"{names} / {editor_names}")
    check("editor is isolated, so its skills declaration is legal",
          editor.get("mode", "isolated") == "isolated")
    check("editor prompt keeps the sheet boundary",
          "never write to the sheet" in editor["system_prompt"])


def test_sandbox() -> None:
    section("sandbox")
    import sandbox as sandbox_module

    options = sandbox_module.sandbox.options
    check("sandbox declared", bool(options), str(sorted(options)))
    check("proxy_config survives normalization", "proxy_config" in options)
    check("idle ttl set so a paused conversation does not hold a VM",
          options.get("idle_ttl_seconds") == 600, str(options.get("idle_ttl_seconds")))

    from managed_deepagents.runtime import _sandbox_create_kwargs

    check("proxy_config is forwarded to create_sandbox",
          "proxy_config" in _sandbox_create_kwargs(sandbox_module.sandbox))

    setup = (ROOT / "sandbox" / "setup.sh").read_text(encoding="utf-8")
    check("setup.sh installs the script dependencies",
          all(p in setup for p in ("gspread", "google-auth", "pandas")))

    # The scripts are baked into the image because /skills/ is a Context Hub
    # mount the shell cannot see. This is the check that catches editing a
    # script and forgetting to regenerate — the failure it prevents is a
    # deployed run using stale code, which looks like a model mistake.
    import subprocess  # noqa: PLC0415

    fresh = subprocess.run([sys.executable, str(ROOT / "build_setup.py"), "--check"],
                           capture_output=True, text=True, cwd=ROOT)
    check("sandbox/setup.sh is regenerated from the current scripts",
          fresh.returncode == 0, fresh.stderr.strip()[:120])
    check("setup.sh bakes the scripts to a fixed absolute path",
          "/opt/cce/scripts" in setup)
    check("setup.sh provides `python`, which a model reaches for first",
          "ln -s" in setup and "python3" in setup)
    # Anything written to disk during bake lands in every thread's image.
    for leak in ("GDRIVE_SA_B64", "GDRIVE_SA_JSON", "SLIDE_CHUNKS_SA_KEY", "service_account.json"):
        check(f"setup.sh does not bake {leak}", leak not in setup.replace("# ", "")
              or leak in setup.split("set -e")[0])


def test_no_credential_in_deployment() -> None:
    section("credential handling")
    text = (ROOT / "sandbox" / "__init__.py").read_text(encoding="utf-8")
    # The module docstring names the env vars in order to explain why they are
    # not used, so scan the code below it rather than the prose above it.
    tree = ast.parse(text)
    code = text if ast.get_docstring(tree) is None else text.split('"""', 2)[-1]
    check("sandbox references the credential by secret name only",
          "workspace_secret" in code
          and not any(v in code for v in ("GDRIVE_SA_B64", "GDRIVE_SA_JSON")))
    check("gcp_auth rule present", "gcp_auth" in text)
    # open_by_key avoids the Drive search that would force a wider grant.
    check("scopes stay narrow (spreadsheets only)",
          "auth/spreadsheets" in text and "auth/drive" not in text)

    common = (SCRIPTS / "_common.py").read_text(encoding="utf-8")
    check("scripts fall back to the proxy when no key is present",
          "AnonymousCredentials" in common and "required=False" in common)
    check("scripts still accept a real key where one exists",
          "gspread.service_account(filename=" in common)


def test_scripts() -> None:
    section("skill scripts")
    expected = {
        "_common.py", "_presentation.py", "check_auth.py", "commit_context.py",
        "commit_workspace.py", "list_tabs.py", "prepare_context_workspace.py",
        "prepare_workspace.py", "present.py",
    }
    present = {p.name for p in SCRIPTS.glob("*.py")}
    check("all scripts present", expected <= present, str(sorted(expected - present)))

    for name in sorted(present):
        path = SCRIPTS / name
        try:
            ast.parse(path.read_text(encoding="utf-8"))
            ok, detail = True, ""
        except SyntaxError as e:  # noqa: PERF203
            ok, detail = False, str(e)
        check(f"{name} parses", ok, detail)

    # The Deep Agents build is a sibling working copy, not a dependency: it is
    # untracked on this branch and may simply not be checked out. Absence is not
    # a failure — there is nothing to disagree with. A file that is missing while
    # the build around it is present still is, because that is a real drift.
    if not DEEPAGENTS.exists():
        check("script parity with the Deep Agents copy", True,
              "skipped — no sibling build checked out")
        return

    for name, other_dir in sorted(SHARED_WITH_DEEPAGENTS.items()):
        other = other_dir / name
        mine = SCRIPTS / name
        if not other.exists():
            check(f"{name} matches the Deep Agents copy", False, f"missing at {other}")
            continue
        same = other.read_bytes() == mine.read_bytes()
        check(f"{name} matches the Deep Agents copy", same, "" if same else "differs")


def test_project_shape() -> None:
    section("project shape")
    check("tools/ removed", not (ROOT / "tools").exists())
    # middleware/ came back for three things, each argued for in agent.py's module
    # docstring: present_bridge.py routes present.py output to the user's screen,
    # editor_identity.py names the signed-in person for the run, and
    # model_select.py runs the session on the model the user picked — the one
    # route MDA leaves for that, since `configurable.model` and `context.model`
    # are both ignored on a deployed run. The assertion is that it holds those
    # and nothing else — middleware is where this build grows accidental
    # machinery, so a fourth file should have to justify itself here.
    middleware_files = sorted(p.name for p in (ROOT / "middleware").glob("*.py"))
    check("middleware/ holds only the three declared middleware",
          middleware_files == ["__init__.py", "editor_identity.py", "model_select.py",
                               "present_bridge.py"],
          str(middleware_files))
    for needed in ("instructions.md", "memory.py", "identity.py", "sandbox/__init__.py"):
        check(f"{needed} present", (ROOT / needed).exists())

    skills = sorted(p.name for p in (ROOT / "skills").iterdir() if p.is_dir())
    check("all three skills present",
          skills == ["editing-research-notes", "editing-slide-chunks",
                     "working-with-google-sheets"], str(skills))
    for name in skills:
        text = (ROOT / "skills" / name / "SKILL.md").read_text(encoding="utf-8")
        check(f"{name} has frontmatter", text.startswith("---") and "description:" in text)

    instructions = (ROOT / "instructions.md").read_text(encoding="utf-8")
    check("instructions no longer promise tools that are gone",
          "present_files" not in instructions and "measure_file" not in instructions)
    check("instructions say a shell exists", "shell" in instructions)
    check("instructions delegate the sheet workflow to the skill",
          "working-with-google-sheets" in instructions)

    sheets_skill = (ROOT / "skills" / "working-with-google-sheets" / "SKILL.md").read_text(
        encoding="utf-8")
    # The deployed agent ran the right script at the wrong path and then
    # searched the whole filesystem for it. The skill must name the runnable
    # path, not the readable one.
    check("sheets skill points at the baked script path",
          "/opt/cce/scripts" in sheets_skill and
          "/skills/working-with-google-sheets/scripts" not in sheets_skill)
    check("sheets skill explains why /skills/ is not runnable",
          "cannot" in sheets_skill and "read_file" in sheets_skill)
    for script in ("prepare_workspace.py", "prepare_context_workspace.py",
                   "commit_workspace.py", "commit_context.py", "list_tabs.py", "present.py"):
        check(f"sheets skill documents {script}", script in sheets_skill)
    check("sheets skill explains why there is no key",
          "no key here" in sheets_skill.lower() or "there is none" in sheets_skill)


def test_skills_only_name_real_capabilities() -> None:
    """No skill may instruct the agent to call something that does not exist.

    Both editing skills spent a refactor telling the editor subagent to call
    `open_course` and `measure_file` after those tools were deleted. Nothing
    failed loudly — the skills were still valid markdown, every other check
    stayed green, and the only symptom was an agent following instructions that
    could not work. A skill naming a capability is a factual claim, so it is
    checkable.
    """
    section("skills name only things that exist")
    removed = ["open_course", "list_course_tabs", "present_files", "measure_file",
               "commit_research_notes", "commit_slide_chunks"]
    for skill_dir in sorted(p for p in (ROOT / "skills").iterdir() if p.is_dir()):
        text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        named = [tool for tool in removed if tool in text]
        check(f"{skill_dir.name} names no deleted tool", not named, str(named))

    instructions = (ROOT / "instructions.md").read_text(encoding="utf-8")
    named = [tool for tool in removed if tool in instructions]
    check("instructions.md names no deleted tool", not named, str(named))

    # Scripts the skill tells the agent to run must actually be there.
    sheets = (ROOT / "skills" / "working-with-google-sheets" / "SKILL.md").read_text(
        encoding="utf-8")
    for referenced in sorted(set(re.findall(r"/opt/cce/scripts/([\w.]+\.py)", sheets))):
        check(f"{referenced} exists", (SCRIPTS / referenced).exists())

    # ...and so must any reference file a skill points at. The first version of
    # this check only looked at scripts and tool names, and so sailed past two
    # live pointers to `references/parsing-rules.md`, a file the port never
    # brought across — dangling in the deployed bundle, not just in the source.
    for skill_dir in sorted(p for p in (ROOT / "skills").iterdir() if p.is_dir()):
        text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        for referenced in sorted(set(re.findall(r"`(references/[\w./-]+\.md)`", text))):
            check(f"{skill_dir.name} -> {referenced} exists",
                  (skill_dir / referenced).exists())


def _script_arguments(script: Path) -> dict[str, bool]:
    """{flag: required} parsed from a script's argparse calls."""
    tree = ast.parse(script.read_text(encoding="utf-8"))
    out: dict[str, bool] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"):
            continue
        flags = [a.value for a in node.args if isinstance(a, ast.Constant)]
        if not flags or not str(flags[0]).startswith("--"):
            continue
        kw = {k.arg: k.value for k in node.keywords}
        out[str(flags[0])] = (
            isinstance(kw.get("required"), ast.Constant) and kw["required"].value is True
        )
    return out


def test_cheatsheet_matches_the_scripts() -> None:
    """The documented commands must match the scripts' real interfaces.

    The first cheatsheet showed optional flags as required and omitted two
    entirely. A deployed run did the reasonable thing and ran `--help` to find
    out — which is the behaviour a prohibition would have blocked, leaving it
    with instructions it could not trust and no way to check them. Documentation
    that disagrees with code is worse than none, and this comparison is
    mechanical, so it should not depend on anyone remembering.
    """
    section("cheatsheet matches the scripts")
    text = (ROOT / "skills" / "working-with-google-sheets" / "SKILL.md").read_text(
        encoding="utf-8")
    start = text.index("## The commands, in full")
    block = text[start:text.index("## Showing your work")]

    for script_path in sorted(SCRIPTS.glob("*.py")):
        if script_path.name.startswith("_"):
            continue
        # The command line for this script: from its mention to the next blank line.
        marker = f"/opt/cce/scripts/{script_path.name}"
        if marker not in block:
            check(f"{script_path.name} appears in the cheatsheet", False)
            continue
        segment = block[block.index(marker):]
        segment = segment[:segment.index("\n\n")] if "\n\n" in segment else segment
        documented = set(re.findall(r"--[\w-]+", segment))
        real = _script_arguments(script_path)

        invented = sorted(documented - set(real))
        check(f"{script_path.name}: cheatsheet invents no flags", not invented, str(invented))

        missing_required = sorted(f for f, req in real.items() if req and f not in documented)
        check(f"{script_path.name}: every required flag is documented",
              not missing_required, str(missing_required))


def main() -> int:
    test_definition()
    test_sandbox()
    test_no_credential_in_deployment()
    test_scripts()
    test_skills_only_name_real_capabilities()
    test_cheatsheet_matches_the_scripts()
    test_project_shape()
    print()
    if failures:
        print(f"{len(failures)} FAILED: " + ", ".join(failures))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
