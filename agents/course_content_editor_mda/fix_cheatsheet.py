"""Rewrite the skill's command cheatsheet so it matches the scripts. Run once.

The first cheatsheet presented optional arguments as required, omitted two flags
entirely, and showed no defaults. A deployed run responded exactly as anyone
would to instructions it could not fully trust: it ran `--help`. The prohibition
I had written against doing that would have blocked the one move that recovered
from my error.

So this generates the block from the scripts' own `argparse` definitions rather
than from memory, and `selftest.py` re-checks the correspondence offline.
"""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
SCRIPTS = PROJECT / "skills" / "working-with-google-sheets" / "scripts"
SKILL = PROJECT / "skills" / "working-with-google-sheets" / "SKILL.md"

START = "## The commands, in full"
END = "## Showing your work"


def arguments(script: str) -> dict[str, tuple[bool, object]]:
    """{flag: (required, default)} for one script, read from its argparse calls."""
    tree = ast.parse((SCRIPTS / script).read_text(encoding="utf-8"))
    constants = {
        target.id: node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    out: dict[str, tuple[bool, object]] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"):
            continue
        flags = [a.value for a in node.args if isinstance(a, ast.Constant)]
        if not flags or not str(flags[0]).startswith("--"):
            continue
        kw = {k.arg: k.value for k in node.keywords}
        required = isinstance(kw.get("required"), ast.Constant) and kw["required"].value is True
        default: object = None
        if "default" in kw:
            node_default = kw["default"]
            if isinstance(node_default, ast.Constant):
                default = node_default.value
            elif isinstance(node_default, ast.Name):
                default = constants.get(node_default.id)
        out[str(flags[0])] = (required, default)
    return out


def optional(flag: str, default: object) -> str:
    return f'[{flag} "{default}"]' if isinstance(default, str) and default else f"[{flag}]"


CHEATSHEET = f'''{START}

Copy these. Square brackets mark optional arguments, with their defaults shown —
so anything unbracketed is required and anything bracketed can be left out.

```bash
# 1. Tab names vary per course sheet. Check rather than assume.
python /opt/cce/scripts/list_tabs.py --sheet-url "<SHEET_URL>"

# 2a. Research notes only. No slide-chunks tab needed.
python /opt/cce/scripts/prepare_context_workspace.py \\
    --sheet-url "<SHEET_URL>" --workspace workspace \\
    {optional("--outline-tab", "Final Outline")} \\
    {optional("--outline-target-tab", "Final Outline (Revised)")}

# 2b. Slide chunks, and the research notes alongside them.
python /opt/cce/scripts/prepare_workspace.py \\
    --sheet-url "<SHEET_URL>" --workspace workspace \\
    {optional("--source-tab", "Slide Chunks")} {optional("--outline-tab", "Final Outline")} \\
    {optional("--target-tab", "Slide Chunks (Revised)")} \\
    {optional("--outline-target-tab", "Final Outline (Revised)")}

# 3. Show the user what you changed. Run this after editing, unprompted.
python /opt/cce/scripts/present.py --workspace workspace \\
    context/topic_01_<slug>.md [--blocks 3,7] [--note "what to look at"]

# 4. Counts only — this output is for you, before and after a
#    tighten / trim / expand request.
python /opt/cce/scripts/present.py --workspace workspace --measure \\
    topics/topic_02_<slug>.md

# 5a. Write research notes to a new tab.
python /opt/cce/scripts/commit_context.py --workspace workspace \\
    [--outline-target-tab "<NEW_TAB_NAME>"]

# 5b. Write slide chunks to a new tab.
python /opt/cce/scripts/commit_workspace.py --workspace workspace \\
    [--target-tab "<NEW_TAB_NAME>"]

# 6. If a sheet call fails, confirm access before retrying anything.
python /opt/cce/scripts/check_auth.py --sheet-url "<SHEET_URL>"
```

**Name the target tab when you commit.** Both commit scripts fall back to the
tab recorded when the workspace was prepared — `Slide Chunks (Revised)` or
`Final Outline (Revised)` — so leaving the flag off does not mean "no tab", it
means "that one", and a second commit overwrites the first. Pass the name the
user asked for.

'''


def main() -> int:
    text = SKILL.read_text(encoding="utf-8")
    start, end = text.index(START), text.index(END)
    SKILL.write_text(text[:start] + CHEATSHEET + text[end:], encoding="utf-8")

    # Report the correspondence this is meant to guarantee.
    for script in sorted(p.name for p in SCRIPTS.glob("*.py") if not p.name.startswith("_")):
        args = arguments(script)
        required = [f for f, (req, _) in args.items() if req]
        print(f"  {script:32} required: {required}")
    print(f"\nrewrote the cheatsheet in {SKILL.relative_to(PROJECT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
