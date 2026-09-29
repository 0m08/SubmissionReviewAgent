"""List the tabs in a Google Sheet, and say which ones hold editable content.

Usage:
    python list_tabs.py --sheet-url "<URL>"
    python list_tabs.py --sheet-url "<URL>" --names-only

Used before prepare_workspace.py, to choose `--source-tab` / `--outline-tab`.
The default names ("Slide Chunks", "Final Outline") vary by course.

## Why this classifies instead of just listing

A course sheet routinely carries several tabs of slide chunks: the generator's
`Slide Chunks`, per-topic tabs someone split out by hand, `(Revised)` tabs left
by an earlier commit, and backups. They are not distinguishable by name — one
observed sheet had `Slide Chunks`, `Topic 1: Pipe Preparation`, `Topic 2:
Threaded Joints` and `Topic 2: Threaded Joints (Revised)` side by side.

A bare list makes picking the default look like the only option, and it is not:
in that session the user meant `Topic 1: Pipe Preparation`, the agent took
`Slide Chunks` without comment, and the whole turn was spent on the wrong
content. So the ambiguity is reported where it is found, rather than left to be
noticed.

Detection is exact, not a guess: a tab holds slide chunks if its header row has
a Topic column and a Slide Chunk column, which is the same test
prepare_workspace.py applies when it reads the tab. Cost is one extra API call
for the whole sheet (~0.4s for 21 tabs), because the header rows are fetched in
a single batch.
"""

from __future__ import annotations

import argparse
import sys

from _common import open_sheet

# The same aliases prepare_workspace.py and _common.py match on. A tab is
# classified by what its header actually says, so a tab named anything at all
# is found, and a tab named "Slide Chunks" holding something else is not.
_TOPIC = ("topic", "Topic")
_SLIDE_CHUNK = ("Slide Chunk", "slide_chunk")
_LEARNING_OBJECTIVES = ("Learning Objectives", "learning_objectives")

DEFAULT_SOURCE_TAB = "Slide Chunks"
DEFAULT_OUTLINE_TAB = "Final Outline"


def _has(header: list[str], aliases: tuple[str, ...]) -> bool:
    cells = {c.strip() for c in header}
    return any(a in cells for a in aliases)


def is_artifact(title: str) -> bool:
    """A slide-chunk tab nobody would mean as a source.

    Two kinds: pipeline backups, and `(Revised)` tabs written by
    commit_workspace.py. Both carry the slide-chunk header, so the header test
    alone flags them, and both appear in almost every course sheet — counting
    them would make the ambiguity warning fire on every sheet, including ones
    with exactly one real source tab. A warning that always fires is one nobody
    reads. They are still listed, and still available to `--source-tab` if the
    user names one.
    """
    t = title.strip().lower()
    return t.startswith("backup") or t.endswith("(revised)")


def classify(header: list[str]) -> str | None:
    """'slide_chunks', 'outline', or None for a tab holding neither."""
    if not header or not _has(header, _TOPIC):
        return None
    if _has(header, _SLIDE_CHUNK):
        return "slide_chunks"
    if _has(header, _LEARNING_OBJECTIVES):
        return "outline"
    return None


def _headers(sheet, titles: list[str]) -> dict[str, list[str]]:
    """Header row of every tab, in one API call.

    Falls back to empty headers rather than failing: a listing that loses its
    annotations is still a listing, and this script is often the first thing
    run against a sheet whose shape nobody knows yet.
    """
    try:
        ranges = [f"'{t}'!A1:Z1" for t in titles]
        got = sheet.values_batch_get(ranges).get("valueRanges", [])
        return {t: ((vr.get("values") or [[]])[0]) for t, vr in zip(titles, got)}
    except Exception as exc:  # noqa: BLE001 — annotation is a bonus, not the job
        print(f"NOTE: could not read header rows ({exc.__class__.__name__}); "
              f"listing tab names only.", file=sys.stderr)
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet-url", required=True)
    parser.add_argument("--names-only", action="store_true",
                        help="Tab names alone, no classification, no extra API call.")
    args = parser.parse_args()

    sheet = open_sheet(args.sheet_url)
    titles = [ws.title for ws in sheet.worksheets()]

    if args.names_only:
        for t in titles:
            print(t)
        return 0

    headers = _headers(sheet, titles)
    kinds = {t: classify(headers.get(t, [])) for t in titles}

    for t in titles:
        marks = []
        if kinds[t] == "slide_chunks":
            marks.append("slide chunks (backup/output)" if is_artifact(t) else "slide chunks")
        elif kinds[t] == "outline":
            marks.append("outline")
        if t == DEFAULT_SOURCE_TAB:
            marks.append("default --source-tab")
        if t == DEFAULT_OUTLINE_TAB:
            marks.append("default --outline-tab")
        print(f"{t}" + (f"    <- {', '.join(marks)}" if marks else ""))

    chunk_tabs = [t for t in titles if kinds[t] == "slide_chunks" and not is_artifact(t)]
    if len(chunk_tabs) > 1:
        print("")
        print(f"NOTE: {len(chunk_tabs)} tabs hold slide chunks, so the source tab is "
              f"ambiguous:", file=sys.stderr)
        for t in chunk_tabs:
            print(f"  - {t}", file=sys.stderr)
        print("Ask the user which one they mean before you prepare a workspace. "
              "Taking the default silently is how the wrong tab gets edited.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
