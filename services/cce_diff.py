"""The Course Content Editor's review layer, shared by both builds.

Two pages drive this product — `course_content_editor.py` (Anthropic Managed
Agents) and `course_content_editor_mda.py` (a Managed Deep Agent on LangGraph)
— and they show the same thing: a course file, before and after the agent
edited it. Only where the bytes come from differs.

So the view lives here, once. The alternative was tried: the MDA page shipped
with a reduced re-implementation, and it drifted immediately — a single
interleaved del/ins stream where this one renders two columns, no plain view,
and no way to select a passage and comment on it. Two implementations of "what
changed" is exactly the kind of second opinion that quietly disagrees with the
file on disk.

Everything here is a pure function of (before, after) except
`render_comment_surface`, which owns the select-and-comment iframe and takes
the caller's session-state key for the message it composes.
"""

from __future__ import annotations

import difflib
import html
import re
import sys
from pathlib import Path

import streamlit as st

# Reuse the exact block parsers the deployed skill scripts use, rather than
# re-implementing the ###LO ID: / ###Block ID: contract a second time here.
# Same sys.path convention agents/slide_chunk_editor/_paths.py established for
# cross-referencing a skill's scripts/ directory from outside the sandbox.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_CCE_DIR = _REPO_ROOT / "agents" / "course_content_editor"
_SCRIPTS_DIR = _CCE_DIR / "skills" / "working-with-google-sheets" / "scripts"
for _path in (_SCRIPTS_DIR, _CCE_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import commit_context as cc_context  # noqa: E402  (parse_lo_block, LO_BLOCK_START)
import commit_workspace as cc_topics  # noqa: E402  (parse_block, BLOCK_START)
import _presentation as pres  # noqa: E402
import _comment_component as _cmt  # noqa: E402  (see its docstring: exec'd pages can't declare)

#: The page stylesheet. An editorial "redline" aesthetic: this is a
#: proofreading tool, not a chatbot, so it leans on real diff coloring and a
#: serif display face instead of a generic chat-app look.
PAGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Public+Sans:wght@400;500;600;700&display=swap');

:root {
    --cce-paper: #faf6ee;
    --cce-ink: #241f18;
    --cce-ink-soft: #6b6153;
    --cce-rule: #ddd3bf;
    --cce-del-bg: #fbe9e7;
    --cce-del-text: #9a3324;
    --cce-ins-bg: #e9f2e3;
    --cce-ins-text: #2f5233;
    --cce-accent: #a8492f;
}

.cce-title {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 1.9rem;
    font-weight: 600;
    color: var(--cce-ink);
    letter-spacing: -0.01em;
    margin-bottom: 0;
}
.cce-subtitle {
    font-family: 'Public Sans', sans-serif;
    color: var(--cce-ink-soft);
    font-size: 0.92rem;
    margin-top: 0.1rem;
    margin-bottom: 0.6rem;
}
/* Card background/border now comes from st.container(border=True)
   itself, not a hand-rolled div — see the comment where it's used.
   Only the label/meta typography stays custom. */
.cce-card-label {
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 600;
    font-size: 1.02rem;
    color: var(--cce-ink);
    margin-bottom: 0.15rem;
}
.cce-card-meta {
    font-size: 0.8rem;
    color: var(--cce-ink-soft);
    margin-bottom: 0.55rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
/* Session picker — shown when the page is opened without a thread in the URL.
   Deliberately reuses .cce-card-label / .cce-card-meta for each row, so a saved
   session reads as the same kind of object as a slide card rather than as a
   different screen. Only the section header and the expiry flag are new. */
.cce-session-head {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--cce-ink);
    margin-bottom: 0.1rem;
}
.cce-session-note {
    font-family: 'Public Sans', sans-serif;
    font-size: 0.85rem;
    color: var(--cce-ink-soft);
    margin-bottom: 0.2rem;
}
/* The one warm-red thing on the screen. A workspace about to be reclaimed is
   the only fact here that expires, so it is the only one that gets the accent. */
.cce-session-expiry {
    color: var(--cce-accent);
    margin-left: 0.35rem;
}
/* Learning Objective — a full sentence, so no uppercase/letter-spacing
   (that reads fine for "RESEARCH NOTES · TOPIC" but not for prose). */
.cce-card-lo {
    font-size: 0.85rem;
    color: var(--cce-ink-soft);
    font-style: italic;
    margin-bottom: 0.6rem;
}
.cce-diff-text {
    font-size: 0.94rem;
    line-height: 1.55;
    color: var(--cce-ink);
}
/* Unchanged paragraphs — the head/tail context around a change. */
.cce-diff-text p.cce-diff-context {
    color: var(--cce-ink-soft);
    margin: 0 0 0.6rem 0;
}
/* Paragraphs containing the actual edit — full ink color so the
   del/ins highlighting inside them reads as the focal point. */
.cce-diff-text p.cce-diff-changed {
    color: var(--cce-ink);
    margin: 0 0 0.6rem 0;
}
/* Inline slide art, plain view only — the diff view deliberately keeps
   the raw ![alt](url) markdown visible, since a changed image URL is an
   edit you need to be able to see. */
/* Capped on BOTH axes. Course art is authored at full slide resolution,
   so max-width alone still let a tall diagram run past a screen height
   and push the surrounding prose out of view — which is the opposite of
   what a review card is for. Thumbnail here, full size one click away
   (each image is wrapped in a link to its own source). */
.cce-plain-img {
    display: block;
    max-width: min(100%, 380px);
    max-height: 240px;
    width: auto;
    height: auto;
    object-fit: contain;
    margin: 0.5rem 0;
    border: 1px solid var(--cce-rule);
    border-radius: 4px;
    /* Many of these diagrams are transparent PNGs drawn in black ink —
       on the paper-toned card background they'd be hard to read. */
    background: #fff;
}
.cce-plain-img-link {
    display: inline-block;
    line-height: 0;
    cursor: zoom-in;
}
.cce-plain-img-missing {
    font-size: 0.8rem;
    color: var(--cce-ink-soft);
    font-style: italic;
}
/* Tool activity woven into the turn. Almost every one of these is a
   shell command or a file path, so it reads as machine text, set apart
   from the agent's prose rather than competing with it. */
.cce-cmd {
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    font-size: 0.78rem;
    line-height: 1.5;
    color: var(--cce-ink-soft);
    background: rgba(0, 0, 0, 0.025);
    border-left: 2px solid var(--cce-rule);
    padding: 0.18rem 0.55rem;
    margin: 0.12rem 0;
    white-space: pre-wrap;
    word-break: break-word;
    border-radius: 0 3px 3px 0;
}
/* The live "what's happening now" line. A real spinner, because a
   static caption during a 60-second delegation is indistinguishable
   from a hung page. */
.cce-working {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.8rem;
    color: var(--cce-ink-soft);
    padding: 0.25rem 0.1rem;
}
.cce-spin {
    width: 0.72rem;
    height: 0.72rem;
    flex: none;
    border: 2px solid var(--cce-rule);
    border-top-color: var(--cce-accent);
    border-radius: 50%;
    animation: cce-spin 0.8s linear infinite;
}
@keyframes cce-spin { to { transform: rotate(360deg); } }
.cce-thread-meta {
    font-size: 0.78rem;
    color: var(--cce-ink-soft);
    margin-bottom: 0.4rem;
}
.cce-diff-text p:last-child {
    margin-bottom: 0;
}
.cce-diff-text del {
    background: var(--cce-del-bg);
    color: var(--cce-del-text);
    text-decoration: line-through;
    text-decoration-thickness: 1.5px;
    padding: 0 1px;
    border-radius: 2px;
}
.cce-diff-text ins {
    background: var(--cce-ins-bg);
    color: var(--cce-ins-text);
    text-decoration: none;
    padding: 0 1px;
    border-radius: 2px;
}
/* Side-by-side paragraph diff table — same structure as
   compare_text_versions in services/helper_functions.py (the diff
   viewer already used for the slide-chunks checklist agent), adapted
   to paragraph granularity. Two independent columns instead of one
   interleaved stream: each side reads as continuous prose even when a
   paragraph was rewritten heavily enough that a single-stream word
   diff would otherwise turn into unreadable word salad. */
.cce-diff-table {
    width: 100%;
    border-collapse: collapse;
    table-layout: fixed;
    margin-top: 0.3rem;
}
.cce-diff-table td {
    padding: 0.5rem 0.7rem;
    vertical-align: top;
    font-size: 0.94rem;
    line-height: 1.5;
    border: 1px solid var(--cce-rule);
    white-space: pre-wrap;
    word-wrap: break-word;
    width: 50%;
}
/* Whole-topic redline: one row per line, so the per-cell grid of the
   paragraph table would box every single line. Drop the horizontal
   rules and tighten the padding — the topic then reads as one
   continuous document with a single rule down the middle, which is
   what makes a 40-line topic legible. */
.cce-diff-doc td {
    border: none;
    border-right: 1px solid var(--cce-rule);
    padding: 0.1rem 0.7rem;
}
.cce-diff-doc td:last-child {
    border-right: none;
}
/* Topic / Subtopic / [Slide Type] Title lines — the spine of the
   document view. */
.cce-diff-doc .cce-diff-head {
    font-weight: 650;
    padding-top: 0.35rem;
}
/* Unchanged paragraph — full-width context row, muted, no color coding. */
.cce-diff-ctx {
    color: var(--cce-ink-soft);
    background: transparent;
}
.cce-diff-old {
    background: var(--cce-del-bg);
    color: var(--cce-ink);
}
.cce-diff-new {
    background: var(--cce-ins-bg);
    color: var(--cce-ink);
}
/* Word-level highlight within a column — solid fill, matches the
   darker-highlight-on-light-background hierarchy the original diff
   viewer used, recolored to this page's palette. */
.cce-word-del {
    background: var(--cce-del-text);
    color: #fff;
    text-decoration: line-through;
    border-radius: 3px;
    padding: 0 3px;
}
.cce-word-ins {
    background: var(--cce-ins-text);
    color: #fff;
    border-radius: 3px;
    padding: 0 3px;
}
.cce-badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 0.1rem 0.5rem;
    border-radius: 999px;
    margin-left: 0.4rem;
}
.cce-badge-accepted { background: var(--cce-ins-bg); color: var(--cce-ins-text); }
.cce-badge-rejected { background: var(--cce-del-bg); color: var(--cce-del-text); }
.cce-badge-pending { background: #f1ece0; color: var(--cce-ink-soft); }
</style>
"""


def diff_card_html(card: dict) -> tuple[str, str, str, str]:
    """(label, meta, extra, body) for one diff card.

    The single source of the card's visual content, used by BOTH the Review
    tab and the present_files cards in chat — the two are the same view of
    the same edit, so they render from the same code rather than from two
    copies that agree until someone edits one of them.

    `extra` is the Learning-Objective line for research-notes cards and ""
    for topic cards. Callers own everything around the content: the Review
    tab appends its status badge to `label` and its accept / reject /
    request-changes row after `body`; the chat cards are read-only, since
    those buttons are keyed to the Review tab's card ids and a second set
    claiming the same decision would be two widgets fighting over one piece
    of state.
    """
    kind = card["kind"]
    if kind == "topic_group":
        before_g, after_g = card["before"], card["after"]
        before_flat = flatten_topic_group(before_g) if before_g else ""
        after_flat = flatten_topic_group(after_g) if after_g else ""
        before_name = before_g["topic"] if before_g else ""
        after_name = after_g["topic"] if after_g else ""
        if not before_g:
            label = f'<ins>{html.escape(after_name)}</ins>'
            meta = f"New topic · {len(after_g['blocks'])} slides"
        elif not after_g:
            label = f'<del>{html.escape(before_name)}</del>'
            meta = f"Removed topic · {len(before_g['blocks'])} slides"
        else:
            label = word_diff_html(before_name, after_name)
            meta = f"Topic · {len(before_g['blocks'])} → {len(after_g['blocks'])} slides"
        return label, meta, "", text_diff_html(before_flat, after_flat)

    if kind == "context_row":
        b, a = card["before"], card["after"]
        return (
            html.escape(b["Subtopic"]),
            f'Research notes · Topic: {b["Topic"]} · Subtopic: {b["Subtopic"]}',
            f'Learning Objective: {html.escape(b["Learning Objective"])}',
            paragraph_diff_html(b["Research Notes"], a["Research Notes"]),
        )

    # context_new
    a = card["after"]
    body = "".join(
        f'<p class="cce-diff-changed"><ins>{html.escape(par)}</ins></p>'
        for par in _split_paragraphs(a["Research Notes"])
    )
    return (
        html.escape(a["Subtopic"]),
        f'Research notes · Topic: {html.escape(a["Topic"])} · Subtopic: {html.escape(a["Subtopic"])}',
        f'Learning Objective: {html.escape(a["Learning Objective"])}',
        body,
    )


_MD_MEDIA_RE = re.compile(r'(!?)\[([^\]]*)\]\(\s*([^)\s]*)(?:\s+"[^"]*")?\s*\)')


_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".avif", ".bmp", ".tif", ".tiff")


def _looks_like_image(url: str) -> bool:
    """Does this url point at an image file?

    Query strings are the norm on this content's urls (`?strip=all`,
    `?itok=…`, `?auto=false`), so the extension check runs on the path
    alone.
    """
    return url.split("#", 1)[0].split("?", 1)[0].lower().endswith(_IMAGE_EXT)


def _media_html(is_image: bool, alt: str, url: str) -> str:
    """One markdown image or link as HTML.

    Whether something renders as a picture is decided by the *url*, not by
    which markdown syntax it was written in. Course content routinely
    points at a diagram with an ordinary link — `[view full size air gap
    diagram](…/air-gap-diagram.png)` — and treating that as text just
    because it lacks a leading `!` shows a reviewer a url where there
    should be a picture. So: anything ending in an image extension is
    rendered as an image, and an explicit `![...]` is trusted as an image
    even when its url carries no extension (CDN urls often don't).

    Everything else becomes a link, which is also where anything that
    can't be displayed lands — a relative url can't resolve from this app,
    so it's a link rather than a guaranteed broken-image icon.

    Schemes are an allow-list — http, https, or a site-relative path.
    `javascript:`, `data:`, and anything else fall through to plain text
    rather than becoming an attribute, which matters because these urls
    come from sheet content the agent rewrites.
    """
    lower = url.lower()
    absolute = lower.startswith(("http://", "https://"))
    relative = url.startswith(("/", "./", "../")) and not lower.startswith("//")
    href = url if (absolute or relative) else None

    # The content wraps alt text in literal quotes ("three-compartment sink
    # …"); they're punctuation from the source, not part of the caption.
    label = (alt or "").strip().strip('"').strip("'")

    if not href:
        return html.escape(label or url)
    safe_href = html.escape(href, quote=True)
    if absolute and (is_image or _looks_like_image(href)):
        # Wrapped in a link to its own source: the card renders a capped
        # thumbnail, and the full-resolution original is one click away for
        # anyone who needs to read the labels inside a diagram.
        return (f'<a class="cce-plain-img-link" href="{safe_href}" target="_blank" '
                f'rel="noopener noreferrer">'
                f'<img class="cce-plain-img" src="{safe_href}" '
                f'alt="{html.escape(label, quote=True)}" loading="lazy"></a>')
    return (f'<a href="{safe_href}" target="_blank" rel="noopener noreferrer">'
            f'{html.escape(label or href)}</a>')


def _plain_paragraph_html(text: str) -> str:
    """Escape a paragraph, rendering markdown images and links as HTML."""
    out, pos = [], 0
    for m in _MD_MEDIA_RE.finditer(text):
        out.append(html.escape(text[pos:m.start()]))
        pos = m.end()
        out.append(_media_html(m.group(1) == "!", m.group(2), m.group(3)))
    out.append(html.escape(text[pos:]))
    return "".join(out)


# NUL-delimited so html.escape() leaves it untouched and _tokenize() keeps it
# as one word — a placeholder that survived escaping but got split across
# <del>/<ins> spans would come back as visible garbage.
_MEDIA_PH = "\x00M{}\x00"


def _stash_media(text: str, store: dict[str, str]) -> str:
    """Swap markdown media out for placeholders before diffing.

    Diffing the raw `![alt](url)` is why images didn't render in the redline:
    a long url is dozens of word tokens, so an image next to a rewritten
    sentence gets sliced across <del>/<ins> spans and can't be turned back
    into an <img> afterwards. Standing in a single token instead lets the
    word diff align on the prose, and the image is restored intact after.

    Identical media reuses its placeholder, so an untouched image is the
    *same* token on both sides and the diff correctly calls it unchanged;
    a swapped url produces a different one and shows up as a real change —
    old image struck, new image inserted.
    """
    def repl(m: re.Match) -> str:
        frag = _media_html(m.group(1) == "!", m.group(2), m.group(3))
        for ph, existing in store.items():
            if existing == frag:
                return ph
        ph = _MEDIA_PH.format(len(store))
        store[ph] = frag
        return ph

    return _MD_MEDIA_RE.sub(repl, text)


def _restore_media(rendered: str, store: dict[str, str]) -> str:
    for ph, frag in store.items():
        rendered = rendered.replace(ph, frag)
    return rendered


def card_plain_html(card: dict) -> str:
    """The card's content as it now reads, with no redline markup.

    The other half of the view toggle. A diff answers "what did you change";
    this answers "what does the slide say now", which is the question when
    you're judging whether the new copy is any good — reading prose with
    struck-through words spliced through it is a different, harder task.

    Renders the *after* state, falling back to *before* for a card whose
    after side is gone (a removed topic), since showing an empty panel would
    just look broken.
    """
    kind = card["kind"]
    if kind == "topic_group":
        group = card["after"] or card["before"]
        if not group:
            return '<p class="cce-diff-context">(nothing to show)</p>'
        out, last_sub = [], None
        for b in group["blocks"]:
            sub = b.get("Subtopic", "")
            if sub != last_sub:
                out.append(f'<div class="cce-card-meta">{html.escape(sub)}</div>')
                last_sub = sub
            title = f"[{b.get('Slide Type', '')}] {b.get('Slide Chunk Title', '')}".strip()
            out.append(f'<div class="cce-card-label">{html.escape(title)}</div>')
            out.append(
                '<div class="cce-diff-text">'
                + "".join(
                    f'<p class="cce-diff-context">{_plain_paragraph_html(par)}</p>'
                    for par in _split_paragraphs(b.get("Slide Chunk", ""))
                )
                + "</div>"
            )
        return "".join(out)

    row = card["after"] or card["before"]
    return "".join(
        f'<p class="cce-diff-context">{_plain_paragraph_html(par)}</p>'
        for par in _split_paragraphs(row.get("Research Notes", ""))
    )


_COMPONENT_DIR = _CCE_DIR / "components" / "comment_selector"


def _comment_component():
    """The select-and-comment component, or a string saying why it's absent.

    A bidirectional custom component rather than components.v1.html: the
    page's own markdown strips <script>, so text selection is invisible
    there, and components.v1.html can run JS but has no channel back to
    Python. declare_component has both, and needs no npm — index.html
    implements the three postMessage calls the protocol actually requires.

    The declaration itself lives in _comment_component.py and cannot move
    here: declare_component asserts on `inspect.getmodule()` of its caller,
    which is unresolvable for a page Streamlit runs via exec(). See that
    module's docstring.
    """
    if _cmt.comment_selector is None:
        return _cmt.load_error or "component not declared"
    return _cmt.comment_selector


def _blocks_for_component(before: str, after: str, kind: str) -> list[dict]:
    """The file's current blocks, prose pre-rendered server-side.

    The HTML is built here with the same _plain_paragraph_html the static
    plain view uses, so images and links behave identically inside the
    iframe and the sanitizing rules are not reimplemented in JavaScript.
    """
    out = []
    for b in pres.parse_blocks(after, kind):
        body = pres.block_body(b, kind)
        out.append({
            "id": b["_id"],
            "label": pres.block_label(b, kind),
            "subtopic": b.get("Subtopic", ""),
            "html": "".join(f"<p>{_plain_paragraph_html(par)}</p>"
                            for par in _split_paragraphs(body)),
        })
    return out


def compose_feedback_message(path: str, comments: list[dict], overall: str = "") -> str:
    """Turn selected-text comments into one message for the agent.

    Each comment quotes the selection verbatim. That is the same handle the
    Review tab's Reject flow uses, and for the same reason: the exact text
    is the only reliable way to point at a passage, since block IDs get
    renumbered and titles get rewritten. A selection spanning two slides
    keeps both block labels, so "this transition doesn't lead into the next
    slide" stays expressible.

    **This function states no scope of its own.** An earlier version closed
    every message with "Apply exactly these changes and nothing else — leave
    every other block in this file, and every other file, as it is." That is
    a reasonable default and completely wrong whenever it isn't: a real run
    whose comments asked for whole-topic rewrites across every topic got
    narrowed by that trailer, and the agent followed the canned sentence
    over the user's own words. The comments are the instruction. Anything
    about how far they reach is the user's to say, in `overall`, which is
    why the composer only relays and never editorializes.
    """
    lines = [f"Feedback on `{path}`. Each item quotes the exact text I selected.", ""]
    for i, c in enumerate(comments, 1):
        where = c.get("where") or ""
        lines.append(f"{i}. {where}".rstrip())
        lines.append(f'   Selected text: """{(c.get("quote") or "").strip()}"""')
        lines.append(f'   What I want: {(c.get("note") or "").strip()}')
        lines.append("")
    if overall.strip():
        lines.append(overall.strip())
    return "\n".join(lines).rstrip() + "\n"


def render_comment_surface(p: dict, before: str, after: str, kind: str, uid: str,
                          *, pending_key: str = "cce_pending_feedback") -> bool:
    """Selectable plain view. Returns True if it rendered.

    Comments live in session_state keyed by uid and are handed back to the
    component on every rerun, because the iframe is re-created each time and
    would otherwise lose them.

    `pending_key` is where the composed message is left for the page to send on
    its next run. The two pages send a turn differently — one posts to a
    Managed Agents session, the other opens a LangGraph run — but neither can
    start that turn from inside this expander, where it would render into the
    middle of an earlier turn. Both queue it instead.
    """
    component = _comment_component()
    if isinstance(component, str):
        # Surfaced, not swallowed. A silent fallback here is indistinguishable
        # from the component loading and simply not responding to selection —
        # which is exactly the confusion it caused the first time out.
        st.warning(f"Select-to-comment unavailable — {component}")
        return False
    blocks = _blocks_for_component(before, after, kind)
    if not blocks:
        st.caption("No parseable blocks in this file, so there is nothing to select.")
        return False

    store = st.session_state.setdefault("cce_comments", {})
    notes = st.session_state.setdefault("cce_comment_notes", {})
    pending = store.get(uid, [])
    try:
        value = component(blocks=blocks, file=p.get("path", ""), pending=pending,
                          overall=notes.get(uid, ""), key=f"cmt_{uid}", default=None)
    except Exception as exc:  # noqa: BLE001 — never take the page down, but never hide it either
        st.warning(f"Select-to-comment failed to render — {exc}")
        return False

    if isinstance(value, dict):
        store[uid] = value.get("comments") or []
        notes[uid] = value.get("overall") or ""
        # Only a "send" click becomes a turn. The component also reports on
        # every add/delete so the comments survive a rerun, and those must
        # not each fire a message — hence the seq guard, since Streamlit
        # replays the last component value on every subsequent rerun too.
        if value.get("action") == "send" and store[uid]:
            seen = st.session_state.setdefault("cce_comment_seq", {})
            if seen.get(uid) != value.get("seq"):
                seen[uid] = value.get("seq")
                st.session_state[pending_key] = compose_feedback_message(
                    p.get("path", ""), store[uid], notes.get(uid, ""))
                store[uid] = []
                notes[uid] = ""
                st.rerun()
    return True


# ---------------------------------------------------------------------------
# Diff building — parses before/after with the SAME regex the deployed
# commit scripts use (imported, not reimplemented), so a card here means
# exactly what commit_context.py / commit_workspace.py will actually do.
# ---------------------------------------------------------------------------
def _parse_lo_blocks(text: str) -> list[dict]:
    starts = list(cc_context.LO_BLOCK_START.finditer(text))
    out = []
    for i, m in enumerate(starts):
        body = text[m.end():starts[i + 1].start() if i + 1 < len(starts) else len(text)]
        block = cc_context.parse_lo_block(body)
        if block:
            out.append(block)
    return out


def _parse_topic_blocks(text: str) -> list[dict]:
    starts = list(cc_topics.BLOCK_START.finditer(text))
    out = []
    for i, m in enumerate(starts):
        body = text[m.end():starts[i + 1].start() if i + 1 < len(starts) else len(text)]
        block = cc_topics.parse_block(body)
        if block:
            out.append(block)
    return out


def diff_context_file(before_text: str, after_text: str) -> list[dict]:
    """Per-row cards. A deleted block is NOT a card — per the skill's own
    contract, a missing block means no change was requested for that row."""
    before_by_key = {(b["Topic"], b["Subtopic"], b["Learning Objective"]): b for b in _parse_lo_blocks(before_text)}
    after_by_key = {(b["Topic"], b["Subtopic"], b["Learning Objective"]): b for b in _parse_lo_blocks(after_text)}
    cards = []
    for key, after_b in after_by_key.items():
        before_b = before_by_key.get(key)
        if before_b is None:
            cards.append({"kind": "context_new", "key": key, "before": None, "after": after_b})
        elif before_b["Research Notes"] != after_b["Research Notes"]:
            cards.append({"kind": "context_row", "key": key, "before": before_b, "after": after_b})
    return cards


def _block_repr(b: dict) -> str:
    return f"{b.get('Slide Type', '')}|{b.get('Slide Chunk Title', '')}|{b.get('Slide Chunk', '')}"


def _topic_groups(blocks: list[dict]) -> list[dict]:
    """Consecutive blocks sharing a Topic, in sheet order. Consecutive, not
    keyed by name: a renamed topic must still group with its old self, and
    grouping by name would instead scatter it into a "deleted" and an
    "added" group."""
    groups: list[dict] = []
    for b in blocks:
        name = b.get("Topic", "")
        if groups and groups[-1]["topic"] == name:
            groups[-1]["blocks"].append(b)
        else:
            groups.append({"topic": name, "blocks": [b]})
    return groups


def flatten_topic_group(group: dict) -> str:
    """One topic rendered as the plain text the diff runs over — topic name,
    then each slide's subtopic/type/title header followed by its chunk.

    Diffing this flattened form (rather than field-by-field per block) is
    what the slide-chunks checklist view does, and it's why that view reads
    well: every kind of edit — a retitled slide, a moved paragraph, a
    renamed topic or subtopic, a slide split in two — shows up as ordinary
    line changes in one continuous document, instead of having to be
    classified into a card type first."""
    lines = [f"Topic: {group['topic']}"]
    last_sub = None
    for b in group["blocks"]:
        sub = b.get("Subtopic", "")
        if sub != last_sub:
            lines += ["", f"Subtopic: {sub}"]
            last_sub = sub
        lines += ["", f"[{b.get('Slide Type', '')}] {b.get('Slide Chunk Title', '')}"]
        lines += b.get("Slide Chunk", "").split("\n")
    return "\n".join(lines)


def diff_topic_file(before_text: str, after_text: str) -> list[dict]:
    """One card per topic, not per slide.

    Per-slide cards forced every edit into a before/after slide pair, which
    can't represent the edits that don't map that way — a renamed topic or
    subtopic, a slide split in two, content moved between slides — and left
    those rendering as an unreadable "structural change" summary. A topic is
    the smallest unit that all of those stay inside, so the card is the
    topic and its body is a full text redline of it.

    Topics are matched between before and after by content similarity (the
    same order-preserving matcher the paragraph diff uses) so a topic whose
    name changed still pairs with its original."""
    before_groups = _topic_groups(_parse_topic_blocks(before_text))
    after_groups = _topic_groups(_parse_topic_blocks(after_text))
    before_flat = [flatten_topic_group(g) for g in before_groups]
    after_flat = [flatten_topic_group(g) for g in after_groups]
    pairs, _unmatched_old, _unmatched_new = _best_paragraph_matching(before_flat, after_flat, threshold=0.2)
    pair_map = dict(pairs)

    cards: list[dict] = []
    ni_cursor = 0
    for oi, group in enumerate(before_groups):
        if oi not in pair_map:
            cards.append({"kind": "topic_group", "before": group, "after": None})
            continue
        ni = pair_map[oi]
        for k in range(ni_cursor, ni):
            cards.append({"kind": "topic_group", "before": None, "after": after_groups[k]})
        if before_flat[oi] != after_flat[ni]:
            cards.append({"kind": "topic_group", "before": group, "after": after_groups[ni]})
        ni_cursor = ni + 1
    for k in range(ni_cursor, len(after_groups)):
        cards.append({"kind": "topic_group", "before": None, "after": after_groups[k]})
    return cards


_TOKEN_SPLIT = re.compile(r"\s+|\S+")
_PARA_SPLIT = re.compile(r"\n\s*\n+")


def _tokenize(text: str) -> list[str]:
    """Words AND the whitespace between them, each as their own token.

    Splitting on a literal " " (the earlier version) silently lumped a
    multi-line paragraph into one giant "word" at every newline, since \\n
    was never a delimiter — that's what made prior diffs look like the
    entire block changed even when only a sentence did. Keeping whitespace
    as real tokens lets SequenceMatcher align text across line breaks.
    """
    return _TOKEN_SPLIT.findall(text)


def _split_paragraphs(text: str) -> list[str]:
    """Blank-line-separated chunks. Approximate (doesn't preserve the exact
    number of blank lines) — fine, since this is only ever used to render a
    diff, never to reconstruct text sent back to the agent; reject/revert
    messages always quote the raw original field, not this rendering."""
    return [p.strip("\n") for p in _PARA_SPLIT.split(text) if p.strip()]


def word_diff_html(before: str, after: str) -> str:
    """Inline <del>/<ins> word diff for the redline look. Intended for
    short, single-paragraph fields (e.g. a slide title) — see
    paragraph_diff_html for anything with multiple paragraphs."""
    before_tok, after_tok = _tokenize(before), _tokenize(after)
    sm = difflib.SequenceMatcher(None, before_tok, after_tok, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out.append(html.escape("".join(before_tok[i1:i2])))
        elif tag == "delete":
            out.append(f"<del>{html.escape(''.join(before_tok[i1:i2]))}</del>")
        elif tag == "insert":
            out.append(f"<ins>{html.escape(''.join(after_tok[j1:j2]))}</ins>")
        elif tag == "replace":
            out.append(f"<del>{html.escape(''.join(before_tok[i1:i2]))}</del>")
            out.append(f"<ins>{html.escape(''.join(after_tok[j1:j2]))}</ins>")
    return "".join(out)


def _word_diff_pair(before: str, after: str) -> tuple[str, str]:
    """Word-level diff of one paragraph pair, returned as two independent
    HTML strings (old, new) instead of one interleaved stream — the point
    is that each side stays readable as continuous prose with its own
    highlighted words, which is what actually fixes a heavily-rewritten
    paragraph reading as word salad. Same tokenizer as word_diff_html."""
    before_tok, after_tok = _tokenize(before), _tokenize(after)
    sm = difflib.SequenceMatcher(None, before_tok, after_tok, autojunk=False)
    old_parts, new_parts = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            seg = html.escape("".join(before_tok[i1:i2]))
            old_parts.append(seg)
            new_parts.append(seg)
        elif tag == "delete":
            old_parts.append(f'<span class="cce-word-del">{html.escape("".join(before_tok[i1:i2]))}</span>')
        elif tag == "insert":
            new_parts.append(f'<span class="cce-word-ins">{html.escape("".join(after_tok[j1:j2]))}</span>')
        elif tag == "replace":
            old_parts.append(f'<span class="cce-word-del">{html.escape("".join(before_tok[i1:i2]))}</span>')
            new_parts.append(f'<span class="cce-word-ins">{html.escape("".join(after_tok[j1:j2]))}</span>')
    return "".join(old_parts), "".join(new_parts)


def _word_diff_lines(before: str, after: str) -> tuple[str, str]:
    """Line-then-word diff for one paragraph pair.

    _word_diff_pair tokenizes the whole paragraph as one flat stream, with
    newlines as just another whitespace token — SequenceMatcher's LCS
    matching isn't line-aware, so a paragraph that mixes a heading, prose,
    and a bullet list (each on its own line, no blank line between them,
    so _split_paragraphs never separates them) could match a word from one
    line against an unrelated line on the other side instead of comparing
    it to its real counterpart line. That's what made a sentence starting
    on a fresh line look like it was "missing" or diffed against the wrong
    line/paragraph.

    Fixed by aligning lines first (a nested SequenceMatcher over
    before.split("\\n") / after.split("\\n")), then running the existing
    word-level diff only within matched line pairs — unmatched lines are
    rendered as fully added/removed rather than word-diffed against a
    stranger. Returned as two independent HTML strings joined by <br>,
    same contract as _word_diff_pair."""
    before_lines = before.split("\n")
    after_lines = after.split("\n")
    sm = difflib.SequenceMatcher(None, before_lines, after_lines, autojunk=False)
    old_parts, new_parts = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for line in before_lines[i1:i2]:
                seg = html.escape(line)
                old_parts.append(seg)
                new_parts.append(seg)
        elif tag == "replace":
            n = min(i2 - i1, j2 - j1)
            for k in range(n):
                o, nw = _word_diff_pair(before_lines[i1 + k], after_lines[j1 + k])
                old_parts.append(o)
                new_parts.append(nw)
            for line in before_lines[i1 + n:i2]:
                old_parts.append(f'<span class="cce-word-del">{html.escape(line)}</span>')
            for line in after_lines[j1 + n:j2]:
                new_parts.append(f'<span class="cce-word-ins">{html.escape(line)}</span>')
        elif tag == "delete":
            for line in before_lines[i1:i2]:
                old_parts.append(f'<span class="cce-word-del">{html.escape(line)}</span>')
        elif tag == "insert":
            for line in after_lines[j1:j2]:
                new_parts.append(f'<span class="cce-word-ins">{html.escape(line)}</span>')
    return "<br>".join(old_parts), "<br>".join(new_parts)


_MD_LINK_RE = re.compile(r"!?\[[^\]]*\]\([^)]*\)")


def _similarity_key(text: str) -> str:
    """Strip markdown image/link syntax before scoring paragraph
    similarity. Used only to decide which old/new paragraphs correspond to
    each other — rendering always uses the raw paragraph text. A research
    note often embeds `![alt](long-url)` / `[text](long-url)` boilerplate;
    those URLs make up a large share of the paragraph's characters but are
    essentially random between an old and new version (or shared by
    coincidence with an unrelated paragraph), so leaving them in skews the
    similarity score away from the actual prose that identifies the match."""
    return _MD_LINK_RE.sub(" ", text)


#: Characters of each side actually compared when scoring correspondence.
_SIMILARITY_SCAN = 2000


def _best_paragraph_matching(
    old_paras: list[str], new_paras: list[str], threshold: float = 0.35
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Pair paragraphs within a "replace" range by content similarity
    instead of raw position — but only accept matches that preserve
    reading order (old index and new index both increasing across accepted
    pairs). Order preservation matters: an earlier version of this that
    accepted the globally-best-similarity match regardless of order, then
    rendered all matched pairs first followed by all leftover deletions
    and insertions, silently reshuffled the paragraph sequence itself
    (a matched pair from later in the document could render before an
    unmatched deletion from earlier in it).

    The top-level SequenceMatcher only ever calls two paragraphs "equal"
    when they're byte-identical, so any edited paragraph — even a single
    reworded sentence — falls into a replace/delete/insert opcode. Inside
    a replace range spanning more than one paragraph on either side, plain
    positional pairing (old[k] with new[k]) is wrong whenever a paragraph
    was inserted or deleted ahead of others in the same range: everything
    after the shift gets falsely paired against unrelated content.

    Considers candidate pairs by descending similarity (quick ratio — cheap,
    good enough to tell "reworded" from "unrelated" at this granularity),
    accepting each only if it doesn't cross an already-accepted pair's
    indices in either direction. What's left below `threshold` or crossing
    an accepted pair renders as a plain delete/insert at its own position
    instead of a misleading word-salad "replace" of two unrelated
    paragraphs. Returns (pairs, unmatched_old_indices, unmatched_new_indices);
    pairs are sorted by old index, and (because of the order constraint)
    are therefore also sorted by new index — the caller can walk both
    sequences in lockstep to render everything in original document
    order."""
    # Scored on a prefix, not the whole string. SequenceMatcher.ratio() is
    # quadratic in length, and this same matcher is called on two granularities:
    # paragraphs (short — the cap never binds) and whole flattened topic groups
    # (multi-kilobyte, where one comparison alone cost seconds and was the
    # slowest thing on the page).
    #
    # Safe because the score only decides *which* old item corresponds to which
    # new one; the rendering always diffs the full text. flatten_topic_group
    # puts "Topic: <name>" and the first slides at the top, so the opening
    # kilobytes are the most identifying part of a topic anyway — and two
    # candidates that agree for 2000 characters and diverge only afterwards
    # would be a coin toss on the full score too.
    clean_old = [_similarity_key(p)[:_SIMILARITY_SCAN] for p in old_paras]
    clean_new = [_similarity_key(p)[:_SIMILARITY_SCAN] for p in new_paras]
    candidates = []
    # One matcher, reused. difflib indexes its `b` sequence on assignment and
    # keeps that index across set_seq1 calls, so holding `b` fixed in the outer
    # loop builds each index once instead of len(clean_old) times.
    matcher = difflib.SequenceMatcher(None, autojunk=False)
    for ni, npv in enumerate(clean_new):
        matcher.set_seq2(npv)
        for oi, op in enumerate(clean_old):
            matcher.set_seq1(op)
            # real_quick_ratio and quick_ratio are both upper bounds on
            # ratio(), so a pair failing either cannot reach the threshold and
            # is skipped without paying for the real comparison. This changes
            # no result — it only avoids computing scores that are already
            # known to lose.
            #
            # It matters most where this is called on whole flattened topic
            # groups rather than paragraphs: that is an all-pairs comparison of
            # multi-kilobyte strings, and it was the single slowest thing on
            # the page — 4.9s to diff one 14KB slide-chunks file, growing
            # quadratically with the number of topics.
            if matcher.real_quick_ratio() < threshold or matcher.quick_ratio() < threshold:
                continue
            # Real ratio(), not quick_ratio(), to *decide*: quick_ratio is a
            # character-multiset bound, not actual similarity — two unrelated
            # paragraphs that happen to share a lot of common words (or are
            # both padded with long markdown image/link URLs) can score
            # deceptively high on it and edge out the real match. ratio()
            # does the actual LCS-based comparison, which is what tells
            # "genuinely reworded" from "coincidentally overlapping words"
            # at this granularity. Scored on _similarity_key'd text so a
            # paragraph's markdown link/image boilerplate (often the bulk
            # of its character count, and never shared between an old URL
            # and a new one) doesn't drown out the actual prose.
            ratio = matcher.ratio()
            if ratio >= threshold:
                candidates.append((ratio, oi, ni))
    candidates.sort(key=lambda c: c[0], reverse=True)
    used_old, used_new, pairs = set(), set(), {}
    for _ratio, oi, ni in candidates:
        if oi in used_old or ni in used_new:
            continue
        # Reject any pair that would cross an already-accepted one — keeps
        # the accepted set monotonic in both indices.
        crosses = any((oi < poi) != (ni < pni) for poi, pni in pairs.items())
        if crosses:
            continue
        used_old.add(oi)
        used_new.add(ni)
        pairs[oi] = ni
    ordered_pairs = [(oi, pairs[oi]) for oi in sorted(pairs)]
    unmatched_old = [oi for oi in range(len(old_paras)) if oi not in used_old]
    unmatched_new = [ni for ni in range(len(new_paras)) if ni not in used_new]
    return ordered_pairs, unmatched_old, unmatched_new


def paragraph_diff_html(before: str, after: str) -> str:
    """The actual redline view: diff at paragraph granularity first —
    unchanged paragraphs render as plain full-width context (the head/tail
    that was missing) — and render every changed paragraph pair side by
    side, old on the left / new on the right, each with its own word-level
    highlighting, rather than one interleaved <del>/<ins> stream.

    That structure (line-then-word diff, side-by-side columns) mirrors
    compare_text_versions in services/helper_functions.py, the diff viewer
    already used for the slide-chunks checklist agent
    (agents/slide_chunks/visualize_slide_chunks_diff.py) — adapted here to
    paragraph granularity since research notes are prose, not code lines.
    Two independent columns are what actually fix a heavily-rewritten
    paragraph reading as word salad: each side stays legible as continuous
    prose instead of both directions being spliced into one run-on line.
    """
    media: dict[str, str] = {}
    before_paras = _split_paragraphs(_stash_media(before, media))
    after_paras = _split_paragraphs(_stash_media(after, media))
    sm = difflib.SequenceMatcher(None, before_paras, after_paras, autojunk=False)
    rows = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for p in before_paras[i1:i2]:
                rows.append(f'<tr><td class="cce-diff-ctx" colspan="2">{html.escape(p)}</td></tr>')
        elif tag == "replace":
            # Pair paragraphs by content similarity, not position — a
            # positional k<->k pairing falsely matched unrelated paragraphs
            # whenever one was inserted/deleted ahead of others in this
            # range. See _best_paragraph_matching for why. Matches are
            # guaranteed order-preserving (old index and new index both
            # increasing across accepted pairs), so walking old_slice in
            # order and flushing each matched pair's "skipped-over" new
            # paragraphs as insertions right before it reconstructs the
            # true document order — unlike rendering all pairs, then all
            # deletions, then all insertions as separate groups, which
            # reshuffled the sequence.
            old_slice = before_paras[i1:i2]
            new_slice = after_paras[j1:j2]
            pairs, _unmatched_old, _unmatched_new = _best_paragraph_matching(old_slice, new_slice)
            pair_map = dict(pairs)
            ni_cursor = 0
            for oi, p in enumerate(old_slice):
                if oi in pair_map:
                    ni = pair_map[oi]
                    for k in range(ni_cursor, ni):
                        rows.append(f'<tr><td></td><td class="cce-diff-new">{html.escape(new_slice[k])}</td></tr>')
                    old_html, new_html = _word_diff_lines(p, new_slice[ni])
                    rows.append(
                        f'<tr><td class="cce-diff-old">{old_html}</td>'
                        f'<td class="cce-diff-new">{new_html}</td></tr>'
                    )
                    ni_cursor = ni + 1
                else:
                    rows.append(f'<tr><td class="cce-diff-old">{html.escape(p)}</td><td></td></tr>')
            for k in range(ni_cursor, len(new_slice)):
                rows.append(f'<tr><td></td><td class="cce-diff-new">{html.escape(new_slice[k])}</td></tr>')
        elif tag == "delete":
            for p in before_paras[i1:i2]:
                rows.append(f'<tr><td class="cce-diff-old">{html.escape(p)}</td><td></td></tr>')
        elif tag == "insert":
            for p in after_paras[j1:j2]:
                rows.append(f'<tr><td></td><td class="cce-diff-new">{html.escape(p)}</td></tr>')
    return _restore_media(
        f'<table class="cce-diff-table"><tbody>{"".join(rows)}</tbody></table>', media)


def text_diff_html(before: str, after: str) -> str:
    """Side-by-side line redline of two flattened topics: unchanged lines as
    full-width context, changed line pairs word-highlighted on each side,
    pure additions/removals in their own column.

    Same shape as compare_text_versions in services/helper_functions.py —
    the slide-chunks checklist diff — so the two views read alike; the
    difference is only that this one uses the page's own cce-diff-* styling
    and the shared word tokenizer."""
    # Images stand in as single placeholder tokens for the duration of the
    # diff and come back as real <img> at the end — see _stash_media.
    media: dict[str, str] = {}
    before_lines = _stash_media(before, media).splitlines()
    after_lines = _stash_media(after, media).splitlines()
    sm = difflib.SequenceMatcher(None, before_lines, after_lines, autojunk=False)
    rows = []

    def classes(line: str, base: str) -> str:
        # The structural lines flatten_topic_group emits (topic name,
        # subtopic name, the [Type] Title header of each slide) carry the
        # reader through a topic that can run dozens of lines, so they get
        # their own weight instead of reading as more body prose.
        head = line.startswith(("Topic: ", "Subtopic: ", "["))
        return f"{base} cce-diff-head" if head else base

    def cell(css: str, inner: str, raw: str) -> str:
        # A visually empty cell still needs a non-breaking space, or the
        # blank lines that separate slides collapse to zero height and the
        # two columns drift out of vertical alignment.
        return f'<td class="{classes(raw, css)}">{inner or "&nbsp;"}</td>'

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for line in before_lines[i1:i2]:
                rows.append(
                    f'<tr><td class="{classes(line, "cce-diff-ctx")}" colspan="2">'
                    f'{html.escape(line) or "&nbsp;"}</td></tr>'
                )
        elif tag == "replace":
            old_lines, new_lines = before_lines[i1:i2], after_lines[j1:j2]
            for k in range(max(len(old_lines), len(new_lines))):
                if k < len(old_lines) and k < len(new_lines):
                    old_html, new_html = _word_diff_pair(old_lines[k], new_lines[k])
                    rows.append(
                        f'<tr>{cell("cce-diff-old", old_html, old_lines[k])}'
                        f'{cell("cce-diff-new", new_html, new_lines[k])}</tr>'
                    )
                elif k < len(old_lines):
                    rows.append(f'<tr>{cell("cce-diff-old", html.escape(old_lines[k]), old_lines[k])}<td></td></tr>')
                else:
                    rows.append(f'<tr><td></td>{cell("cce-diff-new", html.escape(new_lines[k]), new_lines[k])}</tr>')
        elif tag == "delete":
            for line in before_lines[i1:i2]:
                rows.append(f'<tr>{cell("cce-diff-old", html.escape(line), line)}<td></td></tr>')
        elif tag == "insert":
            for line in after_lines[j1:j2]:
                rows.append(f'<tr><td></td>{cell("cce-diff-new", html.escape(line), line)}</tr>')
    return _restore_media(
        f'<table class="cce-diff-table cce-diff-doc"><tbody>{"".join(rows)}</tbody></table>', media)


def format_topic_block_for_message(b: dict) -> str:
    parts = ["####**Topic:**", b.get("Topic", ""), "####**Subtopic:**", b.get("Subtopic", ""),
              "####**Slide Chunk:**"]
    if b.get("Slide Type"):
        parts.append(f"Slide Type: {b['Slide Type']}")
    if b.get("Slide Chunk Title"):
        parts.append(f"Title: {b['Slide Chunk Title']}")
    if b.get("Slide Chunk"):
        parts.append(f"Content: {b['Slide Chunk']}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Counts. Only the MDA build carries these: its `present.py` measures the file
# inside the sandbox and ships the numbers with the payload, so the page states
# what the deployment measured rather than counting a second time out here and
# risking a different answer to the same question.
# ---------------------------------------------------------------------------


def counts_line(counts: dict) -> str:
    """A one-line summary of a file's measured change."""
    blocks = counts.get("blocks") or {}
    words = counts.get("words") or {}
    parts = []
    if blocks:
        arrow = "→"
        parts.append(f"blocks {blocks.get('before')} {arrow} {blocks.get('after')}")
    if words:
        pct = words.get("change_pct")
        suffix = f" ({pct:+.1f}%)" if isinstance(pct, (int, float)) else ""
        parts.append(f"words {words.get('before')} → {words.get('after')}{suffix}")
    types = counts.get("slide_types")
    if types:
        before_types = types.get("before") or {}
        after_types = types.get("after") or {}
        if before_types != after_types:
            parts.append(f"slide types {before_types} → {after_types}")
    return " · ".join(parts)


# ---------------------------------------------------------------------------
# Caching
#
# Streamlit re-runs the whole script on every interaction, and both pages
# re-render every turn of the conversation each time. Nothing here is cheap:
# a presented file is diffed at paragraph granularity, then every changed
# paragraph is diffed again word by word, and that ran from scratch on each
# rerun for every file the agent had ever shown — so a chat got measurably
# slower with each turn, for output that had not changed since it was first
# computed.
#
# These are pure functions of (before, after, kind), which is exactly what a
# cache key wants. The content is the key, so an edited file misses the cache
# and is recomputed, while an untouched one is free forever.
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False, max_entries=512)
def rendered_cards(before: str, after: str, kind: str) -> list[tuple[str, str, str, str, str]]:
    """(label, meta, extra, diff_body, plain_body) per card, built once.

    Returns finished HTML rather than card dicts because the HTML is the
    expensive half — `diff_card_html` runs the word-level diff — and because
    both views are built together, so flipping the toggle costs nothing.
    """
    cards = diff_context_file(before, after) if kind == "context" else diff_topic_file(before, after)
    out = []
    for card in cards:
        label, meta, extra, body = diff_card_html(card)
        out.append((label, meta, extra, body, card_plain_html(card)))
    return out
