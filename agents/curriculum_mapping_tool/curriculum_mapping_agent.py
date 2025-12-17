import logging
from typing import Any, Dict, List, Optional
from langchain.schema import Document
from modules.chain import Chain


course_mapping_prompt = """
You are an expert curriculum mapping reviewer focused on {source_name}.
Analyze how well each candidate resource serves the learner query. For every candidate:
- Reference it by "ID <number>"
- Briefly state why it is or is not a strong match
- Call out missing details, wrong scope, or broken links when relevant

After reviewing all candidates, pick the single best option (or NONE) and explain your decision.
Respond exactly in this format:
<analysis>
ID 0: ...
ID 1: ...
</analysis>
<best_id>...</best_id>
<best_reason>...</best_reason>
"""


video_mapping_prompt = """
You are reviewing YouTube training clips for a curriculum mapping task.
Use the transcript excerpt plus the timestamp window to judge each candidate.
For every entry:
- Reference "ID <number>"
- Mention whether the transcript segment truly teaches the requested concept
- Note the timestamp window (start-end) if it helps the learner jump to the right spot

Then pick the single best clip (or NONE) and justify your choice.
Output format:
<analysis>
ID 0: ...
ID 1: ...
</analysis>
<best_id>...</best_id>
<best_reason>...</best_reason>
"""


def _trim_text(text: Optional[str], limit: int = 750) -> str:
    if not text:
        return ""
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= limit:
        return collapsed
    trimmed = collapsed[:limit].rsplit(" ", 1)[0]
    return trimmed + " ..."


def _format_course_docs(docs: List[Document]) -> str:
    if not docs:
        return "No course candidates were retrieved."

    lines: List[str] = []
    for idx, doc in enumerate(docs):
        md = getattr(doc, "metadata", {}) or {}
        name = md.get("course_name") or md.get("title") or md.get("name") or "Untitled"
        link = md.get("course_link") or md.get("url") or md.get("link") or "N/A"
        snippet = _trim_text(getattr(doc, "page_content", ""))

        lines.append(f"ID {idx}: {name}")
        lines.append(f"Link: {link}")
        if md:
            extra = {
                k: v
                for k, v in md.items()
                if k not in {"course_name", "title", "name", "course_link", "url", "link"}
            }
            if extra:
                lines.append(f"Metadata: {extra}")
        lines.append(f"Snippet: {snippet or 'No summary available.'}")
        lines.append("")

    return "\n".join(lines).strip()


def _format_video_docs(videos: List[Dict[str, Any]]) -> str:
    if not videos:
        return "No video candidates were retrieved."

    lines: List[str] = []
    for idx, video in enumerate(videos):
        title = video.get("video_title") or video.get("title") or "Untitled video"
        channel = video.get("channel") or "Unknown channel"
        link = video.get("url") or video.get("video_link") or "N/A"
        start = video.get("start_time")
        end = video.get("end_time")
        transcript = _trim_text(video.get("transcript") or video.get("text_0"))

        lines.append(f"ID {idx}: {title}")
        lines.append(f"Channel: {channel}")
        lines.append(f"Timestamps: start={start}, end={end}")
        lines.append(f"Link: {link}")
        lines.append(f"Transcript excerpt: {transcript or 'Transcript unavailable.'}")
        lines.append("")

    return "\n".join(lines).strip()


def _parse_index(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    cleaned = text.strip().upper()
    if cleaned in {"", "NONE"}:
        return None
    try:
        return int(cleaned)
    except ValueError:
        return None


def select_best_course_resource(
    source_name: str,
    query: str,
    docs: List[Document],
    llm: str = "gemini_2_flash",
) -> Dict[str, Optional[str]]:
    logger = logging.getLogger(__name__)
    logger.debug(
        "Evaluating %s candidates for query='%s' (%d docs)",
        source_name,
        query,
        len(docs),
    )

    if not docs:
        reason = f"No {source_name} candidates were available."
        logger.warning("%s selection skipped: %s", source_name, reason)
        return {"index": None, "reason": reason}

    chain = Chain(
        llm=llm,
        tags=["analysis", "best_id", "best_reason"],
    )
    chain.add_message(
        role="system",
        content=course_mapping_prompt.format(source_name=source_name).strip(),
    )
    chain.add_message(
        role="user",
        content=f"Query: {query}\n\nCandidates:\n{_format_course_docs(docs)}",
    )
    response = chain.run()
    logger.debug(
        "%s agent output for query='%s': best=%s",
        source_name,
        query,
        response.get("best_id"),
    )

    return {
        "index": _parse_index(response.get("best_id")),
        "reason": response.get("best_reason", "").strip(),
        "analysis": response.get("analysis", "").strip(),
    }


def select_best_video_resource(
    query: str,
    videos: List[Dict[str, Any]],
    llm: str = "gemini_2_flash",
) -> Dict[str, Optional[str]]:
    logger = logging.getLogger(__name__)
    logger.debug("Evaluating video candidates for query='%s' (%d items)", query, len(videos))

    if not videos:
        reason = "No video candidates were available."
        logger.warning("Video selection skipped: %s", reason)
        return {"index": None, "reason": reason}

    chain = Chain(
        llm=llm,
        tags=["analysis", "best_id", "best_reason"],
    )
    chain.add_message(role="system", content=video_mapping_prompt.strip())
    chain.add_message(
        role="user",
        content=f"Query: {query}\n\nVideo candidates:\n{_format_video_docs(videos)}",
    )
    response = chain.run()
    logger.debug(
        "Video agent output for query='%s': best=%s",
        query,
        response.get("best_id"),
    )

    return {
        "index": _parse_index(response.get("best_id")),
        "reason": response.get("best_reason", "").strip(),
        "analysis": response.get("analysis", "").strip(),
    }
