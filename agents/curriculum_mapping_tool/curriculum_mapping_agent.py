import logging
from typing import Any, Dict, List, Optional
from langchain_classic.schema import Document
from modules.chain import Chain

resource_mapping_prompt = """
You are an expert curriculum mapping reviewer.

The learner query is created by combining the category and the course.
Your task is to select the single best resource from EACH source below.
For each source, analyze how well each candidate resource serves the learner query. Make sure the resource closely aligns with both the category and the course.
Beware of resources that closely match keywords but are wrong in scope or topic.
For every candidate in each source:
1. Reference it by "ID <number>"
2. Briefly state why it is or is not a strong match.
3. Call out missing details, wrong scope, or broken links when relevant.

Sources:
1. SkillCat courses
2. NexTech courses
3. YouTube video clips

Rules:
- Each source must be evaluated independently
- If no candidate matches well, respond with NONE for that source
- Beware of candidates that match keywords but are wrong in scope or topic

After selecting the best from each source, determine which ONE resource is the OVERALL BEST match for the learner query. Consider:
- Relevance to the category and course
- Completeness of content
- Quality of the resource
If all sources have NONE, respond with NONE for overall_best_source.
**Preference factor:**
- User preference is as follows: Source - skillcat courses > nextech courses > youtube videos.
- Give slight preference to higher-preference sources when coverage is adequate.

Respond EXACTLY in the following format:

<analysis>
SkillCat:
ID 0: ...
ID 1: ...

NexTech:
ID 0: ...
ID 1: ...

Video:
ID 0: ...
ID 1: ...
</analysis>

<skillcat_best_id>...</skillcat_best_id>
<nextech_best_id>...</nextech_best_id>
<video_best_id>...</video_best_id>

<skillcat_reason>...</skillcat_reason>
<nextech_reason>...</nextech_reason>
<video_reason>...</video_reason>

<overall_best_source>skillcat OR nextech OR video OR NONE</overall_best_source>
<overall_best_reason>Brief explanation of why this is the best overall match</overall_best_reason>
"""

def select_best_resources_unified(
    query: str,
    skillcat_docs: List[Document],
    nextech_docs: List[Document],
    video_docs: List[Dict[str, Any]],
    llm: str = "gemini_2_5_flash",
) -> Dict[str, Any]:

    chain = Chain(
        llm=llm,
        tags=[
            "skillcat_best_id", "nextech_best_id", "video_best_id",
            "overall_best_source", "overall_best_reason"
        ],
    )

    chain.add_message(
        role="system",
        content=resource_mapping_prompt.strip(),
    )

    chain.add_message(
        role="user",
        content=(
            f"Query: {query}\n\n"
            f"SkillCat Candidates:\n{_format_course_docs(skillcat_docs)}\n\n"
            f"NexTech Candidates:\n{_format_course_docs(nextech_docs)}\n\n"
            f"Video Candidates:\n{_format_video_docs(video_docs)}"
        ),
    )

    response = chain.run()

    # Parse overall_best_source (normalize to lowercase)
    overall_source = (response.get("overall_best_source") or "").strip().lower()
    if overall_source not in {"skillcat", "nextech", "video"}:
        overall_source = None

    return {
        "skillcat_idx": _parse_index(response.get("skillcat_best_id")),
        "nextech_idx": _parse_index(response.get("nextech_best_id")),
        "video_idx": _parse_index(response.get("video_best_id")),
        "overall_best_source": overall_source,
        "overall_best_reason": (response.get("overall_best_reason") or "").strip(),
    }




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


