"""
Source-agnostic candidate wrapper for Graphics Definition V2.

Pairs with search_wrapper.py: search_wrapper fills source columns; this candidate_wrapper reads those columns into normalized candidate items for scoring, selection, review and regeneration.

Adding a new source later means registering it here (column + parser + item shape) plus populating its column in Section 5. Downstream steps do not change unless the new source needs a genuinely new way of being shown to the model.
"""

from agents.graphics_definition_v2.candidate_search.pool_registry import (
    SOURCE_DRIVE_IMAGES,
    SOURCE_WEB_IMAGES,
    SOURCE_EXTERNAL_REFERENCES,
    SOURCE_HVAC_YOUTUBE,
    SOURCE_YOUTUBE_OTHER_CHANNELS,
    SOURCE_DRIVE_VIDEOS,
)

# Which sheet column feeds each source, per modality.
IMAGE_SOURCE_COLUMNS = {
    SOURCE_DRIVE_IMAGES: "drive_results",
    SOURCE_WEB_IMAGES: "web_results",
    SOURCE_EXTERNAL_REFERENCES: "external_ref_pool",
}

VIDEO_SOURCE_COLUMNS = {
    SOURCE_HVAC_YOUTUBE: "video_pool",
    SOURCE_YOUTUBE_OTHER_CHANNELS: "video_pool_other_channels",
    SOURCE_DRIVE_VIDEOS: "drive_video_pool",
    # External refs mix images + videos in the same column; read in get_video_candidates.
    SOURCE_EXTERNAL_REFERENCES: "external_ref_pool",
}

# Legacy batch "type" tag kept so downstream batching/multimodal code is unchanged.
_VIDEO_SOURCE_TYPE = {
    SOURCE_HVAC_YOUTUBE: "pool",
    SOURCE_YOUTUBE_OTHER_CHANNELS: "other",
    SOURCE_DRIVE_VIDEOS: "drive",
    # External-ref videos are frames-only (still frames), but keep their start/end window for loading.
    SOURCE_EXTERNAL_REFERENCES: "external_frames",
}

# Durable marker written into video_pool_filtered for external-ref videos.
EXTERNAL_FRAMES_ONLY_MARKER = "FramesOnly: yes"


def _include_set(enabled_sources, modality_sources):
    """
    Resolve which source ids to read for a modality.

    enabled_sources None means read every known column that is present (this
    matches historical behavior and keeps non-UI callers working unchanged).

    :param enabled_sources: Optional list of enabled source ids, or None
    :param modality_sources: Iterable of source ids valid for this modality
    :return: Set of source ids to include
    """
    modality = set(modality_sources)
    if enabled_sources is None:
        return modality
    return set(s for s in enabled_sources if s in modality)


def _cell(row, column):
    """
    Read a sheet cell as a clean string.

    :param row: Pandas Series (a Slide Chunks row)
    :param column: Column name
    :return: Stripped string value ("" when missing/nan)
    """
    value = str(row.get(column, "")).strip()
    if value == "nan":
        return ""
    return value


def get_image_candidates(row, segment_num, enabled_sources=None):
    """
    Read normalized image candidates for one segment across enabled image sources.

    :param row: Pandas Series (a Slide Chunks row)
    :param segment_num: 1-based segment number
    :param enabled_sources: Optional list of enabled source ids (None reads all present)
    :return: Deduped list of dicts with title, url, source_id
    """
    from agents.graphics_definition_v2.image_graphics_agent.image_selection_from_all_images import (
        parse_urls_from_results,
    )
    from agents.graphics_definition_v2.external_references.external_ref_search_from_queries import (
        is_external_ref_video_url,
    )

    include = _include_set(enabled_sources, IMAGE_SOURCE_COLUMNS)
    seen = set()
    out = []
    # Drive images, then web, then external refs (dedupe by URL across sources).
    for source_id in (SOURCE_DRIVE_IMAGES, SOURCE_WEB_IMAGES, SOURCE_EXTERNAL_REFERENCES):
        if source_id not in include:
            continue
        text = _cell(row, IMAGE_SOURCE_COLUMNS[source_id])
        for item in parse_urls_from_results(text, segment_num):
            url = item.get("url", "")
            if not url or url in seen:
                continue
            # external_ref_pool mixes modalities — skip video clip lines for image selection.
            if source_id == SOURCE_EXTERNAL_REFERENCES and is_external_ref_video_url(url):
                continue
            seen.add(url)
            out.append({
                "title": item.get("title", "Untitled"),
                "url": url,
                "source_id": source_id,
            })
    return out


def get_video_candidates(row, segment_num, enabled_sources=None):
    """
    Read normalized video candidates for one segment across enabled video sources.

    Items keep a legacy "type" tag (pool/other/drive/external_frames) so existing batching and multimodal-label code needs no change. External-ref YouTube and Drive clips use type "external_frames": load with start/end window, but frames-only selection permission.

    :param row: Pandas Series (a Slide Chunks row)
    :param segment_num: 1-based segment number
    :param enabled_sources: Optional list of enabled source ids (None reads all present)
    :return: List of dicts with type, url, meta, title, source_id
    """
    from agents.graphics_definition_v2.aggregation_agent.aggregation_agent import (
        parse_urls_from_video_pool,
    )
    from agents.graphics_definition_v2.video_graphics_agent.video_selection_from_all_videos import (
        parse_video_items_from_pool_other_channels,
        parse_drive_video_items_from_pool,
    )
    from agents.graphics_definition_v2.image_graphics_agent.image_selection_from_all_images import (
        parse_urls_from_results,
    )
    from agents.graphics_definition_v2.external_references.external_ref_search_from_queries import (
        is_external_ref_video_url,
    )

    include = _include_set(enabled_sources, VIDEO_SOURCE_COLUMNS)
    out = []

    if SOURCE_HVAC_YOUTUBE in include:
        text = _cell(row, VIDEO_SOURCE_COLUMNS[SOURCE_HVAC_YOUTUBE])
        for url in parse_urls_from_video_pool(text, segment_num):
            if url:
                out.append({
                    "type": _VIDEO_SOURCE_TYPE[SOURCE_HVAC_YOUTUBE],
                    "url": url,
                    "meta": None,
                    "title": "",
                    "source_id": SOURCE_HVAC_YOUTUBE,
                })

    if SOURCE_YOUTUBE_OTHER_CHANNELS in include:
        text = _cell(row, VIDEO_SOURCE_COLUMNS[SOURCE_YOUTUBE_OTHER_CHANNELS])
        for item in parse_video_items_from_pool_other_channels(text, segment_num):
            url = item.get("url", "")
            if url:
                out.append({
                    "type": _VIDEO_SOURCE_TYPE[SOURCE_YOUTUBE_OTHER_CHANNELS],
                    "url": url,
                    "meta": item,
                    "title": item.get("title", ""),
                    "source_id": SOURCE_YOUTUBE_OTHER_CHANNELS,
                })

    if SOURCE_DRIVE_VIDEOS in include:
        text = _cell(row, VIDEO_SOURCE_COLUMNS[SOURCE_DRIVE_VIDEOS])
        for item in parse_drive_video_items_from_pool(text, segment_num):
            url = item.get("url", "")
            if url:
                out.append({
                    "type": _VIDEO_SOURCE_TYPE[SOURCE_DRIVE_VIDEOS],
                    "url": url,
                    "meta": item,
                    "title": item.get("title", ""),
                    "source_id": SOURCE_DRIVE_VIDEOS,
                })

    if SOURCE_EXTERNAL_REFERENCES in include:
        text = _cell(row, VIDEO_SOURCE_COLUMNS[SOURCE_EXTERNAL_REFERENCES])
        for item in parse_urls_from_results(text, segment_num):
            url = item.get("url", "")
            if not url or not is_external_ref_video_url(url):
                continue
            title = item.get("title", "") or "External ref video"
            meta = {"title": title, "url": url}
            out.append({
                "type": _VIDEO_SOURCE_TYPE[SOURCE_EXTERNAL_REFERENCES],
                "url": url,
                "meta": meta,
                "title": title,
                "source_id": SOURCE_EXTERNAL_REFERENCES,
            })

    return out


def split_video_candidates(items):
    """
    Split normalized video candidates into legacy buckets plus external-ref frames-only items.

    :param items: List of items from get_video_candidates
    :return: Tuple (pool_urls, other_items, drive_items, external_frames_items)
    """
    pool_urls = [x["url"] for x in items if x.get("type") == "pool"]
    other_items = [x["meta"] for x in items if x.get("type") == "other" and x.get("meta")]
    drive_items = [x["meta"] for x in items if x.get("type") == "drive" and x.get("meta")]
    external_frames_items = [
        x["meta"] for x in items if x.get("type") == "external_frames" and x.get("meta")
    ]
    return pool_urls, other_items, drive_items, external_frames_items


def image_row_has_candidates(row, enabled_sources=None):
    """
    Report whether any enabled image source has content on this row.

    :param row: Pandas Series (a Slide Chunks row)
    :param enabled_sources: Optional list of enabled source ids (None checks all present)
    :return: True if at least one enabled image column is non-empty
    """
    include = _include_set(enabled_sources, IMAGE_SOURCE_COLUMNS)
    return any(_cell(row, IMAGE_SOURCE_COLUMNS[s]) for s in include)


def video_row_has_candidates(row, enabled_sources=None):
    """
    Report whether any enabled video source has content on this row.

    :param row: Pandas Series (a Slide Chunks row)
    :param enabled_sources: Optional list of enabled source ids (None checks all present)
    :return: True if at least one enabled video column is non-empty
    """
    include = _include_set(enabled_sources, VIDEO_SOURCE_COLUMNS)
    return any(_cell(row, VIDEO_SOURCE_COLUMNS[s]) for s in include)
