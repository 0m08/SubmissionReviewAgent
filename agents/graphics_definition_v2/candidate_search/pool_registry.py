"""
Enabled-source ids for Graphics Definition V2 candidate search.

Maps the legacy use_only_drive_and_hvac boolean onto an explicit source list.
UI asset-library checkboxes live in agent_ui_template; this module holds the ids.
"""

import re

SOURCE_DRIVE_IMAGES = "drive_images"
SOURCE_HVAC_YOUTUBE = "hvac_youtube"
SOURCE_DRIVE_VIDEOS = "drive_videos"
SOURCE_EXTERNAL_REFERENCES = "external_references"
SOURCE_WEB_IMAGES = "web_images"
SOURCE_YOUTUBE_OTHER_CHANNELS = "youtube_other_channels"

# Primary libraries searched in Section 5 (candidate finding).
DEFAULT_ENABLED_SOURCES = [
    SOURCE_DRIVE_IMAGES,
    SOURCE_HVAC_YOUTUBE,
    SOURCE_DRIVE_VIDEOS,
]

# Web / other-YouTube are not Section 5 defaults; they power Section 9 fallback when enabled in UI.
OPTIONAL_SOURCES = [
    SOURCE_WEB_IMAGES,
    SOURCE_YOUTUBE_OTHER_CHANNELS,
]

ALL_KNOWN_SOURCES = DEFAULT_ENABLED_SOURCES + [
    SOURCE_EXTERNAL_REFERENCES,
] + OPTIONAL_SOURCES

# Session-state keys for the Graphics Definition V2 asset-library UI.
UI_KEY_DRIVE_IMAGES = "graphics_v2_asset_drive_images"
UI_KEY_HVAC_YOUTUBE = "graphics_v2_asset_hvac_youtube"
UI_KEY_WEB_AND_OTHER = "graphics_v2_asset_web_and_other_youtube"
UI_KEY_DRIVE_VIDEOS = "graphics_v2_asset_drive_videos"
UI_KEY_EXTERNAL_REFERENCES = "graphics_v2_asset_external_references"
UI_KEY_DRIVE_VIDEO_MODE = "graphics_v2_drive_video_mode"
UI_KEY_WEB_FALLBACK_ENABLED = "graphics_v2_web_fallback_enabled"
UI_KEY_ENABLED_SOURCES = "graphics_v2_enabled_sources"

DRIVE_VIDEO_MODE_ALL = "all"
DRIVE_VIDEO_MODE_NEXTECH = "nextech"

# Course info multi-select column (source of truth for Graphics UI + Human Feedback).
COURSE_INFO_ENABLED_SOURCES_KEY = "Allowed Asset Search Libraries for the Graphics Agent"

# Human-readable multi-select option labels (stored in the Course info cell).
LABEL_DRIVE_IMAGES = "Drive Images"
LABEL_HVAC_YOUTUBE = "HVAC YouTube Videos"
LABEL_DRIVE_VIDEOS_ALL = "Google Drive Videos (All)"
LABEL_DRIVE_VIDEOS_NEXTECH = "Google Drive Videos (NexTech Only)"
LABEL_WEB = "Web Images and Videos"
LABEL_EXTERNAL = "External References"

COURSE_INFO_ASSET_LIBRARY_OPTIONS = [
    LABEL_DRIVE_IMAGES,
    LABEL_HVAC_YOUTUBE,
    LABEL_DRIVE_VIDEOS_ALL,
    LABEL_DRIVE_VIDEOS_NEXTECH,
    LABEL_WEB,
    LABEL_EXTERNAL,
]


def default_enabled_sources_for_course_info():
    """
    Product defaults for a blank Course info asset-libraries cell.

    Drive Images, HVAC YouTube, Google Drive Videos (All), and Web are on.
    External References is off until selected.

    :return: Tuple of (enabled_sources list, drive_video_mode)
    """
    return (
        [
            SOURCE_DRIVE_IMAGES,
            SOURCE_HVAC_YOUTUBE,
            SOURCE_DRIVE_VIDEOS,
            SOURCE_WEB_IMAGES,
            SOURCE_YOUTUBE_OTHER_CHANNELS,
        ],
        DRIVE_VIDEO_MODE_ALL,
    )


def encode_enabled_sources_for_course_info(sources, drive_video_mode=""):
    """
    Encode enabled sources (+ Drive video mode) as Course info multi-select labels.

    :param sources: List of enabled source ids.
    :param drive_video_mode: "all" / "nextech" when Drive videos are enabled, else "".
    :return: Comma-separated label string for the Course info cell.
    """
    source_set = {str(s).strip() for s in (sources or []) if str(s).strip()}
    labels = []
    if SOURCE_DRIVE_IMAGES in source_set:
        labels.append(LABEL_DRIVE_IMAGES)
    if SOURCE_HVAC_YOUTUBE in source_set:
        labels.append(LABEL_HVAC_YOUTUBE)
    if SOURCE_DRIVE_VIDEOS in source_set:
        mode = (drive_video_mode or DRIVE_VIDEO_MODE_ALL).strip().lower()
        if mode == DRIVE_VIDEO_MODE_NEXTECH:
            labels.append(LABEL_DRIVE_VIDEOS_NEXTECH)
        else:
            labels.append(LABEL_DRIVE_VIDEOS_ALL)
    if SOURCE_WEB_IMAGES in source_set or SOURCE_YOUTUBE_OTHER_CHANNELS in source_set:
        labels.append(LABEL_WEB)
    if SOURCE_EXTERNAL_REFERENCES in source_set:
        labels.append(LABEL_EXTERNAL)
    # Sheets multi-select dropdowns store selections as comma-separated (not newlines).
    return ", ".join(labels)


def decode_enabled_sources_from_course_info(raw):
    """
    Decode a Course info asset-libraries cell into (enabled_sources, drive_video_mode)..

    :param raw: Cell value from Course info.
    :return: Tuple (list of source ids, mode string).
    """
    text = (raw or "").strip()
    if not text:
        return default_enabled_sources_for_course_info()

    tokens = [t.strip() for t in re.split(r"[\n,]+", text) if t.strip()]
    labels_lower = {t.lower() for t in tokens}

    def _has(label):
        return label.lower() in labels_lower

    sources = []
    if _has(LABEL_DRIVE_IMAGES):
        sources.append(SOURCE_DRIVE_IMAGES)
    if _has(LABEL_HVAC_YOUTUBE):
        sources.append(SOURCE_HVAC_YOUTUBE)

    has_all = _has(LABEL_DRIVE_VIDEOS_ALL)
    has_nextech = _has(LABEL_DRIVE_VIDEOS_NEXTECH)
    mode = ""
    if has_all or has_nextech:
        sources.append(SOURCE_DRIVE_VIDEOS)
        # Both selected → All (broader set; NexTech is included anyway).
        mode = DRIVE_VIDEO_MODE_ALL if has_all else DRIVE_VIDEO_MODE_NEXTECH

    if _has(LABEL_WEB):
        sources.append(SOURCE_WEB_IMAGES)
        sources.append(SOURCE_YOUTUBE_OTHER_CHANNELS)
    if _has(LABEL_EXTERNAL):
        sources.append(SOURCE_EXTERNAL_REFERENCES)

    if not sources:
        return default_enabled_sources_for_course_info()
    return sources, mode


SOURCE_DISPLAY_NAMES = {
    SOURCE_DRIVE_IMAGES: "Drive Search",
    SOURCE_HVAC_YOUTUBE: "Video Search (HVAC Channels)",
    SOURCE_DRIVE_VIDEOS: "Video Search (Google Drive)",
    SOURCE_EXTERNAL_REFERENCES: "External References",
    SOURCE_WEB_IMAGES: "Web Search",
    SOURCE_YOUTUBE_OTHER_CHANNELS: "Video Search (Other Channels)",
}

UI_ASSET_LABELS = {
    UI_KEY_DRIVE_IMAGES: "Drive Images",
    UI_KEY_HVAC_YOUTUBE: "HVAC YouTube Videos",
    UI_KEY_WEB_AND_OTHER: "Web Images and Videos",
    UI_KEY_DRIVE_VIDEOS: "Google Drive Videos",
    UI_KEY_EXTERNAL_REFERENCES: "External References",
}


def resolve_enabled_sources(enabled_sources=None, use_only_drive_and_hvac=None):
    """
    Resolve which candidate pools to run.

    Priority:
      1. Explicit enabled_sources if provided (non-None)
      2. Else use_only_drive_and_hvac:
         - True  -> drive_images + hvac_youtube + drive_videos
         - False -> all known sources
         - None  -> default primary pools (no external references)

    :param enabled_sources: Optional explicit list of source ids
    :param use_only_drive_and_hvac: Legacy boolean mapped to enabled sources
    :return: List of enabled source id strings
    """
    if enabled_sources is not None:
        resolved = []
        seen = set()
        for source_id in enabled_sources:
            sid = str(source_id).strip()
            if not sid or sid in seen:
                continue
            if sid not in ALL_KNOWN_SOURCES:
                print(f"⚠️ Unknown candidate source id ignored: {sid}")
                continue
            seen.add(sid)
            resolved.append(sid)
        return resolved

    if use_only_drive_and_hvac is False:
        return list(ALL_KNOWN_SOURCES)

    # True or None -> default internal pools only (External Refs opt-in)
    return list(DEFAULT_ENABLED_SOURCES)


def apply_enabled_sources_to_session_state(sources, drive_video_mode, session_state):
    """
    Mirror Course info selection onto Graphics V2 UI session-state checkbox keys.

    :param sources: List of enabled source ids
    :param drive_video_mode: "all" / "nextech" / ""
    :param session_state: Streamlit session_state mapping
    :return: None
    """
    source_set = {str(s).strip() for s in (sources or []) if str(s).strip()}
    session_state[UI_KEY_DRIVE_IMAGES] = SOURCE_DRIVE_IMAGES in source_set
    session_state[UI_KEY_HVAC_YOUTUBE] = SOURCE_HVAC_YOUTUBE in source_set
    session_state[UI_KEY_DRIVE_VIDEOS] = SOURCE_DRIVE_VIDEOS in source_set
    session_state[UI_KEY_EXTERNAL_REFERENCES] = SOURCE_EXTERNAL_REFERENCES in source_set
    session_state[UI_KEY_WEB_AND_OTHER] = (
        SOURCE_WEB_IMAGES in source_set or SOURCE_YOUTUBE_OTHER_CHANNELS in source_set
    )
    mode = (drive_video_mode or DRIVE_VIDEO_MODE_ALL).strip().lower()
    if mode not in (DRIVE_VIDEO_MODE_ALL, DRIVE_VIDEO_MODE_NEXTECH):
        mode = DRIVE_VIDEO_MODE_ALL
    session_state[UI_KEY_DRIVE_VIDEO_MODE] = mode
    session_state["graphics_v2_drive_video_mode_label"] = (
        "Only NexTech videos" if mode == DRIVE_VIDEO_MODE_NEXTECH else "All Google Drive videos"
    )


def enabled_sources_from_session_state(session_state):
    """
    Build (sources, drive_video_mode) from Graphics V2 UI checkbox session keys.

    :param session_state: Streamlit session_state mapping
    :return: Tuple of (enabled_sources list, drive_video_mode)
    """
    sources = []
    if session_state.get(UI_KEY_DRIVE_IMAGES, True):
        sources.append(SOURCE_DRIVE_IMAGES)
    if session_state.get(UI_KEY_HVAC_YOUTUBE, True):
        sources.append(SOURCE_HVAC_YOUTUBE)
    if session_state.get(UI_KEY_DRIVE_VIDEOS, True):
        sources.append(SOURCE_DRIVE_VIDEOS)
    if session_state.get(UI_KEY_EXTERNAL_REFERENCES, False):
        sources.append(SOURCE_EXTERNAL_REFERENCES)
    if session_state.get(UI_KEY_WEB_AND_OTHER, True):
        sources.append(SOURCE_WEB_IMAGES)
        sources.append(SOURCE_YOUTUBE_OTHER_CHANNELS)
    mode = ""
    if SOURCE_DRIVE_VIDEOS in sources:
        mode = str(session_state.get(UI_KEY_DRIVE_VIDEO_MODE, DRIVE_VIDEO_MODE_ALL) or DRIVE_VIDEO_MODE_ALL)
    return sources, mode
