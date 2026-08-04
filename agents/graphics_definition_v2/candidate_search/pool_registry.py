"""
Enabled-source ids for Graphics Definition V2 candidate search.

Maps the legacy use_only_drive_and_hvac boolean onto an explicit source list.
UI asset-library checkboxes live in agent_ui_template; this module holds the ids.
"""

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
    SOURCE_EXTERNAL_REFERENCES,
]

# Web / other-YouTube are not Section 5 defaults; they power Section 9 fallback when enabled in UI.
OPTIONAL_SOURCES = [
    SOURCE_WEB_IMAGES,
    SOURCE_YOUTUBE_OTHER_CHANNELS,
]

ALL_KNOWN_SOURCES = DEFAULT_ENABLED_SOURCES + OPTIONAL_SOURCES

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

# Course info header used to persist a run's enabled sources (+ Drive video mode) so that the human-feedback review app can respect the same libraries.
COURSE_INFO_ENABLED_SOURCES_KEY = "graphics_v2_enabled_sources"


def encode_enabled_sources_for_course_info(sources, drive_video_mode=""):
    """
    Encode enabled sources (+ optional Drive video mode) into one Course info cell.

    :param sources: List of enabled source ids.
    :param drive_video_mode: "all" / "nextech" when Drive videos are enabled, else "".
    :return: Cell value string.
    """
    sources = [str(s).strip() for s in (sources or []) if str(s).strip()]
    value = ",".join(sources)
    mode = (drive_video_mode or "").strip()
    if mode and SOURCE_DRIVE_VIDEOS in sources:
        value = f"{value}|{mode}"
    return value


def decode_enabled_sources_from_course_info(raw):
    """
    Decode a Course info cell into (enabled_sources list, drive_video_mode).

    :param raw: Cell value from Course info.
    :return: Tuple (list_or_None, mode_string). list is None when raw is empty.
    """
    text = (raw or "").strip()
    if not text:
        return None, ""
    sources_part, _, mode_part = text.partition("|")
    sources = [s.strip() for s in sources_part.split(",") if s.strip()]
    mode = mode_part.strip()
    return (sources or None), mode

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
    UI_KEY_HVAC_YOUTUBE: "HVAC School YouTube Videos",
    UI_KEY_WEB_AND_OTHER: "Web Images and Other YouTube Channel Videos",
    UI_KEY_DRIVE_VIDEOS: "Google Drive Videos",
    UI_KEY_EXTERNAL_REFERENCES: "External References",
}


def resolve_enabled_sources(enabled_sources=None, use_only_drive_and_hvac=None):
    """
    Resolve which candidate pools to run.

    Priority:
      1. Explicit enabled_sources if provided (non-None)
      2. Else use_only_drive_and_hvac:
         - True  -> drive_images + hvac_youtube + drive_videos + external_references
         - False -> all known sources
         - None  -> default primary pools (includes external_references)

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

    # True or None -> default internal pools only
    return list(DEFAULT_ENABLED_SOURCES)
