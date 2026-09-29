import os
import re
import json
import threading
from typing import Optional, Tuple, Dict, Any
from PIL import Image
import pandas as pd

from agents.graphics_asset_creation.automated.automated_voiceover_reviewer import (
    review_and_edit_image,
    _download_drive_image,
    _save_image_to_drive,
    _create_drive_subfolder,
)
from agents.graphics_asset_creation.automated.reference_image_pipeline import _extract_best_voiceover
from agents.graphics_asset_creation.gac_utils import (
    get_or_create_drive_folder,
    is_valid_web_image_link,
)

def process_reference_image_path(
    df: pd.DataFrame,
    df_idx: int,
    ref_image_url: str,
    drive,
    output_folder_id: str,
    drive_lock: threading.Lock,
    llm: str = "gemini_3_flash_thinking",
    voiceover_override: Optional[str] = None,
    skip_technical_accuracy: bool = True,
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Executes the Reference Image Pipeline logic for a single row.

    :param voiceover_override: When provided (e.g., called from Step 5 Aggregation),
        the full slide chunk is used directly as the voiceover anchor, skipping the
        context-window _extract_best_voiceover step.

    Returns:
        (is_relevant, final_graphics_definition_text, metadata_dict)
    """
    # 1. Basic Validation
    plain_link = re.sub(r'\s*\(snapshot\)\s*', '', ref_image_url, flags=re.IGNORECASE).strip()
    link_is_valid = is_valid_web_image_link(plain_link)
    if not link_is_valid:
        if any(domain in plain_link.lower() for domain in ['drive.google.com', 'docs.google.com']):
            link_is_valid = True

    if not link_is_valid:
        return False, None, None

    # 2. Download the Reference Image (always required)
    try:
        with drive_lock:
            ref_image = _download_drive_image(plain_link, drive=drive)
        if not ref_image:
            return False, None, None
    except Exception:
        return False, None, None

    if voiceover_override is not None:
        # ── OVERRIDE PATH (called from Step 5 / Aggregation) ────────────────
        # Use the provided full slide chunk directly as the voiceover anchor.
        # Skip context-window VO extraction entirely.
        actual_row = df.iloc[df_idx]
        relevant_source_chunk = voiceover_override
        actual_slide_title = str(actual_row.get("Slide Chunk Title", "")).strip()
        actual_topic = str(actual_row.get("Topic", "")).strip()
        extracted_vo_text = voiceover_override
    else:
        # ── CONTEXT-WINDOW PATH (legacy / classic call) ──────────────────────
        # Build sliding-window context (2 above, 2 below) and ask the LLM which
        # chunk the reference image belongs to.
        col_slide_chunk = "Slide Chunk"
        context_list = []

        for offset in range(-2, 3):
            neighbor_idx = df_idx + offset
            if 0 <= neighbor_idx < len(df):
                n_row = df.iloc[neighbor_idx]
                n_chunk = str(n_row.get(col_slide_chunk, "")).strip()
                if n_chunk:
                    marker = f"CHUNK_ID_{neighbor_idx+1}"
                    context_list.append(f"<{marker}>\n{n_chunk}\n</{marker}>")

        context_chunks_str = "\n\n---\n\n".join(context_list)

        # Relevance Check & VO Extraction
        extracted_vo_raw = _extract_best_voiceover(
            ref_image, context_chunks_str, row_idx=df_idx + 1
        )

        if extracted_vo_raw.strip().upper() in ["NOT_RELEVANT", "IRRELEVANT IMAGE"]:
            return False, None, None

        # Parse CHUNK_ID and text
        source_id = None
        if ": " in extracted_vo_raw and "CHUNK_ID_" in extracted_vo_raw:
            source_id, _ = extracted_vo_raw.split(": ", 1)
            source_id = source_id.strip()

        # Determine the source chunk and metadata
        actual_idx = df_idx
        if source_id and source_id.startswith("CHUNK_ID_"):
            try:
                actual_idx = int(source_id.replace("CHUNK_ID_", "")) - 1
            except ValueError:
                actual_idx = df_idx

        if actual_idx < 0 or actual_idx >= len(df):
            actual_idx = df_idx

        actual_row = df.iloc[actual_idx]
        relevant_source_chunk = str(actual_row.get(col_slide_chunk, "")).strip()
        actual_slide_title = str(actual_row.get("Slide Chunk Title", "")).strip()
        actual_topic = str(actual_row.get("Topic", "")).strip()
        extracted_vo_text = extracted_vo_raw.split(": ", 1)[1] if ": " in extracted_vo_raw else extracted_vo_raw

    # 3. Run Accuracy + Copyright Pipeline
    folder_name = f"RefPath_Row_{df_idx + 1}_{actual_slide_title[:30].replace('/', '_')}"
    row_folder_id = _create_drive_subfolder(
        drive, output_folder_id, folder_name, drive_lock=drive_lock
    )

    _, final_image, _ = review_and_edit_image(
        reference_image     = ref_image,
        slide_title         = actual_slide_title,
        slide_content       = relevant_source_chunk,
        voiceover           = extracted_vo_text,
        visual_instruction  = "",
        target_stage        = "full",
        callback            = None,
        skip_accuracy_validation = skip_technical_accuracy,
    )

    if not final_image:
        return False, None, None

    # 4. Upload Final Image to Drive
    final_drive_link = _save_image_to_drive(
        final_image, "FINAL_RefImage.png", drive, row_folder_id, drive_lock=drive_lock
    )

    if not final_drive_link:
        return False, None, None

    # 5. Format Output
    final_graphics_definition = f"When VO: {relevant_source_chunk}\nGraphics to use: {final_drive_link}"

    return True, final_graphics_definition, {
        "voiceover_segment": relevant_source_chunk,
        "Slide Chunk Title": actual_slide_title,
        "Topic": actual_topic
    }
