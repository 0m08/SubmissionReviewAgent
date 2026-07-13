"""Edge-TTS narration for the review player (same voice as slideshow_video)."""

from __future__ import annotations

import hashlib
import os
import tempfile
from typing import Dict, Tuple

from agents.graphics_definition_v2.slideshow_manifest.slideshow_video import synthesize_narration

# Same voice as pptx_exporter / slideshow_streamlit player.
PLAYER_VOICE = "en-CA-LiamNeural"

_TTS_BYTES_CACHE: Dict[str, bytes] = {}


def synthesize_tts_mp3_bytes(text: str, *, voice: str = PLAYER_VOICE) -> Tuple[bytes, float]:
    """Return (mp3 bytes, duration seconds) for narration text."""
    normalized = (text or "").strip()
    if not normalized:
        raise ValueError("voiceover text is required")

    cache_key = hashlib.sha256((voice + "\n" + normalized).encode("utf-8")).hexdigest()
    if cache_key in _TTS_BYTES_CACHE:
        # Duration not cached separately; clients use audio element duration.
        return _TTS_BYTES_CACHE[cache_key], 0.0

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        out_path = tmp.name

    try:
        audio_path, duration = synthesize_narration(normalized, out_path, voice=voice)
        if not audio_path or not os.path.exists(audio_path):
            raise RuntimeError("TTS synthesis failed")
        with open(audio_path, "rb") as fh:
            data = fh.read()
        if len(_TTS_BYTES_CACHE) > 500:
            _TTS_BYTES_CACHE.clear()
        _TTS_BYTES_CACHE[cache_key] = data
        return data, float(duration or 0.0)
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass
