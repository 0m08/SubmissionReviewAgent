"""Edge-TTS narration for the review player (same voice as slideshow_video)."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import threading
from typing import Any, Dict, List, Optional, Tuple

import edge_tts

# Same voice as pptx_exporter / slideshow_streamlit player.
PLAYER_VOICE = "en-CA-LiamNeural"

# edge-tts offsets are 100-nanosecond ticks.
_TICKS_PER_SECOND = 10_000_000.0

_TTS_CACHE: Dict[str, Tuple[bytes, List[Dict[str, Any]]]] = {}
_TTS_LOCK = threading.Lock()
_TTS_INFLIGHT: Dict[str, threading.Event] = {}


def _cache_key(text: str, voice: str) -> str:
    return hashlib.sha256((voice + "\n" + text).encode("utf-8")).hexdigest()


async def _synthesize_with_words(text: str, voice: str) -> Tuple[bytes, List[Dict[str, Any]]]:
    communicate = edge_tts.Communicate(text, voice, boundary="WordBoundary")
    audio = bytearray()
    words: List[Dict[str, Any]] = []
    async for chunk in communicate.stream():
        kind = chunk.get("type")
        if kind == "audio":
            audio.extend(chunk.get("data") or b"")
        elif kind == "WordBoundary":
            words.append({
                "text": str(chunk.get("text") or "").strip(),
                "offset": float(chunk.get("offset") or 0) / _TICKS_PER_SECOND,
                "duration": float(chunk.get("duration") or 0) / _TICKS_PER_SECOND,
            })
    if not audio:
        raise RuntimeError("TTS synthesis returned no audio")
    return bytes(audio), words


def synthesize_tts_with_words(text: str, *, voice: str = PLAYER_VOICE) -> Tuple[bytes, List[Dict[str, Any]]]:
    """Return (mp3 bytes, word timings in seconds) for narration text."""
    normalized = (text or "").strip()
    if not normalized:
        raise ValueError("voiceover text is required")

    key = _cache_key(normalized, voice)
    owner = False
    wait_ev: Optional[threading.Event] = None

    with _TTS_LOCK:
        cached = _TTS_CACHE.get(key)
        if cached:
            return cached
        wait_ev = _TTS_INFLIGHT.get(key)
        if wait_ev is None:
            wait_ev = threading.Event()
            _TTS_INFLIGHT[key] = wait_ev
            owner = True

    if not owner:
        wait_ev.wait(timeout=120)
        with _TTS_LOCK:
            cached = _TTS_CACHE.get(key)
        if cached:
            return cached
        raise RuntimeError("TTS synthesis failed")

    try:
        loop = asyncio.new_event_loop()
        try:
            data, words = loop.run_until_complete(_synthesize_with_words(normalized, voice))
        finally:
            loop.close()
        with _TTS_LOCK:
            if len(_TTS_CACHE) > 500:
                _TTS_CACHE.clear()
            _TTS_CACHE[key] = (data, words)
        return data, words
    finally:
        with _TTS_LOCK:
            _TTS_INFLIGHT.pop(key, None)
        wait_ev.set()


def encode_tts_words_header(words: List[Dict[str, Any]]) -> Optional[str]:
    """Compact wire form for X-TTS-Words. None if too large for a header."""
    if not words:
        return None
    raw = json.dumps(words, separators=(",", ":")).encode("utf-8")
    encoded = base64.b64encode(raw).decode("ascii")
    if len(encoded) > 7000:
        return None
    return encoded


def synthesize_tts_mp3_bytes(text: str, *, voice: str = PLAYER_VOICE) -> Tuple[bytes, float]:
    """Return (mp3 bytes, duration seconds) for narration text."""
    data, words = synthesize_tts_with_words(text, voice=voice)
    duration = 0.0
    if words:
        last = words[-1]
        duration = float(last["offset"]) + float(last["duration"])
    return data, duration
