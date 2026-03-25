"""
Agent 2 — Technical Accuracy Validator (v2)
============================================
Key improvements over v1:
  1. Significance filter  — runs immediately after detection, drops trivial
     cosmetic diffs (lighting, compression artefacts, minor contrast) before
     they enter the 4-round reasoning loop. Only domain-material differences
     proceed.
  2. Severity ranking     — surviving differences are ranked CRITICAL / MAJOR
     / MINOR so corrections are prioritised correctly.
  3. Batched corrections  — image-gen calls are capped at MAX_FIXES_PER_CALL
     corrections each. Multiple passes are made if needed, each building on
     the previous result, so the model is never overwhelmed.
  4. Correction verifier  — after each batch the corrected image is re-checked
     against the specific fixes that were requested; if a fix is missing it is
     retried once before moving on.

Run standalone:
    streamlit run agent2_validator.py

Import into another file:
    from agent2_validator import run_agent2, render_agent2_page, AgentResult
"""

from __future__ import annotations

import base64
import importlib
import io
import json
import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

import requests
import streamlit as st
from PIL import Image

try:
    genai_legacy = importlib.import_module("google.generativeai")
except Exception:
    genai_legacy = None

try:
    genai_new = importlib.import_module("google.genai")
    genai_new_types = genai_new.types
except Exception:
    genai_new = None
    genai_new_types = None

log = logging.getLogger("Agent2")

# ─────────────────────────────────────────────────────────────────────────────
# SDK bootstrap
# ─────────────────────────────────────────────────────────────────────────────

_SDK_MODE: str | None = None
_NEW_CLIENT = None


def _resolve_api_key() -> str:
    return (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or st.session_state.get("gemini_api_key", "")
        or st.session_state.get("google_api_key", "")
        or ""
    ).strip()


def _ensure_sdk_ready() -> None:
    global _SDK_MODE, _NEW_CLIENT
    if _SDK_MODE:
        return
    api_key = _resolve_api_key()
    if not api_key:
        raise ValueError("Missing API key — set GEMINI_API_KEY or enter it in the UI.")
    if genai_legacy is not None:
        genai_legacy.configure(api_key=api_key)
        _SDK_MODE = "legacy"
        return
    if genai_new is not None:
        _NEW_CLIENT = genai_new.Client(api_key=api_key)
        _SDK_MODE = "new"
        return
    raise ImportError("No Gemini SDK found. Install google-generativeai or google-genai.")


ANALYSIS_MODEL  = "gemini-3-flash-preview"
IMAGE_GEN_MODEL = "gemini-3-pro-image-preview"

N_REASONING_ROUNDS    = 4
CONFIDENCE_THRESHOLD  = 0.72
MAX_CORRECTION_RETRIES = 2
MAX_FIXES_PER_CALL    = 3      # image-gen call cap; batches if more needed
REQUEST_DELAY_S       = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

class Verdict(str, Enum):
    CREATIVE_LIBERTY = "CREATIVE_LIBERTY"
    INACCURACY       = "TECHNICAL_INACCURACY"
    UNCERTAIN        = "UNCERTAIN"


class Severity(str, Enum):
    CRITICAL = "CRITICAL"   # safety codes, electrical colours, refrigerant labels
    MAJOR    = "MAJOR"      # valve states, gauge readings, flow arrows
    MINOR    = "MINOR"      # nameplate text, minor label changes
    TRIVIAL  = "TRIVIAL"    # filtered before reasoning rounds


@dataclass
class Difference:
    diff_id:                str
    description:            str
    location_hint:          str
    significance:           Severity  = Severity.TRIVIAL
    initial_verdict:        Verdict   = Verdict.UNCERTAIN
    final_verdict:          Verdict   = Verdict.UNCERTAIN
    confidence:             float     = 0.0
    hvac_rule_cited:        str       = ""
    correction_instruction: str       = ""
    correction_verified:    bool      = False
    reasoning_trace:        list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    reference_image_path:         str
    output_image_path:            str
    corrected_image_path:         str | None
    differences:                  list[Difference]
    trivial_filtered:             int
    inaccuracies_found:           int
    inaccuracies_corrected:       int
    creative_liberties_preserved: int
    validation_passed:            bool
    audit_log:                    list[str]


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

_DIFF_DETECTOR_SYSTEM = """
You are a meticulous HVAC visual-difference detector with deep expertise in
HVAC systems, P&ID conventions, refrigerant circuits, electrical wiring, and
safety-critical labelling.

Your ONLY task: produce an exhaustive list of every visual difference between
the REFERENCE image and the OUTPUT image.

Rules
-----
1. Be exhaustive — missing a real difference is worse than listing a minor one.
2. Use precise, domain-specific HVAC terminology.
   BAD:  "The colour changed."
   GOOD: "The low-side service hose changed from blue to grey."
3. Note the spatial location of each difference.
4. Do NOT classify or filter yet.
5. Number differences starting at DIFF-001.

Output — strict JSON, no markdown fences:
{
  "differences": [
    {
      "diff_id": "DIFF-001",
      "description": "<precise HVAC-language description>",
      "location_hint": "<spatial location>"
    }
  ],
  "total_count": <integer>,
  "detection_confidence": "LOW|MEDIUM|HIGH"
}
"""

_SIGNIFICANCE_FILTER_SYSTEM = """
You are an HVAC image QA lead reviewing a list of visual differences detected
between a reference and an edited HVAC image.

Your job: decide which differences are DOMAIN-MATERIAL and which are TRIVIAL
noise that does not affect HVAC technical accuracy.

TRIVIAL — discard these (they must NOT enter the reasoning rounds):
  • Subtle brightness, contrast, or saturation shifts on non-coded surfaces
  • JPEG / PNG compression artefacts, slight blur, or rendering softness
  • Minor anti-aliasing or edge-smoothing differences
  • Negligible shadow or highlight variation on background elements
  • Stylistic rendering differences on non-functional surfaces
    (e.g. slight texture variation on unpainted sheet-metal cabinet panels)
  • Sub-pixel positional shifts with no semantic meaning
  • Minor scale or proportion changes under 5% that don't change function

DOMAIN-MATERIAL — must proceed to reasoning rounds. Assign severity:
  CRITICAL: anything involving safety-coded colours (refrigerant service hoses,
    electrical wires, refrigerant cylinder bodies), safety labels / placards,
    certification marks (UL/CE/CSA), refrigerant type identifiers
  MAJOR: valve open/closed state, gauge needle positions, flow-direction
    arrows on ducts or components, pipe insulation presence/absence,
    component orientation (TXV inlet/outlet, check valve direction)
  MINOR: nameplate text legibility, minor label wording changes, small
    dimensional proportion changes that could affect interpretation

Output strict JSON, no markdown fences:
{
  "material_differences": [
    {
      "diff_id": "DIFF-001",
      "description": "<original description>",
      "location_hint": "<original location>",
      "severity": "CRITICAL|MAJOR|MINOR",
      "reason_material": "<one sentence why this matters>"
    }
  ],
  "trivial_diff_ids": ["DIFF-003", "DIFF-005"],
  "trivial_count": <integer>,
  "material_count": <integer>
}
"""

_ROUND1_SYSTEM = """
You are a senior HVAC domain expert with thorough knowledge of ASHRAE standards
(15, 34, 90.1, 62.1), ARI/AHRI refrigerant and component standards, NEC and IEC
electrical wiring colour conventions, OSHA safety labelling requirements, and
standard P&ID and training-diagram conventions.

Task — Round 1 — Initial Classification
----------------------------------------
You receive only DOMAIN-MATERIAL differences (trivial items already removed).

Classify each as:
  CREATIVE_LIBERTY  — valid stylistic change, no domain rule violated
  TECHNICAL_INACCURACY — contradicts an HVAC standard, convention, or
    safety requirement

For each provide:
  • The specific rule, standard, or convention being applied.
  • A one-sentence justification.
  • Confidence 0.0–1.0.
  • Confirm or adjust the severity from the filter round.

Output strict JSON, no markdown fences:
{
  "classifications": [
    {
      "diff_id": "DIFF-001",
      "verdict": "TECHNICAL_INACCURACY",
      "rule_cited": "<specific standard / convention>",
      "justification": "<one sentence>",
      "confidence": 0.85,
      "severity": "CRITICAL"
    }
  ]
}
"""

_ROUND2_SYSTEM = """
You are a contrarian HVAC expert. Challenge every TECHNICAL_INACCURACY verdict.

For every TECHNICAL_INACCURACY:
  Q1. Could this be a valid stylistic choice that only superficially looks like
      an error?
  Q2. Is the cited rule genuinely violated, or is applying it here a stretch?
  Q3. Does the voiceover / scene context make this change acceptable?
  Q4. Is there ANY interpretation under which this is NOT an inaccuracy?

Downgrade to CREATIVE_LIBERTY only if you find a genuinely convincing argument.
Uphold if you cannot — after truly trying.

Also spot-check CREATIVE_LIBERTY items for quiet misclassifications.

Output strict JSON, no markdown fences:
{
  "revised_classifications": [
    {
      "diff_id": "DIFF-001",
      "previous_verdict": "TECHNICAL_INACCURACY",
      "revised_verdict": "TECHNICAL_INACCURACY",
      "devil_advocate_argument": "<best counter-argument>",
      "rebuttal": "<why it fails or succeeds>",
      "confidence": 0.91
    }
  ]
}
"""

_ROUND3_SYSTEM = """
You are the definitive HVAC domain authority — a standards committee member
with field experience and code expertise.

Make the FINAL classification for every difference.

Hard rules:
  • Safety-coded colours → ALMOST ALWAYS TECHNICAL_INACCURACY.
  • Valve state contradicting described flow / isolation → TECHNICAL_INACCURACY.
  • Removed / altered safety label / placard → TECHNICAL_INACCURACY.
  • For CREATIVE_LIBERTY: state explicitly "This does not affect any safety
    code or domain convention because ___."

For every TECHNICAL_INACCURACY, write a FOCUSED correction instruction:
  — One specific change only per instruction
  — State: what to change, the target appearance, where in the image
  — Keep it under 40 words so the image model can execute it precisely

Output strict JSON, no markdown fences:
{
  "definitive_classifications": [
    {
      "diff_id": "DIFF-001",
      "final_verdict": "TECHNICAL_INACCURACY",
      "hvac_rule_cited": "<standard + clause>",
      "definitive_justification": "<2-3 sentences>",
      "confidence": 0.94,
      "severity": "CRITICAL",
      "correction_instruction": "<focused, max 40-word instruction>"
    }
  ]
}
"""

_ROUND4_SYSTEM = f"""
You are a senior QA engineer making the final consolidated decision.

Rules
-----
  • Action = CORRECT only when final_verdict = TECHNICAL_INACCURACY
    AND confidence >= {CONFIDENCE_THRESHOLD}.
  • Sort CORRECT items by severity: CRITICAL first, then MAJOR, then MINOR.
  • UNCERTAIN or low-confidence → PRESERVE.
  • Write a concise audit summary.

Output strict JSON, no markdown fences:
{{
  "final_verdicts": [
    {{
      "diff_id": "DIFF-001",
      "action": "CORRECT",
      "reason": "<one sentence>",
      "correction_instruction": "<verbatim from round 3>",
      "confidence": 0.94,
      "severity": "CRITICAL"
    }},
    {{
      "diff_id": "DIFF-002",
      "action": "PRESERVE",
      "reason": "<one sentence>",
      "confidence": 0.88,
      "severity": "MINOR"
    }}
  ],
  "corrections_required": <integer>,
  "liberties_preserved": <integer>,
  "audit_summary": "<paragraph>"
}}
"""

_IMAGE_CORRECTION_SYSTEM = """
You are a precision HVAC technical image editor making TARGETED corrections.

You receive:
  1. The REFERENCE image — source of truth for correct appearance.
  2. The CURRENT image — apply the listed corrections to this.
  3. A SHORT numbered list of specific corrections (3 or fewer).

Rules
-----
  • Apply ONLY the listed corrections. Touch NOTHING else.
  • Each correction is a single focused change — do not infer additional fixes.
  • Match colours, valve positions, and labels exactly to the reference.
  • Keep all unchanged areas pixel-perfect.
  • Output a photorealistic, professional HVAC training image.
"""

_CORRECTION_VERIFIER_SYSTEM = """
You are an HVAC image QA verifier. Check whether specific corrections were
successfully applied to a corrected image.

For each correction check the corrected image and report:
  • applied: true if the correction is clearly visible in the corrected image
  • confidence: 0.0–1.0
  • note: brief observation (max 15 words)

Output strict JSON, no markdown fences:
{
  "verification": [
    {
      "diff_id": "DIFF-001",
      "applied": true,
      "confidence": 0.92,
      "note": "<brief observation>"
    }
  ],
  "all_applied": true
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# SDK helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_b64(source: str | Image.Image) -> tuple[str, str]:
    if isinstance(source, str):
        suffix = Path(source).suffix.lower()
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".webp": "image/webp"}.get(suffix, "image/png")
        return base64.b64encode(Path(source).read_bytes()).decode(), mime
    buf = io.BytesIO()
    source.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), "image/png"


def _img_part(source: str | Image.Image) -> dict:
    data, mime = _to_b64(source)
    return {"inline_data": {"mime_type": mime, "data": data}}


def _parts_to_new_sdk(parts: list) -> list:
    converted = []
    for part in parts:
        if isinstance(part, str):
            converted.append(genai_new_types.Part.from_text(text=part))
        elif isinstance(part, dict):
            inline = part.get("inline_data") or {}
            data = inline.get("data")
            mime = inline.get("mime_type", "image/png")
            if data:
                converted.append(genai_new_types.Part.from_bytes(
                    data=base64.b64decode(data), mime_type=mime))
    return converted


def _text_from_new_response(response) -> str:
    text = (getattr(response, "text", None) or "").strip()
    if text:
        return text
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return ""
    return "\n".join(
        str(p.text).strip()
        for p in (getattr(candidates[0].content, "parts", []) or [])
        if getattr(p, "text", None)
    ).strip()


def _parse_json(raw: str, ctx: str = "") -> dict:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    m = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
    if m:
        cleaned = m.group(1)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        log.warning("JSON parse [%s]: %s — %.200s", ctx, e, raw)
        return {}


def _text_call(system: str, parts: list, temperature: float = 0.2) -> str:
    _ensure_sdk_ready()
    time.sleep(REQUEST_DELAY_S)
    if _SDK_MODE == "legacy":
        model = genai_legacy.GenerativeModel(ANALYSIS_MODEL, system_instruction=system)
        cfg = genai_legacy.GenerationConfig(temperature=temperature)
        return model.generate_content(parts, generation_config=cfg).text
    resp = _NEW_CLIENT.models.generate_content(
        model=ANALYSIS_MODEL,
        contents=[genai_new_types.Content(role="user", parts=_parts_to_new_sdk(parts))],
        config=genai_new_types.GenerateContentConfig(
            system_instruction=system, temperature=temperature),
    )
    return _text_from_new_response(resp)


def _image_call(system: str, parts: list, temperature: float = 0.05) -> Image.Image | None:
    _ensure_sdk_ready()
    time.sleep(REQUEST_DELAY_S)
    if _SDK_MODE == "legacy":
        model = genai_legacy.GenerativeModel(IMAGE_GEN_MODEL, system_instruction=system)
        cfg = genai_legacy.GenerationConfig(temperature=temperature,
                                             response_modalities=["IMAGE", "TEXT"])
        resp = model.generate_content(parts, generation_config=cfg)
        for p in resp.candidates[0].content.parts:
            if hasattr(p, "inline_data") and p.inline_data:
                return Image.open(
                    io.BytesIO(base64.b64decode(p.inline_data.data))
                ).convert("RGB")
        return None

    resp = _NEW_CLIENT.models.generate_content(
        model=IMAGE_GEN_MODEL,
        contents=[genai_new_types.Content(role="user", parts=_parts_to_new_sdk(parts))],
        config=genai_new_types.GenerateContentConfig(
            system_instruction=system, temperature=temperature,
            response_modalities=["IMAGE", "TEXT"]),
    )
    candidates = getattr(resp, "candidates", None) or []
    if not candidates:
        return None
    for p in getattr(candidates[0].content, "parts", []) or []:
        inline = getattr(p, "inline_data", None)
        if inline and getattr(inline, "data", None):
            raw = (inline.data if isinstance(inline.data, (bytes, bytearray))
                   else base64.b64decode(inline.data))
            return Image.open(io.BytesIO(raw)).convert("RGB")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Validator class
# ─────────────────────────────────────────────────────────────────────────────

class TechnicalAccuracyValidator:
    """
    Agent 2 — Technical Accuracy Validator (v2).

    from agent2_validator import TechnicalAccuracyValidator, AgentResult
    result = TechnicalAccuracyValidator().run(
        reference_image_path = "ref.png",
        output_image_path    = "out.png",
        context              = "Voiceover: ...",
        save_corrected_to    = "corrected.png",
        progress_cb          = print,
    )
    """

    def __init__(self):
        self._log_lines: list[str] = []
        self._cb: Callable[[str], None] = lambda _: None

    # ── entry point ────────────────────────────────────────────────────────

    def run(
        self,
        reference_image_path: str,
        output_image_path: str,
        context: str = "",
        save_corrected_to: str = "validated_output.png",
        progress_cb: Callable[[str], None] | None = None,
    ) -> AgentResult:

        self._log_lines = []
        self._cb = progress_cb or (lambda _: None)
        self._emit("Agent 2 v2 — START")

        # 1. Detect all differences
        raw_diffs = self._detect(reference_image_path, output_image_path, context)
        all_diffs = [
            Difference(
                diff_id       = d.get("diff_id", f"DIFF-{i+1:03d}"),
                description   = d.get("description", ""),
                location_hint = d.get("location_hint", ""),
            )
            for i, d in enumerate(raw_diffs)
        ]
        self._emit(f"[S1] {len(all_diffs)} raw difference(s) detected")

        if not all_diffs:
            self._emit("No differences — passing image through.")
            shutil.copy(output_image_path, save_corrected_to)
            return self._result(reference_image_path, output_image_path,
                                save_corrected_to, [], 0, 0)

        # 2. Significance filter — drop cosmetic noise before reasoning
        material, trivial_count = self._significance_filter(all_diffs)
        self._emit(f"[S2] {trivial_count} trivial diff(s) filtered out")
        self._emit(f"[S2] {len(material)} domain-material diff(s) proceeding")

        if not material:
            self._emit("All differences cosmetic — image is technically accurate.")
            shutil.copy(output_image_path, save_corrected_to)
            return self._result(reference_image_path, output_image_path,
                                save_corrected_to, all_diffs, trivial_count, 0)

        # 3–6. Four-round skeptical reasoning (on material diffs only)
        r1 = self._round1(material, reference_image_path, output_image_path, context)
        r2 = self._round2(material, r1, reference_image_path, output_image_path, context)
        r3 = self._round3(material, r1, r2, reference_image_path, output_image_path, context)
        decisions = self._round4(material, r1, r2, r3, context)

        for d in material:
            dec = decisions.get(d.diff_id, {})
            d.final_verdict          = (Verdict.INACCURACY if dec.get("action") == "CORRECT"
                                        else Verdict.CREATIVE_LIBERTY)
            d.confidence             = dec.get("confidence", 0.0)
            d.correction_instruction = dec.get("correction_instruction", "")
            try:
                d.significance = Severity(dec.get("severity", d.significance.value))
            except ValueError:
                pass

        # Sort to_fix: CRITICAL → MAJOR → MINOR
        _sev_order = {Severity.CRITICAL: 0, Severity.MAJOR: 1,
                      Severity.MINOR: 2, Severity.TRIVIAL: 3}
        to_fix = sorted(
            [d for d in material if d.final_verdict == Verdict.INACCURACY],
            key=lambda d: _sev_order.get(d.significance, 9)
        )

        n_crit  = sum(1 for d in to_fix if d.significance == Severity.CRITICAL)
        n_major = sum(1 for d in to_fix if d.significance == Severity.MAJOR)
        n_minor = sum(1 for d in to_fix if d.significance == Severity.MINOR)
        self._emit(f"[S6] {len(to_fix)} inaccuracy/ies to correct "
                   f"({n_crit} critical / {n_major} major / {n_minor} minor)")
        self._emit(f"[S6] {len(material) - len(to_fix)} creative liberty/ies preserved")

        # 7. Batched image correction
        corrected_path  = None
        corrections_done = 0

        if to_fix:
            corrected_img, corrections_done = self._batched_correct(
                reference_image_path, output_image_path, to_fix
            )
            if corrected_img:
                corrected_img.save(save_corrected_to)
                corrected_path = save_corrected_to
                self._emit(f"[S7] Corrected image saved → {save_corrected_to}")
            else:
                self._emit("[S7] WARNING: batched correction failed — passing through")
        else:
            shutil.copy(output_image_path, save_corrected_to)
            corrected_path = save_corrected_to
            self._emit("[S7] No corrections needed")

        return self._result(reference_image_path, output_image_path,
                            corrected_path, all_diffs, trivial_count, corrections_done)

    # ── Step 1: Detection ──────────────────────────────────────────────────

    def _detect(self, ref: str, out: str, ctx: str) -> list[dict]:
        self._emit("[S1] Detecting visual differences…")
        t0 = time.time()
        parsed = _parse_json(_text_call(
            _DIFF_DETECTOR_SYSTEM,
            [
                "REFERENCE IMAGE (original, technically correct):",
                _img_part(ref),
                "OUTPUT IMAGE (post-pipeline, to validate):",
                _img_part(out),
                f"\nVOICEOVER / SCENE CONTEXT:\n{ctx}\n\n"
                "List every visual difference. Use HVAC terminology. JSON only.",
            ],
            temperature=0.1,
        ), "detect")
        self._emit(f"[S1] Done in {time.time()-t0:.1f}s")
        return parsed.get("differences", [])

    # ── Step 2: Significance filter ────────────────────────────────────────

    def _significance_filter(
        self, diffs: list[Difference]
    ) -> tuple[list[Difference], int]:
        self._emit("[S2] Significance filter — dropping cosmetic noise…")
        diff_json = json.dumps(
            [{"diff_id": d.diff_id, "description": d.description,
              "location_hint": d.location_hint} for d in diffs], indent=2)
        t0 = time.time()
        parsed = _parse_json(_text_call(
            _SIGNIFICANCE_FILTER_SYSTEM,
            [
                f"DIFFERENCES TO EVALUATE:\n{diff_json}\n\n"
                "Classify each as domain-material or trivial. JSON only.",
            ],
            temperature=0.1,
        ), "filter")
        self._emit(f"[S2] Filter done in {time.time()-t0:.1f}s")

        trivial_ids  = set(parsed.get("trivial_diff_ids", []))
        severity_map = {
            m["diff_id"]: m.get("severity", "MINOR")
            for m in parsed.get("material_differences", [])
        }

        material: list[Difference] = []
        for d in diffs:
            if d.diff_id in trivial_ids:
                d.significance = Severity.TRIVIAL
                self._emit(f"  TRIVIAL  {d.diff_id}: {d.description[:60]}")
            else:
                try:
                    d.significance = Severity(severity_map.get(d.diff_id, "MINOR"))
                except ValueError:
                    d.significance = Severity.MINOR
                material.append(d)
                self._emit(f"  {d.significance.value:<8} {d.diff_id}: {d.description[:60]}")

        return material, len(trivial_ids)

    # ── Round 1 ────────────────────────────────────────────────────────────

    def _round1(self, diffs, ref, out, ctx) -> dict:
        self._emit("[S3] Round 1 — Initial classification…")
        diff_json = json.dumps(
            [{"diff_id": d.diff_id, "description": d.description,
              "location_hint": d.location_hint, "severity": d.significance.value}
             for d in diffs], indent=2)
        t0 = time.time()
        parsed = _parse_json(_text_call(
            _ROUND1_SYSTEM,
            [
                f"VOICEOVER / CONTEXT:\n{ctx}\n\n"
                f"DOMAIN-MATERIAL DIFFERENCES:\n{diff_json}\n\n"
                "REFERENCE IMAGE:", _img_part(ref),
                "OUTPUT IMAGE:", _img_part(out),
                "\nClassify each difference. JSON only.",
            ],
            temperature=0.2,
        ), "r1")
        self._emit(f"[S3] Round 1 done in {time.time()-t0:.1f}s")
        result = {c["diff_id"]: c for c in parsed.get("classifications", [])}
        for did, c in result.items():
            self._emit(f"  R1 {did}: {c.get('verdict','?')} "
                       f"conf={c.get('confidence',0):.2f} [{c.get('severity','?')}]")
        return result

    # ── Round 2 ────────────────────────────────────────────────────────────

    def _round2(self, diffs, r1, ref, out, ctx) -> dict:
        self._emit("[S4] Round 2 — Devil's advocate…")
        t0 = time.time()
        parsed = _parse_json(_text_call(
            _ROUND2_SYSTEM,
            [
                f"VOICEOVER / CONTEXT:\n{ctx}\n\n"
                f"ROUND 1 CLASSIFICATIONS:\n{json.dumps(r1, indent=2)}\n\n"
                "REFERENCE IMAGE:", _img_part(ref),
                "OUTPUT IMAGE:", _img_part(out),
                "\nChallenge every TECHNICAL_INACCURACY verdict. JSON only.",
            ],
            temperature=0.4,
        ), "r2")
        self._emit(f"[S4] Round 2 done in {time.time()-t0:.1f}s")
        result = {c["diff_id"]: c for c in parsed.get("revised_classifications", [])}
        for did, c in result.items():
            changed = c.get("revised_verdict") != c.get("previous_verdict")
            self._emit(f"  R2 {did}: {'⟳ CHANGED → ' if changed else '  upheld   '}"
                       f"{c.get('revised_verdict','?')} conf={c.get('confidence',0):.2f}")
        return result

    # ── Round 3 ────────────────────────────────────────────────────────────

    def _round3(self, diffs, r1, r2, ref, out, ctx) -> dict:
        self._emit("[S5] Round 3 — Definitive verdict…")
        t0 = time.time()
        parsed = _parse_json(_text_call(
            _ROUND3_SYSTEM,
            [
                f"VOICEOVER / CONTEXT:\n{ctx}\n\n"
                f"PRIOR ROUNDS:\n{json.dumps({'r1': r1, 'r2': r2}, indent=2)}\n\n"
                "REFERENCE IMAGE:", _img_part(ref),
                "OUTPUT IMAGE:", _img_part(out),
                "\nDefinitive classification + focused correction instructions. JSON only.",
            ],
            temperature=0.1,
        ), "r3")
        self._emit(f"[S5] Round 3 done in {time.time()-t0:.1f}s")
        result = {c["diff_id"]: c for c in parsed.get("definitive_classifications", [])}
        for did, c in result.items():
            self._emit(f"  R3 {did}: ★ {c.get('final_verdict','?')} "
                       f"conf={c.get('confidence',0):.2f} | "
                       f"{c.get('hvac_rule_cited','')[:55]}")
        return result

    # ── Round 4 ────────────────────────────────────────────────────────────

    def _round4(self, diffs, r1, r2, r3, ctx) -> dict:
        self._emit("[S6] Round 4 — Final consolidation…")
        t0 = time.time()
        parsed = _parse_json(_text_call(
            _ROUND4_SYSTEM,
            [
                f"VOICEOVER / CONTEXT:\n{ctx}\n\n"
                f"ALL ROUNDS:\n{json.dumps({'r1': r1, 'r2': r2, 'r3': r3}, indent=2)}\n\n"
                f"Confidence threshold: {CONFIDENCE_THRESHOLD}\n\n"
                "Final CORRECT/PRESERVE list, sorted by severity. JSON only.",
            ],
            temperature=0.05,
        ), "r4")
        self._emit(f"[S6] Consolidation done in {time.time()-t0:.1f}s")
        self._emit(f"  Audit: {parsed.get('audit_summary', '')[:220]}")
        return {v["diff_id"]: v for v in parsed.get("final_verdicts", [])}

    # ── Step 7: Batched image correction ───────────────────────────────────

    def _batched_correct(
        self,
        ref: str,
        out: str,
        to_fix: list[Difference],
    ) -> tuple[Image.Image | None, int]:
        """
        Apply corrections in batches of MAX_FIXES_PER_CALL.
        Each batch receives the output of the previous batch as its base image,
        so the model only handles a small focused set of changes at a time.
        Returns (final_image, confirmed_correction_count).
        """
        batches = [to_fix[i: i + MAX_FIXES_PER_CALL]
                   for i in range(0, len(to_fix), MAX_FIXES_PER_CALL)]
        self._emit(f"[S7] {len(to_fix)} fix(es) → "
                   f"{len(batches)} batch(es) of ≤{MAX_FIXES_PER_CALL}")

        current: str | Image.Image = out   # starts as file path, becomes PIL after first batch
        total_confirmed = 0

        for b_idx, batch in enumerate(batches):
            ids = [d.diff_id for d in batch]
            self._emit(f"  Batch {b_idx+1}/{len(batches)}: {ids}")

            correction_text = "\n".join(
                f"[{i+1}] {d.diff_id} ({d.significance.value}): "
                f"{d.correction_instruction}"
                for i, d in enumerate(batch)
            )

            corrected: Image.Image | None = None
            for attempt in range(1, MAX_CORRECTION_RETRIES + 1):
                self._emit(f"    image-gen attempt {attempt}/{MAX_CORRECTION_RETRIES}…")
                try:
                    corrected = _image_call(
                        _IMAGE_CORRECTION_SYSTEM,
                        [
                            f"CORRECTIONS TO APPLY ({len(batch)} item(s)):\n"
                            f"{correction_text}\n\n"
                            "REFERENCE IMAGE (source of truth for correct appearance):",
                            _img_part(ref),
                            "CURRENT IMAGE (apply corrections here):",
                            _img_part(current),
                            f"\nApply ONLY these {len(batch)} correction(s). "
                            "Leave everything else unchanged.",
                        ],
                        temperature=0.05,
                    )
                    if corrected:
                        break
                except Exception as exc:
                    self._emit(f"    attempt {attempt} error: {exc}")
                    time.sleep(2 ** attempt)

            if corrected is None:
                self._emit(f"  Batch {b_idx+1} FAILED — skipping, continuing with next")
                continue

            # Verify each fix in this batch
            confirmed = self._verify_batch(corrected, ref, batch)
            total_confirmed += confirmed

            # Single retry for any unverified fixes
            failed = [d for d in batch if not d.correction_verified]
            if failed:
                self._emit(f"    {len(failed)} fix(es) not verified — retrying…")
                retry_text = "\n".join(
                    f"[{i+1}] {d.diff_id}: {d.correction_instruction}"
                    for i, d in enumerate(failed)
                )
                retry_img = _image_call(
                    _IMAGE_CORRECTION_SYSTEM,
                    [
                        f"RETRY — {len(failed)} correction(s) not yet applied:\n"
                        f"{retry_text}\n\n"
                        "REFERENCE IMAGE:", _img_part(ref),
                        "CURRENT IMAGE:", _img_part(corrected),
                        "\nApply ONLY these corrections.",
                    ],
                    temperature=0.05,
                )
                if retry_img:
                    retry_confirmed = self._verify_batch(retry_img, ref, failed)
                    if retry_confirmed:
                        corrected = retry_img
                        total_confirmed += retry_confirmed

            current = corrected  # pass to next batch

        final = current if not isinstance(current, str) else None
        self._emit(f"[S7] {total_confirmed}/{len(to_fix)} correction(s) verified")
        return final, total_confirmed

    # ── Verification ───────────────────────────────────────────────────────

    def _verify_batch(
        self,
        corrected: Image.Image,
        ref: str,
        batch: list[Difference],
    ) -> int:
        """Returns number of confirmed corrections in this batch."""
        fix_list = json.dumps(
            [{"diff_id": d.diff_id,
              "correction_instruction": d.correction_instruction}
             for d in batch], indent=2)
        parsed = _parse_json(_text_call(
            _CORRECTION_VERIFIER_SYSTEM,
            [
                f"CORRECTIONS THAT WERE REQUESTED:\n{fix_list}\n\n"
                "REFERENCE IMAGE (correct appearance):", _img_part(ref),
                "CORRECTED IMAGE (check these corrections were applied):",
                _img_part(corrected),
                "\nVerify each correction. JSON only.",
            ],
            temperature=0.1,
        ), "verify")

        ver_map = {v["diff_id"]: v for v in parsed.get("verification", [])}
        confirmed = 0
        for d in batch:
            v = ver_map.get(d.diff_id, {})
            d.correction_verified = v.get("applied", False) and v.get("confidence", 0) >= 0.65
            mark = "✔" if d.correction_verified else "✘"
            self._emit(f"    verify {d.diff_id}: {mark} "
                       f"conf={v.get('confidence',0):.2f} — {v.get('note','')[:60]}")
            if d.correction_verified:
                confirmed += 1
        return confirmed

    # ── Internals ──────────────────────────────────────────────────────────

    def _emit(self, msg: str) -> None:
        log.info(msg)
        self._log_lines.append(msg)
        self._cb(msg)

    def _result(
        self,
        ref: str,
        out: str,
        corrected: str | None,
        all_diffs: list[Difference],
        trivial_count: int,
        corrections_done: int,
    ) -> AgentResult:
        material = [d for d in all_diffs if d.significance != Severity.TRIVIAL]
        return AgentResult(
            reference_image_path          = ref,
            output_image_path             = out,
            corrected_image_path          = corrected,
            differences                   = all_diffs,
            trivial_filtered              = trivial_count,
            inaccuracies_found            = sum(1 for d in material
                                               if d.final_verdict == Verdict.INACCURACY),
            inaccuracies_corrected        = corrections_done,
            creative_liberties_preserved  = sum(1 for d in material
                                               if d.final_verdict == Verdict.CREATIVE_LIBERTY),
            validation_passed             = corrections_done == sum(
                                               1 for d in material
                                               if d.final_verdict == Verdict.INACCURACY),
            audit_log                     = list(self._log_lines),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_agent2(
    reference_image_path: str,
    output_image_path: str,
    context: str = "",
    save_corrected_to: str = "validated_output.png",
    progress_cb: Callable[[str], None] | None = None,
) -> AgentResult:
    """Pipeline-friendly one-liner.

    Accepts either local file paths or http(s) image links for both inputs.
    """
    is_ref_url = bool(re.match(r"^https?://", (reference_image_path or "").strip(), re.IGNORECASE))
    is_out_url = bool(re.match(r"^https?://", (output_image_path or "").strip(), re.IGNORECASE))

    if is_ref_url or is_out_url:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ref_path = tmp_path / "reference_input.png"
            out_path = tmp_path / "output_input.png"

            if is_ref_url:
                _resolve_image_input(None, reference_image_path, ref_path, "reference")
            else:
                ref_path.write_bytes(Path(reference_image_path).read_bytes())

            if is_out_url:
                _resolve_image_input(None, output_image_path, out_path, "output")
            else:
                out_path.write_bytes(Path(output_image_path).read_bytes())

            return TechnicalAccuracyValidator().run(
                reference_image_path=str(ref_path),
                output_image_path=str(out_path),
                context=context,
                save_corrected_to=save_corrected_to,
                progress_cb=progress_cb,
            )

    return TechnicalAccuracyValidator().run(
        reference_image_path = reference_image_path,
        output_image_path    = output_image_path,
        context              = context,
        save_corrected_to    = save_corrected_to,
        progress_cb          = progress_cb,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

def _extract_drive_file_id(url: str) -> str | None:
    for pat in [r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
                r"[?&]id=([a-zA-Z0-9_-]+)"]:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None


def _resolve_image_input(uploaded, url: str, target: Path, label: str) -> None:
    if uploaded:
        Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB").save(target, "PNG")
        return
    if url:
        candidates = [url]
        if "drive.google.com" in url or "docs.google.com" in url:
            fid = _extract_drive_file_id(url)
            if fid:
                candidates.append(
                    f"https://drive.google.com/uc?export=download&id={fid}")
        last_err = None
        for u in candidates:
            try:
                r = requests.get(u, timeout=30, allow_redirects=True,
                                 headers={"User-Agent": "Agent2/2.0"})
                r.raise_for_status()
                ct = r.headers.get("Content-Type", "")
                if ct and "image/" not in ct and "octet-stream" not in ct:
                    raise ValueError(f"Non-image content-type: {ct}")
                Image.open(io.BytesIO(r.content)).convert("RGB").save(target, "PNG")
                return
            except Exception as e:
                last_err = e
        raise ValueError(f"Could not fetch {label} image: {last_err}")
    raise ValueError(f"No {label} image provided.")


def render_agent2_page() -> None:
    """
    Importable page renderer.
    Usage in a multi-page app:
        from agent2_validator import render_agent2_page
        render_agent2_page()
    """
    st.title("🔍 Agent 2 — Technical Accuracy Validator")
    st.caption("Significance-filtered · severity-ranked · batched corrections · "
               "4-round skeptical reasoning")

    # API key
    env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    key_in = st.text_input("Gemini API key", type="password", value=env_key,
                           help="Falls back to GEMINI_API_KEY env var.").strip()
    if key_in:
        os.environ["GEMINI_API_KEY"] = key_in
        st.session_state["gemini_api_key"] = key_in

    # Image inputs
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Reference image")
        ref_file = st.file_uploader("Upload", type=["png", "jpg", "jpeg", "webp"],
                                    key="a2_ref_f")
        ref_url  = st.text_input("or paste URL", key="a2_ref_u",
                                 placeholder="https://…").strip()
        if ref_file:
            st.image(ref_file, use_container_width=True)
        elif ref_url:
            st.caption(ref_url[:80])

    with col2:
        st.subheader("Output image (post-Agent 1)")
        out_file = st.file_uploader("Upload", type=["png", "jpg", "jpeg", "webp"],
                                    key="a2_out_f")
        out_url  = st.text_input("or paste URL", key="a2_out_u",
                                 placeholder="https://…").strip()
        if out_file:
            st.image(out_file, use_container_width=True)
        elif out_url:
            st.caption(out_url[:80])

    context   = st.text_area("Voiceover / scene context", height=100,
                              placeholder="Paste the voiceover transcript here…")
    save_name = st.text_input("Output filename", value="validated_output.png")

    can_run = bool((ref_file or ref_url) and (out_file or out_url)
                   and _resolve_api_key())
    if not can_run:
        st.info("Provide both images (upload or URL) and an API key to proceed.")

    if not st.button("▶  Run Validation", type="primary", disabled=not can_run):
        return

    logs: list[str] = []
    log_box = st.empty()
    ref_bytes: bytes | None = None
    out_bytes: bytes | None = None
    corrected_bytes: bytes | None = None

    def progress(msg: str) -> None:
        logs.append(msg)
        log_box.code("\n".join(logs[-80:]), language="text")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path  = Path(tmp)
        ref_path  = tmp_path / "ref.png"
        out_path  = tmp_path / "out.png"
        save_path = tmp_path / save_name

        try:
            _resolve_image_input(ref_file, ref_url, ref_path, "reference")
            _resolve_image_input(out_file, out_url, out_path, "output")
        except Exception as e:
            st.error(str(e))
            return

        with st.spinner("Validating…"):
            try:
                result = run_agent2(
                    reference_image_path = str(ref_path),
                    output_image_path    = str(out_path),
                    context              = context,
                    save_corrected_to    = str(save_path),
                    progress_cb          = progress,
                )
            except Exception as e:
                st.error(f"Agent 2 error: {e}")
                return

        # Materialize bytes before temp directory is cleaned up.
        ref_bytes = ref_path.read_bytes() if ref_path.exists() else None
        out_bytes = out_path.read_bytes() if out_path.exists() else None
        corrected_path = Path(result.corrected_image_path) if result.corrected_image_path else None
        if corrected_path and corrected_path.exists():
            corrected_bytes = corrected_path.read_bytes()

    st.success("Validation complete")

    # Metrics row
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Trivial filtered",  result.trivial_filtered)
    mc2.metric("Inaccuracies found", result.inaccuracies_found)
    mc3.metric("Corrected",          result.inaccuracies_corrected)
    mc4.metric("Liberties kept",     result.creative_liberties_preserved)
    mc5.metric("Passed",             "✔ Yes" if result.validation_passed else "✘ Partial")

    # Three-way image comparison
    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        st.caption("Reference (original)")
        if ref_bytes:
            st.image(Image.open(io.BytesIO(ref_bytes)), use_container_width=True)
        else:
            st.warning("Reference image unavailable.")
    with ic2:
        st.caption("Before (post-Agent 1)")
        if out_bytes:
            st.image(Image.open(io.BytesIO(out_bytes)), use_container_width=True)
        else:
            st.warning("Output image unavailable.")
    with ic3:
        st.caption("After (Agent 2 corrected)")
        if corrected_bytes:
            corrected_img = Image.open(io.BytesIO(corrected_bytes))
            st.image(corrected_img, use_container_width=True)
            st.download_button("⬇ Download corrected image", corrected_bytes,
                               file_name=save_name, mime="image/png",
                               use_container_width=True)
        else:
            st.warning("No corrected image produced.")

    # Difference breakdown
    material = [d for d in result.differences if d.significance != Severity.TRIVIAL]
    trivial  = [d for d in result.differences if d.significance == Severity.TRIVIAL]

    if material:
        st.subheader("Domain-material differences")
        tab_all, tab_fix, tab_keep = st.tabs([
            f"All ({len(material)})",
            f"Inaccuracies ({result.inaccuracies_found})",
            f"Creative liberties ({result.creative_liberties_preserved})",
        ])

        def _render_diffs(diffs: list[Difference]) -> None:
            if not diffs:
                st.write("None.")
                return
            for d in diffs:
                icon     = "🔴" if d.final_verdict == Verdict.INACCURACY else "🟢"
                verified = "  ✔ verified" if d.correction_verified else ""
                with st.expander(
                    f"{icon} [{d.significance.value}]  {d.diff_id}  —  "
                    f"{d.description[:70]}{verified}"
                ):
                    cols = st.columns(2)
                    cols[0].write(f"**Verdict:** {d.final_verdict.value}")
                    cols[0].write(f"**Confidence:** {d.confidence:.0%}")
                    cols[1].write(f"**Location:** {d.location_hint}")
                    cols[1].write(f"**Severity:** {d.significance.value}")
                    if d.hvac_rule_cited:
                        st.write(f"**HVAC rule:** {d.hvac_rule_cited}")
                    if d.correction_instruction and d.final_verdict == Verdict.INACCURACY:
                        st.info(f"**Correction applied:** {d.correction_instruction}")

        with tab_all:
            _render_diffs(material)
        with tab_fix:
            _render_diffs([d for d in material if d.final_verdict == Verdict.INACCURACY])
        with tab_keep:
            _render_diffs([d for d in material if d.final_verdict == Verdict.CREATIVE_LIBERTY])

    if trivial:
        with st.expander(f"Trivial / cosmetic — filtered out ({len(trivial)})"):
            for d in trivial:
                st.write(f"- `{d.diff_id}` {d.description}")

    with st.expander("Full audit log"):
        st.code("\n".join(result.audit_log), language="text")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

def _standalone_main() -> None:
    st.set_page_config(
        page_title = "Agent 2 — Technical Accuracy Validator",
        page_icon  = "🔍",
        layout     = "wide",
        initial_sidebar_state = "collapsed",
    )
    render_agent2_page()


if __name__ == "__main__":
    _standalone_main()