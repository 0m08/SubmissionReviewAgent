import os
import json
import time
import re
import tempfile
import importlib.util
import traceback
import concurrent.futures
from io import BytesIO
from PIL import Image
from typing import Optional, Dict, Any, Callable
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langsmith import traceable
import gspread

# Import local modules
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))  # .../automated
agents_dir = os.path.dirname(current_dir)                  # .../graphics_asset_creation
root_dir = os.path.dirname(os.path.dirname(agents_dir))     # workspace root (2 levels above graphics_asset_creation)
sys.path.insert(0, root_dir)

from agents.graphics_asset_creation.reviewers.voiceover_reviewer import (
    reviewer_agent,
    VoiceoverReviewResult,
    RecommendedInstructions,
    prepare_image_for_gemini,
    _get_client,
    _extract_json,
    styling_guide,
    REVIEWER_MODEL,
    build_review_prompt
)
from agents.graphics_asset_creation.reviewers.copyright_reviewer import (
    copyright_reviewer_agent,
    build_copyright_system_instruction,
    CopyrightReviewResult
)
from agents.graphics_asset_creation.reviewers.agent2_ui import run_agent2
from agents.graphics_asset_creation.reviewers.voiceover_focus_agent import run_adaptive_focus_or_illustrator
from agents.graphics_asset_creation.automated.llm_call_tracker import tracker as llm_tracker
from services.llm_service import call_llm_with_retry
# [STRATEGIES DISABLED] — comment back in to re-enable the strategy/shift system
# from agents.graphics_asset_creation.automated.strategy_manager import (
#     StrategyCondition, select_strategy, apply_strategy, load_strategies
# )


# [STRATEGIES DISABLED] — _apply_strategy_filter commented out with strategy system
# def _apply_strategy_filter(instructions_dict: dict, strategy_id: str) -> dict:
#     ... (see git history to restore)

load_dotenv()

PIPELINE_ASPECT_RATIO = "16:9"


def _run_with_timeout(func, timeout_seconds: int, *args, **kwargs):
    """Run a blocking call with a hard timeout to avoid stuck round-1 reviews."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError(f"Call exceeded {timeout_seconds}s timeout")


def _send_accuracy_message_with_timeout(accuracy_chat, msg_parts):
    """Send accuracy reviewer message with a configurable hard timeout."""
    timeout_seconds = int(os.getenv("VOICEOVER_ACCURACY_REVIEW_TIMEOUT_SECONDS", "120"))
    return _run_with_timeout(accuracy_chat.send_message, timeout_seconds, msg_parts)

# =============================================================================
# GOOGLE DRIVE DOWNLOAD HELPER
# Handles the Drive virus-scan confirmation page that Google returns for files
# larger than ~100 KB when using `?export=download`.
# =============================================================================

def _download_drive_image(url: str, drive=None) -> Image.Image:
    """
    Download an image from either a Google Drive URL or a plain web URL.

    - Google Drive URLs  → authenticated PyDrive2 service-account download.
    - All other http(s) URLs → plain HTTP GET via requests (public web images).

    Raises ValueError with a descriptive message on failure.
    """
    import requests as _requests

    url = url.strip()

    # ── Detect Drive URLs ────────────────────────────────────────────────────
    file_id = None
    if 'drive.google.com' in url or 'docs.google.com' in url:
        for pattern in [
            r'drive\.google\.com/file/d/([a-zA-Z0-9_-]+)',
            r'[?&]id=([a-zA-Z0-9_-]+)',
        ]:
            m = re.search(pattern, url)
            if m:
                file_id = m.group(1)
                break

    # ── Branch: Google Drive (authenticated) ────────────────────────────────
    if file_id:
        if drive is None:
            raise ValueError(
                'A PyDrive2 `drive` client is required to download private Drive files. '
                'Make sure run_automation passes the authenticated drive object through.'
            )
        print(f'  Downloading Drive file {file_id} via service account...')
        gfile = drive.CreateFile({'id': file_id})
        content_bytes = gfile.GetContentString(encoding='latin-1').encode('latin-1')

        try:
            img = Image.open(BytesIO(content_bytes))
        except Exception as exc:
            snippet = content_bytes[:300].decode('utf-8', errors='replace')
            raise ValueError(
                f'Drive API returned content that could not be decoded as an image.\n'
                f'File ID: {file_id}\n'
                f'Size received: {len(content_bytes):,} bytes\n'
                f'First 300 bytes: {snippet!r}'
            ) from exc

        print(f'  ✅ Drive download succeeded ({len(content_bytes):,} bytes)')
        return img

    # ── Branch: Plain web URL (public) ──────────────────────────────────────
    if url.startswith(('http://', 'https://')):
        print(f'  Downloading web image: {url[:100]}...')
        try:
            resp = _requests.get(url, timeout=30, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; ContentBot/1.0)'
            })
            resp.raise_for_status()
            content_bytes = resp.content
        except Exception as exc:
            raise ValueError(f'HTTP download failed for URL {url!r}: {exc}') from exc

        try:
            img = Image.open(BytesIO(content_bytes))
        except Exception as exc:
            raise ValueError(
                f'Could not decode web response as an image.\n'
                f'URL: {url}\n'
                f'Status: {resp.status_code}, Content-Type: {resp.headers.get("Content-Type", "unknown")}'
            ) from exc

        print(f'  ✅ Web download succeeded ({len(content_bytes):,} bytes)')
        return img

    raise ValueError(
        f'Unsupported URL format: {url!r}. '
        'Expected a Google Drive URL or a plain http(s) web URL.'
    )


def _run_agent2_accuracy_validation(
    reference_image: Image.Image,
    output_image: Image.Image,
    context: str,
) -> Dict[str, Any]:
    """
    Run Agent2 accuracy validator on in-memory images.

    Returns a dict with:
      - result: AgentResult | None
      - correction_instructions: list[str]
      - error: str | None
    """
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ref_path = os.path.join(tmp_dir, "agent2_ref.png")
            out_path = os.path.join(tmp_dir, "agent2_out.png")
            corrected_path = os.path.join(tmp_dir, "agent2_corrected.png")

            reference_image.convert("RGB").save(ref_path, format="PNG")
            output_image.convert("RGB").save(out_path, format="PNG")

            result = run_agent2(
                reference_image_path=ref_path,
                output_image_path=out_path,
                context=context,
                save_corrected_to=corrected_path,
            )

        print("🔎 Agent2 attribute fetch check (result-level):")
        print(
            "   has.validation_passed="
            f"{hasattr(result, 'validation_passed')} | "
            "has.inaccuracies_found="
            f"{hasattr(result, 'inaccuracies_found')} | "
            "has.inaccuracies_corrected="
            f"{hasattr(result, 'inaccuracies_corrected')} | "
            "has.differences="
            f"{hasattr(result, 'differences')}"
        )

        correction_instructions = []
        diffs = getattr(result, "differences", []) or []
        print(f"🔎 Agent2 diff list fetched: count={len(diffs)}")
        for i, diff in enumerate(diffs, 1):
            has_final_verdict = hasattr(diff, "final_verdict")
            has_correction_instruction = hasattr(diff, "correction_instruction")
            diff_id = getattr(diff, "diff_id", f"DIFF-{i:03d}")
            print(
                f"   diff[{i}] {diff_id}: "
                f"has.final_verdict={has_final_verdict}, "
                f"has.correction_instruction={has_correction_instruction}"
            )
            final_verdict = getattr(diff, "final_verdict", "")
            verdict_text = getattr(final_verdict, "value", str(final_verdict)).strip().upper()
            instruction = (getattr(diff, "correction_instruction", "") or "").strip()
            if verdict_text == "TECHNICAL_INACCURACY" and instruction:
                correction_instructions.append(instruction)

        print(
            "🔎 Agent2 extracted correction instructions: "
            f"count={len(correction_instructions)}"
        )

        return {
            "result": result,
            "correction_instructions": correction_instructions,
            "error": None,
        }
    except Exception as exc:
        return {
            "result": None,
            "correction_instructions": [],
            "error": str(exc),
        }



# =============================================================================
# ORCHESTRATOR: review_and_edit_image
# Manages separate sessions for Accuracy and Copyright reviewers,
# and bridges to the image editor.
# =============================================================================

@traceable(
    metadata={
        "agent_name": "voiceover_reviewer",
        "function": "review_and_edit_image"
    }
)
def review_and_edit_image(
    reference_image: Image.Image,
    slide_title: str = "",
    slide_content: str = "",
    voiceover: str = "",
    visual_instruction: str = "",
    image_size: str = "1K",
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    target_stage: str = "full",
    skip_accuracy_validation: bool = False
) -> tuple[VoiceoverReviewResult, Optional[Image.Image], list]:
    """
    Orchestrator: Review and Edit workflow using separate agent sessions.
    
    Creates dedicated chat sessions for:
    - Accuracy Reviewer (technical fidelity)
    - Copyright Reviewer (IP compliance)
    
    Manages the review-edit loop and bridges feedback to the image editor.
    Returns: (FinalReview, FinalImage, HistoryList)
    """
    history = []

    client = _get_client()
    if not client:
        raise ValueError("Google API Client not initialized.")

    # =========================================================================
    # 1. Build System Instructions for Each Session
    # =========================================================================
    accuracy_system_instruction = f"""
[CORE MISSION: CRITICAL ACCURACY ONLY]
Most reference images are already excellent. Your goal is to identify ONLY critical technical errors that would mislead learners and not nitpicking.

[STRICT PROHIBITION: NO ANNOTATIONS]
**DO NOT suggest adding text, labels, arrows, or callouts to the image.** The final output must be a clean technical asset without any digital overlays. Focus only on the physical accuracy of the subject.

DEFAULT TO YES: Unless there are clear factual errors, wrong values, or missing critical elements.

[WHAT TO FLAG]
ONLY flag issues that would cause learner confusion or technical misunderstanding:
1. WRONG VALUES: Gauge shows 0-200 PSI but voiceover says 0-150 PSI
2. MISSING CRITICAL ELEMENTS: Voiceover mentions "red button" but no button is visible
3. CONTRADICTORY STATE: Image shows clean component but voiceover describes corrosion
4. INCORRECT LABELS: Text reads "Input" but voiceover identifies it as "Output"

[WHAT NOT TO FLAG]
Do NOT flag these - they are acceptable:
- Composition or framing that could be "better"
- Elements that are visible but could be "more prominent"
- Adequate technical accuracy even if not "perfect"
- Minor visibility or readability issues that don't prevent understanding
- Any suggestion that is primarily about polish, emphasis, or aesthetics

[VERDICT STANDARD]
YES: If the image adequately supports the voiceover without critical technical errors (even if not perfect)
NO: ONLY if there are factual errors or critical missing elements that would mislead learners

[STYLING GUIDE - SECONDARY FILTER]
{styling_guide}
Only suggest styling changes if they improve clarity or correct an educational error.

[CRITICAL INSTRUCTION: IGNORE BRANDING]
- Assume any missing logo, brand name, or trademark was removed intentionally for legal reasons.
- NEVER flag missing branding as an error, even if the voiceover mentions it (e.g., "The Fieldpiece VP67").
- Evaluate ONLY the functional accuracy (e.g., Is it a vacuum pump? Yes. Is the brand missing? IGNORE).

[REGRESSION DETECTION]
When re-evaluating an edited image, compare it to what you saw in the PREVIOUS round.
Set regression_detected = true ONLY if the edit made things WORSE:
- A previously correct element is now broken (e.g., gauge was readable, now it's garbled)
- The edit introduced NEW critical errors that did not exist before
- The overall educational accuracy regressed compared to the previous version

Set regression_detected = false if:
- This is the first review (nothing to compare against)
- The edit improved or maintained quality, even if issues remain
- The image still has the same problems as before (no improvement, but no regression)
"""

    copyright_system_instruction = build_copyright_system_instruction()

    # =========================================================================
    # 2. Create Separate Chat Sessions
    # =========================================================================
    
    # A. Accuracy Reviewer Session
    accuracy_chat = client.chats.create(
        model=REVIEWER_MODEL,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VoiceoverReviewResult,
            system_instruction=accuracy_system_instruction,
            # thinking_config=types.ThinkingConfig(
            #     thinking_level="medium",
            # )
        )
    )

    def _make_accuracy_chat():
        """Create a fresh accuracy reviewer chat session with the same config."""
        return client.chats.create(
            model=REVIEWER_MODEL,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=VoiceoverReviewResult,
                system_instruction=accuracy_system_instruction,
            )
        )

    # B. Copyright Reviewer Session
    copyright_chat = client.chats.create(
        model=REVIEWER_MODEL,
        config=types.GenerateContentConfig(
            # tools=[{"google_search": {}}],
            response_mime_type="application/json",
            response_schema=CopyrightReviewResult,
            system_instruction=copyright_system_instruction,
            thinking_config=types.ThinkingConfig(
                thinking_level="medium",
            )
        )
    )

    # =========================================================================
    # 3. Intelligent Orchestration Loop
    # =========================================================================
    current_image = reference_image
    last_review_result = None
    round_num = 1
    
    # Coordination state for intelligent conflict prevention
    copyright_transformations_applied = []  # Track what copyright changed (for accuracy awareness)
    similarity_score_history = []  # Track similarity scores for convergence detection
    previous_copyright_instructions = []  # Track copyright instructions history
    copyright_approved_image = None  # Store copyright-approved version
    
    # Rollback state — revert to previous image if an edit causes disaster
    image_stack = []              # Images before each edit (for rollback)
    last_edit_instructions = None # Last applied editing instructions
    last_edit_type = None         # "accuracy" or "copyright"
    failed_edits = []             # [{type, instructions, reason}] for failed edits
    rollback_count = 0
    MAX_ROLLBACKS = 3
    
    print(f"\n🚀 Starting Intelligent Orchestrated Workflow (Accuracy + Copyright Sessions)")
    print(f"   Strategy: Smart coordination to prevent conflicts through context awareness")
    
    last_action_summary = "None (First round)"

    # ── Strategy shift state (DISABLED) ──────────────────────────────────────
    # load_strategies()  # warm cache
    # tried_strategy_ids:         list = []    # strategies CONFIRMED failed (never retry)
    # _pending_accuracy_sid:      str  = ""   # selected-but-not-yet-confirmed accuracy strategy ID
    # _pending_copyright_sid:     str  = ""   # idem for copyright
    # _active_accuracy_sid:       str  = ""   # strategy whose directive is in effect THIS round
    # _current_acc_failure_class:  str  = ""  # track to detect cross-class shifts
    # _accuracy_strategy_directive: str = ""  # injected into accuracy reviewer prompt
    # _copyright_strategy_directive: str = "" # injected into copyright reviewer context
    consecutive_accuracy_failures:   int = 0
    consecutive_copyright_failures:  int = 0
    stale_similarity_rounds:         int = 0
    _prev_similarity                = None
    _accuracy_strategy_directive   = ""  # always empty (strategies disabled)
    _copyright_strategy_directive  = ""  # always empty (strategies disabled)
    # Wall-clock hard cap per subsegment (prevents infinite loops)
    SUBSEG_TIMEOUT_SECONDS = 20 * 60  # 20 minutes
    _subseg_start          = time.time()

    def _finalize_before_return(final_review, final_image, final_history, reason: str):
        """
        Run Voiceover Focus Agent only for full-stage exits.
        Keep existing return contract unchanged on any failure.
        """
        if target_stage != "full":
            return final_review, final_image, final_history

        if final_image is None:
            print(f"[VoiceoverFocus] Skipped ({reason}): final image is None")
            return final_review, final_image, final_history

        try:
            print(f"[VoiceoverFocus] Running post-pipeline adaptive routing pass ({reason})...")
            focus_result = run_adaptive_focus_or_illustrator(
                image=final_image,
                voiceover=voiceover,
                slide_title=slide_title,
                slide_content=slide_content,
                visual_instruction=visual_instruction,
            )

            was_modified = bool(getattr(focus_result, "was_modified", False))
            notes = (getattr(focus_result, "modification_notes", "") or "").strip()
            print(
                "[VoiceoverFocus] Completed | "
                f"was_modified={was_modified} | notes={notes[:160]}"
            )

            candidate_image = getattr(focus_result, "output_image", None)
            if isinstance(candidate_image, Image.Image):
                return final_review, candidate_image, final_history

            print("[VoiceoverFocus] output_image missing/invalid; keeping pre-focus image.")
            return final_review, final_image, final_history
        except Exception as focus_exc:
            print(f"[VoiceoverFocus] Failed ({reason}): {focus_exc}. Keeping pre-focus image.")
            return final_review, final_image, final_history

    while True:  # Dynamic termination based on convergence, not hard limits
        # ── Wall-clock safety valve ─────────────────────────────────────────
        elapsed = time.time() - _subseg_start
        if elapsed > SUBSEG_TIMEOUT_SECONDS:
            print(f"\n⏱ Subsegment wall-clock timeout "
                  f"({SUBSEG_TIMEOUT_SECONDS // 60} min elapsed). "
                  "Returning best available image.")
            _best_result = last_review_result if last_review_result is not None else None
            return _finalize_before_return(_best_result, current_image, history, "timeout")
        is_copyright_round = False  # FIX: initialize before any conditional branch that reads it
        print(f"\n--- ROUND {round_num} ---")
        
        # A. Prepare Message
        image_bytes = prepare_image_for_gemini(current_image)
        
        if round_num == 1:
            prompt = build_review_prompt(slide_title=slide_title, slide_content=slide_content, voiceover=voiceover, visual_instruction=visual_instruction)
        else:
            # Build copyright-aware context - CRITICAL for preventing conflicts
            copyright_context = ""
            if copyright_transformations_applied:
                copyright_context = f"""

[COPYRIGHT TRANSFORMATION CONTEXT]
The following transformations were applied for copyright compliance:
{chr(10).join(f'- {t}' for t in copyright_transformations_applied[-3:])}

IMPORTANT: These are LEGAL transformations that must be preserved:
- Branding removal (logos, trademarks, model numbers)
- Style changes (photo → illustration/render)
- Camera angle changes for differentiation
- Color palette changes

EVALUATION RULES:
1. DO NOT flag missing branding elements (they were removed intentionally)
2. DO NOT request restoration of copyrighted visual styles
3. ONLY flag if functional/educational accuracy is compromised
4. Focus on technical correctness, not commercial appearance

Example VALID concerns:
- "Gauge reading is incorrect" ✓
- "Critical component is functionally wrong" ✓

Example INVALID concerns:
- "Missing brand logo" ✗ (intentional)
- "Different visual style" ✗ (intentional)
- "Camera angle changed" ✗ (intentional)
"""
            
            base_prompt = f"""Re-evaluate this edited image against the voiceover.

Does this image accurately demonstrate what is described in the voiceover?

CONTEXT FROM PREVIOUS ACTIONS:
{last_action_summary}{copyright_context}

Set verdict to 'Yes' if the image is technically accurate and supports the voiceover.
Set verdict to 'No' if any CRITICAL technical errors remain, and provide specific new instructions.
"""
            prompt = base_prompt

        # B. Send to Accuracy Session
        try:
            # [STRATEGIES DISABLED] directive injection commented out — prompt sent as-is
            # if _accuracy_strategy_directive.strip():
            #     print(f"   [strategy] Injecting accuracy directive "
            #           f"({len(_accuracy_strategy_directive)} chars) into prompt.")
            msg_parts = [
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                types.Part.from_text(text=prompt)  # no strategy directive appended
            ]
            
            with llm_tracker.call(REVIEWER_MODEL, "Accuracy Reviewer") as usage:
                response = call_llm_with_retry(
                    _send_accuracy_message_with_timeout,
                    accuracy_chat,
                    msg_parts,
                    max_retries=int(os.getenv("VOICEOVER_ACCURACY_REVIEW_RETRIES", "2")),
                    initial_wait=2,
                )
                usage.set_response(response)
            
            # C. Parse Accuracy Result
            if hasattr(response, 'parsed') and response.parsed:
                review_result = response.parsed
            else:
                data = _extract_json(response.text)
                review_result = VoiceoverReviewResult(**data)

            last_review_result = review_result
            
            # Record in history
            round_data = {
                "round": round_num,
                "review": review_result,
                "image": current_image,
                "agent": "Voiceover Reviewer"
            }
            history.append(round_data)
            if callback: callback(round_data)
            
            print(f"Round {round_num} [Voiceover] Verdict: {review_result.get('verdict', 'No') if isinstance(review_result, dict) else review_result.verdict}")
            
            # D. Intelligent Branching Logic
            v_verdict = str(review_result.verdict).strip().lower()
            
            if v_verdict == "yes":
                # D.5 Secondary validation (Agent2) - Only run as a safety gate for YES verdicts
                if not skip_accuracy_validation:
                    print("🔍 Running technical accuracy validation check (Safety Gate)...")
                    validator_context = (
                        f"Slide Title: {slide_title}\n"
                        f"Slide Content: {slide_content}\n"
                        f"Voiceover: {voiceover}\n"
                        f"Visual Instruction: {visual_instruction}"
                    )
                    agent2_validation = _run_agent2_accuracy_validation(
                        reference_image=reference_image,
                        output_image=current_image,
                        context=validator_context,
                    )

                    if agent2_validation["error"]:
                        print(f"⚠️ Agent2 validation skipped due to error: {agent2_validation['error']}")
                    else:
                        agent2_result = agent2_validation["result"]
                        print(
                            "✅ Agent2 validation complete | "
                            f"passed={agent2_result.validation_passed} | "
                            f"inaccuracies_found={agent2_result.inaccuracies_found} | "
                            f"inaccuracies_corrected={agent2_result.inaccuracies_corrected}"
                        )

                        # Safety net: prevent false-positive YES from accuracy reviewer.
                        if (
                            not agent2_result.validation_passed
                            or getattr(agent2_result, "inaccuracies_found", 0) > 0
                        ):
                            print(
                                "⚠️ Agent2 found technical inaccuracies after a YES verdict. "
                                "Overriding verdict to NO and routing through normal edit path."
                            )

                            synthesized_notes = agent2_validation["correction_instructions"][:3]
                            synthesized_text = " | ".join(synthesized_notes)

                            # Update verdict/analysis/instructions while preserving existing structure.
                            review_result.verdict = "No"
                            review_result.analysis = (
                                (review_result.analysis or "")
                                + "\n\n[Agent2 Validation] "
                                + f"Detected {agent2_result.inaccuracies_found} technical inaccuracy(ies). "
                                + "Applying corrective edit cycle before copyright review."
                            )
                            if synthesized_text:
                                existing = review_result.recommended_instructions.additional_comments or ""
                                review_result.recommended_instructions.additional_comments = (
                                    (existing + "\n" if existing else "")
                                    + "Agent2 corrective instructions: "
                                    + synthesized_text
                                )

                            print(
                                "🛠 Injected Agent2 correction hints into accuracy instructions "
                                "(additional_comments)."
                            )
                            # Re-calculate verdict after override
                            v_verdict = "no"
                else:
                    print("⏩ Skipping technical accuracy validation check (Agent2) as requested.")

                # Check verdict again in case Agent2 overrode it to 'no'
                if v_verdict == "yes":
                    if target_stage == "accuracy":
                        print(f"🎯 Target Stage 'accuracy' reached. Stopping early.")
                        return _finalize_before_return(review_result, current_image, history, "target-stage-accuracy")
                    
                # If copyright already approved this exact image, we're done (both approved same version)
                if copyright_approved_image is not None and current_image == copyright_approved_image:
                    print(f"✅ Both reviewers approved same image version. FINAL APPROVAL.")
                    return _finalize_before_return(review_result, current_image, history, "both-approved-same-image")
                    
                # --- STAGE 2: COPYRIGHT REVIEW (separate session) ---
                print(f"✅ Voiceover Approved. Triggering Copyright Audit...")
                # VO approved → accuracy strategy (if any) WORKED. Do NOT mark it as tried.
                # It should remain available for future problems of the same class.
                consecutive_accuracy_failures = 0
                _accuracy_strategy_directive  = ""  # always empty (strategies disabled)
                # _pending_accuracy_sid         = ""   # [STRATEGIES DISABLED]
                # _current_acc_failure_class    = ""   # [STRATEGIES DISABLED]
                
                # Provide copyright reviewer with accuracy requirements for preservation
                # Build failed edits context for re-planning
                failed_edits_context = ""
                if failed_edits:
                    failed_edits_context = "\n\nPREVIOUSLY FAILED APPROACHES (DO NOT REPEAT):\n"
                    for i, fe in enumerate(failed_edits, 1):
                        instr_str = ""
                        if fe.get("instructions"):
                            instr_str = ", ".join(f"{k}: {v[:50]}" for k, v in fe["instructions"].items() if v)
                        failed_edits_context += f"{i}. Failed ({fe['type']}): {fe['reason'][:120]}"
                        if instr_str:
                            failed_edits_context += f" | Instructions: {instr_str}"
                        failed_edits_context += "\n"
                    failed_edits_context += "Use a COMPLETELY DIFFERENT transformation strategy.\n"

                accuracy_context = f"""ACCURACY REQUIREMENTS (Preserve these):
- Voiceover: {voiceover}
- Visual Instruction: {visual_instruction}
- Technical validation: Passed by accuracy reviewer

DO NOT transform in ways that break functional accuracy.
Example: If gauge is critical, preserve gauge functionality even if changing style."""
                
                # [STRATEGIES DISABLED] — copyright directive injection commented out
                # if _copyright_strategy_directive.strip():
                #     print(f"   [strategy] Injecting copyright directive "
                #           f"({len(_copyright_strategy_directive)} chars) into copyright context.")
                with llm_tracker.call(REVIEWER_MODEL, "Copyright Reviewer"):
                    copyright_result = copyright_reviewer_agent(
                        current_image, 
                        slide_title=slide_title, 
                        voiceover=voiceover, 
                        chat_session=copyright_chat,
                        reference_images=[reference_image],  # Original input for similarity comparison
                        previous_instructions=previous_copyright_instructions if previous_copyright_instructions else None,
                        previous_similarity_scores=similarity_score_history if similarity_score_history else None,
                        previous_context=accuracy_context  # Pass accuracy requirements to preserve
                    )
                
                # Extract and track similarity score for convergence detection
                if hasattr(copyright_result, 'similarity_scores') and copyright_result.similarity_scores:
                    import re
                    score_match = re.search(r'(\d+\.\d+)%', copyright_result.similarity_scores)
                    if score_match:
                        current_score = float(score_match.group(1))
                        similarity_score_history.append(current_score)
                        print(f"   Similarity score: {current_score:.1f}% (History: {[f'{s:.1f}%' for s in similarity_score_history]})")
                
                # Record in history
                cp_round_data = {
                    "round": round_num,
                    "review": copyright_result,
                    "image": current_image,
                    "agent": "Copyright Reviewer"
                }
                history.append(cp_round_data)
                if callback: callback(cp_round_data)
                
                print(f"Round {round_num} [Copyright] Verdict: {copyright_result.verdict}")
                
                if str(copyright_result.verdict).strip().lower() == "yes":
                    print(f"🎉 Final Approval (VO + Copyright) in Round {round_num}")
                    copyright_approved_image = current_image
                    # CP approved → copyright strategy (if any) WORKED. Do NOT mark as tried.
                    consecutive_copyright_failures = 0
                    stale_similarity_rounds        = 0
                    # _copyright_strategy_directive  = ""  # no-op: always empty
                    # _pending_copyright_sid         = ""   # [STRATEGIES DISABLED]
                    return _finalize_before_return(review_result, current_image, history, "vo-and-copyright-approved")
                else:
                    # ── Track copyright failure + compute stale rounds ────────
                    consecutive_copyright_failures += 1
                    consecutive_accuracy_failures = 0  # copyright failure resets accuracy streak

                    # Determine if similarity has improved since the last round.
                    # Require >= _SIMILARITY_IMPROVEMENT_THRESHOLD pp drop before
                    # resetting the stale counter — micro-fluctuations (e.g. ±0.5 pp)
                    # are noise and must not prevent copyright strategies from activating.
                    _SIMILARITY_IMPROVEMENT_THRESHOLD = 1.5   # percentage points
                    if similarity_score_history:
                        current_sim = similarity_score_history[-1]
                        if _prev_similarity is not None:
                            improvement = _prev_similarity - current_sim  # positive = got better
                            if improvement < _SIMILARITY_IMPROVEMENT_THRESHOLD:
                                stale_similarity_rounds += 1
                            else:
                                stale_similarity_rounds = 0   # reset only on meaningful improvement
                        _prev_similarity = current_sim

                    # [STRATEGIES DISABLED] — convergence detection + copyright strategy selection commented out.
                    # To re-enable: uncomment block below and restore state vars above.
                    # -- Convergence detection --
                    # if len(similarity_score_history) >= 3:
                    #     last_three = similarity_score_history[-3:]
                    #     if last_three[-1] >= last_three[-2] and last_three[-2] >= last_three[-3]:
                    #         print(f"Similarity not improving: {last_three}")
                    #
                    # -- Copyright strategy selection --
                    # _cp_instr = (getattr(copyright_result, 'transformation_instructions', None)
                    #              or getattr(copyright_result, 'recommended_instructions', None))
                    # _cp_class_from_struct = None
                    # _cp_structured = ""
                    # if _cp_instr:
                    #     has_color   = bool(getattr(_cp_instr, 'color_grading', '').strip())
                    #     has_angle   = bool(getattr(_cp_instr, 'perspective_and_camera', '').strip())
                    #     has_style   = bool(getattr(_cp_instr, 'visual_style', '').strip())
                    #     has_subject = bool(getattr(_cp_instr, 'subject_focus', '').strip())
                    #     if has_color and not has_angle and not has_style:
                    #         _cp_class_from_struct = "color"
                    #         _cp_structured = "color palette should be different colour"
                    #     elif has_angle or has_style:
                    #         _cp_class_from_struct = "structural"
                    #         _cp_structured = ("perspective angle should change " if has_angle else "") + ("visual style should change" if has_style else "")
                    #     elif has_subject:
                    #         _cp_structured = "subject_focus transformation requested"
                    # _cp_feedback = _cp_structured + " " + (getattr(copyright_result, 'analysis', '') or '')
                    # if _pending_copyright_sid:
                    #     tried_strategy_ids.append(_pending_copyright_sid)
                    #     _pending_copyright_sid = ""
                    # cp_condition = StrategyCondition(
                    #     is_copyright_round=True, consecutive_failures=consecutive_copyright_failures,
                    #     stale_similarity_rounds=stale_similarity_rounds, tried_strategy_ids=tried_strategy_ids,
                    #     regression_detected=False, last_feedback=_cp_feedback.strip()
                    # )
                    # _cp_strategy = select_strategy(cp_condition)
                    # if _cp_strategy:
                    #     _directive = apply_strategy(_cp_strategy, is_copyright_round=True)
                    #     if _directive:
                    #         _pending_copyright_sid        = _cp_strategy.id
                    #         _copyright_strategy_directive = _directive
                    #         print(f"   Copyright strategy '{_cp_strategy.name}' -> next copyright review.")

                    review_result = copyright_result 
                    is_copyright_round = True
                    
                    # Store copyright instructions for next iteration
                    if hasattr(copyright_result, 'recommended_instructions'):
                        previous_copyright_instructions.append(copyright_result.recommended_instructions)
                    
                    # Track what transformations are being applied for accuracy reviewer awareness
                    instr = getattr(copyright_result, 'recommended_instructions', None) or getattr(copyright_result, 'transformation_instructions', None)
                    if instr:
                        for field in ['perspective_and_camera', 'visual_style', 'subject_focus']:
                            val = getattr(instr, field, '')
                            if val and val.strip():
                                # Store transformation summary for accuracy reviewer
                                copyright_transformations_applied.append(f"{field.replace('_', ' ').title()}: {val[:80]}...")
                    
                    print(f"❌ Copyright Issues Found. Applying transformation strategy...")
            else:
                # ROLLBACK: Check if the voiceover reviewer explicitly flagged
                # regression via the regression_detected field.  Only act when
                # there is actually a previous image to revert to.
                is_regression = getattr(review_result, 'regression_detected', False)
                if isinstance(review_result, dict):
                    is_regression = review_result.get('regression_detected', False)
                
                if (is_regression
                        and round_num > 1
                        and image_stack
                        and rollback_count < MAX_ROLLBACKS):
                    analysis_text = getattr(review_result, 'analysis', '') or ''
                    edit_label = (last_edit_type or 'unknown').title()
                    print(f"\n🔙 ROLLBACK: Voiceover reviewer flagged regression after {edit_label} edit")
                    print(f"   Analysis: {analysis_text[:120]}")
                    current_image = image_stack.pop()
                    rollback_count += 1
                    failed_edits.append({
                        "type": last_edit_type or 'unknown',
                        "instructions": last_edit_instructions,
                        "reason": f"Regression detected: {analysis_text[:150]}"
                    })
                    last_edit_type = None
                    # [STRATEGIES DISABLED] — no strategy state to clear
                    _accuracy_strategy_directive = ""
                    # _pending_accuracy_sid        = ""   # [STRATEGIES DISABLED]
                    print(f"   Rollback #{rollback_count}/{MAX_ROLLBACKS}. Will re-assess.")
                    round_num += 1
                    continue

                is_copyright_round = False
                consecutive_accuracy_failures += 1
                consecutive_copyright_failures = 0

                # [STRATEGIES DISABLED] — strategy selection and shift logic commented out.
                # To re-enable: uncomment the block below and restore strategy state vars above.
                # _active_accuracy_sid = _pending_accuracy_sid
                # if _pending_accuracy_sid:
                #     tried_strategy_ids.append(_pending_accuracy_sid)
                #     _pending_accuracy_sid = ""
                # acc_condition = StrategyCondition(
                #     is_copyright_round      = False,
                #     consecutive_failures    = consecutive_accuracy_failures,
                #     stale_similarity_rounds = 0,
                #     tried_strategy_ids      = tried_strategy_ids,
                #     regression_detected     = getattr(review_result, 'regression_detected', False),
                #     last_feedback           = getattr(review_result, 'analysis', '') or ''
                # )
                # _acc_strategy = select_strategy(acc_condition)
                # if _acc_strategy:
                #     _directive = apply_strategy(_acc_strategy, is_copyright_round=False)
                #     if _directive:
                #         if (_acc_strategy.failure_class != _current_acc_failure_class
                #                 and _current_acc_failure_class != ""):
                #             accuracy_chat = _make_accuracy_chat()
                #         _current_acc_failure_class = _acc_strategy.failure_class
                #         _pending_accuracy_sid      = _acc_strategy.id
                #         _accuracy_strategy_directive = _directive
                #         print(f"   🔄 Accuracy strategy '{_acc_strategy.name}' → next review.")

                print(f"❌ Accuracy Issues Found. Applying technical corrections...")

            # E. If either says NO, edit and continue
            print(f"🔧 Applying {'Copyright' if is_copyright_round else 'Accuracy'} edits for Round {round_num}...")
            
            # Load image editor — use image_editing.py (Gemini-based editor)
            image_editing_path = os.path.join(agents_dir, 'image_editing', 'image_editing.py')
            if not os.path.exists(image_editing_path):
                # Fallback to top-level path if local copy somehow missing
                image_editing_path = os.path.join(os.path.dirname(agents_dir), 'agents', 'image_editing', 'image_editing.py')
            spec = importlib.util.spec_from_file_location("image_editing", image_editing_path)
            image_editing_module = importlib.util.module_from_spec(spec)
            # Patch out st.spinner so it doesn't crash in background worker threads
            import contextlib
            if not hasattr(image_editing_module, '_spinner_patched'):
                import types as _types
                import streamlit as _st
                _noop_ctx = contextlib.nullcontext
                image_editing_module.st = _st
                # Override spinner with a no-op so background threads don't crash
                _st_patch = _types.SimpleNamespace(**{k: getattr(_st, k) for k in dir(_st) if not k.startswith('__')})
                _st_patch.spinner = lambda *a, **kw: _noop_ctx()
                _st_patch.error = lambda *a, **kw: print(f"[image_editing] ERROR: {a}")
                image_editing_module.st = _st_patch
            spec.loader.exec_module(image_editing_module)
            image_editing_with_review_loop = image_editing_module.image_editing_with_review_loop
            
            # Handle both VoiceoverReviewResult and CopyrightReviewResult schemas
            # Extract ALL fields dynamically — TransformationInstructions has 7 fields,
            # RecommendedInstructions has 3. Hardcoding 3 would silently drop copyright fields.
            instr = getattr(review_result, 'recommended_instructions', None) or getattr(review_result, 'transformation_instructions', None)
            if instr is not None:
                # Pydantic v2: model_dump() is the safe way to get all field values
                raw = instr.model_dump() if hasattr(instr, 'model_dump') else vars(instr)
                instructions_dict = {k: (v if isinstance(v, str) else "") for k, v in raw.items()}
            else:
                instructions_dict = {}
            # Ensure perspective_and_camera key always present for two-pass logic
            instructions_dict.setdefault("perspective_and_camera", "")

            # [STRATEGIES DISABLED] — editor-side filter commented out.
            # Full reviewer output goes to editor unchanged.
            # if not is_copyright_round and _active_accuracy_sid:
            #     instructions_dict = _apply_strategy_filter(
            #         instructions_dict, _active_accuracy_sid
            #     )
            #     _active_accuracy_sid = ""  # consumed — reset for next round
            
            # Build context instruction (ATM or Surgical)
            if is_copyright_round:
                last_action_summary = f"Copyright Reviewer requested changes: {instructions_dict}"
                
                # Provide convergence feedback to editor for smarter transformations
                convergence_hint = ""
                if len(similarity_score_history) >= 2:
                    recent_scores = similarity_score_history[-3:] if len(similarity_score_history) >= 3 else similarity_score_history[-2:]
                    is_improving = recent_scores[-1] < recent_scores[0]
                    trend = "improving" if is_improving else "not improving"
                    convergence_hint = f"""

CONVERGENCE STATUS: Similarity is {trend} (recent: {', '.join(f'{s:.1f}%' for s in recent_scores)}).
Strategy: {'Continue current approach - it is working' if is_improving else 'Try complementary transformation (if angle changed, adjust style; if style changed, adjust composition)'}"""
                
                # Truncate analysis to fit Grok's 8000-char prompt limit
                _desc = review_result.description[:400] if review_result.description else ""
                _analysis = review_result.analysis[:800] if review_result.analysis else ""
                context_instruction = f"""
Transform this reference image into a High-Fidelity Technical Asset.

[IMAGE ANALYSIS & CONTEXT]
- CURRENT IMAGE DESCRIPTION: {_desc}
- COPYRIGHT AUDIT ANALYSIS: {_analysis}
- SLIDE CONTEXT: {slide_title} | {voiceover}
- EDUCATIONAL REQUIREMENT: {visual_instruction}{convergence_hint}

OBJECTIVE: Create a legally safe, professional technical visual that:
1. Retains technical accuracy for educational purposes (preserve functional elements)
2. Migrates design to copyright-compliant style
3. Preserves critical components needed for learning

CRITICAL: Maintain educational integrity while transforming visual appearance.
"""
            else:
                last_action_summary = f"Voiceover Reviewer requested changes: {instructions_dict}"
                
                # Warn editor about copyright constraints that must be preserved
                copyright_warning = ""
                if copyright_transformations_applied:
                    copyright_warning = f"""

COPYRIGHT CONSTRAINTS (DO NOT UNDO):
The following transformations are legally required and must be preserved:
{chr(10).join(f'- {t}' for t in copyright_transformations_applied[-3:])}

Fix technical accuracy WITHOUT undoing these copyright transformations.
Example: If gauge needs correction, fix the gauge WITHOUT adding brand logos.
"""
                
                # Truncate analysis to fit Grok's 8000-char prompt limit
                _desc = review_result.description[:400] if review_result.description else ""
                _analysis = review_result.analysis[:800] if review_result.analysis else ""
                context_instruction = f"""
Maintain the overall visual theme and artistic essence. Fix the technical errors surgically.

[IMAGE ANALYSIS & CONTEXT]
- CURRENT IMAGE DESCRIPTION: {_desc}
- ACCURACY ANALYSIS: {_analysis}
- SLIDE CONTEXT: {slide_title} | {voiceover}{copyright_warning}

STRICT RULE: Do not regenerate the entire scene. Only surgically adjust the components or functional states requested in the Analysis.
"""
            
            # Save current image for potential rollback before editing
            image_stack.append(current_image)
            if len(image_stack) > MAX_ROLLBACKS + 1:
                image_stack.pop(0)  # Cap memory: keep only recent snapshots
            last_edit_instructions = instructions_dict.copy()
            last_edit_type = "copyright" if is_copyright_round else "accuracy"
            
            # ── Two-pass Grok editing ────────────────────────────────────────
            # Pass 1: Perspective/camera only (structural changes first)
            # Pass 2: All remaining instructions on the result of Pass 1
            # This prevents Grok from being overloaded with mixed instructions.

            has_perspective = bool((instructions_dict.get("perspective_and_camera") or "").strip())
            pass1_dict = {"perspective_and_camera": instructions_dict.get("perspective_and_camera", "")}
            pass2_dict = {k: v for k, v in instructions_dict.items() if k != "perspective_and_camera"}
            has_pass2 = any(v.strip() for v in pass2_dict.values())

            editing_image = current_image  # working image passed through passes

            if has_perspective:
                print(f"  🔧 Pass 1 (Perspective/Camera)...")
                _pass1_raw = call_llm_with_retry(
                    image_editing_with_review_loop,
                    reference_image=editing_image,
                    editing_instructions=pass1_dict,
                    quick_mode=True,
                    image_size=image_size,
                    aspect_ratio=PIPELINE_ASPECT_RATIO,
                    system_instruction=context_instruction,
                    # image_editing_with_review_loop already has internal timeout/retry.
                    # Keep outer retry to 1 to avoid multiplicative latency.
                    max_retries=1
                )
                pass1_result = _pass1_raw[0] if _pass1_raw is not None else None
                if not pass1_result:
                    print("❌ Pass 1 (perspective) failed after retries. Exiting loop.")
                    image_stack.pop()
                    return _finalize_before_return(last_review_result, current_image, history, "edit-pass1-failed")
                editing_image = pass1_result
                print(f"  ✅ Pass 1 complete.")

            if has_pass2:
                p_num = '2' if has_perspective else '1'
                print(f"  🔧 Pass {p_num} (Subject/Style/Comments)...")
                _pass2_raw = call_llm_with_retry(
                    image_editing_with_review_loop,
                    reference_image=editing_image,
                    editing_instructions=pass2_dict,
                    quick_mode=True,
                    image_size=image_size,
                    aspect_ratio=PIPELINE_ASPECT_RATIO,
                    system_instruction=context_instruction,
                    # image_editing_with_review_loop already has internal timeout/retry.
                    # Keep outer retry to 1 to avoid multiplicative latency.
                    max_retries=1
                )
                pass2_result = _pass2_raw[0] if _pass2_raw is not None else None
                if not pass2_result:
                    # If pass 2 fails but pass 1 succeeded, keep pass 1 result
                    if has_perspective:
                        print(f"⚠️ Pass 2 failed after retries. Keeping Pass 1 result.")
                    else:
                        print("❌ Image editing failed after retries. Exiting loop.")
                        image_stack.pop()
                        return _finalize_before_return(last_review_result, current_image, history, "edit-pass2-failed")
                else:
                    editing_image = pass2_result
                    print(f"  ✅ Pass {p_num} complete.")

            edited_image = editing_image

            if not edited_image or edited_image is current_image:
                print("❌ Image editing produced no result. Exiting loop.")
                image_stack.pop()
                return _finalize_before_return(last_review_result, current_image, history, "edit-produced-no-result")

            current_image = edited_image
            round_num += 1
            
            # CONSECUTIVE FAILURES - Stop if persistent rejection pattern detected
            # Scan the entire history per agent type — VO and CP reviews don't
            # interleave so a sliding window would miss entries of one type.
            recent_vo_reviews = [h for h in history if h['agent'] == 'Voiceover Reviewer'][-5:]
            if len(recent_vo_reviews) >= 5:
                if all(str(h['review'].verdict).strip().lower() == 'no' for h in recent_vo_reviews):
                    print(f"\n🛑 STOPPING: 5 consecutive accuracy failures detected")
                    print(f"   The voiceover reviewer rejected the last 5 versions in a row.")
                    print(f"   This suggests fundamental incompatibility between the image and requirements.")
                    return _finalize_before_return(last_review_result, current_image, history, "five-consecutive-accuracy-failures")

            recent_cp_reviews = [h for h in history if h['agent'] == 'Copyright Reviewer'][-5:]
            if len(recent_cp_reviews) >= 5:
                if all(str(h['review'].verdict).strip().lower() == 'no' for h in recent_cp_reviews):
                    print(f"\n🛑 STOPPING: 5 consecutive copyright failures detected")
                    print(f"   The copyright reviewer rejected the last 5 versions in a row.")
                    print(f"   This suggests the image cannot be sufficiently transformed for copyright compliance.")
                    return _finalize_before_return(last_review_result, current_image, history, "five-consecutive-copyright-failures")
            
        except Exception as e:
            print(f"Error in Round {round_num}: {e}")
            traceback.print_exc()
            if last_review_result:
                return _finalize_before_return(last_review_result, current_image, history, "round-exception-with-last-review")
            raise e

# =============================================================================
# HELPER FUNCTIONS FOR DRIVE MANAGEMENT
# =============================================================================

def _create_drive_subfolder(drive, parent_folder_id: str, folder_name: str, drive_lock=None) -> str:
    """Create a subfolder in Drive and return its ID."""
    folder_metadata = {
        'title': folder_name,
        'parents': [{'id': parent_folder_id}],
        'mimeType': 'application/vnd.google-apps.folder'
    }
    def _upload():
        folder = drive.CreateFile(folder_metadata)
        folder.Upload()
        return folder['id']

    if drive_lock:
        with drive_lock:
            return _upload()
    return _upload()

def _get_drive_folder_link(folder_id: str) -> str:
    """Generate shareable link for a Drive folder."""
    return f"https://drive.google.com/drive/folders/{folder_id}"

def _save_image_to_drive(image: Image.Image, filename: str, drive, folder_id: str, drive_lock=None) -> str:
    """Save image to Drive folder and return a shareable view link."""
    from agents.graphics_asset_creation.gac_utils import upload_image_to_drive
    if drive_lock:
        with drive_lock:
            return upload_image_to_drive(image, filename, drive, folder_id)
    return upload_image_to_drive(image, filename, drive, folder_id)

# =============================================================================
# ROW PROCESSING LOGIC
# =============================================================================

def _rebuild_final_graphics_text(
    original_text: str,
    subseg_drive_links: dict
) -> str:
    """
    Rebuild the final_graphics_definition text by replacing only the
    'Graphics to use:' link in each subsegment with the transformed Drive
    link (if one was produced).  Everything else — SEGMENT headers, ----
    separators, When VO, Visual Instructions, Selection Justification —
    is kept byte-for-byte from the original.

    Args:
        original_text: The raw final_graphics_definition text from the source sheet.
        subseg_drive_links: Dict mapping 0-based subsegment index -> Drive thumbnail URL
                            (or None/empty-string if that subseg was skipped).
    Returns:
        Reconstructed text block ready to write to the `final_graphics` column.
    """
    if not original_text:
        return original_text

    # ── Split into SEGMENT blocks ────────────────────────────────────────
    # Each SEGMENT header looks like:
    #   ===============...\nSEGMENT N\n===============...
    seg_header_pat = re.compile(
        r'(={50,}[\r\n]+SEGMENT\s+\d+[\r\n]+={50,})',
        re.IGNORECASE
    )
    # interleave: [pre, hdr1, body1, hdr2, body2, ...]
    parts = seg_header_pat.split(original_text)

    # Flatten: if no SEGMENT headers, treat whole thing as one body.
    if len(parts) == 1:
        # No SEGMENT headers — plain list of ---- separated subsegments
        bodies_with_headers = [('', original_text)]
    else:
        # parts[0] is text before first SEGMENT header (usually empty)
        # then alternating: header, body
        bodies_with_headers = []
        i = 1
        while i < len(parts):
            hdr  = parts[i]     if i   < len(parts) else ''
            body = parts[i+1]   if i+1 < len(parts) else ''
            bodies_with_headers.append((hdr, body))
            i += 2

    global_sub_idx = 0
    output_segments = []

    if parts[0].strip():
        output_segments.append(parts[0])  # text before first SEGMENT header

    for seg_hdr, seg_body in bodies_with_headers:
        # Split by ---- inside this segment body
        raw_subsegments = seg_body.split('----')

        rebuilt_subsegments = []
        for raw_sub in raw_subsegments:
            if not raw_sub.strip():
                rebuilt_subsegments.append(raw_sub)
                continue

            drive_link = subseg_drive_links.get(global_sub_idx)
            global_sub_idx += 1

            if drive_link:
                # Replace the 'Graphics to use:' line.
                # Add a trailing \n after the new link so the next field
                # (e.g. 'When VO starts:') is always separated by a blank line.
                raw_sub = re.sub(
                    r'(Graphics to use:\s*).+?(?:\s*\(snapshot\))?(?=\n|$)',
                    lambda m: m.group(1) + drive_link + '\n',
                    raw_sub,
                    count=1,
                    flags=re.IGNORECASE
                )
                # Remove a standalone (snapshot) line that may follow
                raw_sub = re.sub(
                    r'^\s*\(snapshot\)\s*$', '',
                    raw_sub,
                    flags=re.IGNORECASE | re.MULTILINE
                )

            # Strip the Selection Justification block, then normalize
            # trailing whitespace to exactly one trailing newline.
            raw_sub = re.sub(
                r'Selection Justification:.*?(?=\n\n|\Z)',
                '',
                raw_sub,
                flags=re.IGNORECASE | re.DOTALL
            ).rstrip('\n').rstrip() + '\n'

            rebuilt_subsegments.append(raw_sub)

        rebuilt_body = '----'.join(rebuilt_subsegments)
        output_segments.append(seg_hdr + rebuilt_body)

    return ''.join(output_segments)


def _process_single_reviewer_row(
    row_data: dict,
    drive,
    output_folder_id: str,
    ws_source=None,           # source worksheet for writing back
    source_col_idx: int = None,   # 1-based column index of 'final_graphics'
    sheet_row_number: int = None, # 1-based sheet row of this data row (header=1)
    sheet_lock=None,          # threading.Lock protecting gspread writes
    drive_lock=None,          # threading.Lock protecting pydrive2 writes
    subseg_workers: int = 1,  # parallel subsegment workers: 1=sequential, -1=auto (one per subsegment), >1=fixed count
    max_retries: int = 4
) -> dict:
    """
    Process a single row using the orchestrated review workflow (Accuracy + Copyright).
    For each subsegment with a valid link:
      - Download the reference image
      - Run the full review+edit pipeline
      - Upload the final image to Drive
    Then rebuilds the final_graphics_definition text with Drive links substituted
    in place of the original 'Graphics to use:' links, and writes it back to the
    source sheet in the 'final_graphics' column.  Intermediate images AND the final
    image are all saved inside per-subsegment Drive subfolders.

    Args:
        subseg_workers: Controls parallel subsegment processing within this row.
                        1  = sequential (default, safest for API rate limits).
                        -1 = auto — number of workers equals the number of valid subsegments
                             detected at runtime (one worker per subsegment).
                        >1 = fixed parallel count (capped at actual subsegment count).
    """
    from agents.graphics_asset_creation.gac_utils import (
        parse_subsegments, is_valid_web_image_link
    )
    import threading

    idx             = row_data['idx']
    slide_title     = row_data['slide_title']
    slide_content   = row_data['slide_content']
    final_graphics_def = row_data['final_graphics_def']

    print(f"\n[Worker {threading.current_thread().name}] Processing Row {idx + 1}: {slide_title}")

    # Create a row-level Drive folder
    row_folder_name = f"Row_{idx + 1}_{slide_title[:30].replace('/', '_')}"
    row_folder_id   = _create_drive_subfolder(drive, output_folder_id, row_folder_name, drive_lock=drive_lock)
    print(f"Created Row folder: {row_folder_name}")

    # Parse sub-segments
    subsegments = parse_subsegments(final_graphics_def)
    print(f"[Worker {threading.current_thread().name}] Found {len(subsegments)} sub-segment(s)")

    # subseg_drive_links[i] = Drive thumbnail URL (or None if skipped)
    subseg_drive_links: dict = {}

    if not subsegments:
        gfx_preview = str(final_graphics_def)[:150].replace('\n', ' ') if final_graphics_def else '(EMPTY)'
        print(f"[Worker {threading.current_thread().name}] ⚠️ No subsegments parsed. Preview: {gfx_preview}")
        return {'status': 'no_subsegments', 'drive_links': {}}

    # ── Pre-create all Drive subfolders sequentially (Drive API is serial) ──
    # Build a work list of valid subsegments first
    valid_subseg_work = []
    for sub_idx, subseg in enumerate(subsegments):
        ref_link = subseg.get('reference_link', '')
        if not is_valid_web_image_link(ref_link):
            print(f"  Sub {sub_idx + 1}: Skipping invalid link: {ref_link[:80]}")
            subseg_drive_links[sub_idx] = None
            continue
        plain_link = re.sub(r'\s*\(snapshot\)\s*', '', ref_link, flags=re.IGNORECASE).strip()
        subseg_folder_name = f"Subsegment_{sub_idx + 1}"
        subseg_folder_id   = _create_drive_subfolder(drive, row_folder_id, subseg_folder_name, drive_lock=drive_lock)
        print(f"  Created Subsegment folder: {subseg_folder_name}")
        valid_subseg_work.append((sub_idx, subseg, plain_link, subseg_folder_id))

    def _process_one_subseg(work_item):
        """Process a single subsegment — used by both sequential and parallel paths."""
        sub_idx, subseg, plain_link, subseg_folder_id = work_item
        final_drive_link = None
        reviewer_counters = {'voiceover': 0, 'copyright': 0}

        for retry in range(max_retries + 1):
            try:
                if retry > 0:
                    time.sleep(retry * 2)
                    print(f"  Retry {retry}/{max_retries} for sub {sub_idx + 1}...")

                # 1. Load image
                if drive_lock:
                    with drive_lock:
                        ref_image = _download_drive_image(plain_link, drive=drive)
                else:
                    ref_image = _download_drive_image(plain_link, drive=drive)

                # 2. Callback: save every intermediate review round
                def save_intermediate_callback(round_data: Dict[str, Any],
                                               _folder_id=subseg_folder_id,
                                               _counters=reviewer_counters):
                    agent_type = round_data['agent']
                    image      = round_data['image']
                    if 'Voiceover' in agent_type:
                        _counters['voiceover'] += 1
                        fname = f"Reviewer_1_Round_{_counters['voiceover']}.png"
                    else:
                        _counters['copyright'] += 1
                        fname = f"Reviewer_2_Round_{_counters['copyright']}.png"
                    try:
                        _save_image_to_drive(image, fname, drive, _folder_id, drive_lock=drive_lock)
                        print(f"    💾 Saved: {fname}")
                    except Exception as cb_err:
                        print(f"    ⚠️ Failed to save {fname}: {cb_err}")

                # 3. Run the full review pipeline
                final_review, final_image, history = review_and_edit_image(
                    ref_image,
                    slide_title=slide_title,
                    slide_content=slide_content,
                    voiceover=subseg['voiceover_focus'],
                    visual_instruction=subseg['visual_instruction'],
                    image_size="1K",
                    target_stage="full",
                    callback=save_intermediate_callback
                )

                # 4. Upload final image
                if final_image:
                    upload_link = _save_image_to_drive(
                        final_image, "FINAL_Image.png", drive, subseg_folder_id, drive_lock=drive_lock
                    )
                    if upload_link:
                        final_drive_link = upload_link
                        print(f"  ✅ Sub {sub_idx + 1}: Final uploaded → {upload_link}")
                    else:
                        print(f"  ⚠️ Sub {sub_idx + 1}: Upload returned empty link")
                else:
                    print(f"  ⚠️ Sub {sub_idx + 1}: No final image generated")

                break  # success — exit retry loop

            except Exception as e:
                if retry == max_retries:
                    print(f"  ❌ Sub {sub_idx + 1} failed after {max_retries} retries: {e}")
                    traceback.print_exc()

        return sub_idx, final_drive_link

    # ── Dispatch subsegments — parallel or sequential ────────────────────
    # Determine effective worker count:
    #   subseg_workers == -1  → auto: one worker per valid subsegment
    #   subseg_workers >  1   → fixed count, capped at actual subsegment count
    #   subseg_workers <= 1   → sequential
    n_valid = len(valid_subseg_work)
    if subseg_workers == -1:
        # AUTO MODE: number of workers == number of subsegments to process
        eff_workers = n_valid
    elif subseg_workers > 1:
        eff_workers = min(subseg_workers, n_valid)
    else:
        eff_workers = 1

    if eff_workers > 1 and n_valid > 1:
        from concurrent.futures import ThreadPoolExecutor as _SubPool, as_completed as _as_completed
        print(f"  🔀 Processing {n_valid} subsegments with {eff_workers} parallel worker(s) "
              f"({'auto' if subseg_workers == -1 else 'fixed'} mode)")
        with _SubPool(max_workers=eff_workers, thread_name_prefix="SubsegWorker") as sub_pool:
            futures = {sub_pool.submit(_process_one_subseg, wi): wi for wi in valid_subseg_work}
            for fut in _as_completed(futures):
                try:
                    s_idx, link = fut.result()
                    subseg_drive_links[s_idx] = link
                except Exception as fut_err:
                    wi = futures[fut]
                    print(f"  ❌ Subseg {wi[0]+1} future failed: {fut_err}")
                    subseg_drive_links[wi[0]] = None
    else:
        # Sequential (default or only 1 subsegment)
        for work_item in valid_subseg_work:
            s_idx, link = _process_one_subseg(work_item)
            subseg_drive_links[s_idx] = link

    # ── Rebuild the original text with Drive links substituted in ────────
    rebuilt_text = _rebuild_final_graphics_text(final_graphics_def, subseg_drive_links)

    # ── Write back to source sheet immediately ───────────────────────────
    if ws_source and source_col_idx is not None and sheet_row_number is not None:
        try:
            from gspread.utils import rowcol_to_a1
            cell_addr = rowcol_to_a1(sheet_row_number, source_col_idx)
            if sheet_lock:
                with sheet_lock:
                    ws_source.update(range_name=cell_addr, values=[[rebuilt_text]])
            else:
                ws_source.update(range_name=cell_addr, values=[[rebuilt_text]])
            print(f"  ✏️  Written 'final_graphics' for row {sheet_row_number} → cell {cell_addr}")
        except Exception as sheet_err:
            print(f"  ❌ Failed to write final_graphics to sheet: {sheet_err}")
            traceback.print_exc()

    return {
        'status': 'done',
        'drive_links': subseg_drive_links,
        'rebuilt_text': rebuilt_text,
    }

# =============================================================================
# BATCH EXECUTION
# =============================================================================

def run_automation(
    sheet_url: str,
    source_tab: str,
    output_tab: str = "",
    output_folder_name: str = "Voiceover Reviewer Automation",
    max_workers: int = None,
    subseg_workers: int = None,
    gc=None,
    drive=None,
    progress_callback=None,   # Optional[Callable[[int, int], None]]
    skip_filled_rows: bool = False,
    input_column_name: str = "final_graphics_definition",
    output_column_name: str = "final_graphics_definition",
    write_final_graphics: bool = True,
):
    """
    Batch-process the entire source sheet with FULL parallelism.

    Architecture (3 phases — worker count is always automatic):
      Phase 1  [Serial]   — Parse every row, identify every valid subsegment,
                            pre-create ALL Drive folders (Drive API is not thread-safe).
                            Counts the total number of subsegments N.
      Phase 2  [Parallel] — Spin up exactly N workers, one per subsegment.
                            Every subsegment across every row runs simultaneously.
                            Each row is written to the sheet AS SOON AS its last
                            subsegment completes — no waiting for other rows.
      Phase 3  [Cleanup]  — Handle any rows that had only skipped subsegments
                            (invalid links) and were not written during Phase 2.

    Args:
        progress_callback: Optional callable invoked after each subsegment completes.
                           Signature: progress_callback(completed: int, total: int)
                           useful for driving a UI progress bar.
        skip_filled_rows: Whether to skip rows that already have content in the output column (`output_column_name`).
        write_final_graphics: Whether to create/update the legacy `final_graphics` column.
                              When False, this pipeline writes only `output_column_name`.
        max_workers / subseg_workers: Accepted for backward-compatibility only;
                                     silently ignored.
    """
    import threading
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pydrive2.drive import GoogleDrive
    from gspread.utils import rowcol_to_a1
    from agents.graphics_asset_creation.gac_utils import (
        get_or_create_drive_folder, parse_subsegments, is_valid_web_image_link
    )
    from services.drive_service import login_with_service_account
    from services.sheets_service import get_sheet_data_and_df

    # Reset tracker so each run starts with fresh stats
    llm_tracker.reset()

    # Thread-safety locks
    sheet_lock = threading.Lock()   # gspread is NOT thread-safe
    drive_lock = threading.Lock()   # pydrive2 is NOT thread-safe

    try:
        if gc is None or drive is None:
            service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
            if not service_account_json:
                raise ValueError("Missing GOOGLE_SERVICE_ACCOUNT_JSON")
            gauth = login_with_service_account(json_str=service_account_json)
            drive = GoogleDrive(gauth)
            gc = gspread.service_account_from_dict(json.loads(service_account_json))

        # ── Phase 1: Setup + serial scan of all rows ───────────────────────────
        output_folder_id = get_or_create_drive_folder(drive, output_folder_name)

        sheet         = gc.open_by_url(sheet_url)
        ws_source, df = get_sheet_data_and_df(sheet, source_tab)

        print(f"\n📊 Source tab '{source_tab}' has {len(df)} rows")
        print(f"📊 Source columns: {list(df.columns)}")
        if len(df) > 0:
            print(f"📊 First row sample: {dict(df.iloc[0])}")

        # Ensure output columns exist
        source_headers     = ws_source.row_values(1)
        FINAL_GRAPHICS_COL = 'final_graphics'
        FINAL_GRAPHICS_DEFINITION_COL = output_column_name
        output_cols = [FINAL_GRAPHICS_DEFINITION_COL]
        if write_final_graphics:
            output_cols.insert(0, FINAL_GRAPHICS_COL)

        for col_name in output_cols:
            if col_name not in source_headers:
                next_col = len(source_headers) + 1
                with sheet_lock:
                    ws_source.update(
                        range_name=rowcol_to_a1(1, next_col),
                        values=[[col_name]]
                    )
                source_headers.append(col_name)
                print(f"📌 Added '{col_name}' column at position {next_col}")

        fg_def_col_idx = source_headers.index(FINAL_GRAPHICS_DEFINITION_COL) + 1   # 1-based
        fg_col_idx = None
        if write_final_graphics:
            fg_col_idx = source_headers.index(FINAL_GRAPHICS_COL) + 1   # 1-based

        if write_final_graphics:
            print(
                f"📌 Writing results to columns "
                f"'{FINAL_GRAPHICS_COL}' (col {fg_col_idx}) and "
                f"'{FINAL_GRAPHICS_DEFINITION_COL}' (col {fg_def_col_idx})"
            )
        else:
            print(
                f"📌 Writing results to column "
                f"'{FINAL_GRAPHICS_DEFINITION_COL}' (col {fg_def_col_idx}) only"
            )

        # Scan all rows → pre-create Drive folders → build flat global work list
        print("\n🔍 Phase 1: Scanning all rows and pre-creating Drive folders…")

        row_meta         = []        # one entry per sheet row
        global_work_items = []       # flat list: one entry per valid subsegment
        rows_written: set = set()   # df_idx values we already handled or skipped

        for df_idx, (_, row) in enumerate(df.iterrows()):
            # Resume/Completion check: Skip if row already has generated content
            if skip_filled_rows:
                existing_result = str(row.get(output_column_name, '')).strip()
                # Check for a Drive link as a proxy for 'successfully processed'
                if existing_result and "drive.google.com" in existing_result.lower():
                    print(
                        f"  ⏭️ Row {df_idx + 1}: Already has '{output_column_name}' content. Skipping."
                    )
                    rows_written.add(df_idx)
                    continue
            slide_title        = str(row.get('Slide Chunk Title', ''))
            slide_content      = str(row.get('Slide Chunk', ''))
            final_graphics_def = str(row.get(input_column_name, ''))
            sheet_row_number   = df_idx + 2   # +1 for 1-based, +1 for header row

            print(f"\n  Row {df_idx + 1}: {slide_title[:60]}")

            subsegments = parse_subsegments(final_graphics_def)
            print(f"    → {len(subsegments)} subsegment(s) parsed")

            meta = {
                'df_idx'            : df_idx,
                'sheet_row_number'  : sheet_row_number,
                'slide_title'       : slide_title,
                'final_graphics_def': final_graphics_def,
                'parsed_subsegments_count': len(subsegments),
                'sub_skipped'       : {},   # sub_idx -> None (invalid link)
            }
            row_meta.append(meta)

            if not subsegments:
                gfx_preview = final_graphics_def[:120].replace('\n', ' ') if final_graphics_def else '(EMPTY)'
                print(f"    ⚠️ No subsegments found. Preview: {gfx_preview}")
                continue

            # Create row-level Drive folder (SERIAL — Drive API is not thread-safe)
            row_folder_name = f"Row_{df_idx + 1}_{slide_title[:30].replace('/', '_')}"
            row_folder_id   = _create_drive_subfolder(drive, output_folder_id, row_folder_name, drive_lock=drive_lock)
            print(f"    📁 Created row folder: {row_folder_name}")

            for sub_idx, subseg in enumerate(subsegments):
                ref_link = subseg.get('reference_link', '')
                if not is_valid_web_image_link(ref_link):
                    print(f"    Sub {sub_idx + 1}: Skipping invalid link: {ref_link[:80]}")
                    meta['sub_skipped'][sub_idx] = None
                    continue

                plain_link = re.sub(r'\s*\(snapshot\)\s*', '', ref_link, flags=re.IGNORECASE).strip()

                # Create subsegment Drive folder (SERIAL)
                subseg_folder_name = f"Subsegment_{sub_idx + 1}"
                subseg_folder_id   = _create_drive_subfolder(
                    drive, row_folder_id, subseg_folder_name, drive_lock=drive_lock
                )
                print(f"    📁 Created subsegment folder: {subseg_folder_name}")

                global_work_items.append({
                    'df_idx'            : df_idx,
                    'sub_idx'           : sub_idx,
                    'slide_title'       : slide_title,
                    'slide_content'     : slide_content,
                    'voiceover_focus'   : subseg.get('voiceover_focus', ''),
                    'visual_instruction': subseg.get('visual_instruction', ''),
                    'plain_link'        : plain_link,
                    'subseg_folder_id'  : subseg_folder_id,
                })

        total_subsegments = len(global_work_items)
        # row_expected_counts[df_idx] = number of workers dispatched for that row
        row_expected_counts: dict = {}   # df_idx -> int
        for wi in global_work_items:
            row_expected_counts[wi['df_idx']] = row_expected_counts.get(wi['df_idx'], 0) + 1

        print(f"\n📊 Phase 1 complete.")
        print(f"   Rows scanned            : {len(df)}")
        print(f"   Total valid subsegments : {total_subsegments}")
        print(f"   Workers to launch       : {total_subsegments}  (1 worker per subsegment)")

        # Ensure UI progress bar updates even when there are 0 eligible subsegments.
        if progress_callback:
            try:
                progress_callback(0, total_subsegments)
            except Exception:
                pass

        if total_subsegments == 0:
            print("⚠️ No valid subsegments found across any row. Rows will be written immediately with original graphics text.")

        # results[df_idx][sub_idx] = final Drive link (or None on failure)
        results: dict = defaultdict(dict)

        def _process_one_global_subseg(work_item: dict):
            """
            Worker function — runs one subsegment end-to-end:
              1. Download reference image from Drive / web.
              2. Run full Accuracy + Copyright review+edit pipeline.
              3. Upload final approved image to the subsegment Drive folder.
            Returns (df_idx, sub_idx, drive_link_or_None).
            """
            df_idx            = work_item['df_idx']
            sub_idx           = work_item['sub_idx']
            slide_title_w     = work_item['slide_title']
            slide_content_w   = work_item['slide_content']
            plain_link        = work_item['plain_link']
            subseg_folder_id  = work_item['subseg_folder_id']
            voiceover_focus   = work_item['voiceover_focus']
            visual_instr      = work_item['visual_instruction']
            final_drive_link  = None
            reviewer_counters = {'voiceover': 0, 'copyright': 0}

            thread_name = threading.current_thread().name
            print(f"\n[{thread_name}] ▶ Row {df_idx + 1} / Sub {sub_idx + 1}: {slide_title_w[:50]}")

            for retry in range(3):
                try:
                    if retry > 0:
                        time.sleep(retry * 2)
                        print(f"  [{thread_name}] Retry {retry}/2 for Row {df_idx+1} Sub {sub_idx+1}…")

                    # 1. Download reference image
                    with drive_lock:
                        ref_image = _download_drive_image(plain_link, drive=drive)

                    # 2. Intermediate callback — saves each reviewer round to Drive
                    def save_intermediate_callback(
                        round_data: Dict[str, Any],
                        _folder_id=subseg_folder_id,
                        _counters=reviewer_counters
                    ):
                        agent_type = round_data['agent']
                        image      = round_data['image']
                        if 'Voiceover' in agent_type:
                            _counters['voiceover'] += 1
                            fname = f"Reviewer_1_Round_{_counters['voiceover']}.png"
                        else:
                            _counters['copyright'] += 1
                            fname = f"Reviewer_2_Round_{_counters['copyright']}.png"
                        try:
                            _save_image_to_drive(image, fname, drive, _folder_id, drive_lock=drive_lock)
                            print(f"    💾 [{thread_name}] Saved: {fname}")
                        except Exception as cb_err:
                            print(f"    ⚠️ [{thread_name}] Failed to save {fname}: {cb_err}")

                    # 3. Full review + edit pipeline
                    final_review, final_image, history = review_and_edit_image(
                        ref_image,
                        slide_title=slide_title_w,
                        slide_content=slide_content_w,
                        voiceover=voiceover_focus,
                        visual_instruction=visual_instr,
                        image_size="1K",
                        target_stage="full",
                        callback=save_intermediate_callback
                    )

                    # 4. Upload final approved image
                    if final_image:
                        upload_link = _save_image_to_drive(
                            final_image, "FINAL_Image.png", drive, subseg_folder_id, drive_lock=drive_lock
                        )
                        if upload_link:
                            final_drive_link = upload_link
                            print(f"  [{thread_name}] ✅ Row {df_idx+1} Sub {sub_idx+1}: uploaded → {upload_link}")
                        else:
                            print(f"  [{thread_name}] ⚠️ Row {df_idx+1} Sub {sub_idx+1}: upload returned empty link")
                    else:
                        print(f"  [{thread_name}] ⚠️ Row {df_idx+1} Sub {sub_idx+1}: no final image produced")

                    break   # success — exit retry loop

                except Exception as e:
                    if retry == 2:
                        print(f"  [{thread_name}] ❌ Row {df_idx+1} Sub {sub_idx+1} failed after 2 retries: {e}")
                        traceback.print_exc()

            return df_idx, sub_idx, final_drive_link

        # Shared per-row completion counter (main thread only — no locks needed)
        row_completed_counts: dict = {}   # df_idx -> completed_count

        def _write_row_to_sheet(meta: dict, df_idx: int) -> None:
            """
            Re-open ws, rebuild text, and write with up to 3 retries.
            Separated so it can be called both from the as_completed loop
            (incremental) and from Phase 3 cleanup.
            """
            nonlocal ws_source
            sheet_row_number   = meta['sheet_row_number']
            final_graphics_def = meta['final_graphics_def']

            row_subseg_links: dict = dict(meta['sub_skipped'])
            row_subseg_links.update(results.get(df_idx, {}))

            rebuilt_text = _rebuild_final_graphics_text(final_graphics_def, row_subseg_links)
            cell_addr_def = rowcol_to_a1(sheet_row_number, fg_def_col_idx)
            cell_addr_fg = rowcol_to_a1(sheet_row_number, fg_col_idx) if fg_col_idx is not None else None

            written = False
            for write_attempt in range(3):
                try:
                    if write_attempt > 0:
                        wait_s = 2 ** write_attempt   # 2 s, 4 s
                        print(f"  ⏳ Retry {write_attempt}/2 for sheet row {sheet_row_number} "
                              f"(waiting {wait_s}s)…")
                        time.sleep(wait_s)
                        try:
                            ws_source = sheet.worksheet(source_tab)
                        except Exception:
                            pass
                    with sheet_lock:
                        ws_source.update(range_name=cell_addr_def, values=[[rebuilt_text]])
                        if write_final_graphics and cell_addr_fg:
                            ws_source.update(range_name=cell_addr_fg, values=[[rebuilt_text]])

                    if write_final_graphics and cell_addr_fg:
                        print(
                            f"  ✏️  Written sheet row {sheet_row_number} → "
                            f"{cell_addr_def} and {cell_addr_fg}"
                        )
                    else:
                        print(
                            f"  ✏️  Written sheet row {sheet_row_number} → {cell_addr_def}"
                        )
                    written = True
                    break
                except Exception as sheet_err:
                    print(f"  ❌ Attempt {write_attempt+1}/3 failed row {sheet_row_number}: {sheet_err}")
                    traceback.print_exc()

            if not written:
                print(f"  🆘 ALL WRITE ATTEMPTS FAILED for sheet row {sheet_row_number}.")
                print(f"  🆘 Rebuilt text (copy manually):\n{rebuilt_text}")

        # ── Phase 2a: Immediate write for rows with NO valid subsegments ───────
        # These rows should not wait until end-of-pipeline cleanup.
        prewritten_rows = 0
        for meta in row_meta:
            df_idx = meta['df_idx']
            if df_idx in rows_written:
                continue
            if row_expected_counts.get(df_idx, 0) > 0:
                continue

            if meta.get('parsed_subsegments_count', 0) == 0:
                reason = "no subsegments parsed"
            else:
                reason = "all subsegments invalid/skipped"

            print(f"  📝 Row {df_idx + 1}: {reason} — writing to sheet immediately")
            _write_row_to_sheet(meta, df_idx)
            rows_written.add(df_idx)
            prewritten_rows += 1

        # ── Phase 2b: Launch exactly one thread per valid subsegment ───────────
        # Rows are written to the sheet the instant their LAST subsegment
        # completes — there is no global wait-for-all before writing.
        if total_subsegments > 0:
            print(f"\n� Phase 2: Launching {total_subsegments} parallel worker(s)…")

            # Re-open worksheet freshly before writes start
            try:
                ws_source = sheet.worksheet(source_tab)
            except Exception as ws_err:
                print(f"  ⚠️ Could not pre-refresh worksheet: {ws_err} — using original handle")

            with ThreadPoolExecutor(
                max_workers=total_subsegments,
                thread_name_prefix="SubWorker"
            ) as pool:
                futures = {
                    pool.submit(_process_one_global_subseg, wi): wi
                    for wi in global_work_items
                }
                completed = 0
                for future in as_completed(futures):
                    wi = futures[future]
                    try:
                        r_df_idx, r_sub_idx, link = future.result()
                        results[r_df_idx][r_sub_idx] = link
                    except Exception as fut_err:
                        r_df_idx = wi['df_idx']
                        r_sub_idx = wi['sub_idx']
                        results[r_df_idx][r_sub_idx] = None
                        print(f"\n  ❌ Row {r_df_idx+1} Sub {r_sub_idx+1} future raised: {fut_err}")

                    completed += 1
                    print(f"\n  ✔ [{completed}/{total_subsegments}] "
                          f"Row {r_df_idx+1} Sub {r_sub_idx+1} complete")

                    # Fire the progress callback for the UI progress bar
                    if progress_callback:
                        try:
                            progress_callback(completed, total_subsegments)
                        except Exception:
                            pass   # never let a UI callback crash the pipeline

                    # Increment per-row completion and write the row immediately
                    # when its last subsegment finishes.
                    row_completed_counts[r_df_idx] = row_completed_counts.get(r_df_idx, 0) + 1
                    if row_completed_counts[r_df_idx] == row_expected_counts.get(r_df_idx, 0):
                        meta_for_row = next(
                            (m for m in row_meta if m['df_idx'] == r_df_idx), None
                        )
                        if meta_for_row and r_df_idx not in rows_written:
                            rows_written.add(r_df_idx)
                            print(f"\n  📝 All subsegments for Row {r_df_idx+1} complete"
                                  f" — writing to sheet immediately…")
                            _write_row_to_sheet(meta_for_row, r_df_idx)

        # ── Phase 3: Cleanup — write rows that had ONLY skipped subsegments ──────
        # (Their workers were never dispatched in Phase 2, so they weren't written.)
        print(f"\n📝 Phase 3: Writing any remaining skipped-only rows…")
        skipped_written = 0
        for meta in row_meta:
            df_idx = meta['df_idx']
            if df_idx in rows_written:
                continue   # already written during Phase 2
            if not meta['sub_skipped']:
                # Row had no subsegments at all — nothing to write
                print(f"  Row {df_idx + 1}: no subsegments — skipped")
                continue
            # Row had subsegments but ALL were invalid links
            print(f"  Row {df_idx + 1}: all subsegments were skipped (invalid links) — writing original text")
            _write_row_to_sheet(meta, df_idx)
            skipped_written += 1

        print(f"\n🎉 Automation complete."
              f" Processed {total_subsegments} subsegment(s) across {len(df)} row(s)."
              f" ({prewritten_rows} no-valid-subsegment row(s) written immediately, "
              f"{skipped_written} row(s) written in Phase 3 cleanup)")

    except Exception as e:
        print(f"Automation Error: {e}")
        traceback.print_exc()
    finally:
        llm_tracker.mark_pipeline_end()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    args = parser.parse_args()
    run_automation(args.url, "Slide Chunksb", "Automated Review Results")


