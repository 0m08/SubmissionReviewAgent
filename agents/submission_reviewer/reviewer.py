import io
import os
import json
import base64
from typing import List, Optional
from PIL import Image

from openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from google import genai
from google.genai import types

from langsmith import traceable

from agents.submission_reviewer.schemas import (
    SubmissionReviewOutput,
    ChecklistResultItem,
)

# --------------------------------------------------------------------------- #
# Model Configuration
# --------------------------------------------------------------------------- #
# Primary: GPT-5.6 Luna with reasoning_effort=high
_PRIMARY_MODEL = "gpt-5.6-luna"
_PRIMARY_REASONING_EFFORT = "high"

# Fallback: Gemini 3.5 Flash (GA stable)
_FALLBACK_MODEL = "gemini-3.5-flash"


# --------------------------------------------------------------------------- #
# System Prompt — mentor-like reviewer, guardrails injected at call time
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# System Prompt — lightweight identity & output contract
# --------------------------------------------------------------------------- #
_BASE_SYSTEM_PROMPT = """You are an expert instructional reviewer and master technician acting as a supportive mentor for students in a vocational training program. Your role is to evaluate student activity submissions with empathy, technical accuracy, and instructional clarity, returning your review strictly in the required JSON schema."""

# --------------------------------------------------------------------------- #
# Dynamic User Prompt Blocks (Injected into user prompt per call)
# --------------------------------------------------------------------------- #
_REVIEW_GUIDELINES_BLOCK = """<mentorship_and_grading_rules>
1. MAJOR VS MINOR IMPACT DISTINCTION:
   - MAJOR CRITICAL FAILURES: Explicit "fail if" conditions triggered, core safety violations, total omission of the primary activity task, or blatantly incorrect work that disregards the main learning objective. Fail ONLY when a major critical point is violated.
   - MINOR IMPERFECTIONS: Small presentation flaws (lighting, formatting oversights, cosmetic imperfections). If the student accomplished the core task, DO NOT fail them for minor flaws — grant a "Pass" and provide gentle coaching feedback in your comment!
   - CAMERA ANGLE & FRAMING RULE:
     * IF <activity_instructions> explicitly specifies a camera angle, photo framing, or viewing distance (e.g. "Take a close-up photo of the low-pressure gauge"), you MUST evaluate it as a mandatory instruction requirement.
     * IF <activity_instructions> does NOT specify camera angle or framing, do NOT invent framing requirements or penalize camera angles.

2. EVIDENCE-BASED AUDITING:
   - Base your evaluation strictly on the submitted media (images/videos) and student comment. Do not assume or hallucinate unverified details.
   - If media is missing, blurry, cropped, or ambiguous, assign "Pass (Unsure)" or "Fail (Unsure)" depending on whether genuine effort on the primary task is visible.
   - A missing or brief student comment is normal — evaluate primarily on the visual media. Do not penalize a student for not writing a comment.

3. GRADING CONTRACT (4 QUALIFIED GRADES):
   - "Pass": Core activity goals are clearly accomplished. Minor presentation imperfections still earn a Pass with mentor advice.
   - "Fail": A major core requirement is clearly missing, incorrect work that disregards the main objective, OR an explicit "fail if" condition was triggered.
   - "Pass (Unsure)": Genuine effort is visible on the primary task, but visual evidence is ambiguous, blurry, cropped, or missing a final confirmation step.
   - "Fail (Unsure)": Major requirements appear missing or incomplete, but media is too blurry or cropped to be 100% certain.

4. MENTOR TONAL GUIDANCE:
   - Keep agent_comment concise (1–2 sentences), informal, conversational, and direct.
   - Name the specific thing that determined the grade. Praise what went well, tip what can improve. Avoid robotic or corporate phrasing.

5. STRICT ANTI-HEDGING RULE — "(Unsure)" IS A LAST RESORT, NOT A DEFAULT:
   ⚠️ You must NOT use "Pass (Unsure)" or "Fail (Unsure)" just because you are not 100% certain or want to hedge. Uncertainty alone is not sufficient justification.
   "(Unsure)" ONLY applies when ALL of the following are true simultaneously:
     [U-1] The specific piece of evidence needed to confirm or deny a CRITICAL requirement is COMPLETELY unreadable or absent from ALL submitted media (not just imperfect — truly illegible or non-existent).
     [U-2] You cannot make a reasonable inference from the rest of the visible media.
     [U-3] The issue directly affects the core pass/fail determination, not a minor sub-step.
   If you can make a reasonable determination from the available evidence — even with some imperfection — you MUST use a decisive "Pass" or "Fail".
   The vast majority of submissions (>85%) should receive a decisive "Pass" or "Fail". If you find yourself reaching for "(Unsure)" frequently, you are over-hedging — stop and make a decision.

6. ACCEPTABLE EDGE CASES & OVERRIDES (HIGHEST PRIORITY REVIEWER PRE-CONDITIONS):
   - BEFORE inspecting media or evaluating activity instructions, check <acceptable_edge_cases_and_overrides>.
   - These edge cases define manual reviewer policy rules dictating acceptable vs. unacceptable submission variations (e.g. "Photo of a portable unit is acceptable").
   - AUTHORITATIVE OVERRIDE PRINCIPLE: If an edge case explicitly permits a certain equipment type, photo variation, or setup, you MUST treat it as VALID and ACCEPTED. You are strictly forbidden from marking an item `followed=False` or assigning a `Fail` grade due to a variation that is explicitly permitted by the Edge Cases!

7. EXPLICIT INSTRUCTION REQUIREMENTS VS. COACHING TIPS (CRITICAL GRADING RULE):
   - EXPLICIT REQUIREMENTS ARE MANDATORY: If an item, photo, measurement, or visual proof is explicitly mandated in <activity_instructions>, omitting or failing it MUST result in `followed=False` and a `Fail` (or `Fail (Unsure)`).
   - NO "PASS THIS TIME" PASS-CARDS FOR MISSING MANDATORY REQUIREMENTS: You MUST NEVER grant a "Pass" while writing "Pass this time, but next time include X" when X was an explicit requirement in <activity_instructions>. If an explicitly requested submission item is missing, it is a Fail, NOT a coaching tip!
   - WHAT MENTOR COACHING TIPS ARE FOR: Comments and coaching tips are strictly for presentation feedback on COMPLETED items (e.g., lighting, slight positioning) or optional recommendations NOT explicitly mandated in <activity_instructions>.
</mentorship_and_grading_rules>"""


_LENIENT_MENTOR_DIRECTIVE_BLOCK = """<human_alignment_directive>
CRITICAL MENTORSHIP & SUBSTANTIAL EFFORT DIRECTIVE:
Human reviewers evaluate with supportive alignment while strictly enforcing explicit activity requirements.
- PASS WHEN ALL EXPLICIT REQUIREMENTS ARE MET: If the student fulfilled all explicitly mandated requirements in <activity_instructions> (or accepted edge cases) and completed the core physical task, grant a "Pass". Put presentation feedback into the mentor comment as friendly advice!
- FAIL FOR OMITTED EXPLICIT REQUIREMENTS: If an explicitly mandated photo, measurement, or core step in <activity_instructions> is missing or omitted, assign a "Fail". Never grant a "Pass this time" pass-card for an omitted explicit requirement.
</human_alignment_directive>"""


_ULTRA_LENIENT_DIRECTIVE_BLOCK = """<ultra_relaxed_mentor_directive>
ULTRA-RELAXED MODE — TWO-TIER EVALUATION FRAMEWORK:
Target: >80% Pass rate. Your job is to ACT AS A SUPPORTIVE MENTOR who passes genuine effort, while still flagging the small set of absolute no-go failures that cannot be overlooked.

TIER 1 — HARD NO-GO (ALWAYS FAIL, no exceptions):
Fail ONLY when ONE of the following is unambiguously true:
  [NO-GO-1] WRONG ACTIVITY / CORE CONCEPT SUBSTITUTION: Student demonstrably performed a completely different concept than instructed (e.g. asked to demonstrate Conduction but student showed only Convection). A minor conceptual gap is NOT this — the substitution must be obvious and total.
  [NO-GO-2] ZERO GENUINE ATTEMPT: No real media submitted, fake/stock/internet images used, or all submitted content is completely unrelated to the activity.
  [NO-GO-3] MISSING MANDATORY REQUIRED MEDIA OR STEP: Activity explicitly required N distinct photos/videos or a specific visual proof, and the student omitted a mandatory required photo/step without an allowed edge-case override.
  [NO-GO-4] SEVERE IMMEDIATE SAFETY VIOLATION: Student is shown actively performing a dangerous unsafe act (e.g. touching live electrical contacts, spinning blades with hands, bypassing safety equipment in a hazardous way).
  [NO-GO-5] BLATANTLY INCORRECT CORE WORK: The primary task was completed in a fundamentally wrong way that invalidates the main learning objective entirely (e.g. connections wired backwards causing system failure, not just a cosmetic deviation).

TIER 2 — DEFAULT TO PASS (everything that doesn't hit Tier 1):
If NONE of the Tier 1 conditions above are triggered, ALWAYS grant a "Pass" (or "Pass (Unsure)" if evidence is blurry/partially unverified) with mentor coaching notes. The following are NEVER grounds for a Fail in Ultra-Relaxed mode:
  - Dry-fit / uncemented connections when cementing was not the explicit focus
  - Photo framing or angle issues when framing was not explicitly mandated in activity instructions, poor lighting, or blurry but identifiable images on completed steps
  - Imperfect technique or suboptimal method (but not dangerously wrong)
  - Ambient conditions outside recommended range when student explains it

WHEN IN DOUBT → PASS (UNSURE): If you are unsure whether a step was verified or whether a Tier 1 condition is met, default to "Pass (Unsure)" with coaching notes.
</ultra_relaxed_mentor_directive>"""


_FEW_SHOT_EXAMPLES_BLOCK = """<reference_review_examples>
Use these real human reviewer decisions as reference for grading logic, mentor tone, and comment style:

  <example index="1" activity="Heat Movement" grade="FAIL" reason="Missing required media count">
    <student_comment>#1- Conduction... #2- Convection... #3- Radiation...</student_comment>
    <media_evidence>1 photo attached (instructions explicitly required 3 distinct photos for conduction, convection, and radiation)</media_evidence>
    <mentor_grade>Fail</mentor_grade>
    <mentor_comment>Great job on your explanations! However, you are missing 2 required photos for the other heat transfer types.</mentor_comment>
  </example>

  <example index="2" activity="Stable Temp Readings" grade="FAIL" reason="Omitted required question response">
    <student_comment>Photo 1 shows probe connected. Photo 2 shows probe on metal knob. Photo 3 on wood table. Photo 4 on vent...</student_comment>
    <media_evidence>4 clear photos attached</media_evidence>
    <mentor_grade>Fail</mentor_grade>
    <mentor_comment>Your photos look good, but you forgot to answer question #4 explaining why probe placement and wait time affect temperature accuracy.</mentor_comment>
  </example>

  <example index="3" activity="System Identification" grade="FAIL" reason="Omitted required classification name">
    <student_comment>- inside is air mover - outside is compressor and fan - two copper lines feeding inside unit - compressor outside</student_comment>
    <media_evidence>3 clear photos attached showing components</media_evidence>
    <mentor_grade>Fail</mentor_grade>
    <mentor_comment>You successfully described the components, but you didn't state what the system type is. This would be a Split system (specifically split system-Furnace).</mentor_comment>
  </example>

  <example index="4" activity="Line Temp to Pressure Pro" grade="PASS" reason="Core task complete despite minor ambient reading anomaly">
    <student_comment>Suction line was semi-cold due to sun exposure. Temperature stayed around 80-81°F giving ~237 psig per PT chart.</student_comment>
    <media_evidence>3 clear photos attached showing gauge and line setup</media_evidence>
    <mentor_grade>Pass</mentor_grade>
    <mentor_comment>At 81 degrees, that line is close to ambient temperature — double-check if the unit is actively running. Nice work taking readings!</mentor_comment>
  </example>

  <example index="5" activity="Any hands-on activity" grade="PASS (UNSURE)" reason="Core physical task clearly attempted with real equipment; one required verification step absent or unverifiable from media">
    <student_comment>I completed the activity and took photos of my setup.</student_comment>
    <media_evidence>Photos show real equipment and genuine physical effort on the primary task. One checklist item — a confirmation step or final-state photo — is absent or cannot be verified from the submitted media. There is no evidence the step was skipped unsafely; it is simply undocumented.</media_evidence>
    <mentor_grade>Pass (Unsure)</mentor_grade>
    <mentor_comment>Looks like you did the work — marking Pass (Unsure) because I can't fully verify one step from the photos. Add a quick note or final photo next time confirming that step and you'll be all set!</mentor_comment>
  </example>
</reference_review_examples>"""


def _build_system_prompt() -> str:
    """Returns system prompt containing persona identity, output schema contract, and reference review examples."""
    return f"{_BASE_SYSTEM_PROMPT}\n\n{_FEW_SHOT_EXAMPLES_BLOCK}"


def _pil_to_base64_url(img: Image.Image) -> str:
    """Convert PIL Image to base64 data URL for API messages."""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"


def _build_multimodal_messages(system_prompt: str, prompt_text: str, images: List[Image.Image]) -> list:
    """Build OpenAI-compatible messages with system prompt, text, and base64 images."""
    user_content = [{"type": "text", "text": prompt_text}]
    for img in images:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": _pil_to_base64_url(img)}
        })
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


# _parse_checklist removed — reviewer_checklist is deprecated as evaluation is done solely on activity_instructions.


@traceable(
    name="Submission Reviewer - GPT Luna Call",
    run_type="llm",
    metadata={"agent_name": "submission_reviewer", "model": _PRIMARY_MODEL}
)
def _call_gpt_luna(
    system_prompt: str,
    prompt_text: str,
    images: List[Image.Image],
) -> Optional[SubmissionReviewOutput]:
    """
    Primary reviewer: GPT-5.6 Luna via OpenAI beta.chat.completions.parse
    with reasoning_effort='high' and structured Pydantic output.
    """
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print(f"  [REVIEWER LOG] ⚠️ OPENAI_API_KEY not set — skipping {_PRIMARY_MODEL}.")
        return None

    print(f"  [REVIEWER LOG] Invoking {_PRIMARY_MODEL} (reasoning_effort={_PRIMARY_REASONING_EFFORT}, images={len(images)})...")
    client = OpenAI(api_key=key)
    messages = _build_multimodal_messages(system_prompt, prompt_text, images)

    completion = client.beta.chat.completions.parse(
        model=_PRIMARY_MODEL,
        messages=messages,
        response_format=SubmissionReviewOutput,
        reasoning_effort=_PRIMARY_REASONING_EFFORT,
    )

    parsed = completion.choices[0].message.parsed
    if parsed and isinstance(parsed, SubmissionReviewOutput):
        return parsed

    raw = completion.choices[0].message.content
    if raw:
        return SubmissionReviewOutput(**json.loads(raw))

    return None


@traceable(
    name="Submission Reviewer - Gemini Call",
    run_type="llm",
    metadata={"agent_name": "submission_reviewer", "model": _FALLBACK_MODEL}
)
def _call_gemini_reviewer(
    system_prompt: str,
    prompt_text: str,
    images: Optional[List[Image.Image]] = None,
    videos: Optional[List[dict]] = None,
    model_name: Optional[str] = None,
) -> Optional[SubmissionReviewOutput]:
    """
    Reviewer call using Gemini via google.genai SDK (preferred)
    or LangChain ChatGoogleGenerativeAI. Supports text, PIL images, and raw video bytes.
    """
    images = images or []
    videos = videos or []
    target_model = model_name or _FALLBACK_MODEL
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    # Attempt 1: google.genai SDK (native PIL image + video bytes support)
    if api_key:
        try:
            print(f"  [REVIEWER LOG] Invoking google.genai SDK ({target_model}) with {len(images)} image(s) and {len(videos)} video(s)...")
            client = genai.Client(api_key=api_key)
            contents = [prompt_text]
            contents.extend(images)

            for vid in videos:
                vid_bytes = vid.get("bytes")
                mime_type = vid.get("mime_type", "video/mp4")
                if vid_bytes:
                    contents.append(types.Part.from_bytes(data=vid_bytes, mime_type=mime_type))

            response = client.models.generate_content(
                model=target_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=SubmissionReviewOutput,
                    temperature=0.2,
                )
            )
            if response.parsed:
                return response.parsed
            elif response.text:
                return SubmissionReviewOutput(**json.loads(response.text))
        except Exception as e:
            print(f"  [REVIEWER LOG] google.genai error: {e}. Trying LangChain...")

    # Attempt 2: LangChain ChatGoogleGenerativeAI (fallback for image/text only)
    if not videos:
        try:
            print(f"  [REVIEWER LOG] Fallback: LangChain ChatGoogleGenerativeAI ({target_model})...")
            llm = ChatGoogleGenerativeAI(
                model=target_model,
                temperature=0.2,
                max_tokens=8192
            ).with_structured_output(SubmissionReviewOutput)

            user_content = [{"type": "text", "text": prompt_text}]
            for img in images:
                user_content.append({"type": "image_url", "image_url": {"url": _pil_to_base64_url(img)}})

            result = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content)
            ])
            if isinstance(result, SubmissionReviewOutput):
                return result
            elif isinstance(result, dict):
                return SubmissionReviewOutput(**result)
        except Exception as e:
            print(f"  [REVIEWER LOG] ❌ LangChain error: {e}")

    return None


@traceable(
    name="Submission Reviewer - Single Review",
    run_type="chain",
    metadata={
        "agent_name": "submission_reviewer",
        "step_name": "Review Single Submission",
        "function_name": "review_single_submission",
    }
)
def review_single_submission(
    activity_name: str,
    activity_instructions: str,
    reviewer_checklist: str = "",
    user_comment: str = "",
    images: Optional[List[Image.Image]] = None,
    videos: Optional[List[dict]] = None,
    guardrails: Optional[List[dict]] = None,
    activity_edge_cases: str = "",
    activity_guidelines: str = "",
    lenient_mode: Optional[bool] = None,
    ultra_lenient_mode: Optional[bool] = None,
    primary_model_choice: Optional[str] = None,
) -> SubmissionReviewOutput:
    """
    Evaluates a single user submission.
    - If submission contains VIDEO files, Gemini 3.5 Flash is used as the PRIMARY model
      (since Gemini natively processes video files).
    - If submission contains ONLY IMAGES, GPT-5.6 Luna is used as primary with reasoning_effort='high',
      falling back to Gemini 3.5 Flash.

    Args:
        guardrails: List of dicts with 'name' and 'description' keys, fetched from
                    the 'Guardrails' tab in the Google Sheet.
        activity_edge_cases: Text containing specific edge-case rules on acceptable/unacceptable
                             submission variations (e.g., 'Photo of a portable unit is acceptable').
        activity_guidelines: Reviewer clarifications or supplementary guidance that refine how
                             activity instructions should be interpreted (not student-facing rules).
        lenient_mode: Optional boolean flag to toggle human-alignment lenient mentor mode.
                      If None, defaults to REVIEWER_LENIENT_MODE env var ("true").
        ultra_lenient_mode: Optional boolean flag to toggle ultra-relaxed high-leeway mode.
                            If None, defaults to REVIEWER_ULTRA_LENIENT_MODE env var ("false").
    """
    images = images or []
    videos = videos or []
    has_video = len(videos) > 0

    comment_preview = (user_comment[:60] + "...") if len(user_comment) > 60 else (user_comment or "(no comment)")
    print(f"\n  [REVIEWER LOG] --- Starting Review: '{activity_name}' ---")
    print(f"  [REVIEWER LOG] Comment preview: '{comment_preview}'")
    print(f"  [REVIEWER LOG] Attachments: {len(images)} image(s), {len(videos)} video(s) | Guardrails: {len(guardrails or [])} | Edge Cases: {'Yes' if activity_edge_cases else 'No'} | Guidelines: {'Yes' if activity_guidelines else 'No'}")

    # Build the lightweight system prompt
    system_prompt = _build_system_prompt()

    # Handle zero media immediately
    if not images and not videos:
        print("  [REVIEWER LOG] ⚠️ No images or videos found. Marking Unsure.")
        return SubmissionReviewOutput(
            checklist_evaluations=[
                ChecklistResultItem(
                    item_id="missing_media",
                    instruction_or_condition="Submission media accessibility check",
                    followed=False,
                    is_fail_if_condition=False,
                    comment="Couldn't load any accessible images or videos from the Drive link provided."
                )
            ],
            agent_comment="Hey, couldn't open any images or videos for this submission — the Drive link might not be shared or accessible. Marking Fail (Unsure) for a human mentor to check.",
            agent_grade="Fail (Unsure)"
        )

    # Build guardrails XML block for user prompt
    guardrails_xml = ""
    if guardrails:
        gr_items = [f"  <rule_{i} name=\"{g.get('name', '').strip()}\">{g.get('description', '').strip()}</rule_{i}>"
                    for i, g in enumerate(guardrails, 1) if g.get('name') and g.get('description')]
        if gr_items:
            guardrails_xml = "<active_guardrails>\n" + "\n".join(gr_items) + "\n</active_guardrails>"

    # Build activity edge cases XML block for user prompt (Top-level Reviewer Pre-Conditions)
    edge_cases_xml = ""
    if activity_edge_cases and activity_edge_cases.strip():
        edge_cases_xml = f"<acceptable_edge_cases_and_overrides>\n{activity_edge_cases.strip()}\n</acceptable_edge_cases_and_overrides>"

    # Build activity guidelines XML block for user prompt (Reviewer clarifications on activity instructions)
    guidelines_xml = ""
    if activity_guidelines and activity_guidelines.strip():
        guidelines_xml = f"\n<activity_guidelines>\n{activity_guidelines.strip()}\n</activity_guidelines>"

    # Check lenient & ultra-lenient modes
    if ultra_lenient_mode is None:
        ultra_lenient_mode = os.environ.get("REVIEWER_ULTRA_LENIENT_MODE", "false").lower() in ("true", "1", "yes")

    if lenient_mode is None:
        lenient_mode = os.environ.get("REVIEWER_LENIENT_MODE", "true").lower() in ("true", "1", "yes")

    if ultra_lenient_mode:
        lenient_block = _ULTRA_LENIENT_DIRECTIVE_BLOCK
    elif lenient_mode:
        lenient_block = _LENIENT_MENTOR_DIRECTIVE_BLOCK
    else:
        lenient_block = ""

    student_comment_text = user_comment.strip() if user_comment and user_comment.strip() else None
    comment_display = student_comment_text if student_comment_text else "(no written comment — evaluate from visual media only)"

    # Build Step 3 Pass / Fail synthesis block based on active mode
    if ultra_lenient_mode:
        step3_block = """Step 3 — SYNTHESIZE OVERALL GRADE (Ultra-Relaxed Two-Tier Framework):

  BEFORE assigning a grade, run through the TIER 1 HARD NO-GO checklist in order:
  → [NO-GO-1] Did the student demonstrably perform a COMPLETELY DIFFERENT concept than instructed (total substitution, not just a gap)?  YES → FAIL
  → [NO-GO-2] Is there ZERO genuine attempt — no real media, fake/stock images, or completely unrelated content?  YES → FAIL
  → [NO-GO-3] Did the activity explicitly mandate specific photos/media/steps and student omitted a mandatory required item (without an allowed edge-case override)?  YES → FAIL
  → [NO-GO-4] Is there a SEVERE IMMEDIATE SAFETY VIOLATION (touching live contacts, active hazard) visible in the media?  YES → FAIL
  → [NO-GO-5] Was the primary task completed in a FUNDAMENTALLY WRONG way that entirely invalidates the learning objective?  YES → FAIL

  If NONE of the above are YES:
  → PASS (UNSURE) — ONLY if a critical verification step is COMPLETELY unreadable or absent from ALL submitted media and you genuinely cannot infer the outcome. Must be a rare exception.
  → PASS — for everything else. Coaching tips in the comment are strictly for presentation quality on COMPLETED steps, NOT for missing mandatory requirements.

  ⚠️ ANTI-HEDGING REMINDER: "(Unsure)" must be rare. If you can make any reasonable determination from the media — even with imperfection — commit to PASS or FAIL. Do NOT reach for PASS (UNSURE) simply because evidence is imperfect or you want to hedge. The strong default here is PASS."""
    else:
        step3_block = """Step 3 — SYNTHESIZE OVERALL GRADE (apply in order — stop at the first rule that matches):

  FAIL — if the student made no genuine attempt at the core physical task (no media, wrong activity, fake images).

  FAIL — if an explicitly mandated requirement, photo, measurement, or core step in <activity_instructions> is missing or omitted. NEVER assign a Pass with "Pass this time, but next time include X" if X was explicitly required in <activity_instructions>!

  FAIL — if a major core conceptual/physical requirement in <activity_instructions> is demonstrably wrong (e.g. asked for Conduction but student showed Convection).

  FAIL (UNSURE) — ONLY if major requirements appear clearly missing but the media is so blurry or cropped that you genuinely cannot be certain. Rare use only.

  PASS (UNSURE) — ONLY if genuine physical effort is visible but a CRITICAL step is completely unverifiable because evidence is absent or truly illegible — not just imperfect. Rare use only.

  PASS — if the student completed ALL explicitly mandated activity instruction requirements with real photo/video evidence and core objectives are met.

  ⚠️ ANTI-HEDGING REMINDER: "(Unsure)" grades must be RARE — the vast majority of submissions should receive a decisive "Pass" or "Fail". Do NOT reach for "(Unsure)" because you feel slightly uncertain or want to hedge. If you can make a reasonable call from the media, COMMIT to Pass or Fail. Only use (Unsure) when evidence for a CRITICAL requirement is completely unreadable or missing entirely."""


    prompt_text = f"""Below is the complete task specification, review guidelines, and student submission to evaluate.

<review_guidelines>
{_REVIEW_GUIDELINES_BLOCK}

{edge_cases_xml}

{guardrails_xml}
</review_guidelines>

<activity_context>
<activity_name>{activity_name or 'N/A'}</activity_name>

<activity_instructions>
{activity_instructions or 'Follow general activity guidelines.'}
</activity_instructions>{guidelines_xml}

<student_submission>
<student_comment>{comment_display}</student_comment>
<attached_media_summary>{len(images)} image(s), {len(videos)} video(s) attached — inspect each carefully</attached_media_summary>
</student_submission>
</activity_context>

{lenient_block}

STEP-BY-STEP EVALUATION INSTRUCTIONS:

Step 0 — INITIALIZE REVIEWER LENS WITH EDGE CASES (PRE-CONDITION OVERRIDES):
  Read <acceptable_edge_cases_and_overrides> FIRST before evaluating media or activity instructions.
  - Note all acceptable equipment types, alternative setups, photo variations, or student submissions explicitly permitted by the reviewer rules.
  - Hold these acceptable variations as your primary baseline — any submission matching an allowed edge case variation MUST be accepted as valid. You are strictly forbidden from marking an item `followed=False` or assigning a `Fail` grade for a variation explicitly allowed in the Edge Cases!

Step 1 — INSPECT MEDIA:
  Look at every attached image/video closely. Note what components, steps, or results are clearly visible.

Step 2 — EXTRACT & EVALUATE ACTIVITY INSTRUCTION REQUIREMENTS (INDEPENDENT OF GRADE):
  Break down the key steps and requirements stated in <activity_instructions>. For each instruction requirement:
  - Factually judge if the visual evidence (and student comment) shows the student completed it (applying <acceptable_edge_cases_and_overrides> as authoritative policy rules).
  - Mark `followed=True` if completed substantially (or with an allowed edge case variation or minor presentation flaw).
  - Mark `followed=False` if omitted, incomplete, or demonstrably wrong.
  CRITICAL: Evaluate each instruction item objectively and independently (`followed=True` or `False`) based on evidence, regardless of what the overall `agent_grade` will be! An overall grade of "Pass" does NOT mean all checklist items must be marked `followed=True`.

{step3_block}

Step 4 — WRITE MENTOR COMMENT:
  1–2 casual sentences. Name the specific evidence or activity instruction requirement that drove the grade.
  Praise what went well; tip on anything minor or missing.

Return your structured evaluation in the required JSON schema, populating `checklist_evaluations` with your objective per-item evaluations.
"""



    # ROUTING DECISION:
    # ROUTING DECISION:
    primary_choice = (primary_model_choice or os.environ.get("REVIEWER_PRIMARY_MODEL", "gpt-5.6-luna")).lower()
    is_gemini_primary = "gemini" in primary_choice
    gemini_model_to_use = primary_choice if is_gemini_primary else _FALLBACK_MODEL

    if has_video or is_gemini_primary:
        model_label = f"{gemini_model_to_use} (Selected Primary)" if is_gemini_primary else f"{_FALLBACK_MODEL} (Video Primary)"
        print(f"  [REVIEWER LOG] 🚀 Primary Model Selected: {model_label}")
        result = None
        try:
            result = _call_gemini_reviewer(system_prompt, prompt_text, images=images, videos=videos, model_name=gemini_model_to_use)
            if result:
                print(f"  [REVIEWER LOG] ✅ Gemini Evaluation done. Grade: '{result.agent_grade}'")
                return result
        except Exception as e:
            print(f"  [REVIEWER LOG] ❌ Gemini Evaluation raised exception: {e}")

        if not result:
            if has_video:
                print(f"  [REVIEWER LOG] ⚠️ Gemini failed for video submission — cannot fallback to GPT-5.6 Luna without video support.")
                return SubmissionReviewOutput(
                    checklist_evaluations=[],
                    agent_comment="Submitted video could not be processed by the AI reviewer. Flagged for manual mentor review.",
                    agent_grade="Fail (Unsure)"
                )
            else:
                print(f"  [REVIEWER LOG] ⚠️ Gemini returned no result. Falling back to GPT-5.6 Luna...")
                try:
                    result = _call_gpt_luna(system_prompt, prompt_text, images)
                    if result:
                        print(f"  [REVIEWER LOG] ✅ GPT-5.6 Luna fallback done. Grade: '{result.agent_grade}'")
                        return result
                except Exception as gpt_err:
                    print(f"  [REVIEWER LOG] ❌ GPT-5.6 Luna fallback failed: {gpt_err}")
    else:
        print(f"  [REVIEWER LOG] 🚀 Primary Model Selected: GPT-5.6 Luna (high reasoning)")
        result = None
        try:
            result = _call_gpt_luna(system_prompt, prompt_text, images)
            if result:
                print(f"  [REVIEWER LOG] ✅ GPT-5.6 Luna done. Grade: '{result.agent_grade}'")
                return result
        except Exception as e:
            print(f"  [REVIEWER LOG] ❌ GPT-5.6 Luna raised exception: {e}")

        if not result:
            print(f"  [REVIEWER LOG] ⚠️ GPT-5.6 Luna returned no result. Falling back to Gemini...")
            try:
                result = _call_gemini_reviewer(system_prompt, prompt_text, images=images, videos=videos, model_name=_FALLBACK_MODEL)
                if result:
                    print(f"  [REVIEWER LOG] ✅ Gemini fallback done. Grade: '{result.agent_grade}'")
                    return result
            except Exception as g_err:
                print(f"  [REVIEWER LOG] ❌ Gemini fallback failed: {g_err}")

    # Complete failure safety net
    print("  [REVIEWER LOG] ⚠️ All model calls failed. Defaulting to Fail.")
    return SubmissionReviewOutput(
        checklist_evaluations=[],
        agent_comment="Ran into an issue calling the AI reviewer — couldn't get a response from any model. Marking Fail.",
        agent_grade="Fail"
    )



