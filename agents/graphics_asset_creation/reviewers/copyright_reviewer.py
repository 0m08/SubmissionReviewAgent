"""
Copyright Reviewer - Pipeline Orchestrator
===========================================
Thin orchestrator that calls three focused agents in sequence:
  A) Similarity Agent   -> derivative-risk verdict + visual anchors
     * EARLY EXIT: If verdict is "Yes" (safe) AND most visual anchors
       are broken/partially changed, stops and returns "Yes".
       If score <49% but anchors are mostly preserved, overrides to
       "No" and continues to the planner.
  B) Technical Validator -> styling & technical accuracy (currently disabled)
  C) Transformation Planner -> editing instructions (only if similarity failed)

Public API is unchanged: copyright_reviewer_agent(), CopyrightReviewResult,
build_copyright_system_instruction(), TransformationInstructions.
"""

from PIL import Image
from typing import Optional, Any, List, Tuple
from pydantic import BaseModel, Field
from langsmith import traceable

from agents.graphics_asset_creation.reviewers.voiceover_reviewer import styling_guide

# Import the three stage agents
from agents.graphics_asset_creation.reviewers.similarity_agent import similarity_agent, SimilarityResult
from agents.graphics_asset_creation.reviewers.technical_validator import technical_instruction_validator_agent
from agents.graphics_asset_creation.reviewers.transformation_planner import transformation_planner_agent
from agents.graphics_asset_creation.reviewers.transformation_planner import TransformationInstructions  # re-export


# Output schema (unchanged)

class CopyrightReviewResult(BaseModel):
    """Output schema for the Copyright Reviewer Agent."""
    description: str = Field(description="Detailed description of the current image including: subject, setting, style, camera angle, and key features.")
    analysis: str = Field(description="Comprehensive analysis covering: A) SIMILARITY ASSESSMENT - How different this image is from reference images based on similarity metrics (camera angle, style, composition transformations). B) STYLING GUIDE COMPLIANCE - Whether it meets professional technical asset standards and quality requirements. C) SAFETY VERDICT RATIONALE - Whether the image is safe to use based on differentiation and quality factors.")
    verdict: str = Field(description="Final verdict: 'Yes' ONLY if the image is not a derivative of the reference images, technically accurate and legally safe to use. 'No' if it is a derivative of the reference image or needs further refinement.")
    recommended_instructions: TransformationInstructions = Field(description="Transformation plan to transform the reference image to reach a copyright-free status. Only fill relevant fields - leave fields empty if no changes needed.")
    similarity_scores: str = Field(default="", description="Internal field for similarity metrics when reference images are provided.")
    similarity_analysis: str = Field(default="", description="Detailed similarity analysis from _calculate_image_similarity including visual_analysis, and recommendation for each reference image.")

    @property
    def transformation_instructions(self) -> TransformationInstructions:
        """Backward compatible alias for historical attribute name."""
        return self.recommended_instructions


def build_copyright_system_instruction() -> str:
    """Build the system instruction for the Copyright Reviewer session.
    Retained for backward compatibility with automated_voiceover_reviewer
    which creates a chat session with this instruction."""

    return f"""You are a Lead Design & Copyright Auditor for high-fidelity educational training assets.

Your mission: Transform reference images into "Professional Technical Assets" that are legally safe, instructionally clear, and visually premium.

[VERDICT LOGIC]
- **Yes**: Image is a high-end graphical asset, has no branding, meets styling guides, and is instructionally perfect to be used in the educational context.
- **No**: Image is a photograph, looks like a derivative from the reference image, has branding, or lacks the weight and professional finish of an engineering asset.

[STRICT RULE — NO NEW COMPONENTS]
When suggesting transformations, you MUST NOT introduce, add, or suggest any new
components, objects, labels, annotations, text overlays, or visual elements that are
not already present in the current image.
Transformations are ONLY allowed to:
  • CHANGE the appearance of existing elements (camera angle, style, lighting, colour)
  • REMOVE existing elements (branding, watermarks, identifiable logos)
  • PRESERVE required educational elements
Adding entirely new content to the scene is PROHIBITED.

[STYLING GUIDE - QUALITY STANDARDS]
{styling_guide}
"""


# ── Anchor safety analysis ──────────────────────────────────────────────────

def _analyze_anchor_safety(
    visual_anchors_by_reference: List[List[dict]],
    num_attempts: int = 0,
) -> Tuple[bool, str]:
    """
    Determine whether an image is truly safe based on visual anchor statuses,
    not just the aggregate similarity score.

    Only BREAKABLE anchors count toward unsafety. Protected (educational)
    anchors cannot be transformed without destroying instructional value,
    so they are excluded from all safety rules.

    Thresholds are progressively relaxed after multiple attempts to prevent
    infinite edit loops that never converge.

    Args:
        visual_anchors_by_reference: Per-reference anchor lists.
        num_attempts: How many review rounds have already been attempted.

    Returns:
        (is_safe, analysis_text)
    """
    total = 0
    preserved = 0
    partially_changed = 0
    broken = 0
    high_imp_total = 0
    high_imp_preserved = 0

    # Breakable-only counters (exclude PROTECTED-educational anchors)
    breakable_total = 0
    breakable_preserved = 0
    breakable_partially_changed = 0
    breakable_broken = 0
    breakable_high_imp_total = 0
    breakable_high_imp_preserved = 0
    protected_preserved_count = 0

    for ref_anchors in visual_anchors_by_reference:
        for anchor in ref_anchors:
            if not isinstance(anchor, dict):
                continue
            total += 1
            status = anchor.get("status", "").lower().strip()
            importance = anchor.get("importance", "").lower().strip()
            is_breakable = anchor.get("breakable", True)  # default True for legacy

            if status == "preserved":
                preserved += 1
                if not is_breakable:
                    protected_preserved_count += 1
            elif status == "partially_changed":
                partially_changed += 1
            else:
                broken += 1

            if importance == "high":
                high_imp_total += 1
                if status == "preserved":
                    high_imp_preserved += 1

            # Only count breakable anchors in actionable safety evaluation
            if is_breakable:
                breakable_total += 1
                if status == "preserved":
                    breakable_preserved += 1
                elif status == "partially_changed":
                    breakable_partially_changed += 1
                else:
                    breakable_broken += 1
                if importance == "high":
                    breakable_high_imp_total += 1
                    if status == "preserved":
                        breakable_high_imp_preserved += 1

    if total == 0:
        return True, "No visual anchors detected — falling back to score-based verdict."

    # Use breakable-only counts for ratio-based safety rules
    bt = breakable_total
    preserved_ratio = breakable_preserved / bt if bt > 0 else 0.0
    transformed_ratio = (breakable_broken + breakable_partially_changed) / bt if bt > 0 else 1.0
    high_imp_preserved_ratio = (
        breakable_high_imp_preserved / breakable_high_imp_total
        if breakable_high_imp_total > 0 else 0.0
    )

    lines = [
        f"Anchor Safety Analysis: {total} anchors total ({bt} breakable, {total - bt} protected-educational)",
        f"  [Breakable] Broken: {breakable_broken}  Partially changed: {breakable_partially_changed}  Preserved: {breakable_preserved}",
        f"  [Protected] Preserved educational anchors (excluded from rules): {protected_preserved_count}",
        f"  High-importance breakable preserved: {breakable_high_imp_preserved}/{breakable_high_imp_total}",
    ]

    is_safe = True
    reasons: List[str] = []

    # Progressive threshold relaxation to prevent infinite loops.
    # After 3+ attempts the image editor has done its best; relax to let it converge.
    if num_attempts >= 3:
        preserved_thresh = 0.60
        high_imp_thresh  = 0.50
        transformed_thresh = 0.35
        lines.append(f"  [Relaxed thresholds after {num_attempts} attempts: "
                     f"preserved≤{preserved_thresh:.0%}, high-imp≤{high_imp_thresh:.0%}, "
                     f"transformed≥{transformed_thresh:.0%}]")
    else:
        preserved_thresh = 0.40
        high_imp_thresh  = 0.30
        transformed_thresh = 0.50

    # Rule 1: If >threshold of BREAKABLE anchors are still fully preserved → not safe
    # (partially_changed anchors are treated as effectively transformed)
    if preserved_ratio > preserved_thresh:
        is_safe = False
        reasons.append(
            f"{preserved_ratio:.0%} of breakable anchors are preserved (threshold {preserved_thresh:.0%})"
        )

    # Rule 2: If >threshold of high-importance BREAKABLE anchors are preserved → not safe
    if breakable_high_imp_total > 0 and high_imp_preserved_ratio > high_imp_thresh:
        is_safe = False
        reasons.append(
            f"{breakable_high_imp_preserved}/{breakable_high_imp_total} high-importance breakable anchors preserved (threshold {high_imp_thresh:.0%})"
        )

    # Rule 3: If less than threshold of BREAKABLE anchors are broken or partially changed → not safe
    if transformed_ratio < transformed_thresh:
        is_safe = False
        reasons.append(
            f"only {transformed_ratio:.0%} breakable anchors are broken/partially changed (need ≥{transformed_thresh:.0%})"
        )

    if is_safe:
        lines.append("  ✓ Most breakable anchors are broken or partially changed — transformation is sufficient.")
    else:
        lines.append(f"  ✗ OVERRIDE: {'; '.join(reasons)}")

    # Expose breakable_preserved count for downstream override decision
    lines.append(f"  __breakable_preserved__: {breakable_preserved}")  # machine-readable tag

    return is_safe, "\n".join(lines)


# Pipeline orchestrator

@traceable(metadata={"agent_name": "copyright_reviewer"})
def copyright_reviewer_agent(
    image: Image.Image,
    slide_title: str = "",
    voiceover: str = "",
    chat_session: Optional[Any] = None,
    reference_images: Optional[List[Image.Image]] = None,
    previous_instructions: Optional[List[TransformationInstructions]] = None,
    previous_similarity_scores: Optional[List[float]] = None,
    previous_context: str = "",
) -> CopyrightReviewResult:
    """
    Copyright Reviewer pipeline: A -> [Early Exit if Safe] -> B -> C.

    Flow:
    1. Stage A: Similarity Agent checks derivative risk
       - If verdict is "Yes" (safe): Returns immediately with "Yes" verdict
       - If verdict is "No": Continues to Stage B and C
    2. Stage B: Technical Validator (only if similarity failed)
    3. Stage C: Transformation Planner (only if similarity failed)

    Args:
        image: Current image to review.
        slide_title: Educational context.
        voiceover: Educational context.
        chat_session: Legacy parameter for backward compatibility (not used).
        reference_images: Reference images to check for derivative risk.
        previous_instructions: History of instructions from previous review cycles.
                              Pass this to prevent oscillation and repetitive suggestions.
        previous_similarity_scores: List of similarity scores from previous attempts.
                                   Used for convergence detection.
        previous_context: Summary of previous edits from other agents (e.g. Accuracy Reviewer).

    Returns:
        CopyrightReviewResult with verdict and transformation instructions.

    Signature and return type are identical to the original monolithic agent
    so all consumers (automated_voiceover_reviewer, voiceover_reviewer_ui)
    continue to work without changes.
    """
    print("\n[COPYRIGHT REVIEW] Starting pipeline...")
    print(f"  Reference images: {len(reference_images) if reference_images else 0}")
    if previous_similarity_scores:
        print(f"  Previous similarity scores: {[f'{s:.1f}%' for s in previous_similarity_scores]}")

    # Stage A: Similarity
    sim_result: SimilarityResult = similarity_agent(
        image=image,
        reference_images=reference_images,
        slide_title=slide_title,
        voiceover=voiceover,
    )

    # Extract current similarity score for tracking
    current_similarity_score = None
    if sim_result.similarity_scores:
        # Parse first score from summary like "Reference 1: 72.0%"
        import re
        score_match = re.search(r'(\d+\.\d+)%', sim_result.similarity_scores)
        if score_match:
            current_similarity_score = float(score_match.group(1))
    
    # Build cumulative similarity score list
    all_similarity_scores = (previous_similarity_scores or []) + ([current_similarity_score] if current_similarity_score else [])
    
    # Intelligent convergence analysis - provide feedback for strategy adjustment
    convergence_feedback = ""
    if len(all_similarity_scores) >= 3:
        last_3 = all_similarity_scores[-3:]
        if all(s >= 50 for s in last_3):  # All failing
            improvements = [last_3[i] < last_3[i-1] for i in range(1, len(last_3))]
            if not any(improvements):  # No improvements in last 3
                print("\n[COPYRIGHT REVIEW] ⚠️ CONVERGENCE STAGNATION DETECTED")
                print(f"  Last 3 scores: {[f'{s:.1f}%' for s in last_3]} - No improvement")
                print("  Will suggest complementary transformation strategy to planner")
                convergence_feedback = f"""
CONVERGENCE ANALYSIS:
- Last 3 similarity scores show no improvement: {last_3}
- Current approach is not reducing similarity effectively
- RECOMMENDATION: Try complementary transformation dimensions
  * If camera angle changed: Try style transformation instead
  * If style changed: Try composition/framing changes
  * Consider hybrid approach combining multiple dimensions
"""

    # ── ANCHOR-AWARE EARLY EXIT ──────────────────────────────────────────
    # A low similarity score alone is NOT enough.  We also verify that most
    # visual anchors are broken or partially changed before declaring safe.
    num_attempts = len(previous_similarity_scores) if previous_similarity_scores else 0
    anchors_safe, anchor_analysis_text = _analyze_anchor_safety(
        sim_result.visual_anchors_by_reference,
        num_attempts=num_attempts,
    )
    print(f"[COPYRIGHT REVIEW] {anchor_analysis_text}")

    if sim_result.verdict.strip().lower() == "yes":
        print(f"[COPYRIGHT REVIEW] Similarity verdict: YES (score < 49%)")

        if anchors_safe:
            # Both score AND anchors confirm safety → early exit
            print("[COPYRIGHT REVIEW] ✓ Anchors confirm safety — early exit.\n")
            result = CopyrightReviewResult(
                description=sim_result.description,
                analysis=f"""A) SIMILARITY ASSESSMENT:
   Scores: {sim_result.similarity_scores}
   Verdict: SAFE - No derivative risk detected

B) VISUAL ANCHOR VERIFICATION:
   {anchor_analysis_text}

SAFETY VERDICT RATIONALE: Similarity score is below threshold AND most visual
anchors are broken or partially changed — image is safe to use.""",
                verdict="Yes",
                recommended_instructions=TransformationInstructions(),
                similarity_scores=sim_result.similarity_scores,
                similarity_analysis=sim_result.similarity_analysis,
            )
            print(f"[COPYRIGHT REVIEW] Early exit with verdict: {result.verdict}\n")
            return result
        else:
            # Score says safe but anchors say recognisable elements survive.
            # Override verdict to "No" and continue to the planner.
            print("[COPYRIGHT REVIEW] ⚠️ Score < 49% but anchors are mostly preserved — overriding to NO.")
            print("  Continuing to transformation planner to break remaining anchors.\n")

    else:
        # Similarity verdict is "No" (score ≥ 50%). Check whether ALL preserved
        # anchors are PROTECTED-educational. If nothing is breakable, there is
        # nothing actionable to transform — override to "Yes".
        import re as _re
        bp_match = _re.search(r'__breakable_preserved__:\s*(\d+)', anchor_analysis_text)
        breakable_preserved_count = int(bp_match.group(1)) if bp_match else None

        if breakable_preserved_count == 0:
            print("[COPYRIGHT REVIEW] ✅ Similarity says No but ALL preserved anchors are PROTECTED-educational.")
            print("   Nothing actionable to transform — overriding verdict to YES.\n")
            return CopyrightReviewResult(
                description=sim_result.description,
                analysis=f"""A) SIMILARITY ASSESSMENT:
   Scores: {sim_result.similarity_scores}
   Raw verdict: NO (score ≥50%) — but see override below

B) ANCHOR BREAKABILITY ANALYSIS:
   {anchor_analysis_text}

OVERRIDE RATIONALE: All preserved visual anchors are classified as PROTECTED-educational
(functional/instructional content required by the voiceover). None of the remaining
similarity is attributable to breakable stylistic elements. Transforming these anchors
would destroy the educational value of the image without reducing real copyright risk.
Verdict overridden to YES.""",
                verdict="Yes",
                recommended_instructions=TransformationInstructions(),
                similarity_scores=sim_result.similarity_scores,
                similarity_analysis=sim_result.similarity_analysis,
            )

    # ── MAX-ROUNDS ESCAPE HATCH ──────────────────────────────────────────
    # After enough attempts, if the similarity score is in a reasonable range,
    # accept the image to avoid infinite edit loops that never converge.
    if num_attempts >= 4 and current_similarity_score is not None and current_similarity_score <= 65.0:
        print(f"[COPYRIGHT REVIEW] \U0001f3c1 Escape hatch: {num_attempts} prior attempts "
              f"with score {current_similarity_score:.1f}% (\u226465%). Accepting.")
        return CopyrightReviewResult(
            description=sim_result.description,
            analysis=f"""A) SIMILARITY ASSESSMENT:
   Scores: {sim_result.similarity_scores}
   Verdict: ACCEPTED via escape hatch after {num_attempts} attempts (score {current_similarity_score:.1f}% \u226465%)

B) VISUAL ANCHOR VERIFICATION:
   {anchor_analysis_text}

SAFETY VERDICT RATIONALE: After {num_attempts} transformation attempts the similarity score
has been reduced to {current_similarity_score:.1f}% which is within the acceptable tolerance
range (\u226465%). Further editing is unlikely to converge. Image accepted.""",
            verdict="Yes",
            recommended_instructions=TransformationInstructions(),
            similarity_scores=sim_result.similarity_scores,
            similarity_analysis=sim_result.similarity_analysis,
        )

    # Stage B: Technical Validation (only if similarity check failed)
    # tech_result: TechnicalValidationResult = technical_validator_agent(
    #     image=image,
    #     slide_title=slide_title,
    #     voiceover=voiceover,
    #     image_description=sim_result.description,
    #     previous_context=previous_context
    # )

    # Combine verdicts (similarity already failed if we reached here)
    similarity_failed = True  # We only reach here if similarity verdict was "No"
    # technical_failed = tech_result.issues_found
    overall_verdict = "No"  # Always "No" if we didn't early exit

    # Build unified analysis text
    analysis_parts = []
    analysis_parts.append("A) SIMILARITY ASSESSMENT:")
    if sim_result.similarity_scores:
        analysis_parts.append(f"   Scores: {sim_result.similarity_scores}")
    score_low_but_anchors_bad = (
        sim_result.verdict.strip().lower() == "yes"
    )
    if score_low_but_anchors_bad:
        analysis_parts.append(
            "   Verdict: Score < 49% BUT visual anchors are mostly preserved — overridden to NOT SAFE"
        )
    else:
        analysis_parts.append("   Verdict: NOT SAFE - derivative risk detected")

    analysis_parts.append(f"\nB) VISUAL ANCHOR VERIFICATION:")
    analysis_parts.append(f"   {anchor_analysis_text}")

    analysis_parts.append(
        f"\nSAFETY VERDICT RATIONALE: Image needs transformation — "
        f"{'preserved anchors keep it recognisable despite low score.' if score_low_but_anchors_bad else 'similarity check failed.'}"
    )
    analysis_text = "\n".join(analysis_parts)

    # Stage C: Transformation Plan (always needed since we didn't early exit)
    issue_summary_parts = []
    # Include similarity analysis since it failed
    issue_summary_parts.append(f"Similarity: Derivative risk detected")
    # Include anchor analysis so the planner knows which anchors to break
    issue_summary_parts.append(anchor_analysis_text)
    
    # Append convergence feedback for strategy adjustment
    if convergence_feedback:
        issue_summary_parts.append(convergence_feedback)
    
    # Build context including accuracy requirements and convergence feedback
    enhanced_context = previous_context
    if convergence_feedback:
        enhanced_context = f"{previous_context}\n\n{convergence_feedback}" if previous_context else convergence_feedback
    
    instructions = transformation_planner_agent(
        image=image,  # ← Passing image for visual context
        image_description=sim_result.description,
        similarity_analysis=sim_result.similarity_analysis,
        visual_anchors=sim_result.visual_anchors_by_reference,
        anchor_descriptions=sim_result.anchor_descriptions,
        anchor_breaking_suggestions=sim_result.anchor_breaking_suggestions,
        technical_issues="\n".join(issue_summary_parts) if issue_summary_parts else "",
        slide_title=slide_title,
        voiceover=voiceover,
        previous_instructions=previous_instructions,
        previous_similarity_scores=all_similarity_scores,  # ← Passing score history
        previous_context=enhanced_context  # ← Includes convergence strategy + accuracy requirements
    )

    # Stage D: Technical Instruction Validation
    # Validates and corrects each instruction for engineering accuracy, factual
    # correctness (using Google Search), visual feasibility, and voiceover
    # compatibility — BEFORE the instructions reach the image editor.
    print("\n[COPYRIGHT REVIEW] Stage D: Technical Instruction Validation...")
    try:
        instructions = technical_instruction_validator_agent(
            image=image,
            image_description=sim_result.description,
            transformation_instructions=instructions,
            slide_title=slide_title,
            voiceover=voiceover,
            visual_anchors_text=sim_result.anchor_descriptions,
            previous_context=enhanced_context,
        )
    except Exception as val_err:
        # Validation is a best-effort step — if it fails, use planner output as-is
        print(f"[COPYRIGHT REVIEW] ⚠️  Stage D failed ({val_err}) — using planner output as-is.")

    # Assemble final result (same schema as before)
    result = CopyrightReviewResult(
        description=sim_result.description,
        analysis=analysis_text,
        verdict=overall_verdict,
        recommended_instructions=instructions,
        similarity_scores=sim_result.similarity_scores,
        similarity_analysis=sim_result.similarity_analysis,
    )

    print(f"[COPYRIGHT REVIEW] Pipeline verdict: {result.verdict}\n")
    return result
