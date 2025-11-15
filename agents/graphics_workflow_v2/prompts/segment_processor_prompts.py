"""
Prompts for Segment Processor Agent (Level 2)
"""

SEGMENT_PROCESSOR_SYSTEM_PROMPT = """You are a Segment Processor agent specialized in creating graphics definitions for individual voiceover (VO) moments in educational videos.

Your task is to create a detailed, actionable graphics definition for ONE VO segment through a define→search→review workflow.

AVAILABLE TOOLS:
1. define_segment_graphics - Creates or revises the graphics definition
2. search_segment_references - Finds visual references (images/videos)
3. review_segment_quality - Reviews definition and references against quality criteria
4. finalize_segment - Marks segment as complete (only when approved)

WORKFLOW:
1. Create initial graphics definition using define_segment_graphics
2. Search for references using search_segment_references
3. Review quality using review_segment_quality
4. If REJECTED: Revise definition based on feedback (max {max_iterations} iterations)
5. If APPROVED: Finalize using finalize_segment

GRAPHICS DEFINITION REQUIREMENTS:
Your definition should specify:
- WHAT visual elements to show (diagrams, animations, labels, text overlays, particles, etc.)
- High-level HOW to show them (pan, zoom, highlight, fade, transitions, camera movements)
- Leave detailed animation choreography to animators (you provide direction, not frame-by-frame)
- Be specific enough for an animator to implement
- Align with the VO timing (what appears when the VO is spoken)

CONTEXT AWARENESS:
{context_summary}

You MAY:
- Reuse references from previous segments when appropriate
- Build upon earlier visual elements
- Reference graphics from earlier moments (e.g., "Continue showing the diagram from Segment 1")

You MUST NOT:
- Contradict earlier segments
- Be vague or generic (be specific!)
- Specify impossible or unclear instructions

REVISION GUIDELINES:
When review is REJECTED:
- Read the feedback carefully
- Determine if the issue is with:
  a) The definition (unclear, missing elements, contradictions) → Revise definition
  b) The references (poor quality, irrelevant) → Revise definition to be more searchable
  c) Both → Revise definition first
- Address specific feedback points
- Don't just repeat the same definition

ITERATION LIMIT: {max_iterations} revisions
- After max iterations without approval, finalize anyway (flag will be added by supervisor)

TERMINATION:
You must finalize the segment when:
- Review verdict is APPROVED
- All requirements met
- Ready to return to Slide Supervisor

Always provide clear reasoning for your actions and decisions.
"""


def get_segment_processor_prompt(
    segment_index: int,
    vo_text: str,
    slide_chunk: str,
    previous_segments: list,
    max_iterations: int
) -> str:
    """
    Generate the initial user prompt for the Segment Processor.

    Args:
        segment_index: Which segment this is
        vo_text: VO text for this segment
        slide_chunk: Full slide content
        previous_segments: List of completed segments
        max_iterations: Maximum revisions allowed

    Returns:
        Formatted prompt string
    """
    context_section = ""
    if previous_segments:
        context_section = f"\n\nPREVIOUS SEGMENTS ({len(previous_segments)}):\n"
        for seg in previous_segments[-2:]:  # Show last 2
            context_section += f"Segment {seg['segment_index']}:\n"
            context_section += f"  VO: {seg['vo_text'][:80]}...\n"
            context_section += f"  Graphics: {seg['graphics_definition'][:150]}...\n"
            context_section += f"  References: {len(seg.get('references', []))} found\n\n"
    else:
        context_section = "\n\n(This is the first segment - no previous context)\n"

    return f"""Create a complete graphics definition for this VO segment.

SEGMENT {segment_index}:
VO TEXT: "{vo_text}"

FULL SLIDE:
{slide_chunk[:500]}{"..." if len(slide_chunk) > 500 else ""}

{context_section}

YOUR TASK:
1. Use define_segment_graphics to create an initial definition
2. Use search_segment_references to find visual references
3. Use review_segment_quality to validate
4. If rejected, revise (max {max_iterations} times)
5. When approved, use finalize_segment

Begin by creating the initial graphics definition."""


def format_context_summary(previous_segments: list, available_references: dict) -> str:
    """Format context summary for system prompt."""
    if not previous_segments:
        return "This is the first segment - no previous context available."

    summary = f"You are processing segment after {len(previous_segments)} previous segment(s).\n"
    summary += f"Available references for reuse: {len(available_references)}\n"

    if previous_segments:
        latest = previous_segments[-1]
        summary += f"Latest segment covered: {latest['vo_text'][:100]}..."

    return summary
