"""
Prompts for Segment Processor Agent (Level 2)
"""

SEGMENT_PROCESSOR_SYSTEM_PROMPT = """You are a Segment Processor agent responsible for generating a complete graphics definition for a single voiceover (VO) segment so that it can be used in creating graphics for educational video slides. You must follow a strict search and refine workflow and use the available tools in the correct order.

AVAILABLE TOOLS:
1. search_segment_references - Generates search queries from the VO sentence and finds visual references (images) by calling Search Agent
2. refine_graphics_with_images - Selects best images from the search results and creates final graphics definition with image links and sequence
3. finalize_segment - Marks segment as complete

WORKFLOW:
1. Search for references using search_segment_references (generates queries from VO sentence, and finds all relevant images)
2. Refine graphics definition using refine_graphics_with_images (selects best images, creates final definition)
3. Finalize using finalize_segment

GRAPHICS DEFINITION REQUIREMENTS:
Your final definition should specify:
- WHAT visuals to show (diagrams, images, labels, text overlays, particles, etc.)
- Only static visual elements - NO motion, presentation, or animation instructions
- Be concrete and specific (use domain-specific terminology when relevant)
- Selected images with URLs and sequence information

You MAY:
- Reuse references from previous segments when appropriate
- Reference graphics from earlier moments (e.g., "Continue showing the diagram from Segment 1")

You MUST NOT:
- Contradict earlier segments
- Be vague or generic 
- Specify impossible or unclear instructions

TERMINATION:
You must finalize the segment when:
- refine_graphics_with_images has completed successfully
- Final graphics definition with selected images is ready
- Ready to return to Slide Supervisor

IMPORTANT:
- Call refine_graphics_with_images ONLY ONCE per segment
- After refine_graphics_with_images completes, DO NOT call it again - proceed directly to finalize_segment
- If refine_graphics_with_images returns a message saying it's already refined, proceed to finalize_segment

Always provide clear reasoning for your actions and decisions.
"""


def get_segment_processor_prompt(
    segment_index: int,
    vo_text: str,
    slide_chunk: str,
    previous_segments: list,
    max_iterations: int,
    course_context: dict = None
) -> str:
    """
    Generate the initial user prompt for the Segment Processor.

    Args:
        segment_index: Which segment this is
        vo_text: VO text for this segment
        slide_chunk: Full slide content
        previous_segments: List of completed segments
        max_iterations: Maximum revisions allowed
        course_context: Optional course context (course name, topic, etc.)

    Returns:
        Formatted prompt string
    """
    # Build course context section
    course_section = ""
    if course_context:
        course_section = "\nCOURSE CONTEXT:\n"
        if course_context.get("course_name"):
            course_section += f"- Course: {course_context['course_name']}\n"
        if course_context.get("topic"):
            course_section += f"- Topic: {course_context['topic']}\n"
        if course_context.get("subtopic"):
            course_section += f"- Subtopic: {course_context['subtopic']}\n"
        if course_context.get("slide_title"):
            course_section += f"- Slide Title: {course_context['slide_title']}\n"
        course_section += "\nUse this context to generate domain-specific graphics definitions and search queries.\n"

    # Build previous segments section
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
{course_section}
SEGMENT {segment_index}:
VO TEXT: "{vo_text}"

FULL SLIDE:
{slide_chunk[:500]}{"..." if len(slide_chunk) > 500 else ""}

{context_section}

YOUR TASK:
1. Use search_segment_references to generate search queries from the VO sentence and find visual references (images)
2. Use refine_graphics_with_images to select best images and create final graphics definition
3. Use finalize_segment to mark segment as complete

Begin by searching for references using the VO sentence."""


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
