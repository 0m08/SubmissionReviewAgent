"""
Segment Processor Tools (Level 2)

Tools for the Segment Processor agent to create graphics definitions for individual VO segments.
"""

from typing import Annotated, Dict, List, Any
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
import time

from agents.graphics_workflow_v2.state.schemas import SegmentProcessorState, SegmentData, ReferenceData
from agents.graphics_workflow_v2.config.settings import (
    DEFINITION_CHAIN_MODEL,
    REVIEW_CHAIN_MODEL,
    MIN_DEFINITION_LENGTH,
    ENABLE_CROSS_SEGMENT_VALIDATION,
)
from modules.chain import Chain
from agents.graphics_workflow_v2.agents.search_agent import run_search_agent


@tool
def define_segment_graphics(
    instruction: str,
    state: Annotated[Dict, InjectedState]
) -> Command[str]:
    """
    Creates or revises the graphics definition for this VO segment.

    Generates detailed instructions specifying:
    - Visual elements to show (diagrams, labels, animations, etc.)
    - How to present them (pan, zoom, highlight, transitions)
    - Relationship to previous segments (if applicable)
    - Required reference types (images, videos, diagrams)

    Args:
        instruction: Guidance ("create initial definition" or "revise based on feedback...")

    Returns:
        Summary of created/revised definition
    """
    vo_text = state.get("vo_text", "")
    slide_chunk = state.get("slide_chunk", "")
    previous_segments = state.get("previous_segments", [])
    review_feedback = state.get("review_feedback", "")
    current_definition = state.get("graphics_definition", "")
    iteration_count = state.get("iteration_count", 0)

    # Build context from previous segments
    previous_context = ""
    if previous_segments:
        previous_context = "\n\nPREVIOUS SEGMENTS CONTEXT:\n"
        for seg in previous_segments[-3:]:  # Last 3 segments for context
            previous_context += f"Segment {seg['segment_index']}:\n"
            previous_context += f"VO: {seg['vo_text'][:100]}...\n"
            previous_context += f"Graphics: {seg['graphics_definition'][:200]}...\n\n"

    # Create Chain
    chain = Chain(
        llm=DEFINITION_CHAIN_MODEL,
        tags=["visual_elements", "presentation", "references_needed"]
    )

    # Build prompt
    if "revise" in instruction.lower() and review_feedback:
        prompt_type = "revision"
        specific_instruction = f"""REVISE the graphics definition based on this feedback:
{review_feedback}

Current definition:
{current_definition}

Make specific improvements to address the feedback."""
    else:
        prompt_type = "initial"
        specific_instruction = "CREATE an initial graphics definition for this VO segment."

    prompt = f"""You are creating graphics instructions for an educational video.

TASK: {specific_instruction}

VO TEXT FOR THIS MOMENT:
"{vo_text}"

FULL SLIDE CONTEXT:
{slide_chunk}

{previous_context}

GUIDELINES:
- Specify WHAT visual elements to show (diagrams, animations, labels, text overlays, etc.)
- Describe high-level HOW to show them (pan, zoom, highlight, fade in/out, transitions)
- Leave detailed animation choreography to animators
- Be specific enough that an animator can implement this
- Consider the VO timing - what should appear when the VO is spoken
- Build upon or reference previous segments when appropriate
- Be concrete, not vague (e.g., "Show evaporator diagram" not "Show relevant diagram")

OUTPUT FORMAT:
<visual_elements>
- Element 1: [Specific description]
- Element 2: [Specific description]
- ...
</visual_elements>

<presentation>
[Detailed instructions on how to present/animate these elements. Include camera movements, transitions, timing cues, etc.]
</presentation>

<references_needed>
- Reference type 1: [What to search for]
- Reference type 2: [What to search for]
- ...
</references_needed>"""

    chain.add_message(role="user", content=prompt)
    response = chain.run()

    # Parse response
    if isinstance(response, dict):
        response = response.get("content", str(response))

    parsed = chain.extract_text_in_tags(response)

    visual_elements = parsed.get("visual_elements", "")
    presentation = parsed.get("presentation", "")
    references_needed = parsed.get("references_needed", "")

    # Combine into full definition
    graphics_definition = f"""VISUAL ELEMENTS:
{visual_elements}

PRESENTATION:
{presentation}

REFERENCES NEEDED:
{references_needed}"""

    # Validate length
    if len(graphics_definition.strip()) < MIN_DEFINITION_LENGTH:
        return Command(
            update={},
            goto="agent"
        )

    # Update iteration count if this is a revision
    new_iteration_count = iteration_count + 1 if prompt_type == "revision" else iteration_count

    # Format response
    result_message = f"""Graphics definition {"created" if prompt_type == "initial" else "revised"}:

VISUAL ELEMENTS ({len(visual_elements.split('-')) - 1} elements):
{visual_elements[:300]}...

PRESENTATION APPROACH:
{presentation[:200]}...

Length: {len(graphics_definition)} characters
"""

    return Command(
        update={
            "graphics_definition": graphics_definition,
            "iteration_count": new_iteration_count
        },
        goto="agent"
    )


@tool
def search_segment_references(
    instruction: str,
    state: Annotated[Dict, InjectedState]
) -> Command[str]:
    """
    Finds relevant visual references for this segment's graphics definition.

    Creates a Search Agent instance that will:
    - Check for reusable references from previous segments
    - Generate search queries
    - Execute searches iteratively
    - Return best matching references

    Args:
        instruction: Search guidance from supervisor

    Returns:
        Summary of found references
    """
    graphics_definition = state.get("graphics_definition", "")
    available_references = state.get("available_references", {})
    drive = state.get("drive")

    if not graphics_definition:
        return Command(
            update={},
            goto="agent"
        )

    # Parse visual elements from definition (simplified - could be more sophisticated)
    visual_elements = extract_visual_elements(graphics_definition)

    # Run Search Agent
    try:
        search_result = run_search_agent(
            graphics_definition=graphics_definition,
            visual_elements=visual_elements,
            available_references=available_references,
            drive=drive,
            filters=state.get("filters"),
            root_folder_id=state.get("root_folder_id"),
        )

        # Extract selected references
        selected_references = search_result.get("selected_references", [])

        # Format response
        result_message = f"""Search completed. Found {len(selected_references)} reference(s):

"""
        for i, ref in enumerate(selected_references[:5]):
            result_message += f"{i+1}. {ref['title']} ({ref['type']})\n"
            result_message += f"   Relevance: {ref['relevance_score']:.2f}\n"
            result_message += f"   {ref['description'][:100]}...\n\n"

        if len(selected_references) > 5:
            result_message += f"... and {len(selected_references) - 5} more references.\n"

        return Command(
            update={"references": selected_references},
            goto="agent"
        )

    except Exception as e:
        error_message = f"Search failed: {str(e)}\nPlease try again or adjust the graphics definition."
        return Command(
            update={"references": []},
            goto="agent"
        )


@tool
def review_segment_quality(
    state: Annotated[Dict, InjectedState]
) -> Command[str]:
    """
    Reviews the segment's graphics definition and references for quality.

    Checks against criteria:
    ✓ Definition aligns with VO text
    ✓ At least 1 relevant reference per visual element
    ✓ No contradictions with previous segments
    ✓ Specific enough for animator to implement
    ✓ All references are accessible

    Returns:
        "APPROVED" or "REJECTED: [detailed feedback]"
    """
    vo_text = state.get("vo_text", "")
    graphics_definition = state.get("graphics_definition", "")
    references = state.get("references", [])
    previous_segments = state.get("previous_segments", [])

    # Check prerequisites
    if not graphics_definition:
        return Command(
            update={
                "review_verdict": "rejected",
                "review_feedback": "No graphics definition to review. Please create a definition first."
            },
            goto="agent"
        )

    if not references:
        return Command(
            update={
                "review_verdict": "rejected",
                "review_feedback": "No references found. Please search for references before reviewing."
            },
            goto="agent"
        )

    # Create review chain
    chain = Chain(
        llm=REVIEW_CHAIN_MODEL,
        tags=["verdict", "vo_alignment", "references_found", "references_relevant",
              "no_contradictions", "specificity", "feedback"]
    )

    # Format references for review
    refs_text = ""
    for i, ref in enumerate(references):
        refs_text += f"{i+1}. {ref['title']} ({ref['type']}) - Relevance: {ref['relevance_score']:.2f}\n"
        refs_text += f"   {ref['description']}\n"

    # Format previous segments
    prev_text = ""
    if previous_segments and ENABLE_CROSS_SEGMENT_VALIDATION:
        prev_text = "\n\nPREVIOUS SEGMENTS (check for contradictions):\n"
        for seg in previous_segments[-2:]:
            prev_text += f"Segment {seg['segment_index']}: {seg['graphics_definition'][:200]}...\n"

    prompt = f"""Review this graphics segment for quality and completeness.

VO TEXT:
"{vo_text}"

GRAPHICS DEFINITION:
{graphics_definition}

REFERENCES ({len(references)}):
{refs_text}

{prev_text}

REVIEW CHECKLIST:
1. VO Alignment: Does the graphics definition match what the VO is describing?
2. References Found: Are there references for the visual elements?
3. References Relevant: Do the references actually show what's needed?
4. No Contradictions: Does this contradict any previous segments?
5. Specificity: Is it specific enough for an animator to implement?

For each criterion, evaluate as PASS or FAIL.

OUTPUT FORMAT:
<verdict>APPROVED</verdict>  OR  <verdict>REJECTED</verdict>

<vo_alignment>PASS or FAIL</vo_alignment>
<references_found>PASS or FAIL</references_found>
<references_relevant>PASS or FAIL</references_relevant>
<no_contradictions>PASS or FAIL</no_contradictions>
<specificity>PASS or FAIL</specificity>

<feedback>
[If REJECTED: Provide specific, actionable feedback on what needs improvement.
If APPROVED: Brief confirmation that all criteria met.]
</feedback>"""

    chain.add_message(role="user", content=prompt)
    response = chain.run()

    # Parse response
    if isinstance(response, dict):
        response = response.get("content", str(response))

    parsed = chain.extract_text_in_tags(response)

    verdict = parsed.get("verdict", "REJECTED").strip().upper()
    feedback = parsed.get("feedback", "")

    # Extract individual criteria
    criteria = {
        "vo_alignment": parsed.get("vo_alignment", "FAIL"),
        "references_found": parsed.get("references_found", "FAIL"),
        "references_relevant": parsed.get("references_relevant", "FAIL"),
        "no_contradictions": parsed.get("no_contradictions", "PASS"),
        "specificity": parsed.get("specificity", "FAIL"),
    }

    # Determine final verdict based on criteria
    all_pass = all(c == "PASS" for c in criteria.values())
    final_verdict = "approved" if (verdict == "APPROVED" and all_pass) else "rejected"

    # Format result message
    criteria_summary = "\n".join([f"  {k}: {v}" for k, v in criteria.items()])

    result_message = f"""{verdict}

Criteria Results:
{criteria_summary}

Feedback:
{feedback}
"""

    # Check for cross-segment issues
    cross_segment_issue = None
    if criteria["no_contradictions"] == "FAIL":
        cross_segment_issue = f"Contradiction detected: {feedback}"

    return Command(
        update={
            "review_verdict": final_verdict,
            "review_feedback": feedback,
            "cross_segment_issue": cross_segment_issue
        },
        goto="agent"
    )


@tool
def finalize_segment(
    state: Annotated[Dict, InjectedState]
) -> Command[str]:
    """
    Marks this segment as completed and ready to return to Slide Supervisor.

    Prerequisites:
    - review_verdict must be "approved"
    - graphics_definition must exist
    - references list must not be empty

    Returns:
        Success confirmation
    """
    review_verdict = state.get("review_verdict", "pending")
    graphics_definition = state.get("graphics_definition", "")
    references = state.get("references", [])

    # Validate prerequisites
    if review_verdict != "approved":
        return Command(
            update={},
            goto="agent"
        )

    if not graphics_definition:
        return "Error: Cannot finalize - no graphics definition exists."

    if not references:
        return "Error: Cannot finalize - no references found."

    # Add timestamp
    timestamp = time.time()

    result_message = f"""✓ Segment finalized successfully!

Summary:
- Graphics definition: {len(graphics_definition)} characters
- References: {len(references)}
- Status: COMPLETED
- Timestamp: {timestamp}

Ready to return to Slide Supervisor.
"""

    # No state update needed - supervisor will handle it
    return result_message


# ============================================================================
# Helper Functions
# ============================================================================

def extract_visual_elements(graphics_definition: str) -> List[str]:
    """
    Extract visual elements from a graphics definition.

    Simplified extraction - looks for bullet points in VISUAL ELEMENTS section.
    """
    elements = []

    # Find VISUAL ELEMENTS section
    if "VISUAL ELEMENTS:" in graphics_definition:
        section = graphics_definition.split("VISUAL ELEMENTS:")[1]
        if "PRESENTATION:" in section:
            section = section.split("PRESENTATION:")[0]

        # Extract bullet points
        lines = section.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("-"):
                element = line.lstrip("- ").split(":")[0].strip()
                if element:
                    elements.append(element)

    # Fallback: if no elements found, return a generic one
    if not elements:
        elements = ["Graphics definition content"]

    return elements
