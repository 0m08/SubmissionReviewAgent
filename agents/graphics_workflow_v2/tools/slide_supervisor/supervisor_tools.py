"""
Slide Supervisor Tools (Level 1)

Tools for the Slide Supervisor to orchestrate the entire graphics definition workflow.
"""

from typing import Annotated, Dict, List, Any
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from agents.graphics_workflow_v2.state.schemas import GraphicsSlideState, SegmentData, ReferenceData
from agents.graphics_workflow_v2.config.settings import (
    SEGMENTATION_CHAIN_MODEL,
    SEGMENTATION_STRATEGY,
    MIN_SEGMENT_LENGTH,
    MAX_SEGMENT_LENGTH,
    MAX_SEGMENTS_PER_SLIDE,
    OUTPUT_FORMAT,
    INCLUDE_METADATA,
    AUTO_FLAG_ON_MAX_ITERATIONS,
)
from modules.chain import Chain
from agents.graphics_workflow_v2.agents.segment_processor import (
    run_segment_processor,
    extract_segment_data_from_state
)
import time


@tool
def segment_slide(
    state: Annotated[Dict, InjectedState]
) -> Command[str]:
    """
    Segments the slide content into individual VO (voiceover) moments.

    Uses LLM to intelligently parse the slide into logical VO segments based on:
    - Sentence boundaries
    - Concept shifts
    - Natural speaking rhythm
    - Logical visual moments

    Returns:
        Success message with number of segments found
    """
    slide_chunk = state.get("slide_chunk", "")

    if not slide_chunk:
        return Command(
            update={},
            goto="agent"
        )

    # Create segmentation chain
    chain = Chain(
        llm=SEGMENTATION_CHAIN_MODEL,
        tags=["segments"],
        use_xml_checker=True
    )

    strategy_guidance = {
        "concept_based": "Segment by major concepts or ideas being explained",
        "sentence_based": "Segment by sentence boundaries (1-3 sentences per segment)",
        "time_based": "Segment by estimated speaking time (10-15 seconds per segment)"
    }

    prompt = f"""Segment this educational slide content into individual voiceover (VO) moments.

SLIDE CONTENT:
{slide_chunk}

SEGMENTATION STRATEGY: {SEGMENTATION_STRATEGY}
{strategy_guidance.get(SEGMENTATION_STRATEGY, "")}

GUIDELINES:
- Each segment should represent ONE distinct moment in the voiceover
- Segments should be {MIN_SEGMENT_LENGTH}-{MAX_SEGMENT_LENGTH} characters
- Maximum {MAX_SEGMENTS_PER_SLIDE} segments total
- Each segment should be a complete thought or concept
- Consider natural speaking rhythm and visual breaks
- Maintain the original text exactly (don't paraphrase)

OUTPUT FORMAT:
<segments>
<segment>First VO moment text here</segment>
<segment>Second VO moment text here</segment>
<segment>Third VO moment text here</segment>
...
</segments>

Segment the slide now:"""

    chain.add_message(role="user", content=prompt)
    response = chain.run()

    # Parse response
    if isinstance(response, dict):
        response = response.get("content", str(response))

    parsed = chain.extract_text_in_tags(response)
    segments_text = parsed.get("segments", "")

    # Extract individual segments
    vo_segments = []
    if "<segment>" in segments_text:
        # Extract each segment tag
        import re
        segment_matches = re.findall(r'<segment>(.*?)</segment>', segments_text, re.DOTALL)
        vo_segments = [seg.strip() for seg in segment_matches if seg.strip()]
    else:
        # Fallback: split by newlines
        vo_segments = [
            line.strip()
            for line in segments_text.split("\n")
            if line.strip() and len(line.strip()) >= MIN_SEGMENT_LENGTH
        ]

    # Validate segments
    if not vo_segments:
        return Command(
            update={
                "status": "failed",
                "flags": ["Segmentation failed - no segments could be extracted"]
            },
            goto="agent"
        )

    if len(vo_segments) > MAX_SEGMENTS_PER_SLIDE:
        vo_segments = vo_segments[:MAX_SEGMENTS_PER_SLIDE]

    # Initialize segments data
    segments_data = [
        {
            "segment_index": i,
            "vo_text": vo_text,
            "graphics_definition": "",
            "references": [],
            "status": "pending",
            "review_verdict": "",
            "review_feedback": "",
            "iteration_count": 0,
            "timestamp": None
        }
        for i, vo_text in enumerate(vo_segments)
    ]

    result_message = f"""Successfully segmented slide into {len(vo_segments)} VO moments:

"""
    for i, seg in enumerate(vo_segments[:5]):
        result_message += f"{i}. {seg[:80]}{'...' if len(seg) > 80 else ''}\n"

    if len(vo_segments) > 5:
        result_message += f"... and {len(vo_segments) - 5} more segments.\n"

    return Command(
        update={
            "vo_segments": vo_segments,
            "segments": segments_data,
            "segmentation_method": "auto",
            "status": "processing_segments"
        },
        goto="agent"
    )


@tool
def process_segment(
    segment_index: int,
    state: Annotated[Dict, InjectedState]
) -> Command[str]:
    """
    Processes a single VO segment through the define→search→review workflow.

    This tool creates a Segment Processor agent instance and runs it to completion.
    The segment processor will handle its own revision loop.

    Args:
        segment_index: Which segment to process (0-indexed)

    Returns:
        Summary of segment processing result
    """
    vo_segments = state.get("vo_segments", [])
    slide_chunk = state.get("slide_chunk", "")
    segments = state.get("segments", [])
    all_references = state.get("all_references", {})
    current_flags = state.get("flags", [])
    max_iterations = state.get("max_iterations_per_segment", 3)

    # Validate segment index
    if segment_index >= len(vo_segments):
        return Command(
            update={},
            goto="agent"
        )

    vo_text = vo_segments[segment_index]

    # Get previous segments for context
    previous_segments = [seg for seg in segments if seg["segment_index"] < segment_index and seg["status"] == "completed"]

    # Run Segment Processor
    try:
        processor_result = run_segment_processor(
            segment_index=segment_index,
            vo_text=vo_text,
            slide_chunk=slide_chunk,
            previous_segments=previous_segments,
            available_references=all_references,
            drive=state.get("drive"),
            filters=state.get("filters"),
            root_folder_id=state.get("root_folder_id"),
        )

        # Extract segment data
        segment_data = extract_segment_data_from_state(processor_result)

        # Check for flags
        new_flags = current_flags.copy()

        # Check if max iterations reached
        if AUTO_FLAG_ON_MAX_ITERATIONS and segment_data["iteration_count"] >= max_iterations:
            if segment_data["review_verdict"] != "approved":
                new_flags.append(
                    f"Segment {segment_index}: Reached max iterations ({max_iterations}) without approval"
                )

        # Check for cross-segment issues
        if processor_result.get("cross_segment_issue"):
            new_flags.append(
                f"Segment {segment_index}: {processor_result['cross_segment_issue']}"
            )
            segment_data["status"] = "flagged"

        # Update all_references with new references
        updated_all_references = all_references.copy()
        for ref in segment_data["references"]:
            ref_id = ref["reference_id"]
            if ref_id not in updated_all_references:
                updated_all_references[ref_id] = ref

        # Update segments list
        updated_segments = segments.copy()
        updated_segments[segment_index] = segment_data

        # Format result message
        result_message = f"""Segment {segment_index} processing complete:

Status: {segment_data['status'].upper()}
Review: {segment_data['review_verdict'].upper()}
Iterations: {segment_data['iteration_count']}
References found: {len(segment_data['references'])}
Graphics definition length: {len(segment_data['graphics_definition'])} chars

VO: "{vo_text[:100]}..."
"""

        if segment_data["status"] == "flagged":
            result_message += f"\n⚠ WARNING: This segment has been flagged for human review\n"

        return Command(
            update={
                "segments": updated_segments,
                "current_segment_index": segment_index,
                "all_references": updated_all_references,
                "flags": new_flags,
            },
            goto="agent"
        )

    except Exception as e:
        # Handle processing failure
        new_flags = current_flags.copy()
        new_flags.append(f"Segment {segment_index}: Processing failed - {str(e)}")

        return Command(
            update={"flags": new_flags},
            goto="agent"
        )


@tool
def flag_for_human_review(
    issue_description: str,
    affected_segments: str,
    state: Annotated[Dict, InjectedState]
) -> Command[str]:
    """
    Flags an issue that requires human intervention.

    Use when:
    - Cross-segment conflicts detected
    - A segment repeatedly fails review
    - Critical resources not found
    - Ambiguity that AI cannot resolve

    Args:
        issue_description: Clear description of the issue
        affected_segments: Comma-separated segment indices (e.g., "1,3,5")

    Returns:
        Confirmation message
    """
    current_flags = state.get("flags", [])

    # Parse affected segments
    try:
        seg_indices = [int(s.strip()) for s in affected_segments.split(",") if s.strip()]
        seg_list = ", ".join([str(i) for i in seg_indices])
    except:
        seg_list = affected_segments

    flag_message = f"Segments [{seg_list}]: {issue_description}"
    updated_flags = current_flags + [flag_message]

    result_message = f"""⚠ Issue flagged for human review:

Affected segments: {seg_list}
Issue: {issue_description}

Total flags: {len(updated_flags)}
"""

    return Command(
        update={"flags": updated_flags},
        goto="agent"
    )


@tool
def finalize_slide_graphics(
    state: Annotated[Dict, InjectedState]
) -> Command[str]:
    """
    Assembles the final formatted graphics definition from all segments.

    Combines all segment definitions into a cohesive, formatted document
    that includes:
    - VO text for each moment
    - Graphics instructions
    - Reference URLs with metadata
    - Timing/sequencing notes

    Returns:
        Success message with summary
    """
    segments = state.get("segments", [])
    slide_chunk = state.get("slide_chunk", "")
    flags = state.get("flags", [])

    if not segments:
        return Command(
            update={"status": "failed"},
            goto="agent"
        )

    # Generate final definition based on OUTPUT_FORMAT
    if OUTPUT_FORMAT == "markdown":
        final_definition = format_as_markdown(segments, slide_chunk, flags)
    elif OUTPUT_FORMAT == "structured_json":
        import json
        final_definition = json.dumps(format_as_structured_dict(segments, slide_chunk, flags), indent=2)
    else:  # plain_text
        final_definition = format_as_plain_text(segments, slide_chunk, flags)

    # Calculate statistics
    total_segments = len(segments)
    completed_segments = sum(1 for seg in segments if seg["status"] == "completed")
    total_references = sum(len(seg["references"]) for seg in segments)
    unique_references = len(state.get("all_references", {}))

    result_message = f"""✓ Slide graphics definition finalized!

Summary:
- Total segments: {total_segments}
- Completed: {completed_segments}
- Flagged: {len(flags)}
- Total references: {total_references}
- Unique references: {unique_references}

Output format: {OUTPUT_FORMAT}
Definition length: {len(final_definition)} characters
"""

    return Command(
        update={
            "final_definition": final_definition,
            "status": "completed"
        },
        goto="agent"
    )


# ============================================================================
# Helper Functions - Output Formatting
# ============================================================================

def format_as_markdown(segments: List[SegmentData], slide_chunk: str, flags: List[str]) -> str:
    """Format the final definition as Markdown."""
    output = "# GRAPHICS DEFINITION\n\n"

    # Metadata
    if INCLUDE_METADATA:
        output += "## Metadata\n\n"
        output += f"- **Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        output += f"- **Total Segments:** {len(segments)}\n"
        total_refs = sum(len(seg['references']) for seg in segments)
        output += f"- **Total References:** {total_refs}\n"
        if flags:
            output += f"- **Flags:** {len(flags)} (see bottom)\n"
        output += "\n---\n\n"

    # Segments
    for i, seg in enumerate(segments):
        output += f"## SEGMENT {i + 1} of {len(segments)}\n\n"
        output += f"**When VO:** \"{seg['vo_text']}\"\n\n"

        output += "### GRAPHICS:\n\n"
        output += f"{seg['graphics_definition']}\n\n"

        if seg['references']:
            output += "### REFERENCES:\n\n"
            for j, ref in enumerate(seg['references']):
                output += f"**[{j + 1}] {ref['title']}**\n"
                output += f"- URL: {ref['url']}\n"
                output += f"- Type: {ref['type']}\n"
                if ref.get('timestamp'):
                    output += f"- Timestamp: {ref['timestamp']}\n"
                output += f"- Description: {ref['description']}\n"
                output += f"- Relevance: {ref['relevance_score']:.2f}\n"
                if ref.get('reused_from_segment') is not None:
                    output += f"- *Reused from Segment {ref['reused_from_segment']}*\n"
                output += "\n"

        output += "---\n\n"

    # Flags
    if flags:
        output += "## FLAGS FOR HUMAN REVIEW\n\n"
        for i, flag in enumerate(flags):
            output += f"{i + 1}. {flag}\n"
        output += "\n"

    return output


def format_as_plain_text(segments: List[SegmentData], slide_chunk: str, flags: List[str]) -> str:
    """Format the final definition as plain text."""
    output = "=" * 80 + "\n"
    output += "GRAPHICS DEFINITION\n"
    output += "=" * 80 + "\n\n"

    for i, seg in enumerate(segments):
        output += f"SEGMENT {i + 1}\n"
        output += f"When VO: \"{seg['vo_text']}\"\n\n"
        output += "GRAPHICS:\n"
        output += f"{seg['graphics_definition']}\n\n"

        if seg['references']:
            output += f"REFERENCES ({len(seg['references'])}):\n"
            for j, ref in enumerate(seg['references']):
                output += f"  [{j + 1}] {ref['title']} ({ref['type']})\n"
                output += f"      {ref['url']}\n"

        output += "\n" + "-" * 80 + "\n\n"

    if flags:
        output += "FLAGS:\n"
        for flag in flags:
            output += f"  - {flag}\n"

    return output


def format_as_structured_dict(segments: List[SegmentData], slide_chunk: str, flags: List[str]) -> Dict:
    """Format the final definition as structured dictionary (for JSON)."""
    return {
        "metadata": {
            "generated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "total_segments": len(segments),
            "total_references": sum(len(seg['references']) for seg in segments),
            "flags_count": len(flags)
        },
        "slide_content": slide_chunk,
        "segments": segments,
        "flags": flags
    }
