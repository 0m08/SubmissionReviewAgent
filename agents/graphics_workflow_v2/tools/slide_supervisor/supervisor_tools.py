"""
Slide Supervisor Tools (Level 1)

Tools for the Slide Supervisor to orchestrate the entire graphics definition workflow.
"""

from typing import Annotated, Dict, List, Any
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_core.messages import ToolMessage

from agents.graphics_workflow_v2.state.schemas import GraphicsSlideState, SegmentData, ReferenceData
from agents.graphics_workflow_v2.config.settings import (
    SEGMENTATION_CHAIN_MODEL,
    OUTPUT_FORMAT,
    INCLUDE_METADATA,
    AUTO_FLAG_ON_MAX_ITERATIONS,
    SEARCH_K,
)
from modules.chain import Chain
from agents.graphics_workflow_v2.agents.segment_processor import (
    run_segment_processor,
    extract_segment_data_from_state
)
import time
import re


segment_slide_prompt = """You are an expert instructional designer. Your task is to split the slide content into individual voiceover (VO) segments using sentence-based segmentation. Each complete sentence becomes one segment.

This is the slide content that needs to be segmented:

Slide Content:
{slide_chunk}

RULES:

1. Sentence-based splitting:
   - Split only at real sentence boundaries: ".", "!", "?"
   - One sentence = one <segment>
   - Do not split mid-sentence
   - Do not merge multiple sentences

2. Text must be copied exactly without any change:
   - No paraphrasing, rewording, or summarizing
   - Preserve all punctuation, capitalization, numbering, technical terms, etc.

3. Full coverage:
   - Include every sentence from the slide
   - Segments must appear in original order

EXAMPLES:

Example 1:
Slide Content: "The low-pressure gauge is on the left side of the manifold set and is typically blue. It measures the pressure on the low side, suction side, of the system."
Segment 1: "The low-pressure gauge is on the left side of the manifold set and is typically blue."
Segment 2: "It measures the pressure on the low side, suction side, of the system."

Example 2:
Slide Content: "First, connect the blue hose to the low-pressure port. Then, open the valve slowly. Finally, read the pressure gauge."
Segment 1: "First, connect the blue hose to the low-pressure port."
Segment 2: "Then, open the valve slowly."
Segment 3: "Finally, read the pressure gauge."

Example 3:
Slide Content: "What is a manifold gauge? A manifold gauge measures pressure in HVAC systems. It has two gauges: one for high pressure and one for low pressure."
Segment 1: "What is a manifold gauge?"
Segment 2: "A manifold gauge measures pressure in HVAC systems."
Segment 3: "It has two gauges: one for high pressure and one for low pressure."

OUTPUT FORMAT:
You must always output segments in the following XML format. Each segment must be wrapped in <segment> tags:

<segments>
<segment>First sentence here - copied exactly from the slide</segment>
<segment>Second sentence here - copied exactly from the slide</segment>
<segment>Third sentence here - copied exactly from the slide</segment>
...
</segments>
"""


@tool
def segment_slide(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[Dict, InjectedState]
) -> Command:
    """
    Segments the slide content into individual VO (voiceover) moments.

    Returns:
        Success message with number of segments found
    """
    slide_chunk = state.get("slide_chunk", "")
    
    print(f"\n   {'─'*60}")
    print(f"   📋 Tool: segment_slide ▶ START")
    print(f"   {'─'*60}")
    print(f"      📝 Input: Slide chunk ({len(slide_chunk)} chars)")
    print(f"      Content: {slide_chunk}")

    if not slide_chunk:
        print("❌ Error: No slide content to segment")
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        "Error: No slide content to segment",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )

    # Create segmentation chain
    chain = Chain(
        llm=SEGMENTATION_CHAIN_MODEL,
        tags=["segments"]
    )

    prompt = segment_slide_prompt.format(
        slide_chunk=slide_chunk
    )

    chain.add_message(role="user", content=prompt)
    response = chain.run()

    # Parse response (Chain auto-extracts tagged sections)
    if isinstance(response, dict):
        parsed = response
    else:
        parsed = chain.extract_text_in_tags(str(response))

    segments_text = parsed.get("segments", parsed.get("text", ""))

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
            if line.strip()
        ]

    # Validate segments
    if not vo_segments:
        return Command(
            update={
                "status": "failed",
                "flags": ["Segmentation failed - no segments could be extracted"],
                "messages": [
                    ToolMessage(
                        "Error: Segmentation failed - no segments could be extracted",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )

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
    
    print(f"\n   {'─'*60}")
    print(f"   📋 Tool: segment_slide ◀ END")
    print(f"   {'─'*60}")
    print(f"      ✅ Output: {len(vo_segments)} VO segments created")
    for i, seg in enumerate(vo_segments):
        print(f"         {i}. \"{seg}\"")
    print(f"   {'─'*60}\n")

    return Command(
        update={
            "vo_segments": vo_segments,
            "segments": segments_data,
            "segmentation_method": "auto",
            "status": "processing_segments",
            "messages": [
                ToolMessage(
                    result_message,
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool
def process_segment(
    segment_index: int,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[Dict, InjectedState]
) -> Command:
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
            update={
                "messages": [
                    ToolMessage(
                        f"Error: Invalid segment index {segment_index}",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )

    vo_text = vo_segments[segment_index]
    
    print(f"\n   {'─'*60}")
    print(f"   🔧 Tool: process_segment ▶ START (segment {segment_index})")
    print(f"   {'─'*60}")
    print(f"      🎙️  VO: \"{vo_text}\"")

    # Get previous segments for context
    previous_segments = [seg for seg in segments if seg["segment_index"] < segment_index and seg["status"] == "completed"]
    print(f"      📚 Context: {len(previous_segments)} completed segments available")
    print(f"      ♻️  Available refs for reuse: {len(all_references)}")

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
            search_k=state.get("search_k", SEARCH_K),
            course_context=state.get("course_context"),
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
        
        status_icon = "✅" if segment_data['status'] == 'completed' else "⚠️"
        print(f"\n   {'─'*60}")
        print(f"   🔧 Tool: process_segment ◀ END (segment {segment_index})")
        print(f"   {'─'*60}")
        print(f"      {status_icon} Status: {segment_data['status']}")
        print(f"      🔍 Review verdict: {segment_data['review_verdict']}")
        print(f"      🔄 Iterations: {segment_data['iteration_count']}")
        print(f"      🖼️  References: {len(segment_data['references'])}")
        if segment_data['references']:
            for i, ref in enumerate(segment_data['references'], 1):
                print(f"         {i}. {ref.get('title', 'Untitled')} (score: {ref.get('relevance_score', 0):.2f})")
        print(f"      📝 Definition: {len(segment_data['graphics_definition'])} chars")
        print(f"   {'─'*60}\n")

        return Command(
            update={
                "segments": updated_segments,
                "current_segment_index": segment_index,
                "all_references": updated_all_references,
                "flags": new_flags,
                "messages": [
                    ToolMessage(
                        result_message,
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )

    except Exception as e:
        # Handle processing failure
        print(f"\n   {'─'*60}")
        print(f"   🔧 Tool: process_segment ◀ FAILED (segment {segment_index})")
        print(f"   {'─'*60}")
        print(f"      ❌ Error: {str(e)}")
        print(f"   {'─'*60}\n")
        new_flags = current_flags.copy()
        new_flags.append(f"Segment {segment_index}: Processing failed - {str(e)}")

        return Command(
            update={
                "flags": new_flags,
                "messages": [
                    ToolMessage(
                        f"Error: Segment {segment_index} processing failed - {str(e)}",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )


# @tool
# def flag_for_human_review(
#     issue_description: str,
#     affected_segments: str,
#     tool_call_id: Annotated[str, InjectedToolCallId],
#     state: Annotated[Dict, InjectedState]
# ) -> Command:
#     """
#     Flags an issue that requires human intervention.

#     Use when:
#     - Cross-segment conflicts detected
#     - A segment repeatedly fails review
#     - Critical resources not found
#     - Ambiguity that AI cannot resolve

#     Args:
#         issue_description: Clear description of the issue
#         affected_segments: Comma-separated segment indices (e.g., "1,3,5")

#     Returns:
#         Confirmation message
#     """
#     current_flags = state.get("flags", [])

#     # Parse affected segments
#     try:
#         seg_indices = [int(s.strip()) for s in affected_segments.split(",") if s.strip()]
#         seg_list = ", ".join([str(i) for i in seg_indices])
#     except:
#         seg_list = affected_segments

#     flag_message = f"Segments [{seg_list}]: {issue_description}"
#     updated_flags = current_flags + [flag_message]

#     result_message = f"""⚠ Issue flagged for human review:

# Affected segments: {seg_list}
# Issue: {issue_description}

# Total flags: {len(updated_flags)}
# """

#     return Command(
#         update={
#             "flags": updated_flags,
#             "messages": [
#                 ToolMessage(
#                     result_message,
#                     tool_call_id=tool_call_id
#                 )
#             ]
#         }
#     )


@tool
def finalize_slide_graphics(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[Dict, InjectedState]
) -> Command:
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
    
    print(f"\n   {'─'*60}")
    print(f"   📦 Tool: finalize_slide_graphics ▶ START")
    print(f"   {'─'*60}")
    print(f"      📊 Segments to finalize: {len(segments)}")
    print(f"      ⚠️  Flags: {len(flags)}")

    if not segments:
        return Command(
            update={
                "status": "failed",
                "messages": [
                    ToolMessage(
                        "Error: No segments to finalize",
                        tool_call_id=tool_call_id
                    )
                ]
            }
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
    
    print(f"\n   {'─'*60}")
    print(f"   📦 Tool: finalize_slide_graphics ◀ END")
    print(f"   {'─'*60}")
    print(f"      ✅ Final definition generated!")
    print(f"      📄 Format: {OUTPUT_FORMAT}")
    print(f"      📏 Length: {len(final_definition)} chars")
    print(f"      🧩 Segments: {total_segments} ({completed_segments} completed)")
    print(f"      🖼️  Total references: {total_references} ({unique_references} unique)")
    print(f"      ⚠️  Flags: {len(flags)}")
    print(f"   {'─'*60}\n")

    return Command(
        update={
            "final_definition": final_definition,
            "status": "completed",
            "messages": [
                ToolMessage(
                    result_message,
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


# ============================================================================
# Helper Functions - Output Formatting
# ============================================================================

def _format_graphics_definition_for_sheet(graphics_def: str) -> str:
    """
    Transform graphics definition XML tags to sheet-friendly format.
    
    Replaces:
    - <description>...</description> with Description: ...
    - <selected_images>...</selected_images> with Images to use for this segment: ...
    - Adds line gap between sections
    """
    # Extract description content
    description_match = re.search(r'<description>(.*?)</description>', graphics_def, re.DOTALL)
    description_content = description_match.group(1).strip() if description_match else ""
    
    # Extract selected_images content
    images_match = re.search(r'<selected_images>(.*?)</selected_images>', graphics_def, re.DOTALL)
    images_content = images_match.group(1).strip() if images_match else ""
    
    # Build formatted output
    formatted = ""
    
    if description_content:
        formatted += f"Description:\n{description_content}\n"
    
    if images_content:
        if description_content:
            formatted += "\n"  # Line gap between sections
        formatted += f"Images to use for this segment:\n{images_content}\n"
    
    # If no XML tags found, return original (fallback)
    if not description_match and not images_match:
        return graphics_def
    
    return formatted


def format_as_markdown(segments: List[SegmentData], slide_chunk: str, flags: List[str]) -> str:
    """Format the final definition as designer-focused output."""
    output = ""
    separator = "=" * 80

    for i, seg in enumerate(segments):
        output += f"{separator}\n"
        output += f"SEGMENT {i + 1}\n"
        output += f"{separator}\n"
        output += f"VO: \"{seg['vo_text']}\"\n\n"

        graphics_def = seg.get('graphics_definition', '')
        
        if graphics_def:
            # Transform XML tags to sheet-friendly format
            formatted_def = _format_graphics_definition_for_sheet(graphics_def)
            output += formatted_def
            output += "\n\n"
        else:
            # Fallback: try to parse old format if refine_graphics_with_images wasn't called
            visuals, action = _parse_graphics_definition(graphics_def)
            output += "VISUALS:\n"
            if visuals:
                for v in visuals:
                    output += f"• {v}\n"
            else:
                output += "• [No graphics definition available]\n"
            output += "\n"
            
            if action:
                output += "ACTION:\n"
                output += f"• {action}\n"
                output += "\n"
            
            # Images section - prominent URLs
            if seg['references']:
                output += "IMAGES:\n"
                for j, ref in enumerate(seg['references']):
                    reuse_marker = " (♻️ reuse)" if ref.get('reused_from_segment') is not None else ""
                    output += f"[{j + 1}] {ref['title']}{reuse_marker}\n"
                    output += f"    {ref['url']}\n"
                    output += "\n"
            else:
                output += "IMAGES:\n"
                output += "[No references found]\n\n"

    return output


def _parse_graphics_definition(graphics_def: str) -> tuple:
    """
    Parse graphics definition to extract visual elements and presentation.
    
    Returns:
        tuple: (list of visual elements, presentation string)
    """
    visuals = []
    action = ""
    
    # Extract VISUAL ELEMENTS section
    if "VISUAL ELEMENTS:" in graphics_def:
        section = graphics_def.split("VISUAL ELEMENTS:")[1]
        if "PRESENTATION:" in section:
            section = section.split("PRESENTATION:")[0]
        
        # Extract bullet points
        for line in section.split("\n"):
            line = line.strip()
            if line.startswith("-"):
                element = line.lstrip("- ").strip()
                # Remove "Element N:" prefix if present
                if element.lower().startswith("element") and ":" in element:
                    element = element.split(":", 1)[1].strip()
                if element:
                    visuals.append(element)
    
    # Extract PRESENTATION section
    if "PRESENTATION:" in graphics_def:
        section = graphics_def.split("PRESENTATION:")[1]
        if "REFERENCES NEEDED:" in section:
            section = section.split("REFERENCES NEEDED:")[0]
        action = section.strip()
        # Clean up - take first paragraph or first few sentences
        if action:
            lines = [l.strip() for l in action.split("\n") if l.strip()]
            action = " → ".join(lines[:3]) if len(lines) > 1 else lines[0] if lines else ""
    
    return visuals, action


def format_as_plain_text(segments: List[SegmentData], slide_chunk: str, flags: List[str]) -> str:
    """Format the final definition as plain text (same as markdown for consistency)."""
    # Use the same designer-focused format
    return format_as_markdown(segments, slide_chunk, flags)


def format_as_structured_dict(segments: List[SegmentData], slide_chunk: str, flags: List[str]) -> Dict:
    """Format the final definition as structured dictionary (for JSON)."""
    # Build clean segment data for JSON
    clean_segments = []
    for seg in segments:
        visuals, action = _parse_graphics_definition(seg.get('graphics_definition', ''))
        clean_segments.append({
            "segment": seg.get('segment_index', 0) + 1,
            "vo": seg.get('vo_text', ''),
            "visuals": visuals,
            "action": action,
            "images": [
                {
                    "title": ref.get('title', ''),
                    "url": ref.get('url', ''),
                    "reused": ref.get('reused_from_segment') is not None
                }
                for ref in seg.get('references', [])
            ]
        })
    
    return {
        "segments": clean_segments
    }
