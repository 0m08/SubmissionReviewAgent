"""
Segment Processor Tools (Level 2)

Tools for the Segment Processor agent to create graphics definitions for individual VO segments.
"""

from typing import Annotated, Dict, List, Any
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_core.messages import ToolMessage
import time

from agents.graphics_workflow_v2.state.schemas import SegmentProcessorState, SegmentData, ReferenceData
from agents.graphics_workflow_v2.config.settings import (
    DEFINITION_CHAIN_MODEL,
    REVIEW_CHAIN_MODEL,
    MIN_DEFINITION_LENGTH,
    ENABLE_CROSS_SEGMENT_VALIDATION,
    SEARCH_K,
)
from agents.graphics_workflow_v2.config.settings import MIN_REFERENCES_PER_SEGMENT
from modules.chain import Chain
from agents.graphics_workflow_v2.agents.search_agent import run_search_agent
from agents.vector_store_image_search.graphics_retriever_agent import pil_to_base64_data_uri
from agents.graphics_workflow_v2.tools.search_agent.search_tools import load_image_from_drive_url
from services.llm_service import llm_with_retry
import re


@tool
def define_segment_graphics(
    instruction: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[Dict, InjectedState]
) -> Command:
    """
    Creates or revises the graphics definition for this VO segment.

    Generates visual elements that describe what to show.

    Args:
        instruction: Guidance ("create initial definition" or "revise based on feedback...")

    Returns:
        Summary of created/revised visual elements
    """
    vo_text = state.get("vo_text", "")
    slide_chunk = state.get("slide_chunk", "")
    previous_segments = state.get("previous_segments", [])
    review_feedback = state.get("review_feedback", "")
    current_definition = state.get("graphics_definition", "")
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 1)
    
    # Extract course context
    course_context = state.get("course_context", {})
    course_name = course_context.get("course_name", "") if course_context else ""
    topic_name = course_context.get("topic", "") if course_context else ""
    subtopic_name = course_context.get("subtopic", "") if course_context else ""
    
    is_revision = "revise" in instruction.lower()
    
    # Hard enforcement: Block revisions if max iterations reached
    if is_revision and iteration_count >= max_iterations:
        print(f"\n      {'─'*50}")
        print(f"      ⛔ Tool: define_segment_graphics ▶ BLOCKED")
        print(f"      {'─'*50}")
        print(f"         Max iterations ({max_iterations}) reached. Cannot revise further.")
        print(f"         Current iteration count: {iteration_count}")
        print(f"         You must call finalize_segment now.")
        
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        f"BLOCKED: Max iterations ({max_iterations}) reached. "
                        f"Cannot revise further. You must call finalize_segment now to complete this segment.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    action = "Creating"  # Always creating visual elements in new workflow
    print(f"\n      {'─'*50}")
    print(f"      ✏️  Tool: define_segment_graphics ▶ START")
    print(f"      {'─'*50}")
    print(f"         🎯 Action: {action} visual elements (1-3 essential elements)")
    print(f"         🎙️  VO: \"{vo_text}\"")

    # Create Chain
    chain = Chain(
        llm=DEFINITION_CHAIN_MODEL,
        tags=["visual_elements"]
    )

    # Build prompt
    if is_revision and review_feedback:
        prompt_type = "revision"
        specific_instruction = f"""REVISE the visual elements based on this feedback:
{review_feedback}

Current visual elements:
{current_definition}

Make specific improvements to address the feedback."""
    else:
        prompt_type = "initial"
        specific_instruction = "Create exactly 1–3 essential visual elements that represent what must appear on screen for this VO segment."

    prompt = f"""You are an expert educational motion-graphics designer. Your job is to extract the essential visual elements that directly support the meaning of the Voiceover sentence. Your output should specify the static visual elements that should appear on screen — never motion, timing, or presentation. Your output will be used to generate image-search queries for selecting the relevant images from a vectorstore, so every word must be concrete, specific, and visual based.

TASK: {specific_instruction}

Course Name: {course_name}

Topic Name: {topic_name}

Subtopic Name: {subtopic_name}

Voiceover text:
"{vo_text}"

FULL SLIDE CONTEXT:
{slide_chunk}

Instructions:

1. Describe WHAT to show, not HOW:
- Forbid motion or presentation words: zoom, fade, pan, highlight, transition, animate, movement, reveal, camera, angle.

2. Be concrete, specific, and technically correct:
- Use precise domain-specific terminology when relevant (e.g., HVAC, electrical, mechanical terms).
- Name the exact object (e.g., "evaporator coil diagram with labeled airflow arrows").

3. Keep each visual element simple and distinct:
- 5–10 words per element.
- No full sentences.
- No duplicated ideas or restated concepts.

Output Format:

Always give your output in the following XML format:

<visual_elements>
- Element 1 text
- Element 2 text
...
</visual_elements>
"""

    chain.add_message(role="user", content=prompt)
    response = chain.run()

    # Parse response (Chain handles tag extraction)
    if isinstance(response, dict):
        parsed = response
    else:
        parsed = chain.extract_text_in_tags(str(response))

    visual_elements = parsed.get("visual_elements", "")

    # Store visual elements 
    graphics_definition = f"""VISUAL ELEMENTS:
{visual_elements}"""

    # Validate length
    if len(graphics_definition.strip()) < MIN_DEFINITION_LENGTH:
        print(f"      ❌ Definition too short: {len(graphics_definition)} chars (min: {MIN_DEFINITION_LENGTH})")
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        f"Error: Graphics definition too short ({len(graphics_definition)} chars, minimum {MIN_DEFINITION_LENGTH})",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )

    # Update iteration count if this is a revision
    new_iteration_count = iteration_count + 1 if prompt_type == "revision" else iteration_count

    # Count elements (handle both "- " and "-" formats)
    element_count = len([line for line in visual_elements.split('\n') if line.strip().startswith('-')])

    # Format response
    result_message = f"""Visual elements {"created" if prompt_type == "initial" else "revised"}:

VISUAL ELEMENTS ({element_count} elements):
{visual_elements[:300]}{'...' if len(visual_elements) > 300 else ''}

Length: {len(graphics_definition)} characters
"""
    
    # Count elements for logging
    element_count = len([line for line in visual_elements.split('\n') if line.strip().startswith('-')])
    
    print(f"\n      {'─'*50}")
    print(f"      ✏️  Tool: define_segment_graphics ◀ END")
    print(f"      {'─'*50}")
    print(f"         ✅ Visual elements {'created' if prompt_type == 'initial' else 'revised'}")
    print(f"         📏 Length: {len(graphics_definition)} chars")
    print(f"         🎨 Visual elements: {element_count} items")
    print(f"         📝 Output:")
    print(f"            {graphics_definition}")
    print(f"      {'─'*50}\n")

    return Command(
        update={
            "graphics_definition": graphics_definition,
            "iteration_count": new_iteration_count,
            "messages": [
                ToolMessage(
                    result_message,
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool
def search_segment_references(
    instruction: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[Dict, InjectedState]
) -> Command:
    """
    Finds relevant visual references for this segment's voiceover sentence.

    Creates a Search Agent instance that will:
    - Check for reusable references from previous segments
    - Generate search queries from the VO sentence
    - Execute searches iteratively
    - Return best matching references

    Args:
        instruction: Search guidance from supervisor

    Returns:
        Summary of found references
    """
    vo_text = state.get("vo_text", "")
    slide_chunk = state.get("slide_chunk", "")
    available_references = state.get("available_references", {})
    drive = state.get("drive")
    
    print(f"\n      {'─'*50}")
    print(f"      🔍 Tool: search_segment_references ▶ START")
    print(f"      {'─'*50}")
    print(f"         🎙️  VO sentence: \"{vo_text}\"")
    print(f"         ♻️  Available refs for reuse: {len(available_references)}")

    if not vo_text:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        "Error: No VO text available to search references for",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )

    # Run Search Agent
    try:
        search_result = run_search_agent(
            vo_text=vo_text,
            slide_chunk=slide_chunk,
            available_references=available_references,
            drive=drive,
            filters=state.get("filters"),
            root_folder_id=state.get("root_folder_id"),
            search_k=state.get("search_k", SEARCH_K),
            course_context=state.get("course_context"),
            min_references=MIN_REFERENCES_PER_SEGMENT,
        )

        # Extract search results 
        all_search_results = search_result.get("search_results", [])
        selected_references = search_result.get("selected_references", [])

        # Merge selected_references (reused from previous segments) into search_results
        # Avoid duplicates by checking reference_id
        existing_ids = {ref.get("reference_id") for ref in all_search_results if ref.get("reference_id")}
        merged_results = all_search_results.copy()
        
        for ref in selected_references:
            ref_id = ref.get("reference_id")
            if ref_id and ref_id not in existing_ids:
                merged_results.append(ref)
                existing_ids.add(ref_id)
        
        # Update all_search_results to include merged results
        all_search_results = merged_results

        # Format response - report based on merged search_results
        new_search_count = len(search_result.get("search_results", []))
        reused_count = len(selected_references)
        total_count = len(all_search_results)
        
        result_message = f"""Search completed. Found {total_count} total reference(s) for refinement:
- {new_search_count} new search result(s)
- {reused_count} reusable reference(s) from previous segments

"""
        # Show top results from merged results
        for i, ref in enumerate(all_search_results):
            reused_marker = " (♻️ reused)" if ref.get('reused_from_segment') is not None else ""
            result_message += f"{i+1}. {ref['title']} ({ref['type']}){reused_marker}\n"
            result_message += f"   Relevance: {ref.get('relevance_score', 'N/A')}\n"
            result_message += f"   {ref.get('description', '')[:100]}...\n\n"
        
        result_message += f"\nAll {total_count} references (including {reused_count} reused) are now available for refine_graphics_with_images"
        
        print(f"\n      {'─'*50}")
        print(f"      🔍 Tool: search_segment_references ◀ END")
        print(f"      {'─'*50}")
        print(f"         ✅ New search results: {new_search_count}")
        print(f"         ♻️  Reusable references: {reused_count}")
        print(f"         📊 Total available: {total_count}")
        for i, ref in enumerate(all_search_results[:5], 1):
            reused = " (♻️ reused)" if ref.get('reused_from_segment') is not None else ""
            print(f"            {i}. {ref.get('title', 'Untitled')} | score: {ref.get('relevance_score', 'N/A')}{reused}")
        if len(all_search_results) > 5:
            print(f"            ... and {len(all_search_results) - 5} more results")
        print(f"      {'─'*50}\n")

        return Command(
            update={
                "search_results": all_search_results,  # Store merged results (new + reused) for refine tool
                "messages": [
                    ToolMessage(
                        result_message,
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )

    except Exception as e:
        print(f"\n      {'─'*50}")
        print(f"      🔍 Tool: search_segment_references ◀ FAILED")
        print(f"      {'─'*50}")
        print(f"         ❌ Error: {str(e)}")
        print(f"      {'─'*50}\n")
        error_message = f"Search failed: {str(e)}\nPlease try again or check the VO sentence."
        return Command(
            update={
                "references": [],
                "messages": [
                    ToolMessage(
                        error_message,
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )


# @tool
# def review_segment_quality(
#     tool_call_id: Annotated[str, InjectedToolCallId],
#     state: Annotated[Dict, InjectedState]
# ) -> Command:
#     """
#     Reviews the segment's graphics definition and references for quality.

#     Checks against criteria:
#     ✓ Definition aligns with VO text
#     ✓ At least 1 relevant reference per visual element
#     ✓ No contradictions with previous segments
#     ✓ Specific enough for animator to implement
#     ✓ All references are accessible

#     Returns:
#         "APPROVED" or "REJECTED: [detailed feedback]"
#     """
#     vo_text = state.get("vo_text", "")
#     graphics_definition = state.get("graphics_definition", "")
#     references = state.get("references", [])
#     previous_segments = state.get("previous_segments", [])
#     slide_chunk = state.get("slide_chunk", "")
    
#     print(f"\n      {'─'*50}")
#     print(f"      🔍 Tool: review_segment_quality ▶ START")
#     print(f"      {'─'*50}")
#     print(f"         🎙️  VO: \"{vo_text}\"")
#     print(f"         📝 Definition: {len(graphics_definition)} chars")
#     print(f"         🖼️  References: {len(references)}")
#     print(f"         📚 Previous segments: {len(previous_segments)}")

#     # Check prerequisites
#     if not graphics_definition:
#         return Command(
#             update={
#                 "review_verdict": "rejected",
#                 "review_feedback": "No graphics definition to review. Please create a definition first.",
#                 "messages": [
#                     ToolMessage(
#                         "Review failed: No graphics definition to review. Please create a definition first.",
#                         tool_call_id=tool_call_id
#                     )
#                 ]
#             }
#         )

#     if not references:
#         return Command(
#             update={
#                 "review_verdict": "rejected",
#                 "review_feedback": "No references found. Please search for references before reviewing.",
#                 "messages": [
#                     ToolMessage(
#                         "Review failed: No references found. Please search for references before reviewing.",
#                         tool_call_id=tool_call_id
#                     )
#                 ]
#             }
#         )

#     # Create review chain
#     chain = Chain(
#         llm=REVIEW_CHAIN_MODEL,
#         tags=["verdict", "vo_alignment", "references_found", "references_relevant",
#               "no_contradictions", "feedback"]
#     )

#     # Format references for review
#     refs_text = ""
#     for i, ref in enumerate(references):
#         refs_text += f"{i+1}. {ref['title']} ({ref['type']}) - Relevance: {ref['relevance_score']:.2f}\n"
#         refs_text += f"   {ref['description']}\n"

#     # Format previous segments
#     prev_text = ""
#     if previous_segments and ENABLE_CROSS_SEGMENT_VALIDATION:
#         prev_text = "\n\nPREVIOUS SEGMENTS (check for contradictions):\n"
#         for seg in previous_segments[-2:]:
#             prev_text += f"Segment {seg['segment_index']}: {seg['graphics_definition'][:200]}...\n"

#     prompt = f"""Review this graphics segment for quality and completeness.

# VO TEXT:
# "{vo_text}"

# FULL SLIDE CONTEXT:
# {slide_chunk}

# GRAPHICS DEFINITION:
# {graphics_definition}

# REFERENCES ({len(references)}):
# {refs_text}

# {prev_text}

# REVIEW CHECKLIST (evaluate each as PASS, WARNING, or FAIL):
# 1. VO Alignment (CRITICAL): Does the graphics definition match what the VO is describing?
#    - PASS: Clearly aligns with VO
#    - WARNING: Mostly aligns but minor gaps
#    - FAIL: Doesn't match or contradicts VO

# 2. References Found (CRITICAL): Are there references for the visual elements?
#    - PASS: References found for all/most visual elements
#    - WARNING: Some references missing but core elements covered
#    - FAIL: No references or critical elements missing

# 3. References Relevant (IMPORTANT): Do the references actually show what's needed?
#    - PASS: References are highly relevant and useful
#    - WARNING: References are somewhat relevant, usable but not perfect
#    - FAIL: References are irrelevant or misleading

# 4. No Contradictions (IMPORTANT): Does this contradict any previous segments?
#    - PASS: No contradictions
#    - WARNING: Minor inconsistencies that don't break continuity
#    - FAIL: Major contradictions that would confuse viewers

# APPROVAL RULES:
# - APPROVE if: Both CRITICAL criteria (VO Alignment, References Found) are either PASS or WARNING, and at least 3/4 total criteria are PASS or WARNING
# - REJECT if: Any CRITICAL criterion is FAIL, OR fewer than 3/4 criteria are PASS or WARNING

# For each criterion, evaluate as PASS, WARNING, or FAIL.

# OUTPUT FORMAT:
# <verdict>APPROVED</verdict>  OR  <verdict>REJECTED</verdict>

# <vo_alignment>PASS or WARNING or FAIL</vo_alignment>
# <references_found>PASS or WARNING or FAIL</references_found>
# <references_relevant>PASS or WARNING or FAIL</references_relevant>
# <no_contradictions>PASS or WARNING or FAIL</no_contradictions>

# <feedback>
# [If REJECTED: Provide specific, actionable feedback on what needs improvement.
# If APPROVED: Brief confirmation that critical criteria met and overall quality acceptable.]
# </feedback>"""

#     chain.add_message(role="user", content=prompt)
#     response = chain.run()

#     # Parse response (Chain handles tag extraction)
#     if isinstance(response, dict):
#         parsed = response
#     else:
#         parsed = chain.extract_text_in_tags(str(response))

#     verdict = parsed.get("verdict", "REJECTED").strip().upper()
#     feedback = parsed.get("feedback", "")

#     # Extract individual criteria
#     criteria = {
#         "vo_alignment": parsed.get("vo_alignment", "FAIL").strip().upper(),
#         "references_found": parsed.get("references_found", "FAIL").strip().upper(),
#         "references_relevant": parsed.get("references_relevant", "FAIL").strip().upper(),
#         "no_contradictions": parsed.get("no_contradictions", "PASS").strip().upper(),
#     }
    
#     # Normalize WARNING to PASS for counting (WARNING is acceptable)
#     criteria_normalized = {k: "PASS" if v in ["PASS", "WARNING"] else "FAIL" for k, v in criteria.items()}
    
#     # Critical criteria (must be PASS or WARNING, not FAIL)
#     critical_criteria = ["vo_alignment", "references_found"]
#     critical_pass = all(criteria_normalized.get(c, "FAIL") == "PASS" for c in critical_criteria)
    
#     # Count total PASS (including WARNING as PASS)
#     total_pass = sum(1 for v in criteria_normalized.values() if v == "PASS")
    
#     # Determine final verdict: Critical must pass AND at least 3/4 total must pass
#     meets_threshold = critical_pass and total_pass >= 3
#     final_verdict = "approved" if (verdict == "APPROVED" and meets_threshold) else "rejected"

#     # Format result message
#     criteria_summary = "\n".join([f"  {k}: {v}" for k, v in criteria.items()])

#     result_message = f"""{verdict}

# Criteria Results:
# {criteria_summary}

# Feedback:
# {feedback}
# """
    
#     verdict_icon = "✅" if final_verdict == "approved" else "❌"
#     print(f"\n      {'─'*50}")
#     print(f"      🔍 Tool: review_segment_quality ◀ END")
#     print(f"      {'─'*50}")
#     print(f"         {verdict_icon} Verdict: {final_verdict.upper()}")
#     print(f"         📋 Criteria results:")
#     for k, v in criteria.items():
#         if v == "PASS":
#             crit_icon = "✅"
#         elif v == "WARNING":
#             crit_icon = "⚠️"
#         else:
#             crit_icon = "❌"
#         print(f"            {crit_icon} {k}: {v}")
#     print(f"         📝 Feedback: {feedback}")
#     print(f"      {'─'*50}\n")

#     # Check for cross-segment issues
#     cross_segment_issue = None
#     if criteria["no_contradictions"] == "FAIL":
#         cross_segment_issue = f"Contradiction detected: {feedback}"

#     return Command(
#         update={
#             "review_verdict": final_verdict,
#             "review_feedback": feedback,
#             "cross_segment_issue": cross_segment_issue,
#             "messages": [
#                 ToolMessage(
#                     result_message,
#                     tool_call_id=tool_call_id
#                 )
#             ]
#         }
#     )


@tool
def refine_graphics_with_images(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[Dict, InjectedState]
) -> Command:
    """
    Refines the graphics definition by selecting the best images from search results
    and creating a simple, readable final graphics definition with image links and sequence.

    This tool:
    - Takes all images found from search
    - Selects the best ones considering VO narration duration
    - Generates a final graphics definition with image references
    - Specifies the order/sequence images should appear

    Returns:
        Final graphics definition with selected images
    """
    vo_text = state.get("vo_text", "")
    slide_chunk = state.get("slide_chunk", "")
    search_results = state.get("search_results", [])
    graphics_definition = state.get("graphics_definition", "")
    course_context = state.get("course_context", {})
    references = state.get("references", [])
    review_verdict = state.get("review_verdict", "pending")
    
    print(f"\n      {'─'*50}")
    print(f"      🎨 Tool: refine_graphics_with_images ▶ START")
    print(f"      {'─'*50}")
    print(f"         📝 VO text: {len(vo_text)} chars")
    print(f"         🔍 Available images: {len(search_results)}")
    print(f"         📋 Current definition: {len(graphics_definition)} chars")
    
    # Check if already completed (has refined definition with selected images)
    already_refined = (
        review_verdict == "approved" and
        references and
        ("Selected Images" in graphics_definition or "Sequence:" in graphics_definition)
    )
    
    if already_refined:
        print(f"         ⚠️  Already refined - returning existing result")
        print(f"         ✅ Selected images: {len(references)}")
        print(f"      {'─'*50}\n")
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        f"Graphics definition already refined with {len(references)} selected image(s). "
                        "No need to refine again. Proceed to finalize_segment.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )

    if not vo_text:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        "Error: No VO text available",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )

    if not search_results:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        "Error: No search results available. Please run search_segment_references first.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )

    # Extract course context
    course_name = course_context.get("course_name", "") if course_context else ""
    topic_name = course_context.get("topic", "") if course_context else ""
    subtopic_name = course_context.get("subtopic", "") if course_context else ""
    
    # Get drive instance for downloading images
    drive = state.get("drive")
    
    # Build multimodal content with images
    content_parts = []
    
    # Add the text prompt first
    prompt_text = f"""You are an expert educational graphics designer in the HVAC industry. Your task is to select the most appropriate images from the provided set to visually support a single voiceover sentence, and to produce a simple, clear graphics definition that specifies which images to show and in what order. Your decisions should prioritize visual clarity and instructional usefulness for the voiceover sentence. 

<course_information>
Course name: {course_name}
Topic name: {topic_name}
Subtopic name: {subtopic_name}
</course_information>

<voiceover_sentence>
{vo_text}
</voiceover_sentence>

<whole_slide_context>
{slide_chunk}
</whole_slide_context>

Instructions:

1. Image Selection Rules:

- Review all provided images before making any selection. Note: Image titles are for reference only and may be inaccurate - always analyze the actual visual content of each image.
- Base image selection strictly on the meaning of the voiceover sentence, not on general topic relevance.
- First, mentally break the voiceover sentence into the distinct visual elements that must be shown for the sentence to be clearly understood.
- Select images only if they are necessary to represent one of those visual elements. An image is necessary only if removing it would make the sentence harder to understand visually.
- The number of images to select must be determined solely by how many distinct visual elements are required to represent the voiceover sentence clearly so that the sentence can be easily understood.
- Do not select images that are:
  - Generic or loosely related
  - Redundant with already selected images
  - Informational but not visually required for this specific sentence

2. Clarity & Simplicity Requirements

After selecting the required images for the voiceover sentence, your next task is to generate a graphics definition that describes how the selected images should appear on screen.

- The graphics definition must be simple, clear, and easy to read.
- Describe only what is directly visible in the selected images.
- Clearly indicate the order in which the selected images should appear.
- Do not introduce new visual elements that are not present in the selected images.
- Do not describe motion, camera movements, transitions, or animations.

3. Sequence Logic

- Treat the voiceover as a single sentence that needs to be visualized from start to finish.
- Order the selected images based on how the meaning of the sentence is most clearly understood.
- When multiple images are selected, arrange them so each image introduces a new visual element in a logical progression.
- Use a sequential order by default.
- Only imply showing multiple images at the same time if the voiceover sentence clearly requires comparing or viewing two elements together.
- Keep the sequence short and focused, including only what is necessary to support the sentence.

The following are the complete set of available images.
AVAILABLE IMAGES ({len(search_results)} images in total):
"""

    content_parts.append({
        "type": "text",
        "text": prompt_text
    })
    
    # Add each image with its metadata and URL, then the actual image
    for i, img_ref in enumerate(search_results, 1):
        img_url = img_ref.get('url', '')
        img_title = img_ref.get('title', 'Untitled')
        img_id = img_ref.get('reference_id', 'N/A')
        
        # Add label text with URL
        label_text = f"\n--- Image {i} of {len(search_results)} ---\n"
        label_text += f"Title: {img_title}\n"
        label_text += f"Reference ID: {img_id}\n"
        label_text += f"URL: {img_url}\n"
        
        # Debug: Print title being sent in metadata
        print(f"         📋 Image {i} metadata - Title: '{img_title}' | URL: {img_url}...")
        
        content_parts.append({
            "type": "text",
            "text": label_text
        })
        
        # Load image - handle web images vs Drive images differently
        pil_image = None
        img_source = img_ref.get('source', '')
        
        # For web search images, use PIL image from metadata (already downloaded)
        if img_source == "Web Search":
            metadata = img_ref.get('metadata', {})
            pil_image = metadata.get('pil_image')
            if pil_image:
                print(f"         ✅ Using web image {i} from metadata: {img_title}")
            else:
                print(f"         ⚠️  Web image {i} PIL not found in metadata: {img_title}")
        
        # For Drive images, download from Drive URL
        elif drive and img_url:
            pil_image = load_image_from_drive_url(img_url, drive, img_id)
            if pil_image:
                print(f"         ✅ Loaded Drive image {i}: {img_title}")
        
        # Add image to content if available
        if pil_image:
            content_parts.append({
                "type": "image_url",
                "image_url": pil_to_base64_data_uri(pil_image)
            })
            print(f"         ✅ Added image {i} to vision model: {img_title}")
        else:
            content_parts.append({
                "type": "text",
                "text": f"[Image {i} could not be loaded; evaluate based on metadata only]\n"
            })
            print(f"         ⚠️  Could not load image {i}: {img_title}")
    
    # Add output format instructions
    output_format_text = """

OUTPUT FORMAT:

Provide your output strictly adhering to this exact XML format:

<evaluation_breakdown>

This section is for the documentation of your internal reasoning and analysis. Follow the structured steps below to ground your decisions before producing the final output.

1. Voiceover Meaning:
- Explain, in your own words, what the voiceover sentence is communicating.

2. Visual Requirements for This Sentence:
- Describe what needs to be visually shown on screen for this sentence to be clearly understood.
- Think in terms of visible objects, components, diagrams, etc.

3. Image-by-Image Visual Scan:
- Review every provided image in the order given.
- For each image, carefully observe the actual visual content of the image.
- Write a short description of what you saw in each of the image.
- Do not infer or assume content based only on the image title or any other metadata, but rather based on the actual visual content of the image.

Use the following format:
   - [Image 1 Title]: Briefly describe what you visually saw in the image.
   - [Image 2 Title]: Briefly describe what you visually saw in the image.
     ...
   Continue for all the available images in the set.

4. Candidate Evaluation and Elimination:
- Compare each image against the visual requirements identified above.
- Identify and explain which of the available images clearly support the required visuals.

5. Final Image Selection Rationale:
- Based on the comparison above, identify all the images that you plan to select for this sentence.
- Explain in detail why these images best represent the required visuals.

6. Sequence Planning:
- Explain the order in which the selected images should appear to best support the voiceover sentence.
- Base this order on clarity and how the sentence is most naturally understood.

</evaluation_breakdown>

(Based on your above evaluation, provide the final graphics definition in the following format)

<graphics_definition>

<description>
Give a simple, clear description of what visuals should appear on screen for this voiceover sentence, written using only the selected images and their order of appearance.
</description>

<selection_justification>
Explain why the selected images, taken together, fully and directly support the voiceover sentence. Describe how the selected images cover all the key ideas or visual requirements expressed in the voiceover sentence,
</selection_justification>

<selected_images>
(List of all selected images in this exact format)
1. [Image Title] | [URL]
2. [Image Title] | [URL]
...
</selected_images>

</graphics_definition>
"""

    content_parts.append({
        "type": "text",
        "text": output_format_text
    })
    
    print(f"         🤖 Using Gemini vision model to analyze {len(search_results)} images...")
    messages = [("user", content_parts)]
    raw_response = llm_with_retry(messages, llm_name="gemini_2_5_pro")
    
    # Extract response
    if hasattr(raw_response, "content"):
        raw_text = raw_response.content
    elif isinstance(raw_response, dict):
        raw_text = raw_response.get("content") or raw_response.get("text", "")
    else:
        raw_text = str(raw_response)
    
    # Debug: Print full raw response
    print(f"\n         📄 Full raw response from model:")
    print(f"         {'─'*50}")
    print(raw_text)
    print(f"         {'─'*50}\n")

    parser_chain = Chain(llm="gemini_2_5_pro", tags=["evaluation_breakdown", "graphics_definition"])
    parsed = parser_chain.extract_text_in_tags(raw_text)
    refined_definition = parsed.get("graphics_definition", "")
    
    if not refined_definition:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        "Error: Failed to generate refined graphics definition",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    # Extract selected image references from the <selected_images> section
    selected_references = []
    selected_images_pattern = r'<selected_images>(.*?)</selected_images>'
    selected_images_match = re.search(selected_images_pattern, refined_definition, re.DOTALL)
    
    if selected_images_match:
        # Extract images from within <selected_images> tag
        selected_images_content = selected_images_match.group(1)
        image_pattern = r'(\d+)\.\s+([^|]+)\s+\|\s+(https?://[^\s]+)'
        matches = re.findall(image_pattern, selected_images_content)
    else:
        matches = []
    
    # Create a mapping of URL to reference data
    url_to_ref = {ref.get('url', ''): ref for ref in search_results if ref.get('url')}
    
    for order, title, url in matches:
        # Find matching reference by URL
        ref = url_to_ref.get(url.strip())
        if ref:
            selected_references.append(ref)
    
    # If no matches found by URL, try to match by title
    if not selected_references:
        title_to_ref = {ref.get('title', '').lower(): ref for ref in search_results}
        for order, title, url in matches:
            ref = title_to_ref.get(title.strip().lower())
            if ref:
                selected_references.append(ref)
    
    # Update state
    result_message = f"""Graphics definition refined with selected images:

{refined_definition[:500]}{'...' if len(refined_definition) > 500 else ''}

Selected {len(selected_references)} image(s).
"""
    
    print(f"\n      {'─'*50}")
    print(f"      🎨 Tool: refine_graphics_with_images ◀ END")
    print(f"      {'─'*50}")
    print(f"         ✅ Refined definition: {len(refined_definition)} chars")
    print(f"         🖼️  Selected images: {len(selected_references)}")
    for i, ref in enumerate(selected_references, 1):
        print(f"            {i}. {ref.get('title', 'Untitled')} | {ref.get('url', 'N/A')[:50]}...")
    print(f"      {'─'*50}\n")

    return Command(
        update={
            "graphics_definition": refined_definition,
            "references": selected_references,
            "review_verdict": "approved",  # Auto-approve since we're selecting best images
            "messages": [
                ToolMessage(
                    result_message,
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool
def finalize_segment(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[Dict, InjectedState]
) -> Command:
    """
    Marks this segment as completed and ready to return to Slide Supervisor.

    Prerequisites:
    - graphics_definition must exist
    - references list must not be empty
    - (review_verdict should be "approved" after refine_graphics_with_images, but we allow finalization if definition and references exist)

    Returns:
        Success confirmation
    """
    print(f"\n      {'─'*50}")
    print(f"      ✅ Tool: finalize_segment ▶ START")
    print(f"      {'─'*50}")
    
    review_verdict = state.get("review_verdict", "pending")
    graphics_definition = state.get("graphics_definition", "")
    references = state.get("references", [])
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 1)
    
    print(f"         📝 Graphics definition: {len(graphics_definition)} chars")
    print(f"         🖼️  References: {len(references)}")
    print(f"         📊 Iterations: {iteration_count}/{max_iterations}")
    print(f"         📋 Review verdict: {review_verdict}")
    
    if not graphics_definition and not references:
        print(f"         ❌ Error: No graphics definition or references")
        print(f"      {'─'*50}\n")
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        "Error: Cannot finalize - no graphics definition or references exist. "
                        "Make sure refine_graphics_with_images has been called first.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )

    if not references:
        print(f"         ❌ Error: No references found")
        print(f"      {'─'*50}\n")
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        "Error: Cannot finalize - no references found",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )

    # Add timestamp
    timestamp = time.time()

    # Determine status based on review verdict
    if review_verdict == "approved":
        status = "COMPLETED"
        status_note = ""
    else:
        status = "FLAGGED (max iterations reached)"
        status_note = "\n⚠️ Note: Finalized despite rejection - will be flagged for human review."

    result_message = f"""✓ Segment finalized!

Summary:
- Graphics definition: {len(graphics_definition)} characters
- References: {len(references)}
- Status: {status}
- Iterations: {iteration_count}/{max_iterations}


✅ SEGMENT PROCESSING COMPLETE - YOUR WORK IS DONE

CRITICAL: Do NOT call any more tools. Your work for this segment is finished.
Return a final answer stating the segment is complete and ready for assembly.
The Slide Supervisor will handle the rest.
"""

    print(f"\n      {'─'*50}")
    print(f"      ✅ Tool: finalize_segment ◀ END")
    print(f"      {'─'*50}")
    print(f"         ✅ Status: {status}")
    print(f"         📝 Graphics definition: {len(graphics_definition)} chars")
    print(f"         🖼️  References: {len(references)}")
    print(f"         📊 Iterations: {iteration_count}/{max_iterations}")
    if status_note:
        print(f"         ⚠️  {status_note.strip()}")
    print(f"      {'─'*50}\n")

    # No state update needed - supervisor will handle it
    return Command(
        update={
            "messages": [
                ToolMessage(
                    result_message,
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


# ============================================================================
# Helper Functions
# ============================================================================

# def extract_visual_elements(graphics_definition: str) -> List[str]:
#     """
#     Extract visual elements from a graphics definition.

#     Extracts full element descriptions from the VISUAL ELEMENTS section.
#     """
#     elements = []

#     # Find VISUAL ELEMENTS section
#     if "VISUAL ELEMENTS:" in graphics_definition:
#         section = graphics_definition.split("VISUAL ELEMENTS:")[1]
#         if "PRESENTATION:" in section:
#             section = section.split("PRESENTATION:")[0]

#         # Extract bullet points - get the FULL element description, not just the label
#         lines = section.split("\n")
#         for line in lines:
#             line = line.strip()
#             if line.startswith("-"):
#                 # Remove the leading "- " and keep the FULL description
#                 element = line.lstrip("- ").strip()
#                 # If it starts with "Element N:", extract the description after the colon
#                 if element.lower().startswith("element") and ":" in element:
#                     # Split only on the FIRST colon to get the full description
#                     parts = element.split(":", 1)
#                     if len(parts) > 1:
#                         element = parts[1].strip()
#                 if element:
#                     elements.append(element)

#     # Fallback: if no elements found, return a generic one
#     if not elements:
#         elements = ["Graphics definition content"]

#     return elements
