"""
Segment Processor Agent (Level 2)

ReAct agent that processes a single VO segment through search→refine workflow.
"""

from typing import Dict, Any, List
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

from agents.graphics_workflow_v2.state.schemas import SegmentProcessorState, SegmentData
from agents.graphics_workflow_v2.config.settings import (
    SEGMENT_PROCESSOR_MODEL,
    SEGMENT_PROCESSOR_RECURSION_LIMIT,
    MAX_ITERATIONS_PER_SEGMENT,
    SEARCH_K,
)
from agents.graphics_workflow_v2.tools.segment_processor.segment_tools import (
    search_segment_references,
    # review_segment_quality,  
    refine_graphics_with_images,
    finalize_segment,
)
from agents.graphics_workflow_v2.prompts.segment_processor_prompts import (
    SEGMENT_PROCESSOR_SYSTEM_PROMPT,
    get_segment_processor_prompt,
)


def create_segment_processor_agent():
    """
    Creates the Segment Processor ReAct agent.

    Returns:
        Compiled LangGraph agent
    """
    # Initialize LLM
    # Note: GPT-5 models only support temperature=1
    llm = init_chat_model(SEGMENT_PROCESSOR_MODEL, temperature=1)

    # System prompt will be customized per invocation
    # We'll use a base prompt here
    base_system_prompt = SEGMENT_PROCESSOR_SYSTEM_PROMPT

    # Create agent with tools
    agent = create_react_agent(
        model=llm,
        tools=[
            search_segment_references,
            # review_segment_quality,  
            refine_graphics_with_images,
            finalize_segment,
        ],
        state_schema=SegmentProcessorState,
        prompt=base_system_prompt,
    )

    return agent


def run_segment_processor(
    segment_index: int,
    vo_text: str,
    slide_chunk: str,
    previous_segments: List[SegmentData] = None,
    available_references: Dict[str, Any] = None,
    drive: Any = None,
    course_context: Dict[str, Any] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run the Segment Processor to create graphics definition for one segment.

    Args:
        segment_index: Which segment this is (0-indexed)
        vo_text: VO text for this segment
        slide_chunk: Full slide content for context
        previous_segments: Completed segments before this one
        available_references: References that could be reused
        drive: Google Drive instance
        course_context: Optional course context (course name, topic, etc.)
        **kwargs: Additional configuration

    Returns:
        Final state with graphics_definition, references, review_verdict
    """
    print(f"\n{'🔶'*40}")
    print(f"🔧 SEGMENT PROCESSOR - Starting Segment {segment_index}")
    print(f"{'🔶'*40}")
    print(f"🎙️  VO Text: \"{vo_text}\"")
    print(f"📄 Full slide context: {len(slide_chunk)} chars")
    print(f"📚 Previous segments for context: {len(previous_segments or [])}")
    if previous_segments:
        for ps in (previous_segments or [])[-2:]:
            print(f"      └─ Segment {ps.get('segment_index')}: {ps.get('vo_text', '')}")
    print(f"♻️  Available references for reuse: {len(available_references or {})}")
    if course_context:
        print(f"📚 Course context: {course_context}")
    print(f"{'🔶'*40}\n")
    
    # Create agent
    agent = create_segment_processor_agent()
    print(f"   ✅ Segment Processor agent created for segment {segment_index}")

    # Prepare initial state
    initial_state = {
        "messages": [
            {
                "role": "user",
                "content": get_segment_processor_prompt(
                    segment_index=segment_index,
                    vo_text=vo_text,
                    slide_chunk=slide_chunk,
                    previous_segments=previous_segments or [],
                    max_iterations=MAX_ITERATIONS_PER_SEGMENT,
                    course_context=course_context,
                ),
            }
        ],
        "segment_index": segment_index,
        "vo_text": vo_text,
        "slide_chunk": slide_chunk,
        "previous_segments": previous_segments or [],
        "available_references": available_references or {},
        "graphics_definition": "",
        "references": [],
        "search_results": [],  # Store all search results for refine_graphics_with_images
        "review_verdict": "pending",
        "review_feedback": "",
        "iteration_count": 0,
        "max_iterations": MAX_ITERATIONS_PER_SEGMENT,
        "cross_segment_issue": None,
        "course_context": course_context,
        # Pass through search parameters
        "drive": drive,
        "filters": kwargs.get("filters"),
        "root_folder_id": kwargs.get("root_folder_id"),
        "search_k": kwargs.get("search_k", SEARCH_K),
    }

    # Run agent
    recursion_limit = kwargs.get("recursion_limit", SEGMENT_PROCESSOR_RECURSION_LIMIT)
    print(f"   🚀 Running Segment Processor agent...")
    print(f"      ⚙️  Recursion limit: {recursion_limit}")
    print(f"      ⚙️  Max iterations: {MAX_ITERATIONS_PER_SEGMENT}\n")
    
    final_state = agent.invoke(
        initial_state,
        config={"recursion_limit": recursion_limit}
    )
    
    # Determine result icon
    status = final_state.get('status', 'unknown')
    verdict = final_state.get('review_verdict', 'pending')
    status_icon = "✅" if status == 'completed' or verdict == 'approved' else "⚠️"
    
    print(f"\n{'🟧'*40}")
    print(f"{status_icon} SEGMENT {segment_index} PROCESSING COMPLETE")
    print(f"{'🟧'*40}")
    print(f"📊 Status: {status}")
    print(f"🔍 Review verdict: {verdict}")
    print(f"🔄 Iterations used: {final_state.get('iteration_count', 0)}")
    print(f"🖼️  References found: {len(final_state.get('references', []))}")
    
    # Show references found
    refs = final_state.get('references', [])
    if refs:
        print(f"\n   📎 References:")
        for i, ref in enumerate(refs[:5], 1):
            print(f"      {i}. {ref.get('title', 'Untitled')} (score: {ref.get('relevance_score', 0):.2f})")
        if len(refs) > 5:
            print(f"      ... and {len(refs) - 5} more")
    
    # Show graphics definition
    gd = final_state.get('graphics_definition', '')
    if gd:
        print(f"\n   📝 Graphics Definition ({len(gd)} chars):")
        print(f"      {gd}")
    
    print(f"{'🟧'*40}\n")

    return final_state


def extract_segment_data_from_state(state: Dict[str, Any]) -> SegmentData:
    """
    Extract SegmentData from the final processor state.

    Args:
        state: Final state from segment processor

    Returns:
        SegmentData dictionary
    """
    import time

    segment_data: SegmentData = {
        "segment_index": state.get("segment_index", 0),
        "vo_text": state.get("vo_text", ""),
        "graphics_definition": state.get("graphics_definition", ""),
        "references": state.get("references", []),
        "status": "completed" if state.get("review_verdict") == "approved" else "flagged",
        "review_verdict": state.get("review_verdict", "pending"),
        "review_feedback": state.get("review_feedback", ""),
        "iteration_count": state.get("iteration_count", 0),
        "timestamp": time.time(),
    }

    return segment_data
