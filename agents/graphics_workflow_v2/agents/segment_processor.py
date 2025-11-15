"""
Segment Processor Agent (Level 2)

ReAct agent that processes a single VO segment through define→search→review loop.
"""

from typing import Dict, Any, List
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

from agents.graphics_workflow_v2.state.schemas import SegmentProcessorState, SegmentData
from agents.graphics_workflow_v2.config.settings import (
    SEGMENT_PROCESSOR_MODEL,
    SEGMENT_PROCESSOR_RECURSION_LIMIT,
    MAX_ITERATIONS_PER_SEGMENT,
)
from agents.graphics_workflow_v2.tools.segment_processor.segment_tools import (
    define_segment_graphics,
    search_segment_references,
    review_segment_quality,
    finalize_segment,
)
from agents.graphics_workflow_v2.prompts.segment_processor_prompts import (
    SEGMENT_PROCESSOR_SYSTEM_PROMPT,
    get_segment_processor_prompt,
    format_context_summary,
)


def create_segment_processor_agent():
    """
    Creates the Segment Processor ReAct agent.

    Returns:
        Compiled LangGraph agent
    """
    # Initialize LLM
    llm = init_chat_model(SEGMENT_PROCESSOR_MODEL)

    # System prompt will be customized per invocation
    # We'll use a base prompt here
    base_system_prompt = SEGMENT_PROCESSOR_SYSTEM_PROMPT.format(
        max_iterations=MAX_ITERATIONS_PER_SEGMENT,
        context_summary="[Will be provided in user prompt]"
    )

    # Create agent with tools
    agent = create_react_agent(
        model=llm,
        tools=[
            define_segment_graphics,
            search_segment_references,
            review_segment_quality,
            finalize_segment,
        ],
        state_schema=SegmentProcessorState,
        state_modifier=base_system_prompt,
    )

    return agent


def run_segment_processor(
    segment_index: int,
    vo_text: str,
    slide_chunk: str,
    previous_segments: List[SegmentData] = None,
    available_references: Dict[str, Any] = None,
    drive: Any = None,
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
        **kwargs: Additional configuration

    Returns:
        Final state with graphics_definition, references, review_verdict
    """
    # Create agent
    agent = create_segment_processor_agent()

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
        "review_verdict": "pending",
        "review_feedback": "",
        "iteration_count": 0,
        "max_iterations": MAX_ITERATIONS_PER_SEGMENT,
        "cross_segment_issue": None,
        # Pass through search parameters
        "drive": drive,
        "filters": kwargs.get("filters"),
        "root_folder_id": kwargs.get("root_folder_id"),
    }

    # Run agent
    final_state = agent.invoke(
        initial_state,
        config={"recursion_limit": SEGMENT_PROCESSOR_RECURSION_LIMIT}
    )

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
