"""
Slide Supervisor Agent (Level 1)

ReAct agent that orchestrates the entire graphics definition workflow for a slide.
"""

from typing import Dict, Any, Optional
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

from agents.graphics_workflow_v2.state.schemas import GraphicsSlideState
from agents.graphics_workflow_v2.config.settings import (
    SLIDE_SUPERVISOR_MODEL,
    SLIDE_SUPERVISOR_RECURSION_LIMIT,
    MAX_ITERATIONS_PER_SEGMENT,
    ENABLE_LANGSMITH_TRACING,
)
from agents.graphics_workflow_v2.tools.slide_supervisor.supervisor_tools import (
    segment_slide,
    process_segment,
    flag_for_human_review,
    finalize_slide_graphics,
)
from agents.graphics_workflow_v2.prompts.slide_supervisor_prompts import (
    SLIDE_SUPERVISOR_SYSTEM_PROMPT,
    get_slide_supervisor_prompt,
)


def create_slide_supervisor_agent():
    """
    Creates the Slide Supervisor ReAct agent.

    Returns:
        Compiled LangGraph agent
    """
    # Initialize LLM
    llm = init_chat_model(SLIDE_SUPERVISOR_MODEL)

    # Create agent with tools
    agent = create_react_agent(
        model=llm,
        tools=[
            segment_slide,
            process_segment,
            flag_for_human_review,
            finalize_slide_graphics,
        ],
        state_schema=GraphicsSlideState,
        prompt=SLIDE_SUPERVISOR_SYSTEM_PROMPT,
    )

    return agent


def run_graphics_workflow(
    slide_chunk: str,
    drive: Any,
    course_context: Optional[Dict[str, Any]] = None,
    filters: Optional[Dict[str, Any]] = None,
    root_folder_id: str = '1QS6PmCESfgFWNNEpRJDUB0E-t8iMAatH',
    **kwargs
) -> Dict[str, Any]:
    """
    Run the complete graphics definition workflow for a slide.

    This is the main entry point for the Graphics Workflow V2 system.

    Args:
        slide_chunk: The slide content to process
        drive: Google Drive instance for image search
        course_context: Optional course context (course name, module, topic, etc.)
        filters: Optional filters for image search
        root_folder_id: Root folder ID for vector store
        **kwargs: Additional configuration overrides

    Returns:
        Final state with complete graphics definition

    Example:
        >>> from pydrive.auth import GoogleAuth
        >>> from pydrive.drive import GoogleDrive
        >>>
        >>> gauth = GoogleAuth()
        >>> gauth.LocalWebserverAuth()
        >>> drive = GoogleDrive(gauth)
        >>>
        >>> slide = "Superheat happens after the refrigerant..."
        >>>
        >>> result = run_graphics_workflow(
        ...     slide_chunk=slide,
        ...     drive=drive,
        ...     course_context={"course_name": "HVAC Fundamentals"}
        ... )
        >>>
        >>> print(result["final_definition"])
    """
    # Create agent
    agent = create_slide_supervisor_agent()

    # Prepare initial state
    initial_state = {
        "messages": [
            {
                "role": "user",
                "content": get_slide_supervisor_prompt(
                    slide_chunk=slide_chunk,
                    course_context=course_context
                ),
            }
        ],
        "slide_chunk": slide_chunk,
        "course_context": course_context,
        "vo_segments": [],
        "segmentation_method": "auto",
        "current_segment_index": 0,
        "segments": [],
        "all_references": {},
        "reference_reuse_map": {},
        "max_iterations_per_segment": kwargs.get("max_iterations_per_segment", MAX_ITERATIONS_PER_SEGMENT),
        "total_tool_calls": 0,
        "flags": [],
        "final_definition": None,
        "status": "initialized",
        # Pass through search parameters
        "drive": drive,
        "filters": filters,
        "root_folder_id": root_folder_id,
    }

    # Configure run
    config = {
        "recursion_limit": kwargs.get("recursion_limit", SLIDE_SUPERVISOR_RECURSION_LIMIT)
    }

    # Add LangSmith tracing if enabled
    if ENABLE_LANGSMITH_TRACING:
        config["tags"] = kwargs.get("tags", ["graphics_workflow_v2"])
        config["run_name"] = kwargs.get("run_name", "Graphics Workflow V2")

    # Run agent
    final_state = agent.invoke(initial_state, config=config)

    return final_state


def get_final_definition(workflow_result: Dict[str, Any]) -> Optional[str]:
    """
    Extract the final graphics definition from workflow result.

    Args:
        workflow_result: Result from run_graphics_workflow

    Returns:
        Final definition string or None if not completed
    """
    return workflow_result.get("final_definition")


def get_workflow_summary(workflow_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a summary of the workflow execution.

    Args:
        workflow_result: Result from run_graphics_workflow

    Returns:
        Summary dictionary
    """
    segments = workflow_result.get("segments", [])
    flags = workflow_result.get("flags", [])

    completed_segments = [seg for seg in segments if seg["status"] == "completed"]
    flagged_segments = [seg for seg in segments if seg["status"] == "flagged"]

    total_references = sum(len(seg["references"]) for seg in segments)
    avg_iterations = sum(seg["iteration_count"] for seg in segments) / len(segments) if segments else 0

    return {
        "status": workflow_result.get("status"),
        "total_segments": len(segments),
        "completed_segments": len(completed_segments),
        "flagged_segments": len(flagged_segments),
        "total_flags": len(flags),
        "total_references": total_references,
        "unique_references": len(workflow_result.get("all_references", {})),
        "avg_iterations_per_segment": round(avg_iterations, 2),
        "flags": flags,
    }
