"""
Graphics Workflow V2

A hierarchical multi-agent system for creating graphics definitions for educational video slides.

Architecture:
- Level 1: Slide Supervisor - Orchestrates entire workflow
- Level 2: Segment Processor - Processes individual VO segments
- Level 3: Search Agent - Finds and validates visual references

Usage:
    from agents.graphics_workflow_v2 import run_graphics_workflow

    result = run_graphics_workflow(
        slide_chunk="Your slide content here...",
        drive=drive_instance,
        course_context={"course_name": "Course Name"}
    )

    print(result["final_definition"])
"""

from agents.graphics_workflow_v2.agents.slide_supervisor import (
    run_graphics_workflow,
    get_final_definition,
    get_workflow_summary,
)

from agents.graphics_workflow_v2.config.settings import (
    get_all_config,
    get_model_config,
    get_limit_config,
    get_search_config,
)

__all__ = [
    "run_graphics_workflow",
    "get_final_definition",
    "get_workflow_summary",
    "get_all_config",
    "get_model_config",
    "get_limit_config",
    "get_search_config",
]

__version__ = "2.0.0"
