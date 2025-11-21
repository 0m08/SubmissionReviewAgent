"""
Paraphraser Orchestrator

Coordinates the multi-agent paraphrasing workflow using LangGraph StateGraph.
"""

from langgraph.graph import StateGraph, END
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Dict, Optional
from pydantic import BaseModel, Field

from .state import ParaphraserState
from .prompts import (
    PARAPHRASER_PROMPT_TEMPLATE,
    REVIEWER_PROMPT_TEMPLATE,
    REFINER_PROMPT_TEMPLATE
)
from .utils import extract_from_xml_tags


class QualityReview(BaseModel):
    """Pydantic model for quality review structured output."""

    passed: bool = Field(..., description="Whether all quality criteria passed")
    issues: list[str] = Field(
        default_factory=list,
        description="List of specific issues found"
    )
    score: int = Field(..., ge=0, le=100, description="Quality score from 0-100")
    feedback: str = Field(..., description="Detailed explanation")


def paraphraser_node(state: ParaphraserState) -> ParaphraserState:
    """
    Paraphraser node - transforms text with trade-specific voice.
    """
    # Extract parameters from state
    text = state.get("original_text", "")
    trade = state.get("trade", "HVAC")
    specialization = state.get("specialization")
    preserve_formatting = state.get("preserve_formatting", True)
    target_length = state.get("target_length", "similar")
    llm_model = state.get("llm_model", "google_genai:gemini-2.5-flash")
    custom_examples = state.get("custom_examples")
    custom_tone_instructions = state.get("custom_tone_instructions")

    # Build prompt
    prompt = PARAPHRASER_PROMPT_TEMPLATE.format(
        trade=trade,
        specialization=specialization or "General",
        input_text=text,
        preserve_formatting="yes" if preserve_formatting else "no",
        target_length=target_length or "similar"
    )

    # Append custom examples if provided
    if custom_examples:
        prompt = prompt.replace(
            "</examples>",
            f"\n{custom_examples}\n</examples>"
        )

    # Append custom tone instructions if provided
    if custom_tone_instructions:
        prompt = prompt.replace(
            "</paraphrasing_rules>",
            f"10. CUSTOM INSTRUCTIONS: {custom_tone_instructions}\n</paraphrasing_rules>"
        )

    # Call LLM directly
    llm = init_chat_model(
        llm_model,
        max_retries=3,
    )
    response = llm.invoke([HumanMessage(content=prompt)])

    # Extract paraphrased text from XML tags
    paraphrased_text = extract_from_xml_tags(response.content, "paraphrased_text")

    # Update state
    return {
        **state,
        "paraphrased_text": paraphrased_text,
        "final_text": paraphrased_text,
    }


def reviewer_node(state: ParaphraserState) -> ParaphraserState:
    """
    Reviewer node - validates quality against criteria.
    """
    # Extract from state
    original_text = state.get("original_text", "")
    paraphrased_text = state.get("paraphrased_text", "")
    trade = state.get("trade", "HVAC")
    llm_model = state.get("llm_model", "google_genai:gemini-2.5-flash")
    custom_quality_criteria = state.get("custom_quality_criteria")

    # Build prompt
    prompt = REVIEWER_PROMPT_TEMPLATE.format(
        original_text=original_text,
        paraphrased_text=paraphrased_text,
        trade=trade
    )

    # Replace quality criteria if custom ones provided
    if custom_quality_criteria:
        prompt = prompt.replace(
            "<quality_criteria>",
            f"<quality_criteria>\n{custom_quality_criteria}"
        )

    # Call LLM with structured output
    llm = init_chat_model(
        llm_model,
        max_retries=3,
    )
    structured_llm = llm.with_structured_output(QualityReview)

    try:
        result = structured_llm.invoke([HumanMessage(content=prompt)])

        quality_report = {
            "passed": result.passed,
            "issues": result.issues,
            "score": result.score,
            "feedback": result.feedback
        }
    except Exception as e:
        # Fallback if structured output fails
        quality_report = {
            "passed": False,
            "issues": [f"Error parsing review: {str(e)}"],
            "score": 0,
            "feedback": "Review parsing failed"
        }

    # Update state
    return {
        **state,
        "quality_report": quality_report,
    }


def refiner_node(state: ParaphraserState) -> ParaphraserState:
    """
    Refiner node - fixes specific quality issues.
    """
    # Extract from state
    paraphrased_text = state.get("paraphrased_text", "")
    original_text = state.get("original_text", "")
    trade = state.get("trade", "HVAC")
    llm_model = state.get("llm_model", "google_genai:gemini-2.5-flash")
    quality_report = state.get("quality_report", {})

    # Get issues from quality report
    issues_list = quality_report.get("issues", [])
    issues_str = "\n".join(f"- {issue}" for issue in issues_list)

    # Build prompt
    prompt = REFINER_PROMPT_TEMPLATE.format(
        paraphrased_text=paraphrased_text,
        issues=issues_str,
        original_text=original_text,
        trade=trade
    )

    # Call LLM directly
    llm = init_chat_model(
        llm_model,
        max_retries=3,
    )
    response = llm.invoke([HumanMessage(content=prompt)])

    # Extract refined text from XML tags
    refined_text = extract_from_xml_tags(response.content, "refined_text")

    # Update state
    refinement_count = state.get("refinement_count", 0) + 1

    return {
        **state,
        "paraphrased_text": refined_text,
        "final_text": refined_text,
        "refinement_count": refinement_count,
    }


def should_continue(state: ParaphraserState) -> str:
    """
    Routing function - decides whether to refine or finish.
    """
    quality_report = state.get("quality_report")
    refinement_count = state.get("refinement_count", 0)
    max_iterations = 3  # Default max iterations

    # No quality report yet - shouldn't happen
    if not quality_report:
        return "end"

    # Quality passed - we're done
    if quality_report.get("passed"):
        return "end"

    # Max iterations reached - finish with current version
    if refinement_count >= max_iterations:
        return "end"

    # Quality failed and we have iterations left - refine
    return "refine"


def create_paraphraser_graph(max_iterations: int = 3):
    """
    Create the paraphraser StateGraph workflow.

    Args:
        max_iterations: Maximum refinement iterations

    Returns:
        Compiled StateGraph
    """
    # Create graph
    workflow = StateGraph(ParaphraserState)

    # Add nodes
    workflow.add_node("paraphrase", paraphraser_node)
    workflow.add_node("review", reviewer_node)
    workflow.add_node("refine", refiner_node)

    # Set entry point
    workflow.set_entry_point("paraphrase")

    # Add edges
    workflow.add_edge("paraphrase", "review")

    # Conditional routing from review
    workflow.add_conditional_edges(
        "review",
        should_continue,
        {
            "refine": "refine",
            "end": END
        }
    )

    # After refinement, review again
    workflow.add_edge("refine", "review")

    return workflow.compile()


def run_paraphraser(
    text: str,
    trade: str = "HVAC",
    specialization: Optional[str] = None,
    preserve_formatting: bool = True,
    target_length: Optional[str] = "similar",
    max_iterations: int = 3,
    llm_model: str = "openai:gpt-5-mini",
    custom_quality_criteria: Optional[str] = None,
    custom_examples: Optional[str] = None,
    custom_tone_instructions: Optional[str] = None
) -> Dict:
    """
    Main entry point for paraphrasing text.

    Args:
        text: Text to paraphrase
        trade: Trade context
        specialization: Optional sub-specialization
        preserve_formatting: Whether to preserve structure
        target_length: "similar", "concise", or "expanded"
        max_iterations: Max refinement loops
        llm_model: LLM model to use (not used in current implementation)
        custom_quality_criteria: Custom quality checklist
        custom_examples: Custom examples
        custom_tone_instructions: Custom tone instructions

    Returns:
        Dictionary with final_text, status, quality_report, etc.
    """
    # Create graph
    graph = create_paraphraser_graph(max_iterations)

    # Initialize state
    initial_state = {
        "original_text": text,
        "trade": trade,
        "specialization": specialization,
        "preserve_formatting": preserve_formatting,
        "target_length": target_length,
        "llm_model": llm_model,
        "custom_quality_criteria": custom_quality_criteria,
        "custom_examples": custom_examples,
        "custom_tone_instructions": custom_tone_instructions,
        "refinement_count": 0,
        "status": "in_progress",
    }

    # Run graph
    try:
        final_state = graph.invoke(initial_state)

        return {
            "final_text": final_state.get("final_text", text),
            "status": "completed",
            "quality_report": final_state.get("quality_report"),
            "refinement_count": final_state.get("refinement_count", 0),
            "messages": [],  # No messages in this architecture
            "metadata": {
                "trade": trade,
                "specialization": specialization,
                "preserve_formatting": preserve_formatting,
                "target_length": target_length,
            }
        }
    except Exception as e:
        return {
            "final_text": text,
            "status": "failed",
            "error": str(e),
            "quality_report": None,
            "refinement_count": 0,
            "messages": []
        }
