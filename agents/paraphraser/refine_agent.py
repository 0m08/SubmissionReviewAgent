"""
Refiner Agent Tool

Fixes specific quality issues in paraphrased text.
"""

from langchain_core.tools import tool
from typing import List, Annotated
from langgraph.prebuilt import InjectedState
import json
from modules.chain import Chain
from .prompts import REFINER_PROMPT_TEMPLATE


@tool
def refine_text(
    state: Annotated[dict, InjectedState]
) -> str:
    """
    Refine paraphrased text to fix specific quality issues.

    Use this tool when the quality reviewer identifies problems that need
    to be fixed. Makes surgical edits only - does NOT re-paraphrase from scratch.

    The tool reads from state:
    - paraphrased_text: Current paraphrased version with issues
    - quality_report: Contains issues to fix
    - original_text: Original text for reference
    - trade: Trade context

    Returns:
        Refined text with issues addressed
    """
    # Extract from state
    paraphrased_text = state.get("paraphrased_text", "")
    original_text = state.get("original_text", "")
    trade = state.get("trade", "HVAC")
    quality_report = state.get("quality_report", {})

    # Get issues from quality report
    issues_list = quality_report.get("issues", [])

    # Format issues as bullet points
    if isinstance(issues_list, list) and len(issues_list) > 0:
        issues_str = "\n".join(f"- {issue}" for issue in issues_list)
    else:
        issues_str = "No specific issues provided"

    # Build prompt from template
    prompt = REFINER_PROMPT_TEMPLATE.format(
        paraphrased_text=paraphrased_text,
        issues=issues_str,
        original_text=original_text,
        trade=trade
    )

    # Execute with Chain
    chain = Chain(llm='gemini_2_5_flash', tags=["refined_text"])
    chain.add_message(role="user", content=prompt)
    response = chain.run()

    # Extract the refined text
    if isinstance(response, dict) and "refined_text" in response:
        refined_text = response["refined_text"]
    elif isinstance(response, str):
        refined_text = response
    else:
        refined_text = str(response)

    # Update state with refined text
    state["paraphrased_text"] = refined_text  # Update current version
    state["final_text"] = refined_text  # Update final output

    # Increment refinement count
    state["refinement_count"] = state.get("refinement_count", 0) + 1

    return refined_text
