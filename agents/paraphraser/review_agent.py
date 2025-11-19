"""
Quality Reviewer Agent Tool

Validates paraphrased text against quality criteria.
"""

from langchain_core.tools import tool
from typing import Dict, Annotated
from langgraph.prebuilt import InjectedState
import json
from modules.chain import Chain
from .prompts import REVIEWER_PROMPT_TEMPLATE


@tool
def review_quality(
    state: Annotated[dict, InjectedState]
) -> str:
    """
    Review paraphrased text for quality against spec criteria.

    Use this tool to validate that paraphrased text meets all quality
    requirements: accuracy, clarity, tone, technical correctness, and safety.

    The tool reads from state:
    - original_text: The original input text (for comparison)
    - paraphrased_text: The paraphrased version to review
    - trade: The trade context for validation

    Returns:
        JSON string with review results containing:
        - passed: boolean indicating if all criteria passed
        - issues: list of specific problems found
        - score: integer 0-100 quality score
        - feedback: detailed explanation
    """
    # Extract from state
    original_text = state.get("original_text", "")
    paraphrased_text = state.get("paraphrased_text", "")
    trade = state.get("trade", "HVAC")

    # Build prompt from template
    prompt = REVIEWER_PROMPT_TEMPLATE.format(
        original_text=original_text,
        paraphrased_text=paraphrased_text,
        trade=trade
    )

    # Execute with Chain using structured output
    chain = Chain(
        llm='gemini_2_5_flash',
        tags=["passed", "issues", "score", "feedback"]
    )
    chain.add_message(role="user", content=prompt)
    response = chain.run()

    # Parse and structure the response
    try:
        if isinstance(response, dict):
            # Already structured from Chain tags
            result = {
                "passed": str(response.get("passed", "false")).lower() == "true",
                "issues": response.get("issues", "").split(",") if isinstance(response.get("issues"), str) else response.get("issues", []),
                "score": int(response.get("score", 0)),
                "feedback": response.get("feedback", "")
            }
        else:
            # Fallback parsing
            result = {
                "passed": False,
                "issues": ["Unable to parse review response"],
                "score": 0,
                "feedback": str(response)[:500]
            }
    except Exception as e:
        # Error during parsing - return safe default
        result = {
            "passed": False,
            "issues": [f"Error parsing review: {str(e)}"],
            "score": 0,
            "feedback": f"Review parsing failed: {str(response)[:200]}"
        }

    # Update state with quality report
    state["quality_report"] = result

    # Return as JSON string for LangGraph tool compatibility
    return json.dumps(result)
