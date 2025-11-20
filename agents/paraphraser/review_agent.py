"""
Quality Reviewer Agent Tool

Validates paraphrased text against quality criteria.
"""

from langchain_core.tools import tool, InjectedToolCallId
from typing import Dict, Annotated
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_core.messages import ToolMessage
import json
from pydantic import BaseModel, Field
from modules.chain import Chain
from .prompts import REVIEWER_PROMPT_TEMPLATE


class QualityReview(BaseModel):
    """Pydantic model for quality review output."""
    
    passed: bool = Field(..., description="Whether all quality criteria passed")
    issues: list[str] = Field(
        default_factory=list,
        description="List of specific issues found, empty list if none"
    )
    score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Quality score from 0-100"
    )
    feedback: str = Field(
        ...,
        min_length=1,
        description="Detailed explanation of the review"
    )



@tool
def review_quality(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState]
) -> Command:
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
    custom_quality_criteria = state.get("custom_quality_criteria")

    # Build prompt from template
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

    # Execute with Chain using structured output
    chain = Chain(
        llm='gemini_2_5_flash',
        use_output_parser=False  # Don't use tag extraction for structured output
    )
    chain.structured_output = QualityReview
    chain.add_message(role="user", content=prompt)
    response = chain.run()

    # Response should already be a QualityReview instance from structured output
    if isinstance(response, QualityReview):
        result = {
            "passed": response.passed,
            "issues": response.issues,
            "score": response.score,
            "feedback": response.feedback
        }
    elif isinstance(response, dict):
        # Fallback in case response is already a dict
        result = {
            "passed": response.get("passed", False),
            "issues": response.get("issues", []),
            "score": response.get("score", 0),
            "feedback": response.get("feedback", "")
        }
    else:
        # Error handling - unable to parse response
        result = {
            "passed": False,
            "issues": ["Unable to parse review response"],
            "score": 0,
            "feedback": str(response)[:500]
        }

    # Return Command with state updates and tool message
    return Command(
        update={
            "quality_report": result,
            "messages": [
                ToolMessage(
                    json.dumps(result),
                    tool_call_id=tool_call_id
                )
            ]
        }
    )
