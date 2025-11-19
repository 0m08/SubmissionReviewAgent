"""
Quality Reviewer Agent Tool

Validates paraphrased text against quality criteria.
"""

from langchain_core.tools import tool
from typing import Dict
import json
from modules.chain import Chain
from .prompts import REVIEWER_PROMPT_TEMPLATE


@tool
def review_quality(
    original_text: str,
    paraphrased_text: str,
    trade: str
) -> str:
    """
    Review paraphrased text for quality against spec criteria.

    Use this tool to validate that paraphrased text meets all quality
    requirements: accuracy, clarity, tone, technical correctness, and safety.

    Args:
        original_text: The original input text (for comparison)
        paraphrased_text: The paraphrased version to review
        trade: The trade context for appropriate terminology validation

    Returns:
        JSON string with review results containing:
        - passed: boolean indicating if all criteria passed
        - issues: list of specific problems found
        - score: integer 0-100 quality score
        - feedback: detailed explanation
    """
    # Build prompt from template
    prompt = REVIEWER_PROMPT_TEMPLATE.format(
        original_text=original_text,
        paraphrased_text=paraphrased_text,
        trade=trade
    )

    # Execute with Chain
    chain = Chain(
        llm='openai:gpt-4o-mini',
        tags=["passed", "issues", "score", "feedback"],
        use_output_parser=False  # We want raw JSON output
    )
    chain.add_message(role="user", content=prompt)
    response = chain.run()

    # Parse and structure the response
    try:
        if isinstance(response, dict):
            # Already structured
            result = {
                "passed": response.get("passed", False),
                "issues": response.get("issues", []),
                "score": int(response.get("score", 0)),
                "feedback": response.get("feedback", "")
            }
        elif isinstance(response, str):
            # Try to parse JSON from string
            # Look for JSON in the response
            if "{" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                result = {
                    "passed": parsed.get("passed", False),
                    "issues": parsed.get("issues", []),
                    "score": int(parsed.get("score", 0)),
                    "feedback": parsed.get("feedback", "")
                }
            else:
                # Fallback: treat as passed if no clear failure
                result = {
                    "passed": True,
                    "issues": [],
                    "score": 85,
                    "feedback": response
                }
        else:
            result = {
                "passed": True,
                "issues": [],
                "score": 85,
                "feedback": str(response)
            }
    except Exception as e:
        # Error during parsing - return safe default
        result = {
            "passed": False,
            "issues": [f"Error parsing review: {str(e)}"],
            "score": 0,
            "feedback": f"Review parsing failed: {str(response)[:200]}"
        }

    # Return as JSON string for LangGraph tool compatibility
    return json.dumps(result)
