"""
Refiner Agent Tool

Fixes specific quality issues in paraphrased text.
"""

from langchain_core.tools import tool
from typing import List
from modules.chain import Chain
from .prompts import REFINER_PROMPT_TEMPLATE


@tool
def refine_text(
    paraphrased_text: str,
    issues: str,
    original_text: str,
    trade: str
) -> str:
    """
    Refine paraphrased text to fix specific quality issues.

    Use this tool when the quality reviewer identifies problems that need
    to be fixed. Makes surgical edits only - does NOT re-paraphrase from scratch.

    Args:
        paraphrased_text: Current paraphrased version with issues
        issues: Description of issues to fix (can be comma-separated list or JSON string)
        original_text: Original text for reference
        trade: Trade context

    Returns:
        Refined text with issues addressed
    """
    # Parse issues if it's a JSON string
    import json
    try:
        if isinstance(issues, str) and (issues.startswith('[') or issues.startswith('{')):
            parsed = json.loads(issues)
            if isinstance(parsed, list):
                issues_list = parsed
            elif isinstance(parsed, dict) and 'issues' in parsed:
                issues_list = parsed['issues']
            else:
                issues_list = [str(parsed)]
        else:
            # Treat as plain string
            issues_list = [issues] if isinstance(issues, str) else issues
    except:
        issues_list = [str(issues)]

    # Format issues as bullet points
    if isinstance(issues_list, list) and len(issues_list) > 0:
        issues_str = "\n".join(f"- {issue}" for issue in issues_list)
    else:
        issues_str = str(issues)

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

    # Return the refined text
    if isinstance(response, dict) and "refined_text" in response:
        return response["refined_text"]
    elif isinstance(response, str):
        return response
    else:
        # Fallback: return the full response as string
        return str(response)
