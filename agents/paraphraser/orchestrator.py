"""
Paraphraser Orchestrator

Coordinates the multi-agent paraphrasing workflow using LangGraph.
"""

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
from typing import Dict, Optional
import json

from .paraphrase_agent import paraphrase_text
from .review_agent import review_quality
from .refine_agent import refine_text
from .prompts import ORCHESTRATOR_SYSTEM_PROMPT
from .state import ParaphraserState


def create_paraphraser_orchestrator(
    max_iterations: int = 3,
    llm_model: str = "openai:gpt-4o-mini"
):
    """
    Create the multi-agent paraphraser orchestrator.

    Args:
        max_iterations: Maximum refinement iterations
        llm_model: LLM to use for orchestrator (default: gpt-4o-mini)

    Returns:
        Compiled LangGraph ReAct agent
    """
    # Initialize LLM
    llm = init_chat_model(llm_model)

    # Define tools available to orchestrator
    tools = [paraphrase_text, review_quality, refine_text]

    # Create ReAct agent with tools
    graph = create_react_agent(
        model=llm,
        tools=tools,
        state_schema=ParaphraserState,
    )

    return graph


def run_paraphraser(
    text: str,
    trade: str = "HVAC",
    specialization: Optional[str] = None,
    preserve_formatting: bool = True,
    target_length: Optional[str] = "similar",
    max_iterations: int = 3,
    llm_model: str = "openai:gpt-4o-mini"
) -> Dict:
    """
    Main entry point for paraphrasing text.

    This function orchestrates the entire paraphrasing workflow:
    1. Paraphrases the text
    2. Reviews quality
    3. Refines if needed
    4. Returns final result

    Args:
        text: Text to paraphrase
        trade: Trade context (e.g., "HVAC", "Electrical")
        specialization: Optional sub-specialization
        preserve_formatting: Whether to preserve structure
        target_length: "similar", "concise", or "expanded"
        max_iterations: Max refinement loops
        llm_model: LLM model to use

    Returns:
        Dictionary with:
        - final_text: The paraphrased result
        - status: "completed" or "failed"
        - quality_report: Final quality assessment
        - refinement_count: Number of refinements performed
        - messages: Full message history
    """
    # Create orchestrator
    orchestrator = create_paraphraser_orchestrator(
        max_iterations=max_iterations,
        llm_model=llm_model
    )

    # Build system prompt with max_iterations
    system_prompt = ORCHESTRATOR_SYSTEM_PROMPT.format(
        max_iterations=max_iterations
    )

    # Build user instructions
    user_instructions = f"""Please paraphrase the following text using this workflow:

1. First, call the paraphrase_text tool with these parameters:
   - text: (the input text below)
   - trade: "{trade}"
   - specialization: "{specialization or 'General'}"
   - preserve_formatting: {preserve_formatting}
   - target_length: "{target_length or 'similar'}"

2. Then call review_quality to check the paraphrased output

3. If the review fails (passed=false), call refine_text to fix the issues

4. Review again if needed (max {max_iterations} refinements)

5. Once quality passes or max iterations reached, present the final text to me

Here's the text to paraphrase:

---
{text}
---

Start by calling paraphrase_text.
"""

    # Initialize state
    initial_state = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_instructions)
        ],
        "original_text": text,
        "trade": trade,
        "specialization": specialization,
        "preserve_formatting": preserve_formatting,
        "target_length": target_length,
        "refinement_count": 0,
        "iteration_history": [],
        "status": "in_progress"
    }

    # Run orchestrator
    try:
        final_state = orchestrator.invoke(
            initial_state,
            {"recursion_limit": 50}
        )

        # Extract results from message history
        result = extract_results_from_state(final_state, text)

        return result

    except Exception as e:
        # Handle errors gracefully
        return {
            "final_text": text,  # Return original on error
            "status": "failed",
            "error": str(e),
            "quality_report": None,
            "refinement_count": 0,
            "messages": []
        }


def extract_results_from_state(state: Dict, original_text: str) -> Dict:
    """
    Extract final results from the orchestrator state.

    Parses the message history to find:
    - The final paraphrased/refined text
    - Quality review results
    - Refinement count

    Args:
        state: Final state from orchestrator
        original_text: Original input (fallback)

    Returns:
        Structured result dictionary
    """
    messages = state.get("messages", [])

    final_text = None
    quality_report = None
    refinement_count = 0

    # Parse messages in reverse to find most recent results
    for msg in reversed(messages):
        content = str(msg.content) if hasattr(msg, 'content') else str(msg)

        # Look for paraphrased or refined text
        if final_text is None:
            if "paraphrase_text" in content or "refine_text" in content:
                # This might be a tool call result
                # Try to extract the text content
                lines = content.split('\n')
                for line in lines:
                    if line.strip() and len(line.strip()) > 50:
                        # Likely the actual text content
                        final_text = line.strip()
                        break

        # Look for quality review
        if quality_report is None and "review_quality" in content:
            try:
                # Try to extract JSON from review
                if "{" in content:
                    json_start = content.find("{")
                    json_end = content.rfind("}") + 1
                    json_str = content[json_start:json_end]
                    quality_report = json.loads(json_str)
            except:
                pass

        # Count refinements
        if "refine_text" in content:
            refinement_count += 1

    # Fallback: extract from last AI message
    if final_text is None:
        for msg in reversed(messages):
            if hasattr(msg, 'type') and msg.type == 'ai':
                content = str(msg.content)
                if len(content) > 50:
                    final_text = content
                    break

    # Final fallback
    if final_text is None:
        final_text = original_text

    return {
        "final_text": final_text,
        "status": "completed",
        "quality_report": quality_report,
        "refinement_count": refinement_count // 2,  # Divide by 2 (call + result)
        "messages": messages,
        "metadata": {
            "trade": state.get("trade"),
            "specialization": state.get("specialization"),
            "preserve_formatting": state.get("preserve_formatting"),
            "target_length": state.get("target_length"),
        }
    }
