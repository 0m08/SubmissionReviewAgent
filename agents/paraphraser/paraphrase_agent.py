"""
Paraphraser Agent Tool

Transforms input text to sound like an experienced tradesperson.
"""

from langchain_core.tools import tool, InjectedToolCallId
from typing import Optional, Annotated
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from modules.chain import Chain
from .prompts import PARAPHRASER_PROMPT_TEMPLATE


@tool
def paraphrase_text(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState]
) -> Command:
    """
    Paraphrase text to sound like an experienced tradesperson.

    Use this tool to transform monotonous or overly formal technical text
    into clear, conversational language that sounds like a real technician.

    The tool reads parameters from the state:
    - original_text: The text to paraphrase
    - trade: Trade context (e.g., "HVAC", "Electrical", "Plumbing")
    - specialization: Optional sub-specialization
    - preserve_formatting: Whether to maintain structure
    - target_length: "similar", "concise", or "expanded"

    Returns:
        Paraphrased text that sounds like a real tradesperson wrote it
    """
    # Extract parameters from state
    text = state.get("original_text", "")
    trade = state.get("trade", "HVAC")
    specialization = state.get("specialization")
    preserve_formatting = state.get("preserve_formatting", True)
    target_length = state.get("target_length", "similar")

    # Build prompt from template
    prompt = PARAPHRASER_PROMPT_TEMPLATE.format(
        trade=trade,
        specialization=specialization or "General",
        input_text=text,
        preserve_formatting="yes" if preserve_formatting else "no",
        target_length=target_length or "similar"
    )

    # Execute with Chain (using gemini_2_5_flash)
    chain = Chain(llm='gemini_2_5_flash', tags=["paraphrased_text"])
    chain.add_message(role="user", content=prompt)
    response = chain.run()

    paraphrased_text = response["paraphrased_text"]

    # Return Command with state updates and tool message
    return Command(
        update={
            "paraphrased_text": paraphrased_text,
            "final_text": paraphrased_text,
            "messages": [
                ToolMessage(
                    f"Text successfully paraphrased for {state.get('trade', 'HVAC')}.",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )
