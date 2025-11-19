"""
Paraphraser Agent Tool

Transforms input text to sound like an experienced tradesperson.
"""

from langchain_core.tools import tool
from typing import Optional
from modules.chain import Chain
from .prompts import PARAPHRASER_PROMPT_TEMPLATE


@tool
def paraphrase_text(
    text: str,
    trade: str,
    specialization: Optional[str] = None,
    preserve_formatting: bool = True,
    target_length: Optional[str] = "similar"
) -> str:
    """
    Paraphrase text to sound like an experienced tradesperson.

    Use this tool to transform monotonous or overly formal technical text
    into clear, conversational language that sounds like a real technician.

    Args:
        text: The text to paraphrase
        trade: The trade context (e.g., "HVAC", "Electrical", "Plumbing")
        specialization: Optional sub-specialization (e.g., "Residential HVAC")
        preserve_formatting: Whether to maintain structure (lists, paragraphs)
        target_length: "similar", "concise", or "expanded"

    Returns:
        Paraphrased text that sounds like a real tradesperson wrote it
    """
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

    # Return the paraphrased text
    if isinstance(response, dict) and "paraphrased_text" in response:
        return response["paraphrased_text"]
    elif isinstance(response, str):
        return response
    else:
        # Fallback: return the full response as string
        return str(response)
