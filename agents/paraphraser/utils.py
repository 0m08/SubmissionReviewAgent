"""
Utility functions for paraphraser agent
"""

import re
from typing import Optional


def extract_from_xml_tags(text: str, tag: str) -> str:
    """
    Extract content from XML tags in text.

    Args:
        text: Text containing XML tags
        tag: Tag name to extract (without angle brackets)

    Returns:
        Extracted content, or original text if tags not found
    """
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip()

    # Fallback: return original text
    return text.strip()
