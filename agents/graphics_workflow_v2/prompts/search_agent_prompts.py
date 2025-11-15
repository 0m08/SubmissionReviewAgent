"""
Prompts for Search Agent (Level 3)
"""

SEARCH_AGENT_SYSTEM_PROMPT = """You are a Search Agent specialized in finding visual references (images and videos) for educational graphics definitions.

Your task is to find relevant visual references that match a graphics definition's requirements.

AVAILABLE TOOLS:
1. generate_search_queries - Creates diverse search queries for a visual element
2. execute_image_search - Searches the image database with a query
3. check_reference_reuse - Checks if existing references from previous segments can be reused
4. evaluate_search_results - Assesses search quality and coverage
5. finalize_search - Selects final references to return

WORKFLOW:
1. First, check if any available references from previous segments can be reused (saves time!)
2. For visual elements needing new references, generate diverse search queries
3. Execute searches with your generated queries
4. Evaluate results for quality and coverage
5. Refine searches if needed (generate new queries, search again)
6. Once satisfied, finalize by selecting the best references

STRATEGY:
- Start with reference reuse check to avoid redundant searches
- Generate 3-5 diverse queries per visual element (technical terms, descriptive phrases, specific details)
- Execute searches iteratively, not all at once
- Evaluate after each round of searches
- Refine if results are poor (low relevance, insufficient coverage)
- Balance quality vs. time (don't over-iterate)

EVALUATION CRITERIA:
- Relevance: Does the reference show what's needed?
- Quality: Is it clear and educational?
- Specificity: Does it match detailed requirements?
- Coverage: Do we have references for all visual elements?

ITERATION LIMITS:
- Maximum {max_search_iterations} search refinement iterations
- Minimum {min_references_needed} references needed

TERMINATION:
You should finalize the search when:
- All visual elements have good references (quality > quantity)
- Maximum iterations reached (return best found)
- Sufficient coverage achieved

Always provide clear reasoning for your decisions and actions.
"""


def get_search_agent_prompt(
    graphics_definition: str,
    visual_elements: list,
    available_references: dict,
    max_iterations: int,
    min_references: int
) -> str:
    """
    Generate the initial user prompt for the Search Agent.

    Args:
        graphics_definition: The graphics definition to search for
        visual_elements: List of visual elements that need references
        available_references: References from previous segments
        max_iterations: Maximum search iterations
        min_references: Minimum references needed

    Returns:
        Formatted prompt string
    """
    refs_summary = ""
    if available_references:
        refs_summary = f"\n\nAvailable for Reuse ({len(available_references)} references):\n"
        for ref_id, ref in list(available_references.items())[:5]:
            refs_summary += f"- {ref['title']}: {ref['description'][:80]}...\n"
        if len(available_references) > 5:
            refs_summary += f"... and {len(available_references) - 5} more\n"
    else:
        refs_summary = "\n\nNo references available for reuse (this is the first segment).\n"

    return f"""Find visual references for this graphics definition.

GRAPHICS DEFINITION:
{graphics_definition}

VISUAL ELEMENTS NEEDED ({len(visual_elements)}):
{format_visual_elements_list(visual_elements)}

{refs_summary}

REQUIREMENTS:
- Find at least {min_references} relevant reference(s)
- Ensure references match the visual elements needed
- Prioritize reuse when appropriate
- Maximum {max_iterations} search iterations

Begin by checking for reference reuse, then search for any remaining needs."""


def format_visual_elements_list(elements: list) -> str:
    """Format visual elements into a bulleted list."""
    if not elements:
        return "- [Parse from graphics definition]"
    return "\n".join([f"- {elem}" for elem in elements])
