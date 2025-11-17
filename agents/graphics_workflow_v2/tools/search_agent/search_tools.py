"""
Search Agent Tools (Level 3)

Tools for the Search Agent to find and validate visual references.
"""

from typing import Annotated, List, Dict, Any
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_core.messages import ToolMessage

from agents.graphics_workflow_v2.state.schemas import SearchAgentState, ReferenceData
from agents.graphics_workflow_v2.config.settings import (
    SEARCH_QUERIES_PER_ELEMENT,
    SEARCH_K,
    RELEVANCE_THRESHOLD,
    DEFINITION_CHAIN_MODEL,
)
from modules.chain import Chain
from agents.vector_store_image_search.graphics_retriever import graphics_retriever
import uuid


@tool
def generate_search_queries(
    visual_element: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[Dict, InjectedState]
) -> Command:
    """
    Generates diverse search queries for a specific visual element.

    Creates 3-5 variations including:
    - Technical terminology
    - Descriptive phrases
    - Specific details from graphics definition
    - Educational context keywords

    Args:
        visual_element: What visual element to generate queries for

    Returns:
        Formatted list of generated queries
    """
    graphics_definition = state.get("graphics_definition", "")

    # Create Chain for query generation
    chain = Chain(
        llm=DEFINITION_CHAIN_MODEL,
        tags=["queries"]
    )

    prompt = f"""Generate {SEARCH_QUERIES_PER_ELEMENT} diverse search queries to find images/videos for this visual element.

Visual Element: {visual_element}

Context from full graphics definition:
{graphics_definition}

Guidelines:
- Create queries with different angles: technical terms, descriptive phrases, specific details
- Consider educational/HVAC context
- Keep queries concise but specific
- Include variations that might match different types of visual references

Output format:
<queries>
- Query 1
- Query 2
- Query 3
</queries>"""

    chain.add_message(role="user", content=prompt)
    response = chain.run()

    # Extract queries
    if isinstance(response, dict):
        response = response.get("content", str(response))

    parsed = chain.extract_text_in_tags(response)
    queries_text = parsed.get("queries", "")

    # Parse queries into list
    queries = [
        q.strip().lstrip("- ").strip()
        for q in queries_text.split("\n")
        if q.strip() and q.strip() != "-"
    ]

    # Update state with generated queries
    existing_queries = state.get("search_queries", [])
    updated_queries = existing_queries + queries

    # Format response
    formatted_queries = "\n".join([f"{i+1}. {q}" for i, q in enumerate(queries)])
    result_message = f"Generated {len(queries)} search queries for '{visual_element}':\n{formatted_queries}"

    return Command(
        update={
            "search_queries": updated_queries,
            "messages": [
                ToolMessage(
                    result_message,
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool
def execute_image_search(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[Dict, InjectedState]
) -> Command:
    """
    Executes semantic/vector search against the image database.

    Args:
        query: Search query string

    Returns:
        Formatted list of results with relevance scores
    """
    # Get drive instance from state (will be passed from parent)
    drive = state.get("drive")
    if not drive:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        "Error: Drive instance not available for search",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )

    # Get configuration
    k = state.get("search_k", SEARCH_K)
    filters = state.get("filters")
    root_folder_id = state.get("root_folder_id")

    # Execute search
    try:
        results = graphics_retriever(
            query=query,
            drive=drive,
            k=k,
            filters=filters,
            root_folder_id=root_folder_id
        )
    except Exception as e:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        f"Error: Search failed - {str(e)}",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )

    # Convert results to ReferenceData format
    reference_data_list = []
    for idx, result in enumerate(results):
        metadata = result.get("metadata", {})

        # Generate unique ID
        ref_id = metadata.get("image_id", str(uuid.uuid4()))

        # Determine type (image or video) from MIME type
        mime_type = metadata.get("mime_type", "")
        ref_type = "video" if "video" in mime_type.lower() else "image"

        # Get URL from metadata (drive_url field)
        url = metadata.get("drive_url", "")

        # Get title from metadata
        title = metadata.get("image_title") or metadata.get("name", "Untitled")

        # Get description
        description = metadata.get("description", "")

        # Calculate relevance score from similarity (lower distance = higher relevance)
        # Similarity is a distance metric, so we invert it
        similarity_score = result.get("similarity", 1.0)
        relevance_score = max(0.0, min(1.0, 1.0 - similarity_score))

        reference_data: ReferenceData = {
            "reference_id": ref_id,
            "url": url,
            "type": ref_type,
            "title": title,
            "description": description,
            "source": metadata.get("source", "Google Drive"),
            "timestamp": None,
            "relevance_score": relevance_score,
            "search_query": query,
            "metadata": metadata,
            "reused_from_segment": None
        }

        reference_data_list.append(reference_data)

    # Filter by relevance threshold
    filtered_results = [
        ref for ref in reference_data_list
        if ref["relevance_score"] >= RELEVANCE_THRESHOLD
    ]

    # Update state
    existing_results = state.get("search_results", [])
    updated_results = existing_results + filtered_results

    # Format response
    result_message = f"Search results for '{query}':\nFound {len(filtered_results)} relevant results (threshold: {RELEVANCE_THRESHOLD})\n\n"

    for i, ref in enumerate(filtered_results[:5]):  # Show top 5
        result_message += f"{i+1}. {ref['title']}\n"
        result_message += f"   Type: {ref['type']} | Relevance: {ref['relevance_score']:.2f}\n"
        result_message += f"   Description: {ref['description'][:100]}...\n\n"

    if len(filtered_results) > 5:
        result_message += f"... and {len(filtered_results) - 5} more results.\n"

    return Command(
        update={
            "search_results": updated_results,
            "messages": [
                ToolMessage(
                    result_message,
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool
def check_reference_reuse(
    visual_element: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[Dict, InjectedState]
) -> Command:
    """
    Checks if existing references from previous segments can be reused.

    Evaluates available references against the current visual element needs.
    Returns matching references if found, avoiding redundant searches.

    Args:
        visual_element: What visual element to check for

    Returns:
        List of reusable references (if any)
    """
    available_references = state.get("available_references", {})

    if not available_references:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        "No available references to check for reuse",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )

    # Use LLM to evaluate if any references match
    chain = Chain(
        llm=DEFINITION_CHAIN_MODEL,
        tags=["matches", "reasoning"]
    )

    # Format available references
    refs_text = ""
    for ref_id, ref in available_references.items():
        refs_text += f"- [{ref_id}] {ref['title']}: {ref['description']}\n"

    prompt = f"""Evaluate if any of these existing references can be reused for the following visual element need.

Visual Element Needed: {visual_element}

Available References:
{refs_text}

Determine which (if any) references are suitable for reuse. Consider:
- Does the reference show what's needed?
- Is it relevant enough to be reused?
- Would it save time compared to searching for new references?

Output format:
<matches>
[List reference IDs that match, one per line, or "NONE" if no matches]
</matches>

<reasoning>
[Brief explanation of why each reference does or doesn't match]
</reasoning>"""

    chain.add_message(role="user", content=prompt)
    response = chain.run()

    if isinstance(response, dict):
        response = response.get("content", str(response))

    parsed = chain.extract_text_in_tags(response)
    matches_text = parsed.get("matches", "NONE").strip()
    reasoning = parsed.get("reasoning", "")

    if matches_text == "NONE" or not matches_text:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        "No reusable references found for this visual element",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )

    # Extract reference IDs
    matched_ids = [
        line.strip()
        for line in matches_text.split("\n")
        if line.strip() and line.strip() != "NONE"
    ]

    # Get matched references
    matched_references = [
        available_references[ref_id]
        for ref_id in matched_ids
        if ref_id in available_references
    ]

    if not matched_references:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        "No valid reusable references found",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )

    # Add to selected references
    existing_selected = state.get("selected_references", [])
    updated_selected = existing_selected + matched_references

    result_message = f"Found {len(matched_references)} reusable reference(s) for '{visual_element}':\n"
    for ref in matched_references:
        result_message += f"- {ref['title']} (from segment {ref.get('reused_from_segment', '?')})\n"
    result_message += f"\nReasoning: {reasoning}"

    return Command(
        update={
            "selected_references": updated_selected,
            "messages": [
                ToolMessage(
                    result_message,
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool
def evaluate_search_results(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[Dict, InjectedState]
) -> Command:
    """
    Evaluates the quality and coverage of current search results.

    Checks:
    - Do results match the visual elements needed?
    - Are relevance scores acceptable?
    - Is there sufficient coverage?
    - Should searches be refined?

    Returns:
        Assessment and recommendations
    """
    graphics_definition = state.get("graphics_definition", "")
    visual_elements = state.get("visual_elements", [])
    search_results = state.get("search_results", [])
    selected_references = state.get("selected_references", [])
    min_needed = state.get("min_references_needed", len(visual_elements))

    total_references = len(search_results) + len(selected_references)

    # Use LLM to evaluate
    chain = Chain(
        llm=DEFINITION_CHAIN_MODEL,
        tags=["assessment", "recommendations"]
    )

    prompt = f"""Evaluate the search results for completeness and quality.

Graphics Definition:
{graphics_definition}

Visual Elements Needed ({len(visual_elements)}):
{', '.join(visual_elements)}

Current Status:
- Search results found: {len(search_results)}
- Reused references: {len(selected_references)}
- Total references: {total_references}
- Minimum needed: {min_needed}

Top Search Results:
{format_results_for_evaluation(search_results[:10])}

Evaluate:
1. Do we have sufficient coverage of visual elements?
2. Are the relevance scores good enough?
3. Should we continue searching or refine queries?
4. Which references should be selected?

Output format:
<assessment>
[Overall quality assessment: EXCELLENT/GOOD/FAIR/POOR]
</assessment>

<recommendations>
[Specific recommendations: continue searching, refine queries, finalize, etc.]
</recommendations>"""

    chain.add_message(role="user", content=prompt)
    response = chain.run()

    if isinstance(response, dict):
        response = response.get("content", str(response))

    parsed = chain.extract_text_in_tags(response)
    assessment = parsed.get("assessment", "FAIR")
    recommendations = parsed.get("recommendations", "Continue searching")

    result_message = f"""
SEARCH EVALUATION:
- Assessment: {assessment}
- Total References: {total_references} / {min_needed} needed
- Search Results Quality: {calculate_avg_relevance(search_results):.2f} avg relevance

Recommendations:
{recommendations}
"""

    return Command(
        update={
            "messages": [
                ToolMessage(
                    result_message,
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool
def finalize_search(
    selected_reference_ids: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[Dict, InjectedState]
) -> Command:
    """
    Finalizes the search by selecting the best references to return.

    Args:
        selected_reference_ids: Comma-separated IDs of references to include

    Returns:
        Summary of selected references
    """
    # Parse IDs
    ids = [id.strip() for id in selected_reference_ids.split(",") if id.strip()]

    search_results = state.get("search_results", [])
    already_selected = state.get("selected_references", [])

    # Find references by ID
    final_selected = already_selected.copy()

    for ref in search_results:
        if ref["reference_id"] in ids:
            final_selected.append(ref)

    # Update state
    result_message = f"Search finalized with {len(final_selected)} selected references:\n"
    for i, ref in enumerate(final_selected):
        result_message += f"{i+1}. {ref['title']} ({ref['type']}) - Relevance: {ref['relevance_score']:.2f}\n"

    return Command(
        update={
            "selected_references": final_selected,
            "messages": [
                ToolMessage(
                    result_message,
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


# ============================================================================
# Helper Functions
# ============================================================================

def format_results_for_evaluation(results: List[ReferenceData]) -> str:
    """Format search results for LLM evaluation."""
    formatted = ""
    for i, ref in enumerate(results):
        formatted += f"{i+1}. {ref['title']} (Score: {ref['relevance_score']:.2f})\n"
        formatted += f"   {ref['description'][:100]}...\n"
    return formatted


def calculate_avg_relevance(results: List[ReferenceData]) -> float:
    """Calculate average relevance score."""
    if not results:
        return 0.0
    return sum(ref["relevance_score"] for ref in results) / len(results)
