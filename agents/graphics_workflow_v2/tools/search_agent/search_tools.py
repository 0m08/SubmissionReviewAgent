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
    WEB_SEARCH_K,
    RELEVANCE_THRESHOLD,
    DEFINITION_CHAIN_MODEL,
)
from modules.chain import Chain
from agents.vector_store_image_search.graphics_retriever import graphics_retriever
from agents.vector_store_image_search.graphics_retriever_agent import pil_to_base64_data_uri
from agents.vector_store_image_search.create_vectorstore import download_image_from_drive
from agents.vector_store_image_search.web_image_search_tool import web_image_search_tool as web_search_function
from services.llm_service import llm_with_retry
from PIL import Image
import requests
from io import BytesIO
import uuid
import re


def extract_drive_file_id(url: str) -> str:
    """
    Extract the Google Drive file ID from various URL formats.
    
    Supported formats:
    - https://drive.google.com/file/d/FILE_ID/view
    - https://drive.google.com/open?id=FILE_ID
    - https://drive.google.com/uc?id=FILE_ID
    - Just the FILE_ID itself
    
    Returns:
        The file ID string, or None if not found
    """
    if not url:
        return None
    
    # Pattern for /file/d/FILE_ID/
    match = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    
    # Pattern for ?id=FILE_ID or &id=FILE_ID
    match = re.search(r'[?&]id=([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    
    # If it looks like just a file ID (no slashes, reasonable length)
    if '/' not in url and 'http' not in url.lower() and len(url) > 10:
        return url
    
    return None


def load_image_from_drive_url(url: str, drive: Any, ref_id: str = "") -> Image.Image:
    """
    Load an image from a Google Drive URL using the Drive API.
    
    Args:
        url: Google Drive URL or file ID
        drive: Google Drive instance
        ref_id: Reference ID for logging
        
    Returns:
        PIL Image or None if loading fails
    """
    if not url or not drive:
        return None
    
    file_id = extract_drive_file_id(url)
    if not file_id:
        print(f"         ⚠️  Could not extract file ID from URL for {ref_id}: {url}")
        return None
    
    try:
        img = download_image_from_drive(drive, file_id)
        if img:
            print(f"         ✅ Loaded image from Drive for {ref_id}")
        return img
    except Exception as e:
        print(f"         ⚠️  Failed to load image from Drive for {ref_id}: {e}")
        return None


@tool
def generate_search_queries(
    vo_text: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[Dict, InjectedState]
) -> Command:
    """
    Generates diverse search queries directly from a voiceover sentence.

    Args:
        vo_text: The voiceover sentence to generate queries for

    Returns:
        Formatted list of generated queries
    """
    slide_chunk = state.get("slide_chunk", "")
    course_context = state.get("course_context", {})
    course_name = course_context.get("course_name", "") if course_context else ""
    topic_name = course_context.get("topic", "") if course_context else ""
    subtopic_name = course_context.get("subtopic", "") if course_context else ""
    
    print(f"\n         {'─'*45}")
    print(f"         📝 Tool: generate_search_queries ▶ START")
    print(f"         {'─'*45}")
    print(f"            🎙️  VO sentence: \"{vo_text}\"")
    print(f"            📄 Slide context: {len(slide_chunk)} chars")

    # Create Chain for query generation
    chain = Chain(
        llm=DEFINITION_CHAIN_MODEL,
        tags=["queries", "evaluation_breakdown", "output"]
    )

#     prompt = f"""You are a Search Query Generator agent responsible for creating high-quality search queries that will be used to retrieve images from a vector store of image embeddings and web-based image search engines.

# YOUR TASK:
# Generate {SEARCH_QUERIES_PER_ELEMENT} diverse, high-signal search queries that accurately represent the visual element. These queries will be passed into a semantic/vector search engine, so they must be clear, specific, and varied.

# Course Name: {course_name}

# Topic Name: {topic_name}

# Subtopic Name: {subtopic_name}

# VISUAL ELEMENT (the exact visual requirement to base your queries on):
# {visual_element}

# GUIDELINES FOR CREATING QUERIES:
# 1. Derive all search queries directly from the visual element.
#    - Use its technical terms, objects, components, etc.
#    - All important details from the visual element must appear in at least one of the queries.

# 2. Use multiple perspectives:
#    - Technical domain-specific terminology when relevant (e.g., HVAC, electrical, mechanical terms)
#    - Descriptive visual characteristics
#    - Diagram-oriented phrasing (e.g., "labeled schematic", "cross-section", etc.)
#    - Functional/operational descriptors if relevant

# 3. Query Length:
#    - Keep queries short (around 4–10 words), but packed with meaningful descriptors.
#    - Avoid extremely long queries or full-sentence descriptions.

# 4. Ensure diversity and relevance:
#    - Queries must NOT be small variations of each other.
#    - Each query must explore a different search angle.
#    - Phrase queries the way images are commonly labeled in textbooks, stock photos, technical diagrams, or typical image searches in a web browser
#    - Ensure the query generated is a proper search phrase that will yield relevant results.

# OUTPUT FORMAT:

# Strictly provide your output in the following format:

# <queries>
# - Query 1 text
# - Query 2 text
# - Query 3 text
# </queries>
# """

    prompt = f"""You are a Search Query Generator agent specializing in the field of HVAC. Your task is to generate concise, high-quality image search queries that can be used to retrieve relevant images from a vector store of image embeddings and web-based image search engines. The retrieved images will then be used as on-screen visuals in an educational e-learning slide while the given voiceover sentence is being narrated.

You will be given a single voiceover sentence from an educational slide, along with its course information and the whole slide content for context. First, reason internally about what visual elements would need to be shown on screen for the voiceover sentence to be clearly understood. Then, based on that reasoning, generate {SEARCH_QUERIES_PER_ELEMENT} image search queries that would retrieve the most relevant visuals.
 
These are the inputs:

Course Name: {course_name}

Topic Name: {topic_name}

Subtopic Name: {subtopic_name}

Voiceover sentence for which you need to generate search queries:
"{vo_text}"

Full slide content:
{slide_chunk}

Instructions:

1. Base all search queries strictly on the meaning of the voiceover sentence.
   - Do not generate queries based on general topic knowledge alone.
   - The queries should help retrieve images that are visually necessary to understand this specific sentence.

2. Generate concise, image-focused search queries.
   - Use concrete object names, components, or diagrams that are likely to appear in images.
   - Avoid abstract, instructional, or process-oriented wording.

3. Phrase queries the way images are commonly searched for or labeled.
   - Use short, keyword-based phrases.
   - Prefer phrasing typical of textbooks, technical diagrams, stock photos, or browser image searches.

4. Ensure each query adds value.
   - Queries should not be near-duplicates of each other.
   - Each query should represent a slightly different but relevant visual angle.

Always provide your output strictly in the following format:

<output>

<evaluation_breakdown>

1. Voiceover Meaning
- Briefly explain, in your own words, what the voiceover sentence is communicating.

2. Query Planning
- Reason about the kinds of image searches that would best retrieve visuals to support this sentence.
- Consider how such images are typically searched for or labeled.

</evaluation_breakdown>

(Based on your above evaluation, provide your final output of search queries)

<queries>
- Query 1 text
- Query 2 text
- Query 3 text
</queries>

</output>
"""
    
    # # Print the prompt being used
    # print(f"\n{'='*80}")
    # print(f"🔍 SEARCH QUERY GENERATOR PROMPT")
    # print(f"{'='*80}")
    # print(prompt)
    # print(f"{'='*80}\n")
    
    chain.add_message(role="user", content=prompt)
    response = chain.run()
    
    # # Print the raw response output
    # print(f"\n{'='*80}")
    # print(f"📤 SEARCH QUERY GENERATOR OUTPUT")
    # print(f"{'='*80}")
    # if isinstance(response, dict):
    #     print(str(response))
    # else:
    #     print(str(response))
    # print(f"{'='*80}\n")

    # Extract queries (Chain handles tag extraction)
    if isinstance(response, dict):
        parsed = response
    else:
        parsed = chain.extract_text_in_tags(str(response))
    queries_text = parsed.get("queries", parsed.get("text", ""))

    # Parse queries into list
    queries = [
        q.strip().lstrip("- ").strip()
        for q in queries_text.split("\n")
        if q.strip() and q.strip() != "-"
    ]

    # Format response
    formatted_queries = "\n".join([f"{i+1}. {q}" for i, q in enumerate(queries)])
    result_message = f"Generated {len(queries)} search queries for VO sentence: '{vo_text}':\n{formatted_queries}"
    
    print(f"\n         {'─'*45}")
    print(f"         📝 Tool: generate_search_queries ◀ END")
    print(f"         {'─'*45}")
    print(f"            ✅ Generated {len(queries)} search queries:")
    for i, q in enumerate(queries, 1):
        print(f"               {i}. \"{q}\"")
    print(f"         {'─'*45}\n")

    return Command(
        update={
            "search_queries": queries,  # Just return new queries, reducer will merge
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
    
    # Check if this specific tool has already executed this query
    executed_drive_queries = state.get("executed_drive_queries", set())
    if query in executed_drive_queries:
        print(f"\n         {'─'*45}")
        print(f"         🔎 Tool: execute_image_search ▶ SKIP (duplicate)")
        print(f"         {'─'*45}")
        print(f"            📝 Query: \"{query}\"")
        print(f"            ⚠️  This query was already executed in Drive search, skipping to avoid duplicate")
        print(f"         {'─'*45}\n")
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        f"Query already executed in Drive search, skipping duplicate: {query}",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    print(f"\n         {'─'*45}")
    print(f"         🔎 Tool: execute_image_search ▶ START")
    print(f"         {'─'*45}")
    print(f"            📝 Query: \"{query}\"")
    
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
        print(f"         Raw results: {len(results)} items")
    except Exception as e:
        print(f"         ❌ Search error: {str(e)}")
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
        # Similarity is a distance metric (distance), so we invert it for display
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
            "metadata": {**metadata, "similarity_distance": similarity_score},  # Store raw distance for filtering
            "reused_from_segment": None
        }

        reference_data_list.append(reference_data)

    # Filter by relevance threshold (compare against raw distance, not converted relevance_score)
    # RELEVANCE_THRESHOLD is a distance metric, so lower values = better matches
    filtered_results = [
        ref for ref in reference_data_list
        if ref["metadata"].get("similarity_distance", 999) <= RELEVANCE_THRESHOLD
    ]
    
    # Remove duplicates by reference_id (in case same search runs multiple times)
    existing_results = state.get("search_results", [])
    existing_ids = {ref["reference_id"] for ref in existing_results}
    new_results = [
        ref for ref in filtered_results
        if ref["reference_id"] not in existing_ids
    ]

    # Format response
    result_message = f"Search results for '{query}':\nFound {len(new_results)} new relevant results (threshold: {RELEVANCE_THRESHOLD})\n\n"
    
    print(f"\n         {'─'*45}")
    print(f"         🔎 Tool: execute_image_search ◀ END")
    print(f"         {'─'*45}")
    print(f"            📊 Raw results from retriever: {len(results)}")
    print(f"            🎯 After relevance filter (≤{RELEVANCE_THRESHOLD}): {len(filtered_results)}")
    print(f"            ✅ New unique results: {len(new_results)}")
    if filtered_results and not new_results:
        print(f"            ℹ️  Note: All {len(filtered_results)} filtered results were duplicates")
    if new_results:
        print(f"            📎 Results:")
        for i, ref in enumerate(new_results, 1):
            print(f"               {i}. [{ref['reference_id']}] {ref['title']} | score: {ref['relevance_score']:.2f}")
    print(f"         {'─'*45}\n")

    for i, ref in enumerate(new_results[:5]):  # Show top 5
        result_message += f"{i+1}. [{ref['reference_id']}] {ref['title']}\n"
        result_message += f"   Type: {ref['type']} | Relevance: {ref['relevance_score']:.2f}\n"
        if ref.get("url"):
            result_message += f"   URL: {ref['url']}\n"
        result_message += f"   Description: {ref['description'][:100]}...\n\n"

    if len(new_results) > 5:
        result_message += f"... and {len(new_results) - 5} more results.\n"
    
    # Mark query as executed in Drive search (separate from web search)
    executed_drive_queries.add(query)
    
    return Command(
        update={
            "executed_drive_queries": executed_drive_queries,  # Track executed Drive queries to prevent duplicates
            "search_results": new_results,  # Just return new results (duplicates filtered), reducer will merge
            "messages": [
                ToolMessage(
                    result_message,
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool
def execute_web_image_search(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[Dict, InjectedState]
) -> Command:
    """
    Executes web image search via Google Images Search API.

    Searches the public web for images matching the query. Results are automatically
    filtered, deduplicated, and converted to ReferenceData format to match Drive search results.

    Args:
        query: Search query string

    Returns:
        Formatted list of results with relevance scores
    """
    # Check if this specific tool has already executed this query
    executed_web_queries = state.get("executed_web_queries", set())
    if query in executed_web_queries:
        print(f"\n         {'─'*45}")
        print(f"         🌐 Tool: execute_web_image_search ▶ SKIP (duplicate)")
        print(f"         {'─'*45}")
        print(f"            📝 Query: \"{query}\"")
        print(f"            ⚠️  This query was already executed in web search, skipping to avoid duplicate")
        print(f"         {'─'*45}\n")
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        f"Query already executed in web search, skipping duplicate: {query}",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    print(f"\n         {'─'*45}")
    print(f"         🌐 Tool: execute_web_image_search ▶ START")
    print(f"         {'─'*45}")
    print(f"            📝 Query: \"{query}\"")

    # Get configuration from state (use WEB_SEARCH_K for web search, separate from Drive search)
    k = state.get("web_search_k", WEB_SEARCH_K)

    # Execute web search using the existing function
    try:
        web_results_raw = web_search_function(
            query=query,
            k=k
        )
        
        print(f"         Raw results from web search: {len(web_results_raw)} items")
    except Exception as e:
        print(f"         ❌ Web search error: {str(e)}")
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        f"Error: Web search failed - {str(e)}",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )

    # Convert web search results to ReferenceData format
    reference_data_list = []
    for idx, result in enumerate(web_results_raw):
        # Extract data from the existing format
        pil_image = result.get("image")
        metadata = result.get("metadata", {})
        
        # Generate unique ID
        ref_id = str(uuid.uuid4())
        
        # Calculate relevance score based on rank (position-based)
        # Rank 1 = 1.0, Rank 2 = 0.92, Rank 3 = 0.84, etc. (decreasing by 0.08 per rank)
        rank = idx + 1
        relevance_score = max(0.0, min(1.0, 1.0 - ((rank - 1) * 0.08)))
        
        # Get URLs - the existing function uses "drive_url" and "source_url" in metadata
        image_url = metadata.get("source_url", "")
        source_url = metadata.get("drive_url", "") or image_url
        title = metadata.get("name", f"Web Image {rank}")
        
        # Get description from metadata if available, otherwise use random code
        description = metadata.get("description", "")
        if not description or not description.strip():
            # Generate short random code for fallback
            random_code = str(uuid.uuid4())[:8]
            description = f"image_{random_code}"
        
        reference_data: ReferenceData = {
            "reference_id": ref_id,
            "url": image_url,
            "type": "image",
            "title": title,
            "description": description,
            "source": "Web Search",
            "timestamp": None,
            "relevance_score": relevance_score,
            "search_query": query,
            "metadata": {
                "source_url": source_url,
                "image_url": image_url,
                "search_rank": rank,
                "search_source": "google_images",
                "pil_image": pil_image,  # Store PIL image for vision model use
            },
            "reused_from_segment": None
        }
        
        reference_data_list.append(reference_data)

    # Remove duplicates by URL (in case same search runs multiple times or overlaps with Drive results)
    existing_results = state.get("search_results", [])
    existing_urls = {ref.get("url", "") for ref in existing_results}
    new_results = [
        ref for ref in reference_data_list
        if ref.get("url") and ref["url"] not in existing_urls
    ]

    # Format response
    result_message = f"Web search results for '{query}':\nFound {len(new_results)} new unique results\n\n"
    
    print(f"\n         {'─'*45}")
    print(f"         🌐 Tool: execute_web_image_search ◀ END")
    print(f"         {'─'*45}")
    print(f"            📊 Raw results from web search: {len(web_results_raw)}")
    print(f"            ✅ New unique results: {len(new_results)}")
    if web_results_raw and not new_results:
        print(f"            ℹ️  Note: All {len(web_results_raw)} results were duplicates")
    if new_results:
        print(f"            📎 Results:")
        for i, ref in enumerate(new_results, 1):
            print(f"               {i}. [{ref['reference_id']}] {ref['title']} | score: {ref['relevance_score']:.2f}")
    print(f"         {'─'*45}\n")

    for i, ref in enumerate(new_results[:5]):  # Show top 5
        result_message += f"{i+1}. [{ref['reference_id']}] {ref['title']}\n"
        result_message += f"   Type: {ref['type']} | Relevance: {ref['relevance_score']:.2f}\n"
        if ref.get("url"):
            result_message += f"   URL: {ref['url']}\n"
        result_message += f"   Description: {ref['description'][:100]}...\n\n"

    if len(new_results) > 5:
        result_message += f"... and {len(new_results) - 5} more results.\n"
    
    # Mark query as executed in web search (separate from Drive search)
    executed_web_queries.add(query)
    
    return Command(
        update={
            "executed_web_queries": executed_web_queries,  # Track executed web queries to prevent duplicates
            "search_results": new_results,  # Just return new results (duplicates filtered), reducer will merge
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
    vo_text: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[Dict, InjectedState]
) -> Command:
    """
    Checks if existing references from previous segments can be reused.

    Evaluates available references against the current voiceover sentence needs, using both the image content and the text metadata.

    Args:
        vo_text: The voiceover sentence to check for reusable references

    Returns:
        List of reusable references (if any) via selected_references
    """
    available_references = state.get("available_references", {})
    
    print(f"\n         {'─'*45}")
    print(f"         ♻️  Tool: check_reference_reuse ▶ START")
    print(f"         {'─'*45}")
    print(f"            🎙️  VO sentence: \"{vo_text}\"")
    print(f"            📚 Available references to check: {len(available_references)}")

    if not available_references:
        print(f"            ℹ️  No references available for reuse")
        print(f"         {'─'*45}\n")
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

    # Get drive instance for loading images
    drive = state.get("drive")
    
    # Load images for ALL available references
    images_to_show: List[Dict[str, Any]] = []
    failed_refs: List[str] = []

    for ref_id, ref in available_references.items():
        url = ref.get("url")
        title = ref.get("title", "Untitled")
        description = ref.get("description", "")
        metadata = ref.get("metadata", {})
        source = ref.get("source", "")

        # Check if PIL image is already available in metadata (from web search)
        pil_image = metadata.get("pil_image")
        
        if pil_image:
            # Web search result - use the stored PIL image directly
            print(f"         ✅ Using pre-loaded image from Web Search for {ref_id}")
            images_to_show.append(
                {
                    "image": pil_image,
                    "ref_id": ref_id,
                    "title": title,
                    "description": description,
                }
            )
            continue

        if not url:
            print(f"         ⚠️  Reference {ref_id} has no URL; using text-only metadata")
            images_to_show.append(
                {
                    "image": None,
                    "ref_id": ref_id,
                    "title": title,
                    "description": description,
                }
            )
            continue

        # Drive search result - use Drive API to load image
        img = load_image_from_drive_url(url, drive, ref_id)
        if img:
            images_to_show.append(
                {
                    "image": img,
                    "ref_id": ref_id,
                    "title": title,
                    "description": description,
                }
            )
        else:
            failed_refs.append(ref_id)
            images_to_show.append(
                {
                    "image": None,
                    "ref_id": ref_id,
                    "title": title,
                    "description": description,
                }
    )

    # Format available references as text (for IDs + descriptions)
    refs_text = ""
    for ref_id, ref in available_references.items():
        refs_text += f"- [{ref_id}] {ref['title']}: {ref['description']}\n"

    base_prompt = f"""Evaluate if any of these existing reference images can be reused for the following voiceover sentence.

Voiceover Sentence: "{vo_text}"

You will be shown {len(images_to_show)} reference images below. Each image is associated with a Reference ID in the list:

Available References (text summary):
{refs_text}

Your job:
- Visually inspect the images.
- Use the text metadata (title + description) and the image content together.
- Decide which references are suitable to reuse for visualizing this voiceover sentence.

Output format:
Always strictly give your output in the following format:

<matches>
[List reference IDs that match, one per line, or "NONE" if no matches]
</matches>

<reasoning>
[Brief explanation of why each reference does or doesn't match]
</reasoning>"""

    # Prepare multimodal content parts: text prompt + all images
    content_parts: List[Dict[str, Any]] = []
    content_parts.append(
        {
            "type": "text",
            "text": base_prompt.strip(),
        }
    )

    # Add each reference with label + image (if available)
    for idx, item in enumerate(images_to_show, start=1):
        ref_id = item["ref_id"]
        title = item.get("title", "Untitled")
        description = item.get("description", "")

        label_text = f"\n--- Reference {idx} of {len(images_to_show)} ---\n"
        label_text += f"Reference ID: [{ref_id}]\n"
        label_text += f"Title: {title}\n"
        if description:
            label_text += f"Description: {description}\n"

        content_parts.append(
            {
                "type": "text",
                "text": label_text,
            }
        )

        if item["image"] is not None:
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": pil_to_base64_data_uri(item["image"]),
                }
            )
        else:
            # Image could not be loaded – still keep the reference in text, but explicitly note missing image.
            content_parts.append(
                {
                    "type": "text",
                    "text": f"[Image for {ref_id} could not be loaded; evaluate based on text only]\n",
                }
            )

    # Call a vision-capable model via llm_with_retry
    messages = [("user", content_parts)]
    raw_response = llm_with_retry(messages, llm_name="gemini_2_5_pro")

    # Normalize raw response to plain text
    if hasattr(raw_response, "content"):
        raw_text = raw_response.content
    elif isinstance(raw_response, dict):
        raw_text = raw_response.get("content") or raw_response.get("text", "")
    else:
        raw_text = str(raw_response)

    # Reuse Chain's tag-extraction helper for <matches> and <reasoning>
    parser_chain = Chain(llm="gemini_2_5_pro", tags=["matches", "reasoning"])
    try:
        parsed = parser_chain.extract_text_in_tags(raw_text)
    except Exception:
        # Fallback: treat whole text as reasoning, no matches
        parsed = {"matches": "NONE", "reasoning": raw_text}

    matches_text = parsed.get("matches", "NONE").strip()
    reasoning = parsed.get("reasoning", "").strip()

    if matches_text == "NONE" or not matches_text:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        "No reusable references found for this visual element",
                        tool_call_id=tool_call_id,
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

    # Get matched references from available_references
    matched_references: List[ReferenceData] = [
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
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    result_message = (
        f"Found {len(matched_references)} reusable reference(s) for VO sentence: '{vo_text[:80]}{'...' if len(vo_text) > 80 else ''}':\n"
    )
    for ref in matched_references:
        result_message += (
            f"- {ref['title']} (from segment {ref.get('reused_from_segment', '?')})\n"
        )
    result_message += f"\nReasoning: {reasoning}"
    
    print(f"\n         {'─'*45}")
    print(f"         ♻️  Tool: check_reference_reuse ◀ END")
    print(f"         {'─'*45}")
    print(f"            ✅ Reusable references found: {len(matched_references)}")
    for ref in matched_references:
        print(f"               - [{ref.get('reference_id', 'N/A')}] {ref.get('title', 'Untitled')}")
    if failed_refs:
        print(f"            ⚠️  Note: {len(failed_refs)} image(s) failed to load (evaluated by text only)")
    print(f"            📝 Reasoning: {reasoning}")
    print(f"         {'─'*45}\n")

    return Command(
        update={
            "selected_references": matched_references,  # reducer will merge
            "messages": [
                ToolMessage(
                    result_message,
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


# @tool
# def evaluate_search_results(
#     tool_call_id: Annotated[str, InjectedToolCallId],
#     state: Annotated[Dict, InjectedState]
# ) -> Command:
#     """
#     Evaluates the quality and coverage of current search results.

#     Checks:
#     - Do results match the visual elements needed?
#     - Are relevance scores acceptable?
#     - Is there sufficient coverage?
#     - Should searches be refined?

#     Returns:
#         Assessment and recommendations (FINALIZE vs CONTINUE_SEARCH)
#     """
#     graphics_definition = state.get("graphics_definition", "")
#     visual_elements = state.get("visual_elements", [])
#     search_results = state.get("search_results", [])
#     selected_references = state.get("selected_references", [])
#     min_needed = state.get("min_references_needed", len(visual_elements))

#     total_references = len(search_results) + len(selected_references)
    
#     print(f"\n         {'─'*45}")
#     print(f"         📊 Tool: evaluate_search_results ▶ START")
#     print(f"         {'─'*45}")
#     print(f"            📝 Graphics definition: {len(graphics_definition)} chars")
#     print(f"            🎯 Visual elements: {len(visual_elements)}")
#     print(f"            🔍 Search results: {len(search_results)}")
#     print(f"            ♻️  Already selected (reused): {len(selected_references)}")
#     print(f"            📊 Total references: {total_references} (need: {min_needed})")

#     base_prompt = f"""You are evaluating image search results for an educational graphics definition.

# IMPORTANT CONSTRAINTS:
# - The vector database may NOT contain perfect or exact matches.
# - Your goal is to decide if the CURRENT results are **good enough to use**, not to chase perfection.
# - Prefer finalizing when results are reasonably relevant, even if not perfect.

# Use these realistic rules:
# - If there is **at least 1–{max(min_needed, 2)} reasonably relevant reference** (relevance score ≥ 0.3),
#   and it roughly matches the needed visual elements, that is **sufficient to finalize**.
# - Only recommend continuing search / refining queries when:
#   * Total references = 0, OR
#   * Nearly all results are clearly irrelevant to the graphics definition.

# Graphics Definition:
# {graphics_definition}

# Visual Elements Needed ({len(visual_elements)}):
# {', '.join(visual_elements)}

# Current Status:
# - Search results found: {len(search_results)}
# - Reused references: {len(selected_references)}
# - Total references: {total_references}
# - Minimum needed: {min_needed}

# You will be shown the top {min(len(search_results), 10)} reference images.
# Each image is labeled with its Reference ID, title, relevance score, and description.

# For quick reference, here is a text summary of the top results:
# {format_results_for_evaluation(search_results[:10])}

# TASK:
# 1. Visually inspect each image to judge whether it matches the graphics definition.
# 2. Judge whether the current results are **good enough to use** given the constraints above.
# 3. Decide whether to:
#    - **FINALIZE** with the current best references, or
#    - **CONTINUE_SEARCH** (only if results are clearly unusable or empty).
# 4. Briefly explain your reasoning, focusing on relevance and coverage at a practical level. When reasoning, refer to images by their Reference ID.

# Output format:
# <assessment>
# [Overall quality assessment: EXCELLENT/GOOD/FAIR/POOR]
# </assessment>

# <recommendations>
# One of:
# - FINALIZE – results are good enough to use (this should be the **default** when in doubt)
# - CONTINUE_SEARCH – results are clearly unusable or empty
# You may add a short explanation after the keyword.
# </recommendations>
# """

#     content_parts: List[Dict[str, Any]] = []

#     # Add main prompt text
#     content_parts.append(
#         {
#             "type": "text",
#             "text": base_prompt.strip(),
#         }
#     )

#     # Get drive instance for loading images
#     drive = state.get("drive")
    
#     # Add each reference with its label + image (if URL available)
#     top_results = search_results[:10]
#     for idx, ref in enumerate(top_results, start=1):
#         ref_id = ref.get("reference_id", f"ref_{idx}")
#         title = ref.get("title", "Untitled")
#         description = ref.get("description", "")
#         score = ref.get("relevance_score", 0.0)
#         url = ref.get("url", "")

#         # Label text BEFORE the image so the LLM can associate them
#         label_text = f"\n--- Reference {idx} of {len(top_results)} ---\n"
#         label_text += f"Reference ID: [{ref_id}]\n"
#         label_text += f"Title: {title}\n"
#         label_text += f"Relevance Score: {score:.2f}\n"
#         if description:
#             label_text += f"Description: {description}\n"

#         content_parts.append(
#             {
#                 "type": "text",
#                 "text": label_text,
#             }
#         )

#         # Try to load and attach image using Drive API
#         if url:
#             img = load_image_from_drive_url(url, drive, ref_id)
#             if img:
#                 content_parts.append(
#                     {
#                         "type": "image_url",
#                         "image_url": pil_to_base64_data_uri(img),
#                     }
#                 )
#             else:
#                 content_parts.append(
#                     {
#                         "type": "text",
#                         "text": f"[Image for {ref_id} could not be loaded; evaluate based on text only]\n",
#                     }
#                 )
#         else:
#             content_parts.append(
#                 {
#                     "type": "text",
#                     "text": f"[No URL available for {ref_id}; evaluate based on text only]\n",
#                 }
#             )

#     messages = [("user", content_parts)]
#     raw_response = llm_with_retry(messages, llm_name="gemini_2_5_pro")

#     # Normalize raw response to plain text
#     if hasattr(raw_response, "content"):
#         raw_text = raw_response.content
#     elif isinstance(raw_response, dict):
#         raw_text = raw_response.get("content") or raw_response.get("text", "")
#     else:
#         raw_text = str(raw_response)

#     # Reuse Chain's tag-extraction helper for <assessment> and <recommendations>
#     parser_chain = Chain(llm="gemini_2_5_pro", tags=["assessment", "recommendations"])
#     try:
#         parsed = parser_chain.extract_text_in_tags(raw_text)
#     except Exception:
#         # Fallback: treat whole text as recommendations, default assessment
#         parsed = {"assessment": "FAIR", "recommendations": raw_text}

#     assessment = parsed.get("assessment", "FAIR")
#     recommendations = parsed.get("recommendations", "Continue searching")
#     print(f"\n         {'─'*45}")
#     print(f"         📊 Tool: evaluate_search_results ◀ END")
#     print(f"         {'─'*45}")
#     print(f"            📈 Assessment: {assessment}")
#     print(f"            🎯 Recommendation: {recommendations}")
#     print(f"            📊 Avg relevance score: {calculate_avg_relevance(search_results):.2f}")
#     print(f"         {'─'*45}\n")

#     result_message = f"""
# SEARCH EVALUATION:
# - Assessment: {assessment}
# - Total References: {total_references} / {min_needed} needed
# - Search Results Quality: {calculate_avg_relevance(search_results):.2f} avg relevance

# Recommendations:
# {recommendations}
# """

#     return Command(
#         update={
#             "messages": [
#                 ToolMessage(
#                     result_message,
#                     tool_call_id=tool_call_id
#                 )
#             ]
#         }
#     )


# @tool
# def finalize_search(
#     selected_reference_ids: str,
#     tool_call_id: Annotated[str, InjectedToolCallId],
#     state: Annotated[Dict, InjectedState]
# ) -> Command:
#     """
#     Finalizes the search by selecting the best references to return.

#     Args:
#         selected_reference_ids: Comma-separated IDs of references to include

#     Returns:
#         Summary of selected references
#     """
#     # Parse IDs
#     ids = [id.strip() for id in selected_reference_ids.split(",") if id.strip()]
    
#     print(f"\n         {'─'*45}")
#     print(f"         🏁 Tool: finalize_search ▶ START")
#     print(f"         {'─'*45}")
#     print(f"            🎯 Reference IDs to select: {len(ids)}")
#     for i, ref_id in enumerate(ids, 1):
#         print(f"               {i}. {ref_id}")

#     search_results = state.get("search_results", [])
#     already_selected = state.get("selected_references", [])
#     already_selected_ids = {ref["reference_id"] for ref in already_selected}

#     # Find references by ID that aren't already selected
#     newly_selected = []
#     for ref in search_results:
#         if ref["reference_id"] in ids and ref["reference_id"] not in already_selected_ids:
#             newly_selected.append(ref)

#     # Update state
#     total_selected = len(already_selected) + len(newly_selected)
    
#     print(f"\n         {'─'*45}")
#     print(f"         🏁 Tool: finalize_search ◀ END")
#     print(f"         {'─'*45}")
#     print(f"            ✅ Total selected: {total_selected}")
#     print(f"            📎 Previously selected: {len(already_selected)}")
#     print(f"            ➕ Newly added: {len(newly_selected)}")
#     print(f"            📋 Final references:")
#     all_selected = already_selected + newly_selected
#     for i, ref in enumerate(all_selected, 1):
#         print(f"               {i}. [{ref.get('reference_id', 'N/A')}] {ref.get('title', 'Untitled')} | score: {ref.get('relevance_score', 0):.2f}")
#     print(f"         {'─'*45}\n")
    
#     result_message = f"Search finalized with {total_selected} selected references ({len(newly_selected)} newly added):\n"
#     for i, ref in enumerate(already_selected + newly_selected):
#         result_message += f"{i+1}. {ref['title']} ({ref['type']}) - Relevance: {ref['relevance_score']:.2f}\n"

#     return Command(
#         update={
#             "selected_references": newly_selected,  # Just return new selections, reducer will merge
#             "messages": [
#                 ToolMessage(
#                     result_message,
#                     tool_call_id=tool_call_id
#                 )
#             ]
#         }
#     )


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



#NEW SEARCH QUERY GENERATOR PROMPT WE WILL BE USING

prompt = """You are a Search Query Generator agent specializing in the field of HVAC. Your task is to generate concise, high-quality image search queries that can be used to retrieve relevant images from a vector store of image embeddings and web-based image search engines. The retrieved images will then be used as on-screen visuals in an educational e-learning slide while the given voiceover sentence is being narrated.

You will be given a single voiceover sentence from an educational slide, along with its course information and the whole slide content for context. First, reason internally about what visual elements would need to be shown on screen for the voiceover sentence to be clearly understood. Then, based on that reasoning, generate {SEARCH_QUERIES_PER_ELEMENT} image search queries that would retrieve the most relevant visuals.
 
These are the inputs:

Course Name: {course_name}

Topic Name: {topic_name}

Subtopic Name: {subtopic_name}

Voiceover sentence for which you need to generate search queries:
"{vo_text}"

Full slide content:
{slide_chunk}

Instructions:

1. Base all search queries strictly on the meaning of the voiceover sentence.
   - Do not generate queries based on general topic knowledge alone.
   - The queries should help retrieve images that are visually necessary to understand this specific sentence.

2. Generate concise, image-focused search queries.
   - Use concrete object names, components, or diagrams that are likely to appear in images.
   - Avoid abstract, instructional, or process-oriented wording.

3. Phrase queries the way images are commonly searched for or labeled.
   - Use short, keyword-based phrases.
   - Prefer phrasing typical of textbooks, technical diagrams, stock photos, or browser image searches.

4. Ensure each query adds value.
   - Queries should not be near-duplicates of each other.
   - Each query should represent a slightly different but relevant visual angle.

Always provide your output strictly in the following format:

<output>

<evaluation_breakdown>

1. Voiceover Meaning
- Briefly explain, in your own words, what the voiceover sentence is communicating.

2. Query Planning
- Reason about the kinds of image searches that would best retrieve visuals to support this sentence.
- Consider how such images are typically searched for or labeled.

</evaluation_breakdown>

(Based on your above evaluation, provide your final output of search queries)

<queries>
- Query 1 text
- Query 2 text
- Query 3 text
</queries>

</output>
"""