from typing import Annotated, Union
import pandas as pd
import re
import threading
import itertools
from google import genai
from google.genai import types
# ─────────────────────────────────────────────────────────────────────────
# TOOL 1  ▸  create_block
# ─────────────────────────────────────────────────────────────────────────
# Global atomic counter for block IDs
_block_id_lock = threading.Lock()
_global_block_id_counter = None

# If you're using LangChain‑v0.2+, the two imports below are the same:
from langchain.tools import tool
from langgraph.prebuilt import InjectedState

# ─────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS  (not exposed as tools)
# ─────────────────────────────────────────────────────────────────────────
def _ensure_order_column(df: pd.DataFrame) -> None:
    """Create an 'order' column if missing; values are numeric starting from 1."""
    if "order" not in df.columns:
        df["order"] = df.index.astype(float) + 1  # Start from 1, not 0


def _recompute_dense_order(df: pd.DataFrame) -> None:
    """Compact the order column to 1,2,3,… **without touching the index**."""
    df.sort_values("order", inplace=True)
    df["order"] = range(1, len(df) + 1)  # Start from 1, not 0


# ─────────────────────────────────────────────────────────────────────────
# TOOL 1  ▸  create_block
# ─────────────────────────────────────────────────────────────────────────
@tool("create_block", parse_docstring=True)
def create_block(
    df: Annotated[pd.DataFrame, InjectedState("df")],
    insert_after_block_id: Union[int, None],
    block_text: str,
    intent: str,
) -> str:
    """Insert a new text block *after* the specified block ID.

    Args:
        insert_after_block_id:
            • ID of the block after which the new block should be placed.
            • `None` (or -1) inserts at the very beginning.
        block_text: The text content of the new block.
        intent: A field that describes the intent of changes the LLM is trying to make.

    Returns:
        A confirmation string such as `"✅ Created block 7."`

    Notes:
        * The original block IDs (DataFrame index) **never change**; a brand-new
          integer ID is allocated (`max(index)+1`).
    """
    print(
        "🔧 TOOL USED: create_block | insert_after_block_id: "
        f"{insert_after_block_id} | block_text_length: {len(block_text)} chars | "
        f"intent: {intent}"
    )

    # Only check for duplicates in research notes agent (check if 'research_notes' column exists)
    if 'research_notes' in df.columns:
        # Try to prevent logical duplicates by converting duplicate creates into updates
        # We identify duplicates by matching Topic + Subtopic + Learning Objectives parsed from block_text
        def _extract_section(text: str, header: str) -> str:
            pattern = rf"####\*\*{re.escape(header)}:\*\*\n([\s\S]*?)(?=\n####\*\*|\Z)"
            m = re.search(pattern, text, re.MULTILINE)
            return m.group(1).strip() if m else ""

        parsed_topic = _extract_section(block_text, "Topic")
        parsed_subtopic = _extract_section(block_text, "Subtopic")
        parsed_lo = _extract_section(block_text, "Learning Objective")

        if all([parsed_topic, parsed_subtopic, parsed_lo]) and all(col in df.columns for col in ["Topic", "Subtopic", "Learning Objectives"]):
            duplicate_matches = df[(df["Topic"].astype(str) == parsed_topic) &
                                   (df["Subtopic"].astype(str) == parsed_subtopic) &
                                   (df["Learning Objectives"].astype(str) == parsed_lo)]
            if len(duplicate_matches) >= 1:
                # Update the first match instead of creating a new block
                target_id = int(duplicate_matches.index[0])
                df.at[target_id, "block text"] = block_text
                msg = f"✏️ Updated existing block {target_id} (matched by Topic/Subtopic/LO)."
                print(msg)
                return msg

    # Normalize “prepend” sentinel
    if insert_after_block_id in (-1, None):
        insert_after_block_id = None
    elif insert_after_block_id not in df.index:
        raise IndexError("insert_after_block_id not found")

    # Thread-safe, process-wide block ID allocation and order assignment
    global _global_block_id_counter
    with _block_id_lock:
        if _global_block_id_counter is None:
            # Use global_max_index from DataFrame attrs if available
            if hasattr(df, 'attrs') and 'global_max_index' in df.attrs:
                start_id = int(df.attrs['global_max_index']) + 1
            else:
                # Try to get the parent DataFrame if df is a slice
                full_df = getattr(df, '_parent', None)
                if full_df is not None and isinstance(full_df, pd.DataFrame):
                    start_id = int(full_df.index.max()) + 1
                else:
                    start_id = int(df.index.max()) + 1 if len(df) > 0 else 0
            _global_block_id_counter = itertools.count(start_id)
        new_id = next(_global_block_id_counter)
        print(f"Allocated Block ID: {new_id}")

        # Compute ordering key with full DataFrame context awareness
        # Get the full DataFrame to check for existing order values
        full_df = getattr(df, '_parent', None)
        if full_df is not None and isinstance(full_df, pd.DataFrame) and 'order' in full_df.columns:
            context_df = full_df  # Use full DataFrame for order conflict checking
        else:
            context_df = df  # Fallback to current slice if no parent available
            
        if insert_after_block_id is None:  # prepend
            first_order = df["order"].min() if len(df) else 1
            proposed_order = first_order - 1
            
            # Handle the first block case and ensure orders start from 1
            if proposed_order < 1:
                # This is the case where we're adding before the very first slide
                new_order = (first_order + proposed_order) / 2
                # But ensure it's positive and greater than 0
                if new_order <= 0:
                    new_order = first_order / 2 
                print(f"First block case: first_order={first_order}, proposed_order={proposed_order}, using new_order={new_order}")
            else:
                # Check if proposed order exists in full context
                if proposed_order in context_df["order"].values:
                    new_order = (first_order + proposed_order) / 2
                    print(f"Order conflict detected: {proposed_order} exists in full DF. Using midpoint: {new_order}")
                else:
                    new_order = proposed_order
                
        else:  # insert after X
            prev_order = df.at[insert_after_block_id, "order"]
            
            # Find the next order value in the FULL context, not just the slice
            next_orders_in_context = context_df[context_df["order"] > prev_order]["order"]
            
            if not next_orders_in_context.empty:
                following = next_orders_in_context.min()
                proposed_order = (prev_order + following) / 2
                
                # Check if proposed order exists in full context
                if proposed_order in context_df["order"].values:
                    # Find an even smaller gap
                    gap = (following - prev_order) / 2
                    new_order = prev_order + gap
                    
                    # If still conflicts, make progressively smaller gaps
                    iteration = 0
                    while new_order in context_df["order"].values and iteration < 10:
                        gap = gap / 2
                        new_order = prev_order + gap
                        iteration += 1
                        
                    print(f"Order conflict resolved after {iteration} iterations: {new_order}")
                else:
                    new_order = proposed_order
            else:
                # Appending at the end - find next safe order after prev_order
                proposed_order = prev_order + 1
                
                # Check if proposed order exists in full context
                while proposed_order in context_df["order"].values:
                    proposed_order += 1  # Keep incrementing until we find a free slot
                
                new_order = proposed_order
                    
        print(f"Assigned order value: {new_order} (using full DataFrame context)")
        
        # Add new row in‑place
        df.loc[new_id, ["block text", "order"]] = [block_text, new_order]

        # Re‑densify if gaps shrink too much (simple check)
        if len(df) > 1 and (df["order"].diff().abs().min() < 1e-9):
            _recompute_dense_order(df)

        # Sort DataFrame by order column to maintain proper ordering after block creation
        if "order" in df.columns and len(df) > 1:
            df.sort_values("order", inplace=True)
            print(f"📋 Sorted DataFrame by 'order' column after creating block {new_id}")

    return f"✅ Created block {new_id}."


# ─────────────────────────────────────────────────────────────────────────
# TOOL 2  ▸  read_blocks
# ─────────────────────────────────────────────────────────────────────────
@tool("read_blocks", parse_docstring=True)
def read_blocks(
    df: Annotated[pd.DataFrame, InjectedState("df")],
    start_block_id: int,
    end_block_id: int,
    intent: str,
) -> str:
    """Read the text of a contiguous range of blocks (inclusive).

    Args:
        start_block_id: First block ID in the range.
        end_block_id:   Last block ID in the range.
        intent: A field that describes the intent of changes the LLM is trying to make.

    Returns:
        The concatenated block texts, separated by newlines, in display order.

    Raises:
        IndexError if either ID is absent.
    """
    print(
        "🔧 TOOL USED: read_blocks | start_block_id: "
        f"{start_block_id} | end_block_id: {end_block_id} | intent: {intent}"
    )
    
    if start_block_id not in df.index or end_block_id not in df.index:
        raise IndexError("start or end block_id not found")

    # Get the numeric order boundaries
    lo = df.at[start_block_id, "order"]
    hi = df.at[end_block_id,   "order"]
    if lo > hi:                   # allow reversed arguments
        lo, hi = hi, lo

    subset = df[(df["order"] >= lo) & (df["order"] <= hi)]          \
              .sort_values("order")["block text"]

    return "\n".join(subset.tolist())


# ─────────────────────────────────────────────────────────────────────────
# TOOL 3  ▸  update_block
# ─────────────────────────────────────────────────────────────────────────
@tool("update_block", parse_docstring=True)
def update_block(
    df: Annotated[pd.DataFrame, InjectedState("df")],
    block_id: int,
    block_text: str,
    intent: str,
) -> str:
    """Replace the text of an existing block.

    Args:
        block_id:   ID of the block to modify.
        block_text: New text content.
        intent: A field that describes the intent of changes the LLM is trying to make.

    Returns:
        `"✏️ Updated block {block_id}."`

    Raises:
        IndexError if `block_id` is absent.
    """
    print(
        "🔧 TOOL USED: update_block | block_id: "
        f"{block_id} | block_text_length: {len(block_text)} chars | intent: {intent}"
    )
    
    if block_id not in df.index:
        raise IndexError("block_id not found")
    df.at[block_id, "block text"] = block_text
    return f"✏️ Updated block {block_id}."


# ─────────────────────────────────────────────────────────────────────────
# TOOL 4  ▸  delete_block
# ─────────────────────────────────────────────────────────────────────────
@tool("delete_block", parse_docstring=True)
def delete_block(
    df: Annotated[pd.DataFrame, InjectedState("df")],
    block_id: int,
    intent: str,
) -> str:
    """Delete a block by ID (leaves a gap in IDs).

    Args:
        block_id: ID of the block to remove.
        intent: A field that describes the intent of changes the LLM is trying to make.

    Returns:
        `"🗑️ Deleted block {block_id}."`

    Raises:
        IndexError if `block_id` is absent.
    """
    print(
        "🔧 TOOL USED: delete_block | block_id: "
        f"{block_id} | intent: {intent}"
    )
    
    if block_id not in df.index:
        raise IndexError("block_id not found")

    # Delete from current DataFrame slice
    df.drop(block_id, inplace=True)
    return f"🗑️ Deleted block {block_id}."


# ─────────────────────────────────────────────────────────────────────────
# TOOL 5  ▸  str_replace
# ─────────────────────────────────────────────────────────────────────────
@tool("str_replace", parse_docstring=True)
def str_replace(
    df: Annotated[pd.DataFrame, InjectedState("df")],
    block_id: int,
    old_text: str,
    new_text: str,
    intent: str,
) -> str:
    """Replace a specific substring within a block's text with precision.

    This tool performs exact string matching and replacement. The old_text must
    match the block content exactly, including all whitespace and indentation.

    Args:
        block_id:  ID of the block to modify.
        old_text:  The exact text to find and replace. Must match the block
                   content exactly, including whitespace and indentation.
                   Must uniquely identify the target location (appear exactly once).
        new_text:  The text to replace old_text with. Must be different from old_text.
        intent:    A field that describes the intent of the change.

    Returns:
        A confirmation message: `"✏️ Replaced text in block {block_id}."`

    Raises:
        IndexError: If block_id does not exist.
        ValueError: If old_text is not found in the block (0 matches).
        ValueError: If old_text equals new_text (no-op replacement).
        ValueError: If old_text appears multiple times (ambiguous match).
    """
    print(
        f"🔧 TOOL USED: str_replace | block_id: {block_id} | "
        f"old_text_length: {len(old_text)} chars | new_text_length: {len(new_text)} chars | "
        f"intent: {intent}"
    )

    # Validate block exists
    if block_id not in df.index:
        raise IndexError(f"block_id {block_id} not found")

    # Validate old_text != new_text (prevent no-op)
    if old_text == new_text:
        raise ValueError("old_text and new_text are identical. No replacement needed.")

    current_text = df.at[block_id, "block text"]

    # Count occurrences
    match_count = current_text.count(old_text)

    # Handle zero matches
    if match_count == 0:
        raise ValueError(
            f"old_text not found in block {block_id}. "
            "Ensure the text matches exactly, including whitespace and indentation."
        )

    # Handle multiple matches (ambiguous)
    if match_count > 1:
        raise ValueError(
            f"old_text appears {match_count} times in block {block_id}. "
            "Include more surrounding context to uniquely identify the target location."
        )

    # Perform replacement (exactly one match)
    updated_text = current_text.replace(old_text, new_text, 1)
    df.at[block_id, "block text"] = updated_text

    return f"✏️ Replaced text in block {block_id}."


# ─────────────────────────────────────────────────────────────────────────
# TOOL 6  ▸  search_web
# ─────────────────────────────────────────────────────────────────────────
@tool("search_web", parse_docstring=True)
def search_web(
    query: str,
    intent: str,
) -> str:
    """Search the web using Google Search with Gemini grounding to find information.

    Args:
        query: The search query to find information about.
        intent: A field that describes the intent of the search.

    Returns:
        The answer to the search query based on web search results.
    """
    print(
        f"🔧 TOOL USED: search_web | query: {query} | intent: {intent}"
    )

    # from dotenv import load_dotenv
    # load_dotenv()  # Load environment variables from .env file

    from google import genai
    from google.genai import types

    client = genai.Client()

    grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )
    url_context_tool = types.Tool(
        url_context=types.UrlContext()
    )

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True
        ),
        tools=[grounding_tool, url_context_tool]
    )

    # Construct a focused prompt to ensure concise, relevant output
    focused_prompt = f"""Search Query: {query}
Intent: {intent}

Instructions: Based on the search results, provide a concise and specific answer that directly addresses the query and intent above.
- Focus only on information relevant to the query
- Avoid unnecessary details, tangents, or general background information
- Be factual and precise
- Keep the response brief and actionable"""


    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=focused_prompt,
        config=config,
    )

    # print("Raw response from Gemini:")
    # print(response)

    # Extract the text response
    if response and response.text:
        return f"🔍 Search results for '{query}':\n\n{response.text}"
    else:
        return f"⚠️ No results found for query: {query}"


# ─────────────────────────────────────────────────────────────────────────
# TOOL 7  ▸  preview_image
# ─────────────────────────────────────────────────────────────────────────
@tool("preview_image", parse_docstring=True)
def preview_image(
    image_url: str,
    intent: str,
):
    """Load an image from a URL so it can be seen in the next turn.

    Args:
        image_url: The URL of the image to preview. Supports regular web URLs
            and Google Drive file links.
        intent: A field that describes why the image is being previewed
            (e.g. "check whether this diagram matches paragraph 2").

    Returns:
        A multimodal content list with a short text part and an inline
        base64-encoded JPEG image part for the model to view. On failure,
        returns a plain string warning so the agent can skip gracefully.
    """
    print(
        f"🔧 TOOL USED: preview_image | url: {image_url} | intent: {intent}"
    )

    try:
        from agents.graphics_definition_v2.layout_agent.layout_agent import (
            load_image_from_url,
            get_drive_instance,
        )
    except Exception as e:
        return f"⚠️ Could not load image from {image_url}: loader import failed ({e})"

    drive = get_drive_instance()

    try:
        pil_image = load_image_from_url(image_url, drive, title="preview_image")
    except Exception as e:
        return f"⚠️ Could not load image from {image_url}: {e}"

    if pil_image is None:
        return f"⚠️ Could not load image from {image_url}: loader returned None (check that the file is shared to the service account or is publicly accessible)"

    try:
        from io import BytesIO
        import base64
        buffered = BytesIO()
        pil_image.convert("RGB").save(buffered, format="JPEG")
        b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        return f"⚠️ Could not encode image from {image_url}: {e}"

    return [
        {"type": "text", "text": f"Loaded image from {image_url}"},
        {
            "type": "image",
            "source_type": "base64",
            "mime_type": "image/jpeg",
            "data": b64,
        },
    ]


# ─────────────────────────────────────────────────────────────────────────
# TOOL 8  ▸  stop
# ─────────────────────────────────────────────────────────────────────────
@tool("stop", parse_docstring=True, return_direct=True)
def stop(
    reason: str,
) -> str:
    """Signal that all edits have been completed and the agent should stop.

    Call this tool when you have finished making all necessary revisions
    to the blocks. This explicitly indicates that no more CRUD operations
    are needed.

    Args:
        reason: A brief explanation of what was accomplished or why no further edits are needed.

    Returns:
        A confirmation message indicating the agent has completed its work.
    """
    print(
        f"🛑 TOOL USED: stop | reason: {reason}"
    )
    return f"✅ Agent completed. Reason: {reason}"


# result = search_web("What is the saturation pressure of R410a at 100 F? What is the temp of R22 at that same pressure?", "Finding refrigerant specific details")
# print(result)