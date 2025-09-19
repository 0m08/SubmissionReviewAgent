from typing import Annotated, Union
import pandas as pd
import re
import threading
import itertools
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
) -> str:
    """Insert a new text block *after* the specified block ID.

    Args:
        insert_after_block_id:  
            • ID of the block after which the new block should be placed.  
            • `None` (or -1) inserts at the very beginning.
        block_text: The text content of the new block.

    Returns:
        A confirmation string such as `"✅ Created block 7."`

    Notes:
        * The original block IDs (DataFrame index) **never change**; a brand-new
          integer ID is allocated (`max(index)+1`).
    """
    print(f"🔧 TOOL USED: create_block | insert_after_block_id: {insert_after_block_id} | block_text_length: {len(block_text)} chars")

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
) -> str:
    """Read the text of a contiguous range of blocks (inclusive).

    Args:
        start_block_id: First block ID in the range.
        end_block_id:   Last block ID in the range.

    Returns:
        The concatenated block texts, separated by newlines, in display order.

    Raises:
        IndexError if either ID is absent.
    """
    print(f"🔧 TOOL USED: read_blocks | start_block_id: {start_block_id} | end_block_id: {end_block_id}")
    
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
) -> str:
    """Replace the text of an existing block.

    Args:
        block_id:   ID of the block to modify.
        block_text: New text content.

    Returns:
        `"✏️ Updated block {block_id}."`

    Raises:
        IndexError if `block_id` is absent.
    """
    print(f"🔧 TOOL USED: update_block | block_id: {block_id} | block_text_length: {len(block_text)} chars")
    
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
) -> str:
    """Delete a block by ID (leaves a gap in IDs).

    Args:
        block_id: ID of the block to remove.

    Returns:
        `"🗑️ Deleted block {block_id}."`

    Raises:
        IndexError if `block_id` is absent.
    """
    print(f"🔧 TOOL USED: delete_block | block_id: {block_id}")
    
    if block_id not in df.index:
        raise IndexError("block_id not found")
    
    # Delete from current DataFrame slice
    df.drop(block_id, inplace=True)
    return f"🗑️ Deleted block {block_id}."