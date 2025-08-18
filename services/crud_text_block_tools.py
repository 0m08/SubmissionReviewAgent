from typing import Annotated, Union
import pandas as pd

# If you're using LangChain‑v0.2+, the two imports below are the same:
from langchain.tools import tool
from langgraph.prebuilt import InjectedState

# ─────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS  (not exposed as tools)
# ─────────────────────────────────────────────────────────────────────────
def _ensure_order_column(df: pd.DataFrame) -> None:
    """Create an 'order' column if missing; values are numeric."""
    if "order" not in df.columns:
        df["order"] = df.index.astype(float)


def _recompute_dense_order(df: pd.DataFrame) -> None:
    """Compact the order column to 0,1,2,… **without touching the index**."""
    df.sort_values("order", inplace=True)
    df["order"] = range(len(df))


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
    
    # _ensure_order_column(df)

    # Normalize “prepend” sentinel
    if insert_after_block_id in (-1, None):
        insert_after_block_id = None
    elif insert_after_block_id not in df.index:
        raise IndexError("insert_after_block_id not found")

    # Allocate fresh block ID
    new_id: int = (df.index.max() + 1) if len(df) else 0

    # Compute ordering key
    if insert_after_block_id is None:                    # prepend
        first_order = df["order"].min() if len(df) else 0
        new_order = first_order - 1
    else:                                                # insert after X
        prev_order = df.at[insert_after_block_id, "order"]
        following = df[df["order"] > prev_order]["order"].min()
        new_order = (prev_order + following) / 2 if pd.notna(following) else prev_order + 1

    # Add new row in‑place
    df.loc[new_id, ["block text", "order"]] = [block_text, new_order]

    # Re‑densify if gaps shrink too much
    if (df["order"].diff().abs().min() < 1e-9):
        _recompute_dense_order(df)

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
    df.drop(block_id, inplace=True)
    return f"🗑️ Deleted block {block_id}."
