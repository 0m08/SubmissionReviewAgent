import streamlit as st
import pandas as pd
from services.sheets_service import get_sheet_data_and_df, safe_get_sheet_data_and_df
from services.helper_functions import compare_text_versions


def build_slide_chunks_string(df: pd.DataFrame) -> str:
    """
    Build a concatenated string from Topic, Subtopic, slide_type, slide_chunk_title and slide_chunk.

    Args:
        df: DataFrame containing the slide chunks data

    Returns:
        Concatenated string of all rows
    """
    lines = []
    for index, row in df.iterrows():
        topic = str(row.get('Topic', '')) if pd.notna(row.get('Topic', '')) else ''
        subtopic = str(row.get('Subtopic', '')) if pd.notna(row.get('Subtopic', '')) else ''
        slide_type = str(row.get('Slide Type', '')) if pd.notna(row.get('Slide Type', '')) else ''
        slide_chunk_title = str(row.get('Slide Chunk Title', '')) if pd.notna(row.get('Slide Chunk Title', '')) else ''
        slide_chunk = str(row.get('Slide Chunk', '')) if pd.notna(row.get('Slide Chunk', '')) else ''

        row_str = f"Topic: {topic}\nSubtopic: {subtopic}\nSlide Type: {slide_type}\nTitle: {slide_chunk_title}\nContent:\n{slide_chunk}\n---"
        lines.append(row_str)

    return "\n".join(lines)


def show_slide_chunks_diff(sheet):
    """
    Shows the diff between current slide chunks (Slide Chunks tab)
    and previous slide chunks (Backup Slide Chunks Sheet for Delete step of Slide Chunks Checklist).

    Constructs input strings by concatenating topic, subtopic, slide_type, slide_chunk_title and slide_chunk
    row by row, then shows the diff using compare_text_versions.

    This function is designed to be used as a pre_exec_func in a manual step.

    Args:
        sheet: The Google Sheet object
    """
    slide_chunks_sheet_name = "Slide Chunks"
    backup_sheet_name = "Backup Slide Chunks Sheet for Delete step of Slide Chunks Checklist"

    # Read current slide chunks
    try:
        _, current_df = get_sheet_data_and_df(sheet=sheet, sheet_name=slide_chunks_sheet_name)
        print(f"✅ Loaded {len(current_df)} rows from '{slide_chunks_sheet_name}'")
    except Exception as e:
        st.error(f"❌ Failed to read '{slide_chunks_sheet_name}' sheet: {e}")
        return

    # Read backup slide chunks
    try:
        _, backup_df = safe_get_sheet_data_and_df(sheet=sheet, sheet_name=backup_sheet_name)
        if backup_df.empty:
            st.warning(f"⚠️ Backup sheet '{backup_sheet_name}' is empty or does not exist. "
                      "This step requires the 'Slide Chunks Checklist Review and Revise' step to have been run at least once.")
            return
        print(f"✅ Loaded {len(backup_df)} rows from backup sheet")
    except Exception as e:
        st.error(f"❌ Failed to read backup sheet: {e}")
        st.info("ℹ️ The backup sheet is created when running the 'Slide Chunks Checklist Review and Revise' step.")
        return

    # Check if slide_chunk column exists in both dataframes
    if 'Slide Chunk' not in current_df.columns:
        st.error("❌ 'Slide Chunk' column not found in Slide Chunks sheet.")
        return

    if 'Slide Chunk' not in backup_df.columns:
        st.error("❌ 'Slide Chunk' column not found in backup sheet.")
        return

    # Build concatenated strings for comparison
    current_string = build_slide_chunks_string(current_df)
    backup_string = build_slide_chunks_string(backup_df)

    # Check if there are any differences
    if current_string == backup_string:
        st.success("✅ No differences found between current and backup slide chunks.")
        return

    # Display the diff
    st.write("### Slide Chunks Diff: Comparing Before and After Checklist Review")
    st.caption("Red = Removed from backup (previous version), Green = Added in current version")

    compare_text_versions(
        text1=backup_string,
        text2=current_string,
        version1_name="Previous (Before Checklist)",
        version2_name="Current (After Checklist)"
    )

    st.info("""
    **Review the changes made by the Slide Chunks Checklist Review and Revise Agents.**
    - Red highlighted text shows content that was removed or changed.
    - Green highlighted text shows content that was added or modified.
    - Click the button below to confirm you've reviewed the diff.
    """)

    return


def manual_review_slide_chunks_diff(sheet, **kwargs):
    """
    Manual step function for reviewing slide chunks diff.
    This is a placeholder function that does nothing - the actual diff is shown via pre_exec_func.

    Args:
        sheet: The Google Sheet object
        **kwargs: Additional keyword arguments (unused but accepted for pipeline compatibility)
    """
    pass


def delete_slide_chunks_diff(**kwargs):
    """
    Placeholder delete function for the diff visualization step.
    This step doesn't modify any data, so there's nothing to delete.
    """
    pass
