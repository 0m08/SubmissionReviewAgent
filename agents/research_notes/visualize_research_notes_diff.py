import streamlit as st
import pandas as pd
from services.sheets_service import get_sheet_data_and_df, safe_get_sheet_data_and_df
from services.helper_functions import compare_text_versions


def build_research_notes_string(df: pd.DataFrame) -> str:
    """
    Build a concatenated string from Topic, Subtopic, Learning Objectives and research_notes.

    Args:
        df: DataFrame containing the research notes data

    Returns:
        Concatenated string of all rows
    """
    lines = []
    for index, row in df.iterrows():
        topic = str(row.get('Topic', '')) if pd.notna(row.get('Topic', '')) else ''
        subtopic = str(row.get('Subtopic', '')) if pd.notna(row.get('Subtopic', '')) else ''
        learning_objective = str(row.get('Learning Objectives', '')) if pd.notna(row.get('Learning Objectives', '')) else ''
        research_notes = str(row.get('research_notes', '')) if pd.notna(row.get('research_notes', '')) else ''

        row_str = f"Topic: {topic}\nSubtopic: {subtopic}\nLearning Objective: {learning_objective}\nResearch Notes:\n{research_notes}\n---"
        lines.append(row_str)

    return "\n".join(lines)


def show_research_notes_diff(sheet):
    """
    Shows the diff between current research notes (Final Outline tab)
    and previous research notes (Backup Final Outline Sheet for Delete step of Research Notes Checklist).

    Constructs input strings by concatenating topic, subtopic, learning objective and research notes
    row by row, then shows the diff using compare_text_versions.

    This function is designed to be used as a pre_exec_func in a manual step.

    Args:
        sheet: The Google Sheet object
    """
    final_outline_sheet_name = "Final Outline"
    backup_sheet_name = "Backup Final Outline Sheet for Delete step of Research Notes Checklist"

    # Read current research notes from Final Outline
    try:
        _, current_df = get_sheet_data_and_df(sheet=sheet, sheet_name=final_outline_sheet_name)
        print(f"✅ Loaded {len(current_df)} rows from '{final_outline_sheet_name}'")
    except Exception as e:
        st.error(f"❌ Failed to read '{final_outline_sheet_name}' sheet: {e}")
        return

    # Read backup research notes
    try:
        _, backup_df = safe_get_sheet_data_and_df(sheet=sheet, sheet_name=backup_sheet_name)
        if backup_df.empty:
            st.warning(f"⚠️ Backup sheet '{backup_sheet_name}' is empty or does not exist. "
                      "This step requires the 'Checklist Based Review and Revise Agents' step to have been run at least once.")
            return
        print(f"✅ Loaded {len(backup_df)} rows from backup sheet")
    except Exception as e:
        st.error(f"❌ Failed to read backup sheet: {e}")
        st.info("ℹ️ The backup sheet is created when running the 'Checklist Based Review and Revise Agents' step.")
        return

    # Check if research_notes column exists in both dataframes
    if 'research_notes' not in current_df.columns:
        st.error("❌ 'research_notes' column not found in Final Outline sheet.")
        return

    if 'research_notes' not in backup_df.columns:
        st.error("❌ 'research_notes' column not found in backup sheet.")
        return

    # Build concatenated strings for comparison
    current_string = build_research_notes_string(current_df)
    backup_string = build_research_notes_string(backup_df)

    # Check if there are any differences
    if current_string == backup_string:
        st.success("✅ No differences found between current and backup research notes.")
        return

    # Display the diff
    st.write("### Research Notes Diff: Comparing Before and After Checklist Review")
    st.caption("Red = Removed from backup (previous version), Green = Added in current version")

    compare_text_versions(
        text1=backup_string,
        text2=current_string,
        version1_name="Previous (Before Checklist)",
        version2_name="Current (After Checklist)"
    )

    st.info("""
    **Review the changes made by the Checklist Based Review and Revise Agents.**
    - Red highlighted text shows content that was removed or changed.
    - Green highlighted text shows content that was added or modified.
    - Click the button below to confirm you've reviewed the diff.
    """)

    return


def manual_review_research_notes_diff(sheet, **kwargs):
    """
    Manual step function for reviewing research notes diff.
    This is a placeholder function that does nothing - the actual diff is shown via pre_exec_func.

    Args:
        sheet: The Google Sheet object
        **kwargs: Additional keyword arguments (unused but accepted for pipeline compatibility)
    """
    pass


def delete_research_notes_diff(**kwargs):
    """
    Placeholder delete function for the diff visualization step.
    This step doesn't modify any data, so there's nothing to delete.
    """
    pass
