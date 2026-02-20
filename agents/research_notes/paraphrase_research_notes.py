"""
Paraphrase Research Notes Agent

This agent paraphrases generated research notes to make them more engaging and conversational.
Uses parallel processing with ThreadPoolExecutor for efficient batch processing.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import streamlit as st
from agents.paraphraser import run_paraphraser
from services.sheets_service import get_sheet_data_and_df, save_to_sheet, clear_worksheet
from services.smart_progress_bar import SmartProgressBar
from langsmith import traceable
import pandas as pd


@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Paraphraser",
    "function_name": "paraphrase_research_notes",
    "user_id": st.session_state.get("role", "anonymous"),
    "user_email": st.session_state.get("user_email", "anonymous")
})
def paraphrase_research_notes(research_notes_text, trade="HVAC", specialization=None, llm_model="google_genai:gemini-2.5-flash"):
    """
    Paraphrase research notes to make them more engaging and conversational.

    :param research_notes_text: The research notes to paraphrase
    :param trade: The trade context for paraphrasing
    :param specialization: Optional specialization within the trade
    :param llm_model: The language model to use
    :return: The paraphrased research notes
    """

    if not research_notes_text or not research_notes_text.strip():
        return research_notes_text

    try:
        result = run_paraphraser(
            text=research_notes_text,
            trade=trade,
            specialization=specialization,
            preserve_formatting=True,
            target_length="similar",
            max_iterations=1,
            llm_model=llm_model,
            quick_mode=False,  # Set to True to skip quality review for faster processing
            progress_callback=None
        )

        return result.get("final_text", research_notes_text)
    except Exception as e:
        print(f"Error paraphrasing text: {str(e)}. Returning original text.")
        return research_notes_text


@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Paraphraser",
    "function_name": "run_paraphrase_research_notes_for_all_rows",
    "user_id": st.session_state.get("role", "anonymous"),
    "user_email": st.session_state.get("user_email", "anonymous")
})
def run_paraphrase_research_notes_for_all_rows(sheet, worksheet_name, course_name, target_audience, llm='google_genai:gemini-2.5-flash', _drive=None):
    """
    Paraphrase research notes for all rows in the course outline using parallel processing.

    :param sheet: Google Sheet object
    :param worksheet_name: Name of the worksheet
    :param course_name: Course name (for trade context)
    :param target_audience: Target audience
    :param llm: Language model to use
    :param _drive: Google Drive client (optional)
    :return: None
    """

    # Get worksheet and data
    course_outline_with_lo_sheet, course_outline_with_lo_df = get_sheet_data_and_df(sheet, worksheet_name)

    # Ensure paraphrased_research_notes column exists
    if 'paraphrased_research_notes' not in course_outline_with_lo_df.columns:
        course_outline_with_lo_df['paraphrased_research_notes'] = ''

    # Check if already paraphrased
    if (course_outline_with_lo_df['paraphrased_research_notes'] != '').all():
        print('Research notes already paraphrased for all rows')
        return

    # Prepare trade context (use generic "Educational" if no specific trade)
    trade_context = "Educational"
    specialization_context = None

    futures_map = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        for index, row in course_outline_with_lo_df.iterrows():
            # Skip if already paraphrased
            if row['paraphrased_research_notes'] != '' and pd.notna(row['paraphrased_research_notes']):
                print(f'Skipping row {index}. Already paraphrased')
                continue

            # Skip if no research notes
            research_notes = str(row.get('research_notes', '')).strip()
            if not research_notes:
                print(f'Skipping row {index}. No research notes to paraphrase')
                continue

            # Submit paraphrasing task
            future = executor.submit(
                paraphrase_research_notes,
                research_notes_text=research_notes,
                trade=trade_context,
                specialization=specialization_context,
                llm_model=llm
            )

            futures_map[future] = index

        # Collect results
        total_tasks = len(futures_map)
        if total_tasks == 0:
            print('No rows to paraphrase')
            return

        progress = SmartProgressBar(total_tasks=total_tasks, description="Paraphrasing progress", save_interval=5)

        for future in tqdm(as_completed(futures_map), total=total_tasks):
            index = futures_map[future]
            try:
                paraphrased_notes = future.result()
                course_outline_with_lo_df.loc[index, 'paraphrased_research_notes'] = paraphrased_notes
            except Exception as e:
                print(f"Error processing row {index}: {str(e)}")
                # Keep original notes if paraphrasing fails
                course_outline_with_lo_df.loc[index, 'paraphrased_research_notes'] = course_outline_with_lo_df.loc[index, 'research_notes']

            progress.update()
            if progress.should_save():
                print(f'Saving partial progress after {progress.completed_count} tasks.')
                save_to_sheet(worksheet=course_outline_with_lo_sheet, df=course_outline_with_lo_df)

    print('All rows paraphrased. Saving final DataFrame to sheet.')
    save_to_sheet(worksheet=course_outline_with_lo_sheet, df=course_outline_with_lo_df)


def delete_paraphrased_research_notes(sheet, worksheet_name="Final Outline"):
    """Remove the paraphrased_research_notes column from the specified worksheet."""
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    if "paraphrased_research_notes" in df.columns:
        df = df.drop(columns=["paraphrased_research_notes"])
        clear_worksheet(ws)
        save_to_sheet(ws, df)
