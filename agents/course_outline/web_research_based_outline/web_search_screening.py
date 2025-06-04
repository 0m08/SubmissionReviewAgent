from services.web_search import web_search_screening
from services.sheets_service import create_or_read_worksheet, get_sheet_data_and_df, save_to_sheet, format_worksheet
import pandas as pd
import streamlit as st
from tqdm import tqdm
from services.helper_functions import get_outline_with_los
from services.smart_progress_bar import SmartProgressBar
from langsmith import traceable


@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Obtain Web Article Links",
    "function_name": "run_web_search_screening",
    "user_id": st.session_state.get("role", "anonymous")
})
def run_web_search_screening(sheet, worksheet_name, course_name, llm='gemini_2_flash'):
    """
    Runs the web search screening step that gets article links for all the search queries.

    :param sheet (object): Google Sheets object.
    :param worksheet_name (str): Name of the worksheet.
    :param course_name (str): Name of the course.
    :param llm (str): Language model to use (default: 'gemini_2_flash').
    :returns: None
    """

    # Read the sheets and df
    preliminary_research_sheet, preliminary_research_df = create_or_read_worksheet(sheet, worksheet_name)
    rough_outline_sheet, rough_outline_df = get_sheet_data_and_df(sheet = sheet, sheet_name = "Base Outline")

    # Create the column if not present
    if 'query' not in preliminary_research_df.columns:
        preliminary_research_df['query'] = ''
        preliminary_research_df['article'] = ''

    # Get the list of search queries
    refined_search_queries = []
    for ind, row in rough_outline_df.iterrows():
        query = row['search_queries']
        if query == '':
            continue
        if query in preliminary_research_df['query'].values:
            continue
        refined_search_queries.append(query)

    # Collect the results as they complete
    total_tasks = len(refined_search_queries)
    save_interval = 5  # how often to save (in number of completed tasks)

    # Skip is no tasks
    if total_tasks == 0:
        print("Skipping Web Search Screening Step")
        return

    # Get the course outline
    course_outline = get_outline_with_los(
        df = rough_outline_df,
        include_learning_objectives = True
    )

    # Initialize the progress tracker
    progress = SmartProgressBar(total_tasks = total_tasks, description = "Percent complete", save_interval = save_interval)

    # Run the for loop
    for query in tqdm(refined_search_queries):
        
        # Get the articles
        articles = web_search_screening(
            course_name = course_name,
            course_outline = course_outline,
            search_query = query,
            llm = llm
        )

        # Populate the df
        for article in articles:
            preliminary_research_df = pd.concat([preliminary_research_df, pd.DataFrame({'query': [query], 'article': [article]})], ignore_index=True)

        # Update progress
        progress.update()

        # Check if we should save
        if progress.should_save():
            print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
            save_to_sheet(worksheet = preliminary_research_sheet, df = preliminary_research_df)

    # Final save to sheet after all tasks
    print('All rows processed. Saving final DataFrame to sheet.')
    save_to_sheet(worksheet = preliminary_research_sheet, df = preliminary_research_df)

    format_worksheet(worksheet = preliminary_research_sheet)

    return

