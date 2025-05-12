from services.sheets_service import get_sheet_data_and_df, create_or_read_worksheet, format_worksheet, save_to_sheet
from tqdm import tqdm
import pandas as pd
import streamlit as st
from services.youtube_search import search_youtube_videos
from services.smart_progress_bar import SmartProgressBar
from langsmith import traceable


@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Retrieve HVAC School Videos",
    "function_name": "run_get_hvac_school_videos",
    "user_id": st.session_state.get("role", "anonymous")
})
def run_get_hvac_school_videos(sheet, worksheet_name):
    
    rough_outline_sheet, rough_outline_df = get_sheet_data_and_df(sheet, 'Rough Outline')

    video_search_queries = []
    for query in rough_outline_df['video_search_queries'].to_list():
        if query == '':
            continue
        video_search_queries.append(query)
    
    videos_research_sheet, videos_research_df = create_or_read_worksheet(sheet, worksheet_name, rows = 1000, cols = 30)

    hvac_school_channel_id = "UCdLlUhD9LUGm0-NTYDASPFA"

    query_list = []
    
    if 'search_query' not in videos_research_df.columns:
        videos_research_df['search_query'] = ''
    else:
        for query in videos_research_df['search_query'].to_list():
            query_list.extend(query.split('\n'))
        query_list = list(set(query_list))
    
    # Skip for loop logic
    if videos_research_df.shape[0] > 10:
        print('Skipping getting HVAC School Videos, sheet already populated')
        return

    total_tasks = len(video_search_queries)

    # Initialize the progress tracker
    progress = SmartProgressBar(total_tasks = total_tasks, description = "Percent complete")

    for query in tqdm(video_search_queries):
        
        # Skip queries that were already searched
        if query in query_list:
            print(f"Skip. Query {query} already searched")
            continue
           
        # Run the search
        search_results = search_youtube_videos(query, max_results=10, channel_id=hvac_school_channel_id)

        print(f"Search results for '{query}':", search_results)  # Debugging print

        # Ensure search results are not empty
        if not search_results:
            print(f"No results found for query: {query}")
            continue

        # Ensure 'video_id' exists in search results
        valid_results = [video for video in search_results if 'video_id' in video]

        if not valid_results:
            print(f"Warning: 'video_id' missing in search results for query: {query}")
            continue  # Skip to next query

        # Add valid results to the dataframe
        videos_research_df = pd.concat([videos_research_df, pd.DataFrame(valid_results)], ignore_index=True)

        # Update query list
        query_list.append(query)

        # Update progress
        progress.update()

    # Ensure 'video_id' column exists before grouping
    if 'video_id' not in videos_research_df.columns:
        raise ValueError("Error: 'video_id' column missing from DataFrame. Check the search results format.")

    agg_dict = {col: 'first' for col in videos_research_df.columns if col not in ['video_id', 'search_query']}
    agg_dict['search_query'] = lambda queries: '\n'.join(queries)

    # Group by video_id and aggregate
    videos_research_df = videos_research_df.groupby('video_id', as_index=False).agg(agg_dict)

    save_to_sheet(worksheet = videos_research_sheet, df = videos_research_df)

    # Format the sheet
    format_worksheet(videos_research_sheet)

    # return videos_research_df, query_list, video_search_queries
    return

