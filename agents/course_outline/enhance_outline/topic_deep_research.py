from modules.chain import Chain
from tqdm import tqdm
from services.sheets_service import (
    get_sheet_data_and_df,
    save_to_sheet,
    clear_worksheet,
)
import pandas as pd
import streamlit as st
from services.smart_progress_bar import SmartProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.llm_service import google_search_with_grounding
from langsmith import traceable

# Prompt for researching topics
topic_research_prompt = """Research the following topic within a course - {course_name} and target audience - {target_audience}.

Topic - {topic}.
"""

@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Topic Deep Research",
    "function_name": "deep_research_topic",
    "user_id": st.session_state.get("role", "anonymous")
})
def deep_research_topic(course_name, target_audience, topic, llm="gemini_with_grounding"):
    """
    Performs deep research on a specific topic for a course.
    
    :param course_name (str): The name of the course
    :param target_audience (str): The target audience for the course
    :param topic (str): The topic to research
    :param llm (str): The language model to use ('pplx_deep_research' or 'gemini_with_grounding')
    :return tuple: A tuple containing (research_content, sources)
    """
    if llm == "gemini_with_grounding":
        research_content, sources = google_search_with_grounding(
            prompt = topic_research_prompt.format(
                course_name = course_name,
                target_audience = target_audience,
                topic = topic
            ),
            # model = "gemini-2.0-flash"
        )
    else:
        research_agent = Chain(llm=llm, use_output_parser=False)
        
        research_agent.add_message(
            role='user', content=topic_research_prompt.format(
                course_name=course_name,
                target_audience=target_audience,
                topic=topic
            )
        )
        
        response = research_agent.run()
        print(f"Completed research for topic: {topic}")
        
        # Extract content and citations from the response
        research_content = response.content if hasattr(response, 'content') else str(response)
        sources = response.additional_kwargs.get("citations", []) if hasattr(response, 'additional_kwargs') else []
    
    return research_content, "\n".join(sources)


def run_topic_deep_research(sheet, worksheet_name, course_name, target_audience, llm="pplx_deep_research"):
    """
    Runs deep research on topics in the Topic Deep Research sheet and populates the research and source columns.
    
    :param sheet (object): Google Sheets object
    :param worksheet_name (str): Name of the worksheet to read and update (default: "Topic Deep Research")
    :param course_name (str): Name of the course
    :param target_audience (str): Target audience for the course
    :param llm (str): Language model to use (default: 'pplx_deep_research')
    :return: None
    """
    # Get the Topic Deep Research sheet
    deep_research_sheet, deep_research_df = get_sheet_data_and_df(sheet, worksheet_name)
    
    # Check if the sheet exists and has the required columns
    if deep_research_df.empty:
        raise Exception(f"The '{worksheet_name}' sheet is empty. Please ensure it is properly created.")
    
    if 'topic_query' not in deep_research_df.columns:
        raise Exception(f"The '{worksheet_name}' sheet is missing the required 'topic_query' column.")
    
    # If research column is already populated for all rows, skip processing
    if 'research' in deep_research_df.columns and deep_research_df['research'].notna().all() and deep_research_df['research'].str.strip().all():
        print("Research column is already populated for all topics. Skipping processing.")
        return
    
    # Ensure research and sources columns exist
    if 'research' not in deep_research_df.columns:
        deep_research_df['research'] = ''
    if 'source' not in deep_research_df.columns:
        deep_research_df['source'] = ''
    
    # Prepare for parallel processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit tasks for each topic
        for index, row in deep_research_df.iterrows():
            topic = row['topic_query']
            
            # Skip if research is already populated for this topic
            if pd.notna(row['research']) and row['research'].strip():
                print(f"Skipping {index}: Research already populated for '{topic}'.")
                continue
                
            if not topic.strip():
                print(f"Skipping {index}: Empty topic.")
                continue
                
            # Submit the task
            future = executor.submit(
                deep_research_topic,
                course_name,
                target_audience,
                topic,
                llm
            )
            
            # Map the Future to the index
            futures_map[future] = index
        
        # Collect the results as they complete
        total_tasks = len(futures_map)
        save_interval = 1  # how often to save (in number of completed tasks)
        
        # Initialize the progress tracker
        progress = SmartProgressBar(total_tasks=total_tasks, description="Percent complete", save_interval=save_interval)
        
        # Process results as they complete
        for future in tqdm(as_completed(futures_map), total=total_tasks):
            index = futures_map[future]  # retrieve the index
            research_content, sources = future.result()
            
            # Update the research and sources columns
            deep_research_df.loc[index, 'research'] = research_content
            deep_research_df.loc[index, 'source'] = sources
            
            # Update progress
            progress.update()
            
            # Check if we should save
            if progress.should_save():
                print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
                save_to_sheet(worksheet=deep_research_sheet, df=deep_research_df)
    
    # Final save to sheet after all tasks
    print('All topics processed. Saving final DataFrame to sheet.')
    save_to_sheet(worksheet=deep_research_sheet, df=deep_research_df)

    return


def delete_topic_deep_research(sheet, worksheet_name="Topic Deep Research"):
    """Remove research and source columns from Topic Deep Research sheet."""
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    cols = ["research", "source"]
    cols = [c for c in cols if c in df.columns]
    if cols:
        df = df.drop(columns=cols)
        clear_worksheet(ws)
        save_to_sheet(ws, df)
