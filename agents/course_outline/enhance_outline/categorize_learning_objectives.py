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
from services.helper_functions import get_topic_outline
from langsmith import traceable


categorize_learning_objectives_prompt = """You are an AI assistant specializing in educational content analysis. Your task is to categorize proposed learning objectives for a specific course topic. Please review the following course information:

<course_outline>
{course_outline}
</course_outline>

<proposed_objectives>
{proposed_objectives}
</proposed_objectives>

<course_name>
{course_name}
</course_name>

<target_audience>
{target_audience}
</target_audience>

The current topic of focus is:
<current_topic>
{current_topic}
</current_topic>

Your task is to analyze these learning objectives and categorize them based on their relevance to the course and current topic. Follow these steps:

1. Summarize the course information and current topic to ensure a clear understanding.
2. List out each proposed learning objective, numbering them for easy reference.
3. Analyze each proposed learning objective in relation to the course information and current topic.
4. Categorize each learning objective into one of the following categories:
   a. Irrelevant for the course
   b. Relevant for current topic and already covered by existing objectives
   c. Relevant for current topic and not covered by existing objectives
   d. Irrelevant for current topic but relevant for another topic and already covered by that topic
   e. Irrelevant for current topic but relevant for another topic and not covered by that topic

Before providing your final categorization, wrap your thought process for each objective inside <objective_analysis> tags. Consider its relevance to the course and current topic, and provide arguments for and against each potential categorization. It's OK for this section to be quite long.

In your analysis:
1. Summarize key points from the course outline and current topic.
2. For each numbered objective:
   - Consider its relevance to the overall course
   - Evaluate its relevance to the current topic
   - Assess its potential relevance to other topics in the course
   - Provide arguments for each possible categorization

After your analysis, present your categorizations in the following format:

<irrelevant_for_course>
[List objectives that fall into this category in this format:
LO: Learning Objective text
]
</irrelevant_for_course>

<relevant_current_covered>
[List objectives that fall into this category, including the current topic name in this format:
LO: Learning Objective text > Topic: Current Topic Name
]
</relevant_current_covered>

<relevant_current_not_covered>
[List objectives that fall into this category, including the current topic name in this format:
LO: Learning Objective text > Topic: Current Topic Name
]
</relevant_current_not_covered>

<irrelevant_current_relevant_other_covered>
[List objectives that fall into this category, specifying the relevant topic name in this format:
LO: Learning Objective text > Topic: Relevant Topic Name
]
</irrelevant_current_relevant_other_covered>

<irrelevant_current_relevant_other_not_covered>
[List objectives that fall into this category, specifying the relevant topic name in this format:
LO: Learning Objective text > Topic: Relevant Topic Name
]
</irrelevant_current_relevant_other_not_covered>

Ensure that your analysis is thorough and your categorizations are consistent with the provided course information. Ensure stating the full learning objective sentence while listing learning objectives within the relevant category tags.
"""


@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Categorize Learning Objectives",
    "function_name": "categorize_learning_objectives",
    "user_id": st.session_state.get("role", "anonymous")
})
def categorize_learning_objectives(course_outline, course_name, target_audience, current_topic, proposed_objectives, llm='gemini_2_flash'):
    """
    Categorize learning objectives for a specific topic.
    
    :param course_outline (str): The full course outline text
    :param course_name (str): The name of the course
    :param target_audience (str): The target audience for the course
    :param current_topic (str): The current topic being analyzed
    :param proposed_objectives (str): The learning objectives to categorize
    :param llm (str): The language model to use (default: 'gemini_2_flash')
    :return str: The categorization results
    """
    categorization_agent = Chain(llm=llm, tags=["irrelevant_for_course", "relevant_current_covered", "relevant_current_not_covered", "irrelevant_current_relevant_other_covered", "irrelevant_current_relevant_other_not_covered"])
    
    categorization_agent.add_message(
        role='user', content=categorize_learning_objectives_prompt.format(
            course_outline=course_outline,
            course_name=course_name,
            target_audience=target_audience,
            current_topic=current_topic,
            proposed_objectives=proposed_objectives
        )
    )
    
    response = categorization_agent.run()
    
    return response


def run_categorize_learning_objectives(sheet, worksheet_name, course_name, target_audience, llm='gemini_2_flash'):
    """
    Categorizes learning objectives for each topic in the Topic Deep Research sheet.
    
    :param sheet (object): Google Sheets object.
    :param worksheet_name (str): Name of the worksheet with the course outline.
    :param course_name (str): Name of the course.
    :param target_audience (str): Target audience for the course.
    :param llm (str): Language model to use (default: 'gemini_2_flash').
    :returns: None
    """
    # Load Topic Deep Research sheet
    deep_research_sheet, deep_research_df = get_sheet_data_and_df(sheet, worksheet_name)
        
    # Ensure relevant columns exist
    if 'irrelevant_for_course' not in deep_research_df.columns:
        deep_research_df['irrelevant_for_course'] = ''
    if 'relevant_current_covered' not in deep_research_df.columns:
        deep_research_df['relevant_current_covered'] = ''
    if 'relevant_current_not_covered' not in deep_research_df.columns:
        deep_research_df['relevant_current_not_covered'] = ''
    if 'irrelevant_current_relevant_other_covered' not in deep_research_df.columns:
        deep_research_df['irrelevant_current_relevant_other_covered'] = ''
    if 'irrelevant_current_relevant_other_not_covered' not in deep_research_df.columns:
        deep_research_df['irrelevant_current_relevant_other_not_covered'] = ''
    
    # Check if categorization has been done for all rows (at least one category populated per row)
    categorization_done = (
        (deep_research_df['irrelevant_for_course'].notna() & deep_research_df['irrelevant_for_course'].str.strip().ne('')) | 
        (deep_research_df['relevant_current_covered'].notna() & deep_research_df['relevant_current_covered'].str.strip().ne('')) | 
        (deep_research_df['relevant_current_not_covered'].notna() & deep_research_df['relevant_current_not_covered'].str.strip().ne('')) | 
        (deep_research_df['irrelevant_current_relevant_other_covered'].notna() & deep_research_df['irrelevant_current_relevant_other_covered'].str.strip().ne('')) | 
        (deep_research_df['irrelevant_current_relevant_other_not_covered'].notna() & deep_research_df['irrelevant_current_relevant_other_not_covered'].str.strip().ne(''))
    ).all()
    
    if categorization_done:
        print("Learning objectives categorization is already populated for all topics. Skipping processing.")
        return

    # Load Topic Outline sheet
    topic_outline_sheet, topic_outline_df = get_sheet_data_and_df(sheet, "Topic Outline")

    # Get the full course outline in text format
    course_outline = get_topic_outline(topic_outline_df)
    
    # Prepare for parallel processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit tasks for each topic in the Deep Research sheet
        for index, row in deep_research_df.iterrows():
            topic = row['topic_query']
            learning_objectives = row['learning_objectives']
            
            # Skip if empty topic or learning objectives
            if not topic.strip() or not pd.notna(learning_objectives) or not learning_objectives.strip():
                print(f"Skipping {index}: Empty topic or learning objectives.")
                continue
                
            # Skip if categorization is already populated for this topic
            if (pd.notna(row['irrelevant_for_course']) and row['irrelevant_for_course'].strip()) or \
               (pd.notna(row['relevant_current_covered']) and row['relevant_current_covered'].strip()) or \
               (pd.notna(row['relevant_current_not_covered']) and row['relevant_current_not_covered'].strip()) or \
               (pd.notna(row['irrelevant_current_relevant_other_covered']) and row['irrelevant_current_relevant_other_covered'].strip()) or \
               (pd.notna(row['irrelevant_current_relevant_other_not_covered']) and row['irrelevant_current_relevant_other_not_covered'].strip()):
                print(f"Skipping {index}: Categorization already populated for '{topic}'.")
                continue
                
            # Submit the task
            future = executor.submit(
                categorize_learning_objectives,
                course_outline,
                course_name,
                target_audience,
                topic,
                learning_objectives,
                llm
            )
            
            # Map the Future to the index
            futures_map[future] = index
        
        # Collect the results as they complete
        total_tasks = len(futures_map)
        save_interval = 5  # how often to save (in number of completed tasks)
        
        # Initialize the progress tracker
        progress = SmartProgressBar(total_tasks=total_tasks, description="Percent complete", save_interval=save_interval)
        
        # Process results as they complete
        for future in tqdm(as_completed(futures_map), total=total_tasks):
            index = futures_map[future]  # retrieve the index
            response = future.result()
            
            # Update the relevant columns
            deep_research_df.loc[index, 'irrelevant_for_course'] = response['irrelevant_for_course']
            deep_research_df.loc[index, 'relevant_current_covered'] = response['relevant_current_covered']
            deep_research_df.loc[index, 'relevant_current_not_covered'] = response['relevant_current_not_covered']
            deep_research_df.loc[index, 'irrelevant_current_relevant_other_covered'] = response['irrelevant_current_relevant_other_covered']
            deep_research_df.loc[index, 'irrelevant_current_relevant_other_not_covered'] = response['irrelevant_current_relevant_other_not_covered']

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


def delete_lo_categorization(sheet, worksheet_name="Topic Deep Research"):
    """Remove learning objective categorization columns."""
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    cols = [
        "irrelevant_for_course",
        "relevant_current_covered",
        "relevant_current_not_covered",
        "irrelevant_current_relevant_other_covered",
        "irrelevant_current_relevant_other_not_covered",
    ]
    cols = [c for c in cols if c in df.columns]
    if cols:
        df = df.drop(columns=cols)
        clear_worksheet(ws)
        save_to_sheet(ws, df)

