from modules.chain import Chain
from tqdm import tqdm
from services.sheets_service import get_sheet_data_and_df, save_to_sheet
import pandas as pd
import streamlit as st
from services.smart_progress_bar import SmartProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed
from langsmith import traceable


generate_learning_objectives_from_deep_research_prompt = """You are an AI assistant specializing in educational design. Your task is to generate clear, specific, and detailed learning objectives based on provided research. These objectives should be tailored to a specific course, audience, and topic.

Please review the following research notes carefully:

<research>
{research}
</research>

Here is the context for the learning objectives you'll be creating:

<course_name>
{course_name}
</course_name>

<target_audience>
{target_audience}
</target_audience>

<topic_name>
{topic_name}
</topic_name>

Before generating the learning objectives, please conduct a thorough analysis of the information provided. Wrap your analysis inside <objective_analysis> tags, including the following steps:

1. Summarize the key concepts, skills, and knowledge areas from the research that are relevant to the topic.
2. Categorize these key concepts into knowledge, skills, and attitudes.
3. Consider how the depth and breadth of the material relate to the course name and target audience.
4. Draft an outline of potential learning objectives, ensuring they cover all significant aspects of the topic.
5. For each objective, consider the appropriate Bloom's Taxonomy level (Remember, Understand, Apply, Analyze, Evaluate, Create).
6. Review your draft to ensure all objectives are within the scope of the given topic.
7. For each objective, draft a sample assessment question to ensure measurability.

After your analysis, create a set of learning objectives that meet the following criteria:

1. Clear: Easy to understand and unambiguous
2. Specific: Focused on particular outcomes or competencies
3. Detailed: Providing enough information to guide both instruction and assessment
4. Measurable: Can be observed, assessed, or evaluated
5. Appropriate: Aligned with the target audience's expected level of knowledge and skills
6. Comprehensive: Covering all significant aspects of the topic as presented in the research
7. Non-redundant: Avoid creating overly similar objectives

Each learning objective should start with an action verb that describes the expected cognitive level (e.g., "explain," "analyze," "evaluate," "create").

Present your final learning objectives in the following format:

<learning_objectives>
1. [Action Verb] [Specific Content] [Context or Condition if applicable]
2. [Action Verb] [Specific Content] [Context or Condition if applicable]
3. [Action Verb] [Specific Content] [Context or Condition if applicable]
...
</learning_objectives>

There is no strict limit on the number of learning objectives. Generate as many as necessary to comprehensively cover the depth of the research notes provided.

Remember, your goal is to create learning objectives that are so clear, specific, and detailed that no uncertainty is left regarding what learners should be able to do or understand after completing this part of the course.
"""

@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Generate Learning Objectives",
    "function_name": "generate_learning_objectives",
    "user_id": st.session_state.get("role", "anonymous")
})
def generate_learning_objectives(course_name, target_audience, topic_name, research, llm='gemini_2_flash'):
    """
    Generate learning objectives based on research for a specific topic.
    
    :param course_name (str): The name of the course
    :param target_audience (str): The target audience for the course
    :param topic_name (str): The name of the topic
    :param research (str): The research content to base learning objectives on
    :param llm (str): The language model to use (default: 'gemini_2_flash')
    :return str: The generated learning objectives
    """
    learning_objectives_agent = Chain(llm=llm, tags=['learning_objectives'])
    
    learning_objectives_agent.add_message(
        role='user', content=generate_learning_objectives_from_deep_research_prompt.format(
            course_name=course_name,
            target_audience=target_audience,
            topic_name=topic_name,
            research=research
        )
    )
    
    response = learning_objectives_agent.run()
    return response['learning_objectives']


def run_generate_learning_objectives(sheet, worksheet_name, course_name, target_audience, llm='gemini_2_flash'):
    """
    Generates learning objectives for each topic in the Topic Deep Research sheet based on the research.
    
    :param sheet (object): Google Sheets object.
    :param worksheet_name (str): Name of the worksheet (default: "Topic Deep Research").
    :param course_name (str): Name of the course.
    :param target_audience (str): Target audience for the course.
    :param llm (str): Language model to use (default: 'gemini_2_flash').
    :returns: None
    """
    # Load Topic Deep Research sheet
    deep_research_sheet, deep_research_df = get_sheet_data_and_df(sheet, worksheet_name)
    
    # Check if the sheet exists and has the required columns
    if deep_research_df.empty:
        raise Exception(f"The '{worksheet_name}' sheet is empty. Please ensure it is properly created.")
    
    if 'topic_query' not in deep_research_df.columns or 'research' not in deep_research_df.columns:
        raise Exception(f"The '{worksheet_name}' sheet is missing required columns: 'topic_query' or 'research'.")
    
    # Ensure learning_objectives column exists
    if 'learning_objectives' not in deep_research_df.columns:
        deep_research_df['learning_objectives'] = ''
    
    # Skip processing if all learning objectives are already populated
    if deep_research_df['learning_objectives'].notna().all() and deep_research_df['learning_objectives'].str.strip().all():
        print("Learning objectives column is already populated for all topics. Skipping processing.")
        return
    
    # Prepare for parallel processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit tasks for each topic
        for index, row in deep_research_df.iterrows():
            topic = row['topic_query']
            research = row['research']
            
            # Skip if learning objectives are already populated for this topic
            if pd.notna(row['learning_objectives']) and row['learning_objectives'].strip():
                print(f"Skipping {index}: Learning objectives already populated for '{topic}'.")
                continue
                
            if not topic.strip() or not pd.notna(research) or not research.strip():
                print(f"Skipping {index}: Empty topic or research.")
                continue
                
            # Submit the task
            future = executor.submit(
                generate_learning_objectives,
                course_name,
                target_audience,
                topic,
                research,
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
            learning_objectives = future.result()
            
            # Update the learning_objectives column
            deep_research_df.loc[index, 'learning_objectives'] = learning_objectives
            
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

