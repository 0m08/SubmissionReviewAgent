from modules.chain import Chain
from tqdm import tqdm
from services.sheets_service import (
    get_sheet_data_and_df,
    save_to_sheet,
    create_or_read_worksheet,
    format_worksheet,
    resize_column_by_name,
    get_worksheet_names,
    clear_worksheet,
    delete_worksheet,
)
import pandas as pd
from services.smart_progress_bar import SmartProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field
from typing import List
from services.helper_functions import get_topic_outline, get_outline_in_table_format, create_and_populate_columns
import re
import streamlit as st
from langsmith import traceable


label_learning_objectives_prompt = """You are tasked with labeling an existing list of learning objectives to insert them into a course outline. Here's the current course outline:

<course_outline>
{course_outline}
</course_outline>

<course_name>
{course_name}
</course_name>

<target_audience>
{target_audience}
</target_audience>

Now, here's the list of learning objectives to be processed:

<learning_objectives>
{learning_objectives}
</learning_objectives>

Your task is to go through each learning objective and determine whether it should be inserted as is or if an existing objective should be edited. For each learning objective, you need to provide the following information:
1. The learning objective text
2. The operation type (add or edit)
3. The line number where the operation should occur

Follow these guidelines:
1. Carefully read each learning objective and any associated comments.
2. Compare the learning objective to the existing course outline.
3. Decide whether to insert the objective as is (add) or edit an existing objective (edit).
4. Prefer insertion (add) unless it would introduce duplicity.
5. If editing, choose the most appropriate existing objective to modify.
6. Determine the line number for the operation based on the original course outline.

Important note about line numbers: When determining line numbers for operations, always refer to the original course outline. Do not factor in any previous insertions or edits you've made. Treat each decision independently as if no changes have been made to the outline.

Your final output for each learning objective should be in this format:

<output>
<learning_objective>
LO text: [The learning objective text]
Analysis: [Your analysis of the learning objective]
Operation type: [add/edit]
Line number: [line number in the original outline]
</learning_objective>

[Repeat the learning_objective tags within the output tags for the remaining learning objectives...]

</output>

Process all learning objectives in the given list within the output tags.
"""


get_learning_objective_structured_prompt  = """Here's the set of learning objectives labeled with the operation type and line number:

<learning_objectives>
{learning_objectives}
</learning_objectives>

Your task is to extract the learning objective text, operation type, and line number for each learning objective.

For each learning objective, provide the following information:
1. The learning objective text
2. The operation type (add or edit)
3. The line number where the operation should occur
"""


@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Label Learning Objectives",
    "function_name": "extract_learning_objective_details_with_regex",
    "user_id": st.session_state.get("role", "anonymous")
})
def extract_learning_objective_details_with_regex(text):
    """
    Extracts learning objective details from the text using regex.
    
    :param text: The text containing learning objective information.
    :return: A DataFrame with the extracted learning objective details.
    """
    # Split text into individual learning objective blocks
    learning_objective_blocks = re.split(r'<learning_objective>|</learning_objective>', text)
    
    # Filter out empty blocks or non-learning objective text
    learning_objective_blocks = [block.strip() for block in learning_objective_blocks if 'LO text:' in block]
    
    results = []
    
    for block in learning_objective_blocks:
        # Extract learning objective text
        lo_text_match = re.search(r'LO text:\s*(.*?)(?=\n|$)', block)
        lo_text = lo_text_match.group(1).strip() if lo_text_match else None
        
        # Extract operation type
        operation_match = re.search(r'Operation type:\s*(add|edit)', block)
        operation_type = operation_match.group(1) if operation_match else None
        
        # Extract line number
        line_match = re.search(r'Line number:\s*(\d+)', block)
        line_number = int(line_match.group(1)) if line_match else None
        
        if lo_text is not None and operation_type is not None and line_number is not None:
            results.append({
                "objective_text": lo_text,
                "line_number": line_number,
                "operation_type": operation_type
            })
    
    return pd.DataFrame(results)

@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Label Learning Objectives",
    "function_name": "get_learning_objective_structured",
    "user_id": st.session_state.get("role", "anonymous")
})
def get_learning_objective_structured(learning_objectives, llm='gemini_2_flash', use_regex=True):
    """
    Get the learning objectives in a structured format.
    """
    if use_regex:
        try:
            # Try regex extraction first
            df = extract_learning_objective_details_with_regex(learning_objectives)
            
            # Validate the results
            if not df.empty and all(df["objective_text"].notna()) and all(df["operation_type"].notna()) and all(df["line_number"].notna()):
                return df
                    
            print("Regex extraction produced incomplete results. Falling back to structured output.")
        except Exception as e:
            print(f"Regex extraction failed with error: {str(e)}. Falling back to structured output.")

    class LearningObjective(BaseModel):
        """
        A single learning objective with objective text, its corresponding line number in the course outline and the operation type.
        """
        objective_text: str = Field(
            description="The text of the learning objective."
        )
        line_number: int = Field(
            description="The line number of this objective in the course outline."
        )
        operation_type: str = Field(
            description="The operation type (add or edit)."
        )

    class LearningObjectives(BaseModel):
        """
        Holds a list of learning objectives.
        """
        objectives: List[LearningObjective] = Field(
            description="List of learning objectives."
        )

    get_learning_objective_structured_agent = Chain(llm=llm)

    get_learning_objective_structured_agent.add_message(
        role='user',
        content=get_learning_objective_structured_prompt.format(
            learning_objectives=learning_objectives
        )
    )

    get_learning_objective_structured_agent.structured_output = LearningObjectives

    response = get_learning_objective_structured_agent.run()

    # Create a dataframe from the response with the columns - objective_text, line_number, operation_type
    learning_objectives_df = pd.DataFrame(columns=['objective_text', 'line_number', 'operation_type'])
    for objective in response.objectives:
        # Add the objective to the dataframe
        learning_objectives_df = pd.concat([learning_objectives_df, pd.DataFrame({
            'objective_text': [objective.objective_text],
            'line_number': [objective.line_number],
            'operation_type': [objective.operation_type]
        })], ignore_index=True)

    return learning_objectives_df

@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Label Learning Objectives",
    "function_name": "label_learning_objectives",
    "user_id": st.session_state.get("role", "anonymous")
})
def label_learning_objectives(course_outline, course_name, target_audience, learning_objectives, llm='gemini_2_flash'):
    """
    Label learning objectives for a course outline.

    :param course_outline: The course outline to be processed.
    :param course_name: The name of the course.
    :param target_audience: The target audience for the course.
    :param learning_objectives: The list of learning objectives to be processed.
    :param llm: The LLM to use for the task.
    :return: A tuple of the original response and the structured learning objectives.
    """

    label_learning_objectives_agent = Chain(llm=llm, tags=['output'])
    
    label_learning_objectives_agent.add_message(
        role='user',
        content=label_learning_objectives_prompt.format(
            course_outline=course_outline,
            course_name=course_name,
            target_audience=target_audience,
            learning_objectives=learning_objectives
        )
    )

    response = label_learning_objectives_agent.run()

    # Get the learning objectives in a structured format
    learning_objectives_df = get_learning_objective_structured(response['output'], llm)

    # Return original response and the structured learning objectives
    return response['output'], learning_objectives_df


def update_course_outline_with_missing_objectives(sheet, topic_outline_sheet_name="Topic Outline", missing_lo_sheet_name="Missing Learning Objectives", revised_outline_sheet_name="Revised Outline"):
    """
    Updates the course outline with missing learning objectives.

    This function reads the course outline from the Topic Outline sheet and the missing learning objectives
    from the Missing Learning Objectives sheet. It then applies the operations (add/edit) from the
    missing learning objectives to update the course outline.

    :param sheet: The Google Sheet object.
    :param topic_outline_sheet_name: The name of the Topic Outline worksheet.
    :param missing_lo_sheet_name: The name of the Missing Learning Objectives worksheet.
    :param revised_outline_sheet_name: The name of the worksheet to save the revised outline.
    :return: None
    """
    # Read the course outline and missing learning objectives
    _, topic_outline_df = get_sheet_data_and_df(sheet, topic_outline_sheet_name)
    _, missing_lo_df = get_sheet_data_and_df(sheet, missing_lo_sheet_name)
    
    # Check if the missing learning objectives dataframe is empty
    if missing_lo_df.empty:
        print("No missing learning objectives to process. Exiting.")
        return
    
    # Make a copy of the original dataframe to avoid modifying it directly
    updated_outline_df = topic_outline_df.copy()
    
    # Add a column to track changes
    updated_outline_df['Change Type'] = 'Original'
    
    # Convert line numbers to 0-based indices for DataFrame operations
    # (sheet shows 1-based line numbers, but DataFrame uses 0-based indices)
    missing_lo_df['index'] = missing_lo_df['line_number'] - 1
    
    # Sort by line number to process in order (important for add operations)
    # We sort in descending order so that adding new rows doesn't affect the
    # indices of rows we haven't processed yet
    missing_lo_df = missing_lo_df.sort_values(by='line_number', ascending=False)
    
    # Track number of rows added for each line number
    added_rows = {}
    
    # Process each operation
    for _, row in missing_lo_df.iterrows():
        operation_type = row['operation_type'].lower()
        line_index = row['index']  # 0-based index
        objective_text = row['objective_text']
        
        if operation_type == 'edit':
            # For edits, simply replace the learning objective at the specified line
            if 0 <= line_index < len(updated_outline_df):
                updated_outline_df.at[line_index, 'Learning Objective'] = objective_text
                updated_outline_df.at[line_index, 'Change Type'] = 'Edited'
            else:
                print(f"Warning: Cannot edit line {line_index + 1} - index out of range")
                
        elif operation_type == 'add':
            # For adds, insert a new row after the specified line
            if 0 <= line_index < len(updated_outline_df):
                # Get the topic from the specified line
                topic = updated_outline_df.iloc[line_index]['Topic']
                
                # Create a new row with the same topic but new learning objective
                new_row = pd.DataFrame({
                    'Topic': [topic],
                    'Learning Objective': [objective_text],
                    'Change Type': ['Added']
                })
                
                # Calculate the insert position
                # We add after the specified line, plus any rows already added at this position
                insert_position = line_index + 1
                if line_index in added_rows:
                    insert_position += added_rows[line_index]
                
                # Insert the new row
                updated_outline_df = pd.concat([
                    updated_outline_df.iloc[:insert_position],
                    new_row,
                    updated_outline_df.iloc[insert_position:]
                ]).reset_index(drop=True)
                
                # Update the tracking of added rows
                if line_index in added_rows:
                    added_rows[line_index] += 1
                else:
                    added_rows[line_index] = 1
            else:
                print(f"Warning: Cannot add after line {line_index + 1} - index out of range")
    
    # Replace NaN with empty string
    updated_outline_df = updated_outline_df.fillna('')

    # Create or read the revised outline sheet
    revised_outline_sheet, _ = create_or_read_worksheet(sheet, revised_outline_sheet_name)
    
    # Save the updated outline to the revised outline sheet
    save_to_sheet(worksheet=revised_outline_sheet, df=updated_outline_df)
    
    # Format the worksheet
    format_worksheet(worksheet=revised_outline_sheet)
    
    # Resize columns
    resize_column_by_name(revised_outline_sheet, "Topic", 300)
    resize_column_by_name(revised_outline_sheet, "Learning Objective", 500)
    resize_column_by_name(revised_outline_sheet, "Change Type", 100)
    
    print(f"Revised course outline saved to '{revised_outline_sheet_name}' sheet.")
    return


def run_label_learning_objectives(sheet, worksheet_name, course_name, target_audience, llm='gemini_2_flash'):
    """
    Run the label learning objectives process and save the results to the missing learning objectives sheet.

    :param sheet: The Google Sheet object.
    :param worksheet_name: The name of the worksheet - Missing Learning Objectives.
    :param course_name: The name of the course.
    :param target_audience: The target audience for the course.
    :param llm: The LLM to use for the task.
    :return: None
    """
    # Create or read missing learning objectives sheet
    learning_objectives_sheet, learning_objectives_df = create_or_read_worksheet(sheet, worksheet_name)

    # Read the topic deep research sheet
    topic_deep_research_sheet, topic_deep_research_df = get_sheet_data_and_df(sheet, 'Topic Deep Research')

    # Add a column to keep track of whether the learning objective has been labeled
    if 'labeled' not in topic_deep_research_df.columns:
        topic_deep_research_df['labeled'] = ""

    # Check if all rows are labeled i.e not empty strings and Revised Outline and Enhanced Outline Review sheets are present
    sheet_names = get_worksheet_names(sheet)
    if topic_deep_research_df['labeled'].str.strip().ne('').all() and 'Revised Outline' in sheet_names and 'Enhanced Outline Review' in sheet_names:
        print('All rows are labeled. Exiting.')
        return

    # Read the topic outline sheet
    topic_outline_sheet, topic_outline_df = get_sheet_data_and_df(sheet, 'Topic Outline')
    
    # Get the course outline in a table format
    course_outline_table = get_outline_in_table_format(topic_outline_df)

    # Prepare for parallel processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit tasks for each topic in the Deep Research sheet
        for index, row in topic_deep_research_df.iterrows():
            topic = row['topic_query']
            learning_objectives = row['relevant_current_not_covered'] + '\n' + row['irrelevant_current_relevant_other_not_covered']
            
            # Skip if empty topic or learning objectives
            if not topic.strip() or not pd.notna(learning_objectives) or not learning_objectives.strip():
                print(f"Skipping {index}: Empty topic or learning objectives.")
                continue

            # Check if the learning objective has already been labeled
            if row['labeled'].strip():
                print(f"Skipping {index}: Learning objective already labeled.")
                continue
                
            # Submit the task
            future = executor.submit(
                label_learning_objectives,
                course_outline_table,
                course_name,
                target_audience,
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
            response_text, response_df = future.result()
            
            # Update the labeled column
            topic_deep_research_df.loc[index, 'labeled'] = response_text

            # Update the learning objectives df
            learning_objectives_df = pd.concat([learning_objectives_df, response_df], ignore_index=True)

            # Update progress
            progress.update()
            
            # Check if we should save
            if progress.should_save():
                print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
                save_to_sheet(worksheet=learning_objectives_sheet, df=learning_objectives_df)
                save_to_sheet(worksheet=topic_deep_research_sheet, df=topic_deep_research_df)
    
    # Final save to sheet after all tasks
    print('All topics processed. Saving final DataFrame to sheet.')
    save_to_sheet(worksheet=learning_objectives_sheet, df=learning_objectives_df)
    save_to_sheet(worksheet=topic_deep_research_sheet, df=topic_deep_research_df)
    
    # Format the learning objectives sheet
    format_worksheet(worksheet=learning_objectives_sheet)

    # Update the course outline with the missing learning objectives
    if 'Revised Outline' not in sheet_names:
        update_course_outline_with_missing_objectives(sheet, topic_outline_sheet_name="Topic Outline", missing_lo_sheet_name="Missing Learning Objectives", revised_outline_sheet_name="Revised Outline")
    else:
        print("Revised Outline sheet already exists. Skipping update.")

    # Update the Enhanced Outline Review sheet
    outline_review_sheet, outline_review_df = create_or_read_worksheet(sheet, 'Enhanced Outline Review')

    if outline_review_df.empty:
        # Get topic outline from the Revised Outline sheet
        _, revised_outline_df = get_sheet_data_and_df(sheet, 'Revised Outline')
        topic_outline = get_topic_outline(revised_outline_df, use_text_labels = True)

        # Create initial DataFrame with Turn column and other columns
        outline_review_df = pd.DataFrame([{
            'Turn': 1,
            'Verdict': '',
            'Manual Feedback': '',
            'AI Suggestions': ''
        }])

        # Use create_and_populate_columns to split large topic outline into multiple columns
        outline_review_df = create_and_populate_columns(
            df=outline_review_df,
            text=topic_outline,
            specific_index=0,
            col_base_name='outline_chunk',
            chunk_size=49000
        )

        # Reorder columns to have Turn first, followed by outline_chunk columns, then the rest
        all_columns = outline_review_df.columns.tolist()
        outline_chunk_columns = [col for col in all_columns if col.startswith('outline_chunk')]
        other_columns = [col for col in all_columns if col != 'Turn' and not col.startswith('outline_chunk')]
        
        # Define the new column order
        new_column_order = ['Turn'] + sorted(outline_chunk_columns) + other_columns
        
        # Reorder the DataFrame
        outline_review_df = outline_review_df[new_column_order]

        save_to_sheet(worksheet = outline_review_sheet, df = outline_review_df)
        format_worksheet(worksheet = outline_review_sheet)
    else:
        print("Enhanced Outline Review sheet already exists and is not empty. Skipping creation.")

    return


def delete_label_learning_objectives(
    sheet,
    missing_lo_ws="Missing Learning Objectives",
    revised_outline_ws="Revised Outline",
    outline_review_ws="Enhanced Outline Review",
    topic_deep_research_ws="Topic Deep Research",
):
    """Clean up worksheets and labeled column from Topic Deep Research."""
    delete_worksheet(sheet, missing_lo_ws)
    delete_worksheet(sheet, revised_outline_ws)
    delete_worksheet(sheet, outline_review_ws)
    ws, df = get_sheet_data_and_df(sheet, topic_deep_research_ws)
    if "labeled" in df.columns:
        df["labeled"] = ""
        save_to_sheet(ws, df)

