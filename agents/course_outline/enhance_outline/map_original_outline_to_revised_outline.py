from modules.chain import Chain
from typing import List, Union, Literal
from pydantic import BaseModel, Field
import pandas as pd
from services.sheets_service import (
    get_sheet_data_and_df,
    create_or_read_worksheet,
    format_worksheet,
    resize_column_by_name,
    save_to_sheet,
    clear_worksheet,
    delete_worksheet,
)
from services.smart_progress_bar import SmartProgressBar
from services.helper_functions import get_outline_in_table_format
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import re
import streamlit as st
from langsmith import traceable


map_original_outline_to_revised_outline_prompt = """You are given two outlines (in Markdown tables): an old outline (partial section) and a new, revised outline.

1. Old Outline: A short section with line number, topic, and learning objective.
<old_outline>
{old_outline}
</old_outline>

2. New/Revised Outline: The updated version with its own line numbers and learning objectives.
<revised_outline>
{new_outline}
</revised_outline>

Your Task:
- For each line in the old outline, determine which line in the revised outline best matches that old line.
- Write down:
  1. The old line number.
  2. The original learning objective from the old line.
  3. Your detailed analysis of how you found the match between old and new lines.
  4. The line number in the new outline that corresponds to the old line.
  5. The new learning objective that corresponds to the old line.
  6. The type of operation that describes the change:
     - **as_is** if the line stayed the same with minimal wording changes
     - **edited** if the line was updated or reworded
     - **removed** if the old line has no counterpart in the new outline

Output Format:
<output>

<line_comparison>
Include all the fields described above as follows:
old_line_number: [The old line number]
old_learning_objective: [The original learning objective from the old line]
analysis: [Your detailed analysis of how you found the match between old and new lines]
new_line_number: [The line number in the new outline that corresponds to the old line]
new_learning_objective: [The new learning objective that corresponds to the old line]
operation_type: [The type of operation that describes the change: as_is, edited, or removed]
</line_comparison>

[Repeat the line_comparison block for each line in the old outline.]

</output>

IMPORTANT: For each line, you MUST include all six elements listed above. Specifically ensure you always include the old_learning_objective and your analysis for each comparison.
"""


map_original_outline_to_revised_outline_structured_output_prompt = """Your task is to extract the relevant data from the following text:

<text>
{text}
</text>

Your output should extract the following data:
- The old line number.
- The line number in the new outline that corresponds to the old line.
- The type of operation that describes the change: as_is, edited, or removed.

Return your findings in the proper format.
"""


@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Map Topic Outline to Enhanced Outline",
    "function_name": "extract_line_mappings_with_regex",
    "user_id": st.session_state.get("role", "anonymous")
})
def extract_line_mappings_with_regex(text):
    """
    Extracts line mappings from the text using regex.
    
    :param text: The text containing line comparison information.
    :return: A DataFrame with the mapping of the old outline to the new outline.
    """
    # Split text into individual line comparison blocks
    line_comparison_blocks = re.split(r'<line_comparison>|</line_comparison>', text)
    
    # Filter out empty blocks or non-comparison text
    line_comparison_blocks = [block.strip() for block in line_comparison_blocks if 'old_line_number' in block]
    
    results = []
    
    for block in line_comparison_blocks:
        # Extract old line number
        old_line_match = re.search(r'old_line_number:\s*(\d+)', block)
        old_line_number = int(old_line_match.group(1)) if old_line_match else None
        
        # Extract new line number (which might be None)
        new_line_match = re.search(r'new_line_number:\s*(\d+)', block)
        new_line_number = int(new_line_match.group(1)) if new_line_match else None
        
        # Extract operation type
        operation_match = re.search(r'operation_type:\s*(as_is|edited|removed)', block)
        operation_type = operation_match.group(1) if operation_match else None
        
        if old_line_number is not None and operation_type is not None:
            results.append({
                "Old Line Number": old_line_number,
                "New Line Number": new_line_number,
                "Operation Type": operation_type
            })
    
    return pd.DataFrame(results)


@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Map Topic Outline to Enhanced Outline",
    "function_name": "map_original_outline_to_revised_outline",
    "user_id": st.session_state.get("role", "anonymous")
})
def map_original_outline_to_revised_outline(old_outline, new_outline, llm = "gemini_2_flash", use_regex=True):
    """
    Maps the original outline to the revised outline.
    :param old_outline: The old outline in Markdown table format.
    :param new_outline: The new outline in Markdown table format.
    :param llm: The LLM to use.
    :param use_regex: Whether to use regex extraction first before falling back to structured output.
    :return: A DataFrame with the mapping of the old outline to the new outline.
    """

    mapping_text_chain = Chain(llm = llm, tags = ["output"])

    mapping_text_chain.add_message(
        role = "user",
        content = map_original_outline_to_revised_outline_prompt.format(
            old_outline = old_outline,
            new_outline = new_outline
        )
    )

    response = mapping_text_chain.run()
    text = response["output"]
    
    if use_regex:
        try:
            # Try regex extraction first
            df = extract_line_mappings_with_regex(text)
            
            # Validate the results
            if not df.empty and all(df["Old Line Number"].notna()) and all(df["Operation Type"].notna()):
                # Check if there's at least one valid mapping for each operation type that should have a mapping
                has_valid_mappings = True
                for _, row in df.iterrows():
                    if row["Operation Type"] in ["as_is", "edited"] and pd.isna(row["New Line Number"]):
                        has_valid_mappings = False
                        break
                
                if has_valid_mappings:
                    return df
                    
            print("Regex extraction produced incomplete results. Falling back to structured output.")
        except Exception as e:
            print(f"Regex extraction failed with error: {str(e)}. Falling back to structured output.")
    
    # Fall back to Pydantic structured output approach
    class LineComparison(BaseModel):
        """
        Represents the comparison (or mapping) of a single line from the old outline 
        to a corresponding (or no) line in the new outline.
        """
        old_line_number: int = Field(
            ...,
            description="Line number of the old outline. Example: 18"
        )
        # old_learning_objective: str = Field(
        #     ...,
        #     description="The original learning objective text from the old outline. Example: 'Understand basic JavaScript syntax'"
        # )
        # analysis: str = Field(
        #     ...,
        #     description=(
        #         "Your detailed reasoning for how you determined the match. Example: 'The old line about JavaScript syntax closely matches line 22 in the new outline with similar wording but more specificity.'"
        #     )
        # )
        new_line_number: Union[int, None] = Field(
            default=None,
            description=(
                "Line number in the new outline that corresponds to this old line. "
                "If not found or removed, this should be None. Example: 22 or None"
            )
        )
        # new_learning_objective: Union[str, None] = Field(
        #     default=None,
        #     description=(
        #         "The new learning objective text that matches the old line. "
        #         "If removed, this should be None. Example: 'Master JavaScript syntax and core concepts'"
        #     )
        # )
        operation_type: Literal["as_is", "edited", "removed"] = Field(
            ...,
            description=(
                "How the old line was handled in the new outline: 'as_is' (minimal changes), "
                "'edited' (significant rewording), or 'removed' (no match in new outline)"
            )
        )

    class OutlineComparison(BaseModel):
        """
        Captures a structured comparison of the old outline's lines to those in the new outline.
        """
        line_comparisons: List[LineComparison] = Field(
            ...,
            description="List of comparisons for each old line against the new outline."
        )

    # Initialize the chain
    mapping_structured_output_chain = Chain(llm = llm)

    mapping_structured_output_chain.add_message(
        role = "user",
        content = map_original_outline_to_revised_outline_structured_output_prompt.format(
            text = text
        )
    )

    mapping_structured_output_chain.structured_output = OutlineComparison

    response = mapping_structured_output_chain.run()

    # Convert extracted data into a DataFrame
    flattened_data = []
    for comparison in response.line_comparisons:
        flattened_data.append({
            "Old Line Number": comparison.old_line_number,
            # "Old Learning Objective": comparison.old_learning_objective,
            # "Analysis": comparison.analysis,
            "New Line Number": comparison.new_line_number,
            # "New Learning Objective": comparison.new_learning_objective,
            "Operation Type": comparison.operation_type
        })
    
    return pd.DataFrame(flattened_data)


@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Map Topic Outline to Enhanced Outline",
    "function_name": "map_additional_columns_to_enhanced_outline",
    "user_id": st.session_state.get("role", "anonymous")
})
def map_additional_columns_to_enhanced_outline(sheet):
    """
    Maps additional columns from the topic outline to the enhanced outline based on the line number mapping.
    :param sheet: The sheet object.
    :return: None
    """
    # Read the Topic Outline worksheet
    topic_outline_sheet, topic_outline_df = get_sheet_data_and_df(sheet, "Topic Outline")
    
    # Read the Topic - Revised Outline Mapping worksheet
    mapping_sheet, mapping_df = get_sheet_data_and_df(sheet, "Topic - Revised Outline Mapping")
    
    # Read the Enhanced Outline with LOs worksheet
    enhanced_outline_sheet, enhanced_outline_df = get_sheet_data_and_df(sheet, "Enhanced Outline with LOs")
    
    # Get columns to map (excluding Topic, Learning Objective and mapping columns)
    columns_to_map = [col for col in topic_outline_df.columns 
                     if col not in ["Topic", "Learning Objective", "mapping"]]
    
    if not columns_to_map:
        print("No additional columns to map from Topic Outline.")
        return
    
    print(f"Mapping additional columns: {', '.join(columns_to_map)}")
    
    # Initialize new columns in enhanced outline if they don't exist
    for col in columns_to_map:
        if col not in enhanced_outline_df.columns:
            enhanced_outline_df[col] = ""
    
    # Convert line numbers to integers where possible
    mapping_df["Old Line Number"] = pd.to_numeric(mapping_df["Old Line Number"], errors="coerce")
    mapping_df["New Line Number"] = pd.to_numeric(mapping_df["New Line Number"], errors="coerce")
    
    # Create a dictionary to track which new lines already have content for each column
    # This helps us know when to append with newline vs. when to set initial value
    mapped_new_lines = {col: set() for col in columns_to_map}
    
    # Map the values based on line numbers
    mapped_count = 0
    for _, mapping_row in mapping_df.iterrows():
        try:
            old_line_number = mapping_row["Old Line Number"]
            new_line_number = mapping_row["New Line Number"]
            operation_type = mapping_row["Operation Type"]
            
            # Update the mapping column in the topic outline sheet with operation type and line number
            if not pd.isna(old_line_number):
                old_idx = int(old_line_number) - 1
                if 0 <= old_idx < len(topic_outline_df):
                    if pd.isna(new_line_number):
                        # If removed (no new line number)
                        topic_outline_df.loc[old_idx, "mapping"] = f"{operation_type}"
                    else:
                        # If mapped to a new line
                        topic_outline_df.loc[old_idx, "mapping"] = f"{operation_type} - {int(new_line_number)}"
            
            # Skip if the line was removed or numbers are invalid for column mapping
            if pd.isna(new_line_number) or pd.isna(old_line_number):
                continue
                
            # Convert to integer and adjust for 0-indexing
            old_idx = int(old_line_number) - 1
            new_idx = int(new_line_number) - 1
            
            # Check if indices are valid
            if old_idx < 0 or old_idx >= len(topic_outline_df) or new_idx < 0 or new_idx >= len(enhanced_outline_df):
                print(f"Warning: Invalid index - Old: {old_line_number}, New: {new_line_number}")
                continue
                
            # Get the values from the old outline
            old_values = topic_outline_df.iloc[old_idx][columns_to_map]
            
            # Update the values in the enhanced outline
            for col in columns_to_map:
                old_value = old_values[col]
                # Skip empty values
                if pd.isna(old_value) or old_value == "":
                    continue
                    
                # If this new line already has content for this column, append with newline
                if new_idx in mapped_new_lines[col]:
                    current_value = enhanced_outline_df.loc[new_idx, col]
                    if current_value and str(current_value).strip():
                        enhanced_outline_df.loc[new_idx, col] = f"{current_value}\n---\n{old_value}"
                    else:
                        enhanced_outline_df.loc[new_idx, col] = old_value
                else:
                    # First time setting this column for this new line
                    enhanced_outline_df.loc[new_idx, col] = old_value
                    mapped_new_lines[col].add(new_idx)
            
            mapped_count += 1
            
        except Exception as e:
            print(f"Error mapping line {old_line_number} to {new_line_number}: {str(e)}")
    
    print(f"Successfully mapped {mapped_count} rows.")
    
    # Save the updated topic outline with mapping information
    save_to_sheet(worksheet=topic_outline_sheet, df=topic_outline_df)
    
    # Save the updated enhanced outline
    save_to_sheet(worksheet=enhanced_outline_sheet, df=enhanced_outline_df)
    
    print(f"Updated Enhanced Outline with {len(columns_to_map)} additional columns.")
    print(f"Updated Topic Outline mapping column with operation type and line number information.")
    return


@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Map Topic Outline to Enhanced Outline",
    "function_name": "map_original_outline_to_revised_outline_for_all_topics",
    "user_id": st.session_state.get("role", "anonymous")
})
def map_original_outline_to_revised_outline_for_all_topics(sheet, worksheet_name = "Enhanced Outline with LOs", llm = "gemini_2_flash"):
    """
    Maps the original outline to the revised outline for all topics.
    :param sheet: The sheet object.
    :param worksheet_name: The name of the worksheet to map the original outline to the revised outline for all topics.
    :param llm: The LLM to use.
    :return: None
    """

    # Read the Topic Outline worksheet
    topic_outline_sheet, topic_outline_df = get_sheet_data_and_df(sheet, "Topic Outline")

    # Read the Enhanced Outline with LOs worksheet
    enhanced_outline_with_los_sheet, enhanced_outline_with_los_df = get_sheet_data_and_df(sheet, worksheet_name)

    # Create the topic revised outline mapping sheet
    topic_revised_outline_mapping_sheet, topic_revised_outline_mapping_df = create_or_read_worksheet(sheet, "Topic - Revised Outline Mapping")

    # Create a mapping column in the topic outline sheet to track the completed mapping
    if "mapping" not in topic_outline_df.columns:
        topic_outline_df["mapping"] = ""

    # Check if all rows in the topic outline sheet have been mapped
    if topic_outline_df["mapping"].all() and not topic_revised_outline_mapping_df.empty:
        print("All rows in the topic outline sheet have been mapped.")
        return

    # Get the rows that have not been mapped
    unmapped_rows = topic_outline_df[topic_outline_df["mapping"] == ""]

    # Get the enhanced outline with LOs in table format
    enhanced_outline_with_los_md_table = get_outline_in_table_format(
        enhanced_outline_with_los_df, 
        topic_col_name="Topic", 
        lo_col_name="Learning Objectives"
    )

    # print(enhanced_outline_with_los_md_table)

    # Prepare for parallel processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers = 5) as executor:
        # Iterate over the topic groups in the unmapped rows
        for topic_group in unmapped_rows["Topic"].unique():
            # Get the rows for the current topic group
            topic_group_rows = unmapped_rows[unmapped_rows["Topic"] == topic_group]

            # Get the old outline as table
            old_outline_table = get_outline_in_table_format(topic_group_rows)

            indices = topic_group_rows.index.tolist()

            # print(old_outline_table)

            # Submit the task
            future = executor.submit(
                map_original_outline_to_revised_outline,
                old_outline_table,
                enhanced_outline_with_los_md_table,
                llm
            )

            # Map the Future to the index
            futures_map[future] = indices

        # Initialize the progress tracker
        progress = SmartProgressBar(total_tasks = len(futures_map), description = "Percent complete", save_interval = 5)

        # Now, pass only the futures (the keys) to as_completed:
        for future in tqdm(as_completed(futures_map), total = len(futures_map)):
            indices = futures_map[future]
            response = future.result()

            # Update the topic outline sheet with the mapping
            for index in indices:
                topic_outline_df.loc[index, "mapping"] = "Done"

            # Update the topic revised outline mapping sheet with the mapping
            topic_revised_outline_mapping_df = pd.concat([topic_revised_outline_mapping_df, response], ignore_index = True)

            # Update progress
            progress.update()
            
            # Check if we should save
            if progress.should_save():
                print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
                save_to_sheet(worksheet = topic_outline_sheet, df = topic_outline_df)
                save_to_sheet(worksheet = topic_revised_outline_mapping_sheet, df = topic_revised_outline_mapping_df)


    # Final save to sheet after all tasks
    print('All topics processed. Saving final DataFrame to sheet.')
    save_to_sheet(worksheet = topic_outline_sheet, df = topic_outline_df)
    # Sort the df by the old line number
    topic_revised_outline_mapping_df = topic_revised_outline_mapping_df.sort_values(by = "Old Line Number")
    save_to_sheet(worksheet = topic_revised_outline_mapping_sheet, df = topic_revised_outline_mapping_df)

    format_worksheet(worksheet = topic_revised_outline_mapping_sheet)

    # Map additional columns from the topic outline to the enhanced outline
    map_additional_columns_to_enhanced_outline(sheet)

    return


def delete_topic_outline_mapping(
    sheet,
    mapping_ws="Topic - Revised Outline Mapping",
    topic_outline_ws="Topic Outline",
    enhanced_outline_ws="Enhanced Outline with LOs",
):
    """Delete the mapping sheet and extra columns added during mapping."""
    delete_worksheet(sheet, mapping_ws)
    ws_topic, df_topic = get_sheet_data_and_df(sheet, topic_outline_ws)
    if "mapping" in df_topic.columns:
        df_topic = df_topic.drop(columns=["mapping"])
        clear_worksheet(ws_topic)
        save_to_sheet(ws_topic, df_topic)
    ws_enhanced, df_enhanced = get_sheet_data_and_df(sheet, enhanced_outline_ws)
    extra_cols = [c for c in df_enhanced.columns if c not in ["Topic", "Learning Objectives"]]
    if extra_cols:
        df_enhanced = df_enhanced.drop(columns=extra_cols)
        clear_worksheet(ws_enhanced)
        save_to_sheet(ws_enhanced, df_enhanced)

