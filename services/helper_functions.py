import pandas as pd
import math
from typing import List, Optional
import difflib
import streamlit as st
import re

from services.sheets_service import get_sheet_data_and_df, create_or_read_worksheet, save_to_sheet, format_worksheet, delete_worksheet

# Get outline as text with topic, subtopic and los (if present)
def get_outline_with_los(df, include_learning_objectives = False, include_prefix = True):
    """
    This function returns full outline as text with topics, subtopics and LOs (if populated in df)
    """
    output_text = ""
    current_topic = None
    current_subtopic = None

    topic_prefix = "Topic: " if include_prefix else ""
    subtopic_prefix = "  Subtopic: " if include_prefix else ""
    lo_prefix = "    Learning Objectives:\n" if include_prefix else ""

    for index, row in df.iterrows():
        if row['Topic'] == "":
            continue
        # Check if the topic has changed
        if row['Topic'] != current_topic:
            if current_topic is not None:
                output_text += "\n"
            current_topic = row['Topic']
            output_text += f"{topic_prefix}{current_topic}\n"

        # Check if the subtopic has changed
        if row['Subtopic'] != current_subtopic:
            current_subtopic = row['Subtopic']
            output_text += f"{subtopic_prefix}{current_subtopic}\n"

        # Check if there are learning objectives and display them

        if include_learning_objectives and pd.notna(row.get('Learning Objectives')) and row['Learning Objectives']:
            learning_objectives = row['Learning Objectives'].split('\n')  # Assuming they are separated by new line
            output_text += f"{lo_prefix}"
            for lo in learning_objectives:
                output_text += f"      {lo.strip()}\n"

    return output_text.strip()


def create_and_populate_columns(df: pd.DataFrame, text: str, specific_index: int, col_base_name: str, chunk_size: int = 49000) -> pd.DataFrame:
    """
    This function creates and populates a specific row in a dataframe.

    :param df: The dataframe to populate.
    :param text: The text to split into columns.
    :param specific_index: The index of the specific row to populate.
    :param col_base_name: The base name of the columns.
    :param chunk_size: The size of each chunk.
    :return: The populated dataframe.
    """

    text_length = len(text)
    print(f'Text length: {text_length}')
    
    # Calculate how many chunks (and thus columns) we need
    needed_columns = math.ceil(text_length / chunk_size) if text_length > 0 else 1
    
    # Split the text into 50,000-char chunks
    chunks = [
        text[i*chunk_size : (i+1)*chunk_size]
        for i in range(needed_columns)
    ]
    
    # Create/populate columns
    for i, chunk in enumerate(chunks):
        col_name = f"{col_base_name}_{i}"
        
        # If the column doesn't exist, create it with empty strings
        if col_name not in df.columns:
            df[col_name] = ""
        
        # Now populate the column with the chunk
        df.loc[specific_index, col_name] = chunk

    return df


def get_short_name(text: str) -> str:
    """
    Return a valid collection name (3-63 chars, a-z0-9, _ or - only,
    starting & ending with an alphanumeric).
    """
    if not text:
        return "untitled"

    # 1. lowercase + collapse whitespace into single underscores
    text = re.sub(r'\s+', '_', text.strip().lower())

    # 2. drop every char that is *not* allowed
    text = re.sub(r'[^a-z0-9_-]', '', text)

    # 3. ensure we don’t start / end with _ or -
    text = text.strip('_-')

    # 4. truncate to 60 chars to stay under the 63-char hard limit
    text = text[:60]

    # 5. if the name is now too short, pad it
    if len(text) < 3:
        text = text.rjust(3, 'x')          # gives ‘xxx’, ‘axx’, … as needed

    return text or "untitled"


def add_list_as_new_column(df: pd.DataFrame, new_values: list, new_col_name: str) -> pd.DataFrame:
    """
    Adds a new column to an existing DataFrame using the values from `new_values`.
    If `new_values` has more items than `df` has rows, it appends new rows
    (with blank values in the existing columns) to accommodate all `new_values`.

    :param: df (pd.DataFrame): The original DataFrame.
    :param: new_values (list): The list of values to be added as a new column.
    :param: new_col_name (str): The name of the new column to be added.
    :returns: df (pd.DataFrame): The updated DataFrame with the new column.
    """
    
    # Number of rows in the original df
    n_original = len(df)

    # Number of new values
    n_new_values = len(new_values)

    # Case 1: If the new_values list is shorter or equal to the existing number of rows
    if n_original >= n_new_values:
        # Pad new_values with "" so it matches exactly the length of df
        diff = n_original - n_new_values
        if diff > 0:
            new_values = new_values + [""] * diff
        df[new_col_name] = new_values
        return df
    
    # Case 2: The new list is longer than the DataFrame's row count
    # 2.1 Assign the first 'n_original' items to the existing DataFrame
    df[new_col_name] = new_values[:n_original]

    # 2.2 Create a separate DataFrame for the extra rows
    n_extra = n_new_values - n_original

    # Build a dict where each existing column has blank ('') values
    extra_data = {
        col: [''] * n_extra for col in df.columns if col != new_col_name
    }
    # The new column in these extra rows has the remaining new values
    extra_data[new_col_name] = new_values[n_original:]

    # Create the extra DataFrame
    df_extra = pd.DataFrame(extra_data)

    # 2.3 Concatenate original and extra DataFrame
    final_df = pd.concat([df, df_extra], ignore_index=True)

    return final_df


def find_blank_followed_by_filled_indices(df, column_name, blank_value=""):
    """
    Finds rows in `df[column_name]` where the current row is blank
    and the next row is filled, then returns the indices of those rows.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to check.
    column_name : str
        The column to inspect for blank -> filled transitions.
    blank_value : Any, optional
        The value to consider as "blank". Default is np.nan (missing data).
        For other use cases (e.g., an empty string), set blank_value="".

    Returns
    -------
    list of tuples or str
        If one or more rows match, returns a list of tuples where each tuple
        is (row_index, next_row_index).
        If no rows match, returns a message (str) indicating no issues found.
    """
    
    # Identify where current row is blank
    if pd.isna(blank_value):
        # Checking for missing (NaN) in the current row
        is_blank = df[column_name].isna()
        # Checking for not missing in the next row
        is_filled_next = df[column_name].shift(-1).notna()
    else:
        # Checking for the specified blank_value in the current row
        is_blank = df[column_name].eq(blank_value)
        # Checking that next row is not the same blank_value (and not NaN)
        next_col = df[column_name].shift(-1)
        is_filled_next = (next_col.notna() & ~next_col.eq(blank_value))

    # Combine conditions: current row blank AND next row filled
    blank_followed_by_filled_mask = is_blank & is_filled_next

    # Get the indices where this condition is True
    # We also map each such index to its subsequent (next) index
    problem_indices = [
        (idx, df.index[i+1]) 
        for i, idx in enumerate(df.index) 
        if i < len(df.index) - 1 and blank_followed_by_filled_mask.iloc[i]
    ]
    
    if problem_indices:
        # Return a list of tuples: (current_row_index, next_row_index)
        error_message = "\n".join(
            [f"Row {current_idx+1} has a blank value followed by a filled value in row {next_idx+1}." 
             for (current_idx, next_idx) in problem_indices]
        )
        error_message = "Found the following issues:\n\n" + error_message + "\n\nTo Fix these issues, make sure to not leave any blank spaces in between the column values."
        raise Exception(error_message)
    else:
        return True


def validate_column_values(
    df: pd.DataFrame,
    filter_column: str,
    validation_column: Optional[str] = None,
    valid_values: Optional[List[str]] = None,
    case_sensitive: bool = True,
    require_populated: bool = False
) -> pd.DataFrame:
    """
    Filters a DataFrame based on non-empty values in filter_column,
    then validates that values in validation_column are within a list of valid values
    and/or that the validation column is populated for all rows.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to validate
    filter_column : str
        The column to use for filtering out empty rows
    validation_column : str, optional
        The column to validate values against valid_values.
        If None, uses the same column as filter_column.
    valid_values : List[str], optional
        List of allowed values for the validation column.
        If None, no validation is performed against a list of values.
    case_sensitive : bool, default=True
        Whether validation should be case sensitive
    require_populated : bool, default=False
        If True, checks that validation_column is populated (not empty or null)
        for all rows after filtering
        
    Returns:
    --------
    pandas.DataFrame
        Filtered DataFrame with only non-empty rows that pass validation
        
    Raises:
    -------
    ValueError
        If columns don't exist or validation fails with details about invalid rows
    """
    # Check if filter column exists
    if filter_column not in df.columns:
        raise ValueError(f"Filter column '{filter_column}' does not exist in the DataFrame")
    
    # If validation column not specified, use filter column
    if validation_column is None:
        validation_column = filter_column
    elif validation_column not in df.columns:
        raise ValueError(f"Validation column '{validation_column}' does not exist in the DataFrame")
    
    # Filter out empty rows (NaN, None, empty string)
    filtered_df = df[df[filter_column].notna() & (df[filter_column].astype(str) != '')]
    
    # Check if any rows remain after filtering
    if filtered_df.empty:
        print(f"No populated rows found in column '{filter_column}'")
        return filtered_df
    
    # Check if validation column is populated for all rows when required
    if require_populated:
        empty_rows = filtered_df[filtered_df[validation_column].isna() | 
                               (filtered_df[validation_column].astype(str) == '')]
        
        if not empty_rows.empty:
            error_messages = []
            for index, _ in empty_rows.iterrows():
                error_messages.append(f"Row {index + 2}: '{validation_column}' is empty or null")
            
            raise ValueError(f"Empty values found in column '{validation_column}':\n" + 
                            "\n".join(error_messages))
    
    # If valid_values provided, validate the specified column
    if valid_values is not None:
        # Only validate non-empty values
        rows_to_validate = filtered_df[filtered_df[validation_column].notna() & 
                                     (filtered_df[validation_column].astype(str) != '')]
        
        if not rows_to_validate.empty:
            if not case_sensitive:
                # Convert everything to lowercase for case-insensitive comparison
                validation_series = rows_to_validate[validation_column].astype(str).str.lower()
                valid_values_lower = [str(val).lower() for val in valid_values]
                invalid_mask = ~validation_series.isin(valid_values_lower)
            else:
                # Case-sensitive comparison
                validation_series = rows_to_validate[validation_column]
                invalid_mask = ~validation_series.isin(valid_values)
            
            invalid_rows = rows_to_validate[invalid_mask]
            
            if not invalid_rows.empty:
                error_messages = []
                valid_values_str = ", ".join([f"'{v}'" for v in valid_values])
                
                for index, row in invalid_rows.iterrows():
                    value = row[validation_column]
                    error_messages.append(f"Row {index + 2}: Found '{value}' instead of one of {valid_values_str}")
                
                raise ValueError(f"Invalid values found in column '{validation_column}':\n" + 
                                "\n".join(error_messages))
    
    return filtered_df


def escape_single_braces(text: str) -> str:
    """
    Escapes single braces '{' and '}' to prevent LangChain template format errors.
    
    This function directly addresses the "Single '}' encountered in format string"
    error that occurs with LangChain ChatPromptTemplate.
    
    The function will:
    1. Replace single '{' with '{{'
    2. Replace single '}' with '}}'
    3. Preserve already-escaped braces '{{' and '}}'
    
    Examples:
      A single { brace } -> A single {{ brace }}
      Double {{ braces }} -> Double {{ braces }}
      LaTeX: \frac{p}{q} -> LaTeX: \frac{{p}}{{q}}
    """
    if not text:
        return text
    
    # First handle closing braces (to avoid issues with nested replacements)
    # Start with a string where we'll accumulate our result
    result = []
    i = 0
    while i < len(text):
        # Check for an already-escaped right brace
        if i < len(text) - 1 and text[i:i+2] == '}}':
            result.append('}}')
            i += 2
        # Check for a single right brace that needs escaping
        elif text[i] == '}':
            result.append('}}')
            i += 1
        # Check for an already-escaped left brace
        elif i < len(text) - 1 and text[i:i+2] == '{{':
            result.append('{{')
            i += 2
        # Check for a single left brace that needs escaping
        elif text[i] == '{':
            result.append('{{')
            i += 1
        # Regular character, no change needed
        else:
            result.append(text[i])
            i += 1
    
    return ''.join(result)


def get_topic_outline(df, use_text_labels=False):
    """
    Create a textual outline from a DataFrame with 'topic' and 'learning objective' columns.
    Returns a single string with the formatted outline.
    
    Args:
        df: DataFrame containing 'Topic' and 'Learning Objective' columns
        use_text_labels: If True, use "Topic:" and "LO:" labels instead of numbers
    """

    # We group by the 'topic' column to get all related learning objectives
    outline_lines = []
    grouped = df.groupby("Topic", sort=False)
    
    for i, (topic, group) in enumerate(grouped, start=1):
        # Add the topic header
        if use_text_labels:
            outline_lines.append(f"Topic: {topic}")
        else:
            outline_lines.append(f"{i}. {topic}")
        
        # List each learning objective
        for j, lo in enumerate(group["Learning Objective"], start=1):
            if use_text_labels:
                outline_lines.append(f"   LO: {lo}")
            else:
                outline_lines.append(f"   {i}.{j} {lo}")

    # Combine into one output string
    return "\n".join(outline_lines)


# Function to get outline in this format
def get_outline_in_table_format(topic_outline_df, topic_col_name="Topic", lo_col_name="Learning Objective"):
    """
    Get the course outline in a table format.

    :param topic_outline_df: The topic outline dataframe with the Topic and Learning Objective columns.
    :return: The course outline in a table format.
    """
    # Add line number column
    topic_outline_df['Line No.'] = topic_outline_df.index + 1

    # Select the required columns
    outline_table = topic_outline_df[['Line No.', topic_col_name, lo_col_name]]

    # Convert to markdown table
    outline_table_md = "| Line No. | Topic | Learning Objective |\n| --- | --- | --- |\n"
    for index, row in outline_table.iterrows():
        outline_table_md += f"| {row['Line No.']} | {row[topic_col_name]} | {row[lo_col_name]} |\n"

    return outline_table_md


def compare_text_versions(text1: str, text2: str, version1_name: str = "Version 1", version2_name: str = "Version 2"):
    """
    Compare two versions of text and display them side by side in Streamlit
    with color-coded highlights for changes.

    :param text1: The first text version.
    :param text2: The second text version.
    :param version1_name: The name of the first version.
    :param version2_name: The name of the second version.
    """
    # Split text into lines
    lines1 = text1.splitlines()
    lines2 = text2.splitlines()

    # Use difflib to get differences
    diff = list(difflib.ndiff(lines1, lines2))

    # Table rows to store the differences
    table_rows = []
    
    # Helper function to preserve spaces and special characters
    def format_content(text):
        # Replace spaces with non-breaking spaces for leading spaces (indentation)
        # This regex looks for spaces at the beginning of a line
        indented_text = ""
        i = 0
        while i < len(text) and text[i] == ' ':
            indented_text += "&nbsp;"
            i += 1
        
        # For the rest of the line, handle special HTML characters
        rest_of_text = text[i:]
        rest_of_text = rest_of_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        
        return indented_text + rest_of_text

    # Process the diff and build table rows
    i = 0
    while i < len(diff):
        line = diff[i]
        
        if line.startswith("- "):  # Line removed from version 1
            content = format_content(line[2:])
            left_cell = f'<td class="removed">{content}</td>'
            right_cell = '<td>&nbsp;</td>'
            table_rows.append(f'<tr>{left_cell}{right_cell}</tr>')
            i += 1
        elif line.startswith("+ "):  # Line added in version 2
            content = format_content(line[2:])
            left_cell = '<td>&nbsp;</td>'
            right_cell = f'<td class="added">{content}</td>'
            table_rows.append(f'<tr>{left_cell}{right_cell}</tr>')
            i += 1
        elif line.startswith("  "):  # Unchanged line
            content = format_content(line[2:])
            left_cell = f'<td>{content}</td>'
            right_cell = f'<td>{content}</td>'
            table_rows.append(f'<tr>{left_cell}{right_cell}</tr>')
            i += 1
        elif line.startswith("? "):  # Hint line (skip)
            i += 1
        else:
            i += 1  # Skip any other lines
    
    # Define CSS styles for the table-based diff view
    diff_styles = """
    <style>
        .diff-table {
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
        }
        .diff-table th, .diff-table td {
            padding: 8px;
            text-align: left;
            vertical-align: top;
            word-wrap: break-word;
            white-space: pre-wrap;
            font-family: inherit;
            border: 1px solid #ddd;
        }
        .diff-table th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        .removed { background-color: #f7b6b6; }
        .added { background-color: #b6f7b6; }
    </style>
    """

    # Create HTML table structure
    html_output = f"""
    {diff_styles}
    <table class="diff-table">
        <thead>
            <tr>
                <th>{version1_name}</th>
                <th>{version2_name}</th>
            </tr>
        </thead>
        <tbody>
            {"".join(table_rows)}
        </tbody>
    </table>
    """

    # Display the HTML
    st.markdown(html_output, unsafe_allow_html=True)

@traceable(
    metadata={
        "agent_name": "course_outline",
        "step_name": "Create the Final Outline Sheet",
        "function_name": "create_final_outline_sheet",
        "user_id": st.session_state.get("role", "anonymous")
    }
)
def create_final_outline_sheet(sheet):
    """
    Creates a 'Final Outline' sheet by reading from one of the source sheets depending on the Outline Stage and splits multiple LOs into separate rows.
    - If Outline Stage is 'Final' - use 'Base Outline' sheet and use standard column names.
    - If Outline Stage is 'Initial' - use 'Enhanced Outline with LOs' or fallback to 'Course Outline with LOs', and use column names with 'before checklist step'.
    """
    # Step 1: Check Outline Stage
    try:
        course_info_ws = sheet.worksheet("Course info")
        course_info_df = pd.DataFrame(course_info_ws.get_all_records())
        course_info_df.columns = [col.strip() for col in course_info_df.columns]

        # Use the first non-empty Outline Stage value
        outline_stage = course_info_df["Outline Stage"].dropna().astype(str).str.strip().str.lower().iloc[0]
    except Exception as e:
        raise ValueError(f" Failed to read 'Outline Stage' from 'Course Info' sheet: {e}")

    # Step 2: Choose source worksheet
    source_sheet_name = None
    if outline_stage == "final":
        source_sheet_name = "Base Outline"
        print(" Using 'Base Outline' as source since Outline Stage is 'Final'")
    else:
        try:
            sheet.worksheet("Enhanced Outline with LOs")
            source_sheet_name = "Enhanced Outline with LOs"
            print(" Using 'Enhanced Outline with LOs' as source since Outline Stage is 'Initial'")
        except:
            try:
                sheet.worksheet("Course Outline with LOs")
                source_sheet_name = "Course Outline with LOs"
                print(" Using 'Course Outline with LOs' as source since Outline Stage is 'Initial'")
            except:
                raise ValueError(" No valid source outline sheet found.")

    # Step 3: Load and normalize
    worksheet = sheet.worksheet(source_sheet_name)
    df = pd.DataFrame(worksheet.get_all_records())
    df.columns = [col.strip() for col in df.columns]

    if not all(col in df.columns for col in ["Topic", "Subtopic", "Learning Objectives"]):
        raise ValueError(f" Required columns missing in '{source_sheet_name}': 'Topic', 'Subtopic', 'Learning Objectives'.")

    # Step 4: Determine final column names
    if outline_stage == "final":
        topic_col = "Topic"
        subtopic_col = "Subtopic"
        lo_col = "Learning Objectives"
    else:
        topic_col = "Topic before checklist step"
        subtopic_col = "Subtopic before checklist step"
        lo_col = "Learning Objectives before checklist step"

    # Step 5: Split LOs into individual rows
    final_rows = []
    for _, row in df.iterrows():
        topic = row["Topic"]
        subtopic = row["Subtopic"]
        los_raw = row["Learning Objectives"]
        if pd.isna(los_raw):
            continue
        los = [lo.strip() for lo in str(los_raw).split("\n") if lo.strip()]
        for lo in los:
            final_rows.append({
                topic_col: topic,
                subtopic_col: subtopic,
                lo_col: lo
            })

    # Step 6: Write output
    final_df = pd.DataFrame(final_rows)
    final_ws, _ = create_or_read_worksheet(sheet, "Final Outline")
    save_to_sheet(worksheet=final_ws, df=final_df)
    format_worksheet(final_ws)

    print(" Final Outline sheet created successfully.")
    return True


def delete_final_outline(sheet, worksheet_name="Final Outline"):
    """Delete the Final Outline worksheet."""
    delete_worksheet(sheet, worksheet_name)
