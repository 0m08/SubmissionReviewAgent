import pandas as pd
import math
from typing import List, Optional
import regex as re

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
    Get a short name for a text.
    Args:
        text (str): The text to get the short name for.
    Returns:
        str: The short name.
    """
    # Remove blanks, dots, and hyphens from text
    text = text.strip()
    text = text.lower()
    text = text.replace(' ', '_')
    text = text.replace('.', '')
    text = text.replace('-', '')

    # text no longer than 60 chars
    if len(text) > 60:
        text = text[:60]

    return text


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
    
    # Case 2: The new list is longer than the DataFrame’s row count
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
    Replaces only truly single '{' or '}' with double braces.
    Existing double/triple braces remain unchanged.
    
    Examples:
      A single { brace } -> A single {{ brace }}
      Double {{ braces }} -> (unchanged) -> {{ braces }}
      Triple {{{ braces }}} -> (unchanged) -> {{{ braces }}}
    """

    # Regex to find a left brace '{' that is NOT preceded or followed by another '{'
    SINGLE_LEFT_BRACE = re.compile(r'(?<!\{)\{(?!\{)')

    # Regex to find a right brace '}' that is NOT preceded or followed by another '}'
    SINGLE_RIGHT_BRACE = re.compile(r'(?<!\})\}(?!\})')
    
    # Replace single '{' with '{{'
    text = SINGLE_LEFT_BRACE.sub('{{', text)
    # Replace single '}' with '}}'
    text = SINGLE_RIGHT_BRACE.sub('}}', text)
    return text

