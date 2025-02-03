import pandas as pd
import math

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

