from services.llm_service import csv_list_parser
from modules.chain import Chain
import pandas as pd
from services.sheets_service import get_sheet_data_and_df
from services.helper_functions import get_outline_with_los

### Construct Video Search Queries

generate_video_search_queries_prompt = """You are tasked with generating a list of search queries to find relevant YouTube videos for a given course. These queries will be used to search for educational content that aligns with the course outline and is suitable for the target audience.

You will be provided with the following information:

<course_name>
{course_name}
</course_name>

<target_audience>
{target_audience}
</target_audience>

<course_outline>
{course_outline}
</course_outline>

First, carefully analyze the course outline within <analysis> tags. Identify the main topics and subtopics. Consider the depth and breadth of each topic based on its position in the outline hierarchy.

Next, generate a list of search queries based on the following guidelines:
1. Create at least one query for each main topic in the outline.
2. For complex or broad topics, create additional queries for important subtopics.
3. Incorporate the course name and target audience into some of the queries to find more specific and relevant content.
4. Use a mix of broad and specific queries to ensure comprehensive coverage.
5. Include relevant keywords, phrases, and concepts from the outline in your queries.
6. Consider adding terms like "tutorial," "lecture," "explanation," or "introduction" to some queries to find instructional content.
7. Aim for a total of 10-20 unique search queries, depending on the complexity of the course outline.

Present your list of search queries (unnumbered) in the following format:

<search_queries>
[First search query]
[Second search query]
[Third search query]
...
</search_queries>


Remember to tailor the language and complexity of your search queries to match the target audience's level of understanding.
"""


# Define a custom output parser
def extract_csv_lines(text: str):
    """
    Function to extract csv lines from text
    :param: text (str): Input text
    :returns: csv_lines (list): List of csv lines
    """
    # Try with simple line split, since csv splitter doesn't handle commas in between search query
    try:
        lines = text.strip().split('\n')
        return [line.strip(',').strip() for line in lines]
    except KeyboardInterrupt:
        print('User stopped action')
    except:
        return csv_list_parser.parse(text)


def generate_video_search_queries(course_name, target_audience, course_outline, llm):
    """
    Function to generate search queries
    
    :param: course_name (str): Course name
    :param: target_audience (str): Target audience
    :param: course_outline (str): Course outline
    :param: llm (str): Language model to use
    :returns: search_queries (list): List of search queries
    """
    generate_query_agent = Chain(llm = llm, tags = ['search_queries'])
    generate_query_agent.add_message(
        role = "user",
        content = generate_video_search_queries_prompt.format(
            course_name = course_name,
            target_audience = target_audience,
            course_outline = course_outline
            )
        )
    response = generate_query_agent.run()
    search_queries = extract_csv_lines(response['search_queries'])

    return search_queries


def create_additional_video_search_queries(rough_outline_df):
    """
    Function to create additional search queries by combining topic - subtopic keywords

    :param: rough_outline_df (pd.DataFrame): Rough outline dataframe
    :returns: additional_video_search_queries (list): List of additional search queries
    """

    additional_video_search_queries = []

    for topic, subtopic in zip(rough_outline_df['Topic'].to_list(), rough_outline_df['Subtopic'].to_list()):
        if topic == '' or subtopic == '':
            continue
        additional_video_search_queries.append(f"{topic} - {subtopic}")

    return additional_video_search_queries


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

    # Case 1: If the original DataFrame has as many (or more) rows than the list
    if n_original >= n_new_values:
        # Directly set the new column (missing rows, if any, become NaN automatically)
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


def run_construct_video_search_queries(sheet, course_name, target_audience, worksheet_name = 'Rough Outline', llm = 'groq'):
    """
    This function creates a new worksheet if not already present. If present, it reads the sheet

    :param sheet: The sheet object.
    :param course_name: The course name.
    :param target_audience: The target audience.
    :param worksheet_name: The worksheet name.
    :param llm: The language model to use.
    :return: worksheet, df
    """

    rough_outline_sheet, rough_outline_df = get_sheet_data_and_df(sheet, worksheet_name)

    course_outline = get_outline_with_los(rough_outline_df, True)

    # Check if video search queries already present in the df or not.
    if 'video_search_queries' not in rough_outline_df.columns:
        # Generate search queries with llm
        video_search_queries = generate_video_search_queries(
            course_name = course_name,
            target_audience = target_audience,
            course_outline = course_outline,
            llm = llm
        )
        # Generate search queries by combining topic - subtopic string.
        video_search_queries.extend(create_additional_video_search_queries(
            rough_outline_df = rough_outline_df
        ))
        # Add to df
        rough_outline_df = add_list_as_new_column(rough_outline_df, video_search_queries, 'video_search_queries')
        # Save to sheet
        rough_outline_df = rough_outline_df.astype(str)
        rough_outline_sheet.update([rough_outline_df.columns.values.tolist()] + rough_outline_df.values.tolist())

    else:
        # Already present, skip
        print('Column - video_search_queries already present. Skipping generate video search queries')


