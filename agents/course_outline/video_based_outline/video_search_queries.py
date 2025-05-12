from services.llm_service import extract_csv_lines
from modules.chain import Chain
import pandas as pd
import streamlit as st
from services.sheets_service import get_sheet_data_and_df, save_to_sheet
from services.helper_functions import get_outline_with_los, add_list_as_new_column, find_blank_followed_by_filled_indices
from langsmith import traceable
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


@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Video Search Query Generator",
    "function_name": "generate_video_search_queries",
    "user_id": st.session_state.get("role", "anonymous")
})
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

@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Video Search Query Generator",
    "function_name": "create_additional_video_search_queries",
    "user_id": st.session_state.get("role", "anonymous")
})
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



@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Video Search Query Generator",
    "function_name": "run_construct_video_search_queries",
    "user_id": st.session_state.get("role", "anonymous")
})
def run_construct_video_search_queries(sheet, course_name, target_audience, worksheet_name = 'Rough Outline', llm = 'groq'):
    """
    This function creates a list of search queries to be used to search for HVAC school youtube videos.

    :param sheet: The sheet object.
    :param course_name: The course name. 
    :param target_audience: The target audience.
    :param worksheet_name: The worksheet name.
    :param llm: The language model to use.
    :return: None
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
        save_to_sheet(worksheet = rough_outline_sheet, df = rough_outline_df)

    else:
        # Already present, skip
        print('Column - video_search_queries already present. Skipping generate video search queries')


def manual_input_review_video_search_queries(sheet, worksheet_name = 'Rough Outline'):
    """
    Manual input to review video search queries.
    :param sheet: The sheet object.
    :param worksheet_name: The worksheet name.
    """

    rough_outline_sheet, rough_outline_df = get_sheet_data_and_df(sheet, worksheet_name)

    return find_blank_followed_by_filled_indices(
        df = rough_outline_df,
        column_name = "video_search_queries",
        blank_value = ""
    )

