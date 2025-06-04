from modules.chain import Chain
from services.llm_service import extract_csv_lines
from services.sheets_service import get_sheet_data_and_df, save_to_sheet
from services.helper_functions import get_outline_with_los, add_list_as_new_column
from services.smart_progress_bar import SmartProgressBar
import streamlit as st
from langsmith import traceable


generate_search_queries_prompt = """You are tasked with generating multiple web search queries to gather resources for a course. You will be provided with the course name and course outline. Your goal is to create a list of search queries that will help retrieve relevant research material and content for the course.

Here is the course name:
<course_name>
{course_name}
</course_name>

Here is the target audience:
<target_audience>
{target_audience}
</target_audience>

And here is the course outline:
<course_outline>
{course_outline}
</course_outline>

Follow these guidelines to create effective search queries:

1. Use the course name and key topics from the outline as the base for your queries.
2. Incorporate relevant keywords and concepts from the course outline.
3. Utilize search operators such as AND, OR to narrow down the search scope.
4. Think about potential subtopics or related fields that might yield useful information.

Generate multiple search queries based on the course outline. Each query should focus on a specific aspect or topic of the course. Aim to cover all major topics and subtopics mentioned in the outline.

Present your list of search queries in a comma-separated format (CSV). Each query should be on a new line. Do not number the queries.


Output format: Structure your response as follows:
<output>
<scratchpad>
Use this section to brainstorm and outline your thought process before finalizing the search query.
</scratchpad>
<answer>
Provide your list of search queries within these tags.
</answer>
</output>

Avoid creating poorly constructed queries that lack context or are too general. Here are examples of what NOT to do:
<poor_examples>
Best practices AND techniques
Industry trends AND updates
Tools AND equipment
Common problems AND solutions
Certification AND training
</poor_examples>

These queries are too general and lack the specific context of the course. Always include relevant keywords from the course name or outline to provide necessary context.

Remember to avoid using double quotes or apostrophes in your queries, as they can be excessively restrictive.
"""


refine_search_queries_prompt = """You are tasked with reviewing and refining a list of web search queries generated for a specific course. Your goal is to ensure that these queries are effective in retrieving relevant research material and content for the course. You will be provided with the course name, course outline, and the initially generated queries.

Here is the course name:
<course_name>
{course_name}
</course_name>

Here is the target audience:
<target_audience>
{target_audience}
</target_audience>

Here is the course outline:
<course_outline>
{course_outline}
</course_outline>

And here are the initially generated queries:
<generated_queries>
{generated_queries}
</generated_queries>

Follow these guidelines to review and refine the search queries:
For individual query:
- Ensure each query is directly related to the course name and key topics from the outline.
- Confirm that each query has sufficient context and is not too general.


As you review, consider these examples of good and poor queries:

Good query examples:
- Flooring Care AND (hardwood OR laminate OR tile)
- Carpet Cleaning Techniques AND (stain removal OR deep cleaning)
- Floor Maintenance AND (commercial OR residential)

Poor query examples (too general, lack context):
- Best practices AND techniques
- Industry trends AND updates
- Tools AND equipment

For each query, decide whether to:
- Keep as is
- Modify to improve relevance or specificity
- Delete if redundant or irrelevant
- Add new queries to cover missing topics

Present your refined list of search queries in a comma-separated format (CSV). Each query should be on a new line. Do not number the queries.

Remember to avoid using double quotes or apostrophes in your queries, as they can be excessively restrictive.


Output format: Structure your response as follows:
<output>
<scratchpad>
<query_analysis>
Put all queries in a markdown table, with each query on a new line. The table should have following columns: Query, Search results (A sentence or two on what would the search results yeild), Review (based on guidelines), Enough context to get results related to {course_name} (Yes , No), Action to take (Keep, Modify, Delete)
</query_analysis>
<overall_analysis>
Use this section to check the queries as a whole:
- Check that all relevant keywords and concepts from the course outline are incorporated.
- Verify the proper use of search operators such as AND, OR to narrow down the search scope.
- Assess if all major topics and subtopics mentioned in the outline are covered.
- Look for and remove any redundant or overly similar queries.
- Add new queries if important topics from the outline are not represented.
</overall_analysis>
</scratchpad>
<answer>
Provide your updated list of search queries within these tags. Each query should be on a new line. Do not number the queries.
</answer>
</output>
"""


@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Web Search Queries Generator",
    "function_name": "generate_search_query",
    "user_id": st.session_state.get("role", "anonymous")
})
def generate_search_query(course_name, target_audience, course_outline, llm = 'gemini_2_flash'):
    """
    Generate search queries
    :param course_name: The course name.
    :param target_audience: The target audience.
    :param course_outline: The outline of the course.
    :param llm: The language model to use.    
    :return: list of search queries
    """

    generate_query_agent = Chain(llm = llm, tags = ['answer'])

    generate_query_agent.add_message(
        role = "user",
        content = generate_search_queries_prompt.format(
            course_name = course_name,
            target_audience = target_audience,
            course_outline = course_outline
        )
    )

    response = generate_query_agent.run()
    search_queries = extract_csv_lines(response['answer'])
    
    return search_queries


@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Web Search Queries Generator",
    "function_name": "refine_search_queries",
    "user_id": st.session_state.get("role", "anonymous")
})
def refine_search_queries(course_name, target_audience, course_outline, search_queries, llm = 'gemini_2_flash'):
    """
    Generate revised search queries
    :param course_name: The course name.
    :param target_audience: The target audience.
    :param course_outline: The outline of the course.
    :param search_queries: The list of search queries to revise
    :param llm: The language model to use.    
    :return: list of revised search queries
    """
    refine_search_queries_agent = Chain(llm = llm, tags = ['answer'])

    refine_search_queries_agent.add_message(
        role = "user",
        content = refine_search_queries_prompt.format(
            course_name = course_name,
            course_outline = course_outline,
            target_audience = target_audience,
            generated_queries = '\n'.join(search_queries)
            )
    )

    response = refine_search_queries_agent.run()
    refined_search_queries = extract_csv_lines(response['answer'])
    
    return refined_search_queries

@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Web Search Queries Generator",
    "function_name": "run_construct_web_search_queries",
    "user_id": st.session_state.get("role", "anonymous")
})
def run_construct_web_search_queries(sheet, worksheet_name, course_name, target_audience, llm='gemini_2_flash'):
    """
    This function creates a list of search queries to be used to search for HVAC school youtube videos.

    :param sheet: The sheet object.
    :param worksheet_name: The worksheet name.
    :param course_name: The course name.
    :param target_audience: The target audience.
    :param llm: The language model to use.
    :return: None
    """

    rough_outline_sheet, rough_outline_df = get_sheet_data_and_df(sheet, worksheet_name)

    course_outline = get_outline_with_los(rough_outline_df, True)

    # Check if the column exists
    if 'search_queries' in rough_outline_df.columns and rough_outline_df['search_queries'][0] != '':
        print("Web search queries already present. Skipping this step.")
        return

    # Initialize the progress tracker
    progress = SmartProgressBar(total_tasks = 2, description = "Percent complete")

    # Generate the search queries
    search_queries = generate_search_query(
        course_name = course_name, 
        target_audience = target_audience, 
        course_outline = course_outline, 
        llm = llm
    )

    # Update progress
    progress.update()

    # Revise the search queries
    refined_search_queries = refine_search_queries(
        course_name = course_name, 
        target_audience = target_audience, 
        course_outline = course_outline, 
        search_queries = search_queries, 
        llm = llm
    )

    # Update progress
    progress.update()

    # Add to df
    rough_outline_df = add_list_as_new_column(rough_outline_df, refined_search_queries, 'search_queries')

    # Save to sheet
    save_to_sheet(worksheet = rough_outline_sheet, df = rough_outline_df)

    return

