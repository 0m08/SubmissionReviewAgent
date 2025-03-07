from modules.chain import Chain
from services.sheets_service import get_sheet_data_and_df, save_to_sheet
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.smart_progress_bar import SmartProgressBar
from services.helper_functions import validate_column_values, get_outline_with_los


summarize_prompt = """We are developing a course on {course_name}.

Your task is to summarize the following information in such a way that no information is lost.
<info_to_summarize>
{info_to_summarize}
</info_to_summarize>

Aim to summarize this information such that little to no critical information wrt the course is lost. At the same time, aim to create a summary that is clear, concise, and free from any repititions.

Output in the following format:
<output>
<scratchpad>
[Place to think through before you respond with actual summary. Use this space to think step by step.]
</scratchpad>
<summary>
[Actual summary]
</summary>
</output>
"""


review_research_summary_prompt = """You are tasked with reviewing and revising a research summary for a course. Your goal is to ensure that the summary only contains relevant and appropriate information based on the course details provided. Follow these steps carefully:

1. Review the course information:

<course_name>
{course_name}
</course_name>

<target_audience>
{target_audience}
</target_audience>

<course_outline>
{course_outline}
</course_outline>

2. Now, examine the research summary:

<research_summary>
{research_summary}
</research_summary>

3. Analyze the research summary for relevance and appropriateness based on the course name, target audience, and outline. Consider the following criteria:
   - Relevance to the course topic
   - Appropriateness for the target audience
   - Alignment with the course outline
   - Scope (whether the information is too broad or too narrow)

4. Identify any content in the research summary that is:
   - Irrelevant to the course topic
   - Inappropriate for the target audience
   - Not aligned with the course outline
   - Out of scope (too broad or too detailed)
   - Redundant or repetitive
   - Factually incorrect or outdated

   List this unwanted content in <unwanted_content> tags, providing a brief reason for each item's exclusion.

5. Create a revised summary by removing all the unwanted content identified in step 4. Do not add any new information or modify the remaining content. Retain everything else as it is in the original summary.

6. Output your results in the following format:

<analysis>
[Do your initial analysis of the research summary based here]
</analysis>

<unwanted_content>
[List the unwanted content here, with reasons for exclusion]
</unwanted_content>

<revised_summary>
[Place the revised research summary here, with all unwanted content removed]
</revised_summary>

Remember, your task is to remove unfit content only. Do not add new information or alter the remaining content in any way.
"""


def summarize_info(course_name, info_to_summarize, llm = 'gemini_2_flash'):
    """
    Summarize the information
        :param info_to_summarize (str): The information to summarize
        :param llm (str): The language model to use
        :return response (str): The summarized information
    """
    summarize_agent = Chain(llm = llm, tags = ['scratchpad', 'summary'])

    summarize_agent.add_message(
        role = 'user', content = summarize_prompt.format(
            course_name = course_name,
            info_to_summarize = info_to_summarize
        )
    )

    response = summarize_agent.run()
    return response


def run_get_research_summary(sheet, worksheet_name, course_name, llm='gemini_2_flash'):
    """
    Generates a research summary for each search query in the rough outline.

    :param sheet (object): Google Sheets object.
    :param worksheet_name (str): Name of the worksheet.
    :param course_name (str): Name of the course.
    :param llm (str): Language model to use (default: 'gemini_2_flash').
    :returns: None
    """

    # Load rough outline and preliminary research data
    rough_outline_sheet, rough_outline_df = get_sheet_data_and_df(sheet, worksheet_name)
    _, preliminary_research_df = get_sheet_data_and_df(sheet, "Preliminary Research")

    # Ensure required columns exist
    if 'research_summary' not in rough_outline_df.columns:
        rough_outline_df['research_summary'] = ''
        rough_outline_df['Manual Extract'] = ''

    # # Skip processing if the last row is already populated
    # if rough_outline_df['research_summary'].iloc[-1].strip():
    #     print("Skipping processing: Research summary is already populated.")
    #     return rough_outline_df
    try:
        validate_column_values(
            df = rough_outline_df,
            filter_column = 'search_queries',
            validation_column = 'research_summary',
            require_populated = True
        )
        print("Skipping processing: Research summary is already populated.")
    except:
        pass

    # Prepare for parallel processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers = 5) as executor:
        # Submit tasks for each row
        for index, row in rough_outline_df.iterrows():

            if row['research_summary'] != '':
                print(f"Skipping {index}: Research summary already populated.")
                continue

            query = row['search_queries']
            if not query.strip():
                print(f"Skipping {index}: Empty search query.")
                continue

            # Extract relevant learning objectives            
            temp_df = preliminary_research_df[preliminary_research_df['query'] == query]
            los = temp_df[temp_df['learning_objectives'].str.contains('<objective>')]['learning_objectives'].to_list()
            info_to_summarize = '\n'.join(los).replace('<objective>', '').replace('</objective>', '').replace('{', '{{').replace('}', '}}')

            # Submit the task
            future = executor.submit(
                summarize_info,
                course_name,
                info_to_summarize,
                llm
            )

            # Map the Future to the index
            futures_map[future] = index

        # Collect the results as they complete
        total_tasks = len(futures_map)
        save_interval = 5  # how often to save (in number of completed tasks)

        # Initialize the progress tracker
        progress = SmartProgressBar(total_tasks = total_tasks, description = "Percent complete", save_interval = save_interval)

        # Now, pass only the futures (the keys) to as_completed:
        for future in tqdm(as_completed(futures_map), total=total_tasks):
            index = futures_map[future]  # retrieve the index
            response = future.result()

            rough_outline_df.loc[index, 'research_summary'] = response['summary']

            # Update progress
            progress.update()

            # Check if we should save
            if progress.should_save():
                print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
                save_to_sheet(worksheet = rough_outline_sheet, df = rough_outline_df)

    # Final save to sheet after all tasks
    print('All rows processed. Saving final DataFrame to sheet.')
    save_to_sheet(worksheet = rough_outline_sheet, df = rough_outline_df)

    return


def review_research_summary(course_name, target_audience, course_outline, research_summary, llm = 'gemini_2_flash'):
    """
    Review the research summary and return the revised summary
    :param course_name: The course name.
    :param target_audience: The target audience.
    :param course_outline: The course outline
    :param research_summary (str): The research summary to review
    :param llm (str): The language model to use
    :returns response (str): The revised research summary
    """

    review_research_summary_agent = Chain(llm = llm, tags = ['revised_summary'])

    review_research_summary_agent.add_message(
        role = 'user', content = review_research_summary_prompt.format(
            course_name = course_name,
            target_audience = target_audience,
            course_outline = course_outline,
            research_summary = research_summary
        )
    )

    response = review_research_summary_agent.run()
    return response['revised_summary']


def run_review_research_summary_for_all_rows(sheet, worksheet_name, course_name, target_audience, llm = "gemini_2_flash"):
    """
    Runs the review of research summary for all search query rows in the rough outline.

    :param sheet (object): Google Sheets object.
    :param worksheet_name (str): Name of the worksheet.
    :param course_name: The course name.
    :param target_audience: The target audience.
    :param llm (str): Language model to use (default: 'gemini_2_flash').
    :returns: None
    """

    # Load rough outline and preliminary research data
    rough_outline_sheet, rough_outline_df = get_sheet_data_and_df(sheet, worksheet_name)

    # Get the course outline
    course_outline = get_outline_with_los(
        df = rough_outline_df,
        include_learning_objectives = False
    )

    # Prepare for parallel processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers = 5) as executor:
        # Submit tasks for each row
        for index, row in rough_outline_df.iterrows():
            
            research_summary = row['research_summary']
            # Skip blank rows 
            if research_summary == '':
                # print(f"Skipping {index}: Blank Research summary row.")
                continue

            # Skip if Manual Extract already populated
            if row['Manual Extract'] != '':
                continue

            # Submit the task
            future = executor.submit(
                review_research_summary,
                course_name,
                target_audience,
                course_outline,
                research_summary,
                llm
            )

            # Map the Future to the index
            futures_map[future] = index

        # Collect the results as they complete
        total_tasks = len(futures_map)
        save_interval = 5  # how often to save (in number of completed tasks)

        # Initialize the progress tracker
        progress = SmartProgressBar(total_tasks = total_tasks, description = "Percent complete", save_interval = save_interval)

        # Now, pass only the futures (the keys) to as_completed:
        for future in tqdm(as_completed(futures_map), total=total_tasks):
            index = futures_map[future]  # retrieve the index
            revised_summary = future.result()

            rough_outline_df.loc[index, 'Manual Extract'] = revised_summary

            # Update progress
            progress.update()

            # Check if we should save
            if progress.should_save():
                print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
                save_to_sheet(worksheet = rough_outline_sheet, df = rough_outline_df)

    # Final save to sheet after all tasks
    print('All rows processed. Saving final DataFrame to sheet.')
    save_to_sheet(worksheet = rough_outline_sheet, df = rough_outline_df)

    return


def manual_input_review_research_summary(sheet, worksheet_name, course_name, target_audience, skip_manual_step = False, llm = "gemini_2_flash"):
    """
    Checks whether the user has properly reviewed the research summary.

    :param sheet: The Google Sheets object.
    :param worksheet_name: The name of the worksheet.
    :param course_name: The course name.
    :param target_audience: The target audience.
    :param skip_manual_step: Bool. If True, it will use ai to perform this step instead.
    :return: True if column is properly populated, raises an error otherwise.     
    """

    if skip_manual_step:
        # Automatically fill the column with help of AI
        run_review_research_summary_for_all_rows(
            sheet = sheet,
            worksheet_name = worksheet_name,
            course_name = course_name,
            target_audience = target_audience,
            llm = llm
        )
        return True

    # Get the sheet and DataFrame
    rough_outline_sheet, rough_outline_df = get_sheet_data_and_df(sheet, worksheet_name)

    # Check if user has properly added inputs - for consolidation_comments columns
    validate_column_values(
        df = rough_outline_df,
        filter_column = "research_summary",
        validation_column = "Manual Extract",
        require_populated = True
    )

    return True


