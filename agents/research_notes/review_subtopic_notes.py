from modules.chain import Chain
from services.helper_functions import get_outline_with_los
from tqdm import tqdm
from services.sheets_service import get_sheet_data_and_df, save_to_sheet
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.smart_progress_bar import SmartProgressBar


review_research_notes_prompt = """You are tasked with reviewing research notes for a specific subtopic of a course. Your goal is to determine if the subtopic is sufficiently covered based on the learning objectives (LOs) and the overall scope of the course. Follow these steps carefully:

1. Review the course information:
<course_name>
{course_name}
</course_name>

<target_audience>
{target_audience}
</target_audience>

2. Examine the full course outline to understand the overall context and scope:
<full_course_outline>
{full_course_outline}
</full_course_outline>

3. Focus on the specific subtopic and its learning objectives:
<subtopic_focus>
{subtopic_focus}
</subtopic_focus>

4. Carefully read the research notes generated for this subtopic:
<research_notes>
{research_notes}
</research_notes>

5. Analyze the coverage of the subtopic based on the learning objectives and the course scope. Consider the following categories:
   - Not covered at all
   - Not covered enough
   - Perfectly covered
   - Covered too much

6. In your analysis, categorize the concepts from the subtopic into these four categories. Think through your reasoning carefully, considering the depth and breadth of coverage for each concept in relation to the learning objectives and the overall course scope.

7. Present your analysis and categorization in the following format:

<analysis>
[Your detailed analysis goes here. Explain your reasoning for categorizing each concept, referring to the learning objectives, course scope, and content of the research notes.
First, present your understanding of the subtopic and its scope based on overall outline and target audience.
Then, summarize the research notes.
Then, formulate and answer a series of questions  to do a thorough analysis (eg. Are the definitions clear? Is there adequate detail on ...? Are there any missing pieces...? Does the depth match the target audience? Are there any parts that appear over-emphasized or excessively detailed?).]
</analysis>

<categorization>
<not_covered_at_all>
[List concepts that are not covered at all in the research notes]
</not_covered_at_all>

<not_covered_enough>
[List concepts that are not covered sufficiently in the research notes]
</not_covered_enough>

<perfectly_covered>
[List concepts that are covered adequately in the research notes]
</perfectly_covered>

<covered_too_much>
[List concepts that are covered in too much detail given the course scope]
</covered_too_much>
</categorization>

<verdict>
["PASS" or "FAIL"]
</verdict>

Ensure that your analysis is thorough and your categorization is well-justified based on the provided information. Incase no items belong to a certain category, leave it blank. Your goal is to provide a clear assessment of how well the research notes cover the subtopic in relation to the course's learning objectives and overall scope.
"""


def review_research_notes(course_name, target_audience, course_outline, subtopic_focus, research_notes, llm = 'groq'):
    """
    This function reviews the research notes for a specific subtopic.

    :param course_name: The name of the course.
    :param target_audience: The target audience of the course.
    :param course_outline: The outline of the course.
    :param subtopic_focus: The subtopic to focus on.
    :param research_notes: The research notes.
    :param llm: The language model to use
    :return: The analysis and categorization of the research notes.
    """

    review_research_notes_agent = Chain(llm = llm, tags = ['analysis', 'not_covered_at_all', 'not_covered_enough', 'perfectly_covered', 'covered_too_much', 'verdict'])

    review_research_notes_agent.add_message(
        role = 'user',
        content = review_research_notes_prompt.format(
            course_name = course_name,
            target_audience = target_audience,
            full_course_outline = course_outline,
            subtopic_focus = subtopic_focus,
            research_notes = research_notes
        )
    )

    response = review_research_notes_agent.run()

    return response


def run_reviewer_agent_for_all_rows(sheet, worksheet_name, course_name, target_audience, llm = 'groq'):
    """
    This function runs the reviewer agent for all rows in the sheet.

    :param sheet: The sheet object.
    :param worksheet_name: The worksheet name.
    :param course_name: The name of the course.
    :param target_audience: The target audience of the course.
    :param llm: The language model to use.
    :return: None
    """

    # Read the sheet and df
    course_outline_with_lo_sheet, course_outline_with_lo_df = get_sheet_data_and_df(sheet = sheet, sheet_name = worksheet_name)


    # Create column for analysis if not already present
    if 'analysis' not in course_outline_with_lo_df.columns:
        course_outline_with_lo_df['analysis'] = ''
        course_outline_with_lo_df['not_covered_at_all'] = ''
        course_outline_with_lo_df['not_covered_enough'] = ''
        course_outline_with_lo_df['perfectly_covered'] = ''
        course_outline_with_lo_df['covered_too_much'] = ''
        course_outline_with_lo_df['verdict'] = ''
        course_outline_with_lo_df['Manual Comments'] = ''

    # Check if this step is already done by checking last row of analysis column
    if course_outline_with_lo_df.iloc[-1]['analysis'] != '':
        print('Analysis already populated')
        return

    # Get the course outline with lo
    course_outline_with_lo = get_outline_with_los(
        df = course_outline_with_lo_df,
        include_learning_objectives = True
    )

    # Prepare for parallel processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit tasks for each row
        for index, row in course_outline_with_lo_df.iterrows():

            # Check if row already populated
            if row['analysis'] != '':
                print(f'Skipping row {index}. Already populated')
                continue

            # Submit the task
            future = executor.submit(
                review_research_notes,
                course_name=course_name,
                target_audience=target_audience,
                course_outline=course_outline_with_lo,
                subtopic_focus=row['Subtopic'] + '\n' + row['Learning Objectives'],
                research_notes=row['research_notes'],
                llm=llm,
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

            # Update the df row with analysis
            course_outline_with_lo_df.loc[index, 'analysis'] = response['analysis']
            course_outline_with_lo_df.loc[index, 'not_covered_at_all'] = response['not_covered_at_all']
            course_outline_with_lo_df.loc[index, 'not_covered_enough'] = response['not_covered_enough']
            course_outline_with_lo_df.loc[index, 'perfectly_covered'] = response['perfectly_covered']
            course_outline_with_lo_df.loc[index, 'covered_too_much'] = response['covered_too_much']
            course_outline_with_lo_df.loc[index, 'verdict'] = response['verdict']

            # Update progress
            progress.update()

            # Check if we should save
            if progress.should_save():
                print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
                save_to_sheet(worksheet = course_outline_with_lo_sheet, df = course_outline_with_lo_df)


    # Final save to sheet after all tasks
    print('All rows processed. Saving final DataFrame to sheet.')
    save_to_sheet(worksheet = course_outline_with_lo_sheet, df = course_outline_with_lo_df)

    return


def manual_input_review_research_notes(sheet, worksheet_name):
    """
    This function allows the user to manually review the research notes.

    :param sheet: The sheet object.
    :param worksheet_name: The worksheet name.
    :return: None
    """
    return True
    # Read the sheet and df
    # course_outline_with_lo_sheet, course_outline_with_lo_df = get_sheet_data_and_df(sheet = sheet, sheet_name = worksheet_name)

    # st.write(f"Review the research notes, and the ai generated review.")
    # st.write(f"Sheet Link - {course_outline_with_lo_sheet.url}")
    # st.write(f"Column names:\n `research_notes`,\n `analysis`,\n `not_covered_at_all`,\n `not_covered_enough`,\n `perfectly_covered`,\n `covered_too_much`,\n `verdict`")
    # st.write("-"*100)
    # st.write("Edit any of the category columns to incorporate your final review.")
    # st.write("You can put any additional comments in the `Manual Comments` column.")
    # st.write("-"*100)
    # st.write("")
    # # _ = input("Press enter to continue: ")
    # if st.button("Done with Edits? Click to continue"):
    #     st.session_state[step_key] = True
    #     return True
    # else:
    #     # raise Exception("Please make the manual edits and press the button to continue")
    #     st.info("Please make the manual edits and press the button to continue")
    #     return False
    



