from modules.chain import Chain
from services.helper_functions import get_outline_with_los
from tqdm import tqdm
from services.sheets_service import get_sheet_data_and_df, save_to_sheet
import streamlit as st
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.smart_progress_bar import SmartProgressBar


review_topic_notes_prompt = """Your task is to review research notes generated for a given topic.

Here's the course related information:
<course_info>
Course name: {course_name}
Target audience: {target_audience}

Course outline:
{course_outline}
</course_info>

Here's the current topic and research notes to focus on:
<topic_focus>
{topic_focus}
</topic_focus>

<research_notes>
{research_notes}
</research_notes>

Your task is to analyze the research notes and identify all the issues with it.
Your review will be used as the base to restructure the notes, make edits, delete repetitions, etc.
Your review should not contain suggestions for additions. Our current focus is to make edits, deletions, restructuring; no additions.

Output your response in the following format:
<input_analysis>
[Place your analysis of the inputs with these tags. This should include:
- Your understanding of the course requirements.
- The scope of current topic based on target audience and course requirements.
- Summary, followed by a deep analysis of the current research notes.]
</input_analysis>
<final_review>
[Place your final review of the research notes with these tags.]
</final_review>
"""


def review_topic_notes(course_name, target_audience, course_outline, topic_focus, research_notes, llm = 'groq'):
    """
    This function reviews the topic notes.

    :param course_name: The name of the course.
    :param target_audience: The target audience of the course.
    :param course_outline: The outline of the course.
    :param topic_focus: The topic focus.
    :param research_notes: The research notes.
    :param llm: The language
    :return: The review of the topic notes.
    """

    review_topic_notes_agent = Chain(llm = llm, tags = ['final_review'])

    review_topic_notes_agent.add_message(
        role = 'user',
        content = review_topic_notes_prompt.format(
            course_name = course_name,
            target_audience = target_audience,
            course_outline = course_outline,
            topic_focus = topic_focus,
            research_notes = research_notes
        )
    )

    review_response = review_topic_notes_agent.run()

    return review_response['text']


def run_review_topic_notes_agent_for_all_rows(sheet, worksheet_name, course_name, target_audience, llm = 'groq'):
    """
    This function runs the review topic notes agent for all rows in the sheet.

    :param sheet: The sheet object.
    :param worksheet_name: The worksheet name.
    :param course_name: The name of the course.
    :param target_audience: The target audience of the course.
    :param llm: The language model to use.
    :return: None
    """

    # Read the sheet and df
    research_notes_sheet, research_notes_df = get_sheet_data_and_df(sheet = sheet, sheet_name = worksheet_name)

    # Create final review col if not already present
    if 'final_review' not in research_notes_df.columns:
        research_notes_df['final_review'] = ''
        research_notes_df['Manual Comments'] = ''

    # Check if this step is already done by checking last row of final_review column
    if research_notes_df.iloc[-1]['final_review'] != '':
        print('Final review already populated')
        return

    # Get the course outline with lo
    course_outline = get_outline_with_los(
        df = research_notes_df,
        include_learning_objectives = False
    )

    # Prepare for parallel processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit tasks for each row
        for index, row in research_notes_df.iterrows():
            
            # Skip if final review already populated
            if row['final_review'] != '':
                print(f'Skipping row {index}. Already populated')
                continue
            
            # Submit the task
            future = executor.submit(
                review_topic_notes,
                course_name = course_name,
                target_audience = target_audience,
                course_outline = course_outline,
                topic_focus = row['Topic'],
                research_notes = row['research_notes'],
                llm = llm
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
            final_review = future.result()

            # Update the df row with analysis
            research_notes_df.loc[index, 'final_review'] = final_review

            # Update progress
            progress.update()

            # Check if we should save
            if progress.should_save():
                print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
                save_to_sheet(worksheet = research_notes_sheet, df = research_notes_df)

    # Final save to sheet after all tasks
    print('All rows processed. Saving final DataFrame to sheet.')
    save_to_sheet(worksheet = research_notes_sheet, df = research_notes_df)

    return



def manual_input_review_topic_notes(sheet, worksheet_name):
    """
    This function allows the user to manually review the research notes.

    :param sheet: The sheet object.
    :param worksheet_name: The worksheet name.
    :return: None
    """

    # # Read the sheet and df
    # research_notes_sheet, research_notes_df = get_sheet_data_and_df(sheet = sheet, sheet_name = worksheet_name)

    # # Create a progress bar in the UI
    # total = research_notes_df.shape[0]
    # progress_bar = st.progress(0, text = "Percent complete: 0%")
    
    # for ind, row in tqdm(research_notes_df.iterrows(), total = total):
    #     time.sleep(5)

    #     # Update progress bar
    #     frac_complete = (ind + 1) / total
    #     progress_bar.progress(frac_complete, text = f"Percent complete: {int(frac_complete * 100)}%")

    # progress_bar.empty()

    # st.write(f"Review the research notes, and the ai generated review.")
    # st.write(f"Sheet Link - {research_notes_sheet.url}")
    # st.write(f"Column names:\n `research_notes`,\n `final_review`")
    # st.write("-"*100)
    # st.write("You can put any additional comments in the `Manual Comments` column.")
    # st.write("-"*100)
    # st.write("")
    # if st.button("Done with Edits? Click to continue"):
    #     return
    return True

