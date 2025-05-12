from modules.chain import Chain
from services.helper_functions import get_outline_with_los
from tqdm import tqdm
from services.sheets_service import get_sheet_data_and_df, save_to_sheet
from agents.research_notes.review_topic_notes import review_topic_notes_prompt
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
from services.smart_progress_bar import SmartProgressBar
from langsmith import traceable


propose_revised_notes_framework_prompt = """Your next task is to define a framework for the research notes.

Output in the following format:
<planning>
[Plan your framework within these tags.]
</planning>
<proposed_framework>
[Place your proposed framework here. This should include:
- Main headings: Create 3–5 major sections or modules
- Subheadings: Under each main heading, list specific topics, concepts, or skills.

Hierarchy & Clarity: Ensure the top-level structure captures the main pillars of your topic, while subheadings handle finer details.
Storytelling Flow: Think of the framework like a story—each section should naturally lead to the next.]
</proposed_framework>
<no_of_sections>
[Enter the no. of sections (main headings). This should be a number and nothing else.]
</no_of_sections>
"""


revise_topic_notes_prompt = """Your next task is to revise the research notes based on the suggested framework. We will generate the revised research notes one section at a time.

Proceed with section no: {section_no}

Output in the following format:
<section_no>
[Enter the section number here. This should be a number and nothing else.]
</section_no>
<section_notes>
[Place the corresponding section's revised research notes here. Structure them as follows: transition slide > multiple content slides > summary slide]
<transition_slide>
Slide Title: [Place title of the slide here]
Slide Content: [Place content of the slide here. This is supposed to create the transition of going into the subtopic / section. A line-two liner.]
</transition_slide>
<content_slides>
<slide>
Slide Title: [Place title of the slide here]
Slide Content: [Place content of the slide here. These slides cover the main content. Paragraph, narration style.]
</slide>
[Repeat above slide block for multiple times to fit in all the content]
</content_slides>
<summary_slide>
Slide Title: [Place title of the slide here]
Slide Content: [Place content of the slide here. This is supposed to revise the key learnings from this subtopic / section. 3-5 bullets.]
</summary_slide>
</section_notes>
"""

@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "revise_topic_notes",
    "function_name": "revise_topic_notes",
    "user_id": st.session_state.get("role", "anonymous")
})
def revise_topic_notes(course_name, target_audience, course_outline, topic_focus, research_notes, ai_review, manual_comments, llm = 'groq'):
    """
    This function reviews the topic notes.

    :param course_name: The name of the course.
    :param target_audience: The target audience of the course.
    :param course_outline: The outline of the course.
    :param topic_focus: The topic focus.
    :param research_notes: The research notes.
    :param ai_review: The AI review.
    :param manual_comments: The manual comments.
    :param llm: The language model to use.
    :return: The review of the topic notes.
    """

    revise_topic_notes_agent = Chain(llm = llm, tags = ['no_of_sections'])

    # Build the chain of messages - review > propose framework > loop of section note generation
    # Review message
    revise_topic_notes_agent.add_message(
        role = 'user',
        content = review_topic_notes_prompt.format(
            course_name = course_name,
            target_audience = target_audience,
            course_outline = course_outline,
            topic_focus = topic_focus,
            research_notes = research_notes
        )
    )
    # AI Review message
    revise_topic_notes_agent.add_message(
        role = 'ai',
        content = f'{ai_review}\n\nManual comments from the user (these take precedence over all the ai review incase of conflicts):\n{manual_comments}'
    )

    # Add the propose framework message
    revise_topic_notes_agent.add_message(
        role = 'user',
        content = propose_revised_notes_framework_prompt
    )

    # Run to get the framework for the topic
    framework_response = revise_topic_notes_agent.run()

    # Extract no of sections inside the generated framework
    no_of_sections = framework_response['no_of_sections']
    # Check if it is digit or not, else raise error
    if no_of_sections.isdigit():
        no_of_sections = int(no_of_sections)
    else:
        raise Exception('No of sections is not a number')

    # Set the tags to now extract section notes > these contain the actual research notes for this topic.
    revise_topic_notes_agent.tags = ['section_notes']

    # List to store the generated sections
    section_notes = []
    # Iterate over the no of sections to generate research notes for all sections within a Topic
    for section_no in range(1, no_of_sections + 1):
        print("-"*100)
        print(f"Generating Section No: {section_no}")
        print("-"*100)
        # Add reviser message
        revise_topic_notes_agent.add_message(
            role = 'user',
            content = revise_topic_notes_prompt.format(
                section_no = section_no,
            )
        )
        # Run the agent to get current section's research notes
        reviser_response = revise_topic_notes_agent.run()
        section_notes.append(reviser_response['section_notes'])


    return framework_response['text'], '\n\n---\n\n'.join(section_notes)

@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "revise_topic_notes",
    "function_name": "run_revise_topic_notes_for_all_rows",
    "user_id": st.session_state.get("role", "anonymous")
})
def run_revise_topic_notes_for_all_rows(sheet, worksheet_name, course_name, target_audience, llm = 'groq'):
    """
    This function runs the revise topic notes agent for all rows in the sheet.

    :param sheet: The sheet object.
    :param worksheet_name: The worksheet name.
    :param course_name: The name of the course.
    :param target_audience: The target audience of the course.
    :param llm: The language model to use.
    :return: None
    """

    # Read the sheet and df
    research_notes_sheet, research_notes_df = get_sheet_data_and_df(sheet = sheet, sheet_name = worksheet_name)

    # Create framework col if not already present
    if 'framework' not in research_notes_df.columns:
        research_notes_df['framework'] = ''
        research_notes_df['section_notes'] = ''

    # Check if this step is already done by checking last row of final_review column
    if research_notes_df.iloc[-1]['framework'] != '':
        print('Framework already populated')
        return

    # Get the course outline with lo
    course_outline = get_outline_with_los(
        df = research_notes_df,
        include_learning_objectives = False
    )

    research_notes_col_count = len([col for col in research_notes_df.columns if 'research_notes_' in col])

    # Prepare for parallel processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit tasks for each row
        for index, row in research_notes_df.iterrows():

            # Skip if framework already populated
            if row['framework'] != '':
                print(f'Skipping row {index}. Already populated')
                continue
            
            research_notes = ''.join(
                    [str(row[f'research_notes_{i}']) for i in range(research_notes_col_count)]
                ).strip()

            # Submit the task
            future = executor.submit(
                revise_topic_notes,
                course_name = course_name,
                target_audience = target_audience,
                course_outline = course_outline,
                topic_focus = row['Topic'],
                research_notes = research_notes,
                ai_review = row['final_review'],
                manual_comments = row['Manual Comments'],
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
            framework, section_notes = future.result()

            # Update the df row
            research_notes_df.loc[index, 'framework'] = framework
            research_notes_df.loc[index, 'section_notes'] = section_notes

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


