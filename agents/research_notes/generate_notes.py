from modules.chain import Chain
from services.helper_functions import get_outline_with_los
from tqdm import tqdm
from services.sheets_service import get_sheet_data_and_df
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
from services.smart_progress_bar import SmartProgressBar


generate_research_notes_prompt = """You are an expert educational content developer tasked with creating comprehensive research notes for a specific subtopic within a larger course. Your goal is to produce well-structured, engaging, and educational notes that align precisely with the given learning objectives while considering the overall course structure and target audience.

Before we begin, please review the following course information:

Relevant Documents:
<relevant_documents>
{relevant_documents}
</relevant_documents>

Course Name:
<course_name>
{course_name}
</course_name>

Target Audience:
<target_audience>
{target_audience}
</target_audience>

Full Course Outline with Learning Objectives:
<full_course_outline>
{full_course_outline}
</full_course_outline>

Subtopic and Learning Objectives to Focus On:
<subtopic_and_los>
{subtopic_and_los}
</subtopic_and_los>

Now, follow these steps to generate the research notes. For each step, wrap your work inside the specified XML tags (i.e - objective_analysis, examine_documents, determine_information, propose_framework, create_notes, and review_and_refine) to show your work:

1. <objective_analysis>
   - Carefully review the learning objectives for the specific subtopic.
   - List each learning objective and break it down into key concepts and skills students should master.
   - Consider how these objectives fit into the broader context of the course.
</objective_analysis>

2. <examine_documents>
   - Thoroughly read and analyze the provided relevant documents.
   - Quote key passages that directly support the learning objectives (in case of too many docs, very large quotes, too many quotes, it is okay to present summarized versions.)
   - Explain the significance of each quoted passage in relation to the learning objectives.
   - Note any examples, definitions, or explanations that could enhance understanding.
</examine_documents>

3. <determine_information>
   - Create a table matching selected information to specific learning objectives.
   - Review the full course outline to identify topics that will be covered in future sections.
   - Exclude information that is more appropriate for later sections of the course.
   - Ensure all selected information is derived from the relevant documents; do not add external information.
</determine_information>

4. <propose_framework>
   - Create a logical structure for the notes that aligns with the learning objectives.
   - Organize the main points and sub-points in a clear, hierarchical manner.
   - Number each main point and sub-point to ensure a clear hierarchy.
   - Ensure the framework provides a comprehensive overview of the subtopic.
   - Consider how to make the structure engaging and interesting for the target audience.
</propose_framework>

5. <create_notes>
   - Fill in the proposed framework with detailed information from the relevant documents.
   - Write in clear, concise paragraphs appropriate for the target audience.
   - Ensure each paragraph directly supports one or more of the learning objectives.
   - Include relevant examples, definitions, and explanations as needed.
   - Maintain a logical flow of information throughout the notes.
   - Focus on making the content interesting, engaging, and educational.
</create_notes>

6. <review_and_refine>
   - Review the notes to ensure they fully address all learning objectives.
   - Check that all information is derived from the relevant documents.
   - Verify that the content is comprehensive, engaging, and educational.
   - Make any necessary refinements to improve clarity, flow, or alignment with objectives.
</review_and_refine>


Remember:
- Focus solely on the specific subtopic and learning objectives provided.
- Consider the broader context of the course and the needs of the target audience.
- Ensure that your research notes are comprehensive, well-structured, and directly aligned with the learning objectives.
- Write in clear, engaging paragraphs that will interest and educate the target audience.
- Do not add any information that is not derived from the provided relevant documents.
"""


def generate_research_notes(course_name, target_audience, course_outline, subtopic_and_los, relevant_documents, llm = 'groq'):
    """
    This function generates research notes for a specific subtopic.

    :param course_name: The name of the course.
    :param target_audience: The target audience of the course.
    :param course_outline: The outline of the course.
    :param subtopic_and_los: The subtopic and learning objectives to focus on.
    :param relevant_documents: The relevant documents.
    :param llm: The language model to use
    :return: The generated research notes.
    """

    generate_research_notes_agent = Chain(llm = llm, tags = ['create_notes'])

    generate_research_notes_agent.add_message(
        role = 'user',
        content = generate_research_notes_prompt.format(
            course_name = course_name,
            target_audience = target_audience,
            full_course_outline = course_outline,
            subtopic_and_los = subtopic_and_los,
            relevant_documents = relevant_documents
        )
    )

    response = generate_research_notes_agent.run()

    return response['create_notes']


def run_research_notes_agent_for_all_rows(sheet, worksheet_name, course_name, target_audience, llm='groq'):
    """
    This function runs the research notes agent for all rows in the sheet.

    :param sheet: The sheet object.
    :param worksheet_name: The worksheet name.
    :param course_name: The name of the course.
    :param target_audience: The target audience of the course.
    :param llm: The language model to use.
    :return: None
    """

    # Read the sheet and df
    course_outline_with_lo_sheet, course_outline_with_lo_df = get_sheet_data_and_df(
        sheet=sheet, 
        sheet_name=worksheet_name
    )

    # Create column for research notes if not already present
    if 'research_notes' not in course_outline_with_lo_df.columns:
        course_outline_with_lo_df['research_notes'] = ''

    # Check if this step is already done by checking last row of research notes column
    if course_outline_with_lo_df.iloc[-1]['research_notes'] != '':
        print('Research notes already populated')
        return

    # Get context column count
    context_col_count = len([col for col in course_outline_with_lo_df.columns if 'context_' in col])

    # Get the course outline with lo
    course_outline_with_lo = get_outline_with_los(
        df=course_outline_with_lo_df,
        include_learning_objectives=True
    )

    # Prepare for parallel processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit tasks for each row
        for index, row in course_outline_with_lo_df.iterrows():
            # Check if row already populated
            if row['research_notes'] != '':
                print(f'Skipping row {index}. Already populated')
                continue

            # Construct the context by joining all the context_n values
            context = ''.join(
                [row[f'context_{i}'] for i in range(context_col_count)]
            )

            # Submit the task
            future = executor.submit(
                generate_research_notes,
                course_name=course_name,
                target_audience=target_audience,
                course_outline=course_outline_with_lo,
                subtopic_and_los=row['Subtopic'] + '\n\n' + row['Learning Objectives'],
                relevant_documents=context,
                llm=llm,
            )

            # Map the Future to the index
            futures_map[future] = index

        # Collect the results as they complete
        total_tasks = len(futures_map)
        save_interval = 5  # how often to save (in number of completed tasks)

        # Initialize the progress tracker
        progress = SmartProgressBar(total_tasks = total_tasks, description = "Percent complete", save_interval = save_interval)

        for future in tqdm(as_completed(futures_map), total=total_tasks):
            index = futures_map[future]
            research_notes = future.result()

            # Update the df row with research notes
            course_outline_with_lo_df.loc[index, 'research_notes'] = research_notes

            # Update progress
            progress.update()

            # Check if we should save
            if progress.should_save():
                print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
                # Convert all columns to string to avoid data-type issues
                course_outline_with_lo_df = course_outline_with_lo_df.astype(str)
                course_outline_with_lo_sheet.update(
                    [course_outline_with_lo_df.columns.values.tolist()] +
                    course_outline_with_lo_df.values.tolist()
                )


    # Final save to sheet after all tasks
    print('All rows processed. Saving final DataFrame to sheet.')
    course_outline_with_lo_df = course_outline_with_lo_df.astype(str)
    course_outline_with_lo_sheet.update(
        [course_outline_with_lo_df.columns.values.tolist()] +
        course_outline_with_lo_df.values.tolist()
    )

    return



