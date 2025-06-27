from modules.chain import Chain
from services.helper_functions import get_outline_with_los
from tqdm import tqdm
from services.sheets_service import get_sheet_data_and_df, save_to_sheet, clear_worksheet
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
from services.smart_progress_bar import SmartProgressBar
from langsmith import traceable

generate_research_notes_prompt = """You are an expert educational content developer tasked with creating comprehensive research notes for a specific subtopic within a larger course. Your goal is to produce well-structured, engaging, and educational notes that align precisely with the given learning objectives while considering the overall course structure and target audience.

Before we begin, please review the following course information:

Relevant Documents/Transcripts:
<relevant_documents_or_transcripts>
{relevant_documents}
</relevant_documents_or_transcripts>

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

Note:
   - The input under <relevant_documents_or_transcripts> may consist of either:
     a) Traditional documents (e.g., articles, manuals, textbook excerpts), or
     b) Timestamped transcripts (e.g., from video or audio sources).
   - If transcripts are provided, ignore the timestamps and treat the text as a normal source of information.
   - Analyze transcript content just like any other document to extract relevant insights for the learning objectives.
   - Regardless of the format, follow the same structured approach for all steps below and maintain the same output format.

1. <objective_analysis>
   - Carefully review the learning objectives for the specific subtopic.
   - List each learning objective and break it down into key concepts and skills students should master.
   - Consider how these objectives fit into the broader context of the course.
</objective_analysis>

2. <examine_documents>
   - Thoroughly read and analyze the provided relevant documents or transcripts. If working with transcripts, ignore the timestamps and treat the content as a standard document.
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

@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Researcher",
    "function_name": "generate_research_notes",
    "user_id": st.session_state.get("role", "anonymous")
})
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

    # Check is research notes is under character limit
    if len(response['create_notes']) > 50000:
        print(f'Research notes are too long - {len(response["create_notes"])}. Summarizing...')
        generate_research_notes_agent.add_message(
            role = 'user',
            content = f"The research notes are too long. The current length is {len(response['create_notes'])} characters. Please summarize them to be under 50000 characters. Make sure to output in the same format as above."
        )
        response = generate_research_notes_agent.run()

    return response['create_notes'] if len(response['create_notes']) < 50000 else response['create_notes'][:49990]


@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Researcher",
    "function_name": "run_research_notes_agent_for_all_rows",
    "user_id": st.session_state.get("role", "anonymous")
})
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
                save_to_sheet(worksheet = course_outline_with_lo_sheet, df = course_outline_with_lo_df)


    # Final save to sheet after all tasks
    print('All rows processed. Saving final DataFrame to sheet.')
    save_to_sheet(worksheet = course_outline_with_lo_sheet, df = course_outline_with_lo_df)

    return


def delete_research_notes(sheet, worksheet_name="Final Outline"):
    """Remove the research_notes column from the specified worksheet."""
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    if "research_notes" in df.columns:
        df = df.drop(columns=["research_notes"])
        clear_worksheet(ws)
        save_to_sheet(ws, df)



