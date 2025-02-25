from modules.chain import Chain
from services.helper_functions import get_outline_with_los
from tqdm import tqdm
from services.sheets_service import get_sheet_data_and_df
from agents.research_notes.retriever import get_compression_retriever, get_web_search_retriever
from agents.research_notes.retriever_agent import retrieve_relevant_docs
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
from services.smart_progress_bar import SmartProgressBar


revise_research_notes_prompt = """Your task is to revise the previously generated research notes based on ai and human review.

Here's the list of documents retrieved for context -
<relevant_documents>
{relevant_documents}
</relevant_documents>

Here's the course related information -
<course_info>
Course name: {course_name}
Target audience: {target_audience}

Course outline:
{course_outline}
</course_info>

Here's the current subtopic we are focussing on:
<subtopic_and_los>
{subtopic_and_los}
</subtopic_and_los>

This is the current iteration of the research notes you previously generated:
<current_notes>
{current_notes}
</current_notes>

Here's the AI generated + human review from the reviewer agent:
<review>
The items in the research notes were categorized into the following:
1) not covered at all:
{not_covered_at_all}

2) not covered enough:
{not_covered_enough}

3) perfectly covered:
{perfectly_covered}

4) covered too much:
{covered_too_much}

And here's the additional comments from the human reviewer:
{manual_comments}
</review>

Your task is to incorporate the review into the research notes and present a revised iteration of it.

Guidelines:
- Items listed in the "not covered at all" and "not covered enough" means that those points need to be covered in the research notes.
- Items listed in the "perfectly covered" are good as is. No modification is needed for these points. Keep them as is in the research notes.
- Items listed in the "covered too much" need to be cut down from the research notes.
- The comments by the human reviewer supersede all other guidelines. Your no. 1 priority should be to follow those comments.
- No "new information" should be added on added on your own. In other words, any addition you do should be derived from the documents.

Present your final output in this format:
<review_analysis>
   - Summarize your understanding the overall course scope and the current subtopic at focus.
   - Summarize the research notes.
   - Analyze the review and the manual comments in relation to the above.
</review_analysis>

<examine_documents>
   - If additional information needs to be added to the research notes, then analyze the list of documents to identify the ones that contain relevant information.
   - If no new information needs to be added, then leave this section blank.
</examine_documents>

<determine_information>
   - Determine the information you intend to add into the research notes based on the review.
   - Similary, determine the information you intend to remove from the research notes based on the reveiw.
</determine_information>

<propose_framework>
   - Create a logical structure for the notes that aligns with the learning objectives.
   - Organize the main points and sub-points in a clear, hierarchical manner.
   - Number each main point and sub-point to ensure a clear hierarchy.
   - Ensure the framework provides a comprehensive overview of the subtopic.
   - Consider how to make the structure engaging and interesting for the target audience.
</propose_framework>

<create_notes>
   - Fill in the proposed framework with detailed information from the relevant documents.
   - Write in clear, concise paragraphs appropriate for the target audience.
   - Ensure each paragraph directly supports one or more of the learning objectives.
   - Include relevant examples, definitions, and explanations as needed.
   - Maintain a logical flow of information throughout the notes.
   - Focus on making the content interesting, engaging, and educational.
</create_notes>

Remember:
- Your main goal is to revise the research notes based on the review provided.
- Focus solely on the specific subtopic and learning objectives provided.
- Consider the broader context of the course and the needs of the target audience.
- Ensure that your research notes are comprehensive, well-structured, and directly aligned with the learning objectives.
- Write in clear, engaging paragraphs that will interest and educate the target audience.
- Do not add any information that is not derived from the provided relevant documents.
"""


def revise_research_notes(compression_retriever, web_search_retriever, course_name, target_audience, course_outline, topic_name, subtopic_name, learning_objectives, current_notes, not_covered_at_all, not_covered_enough, perfectly_covered, covered_too_much, manual_comments, llm = 'groq'):
    """
    This function revises the research notes.

    :param compression_retriever: The compression retriever.
    :param web_search_retriever: The web search retriever.
    :param course_name: The name of the course.
    :param target_audience: The target audience of the course.
    :param course_outline: The outline of the course.
    :param topic_name: The topic name.
    :param subtopic_name: The subtopic name.
    :param learning_objectives: The learning objectives.
    :param current_notes: The current
    :param not_covered_at_all: The not covered at all.
    :param not_covered_enough: The not covered enough.
    :param perfectly_covered: The perfectly covered.
    :param covered_too_much: The covered too much.
    :param manual_comments: The manual comments.
    :param llm: The language model to use.
    :return: The revised research notes.
    """

    revise_research_notes_agent = Chain(llm = llm, tags = ['create_notes'])

    # Get the list of items to retrieve documents for
    queries = []
    # Check if it is an empty string
    if not_covered_at_all != '':
        # Split it into a list
        for item in not_covered_at_all.split('\n'):
            # Check if the text item length is more than 5 chars, otherwise it may be some junk like - NA, Null, [], etc.
            if len(item) > 5:
                queries.append(item)
    # Check if it is an empty string
    if not_covered_enough != '':
        # Split it into a list
        for item in not_covered_enough.split('\n'):
            # Check if the text item length is more than 5 chars, otherwise it may be some junk like - NA, Null, [], etc.
            if len(item) > 5:
                queries.append(item)


    # Run the retriever for these queries
    context = ""
    context_docs = []
    for query in queries:
        # Run the retriever agent
        selected_doc_ids, all_docs = retrieve_relevant_docs(
            compression_retriever = compression_retriever,
            web_search_retriever = web_search_retriever,
            course_name = course_name,
            target_audience = target_audience,
            course_outline = course_outline,
            topic_name = topic_name,
            subtopic_name = subtopic_name,
            learning_objective = query,
            llm = llm
        )

        for id in selected_doc_ids:
            # Check if id can be converted to int
            if id.isdigit():
                context_doc = all_docs[int(id)]
                # Check if doc already added
                if context_doc in context_docs:
                    print(f'Skipping doc id {id} as it is already added')
                    continue
                # Else add the doc to the context
                else:
                    context += f'============= Doc id: {len(context_docs)} =============\n'
                    context += context_doc.page_content + '\n\n'
                    context_docs.append(context_doc)
            else:
                print(f'Skipping doc id {id} as it is not an integer')
                continue


    # Add message to reviser
    revise_research_notes_agent.add_message(
        role = 'user',
        content = revise_research_notes_prompt.format(
            course_name = course_name,
            target_audience = target_audience,
            course_outline = course_outline,
            subtopic_and_los = subtopic_name + '\n' + learning_objectives,
            current_notes = current_notes,
            not_covered_at_all = not_covered_at_all,
            not_covered_enough = not_covered_enough,
            perfectly_covered = perfectly_covered,
            covered_too_much = covered_too_much,
            manual_comments = manual_comments,
            relevant_documents = context
        )
    )

    response = revise_research_notes_agent.run()

    return response['create_notes']


def run_reviser_agent_for_all_rows(root_folder_id, drive, sheet, worksheet_name, course_name, target_audience, llm = 'groq'):
    """
    This function runs the reviser agent for all rows in the sheet.

    :param root_folder_id: The ID of the root folder containing 'Pickle files' and 'Vectorstore files'.
    :param drive: Authenticated GoogleDrive instance (PyDrive2).
    :param sheet: The sheet object.
    :param worksheet_name: The worksheet name.
    :param course_name: The name of the course.
    :param target_audience: The target audience of the course.
    :param llm: The language model to use.
    :return: None
    """

    # Read the sheet and df
    course_outline_with_lo_sheet, course_outline_with_lo_df = get_sheet_data_and_df(sheet = sheet, sheet_name = worksheet_name)

    # Create revised research notes col if not already present
    if 'revised_research_notes' not in course_outline_with_lo_df.columns:
        course_outline_with_lo_df['revised_research_notes'] = ''

    # Check if this step is already done by checking last row of revised_research_notes column
    if course_outline_with_lo_df.iloc[-1]['revised_research_notes'] != '':
        print('Reviser already populated')
        return

    # Load the vector retriever
    compression_retriever = get_compression_retriever(
        course_name = course_name,
        root_folder_id = root_folder_id,
        drive = drive,
        sheet = sheet,
        retriever_1_weight = 0.5,
        retriever_2_weight = 0.5
    )

    # Load the web retriever
    web_search_retriever = get_web_search_retriever()

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

            # Skip if verdict = PASS
            if 'pass' in row['verdict'].lower():
                print(f'Skipping row {index} as verdict is PASS')
                # Update the df
                course_outline_with_lo_df.loc[index, 'revised_research_notes'] = row['research_notes']
                continue

            # Check if row already populated
            if row['revised_research_notes'] != '':
                print(f'Skipping row {index}. Already populated')
                continue

            # Submit the task
            future = executor.submit(
                revise_research_notes,
                compression_retriever = compression_retriever,
                web_search_retriever = web_search_retriever,
                course_name = course_name,
                target_audience = target_audience,
                course_outline = course_outline_with_lo,
                topic_name = row['Topic'],
                subtopic_name = row['Subtopic'],
                learning_objectives = row['Learning Objectives'],
                current_notes = row['research_notes'],
                not_covered_at_all = row['not_covered_at_all'],
                not_covered_enough = row['not_covered_enough'],
                perfectly_covered = row['perfectly_covered'],
                covered_too_much = row['covered_too_much'],
                manual_comments = row['Manual Comments'],
                llm = llm
            )

            # Map the Future to the index
            futures_map[future] = index

        # Collect the results as they complete
        total_tasks = len(futures_map)
        save_interval = 1  # how often to save (in number of completed tasks)

        # Initialize the progress tracker
        progress = SmartProgressBar(total_tasks = total_tasks, description = "Percent complete", save_interval = save_interval)

        # Now, pass only the futures (the keys) to as_completed:
        for future in tqdm(as_completed(futures_map), total=total_tasks):
            index = futures_map[future]  # retrieve the index
            revised_notes = future.result()

            # Update the df row with analysis
            course_outline_with_lo_df.loc[index, 'revised_research_notes'] = revised_notes

            # Update progress
            progress.update()

            # Check if we should save
            if progress.should_save():
                print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
                course_outline_with_lo_df = course_outline_with_lo_df.astype(str)
                course_outline_with_lo_sheet.update([course_outline_with_lo_df.columns.values.tolist()] + course_outline_with_lo_df.values.tolist())


    # Final save to sheet after all tasks
    print('All rows processed. Saving final DataFrame to sheet.')
    course_outline_with_lo_df = course_outline_with_lo_df.astype(str)
    course_outline_with_lo_sheet.update([course_outline_with_lo_df.columns.values.tolist()] + course_outline_with_lo_df.values.tolist())

    return



