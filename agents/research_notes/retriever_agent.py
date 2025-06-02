from modules.chain import Chain
from services.llm_service import csv_list_parser
from agents.research_notes.retriever import get_compression_retriever, get_web_search_retriever
from services.sheets_service import get_sheet_data_and_df, save_to_sheet, hide_columns_by_name, format_worksheet
from tqdm import tqdm
from services.helper_functions import create_and_populate_columns, get_outline_with_los
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.smart_progress_bar import SmartProgressBar
from langsmith import traceable

retriver_agent_system_prompt = """You are a retriever agent with access to a knowledge base. Your task is to retrieve the best results for a given query.

Here are the details of this task you should know:

<course_details>
We are creating a course with following details -
Course Name: {course_name}
Target audience: {target_audience}
Course outline:
{course_outline}
</course_details>

<current_focus>
Within this course, the current focus is to create research notes for the following topic, subtopic and this specific learning objective:
Topic: {topic_name}
Subtopic: {subtopic_name}
Learning objective: {learning_objective}
</current_focus>

Your task is to retrieve the best results for the given learning objective while keeping the topic, subtopic and the overall course context in mind.

Thus, formulate a search query that can effectively retrieve the relevant information.

Best practices for formulating search query:
- The search query you create in most cases can be same / very similar to the learning objective.
- The search query will be executed in a vectorstore (similarity search). Thus, you can employ techniques such as query expansion, query decomposition, HyDE, etc. incase the simple query is not yeilding good results.
- Modify search query to include additional context incase the learning objective if looked in isolation is vague / too generic.
- Evolve and adapt your search queries incase they don't yeild relevant results or incase you need some more surrounding information to cover the LO properly.

Present yout output in the following format:
<observations>
[Place your observations of the step taken within these tags. It is ok for this section to be quite long and comprehensive.
Start by analysing each and every doc and then formulate overall observations.]
</observations>
<verdict>
[Based on your observations so far, decide whether to continue with retrieval or to stop. Verdict should be one of these - TERMINATE or CONTINUE]
</verdict>
<selected_doc_ids>
[Put the ids of all the docs that are highly relevant for the current LO we are focussing on. Each doc id should be in a new line. New doc ids should be appended to the same list.]
</selected_doc_ids>
<action>
[In case the verdict is to CONTINUE, think and ideate how / what query to create next to get relevant retrievals for the current LO. If verdict is to terminate, leave this blank.]
</action>
<query>
[If continue, then place the query within these tags, else leave this as blank.]
</query>
<search_on>
["VECTOR STORE" or "WEB SEARCH". Only choose web search if multiple queries against vector store did not retrieve relevant docs. Leave blank if no query.]
</search_on>

Remember, your goal is to retrieve the best information possible for the given learning objective. Be mindful of the docs you select. Make sure you follow the output format.
"""


# Helper Function to wrap docs in a readable manner while removing any duplicate docs
def get_docs_as_string(docs, all_docs):
    """
    This function wraps the docs in a readable manner while removing any duplicate docs.

    :param docs: The docs to be wrapped.
    :param all_docs: The list of all docs.
    :return: The docs as a string and the updated list of all docs.
    """
    docs_as_string = ''
    for doc in docs:
        if doc in all_docs:
            # Still add to doc string but don't add actual content
            doc_id = all_docs.index(doc)
            docs_as_string += f'============= Doc id: {doc_id} =============\n'
            docs_as_string += 'Page content not added since this doc is repeating and has already appeared earlier.\n\n'
        else:
            docs_as_string += f'============= Doc id: {len(all_docs)} =============\n'
            docs_as_string += doc.page_content + '\n\n'
            all_docs.append(doc)

    # Replace {} with {{}}
    docs_as_string = docs_as_string.replace('{', '{{')
    docs_as_string = docs_as_string.replace('}', '}}')

    return docs_as_string, all_docs

@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Retriever",
    "function_name": "retrieve_relevant_docs",
    "user_id": st.session_state.get("role", "anonymous")
})
def retrieve_relevant_docs(compression_retriever, web_search_retriever, course_name, target_audience, course_outline, topic_name, subtopic_name, learning_objective, max_turns = 5, llm = 'groq'):
    """
    This function retrieves the relevant documents for the given learning objective.

    :param compression_retriever: The compression retriever.
    :param web_search_retriever: The web search retriever.
    :param course_name: The name of the course.
    :param target_audience: The target audience of the course.
    :param course_outline: The outline of the course.
    :param topic_name: The name of the topic.
    :param subtopic_name: The name of the subtopic.
    :param learning_objective: The learning objective.
    :param max_turns: The maximum number of turns to retrieve the relevant documents.
    :param llm: The language model to use.
    :return: The selected doc ids and all docs.
    """

    # Construct the chain
    retriver_agent = Chain(llm = llm, tags = ['verdict', 'selected_doc_ids', 'action', 'query', 'search_on'])

    # Add system message
    retriver_agent.add_message(
        role = 'system',
        content = retriver_agent_system_prompt.format(
            course_name = course_name,
            target_audience = target_audience,
            course_outline = course_outline,
            topic_name = topic_name,
            subtopic_name = subtopic_name,
            learning_objective = learning_objective
        )
    )

    # Run the retriver first time with LO as query
    # docs = retriever_with_filter(query = learning_objective)
    docs = compression_retriever.invoke(learning_objective)

    # List of all docs captured
    all_docs = []

    # List to store selected doc ids
    selected_doc_ids = []

    # For first user message, query is same as LO
    query = learning_objective

    for i in range(max_turns):

        # Get the docs as string and update all docs list
        docs_as_string, all_docs = get_docs_as_string(docs, all_docs)

        # Add user message - query + retrieved docs
        retriver_agent.add_message(
            role = 'user',
            content = f'Query: {query}\n\nRetrived_docs:\n{docs_as_string}'
        )

        # Run the agent
        response = retriver_agent.run()

        # Get selected doc ids
        selected_doc_ids.extend(
            csv_list_parser.invoke(response['selected_doc_ids'])
        )

        # Check the verdict
        verdict = response['verdict']
        # Stop the loop if terminate
        if 'TERMINATE' in verdict:
            print('Terminating the loop')
            break
        # Else continue the loop
        else:
            # Extract the next query and get the docs
            query = response['query']
            # Check search type
            search_on = response['search_on']
            if "web search" in search_on.lower():
                print('Using web search retriever')
                try:
                    docs = web_search_retriever.invoke(query)
                except:
                    docs = []
            else:
                print('Using compression retriever')
                try:
                    docs = compression_retriever.invoke(query)
                except:
                    docs = []

    return selected_doc_ids, all_docs

@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Retriever",
    "function_name": "process_single_row",
    "user_id": st.session_state.get("role", "anonymous")
})
def process_single_row(index, row, compression_retriever, web_search_retriever,
                       course_name, target_audience, course_outline, llm):
    """
    Processes a single row: runs the retriever agent for each Learning Objective
    and accumulates the context string. Returns (index, context_string, source_links, as_is_sources, content_sources, web_links, video_links)
    """
    # # If row is already populated, just return existing context
    # if row['context_0'] != '':
    #     return index, row['context_0']

    # Get the LOs for this row / subtopic
    # If row is blank, search by subtopic
    if row['Learning Objectives'] == '':
        learning_objectives = [row['Subtopic']]
    else:
        learning_objectives = row['Learning Objectives'].split('\n')

    context = ""
    context_docs = []
    source_links = []  # List to store all source links
    as_is_sources = []  # List to store sources used as is
    content_sources = []  # List to store sources for content
    web_links = []  # List to store web links
    video_links = []  # List to store video links   

    for lo in learning_objectives:
        # Run the retriever agent
        selected_doc_ids, all_docs = retrieve_relevant_docs(
            compression_retriever=compression_retriever,
            web_search_retriever=web_search_retriever,
            course_name=course_name,
            target_audience=target_audience,
            course_outline=course_outline,
            topic_name=row['Topic'],
            subtopic_name=row['Subtopic'],
            learning_objective=lo,
            llm=llm
        )

        for doc_id in selected_doc_ids:
            # Check if doc_id is an integer
            if doc_id.isdigit():
                doc_index = int(doc_id)
                if len(all_docs) <= doc_index < 0:
                    print("Index not present within list.")
                    continue
                context_doc = all_docs[doc_index]
                # Skip if doc already added
                if context_doc in context_docs:
                    print(f'Skipping doc id {doc_id} as it is already added')
                    continue
                # Otherwise add the doc
                context += f'============= Doc id: {len(context_docs)} =============\n'
                context += context_doc.page_content + '\n\n'
                context_docs.append(context_doc)
                
                # Extract source link from metadata if available
                if hasattr(context_doc, 'metadata') and 'source' in context_doc.metadata:
                    source_link = context_doc.metadata['source']
                    if source_link:
                        # Format the source link with the doc ID
                        formatted_source = f"[{len(context_docs)}] {source_link}"
                        source_links.append(formatted_source)
                        
                        # Categorize the source based on criteria
                        is_hvac_school = False
                        
                        # Check if the source is from HVAC School
                        if hasattr(context_doc, 'metadata') and 'channel' in context_doc.metadata:
                            if context_doc.metadata['channel'] == 'HVAC School':
                                is_hvac_school = True
                        
                        # Check if the source URL contains hvacrschool.com
                        if 'https://hvacrschool.com/' in source_link:
                            is_hvac_school = True
                            
                        # Add to appropriate category
                        if is_hvac_school:
                            as_is_sources.append(formatted_source)
                        else:
                            content_sources.append(formatted_source)
                        
                        # Remove the doc ID prefix
                        source_link = source_link.split('] ', 1)[1] if '] ' in source_link else source_link

                        # Separate into video or web link
                        if 'youtube.com' in source_link:
                            # Check if start and end parameters exist in the URL
                            if 'start=' in source_link and 'end=' in source_link:
                                video_links.append(source_link)  # Keep the full URL with timestamps
                            else:
                                # If no timestamps in URL, check if they exist in metadata
                                if hasattr(context_doc, 'metadata'):
                                    start_time = context_doc.metadata.get('start_time')
                                    end_time = context_doc.metadata.get('end_time')
                                    if start_time is not None and end_time is not None:
                                        # Add timestamps to the URL
                                        separator = '&' if '?' in source_link else '?'
                                        video_links.append(f"{source_link}{separator}start={start_time}&end={end_time}")
                                    else:
                                        video_links.append(source_link)
                                else:
                                    video_links.append(source_link)
                        else:
                            web_links.append(source_link)
            else:
                print(f'Skipping doc id {doc_id} as it is not an integer')
                continue

    # Join all source links with newline character
    source_links_text = '\n'.join(source_links)
    as_is_sources_text = '\n'.join(as_is_sources)
    content_sources_text = '\n'.join(content_sources)
    web_links_text = '\n'.join(web_links)
    video_links_text = '\n'.join(video_links)
    
    return index, context, source_links_text, as_is_sources_text, content_sources_text, web_links_text, video_links_text


@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Retriever",
    "function_name": "run_retriever_agent_for_all_rows",
    "user_id": st.session_state.get("role", "anonymous")
})
def run_retriever_agent_for_all_rows(root_folder_id, drive, sheet, worksheet_name,
                                     course_name, target_audience, llm):
    """
    This function runs the retriever agent for all df rows, in parallel using ThreadPoolExecutor.
    """

    # Read the sheet and df
    course_outline_with_lo_sheet, course_outline_with_lo_df = get_sheet_data_and_df(
        sheet=sheet,
        sheet_name=worksheet_name
    )

    # Create columns in df if not already present
    if 'context_0' not in course_outline_with_lo_df.columns:
        course_outline_with_lo_df['context_0'] = ''
    
    # Create source_links column if not already present
    if 'source_links' not in course_outline_with_lo_df.columns:
        course_outline_with_lo_df['source_links'] = ''
        
    # Create as_is_sources and content_sources columns if not already present
    if 'as_is_sources' not in course_outline_with_lo_df.columns:
        course_outline_with_lo_df['as_is_sources'] = ''
    if 'content_sources' not in course_outline_with_lo_df.columns:
        course_outline_with_lo_df['content_sources'] = ''
    if 'web_links' not in course_outline_with_lo_df.columns:
        course_outline_with_lo_df['web_links'] = ''
    if 'video_links' not in course_outline_with_lo_df.columns:
        course_outline_with_lo_df['video_links'] = ''

    # Check if this step is already done by checking the last row of 'context_0'
    if course_outline_with_lo_df.iloc[-1]['context_0'] != '':
        print('Context already populated')
        return

    # Load the vector retriever
    compression_retriever = get_compression_retriever(
        course_name=course_name,
        root_folder_id=root_folder_id,
        drive=drive,
        sheet=sheet,
        retriever_1_weight=0.5,
        retriever_2_weight=0.5
    )

    # Load the web retriever
    web_search_retriever = get_web_search_retriever()

    # Get the course outline
    course_outline = get_outline_with_los(
        df=course_outline_with_lo_df,
        include_learning_objectives=False
    )

    # Prepare for parallel processing
    futures = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit tasks for each row
        for index, row in course_outline_with_lo_df.iterrows():
            # Skip if already populated
            if row['context_0'] != '':
                continue
            futures.append(
                executor.submit(
                    process_single_row,
                    index,
                    row,
                    compression_retriever,
                    web_search_retriever,
                    course_name,
                    target_audience,
                    course_outline,
                    llm
                )
            )

        # Collect the results as they complete
        save_interval = 1  # how often to save
        total_tasks = len(futures)
        
        # Initialize the progress tracker
        progress = SmartProgressBar(total_tasks = total_tasks, description = "Percent complete", save_interval = save_interval)

        for future in tqdm(as_completed(futures), total=total_tasks):
            index, context, source_links, as_is_sources, content_sources, web_links, video_links = future.result()

            # Update the row in the DataFrame
            course_outline_with_lo_df = create_and_populate_columns(
                df=course_outline_with_lo_df,
                text=context,
                specific_index=index,
                col_base_name='context',
                chunk_size=49000
            )

            # Convert to list if string
            if isinstance(web_links, str):
                web_links = web_links.split('\n')
            if isinstance(video_links, str):
                video_links = video_links.split('\n')

            # Normalize and deduplicate web and video links before joining
            def clean_url(url):
                return url.strip().lower()
                
            web_links = list({clean_url(link): link for link in web_links}.values())
            video_links = list({clean_url(link): link for link in video_links}.values())

            # Join the links
            web_links_text = '\n'.join(web_links)
            video_links_text = '\n'.join(video_links)
            
            # Update the source links columns
            course_outline_with_lo_df.at[index, 'source_links'] = source_links
            course_outline_with_lo_df.at[index, 'as_is_sources'] = as_is_sources
            course_outline_with_lo_df.at[index, 'content_sources'] = content_sources
            course_outline_with_lo_df.at[index, 'web_links'] = web_links_text
            course_outline_with_lo_df.at[index, 'video_links'] = video_links_text

            # Update progress
            progress.update()

            # Check if we should save
            if progress.should_save():
                print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
                # Convert all columns to string to avoid data-type issues
                save_to_sheet(worksheet = course_outline_with_lo_sheet, df = course_outline_with_lo_df)


    # Final save to sheet after all tasks
    print('All rows processed. Saving final DataFrame to sheet.')
    save_to_sheet(worksheet = course_outline_with_lo_sheet, df = course_outline_with_lo_df)

    format_worksheet(course_outline_with_lo_sheet)

    column_names = [column_name for column_name in course_outline_with_lo_df.columns if ("context" in column_name or column_name in ["source_links", "as_is_sources", "content_sources"])]

    # Hide the columns
    hide_columns_by_name(worksheet = course_outline_with_lo_sheet, column_names = column_names, df = course_outline_with_lo_df)

    return


def manual_input_review_context(sheet, worksheet_name):
    """
    This function checks if the user has properly reviewed the context and marked the review as Done or not.

    :param sheet: The sheet object.
    :param worksheet_name: The worksheet name.
    """

    course_outline_with_lo_sheet, course_outline_with_lo_df = get_sheet_data_and_df(sheet = sheet, sheet_name = worksheet_name)

    if "Context Review" not in course_outline_with_lo_df.columns:
        raise Exception(f"Manually add the following column `Context Review` inside this sheet - {course_outline_with_lo_sheet.url}.\nReview the context_n columns and enter `Done` in the first row of `Context Review` column.")
        # st.write(f"Manually add the following column `Context Review` inside this sheet - {course_outline_with_lo_sheet.url}.\nReview the context_n columns and enter `Done` in the first row of `Context Review` column.")
        # return False
    elif course_outline_with_lo_df["Context Review"].values[0] != "Done":
        raise ValueError("Review the context_n columns and enter `Done` in the first row of `Context Review` column.")
        # return False
    else:
        print("Context is reviewed")
        return True

