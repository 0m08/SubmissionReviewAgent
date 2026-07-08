from modules.chain import Chain
from services.llm_service import csv_list_parser
from agents.research_notes.retriever import get_compression_retriever, get_web_search_retriever
from services.sheets_service import (
    get_sheet_data_and_df,
    save_to_sheet,
    hide_columns_by_name,
    format_worksheet,
    hide_worksheet_by_name,
    clear_worksheet,
    delete_worksheet,
    get_worksheet_names,
    create_or_read_worksheet,
)
from tqdm import tqdm
from services.helper_functions import create_and_populate_columns, get_outline_with_los, normalize_youtube_url, extract_transcript_segment
from services.youtube_video_loader import convert_time_to_sec
import json
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.smart_progress_bar import SmartProgressBar
import urllib.parse as urlparse
from langsmith import traceable
import pandas as pd
from services.helper_functions import extract_drive_file_id
from services.web_page_loaders import get_docs_from_url
from services.youtube_video_loader import get_video_id_from_url, convert_time, get_transcript_with_fallback, get_transcript_assemblyai_drive
import json
import os
import shutil
import gspread
import base64
import re
from services.helper_functions import get_short_name


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
        if response.get('selected_doc_ids'):
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

# Helper to format transcript segments into 6-word groups with timestamps
def format_transcript_segments(transcript_json, group_size=6):
    segments = json.loads(transcript_json) if isinstance(transcript_json, str) else transcript_json
    lines = []
    group = []
    group_start = None
    for i, word in enumerate(segments):
        if not group:
            group_start = word['start']
        group.append(word['text'])
        if len(group) == group_size:
            timestamp = convert_time(group_start, format='HH:MM:SS' if group_start >= 3600 else 'MM:SS')
            lines.append(f"- '{timestamp}': {' '.join(group)}")
            group = []
    if group:
        timestamp = convert_time(group_start, format='HH:MM:SS' if group_start >= 3600 else 'MM:SS')
        lines.append(f"- '{timestamp}': {' '.join(group)}")
    return '\n'.join(lines)


relevant_link_selection_if_multiple_matching_reference_links_prompt = """You are an expert instructional designer and content analyst. Your task is to select the single most relevant reference link for a specific Learning Objective (LO) from a set of candidate links. You will be provided with the course context and the content associated with each candidate link.

Below is the course information:

<course_information>
Course name: {course_name}
Target audience: {target_audience}
</course_information>

Topic:
<topic>
{topic}
</topic>

Subtopic:
<subtopic>
{subtopic}
</subtopic>

Learning Objective:
<learning_objective>
{learning_objective}
</learning_objective>

Below are the candidate reference links and their content:
Note: All candidate links will be of the same type - either all YouTube video links (with their transcript segments) or all web article links (with their article content).

<candidate_links_and_content>
{candidate_links_and_content}
</candidate_links_and_content>

Your task:

- Carefully read the Learning Objective and the content of each candidate link.
- Analyze which link provides the most relevant, comprehensive, and directly useful information for achieving the Learning Objective, considering the course name and target audience.
- Select only one link as the single most relevant reference for the given Learning Objective.
- Do not add, modify, or paraphrase the content. Only select from the provided links.

Provide your output strictly in the following format:

<final_output>

<analysis>
[Provide a brief analysis of your reasoning for selecting the most relevant link. Explain why this link is the best fit for the Learning Objective, referencing specific content if needed.]
</analysis>

<selected_link>
[Paste the single most relevant link here. Only include the URL as it is, nothing else.]
</selected_link>

</final_output>
"""


def relevant_link_selection_if_multiple_matching_reference_links(course_name, target_audience, topic, subtopic, learning_objective, candidate_links_and_content, llm = 'gemini_2_5_flash'):
    """
    This function runs the relevant link selection prompt for multiple matching reference links for a single Learning Objective
    :param course_name: The name of the course.
    :param target_audience: The target audience for the course.
    :param topic: The topic for the LO.
    :param subtopic: The subtopic for the LO.
    :param learning_objective: The learning objective.
    :param candidate_links_and_content: The formatted string of candidate links and their content.
    :param llm: The language model to use.
    :return: The selected link (content inside <selected_link>)
    """
    
    # Construct the chain
    relevant_link_selection_agent = Chain(llm = llm, tags = ['final_output'])

    # # Format the prompt for debugging (printing)
    # formatted_prompt = relevant_link_selection_if_multiple_matching_reference_links_prompt.format(
    #         course_name = course_name,
    #         target_audience = target_audience,
    #         topic = topic,
    #         subtopic = subtopic,
    #         learning_objective = learning_objective,
    #         candidate_links_and_content = candidate_links_and_content
    #     ) 

    # # Print the formatted prompt for debugging
    # print("\n🔍 Prompt Being Sent to LLM:\n")
    # print(formatted_prompt)
    # print("\n" + "=" * 100 + "\n")

    # Add the user message
    relevant_link_selection_agent.add_message(
        role = 'user',
        content = relevant_link_selection_if_multiple_matching_reference_links_prompt.format(
            course_name = course_name,
            target_audience = target_audience,
            topic = topic,
            subtopic = subtopic,
            learning_objective = learning_objective,
            candidate_links_and_content = candidate_links_and_content
        )
    )
    response = relevant_link_selection_agent.run()
    match = re.search(r'<selected_link>(.*?)</selected_link>', response['text'], re.DOTALL)
    if match:
        return match.group(1).strip()
    return response['text']


def get_transcript_from_youtube_transcript_csv(video_id):
    """
    Fetch transcript from local youtube_transcript.csv file.
    Reconstructs full transcript from JSON arrays and converts to seconds format.
    
    Args:
        video_id (str): YouTube video ID to search for
        
    Returns:
        list: List of transcript segments in format [{"timestamp": "seconds", "text": "content"}]
        None: If video not found or error occurs
    """
    try:
        import pandas as pd
        
        # Read the youtube transcript CSV
        csv_path = "assets/youtube_transcript.csv"
        df = pd.read_csv(csv_path)
        
        # Filter for the specific video_id
        video_row = df[df['video_id'] == video_id]
        
        if video_row.empty:
            print(f"Video {video_id} not found in youtube_transcript.csv")
            return None
        
        # Reconstruct transcript from JSON arrays in transcript columns
        transcript_segments = []
        
        # Process all transcript columns
        transcript_columns = [col for col in video_row.columns if col.startswith('transcript_')]
        
        if not transcript_columns:
            print(f"No transcript columns found for video {video_id}")
            return None
        
        for col in transcript_columns:
            if col in video_row.columns and pd.notna(video_row[col].iloc[0]):
                try:
                    # Parse JSON array from transcript column
                    transcript_data = json.loads(video_row[col].iloc[0])
                    
                    for segment in transcript_data:
                        # Convert HH:MM:SS timestamp to seconds
                        timestamp_str = segment.get('timestamp', '00:00')
                        timestamp_seconds = convert_time_to_sec(timestamp_str)
                        
                        # Get text content
                        text_content = str(segment.get('text', '')).strip()
                        
                        if text_content and text_content != 'nan':
                            transcript_segments.append({
                                "timestamp": str(timestamp_seconds),
                                "text": text_content,
                                "start_time": timestamp_seconds  # Add this for sorting
                            })
                            
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    print(f"Error processing transcript column {col} for video {video_id}: {e}")
                    continue
        
        # Sort by start_time to ensure chronological order
        transcript_segments.sort(key=lambda x: x['start_time'])
        
        if transcript_segments:
            print(f"Successfully loaded {len(transcript_segments)} segments for video {video_id} from youtube_transcript.csv")
            return transcript_segments
        else:
            print(f"No valid transcript segments found for video {video_id}")
            return None
            
    except Exception as e:
        print(f"Error reading youtube_transcript.csv: {e}")
        return None


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

    required_cols = ["References", "Reference type", "Reference usage"]
    if all(col in row.index for col in required_cols):
        ref = str(row["References"]).strip()
        ref_type = str(row["Reference type"]).strip()
        ref_usage = str(row["Reference usage"]).strip()
        
        # Process Web Article with "Content" usage - use actual article content
        if ref and ref_type == "Web Article" and ref_usage == "Content":
            context_chunks = []
            try:
                # Split references by newline to handle multiple web articles
                web_urls = [url.strip() for url in ref.split('\n') if url.strip()]
                all_web_content = []
                
                for article_num, web_url in enumerate(web_urls, 1):
                    try:
                        docs = get_docs_from_url(web_url, query="")
                        article_content_parts = []
                        
                        for doc in docs:
                            # If chunk is too large, further split
                            if len(doc.page_content) > 49000:
                                # Use create_and_populate_columns on a temp df
                                temp_df = pd.DataFrame({"dummy": [""]})
                                temp_df = create_and_populate_columns(temp_df, doc.page_content, 0, "context", 49000)
                                for col in temp_df.columns:
                                    if col.startswith("context_"):
                                        article_content_parts.append(temp_df.at[0, col])
                            else:
                                article_content_parts.append(doc.page_content)
                        
                        # Add article header and content
                        if article_content_parts:
                            article_content = f"Article {article_num}:\n\n" + "\n\n".join(article_content_parts)
                            all_web_content.append(article_content)
                            
                    except Exception as e:
                        print(f"Error processing web article {article_num} ({web_url}): {e}")
                        continue
                
                # Join all articles with separator
                if all_web_content:
                    combined_content = "\n\n---\n\n".join(all_web_content)
                    
                    # Split and fill context_n columns
                    temp_df = pd.DataFrame({"dummy": [""]})
                    temp_df = create_and_populate_columns(temp_df, combined_content, 0, "context", 49000)
                    for col in temp_df.columns:
                        if col.startswith("context_"):
                            context_chunks.append(temp_df.at[0, col])
                
            except Exception as e:
                print(f"Error using web loader for row {index}: {e}")

            context_combined = "\n\n".join(context_chunks)
            return index, context_combined, "", "", "", "", ""
        
        # Process YouTube Video with "Video" or "Content" usage - use actual video transcript
        elif ref and ref_type == "Youtube Video" and ref_usage in ["Video", "Content"]:
            context_chunks = []
            try:
                # Split references by newline to handle multiple videos
                video_urls = [url.strip() for url in ref.split('\n') if url.strip()]
                all_video_content = []
                
                for video_num, video_url in enumerate(video_urls, 1):
                    try:
                        video_id = get_video_id_from_url(video_url)

                        # Extract start and end timestamps from URL if present
                        start_time = None
                        end_time = None
                        try:
                            parsed_url = urlparse.urlparse(video_url)
                            query_params = urlparse.parse_qs(parsed_url.query)
                            start_param = query_params.get('start', [None])[0]
                            end_param = query_params.get('end', [None])[0]
                            if start_param:
                                start_time = int(start_param)
                            if end_param:
                                end_time = int(end_param)
                            if start_time is not None or end_time is not None:
                                print(f"Video {video_id}: Filtering transcript to start={start_time}, end={end_time}")
                        except Exception as e:
                            print(f"Could not parse timestamps from URL {video_url}: {e}")

                        #First try youtube transcripts CSV fallback
                        transcript = get_transcript_from_youtube_transcript_csv(video_id)

                        # If not found in youtube transcript CSV, fall back to existing methods
                        if not transcript:
                            print(f"Video {video_id} not found in youtube_transcript.csv, trying other methods...")
                            transcript = get_transcript_with_fallback(video_id, return_text_only=False)

                        if not transcript:
                            print(f"No transcript returned for video {video_id}")
                            continue

                        # Convert all timestamps to seconds and prepare for filtering
                        items_with_seconds = []
                        for item in transcript:
                            # Convert timestamp from HH:MM:SS to seconds (if needed)
                            if ':' in str(item['timestamp']):
                                item_time = convert_time_to_sec(item['timestamp'])
                            else:
                                # Already in seconds format from video chunks CSV
                                item_time = int(item['timestamp'])
                            items_with_seconds.append({
                                'time': item_time,
                                'text': item['text']
                            })

                        # Filter transcript based on start/end times if specified
                        if start_time is not None or end_time is not None:
                            # Find the starting index (include overlap from start)
                            start_idx = 0
                            if start_time is not None:
                                # Find the last segment that starts BEFORE start_time (it's likely still playing)
                                last_before_start = None
                                for i, item in enumerate(items_with_seconds):
                                    if item['time'] < start_time:
                                        last_before_start = i
                                    elif item['time'] >= start_time:
                                        break

                                # If there's a segment before start_time, include it (overlap)
                                if last_before_start is not None:
                                    start_idx = last_before_start
                                else:
                                    # No segment before start_time, find first segment at or after start_time
                                    for i, item in enumerate(items_with_seconds):
                                        if item['time'] >= start_time:
                                            start_idx = i
                                            break

                            # Filter based on start_idx and end_time
                            filtered_items = []
                            for i in range(start_idx, len(items_with_seconds)):
                                item = items_with_seconds[i]
                                # For end_time: include segments that START at or before end_time (overlap)
                                if end_time is not None and item['time'] > end_time:
                                    break
                                filtered_items.append(item)
                            items_with_seconds = filtered_items

                        # Format each segment as "- '{seconds}': {text}"
                        lines = []
                        for item in items_with_seconds:
                            total_seconds = item['time']
                            lines.append(f"- '{total_seconds}': {item['text']}")
                        
                        # Add video header and content - include video ID only for "Video" usage
                        if ref_usage == "Video":
                            video_content = f"Video {video_num} (ID: {video_id}):\n\n" + '\n'.join(lines)
                        else:
                            video_content = f"Video {video_num}:\n\n" + '\n'.join(lines)
                        all_video_content.append(video_content)
                        
                    except Exception as e:
                        print(f"Error processing video {video_num} ({video_url}): {e}")
                        continue
                
                # Join all videos with separator
                if all_video_content:
                    combined_transcript = "\n\n---\n\n".join(all_video_content)
                    
                    # Split and fill context_n columns
                    temp_df = pd.DataFrame({"dummy": [""]})
                    temp_df = create_and_populate_columns(temp_df, combined_transcript, 0, "context", 49000)
                    for col in temp_df.columns:
                        if col.startswith("context_"):
                            context_chunks.append(temp_df.at[0, col])
                
            except Exception as e:
                print(f"Error using YouTube loader for row {index}: {e}")

            context_combined = "\n\n".join(context_chunks)
            return index, context_combined, "", "", "", "", ""
        # --- Google Drive Video logic ---
        elif ref and ref_type == "Google Drive Video" and ref_usage and ref_usage != 'nan':
            context_chunks = []
            try:
                # Split references by newline to handle multiple videos
                video_urls = [url.strip() for url in ref.split('\n') if url.strip()]
                all_video_content = []
                
                for video_num, video_url in enumerate(video_urls, 1):
                    try:
                        # 1. Try to fetch transcript from MAIN GRIT VIDEOS sheet
                        found = False
                        video_content = None
                        
                        try:
                            # --- Use gc from session state ---
                            gc = st.session_state.get("gc")
                            if not gc:
                                raise Exception("No gspread client found in session state.")
                            grit_sheet = gc.open("MAIN GRIT VIDEOS")
                            worksheet_names = get_worksheet_names(grit_sheet)
                            
                            # Extract file ID from the References column (outline sheet)
                            ref_file_id = extract_drive_file_id(video_url.strip())
                            for ws_name in worksheet_names:
                                ws, df = get_sheet_data_and_df(grit_sheet, ws_name)
                                # Make matching robust by stripping whitespace
                                df['File Link'] = df['File Link'].astype(str).str.strip()
                                # Extract file IDs from the File Link column (MAIN GRIT VIDEOS sheet)
                                df['File Link File ID'] = df['File Link'].apply(extract_drive_file_id)
                                # Compare file IDs for matching
                                match = df[df["File Link File ID"] == ref_file_id]
                                if not match.empty:
                                    print(f"Match found for file ID '{ref_file_id}' in worksheet '{ws_name}' (References vs File Link)")
                                    found = True
                                    transcript_parts = []
                                    for col in match.columns:
                                        if col.startswith("Transcript ") and col[len("Transcript ") :].strip().isdigit():
                                            transcript_json = match.iloc[0][col]
                                            if transcript_json and transcript_json != 'nan':
                                                print(f"Fetched transcript from MAIN GRIT VIDEOS for file ID {ref_file_id} in worksheet {ws_name}, column {col}")
                                                formatted = format_transcript_segments(transcript_json, group_size=6)
                                                transcript_parts.append(formatted)
                                    if transcript_parts:
                                        video_content = "\n\n".join(transcript_parts)
                                    break
                                else:
                                    print(f"No match found for file ID '{ref_file_id}' in worksheet '{ws_name}' (References vs File Link)")
                        except Exception as e:
                            print(f"Error fetching from MAIN GRIT VIDEOS for video {video_num}: {e}")
                        
                        # 2. Fallback: Download from Drive, transcribe with AssemblyAI
                        if not found or not video_content:
                            try:
                                # Assume video_url is a Google Drive file link or ID
                                transcript = get_transcript_assemblyai_drive(video_url)
                                # transcript is already formatted for context_n columns (convert timestamp to seconds)
                                lines = []
                                for item in transcript:
                                    # Convert timestamp from HH:MM:SS to seconds
                                    total_seconds = convert_time_to_sec(item['timestamp'])
                                    lines.append(f"- '{total_seconds}': {item['text']}")
                                video_content = '\n'.join(lines)
                            except Exception as e:
                                print(f"AssemblyAI fallback failed for Drive Video {video_num}: {e}")
                                continue
                        
                        # Add video header and content
                        if video_content:
                            formatted_video_content = f"Video {video_num}:\n\n" + video_content
                            all_video_content.append(formatted_video_content)
                        
                    except Exception as e:
                        print(f"Error processing Google Drive video {video_num} ({video_url}): {e}")
                        continue
                
                # Join all videos with separator and split into context chunks
                if all_video_content:
                    combined_transcript = "\n\n---\n\n".join(all_video_content)
                    
                    # Split and fill context_n columns
                    temp_df = pd.DataFrame({"dummy": [""]})
                    temp_df = create_and_populate_columns(temp_df, combined_transcript, 0, "context", 49000)
                    for col in temp_df.columns:
                        if col.startswith("context_"):
                            context_chunks.append(temp_df.at[0, col])
                    
                    context_combined = "\n\n".join(context_chunks)
                    return index, context_combined, "", "", "", "", ""
                else:
                    return index, '', '', '', '', '', ''
                    
            except Exception as e:
                print(f"Error processing Google Drive videos for row {index}: {e}")
                return index, '', '', '', '', '', ''

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
                            parsed = urlparse.urlparse(source_link)
                            query = urlparse.parse_qs(parsed.query)
                            vid_id = query.get('v', [None])[0]

                            if vid_id:  # Only proceed if video ID found
                                # Extract start/end from URL
                                start_time = query.get('start', [None])[0]
                                end_time = query.get('end', [None])[0]

                                # If not in URL, check metadata
                                if (start_time is None or end_time is None) and hasattr(context_doc, 'metadata'):
                                    start_time = start_time or context_doc.metadata.get('start_time')
                                    end_time = end_time or context_doc.metadata.get('end_time')

                                # Build embed link
                                params = []
                                if start_time is not None:
                                    params.append(f"start={start_time}")
                                if end_time is not None:
                                    params.append(f"end={end_time}")

                                embed_url = f"https://www.youtube.com/embed/{vid_id}"
                                if params:
                                    embed_url += "?" + "&".join(params)

                                video_links.append(embed_url)
                            else:
                                video_links.append(source_link)  # fallback if no vid id
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

    # Check if this step is already done by checking if ALL rows have 'context_0' populated
    if (course_outline_with_lo_df['context_0'] != '').all():
        print('Context already populated for all rows')
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
            
            # Skip if References column is not empty
            if str(row.get("References", "")).strip():
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

    # Reference link matching and usage for Initial Outline Stage
    try:
        course_info_ws = sheet.worksheet("Course info")
        course_info_df = pd.DataFrame(course_info_ws.get_all_records())
        course_info_df.columns = [col.strip() for col in course_info_df.columns]
        outline_stage = course_info_df["Outline Stage"].dropna().astype(str).str.strip().str.lower().iloc[0]
    except Exception as e:
        print(f"Failed to read 'Outline Stage' from 'Course Info' sheet: {e}")
        outline_stage = None

    if outline_stage == "initial":
        print("Running Reference link matching and usage for Initial Outline Stage")
        
        # Load Base Outline
        try:
            base_outline_ws, base_outline_df = get_sheet_data_and_df(sheet, "Base Outline")
        except Exception as e:
            print(f"Failed to load Base Outline: {e}")
            return
        
        # Reload Final Outline
        final_ws, final_df = get_sheet_data_and_df(sheet, worksheet_name)
        
        # Ensure reference columns exist 
        for col in reversed(["References", "Reference type", "Reference usage"]):
            if col not in final_df.columns:
                final_df.insert(3, col, "")
        
        # For each LO, cross-reference links
        for idx, row in final_df.iterrows():
            web_links = str(row.get("web_links", "")).split("\n") if row.get("web_links", "") else []
            video_links = str(row.get("video_links", "")).split("\n") if row.get("video_links", "") else []
            
            # Normalize all links
            web_links = [w.strip() for w in web_links if w.strip()]
            video_links = [v.strip() for v in video_links if v.strip()]
            
            # Prepare Base Outline reference set
            base_refs = base_outline_df["References"].dropna().astype(str).str.strip().tolist()
            base_refs_youtube = [normalize_youtube_url(ref) for ref in base_refs if "youtube.com" in ref]
            base_refs_web = [ref for ref in base_refs if "youtube.com" not in ref]
            
            # Find matches
            matched_youtube = []
            for v in video_links:
                base_url = normalize_youtube_url(v)
                if base_url in base_refs_youtube:
                    matched_youtube.append(v)
            matched_web = [w for w in web_links if w in base_refs_web]
            
            # Selection logic
            selected_link = None
            selected_type = None
            selected_usage = None
            if matched_youtube:
                if len(matched_youtube) == 1:
                    selected_link = matched_youtube[0]
                else:

                    # Prepare prompt input: fetch transcript segments for each
                    candidate_links_and_content = ""
                    for i, link in enumerate(matched_youtube, 1):
                        video_id = get_video_id_from_url(link)
                        transcript = get_transcript_with_fallback(video_id, return_text_only=False)
                        # Extract segment if start/end present
                        m = re.search(r"[?&]start=(\d+)", link)
                        n = re.search(r"[?&]end=(\d+)", link)
                        if m and n:
                            start_sec = int(m.group(1))
                            end_sec = int(n.group(1))
                            transcript = extract_transcript_segment(transcript, start_sec, end_sec)
                        
                        # Format as lines
                        lines = [f"- '{item['timestamp']}': {item['text']}" for item in transcript]
                        transcript_text = '\n'.join(lines)
                        candidate_links_and_content += (
                            f"<link_{i}>\n\n"
                            f"<link>\nLink: {link}\n</link>\n\n"
                            f"<content>\nContent:\n{transcript_text}\n</content>\n\n"
                            f"</link_{i}>\n\n"
                        )
                    selected_link = relevant_link_selection_if_multiple_matching_reference_links(
                        course_name, target_audience, row["Topic"], row["Subtopic"], row["Learning Objectives"], candidate_links_and_content, llm=llm
                    )
                
                # Find Reference type/usage from Base Outline (match on base URL)
                base_url = normalize_youtube_url(selected_link)
                match_row = base_outline_df[base_outline_df["References"].apply(lambda x: normalize_youtube_url(str(x)) == base_url)]
                if not match_row.empty:
                    selected_type = match_row.iloc[0]["Reference type"] if "Reference type" in match_row.columns else ""
                    selected_usage = match_row.iloc[0]["Reference usage"] if "Reference usage" in match_row.columns else ""
            elif matched_web:
                if len(matched_web) == 1:
                    selected_link = matched_web[0]
                else:
                    
                    # Prepare prompt input: fetch web content for each
                    candidate_links_and_content = ""
                    for i, link in enumerate(matched_web, 1):
                        docs = get_docs_from_url(link, "")
                        content = "\n".join([doc.page_content for doc in docs])
                        candidate_links_and_content += (
                            f"<link_{i}>\n\n"
                            f"<link>\nLink: {link}\n</link>\n\n"
                            f"<content>\nContent:\n{content}\n</content>\n\n"
                            f"</link_{i}>\n\n"
                        )
                    selected_link = relevant_link_selection_if_multiple_matching_reference_links(
                        course_name, target_audience, row["Topic"], row["Subtopic"], row["Learning Objectives"], candidate_links_and_content, llm=llm
                    )
                
                # Find Reference type/usage from Base Outline
                match_row = base_outline_df[base_outline_df["References"] == selected_link]
                if not match_row.empty:
                    selected_type = match_row.iloc[0]["Reference type"] if "Reference type" in match_row.columns else ""
                    selected_usage = match_row.iloc[0]["Reference usage"] if "Reference usage" in match_row.columns else ""
            
            # If a link was selected, update the row and context
            if selected_link:
                final_df.at[idx, "References"] = selected_link
                final_df.at[idx, "Reference type"] = selected_type
                final_df.at[idx, "Reference usage"] = selected_usage
                
                # Clear all existing context_n columns for this LO before updating
                for col in final_df.columns:
                    if col.startswith("context_"):
                        final_df.at[idx, col] = ""
                
                # Update context_n columns for this LO
                context_chunks = []
                if "youtube.com" in selected_link:
                    video_id = get_video_id_from_url(selected_link)
                    transcript = get_transcript_with_fallback(video_id, return_text_only=False)
                    m = re.search(r"[?&]start=(\d+)", selected_link)
                    n = re.search(r"[?&]end=(\d+)", selected_link)
                    if m and n:
                        start_sec = int(m.group(1))
                        end_sec = int(n.group(1))
                        transcript = extract_transcript_segment(transcript, start_sec, end_sec)
                    lines = [f"- '{item['timestamp']}': {item['text']}" for item in transcript]
                    transcript_text = '\n'.join(lines)
                    temp_df = pd.DataFrame({"dummy": [""]})
                    temp_df = create_and_populate_columns(temp_df, transcript_text, 0, "context", 49000)
                    for col in temp_df.columns:
                        if col.startswith("context_"):
                            final_df.at[idx, col] = temp_df.at[0, col]
                
                else:
                    docs = get_docs_from_url(selected_link, "")
                    content = "\n".join([doc.page_content for doc in docs])
                    temp_df = pd.DataFrame({"dummy": [""]})
                    temp_df = create_and_populate_columns(temp_df, content, 0, "context", 49000)
                    for col in temp_df.columns:
                        if col.startswith("context_"):
                            final_df.at[idx, col] = temp_df.at[0, col]
                
                # Save and format after each row
                save_to_sheet(worksheet=final_ws, df=final_df)
                format_worksheet(final_ws)

    # Hide the columns in the most up-to-date Final Outline (after all processing)

    final_ws, final_df = get_sheet_data_and_df(sheet, worksheet_name)
    column_names = [column_name for column_name in final_df.columns if ("context" in column_name or column_name in ["source_links", "as_is_sources", "content_sources"])]
    hide_columns_by_name(worksheet=final_ws, column_names=column_names, df=final_df)

    # Hide specified sheets if they exist
    sheets_to_hide = [
        "Agents",
        "Agent logs",
        "Videos Research",
        "Video Chunks",
        "Outline Consolidation",
        "Outline Review",
        "Preliminary Research",
        "Deep Research",
        "Course Outline with LOs",
        "Topic Outline",
        "Topic Deep Research",
        "Missing Learning Objectives",
        "Revised Outline",
        "Enhanced Outline Review",
        "Enhanced Outline with LOs",
        "Topic - Revised Outline Mapping",
        "All References"
    ]

    for sheet_name in sheets_to_hide:
        hide_worksheet_by_name(sheet, sheet_name)

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


@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Generate Context from Provided References",
    "function_name": "run_reference_based_context_generator_for_all_rows",
    "user_id": st.session_state.get("role", "anonymous")
})
def run_reference_based_context_generator_for_all_rows(root_folder_id, drive, sheet, worksheet_name, course_name, target_audience, llm):
    """
    This function generates context_n columns for rows with provided references.

    :param root_folder_id: The root folder ID for Google Drive.
    :param drive: The Google Drive client.
    :param sheet: The Google Sheet object.
    :param worksheet_name: The name of the worksheet.
    :param course_name: The name of the course.
    :param target_audience: The target audience of the course.
    :param llm: The language model to use.
    """
    
    # Read the sheet
    course_outline_with_lo_sheet, course_outline_with_lo_df = get_sheet_data_and_df(
        sheet=sheet,
        sheet_name=worksheet_name
    )

    # Load Base Outline once so we can inject reference columns per-row at submission time.
    # (Final Outline does not carry References / Reference type / Reference usage — those live
    # in Base Outline.  We do NOT merge them into the DataFrame because pandas merge resets
    # the index, breaking all subsequent .at[index, col] writes.)
    ref_cols = ["References", "Reference type", "Reference usage"]
    base_outline_ref_lookup = {}   # keyed by (Topic, Subtopic) → dict of ref values
    try:
        _, base_outline_df = get_sheet_data_and_df(sheet=sheet, sheet_name="Base Outline")
        has_ref_cols = all(c in base_outline_df.columns for c in ref_cols)
        has_keys = "Subtopic" in base_outline_df.columns and "Topic" in base_outline_df.columns
        if has_ref_cols and has_keys:
            for _, bo_row in base_outline_df.iterrows():
                key = (str(bo_row.get("Topic", "")).strip(), str(bo_row.get("Subtopic", "")).strip())
                if key[1]:  # only if Subtopic is non-empty
                    ref_val = str(bo_row.get("References", "")).strip()
                    if ref_val and ref_val != "nan":
                        base_outline_ref_lookup[key] = {
                            "References": ref_val,
                            "Reference type": str(bo_row.get("Reference type", "")).strip(),
                            "Reference usage": str(bo_row.get("Reference usage", "")).strip(),
                        }
            print(f"[run_reference_based_context_generator] Built Base Outline ref lookup with {len(base_outline_ref_lookup)} entries.")
    except Exception as e:
        print(f"[run_reference_based_context_generator] Could not load Base Outline references: {e}")
    
    # BACKUP: Store current context_n state for ALL rows
    backup_data = []
    for index, row in course_outline_with_lo_df.iterrows():
        backup_row = {"row_index": index}
        # Store all context_n columns for this row 
        for col in course_outline_with_lo_df.columns:
            if col.startswith('context_'):
                backup_row[col] = row[col]
        backup_data.append(backup_row)
    
    # Create backup DataFrame and save to hidden sheet
    if backup_data:
        backup_df = pd.DataFrame(backup_data)
        backup_ws, _ = create_or_read_worksheet(sheet, "Context Column Backup")
        save_to_sheet(backup_ws, backup_df)
        format_worksheet(backup_ws)
        hide_worksheet_by_name(sheet, "Context Column Backup")
    
    # Check if there are any rows with references — check both the sheet's own columns
    # and the Base Outline lookup we built above.
    def _has_reference(row):
        # Check columns already present in the target sheet
        ref_in_sheet = str(row.get("References", "")).strip()
        if ref_in_sheet and ref_in_sheet != "nan":
            return True
        # Check Base Outline lookup
        key = (str(row.get("Topic", "")).strip(), str(row.get("Subtopic", "")).strip())
        return key in base_outline_ref_lookup

    rows_with_refs = course_outline_with_lo_df[
        course_outline_with_lo_df.apply(_has_reference, axis=1)
    ]
    
    if rows_with_refs.empty:
        print('No rows with references found')
        return
    
    # Process ALL rows with references
    rows_needing_processing = rows_with_refs
    
    print(f'Found {len(rows_needing_processing)} rows with references to process')
    
    # Load the vector retriever (for fallback cases)
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
        # Submit tasks for each row that needs processing
        for index, row in rows_needing_processing.iterrows():
            # Inject reference columns from Base Outline if not already in this row
            row = row.copy()
            ref_in_row = str(row.get("References", "")).strip()
            if (not ref_in_row or ref_in_row == "nan"):
                key = (str(row.get("Topic", "")).strip(), str(row.get("Subtopic", "")).strip())
                bo_refs = base_outline_ref_lookup.get(key, {})
                if bo_refs:
                    for col, val in bo_refs.items():
                        row[col] = val
                    print(f"[run_reference_based_context_generator] Injected refs for row {index}: {bo_refs.get('Reference type')} / {bo_refs.get('References', '')[:60]}")
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
    save_interval = 1
    total_tasks = len(futures)
    
    # Initialize the progress tracker
    progress = SmartProgressBar(total_tasks=total_tasks, description="Percent complete", save_interval=save_interval)
    
    for future in tqdm(as_completed(futures), total=total_tasks):
        index, context, source_links, as_is_sources, content_sources, web_links, video_links = future.result()
        
        # Clear ALL context_n columns for this row first
        for col in course_outline_with_lo_df.columns:
            if col.startswith('context_'):
                course_outline_with_lo_df.at[index, col] = ""
        
        # Update the row in the DataFrame with new content
        course_outline_with_lo_df = create_and_populate_columns(
            df=course_outline_with_lo_df,
            text=context,
            specific_index=index,
            col_base_name='context',
            chunk_size=49000
        )
        
        # Update progress
        progress.update()
        
        # Check if we should save
        if progress.should_save():
            print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
            save_to_sheet(worksheet=course_outline_with_lo_sheet, df=course_outline_with_lo_df)
    
    # Final save to sheet after all tasks
    print('All rows processed. Saving final DataFrame to sheet.')
    save_to_sheet(worksheet=course_outline_with_lo_sheet, df=course_outline_with_lo_df)
    format_worksheet(course_outline_with_lo_sheet)
    
    return


def delete_reference_based_context(sheet, worksheet_name="Final Outline"):
    """Restore context_n columns for ALL rows to their previous state."""
    
    # Read current sheet
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    
    # Read backup from hidden sheet
    try:
        backup_ws = sheet.worksheet("Context Column Backup")
        backup_df = pd.DataFrame(backup_ws.get_all_records())
        
        # Restore context_n columns for ALL rows
        for _, backup_row in backup_df.iterrows():
            index = backup_row["row_index"]
            if index in df.index:
                # First, clear ALL context_n columns for this row
                for col in df.columns:
                    if col.startswith('context_'):
                        df.at[index, col] = ""
                
                # Then restore only the columns that existed in backup
                for col in backup_row.index:
                    if col.startswith('context_'):
                        df.at[index, col] = backup_row[col]
        
        # Delete the backup sheet
        delete_worksheet(sheet, "Context Column Backup")
        
    except Exception as e:
        print(f"Could not restore backup: {e}")
        # Fallback: just clear context_n columns for rows with references
        for index, row in df.iterrows():
            if str(row.get("References", "")).strip():
                for col in df.columns:
                    if col.startswith('context_'):
                        df.at[index, col] = ""
    
    # Save the restored sheet
    clear_worksheet(ws)
    save_to_sheet(ws, df)


def delete_retriever_context(sheet, worksheet_name="Final Outline", drive=None, root_folder_id=None, course_name=None):
    """Remove context columns and reference links from Final Outline sheet.

    Also deletes:
    - 'All References' sheet — so the next run of Step 1 rebuilds it from current
      External References / Base Outline references (avoids stale vector DB).
    - 'Vectorstore files' and 'Pickle files' folders from Google Drive (if drive + root_folder_id provided)
    - Local /tmp Chroma DB folder for the course
    - Local /tmp BM25 pickle file
    """
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    # Check Outline Stage
    try:
        course_info_ws = sheet.worksheet("Course info")
        course_info_df = pd.DataFrame(course_info_ws.get_all_records())
        course_info_df.columns = [col.strip() for col in course_info_df.columns]
        outline_stage = course_info_df["Outline Stage"].dropna().astype(str).str.strip().str.lower().iloc[0]
    except Exception as e:
        print(f"Failed to read 'Outline Stage' from 'Course Info' sheet: {e}")
        outline_stage = None
    cols = [c for c in df.columns if c.startswith("context_") or c in ["source_links", "as_is_sources", "content_sources", "web_links", "video_links"]]
    # If Initial Outline, also delete reference columns
    if outline_stage == "initial":
        cols += [c for c in ["References", "Reference type", "Reference usage"] if c in df.columns]
    if cols:
        df = df.drop(columns=cols)
        clear_worksheet(ws)
        save_to_sheet(ws, df)

    # Delete the All References sheet so Step 1 must regenerate it from current
    # External References / Base Outline references on the next run.
    # Without this, re-running Step 2 would rebuild the same stale vector DB from
    # an unchanged All References sheet, ignoring any edits to External References.
    try:
        delete_worksheet(sheet, "All References")
        print("[delete_retriever_context] Deleted 'All References' sheet — will be rebuilt fresh on next run.")
    except Exception as e:
        print(f"[delete_retriever_context] Could not delete 'All References' sheet (may not exist): {e}")


    if drive is not None and root_folder_id:
        for folder_name in ["Vectorstore files", "Pickle files"]:
            try:
                query = (
                    f"title='{folder_name}' and '{root_folder_id}' in parents "
                    f"and mimeType='application/vnd.google-apps.folder' and trashed=false"
                )
                folders = drive.ListFile({'q': query}).GetList()
                if not folders:
                    print(f"[delete_retriever_context] '{folder_name}' folder not found in Drive — skipping.")
                    continue
                for folder in folders:
                    drive.CreateFile({'id': folder['id']}).Delete()
                    print(f"[delete_retriever_context] Deleted Drive folder: '{folder_name}' (id={folder['id']})")
            except Exception as e:
                print(f"[delete_retriever_context] Failed to delete '{folder_name}' from Drive: {e}")

    # Clear local /tmp Chroma DB
    try:
        if course_name:
            short_name = get_short_name(course_name)
            local_chroma_path = os.path.join("/tmp", "temp_chroma_folder", f"{short_name}_chroma_research_db")
        else:
            local_chroma_path = os.path.join("/tmp", "temp_chroma_folder")

        if os.path.exists(local_chroma_path):
            # Step 1: Try to release ChromaDB file handles by clearing its singleton cache.
            # The internal attribute name varies across chromadb versions.
            try:
                import gc
                from chromadb.api.client import SharedSystemClient
                # Try known attribute names across different chromadb versions
                released = False
                for attr in ("_identifer_to_system", "_identifier_to_system", "_instances", "_system_cache"):
                    cache = getattr(SharedSystemClient, attr, None)
                    if cache is not None:
                        cache.clear()
                        released = True
                        break
                gc.collect()
                print(
                    f"[delete_retriever_context] ChromaDB singleton cache cleared (released={released})."
                )
            except Exception as _e:
                print(f"[delete_retriever_context] Could not release ChromaDB handles: {_e}")

            # Step 2: Attempt deletion
            try:
                shutil.rmtree(local_chroma_path)
                st.session_state.pop("chroma_db_path_override", None)
                print(f"[delete_retriever_context] Cleared local Chroma DB at: {local_chroma_path}")
            except OSError as e:
                if getattr(e, "winerror", None) == 32 or "WinError 32" in str(e):
                    # Still locked — store a fresh path in session state for the next run
                    import time
                    fresh_path = local_chroma_path + f"_fresh_{int(time.time())}"
                    st.session_state["chroma_db_path_override"] = fresh_path
                    print(f"[delete_retriever_context] DB locked — next run uses: {fresh_path}")
                    st.warning(
                        "⚠️ The local vector DB is in use and could not be deleted. "
                        "It will be **rebuilt fresh** on the next run automatically."
                    )
                else:
                    raise
        else:
            st.session_state.pop("chroma_db_path_override", None)
            print(f"[delete_retriever_context] Local Chroma DB not found — skipping.")
    except Exception as e:
        print(f"[delete_retriever_context] Failed to clear local Chroma DB: {e}")

    # Clear local /tmp BM25 pickle file
    try:
        local_pickle_path = "/tmp/bm25_retriever.pkl"
        if os.path.exists(local_pickle_path):
            os.remove(local_pickle_path)
            print(f"[delete_retriever_context] Cleared local BM25 pickle at: {local_pickle_path}")
        else:
            print(f"[delete_retriever_context] Local BM25 pickle not found — skipping.")
    except Exception as e:
        print(f"[delete_retriever_context] Failed to clear local BM25 pickle: {e}")