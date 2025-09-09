from modules.chain import Chain
from services.helper_functions import get_outline_with_los
from tqdm import tqdm
from services.sheets_service import get_sheet_data_and_df, save_to_sheet, clear_worksheet
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
from services.smart_progress_bar import SmartProgressBar
from langsmith import traceable
from services.youtube_video_loader import convert_time_to_sec, get_video_id_from_url
import re
import pandas as pd


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


generate_transcript_chunk_extraction_prompt = """You are an expert educational content developer tasked with analyzing a timestamped transcript to extract the most relevant segment for a specific learning objective within a subtopic of an E-learning course. Your goal is to identify and return the exact transcript chunk that best supports the given learning objective.

Before we begin, please review the following course information:

Timestamped Transcript:
<timestamped_transcript>
{relevant_documents}
</timestamped_transcript>

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

Subtopic and Learning Objective to Focus On:
<subtopic_and_los>
{subtopic_and_los}
</subtopic_and_los>

Now, follow these steps to identify and extract the relevant chunk from the given timestamped transcript. For each step, wrap your reasoning and internal analysis inside the specified XML tags to show your thinking process:

1. <objective_analysis>
   - Break down the learning objective into key ideas or skills the learner should understand and master.
   - Clarify what kind of transcript content would fulfill this objective.
</objective_analysis>

2. <examine_transcript>
   - Carefully read the transcript and identify the one segment that most directly supports the learning objective.
   - Do not summarize or rephrase the transcript — just locate the most relevant portion.
</examine_transcript>

3. <extract_relevant_chunk>
   - Extract the most relevant chunk of transcript.
   - Format your output as follows:

       Start: (insert start timestamp in seconds)
       End: (insert end timestamp in seconds)
       Transcript:
         - '(insert timestamp in seconds)': (insert transcript text for this timestamp)
         - '(insert timestamp in seconds)': (insert transcript text for this timestamp)

         - ...
         - '(insert timestamp in seconds)': (insert transcript text for this timestamp)

   - Formatting rules:
     • Use exactly 2 spaces before each transcript line
     • Use single quotes around each timestamp
     • Do not add any commentary, notes, or tags in the output
</extract_relevant_chunk>

<output>
Paste your final result here using the format shown above. Only one chunk should be returned for the single learning objective. Do not include anything else outside the output.
</output>

Note: Strictly remember to always enclose your final output of timestamped transcript chunk inside the <output> .... </output> tags as shown above.

Important Rules and Constraints:

- Do not paraphrase, rephrase, or summarize the transcript in the output.  
- Do not generate new sentences or explanations — all output must be copied directly from the provided transcript lines.
- Only include the single most relevant chunk that clearly and directly supports the learning objective.
- Do not include loosely related or general content — the match must be tight and objective-specific.
- Do not include any unrelated lines before or after the chunk — trim precisely to the relevant start and end.
- Do NOT include any commentary, explanations, labels, or metadata outside the format shown.
- The value you provide for "Start" must match the timestamp of the first transcript line you return.
- The value you provide for "End" must represent when the last transcript line ends — not just its timestamp. For example, if the final transcript line starts at '621' and continues until 625, then End should be 625.
- There is no constraint on the length of the chunk — it may be as short or as long as needed to fully satisfy the given learning objective.
- Format exactly as shown in the output example: start/end timestamps and indented line-by-line transcript with single quotes and exact spacing.
"""


def generate_transcript_chunks(course_name, target_audience, course_outline, subtopic_and_los, relevant_documents, llm='gemini_2_flash'):
    """
    This function extracts relevant transcript chunks aligned with each learning objective.

    :param course_name: The name of the course.
    :param target_audience: The target audience of the course.
    :param course_outline: The outline of the course.
    :param subtopic_and_los: The subtopic and learning objectives to focus on.
    :param relevant_documents: The timestamped transcript as input.
    :param llm: The language model to use.
    :return: Extracted transcript chunks.
    """

    generate_transcript_chunks_agent = Chain(llm = llm, tags = ['output'])

    generate_transcript_chunks_agent.add_message(
        role ='user',
        content = generate_transcript_chunk_extraction_prompt.format(
            course_name = course_name,
            target_audience = target_audience,
            full_course_outline = course_outline,
            subtopic_and_los = subtopic_and_los,
            relevant_documents = relevant_documents
        )
    )

    response = generate_transcript_chunks_agent.run()
    return response['output']


@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Researcher",
    "function_name": "run_research_notes_agent_for_all_rows",
    "user_id": st.session_state.get("role", "anonymous")
})
def run_research_notes_agent_for_all_rows(sheet, worksheet_name, course_name, target_audience, llm='gemini_2_flash'):
    """
    This function is used to generate research_notes for all rows.

    :param sheet: The sheet object.
    :param worksheet_name: The worksheet name.
    :param course_name: The name of the course.
    :param target_audience: The target audience of the course.
    :param llm: The language model to use.
    :return: None    
    """
    
    # Read Outline Stage from Course info
    try:
        course_info_ws = sheet.worksheet("Course info")
        course_info_df = pd.DataFrame(course_info_ws.get_all_records())
        course_info_df.columns = [col.strip() for col in course_info_df.columns]
        outline_stage = course_info_df["Outline Stage"].dropna().astype(str).str.strip().str.lower().iloc[0]
    except Exception as e:
        print(f"Failed to read 'Outline Stage' from 'Course Info' sheet: {e}")
        outline_stage = None

    if outline_stage == "final":

        # Read the sheet and df
        course_outline_with_lo_sheet, course_outline_with_lo_df = get_sheet_data_and_df(
            sheet=sheet, 
            sheet_name=worksheet_name
        )

        # Create column for research notes if not already present
        if 'research_notes' not in course_outline_with_lo_df.columns:
            course_outline_with_lo_df['research_notes'] = ''

        # Check if this step is already done by checking if ALL rows have research_notes populated
        if (course_outline_with_lo_df['research_notes'] != '').all():
            print('Research notes already populated for all rows')
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

                # Check if the three reference columns exist and have values
                required_cols = ["References", "Reference type", "Reference usage"]
                if all(col in row.index for col in required_cols):
                    ref = str(row["References"]).strip()
                    ref_type = str(row["Reference type"]).strip()
                    ref_usage = str(row["Reference usage"]).strip()
                    
                    # Check if the columns have values
                    if ref and ref_type and ref_usage:
                        # Case: Youtube Video or Google Drive Video + Video usage - use transcript chunk extraction
                        if (ref_type == "Youtube Video" or ref_type == "Google Drive Video") and ref_usage == "Video":
                            # Convert transcript format from MM:SS to seconds for generate_transcript_chunks
                            pattern = r"- '(\d{1,2}:\d{2}(?::\d{2})?)': (.+)"
                            
                            # Replace MM:SS format to seconds format
                            converted_context = re.sub(pattern, lambda match: f"- {convert_time_to_sec(match.group(1))}: {match.group(2)}", context)
                            
                            future = executor.submit(
                                generate_transcript_chunks,
                                course_name=course_name,
                                target_audience=target_audience,
                                course_outline=course_outline_with_lo,
                                subtopic_and_los=row['Subtopic'] + '\n\n' + row['Learning Objectives'],
                                relevant_documents=converted_context,
                                llm=llm
                            )
                        else:
                            # For all other cases, use the existing workflow
                            future = executor.submit(
                                generate_research_notes,
                                course_name=course_name,
                                target_audience=target_audience,
                                course_outline=course_outline_with_lo,
                                subtopic_and_los=row['Subtopic'] + '\n\n' + row['Learning Objectives'],
                                relevant_documents=context,
                                llm=llm,
                            )
                    else:
                        # Reference columns exist but are empty, use existing workflow
                        future = executor.submit(
                            generate_research_notes,
                            course_name=course_name,
                            target_audience=target_audience,
                            course_outline=course_outline_with_lo,
                            subtopic_and_los=row['Subtopic'] + '\n\n' + row['Learning Objectives'],
                            relevant_documents=context,
                            llm=llm,
                        )
                else:
                    # Reference columns don't exist, use existing workflow
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

                # Check if this row used transcript chunk extraction (YouTube Video + Video usage)
                row = course_outline_with_lo_df.iloc[index]
                required_cols = ["References", "Reference type", "Reference usage"]
                if all(col in row.index for col in required_cols):
                    ref = str(row["References"]).strip()
                    ref_type = str(row["Reference type"]).strip()
                    ref_usage = str(row["Reference usage"]).strip()
                    
                    if ref and ref_type and ref_usage and (ref_type == "Youtube Video" or ref_type == "Google Drive Video") and ref_usage == "Video":
                        # Extract video ID from the URL or file ID from Google Drive link
                        if ref_type == "Youtube Video":
                            video_id = get_video_id_from_url(ref)
                            start_match = re.search(r'Start: (\d+)', research_notes)
                            end_match = re.search(r'End: (\d+)', research_notes)
                            if start_match and end_match:
                                start_time = start_match.group(1)
                                end_time = end_match.group(1)
                                link_with_params = f"https://www.youtube.com/embed/{video_id}?start={start_time}&end={end_time}"
                                research_notes = f"Link: {link_with_params}\nVideo_Id: {video_id}\n{research_notes}"
                        elif ref_type == "Google Drive Video":
                            match = re.search(r'/d/([\w-]+)', ref)
                            file_id = match.group(1) if match else ''
                            research_notes = f"Link: {ref}\nVideo_ID: {file_id}\n{research_notes}"


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
    elif outline_stage == "initial":
        # Read the sheet and df
        course_outline_with_lo_sheet, course_outline_with_lo_df = get_sheet_data_and_df(
            sheet=sheet, 
            sheet_name=worksheet_name
        )

        # Create column for research notes if not already present
        if 'research_notes' not in course_outline_with_lo_df.columns:
            course_outline_with_lo_df['research_notes'] = ''

        # Check if this step is already done by checking if ALL rows have research_notes populated
        if (course_outline_with_lo_df['research_notes'] != '').all():
            print('Research notes already populated for all rows')
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

                # Check if the three reference columns exist and have values
                required_cols = ["References", "Reference type", "Reference usage"]
                if all(col in row.index for col in required_cols):
                    ref = str(row["References"]).strip()
                    ref_type = str(row["Reference type"]).strip()
                    ref_usage = str(row["Reference usage"]).strip()
                    
                    # If all reference columns are non-empty
                    if ref and ref_type and ref_usage:
                        
                        # Case: Web Article + Content usage - use generate_research_notes
                        if ref_type == "Web Article" and ref_usage == "Content":
                            future = executor.submit(
                                generate_research_notes,
                                course_name=course_name,
                                target_audience=target_audience,
                                course_outline=course_outline_with_lo,
                                subtopic_and_los=row['Subtopic'] + '\n\n' + row['Learning Objectives'],
                                relevant_documents=context,
                                llm=llm,
                            )
                        
                        # Case: Youtube Video + Content usage - use generate_research_notes
                        elif ref_type == "Youtube Video" and ref_usage == "Content":
                            future = executor.submit(
                                generate_research_notes,
                                course_name=course_name,
                                target_audience=target_audience,
                                course_outline=course_outline_with_lo,
                                subtopic_and_los=row['Subtopic'] + '\n\n' + row['Learning Objectives'],
                                relevant_documents=context,
                                llm=llm,
                            )
                        
                        # Case: Youtube Video + Video usage - custom transcript formatting
                        elif ref_type == "Youtube Video" and ref_usage == "Video":
                            def format_youtube_video_notes(ref, context):
                                # ✅ Extract video id from embed link
                                video_id_match = re.search(r"/embed/([\w-]+)", ref)
                                video_id = video_id_match.group(1) if video_id_match else ''

                                # ✅ Extract start/end parameters directly from the Reference link
                                start_match = re.search(r"[?&]start=(\d+)", ref)
                                end_match = re.search(r"[?&]end=(\d+)", ref)
                                start = start_match.group(1) if start_match else ''
                                end = end_match.group(1) if end_match else ''

                                # ✅ Convert transcript timestamps (MM:SS → seconds)
                                pattern = r"- '([\d:]+)': (.+)"
                                def mmss_to_sec(ts):
                                    parts = ts.split(":")
                                    if len(parts) == 2:  # MM:SS
                                        return str(int(parts[0]) * 60 + int(parts[1]))
                                    elif len(parts) == 3:  # HH:MM:SS
                                        return str(int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2]))
                                    return ts

                                transcript_lines = []
                                for match in re.finditer(pattern, context):
                                    sec = mmss_to_sec(match.group(1))
                                    text = match.group(2)
                                    transcript_lines.append(f"  - '{sec}': {text}")
                                transcript = '\n'.join(transcript_lines)

                                # ✅ Format the research notes — always keep Reference start/end
                                return (
                                    f"Link: {ref}\n"
                                    f"Video_Id: {video_id}\n"
                                    f"Start: {start}\n"
                                    f"End: {end}\n"
                                    f"Transcript:\n{transcript}"
                                )


                            future = executor.submit(format_youtube_video_notes, ref, context)
                        
                        # All other cases, use generate_research_notes
                        else:
                            future = executor.submit(
                                generate_research_notes,
                                course_name=course_name,
                                target_audience=target_audience,
                                course_outline=course_outline_with_lo,
                                subtopic_and_los=row['Subtopic'] + '\n\n' + row['Learning Objectives'],
                                relevant_documents=context,
                                llm=llm,
                            )
                    
                    # If any reference column is empty, use generate_research_notes
                    else:
                        future = executor.submit(
                            generate_research_notes,
                            course_name=course_name,
                            target_audience=target_audience,
                            course_outline=course_outline_with_lo,
                            subtopic_and_los=row['Subtopic'] + '\n\n' + row['Learning Objectives'],
                            relevant_documents=context,
                            llm=llm,
                        )
                
                # If reference columns are missing, use generate_research_notes
                else:
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
            save_interval = 5
            progress = SmartProgressBar(total_tasks = total_tasks, description = "Percent complete", save_interval = save_interval)
            for future in tqdm(as_completed(futures_map), total=total_tasks):
                index = futures_map[future]
                research_notes = future.result()
                course_outline_with_lo_df.loc[index, 'research_notes'] = research_notes
                progress.update()
                if progress.should_save():
                    print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
                    save_to_sheet(worksheet = course_outline_with_lo_sheet, df = course_outline_with_lo_df)
        
        # Final save to sheet after all tasks
        print('All rows processed. Saving final DataFrame to sheet.')
        save_to_sheet(worksheet = course_outline_with_lo_sheet, df = course_outline_with_lo_df)
        return
    else:
        raise ValueError(f"Unknown outline stage: {outline_stage}")


def delete_research_notes(sheet, worksheet_name="Final Outline"):
    """Remove the research_notes column from the specified worksheet."""
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    if "research_notes" in df.columns:
        df = df.drop(columns=["research_notes"])
        clear_worksheet(ws)
        save_to_sheet(ws, df)