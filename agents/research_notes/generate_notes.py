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
from services.youtube_video_loader import get_transcript_with_fallback, load_video_chunks_from_local_or_drive
from agents.research_notes.review_revise_research_notes import generate_review_revise_research_notes
from agents.research_notes.retriever import get_compression_retriever, get_web_search_retriever



generate_research_notes_prompt = """You are an expert educational content developer tasked with creating comprehensive research notes for a specific subtopic within a larger course.

Relevant Documents/Transcripts:
<relevant_documents_or_transcripts>
{relevant_documents}
</relevant_documents_or_transcripts>

Course Name: {course_name}

Target Audience: {target_audience}

Full Course Outline with Learning Objectives:
<full_course_outline>
{full_course_outline}
</full_course_outline>

Subtopic and Learning Objectives to Focus On:
<subtopic_and_los>
{subtopic_and_los}
</subtopic_and_los>

Follow these Paraphrasing Rules when generating the research notes:
<paraphrasing_rules>
1. Use plain language, not corporate jargon
2. Sound like a real technician, not an AI
3. Keep all factual information accurate (never add or remove facts)
4. Use active voice and short sentences
5. Explain trade terminology when first used
6. Be conversational but professional
7. No buzzwords like "optimal," "leverage," "utilize," "facilitate"
8. No hype or fluff - every sentence adds value

Examples:
BAD: "Technicians must de-energize the system to mitigate hazards prior to engaging with components."
GOOD: "Shut the power off before you touch anything—it keeps you safe."

BAD: "Ensure optimal airflow parameters."
GOOD: "Good airflow keeps the system running right."

BAD: "Apply pookie to the joints."
GOOD: "Apply mastic—also called pookie by tradesman—to seal the joints."
</paraphrasing_rules>


Generate research notes for the given subtopic and learning objectives. Always provide your output strictly in the following format:

<output>

<analysis>
Briefly identify the key concepts from the learning objectives and note which parts of the provided documents are most relevant.
</analysis>

<research_notes>
Write well-structured, comprehensive notes that:
- Directly address each learning objective
- Use only information from the provided documents (ignore timestamps if transcripts are provided)
- Are written in clear, engaging paragraphs appropriate for the target audience
- Include relevant examples and definitions from the source material
- Exclude information better suited for other sections of the course outline
</research_notes>

</output>

CRITICAL FORMAT REQUIREMENT: Return output wrapped in <output>...</output>. Inside it, use exactly these XML tags in this order and close all tags: <analysis>...</analysis> then <research_notes>...</research_notes>. Do not include any text outside <output>...</output>.
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

    generate_research_notes_agent = Chain(llm = llm, tags = ['research_notes'])

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

    # Check if research notes is under character limit
    if len(response['research_notes']) > 50000:
        print(f'Research notes are too long - {len(response["research_notes"])}. Summarizing...')
        generate_research_notes_agent.add_message(
            role = 'user',
            content = f"The research notes are too long. The current length is {len(response['research_notes'])} characters. Please summarize them to be under 50000 characters. Make sure to output in the same format as above."
        )
        response = generate_research_notes_agent.run()

    return response['research_notes'] if len(response['research_notes']) < 50000 else response['research_notes'][:49990]


generate_transcript_chunk_extraction_prompt = """You are an expert educational content developer tasked with analyzing timestamped transcripts from one or more videos to extract the most relevant segments for a specific learning objective within a subtopic of an E-learning course. Your goal is to identify and return the exact transcript chunks that best support the given learning objective.

Before we begin, please review the following course information:

The timestamped transcripts from the referenced videos are provided below:
<timestamped_transcripts>
{relevant_documents}
</timestamped_transcripts>

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

Now, follow these steps to identify and extract the relevant chunks from the given timestamped transcripts. For each step, wrap your reasoning and internal analysis inside the specified XML tags to show your thinking process:

Make sure your output is strictly enclosed within the <output> ... </output> tags as shown below. Do NOT include any other text or commentary outside these tags.
<output>
1. <objective_analysis>
- Break down the learning objective into key ideas or skills the learner should understand and master.
- Clarify what kind of transcript content would fulfill this objective.
</objective_analysis>

2. <examine_transcripts>
- Carefully read the video transcript(s) and identify which video(s) contain content that directly supports the learning objective.
- If transcripts of multiple videos are provided, it is not mandatory to extract relevant chunks from every video - only select videos that are directly relevant to the learning objective.
- If only one video transcript is provided, carefully read the transcript and identify the one segment that most directly supports the learning objective.
- For each relevant video, identify the one segment that most directly supports the learning objective. Ensure that there is no overlapping or duplicate content between the chunks that you identify as relevant in case of multiple videos.

</examine_transcripts>

3. <extract_relevant_chunks>
- Extract the most relevant chunk of transcript.
- Format your output as follows:

Video_Id: (insert the exact video id of the video chunk)
    
Start: (insert start timestamp in seconds)
End: (insert end timestamp in seconds)
Transcript:
    - '(insert timestamp in seconds)': (insert transcript text for this timestamp)
    - '(insert timestamp in seconds)': (insert transcript text for this timestamp)
    - ...

---

Video id: (insert the exact video id of the video chunk)

Start: (insert start timestamp in seconds)
End: (insert end timestamp in seconds)
Transcript:
    - '(insert timestamp in seconds)': (insert transcript text for this timestamp)
    - '(insert timestamp in seconds)': (insert transcript text for this timestamp)
    - ...

---
(Continue this pattern for each relevant video chunk that you identify)

- Formatting rules:
     • Use exactly 2 spaces before each transcript line
     • Use single quotes around each timestamp
     • Include the exact video_id from the transcript header (e.g., "Video 1 (ID: dQw4w9WgXcQ):" - use "dQw4w9WgXcQ")
     • Separate each video chunk with "---" on its own line
     • Do not add any commentary, notes, or tags in the output
     • If only one video is relevant, output only one chunk without "---"
</extract_relevant_chunks>
</output>

Important Rules and Constraints:

- Do not paraphrase, rephrase, or summarize the transcripts in the output. The transcript lines of the identified chunks must be identical to the ones in the input.
- Do not generate new sentences or explanations — all output must be copied directly from the provided transcript lines.  
- In case of a single video, only include the single most relevant chunk that clearly and directly supports the learning objective
- In case of multiple videos, only identify and extract chunks from videos that are directly relevant to the learning objective - you do not need to use every video. The transcript content of each chunk should be distinct and non-overlapping with other chunks.
- If the relevant chunk spans from second 10 to 25, your output must start **exactly at 10** and end **exactly at 25** — do not include unrelated transcript lines before or after.  
- The value you provide for "Start" must match the timestamp of the first transcript line you return.  
- The value you provide for "End" must represent when the last transcript line ends — not just its timestamp. For example, if the final transcript line starts at '621' and continues until 625, then End should be 625 (not 621).  
- There is no constraint on the length of each chunk — it may be as short or as long as needed to fully satisfy the given learning objective.  
- Format exactly as shown in the output example: video ID, start/end timestamps and indented line-by-line transcript with single quotes and exact spacing.  
- Use "---" to separate chunks from different videos, but only if you're extracting from multiple videos.

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
    
    # Extract only the content inside <extract_relevant_chunks> tags
    match = re.search(r'<extract_relevant_chunks>(.*?)</extract_relevant_chunks>', response['output'], re.DOTALL)
    return match.group(1).strip()


@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Researcher",
    "function_name": "run_research_notes_agent_for_all_rows",
    "user_id": st.session_state.get("role", "anonymous")
})
def run_research_notes_agent_for_all_rows(sheet, worksheet_name, course_name, target_audience, llm='gemini_2_flash', _drive=None):
    """
    Generate research_notes for all rows in the course outline.

    :param sheet: Google Sheet object
    :param worksheet_name: Name of the worksheet
    :param course_name: Course name
    :param target_audience: Target audience
    :param llm: Language model to use
    :param _drive: Google Drive client (required for loading video transcripts)
    :return: None
    """

    def format_chunks_to_transcript(chunks):
        """Format list of video chunks into timestamped transcript string."""
        lines = []
        for chunk in chunks:
            start = chunk["metadata"].get("start_time")
            text = chunk.get("text", "").strip()
            if start is not None and text:
                lines.append(f"- '{start}': {text}")
        return "\n".join(lines)

    # Read Outline Stage from Course info
    try:
        course_info_ws = sheet.worksheet("Course info")
        course_info_df = pd.DataFrame(course_info_ws.get_all_records())
        course_info_df.columns = [col.strip() for col in course_info_df.columns]
        outline_stage = course_info_df["Outline Stage"].dropna().astype(str).str.strip().str.lower().iloc[0]
    except Exception as e:
        print(f"Failed to read 'Outline Stage' from 'Course Info' sheet: {e}")
        outline_stage = None

    # Common function for both stages
    def process_rows(course_outline_with_lo_sheet, course_outline_with_lo_df):
        # Ensure research_notes column exists
        if 'research_notes' not in course_outline_with_lo_df.columns:
            course_outline_with_lo_df['research_notes'] = ''

        if (course_outline_with_lo_df['research_notes'] != '').all():
            print('Research notes already populated for all rows')
            return

        context_col_count = len([col for col in course_outline_with_lo_df.columns if 'context_' in col])
        course_outline_with_lo = get_outline_with_los(course_outline_with_lo_df, include_learning_objectives=True)

        # Load all transcripts once
        video_chunks_dict = load_video_chunks_from_local_or_drive(_drive) if _drive else {}

        # Load retrievers once for the review-revise loop
        compression_retriever = None
        web_search_retriever = None
        try:
            import streamlit as st
            course_drive_folder_id = st.session_state.get("root_folder_id")
            compression_retriever = get_compression_retriever(course_name, course_drive_folder_id, _drive, sheet)
        except Exception as e:
            print(f"Could not load compression retriever: {e}")
        try:
            web_search_retriever = get_web_search_retriever()
        except Exception as e:
            print(f"Could not load web search retriever: {e}")

        futures_map = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            for index, row in course_outline_with_lo_df.iterrows():
                if row['research_notes'] != '':
                    print(f'Skipping row {index}. Already populated')
                    continue

                # Build fallback context
                context = ''.join([row[f'context_{i}'] for i in range(context_col_count)])

                ref = str(row.get("References", "")).strip()
                ref_type = str(row.get("Reference type", "")).strip()
                ref_usage = str(row.get("Reference usage", "")).strip()

                future = None

                if ref and ref_type and ref_usage:
                    # YouTube Video
                    if ref_type == "Youtube Video":
                        if ref_usage == "Video":
                            # Use the context that was already retrieved and processed
                            future = executor.submit(
                                generate_transcript_chunks,
                                course_name=course_name,
                                target_audience=target_audience,
                                course_outline=course_outline_with_lo,
                                subtopic_and_los=row['Subtopic'] + '\n\n' + row['Learning Objectives'],
                                relevant_documents=context,
                                llm=llm
                            )
                        elif ref_usage == "Content":
                            future = executor.submit(
                                generate_review_revise_research_notes,
                                course_name=course_name,
                                target_audience=target_audience,
                                course_outline=course_outline_with_lo,
                                subtopic_and_los=row['Subtopic'] + '\n\n' + row['Learning Objectives'],
                                relevant_documents=context,
                                row_id=index,
                                sheet=sheet,
                                compression_retriever=compression_retriever,
                                web_search_retriever=web_search_retriever,
                                llm=llm
                            )

                    # Google Drive Video
                    elif ref_type == "Google Drive Video":
                        if ref_usage == "Video":
                            # Use the context that was already retrieved and processed
                            future = executor.submit(
                                generate_transcript_chunks,
                                course_name=course_name,
                                target_audience=target_audience,
                                course_outline=course_outline_with_lo,
                                subtopic_and_los=row['Subtopic'] + '\n\n' + row['Learning Objectives'],
                                relevant_documents=context,
                                llm=llm
                            )
                        elif ref_usage == "Content":
                            future = executor.submit(
                                generate_review_revise_research_notes,
                                course_name=course_name,
                                target_audience=target_audience,
                                course_outline=course_outline_with_lo,
                                subtopic_and_los=row['Subtopic'] + '\n\n' + row['Learning Objectives'],
                                relevant_documents=context,
                                row_id=index,
                                sheet=sheet,
                                compression_retriever=compression_retriever,
                                web_search_retriever=web_search_retriever,
                                llm=llm
                            )

                    elif ref_type == "Web Article" and ref_usage == "Content":
                        future = executor.submit(
                            generate_review_revise_research_notes,
                            course_name=course_name,
                            target_audience=target_audience,
                            course_outline=course_outline_with_lo,
                            subtopic_and_los=row['Subtopic'] + '\n\n' + row['Learning Objectives'],
                            relevant_documents=context,
                            row_id=index,
                            sheet=sheet,
                            compression_retriever=compression_retriever,
                            web_search_retriever=web_search_retriever,
                            llm=llm,
                        )

                # Fallback when no reference is provided
                if not future:
                    future = executor.submit(
                        generate_review_revise_research_notes,
                        course_name=course_name,
                        target_audience=target_audience,
                        course_outline=course_outline_with_lo,
                        subtopic_and_los=row['Subtopic'] + '\n\n' + row['Learning Objectives'],
                        relevant_documents=context,
                        row_id=index,
                        sheet=sheet,
                        compression_retriever=compression_retriever,
                        web_search_retriever=web_search_retriever,
                        llm=llm
                    )

                futures_map[future] = index

            # Collect results
            total_tasks = len(futures_map)
            progress = SmartProgressBar(total_tasks=total_tasks, description="Percent complete", save_interval=5)

            for future in tqdm(as_completed(futures_map), total=total_tasks):
                index = futures_map[future]
                research_notes = future.result()

                # Post-process links
                row = course_outline_with_lo_df.iloc[index]
                ref = str(row.get("References", "")).strip()
                ref_type = row.get("Reference type")
                ref_usage = row.get("Reference usage")

                if ref and ref_type and ref_usage and (ref_type in ["Youtube Video", "Google Drive Video"]) and ref_usage == "Video":
                        if ref_type == "Youtube Video":
                            # Parse multiple video chunks from LLM output
                            video_chunks = []
                            chunks_text = research_notes
                            
                            # Split by "---" to get individual chunks
                            chunk_sections = chunks_text.split('---')
                            
                            for chunk_section in chunk_sections:
                                chunk_section = chunk_section.strip()
                                if not chunk_section:
                                    continue
                                    
                                # Extract Video id, Start, and End from each chunk
                                video_id_match = re.search(r'Video[ _]?[iI]d:\s*([^\n]+)', chunk_section)
                                start_match = re.search(r'Start:\s*(\d+)', chunk_section)
                                end_match = re.search(r'End:\s*(\d+)', chunk_section)
                                
                                if video_id_match and start_match and end_match:
                                    video_id = video_id_match.group(1).strip()
                                    start_time = start_match.group(1)
                                    end_time = end_match.group(1)
                                    
                                    # Create embed link with start and end parameters
                                    link_with_params = f"https://www.youtube.com/embed/{video_id}?start={start_time}&end={end_time}"
                                    video_chunks.append({
                                        'link': link_with_params,
                                        'video_id': video_id,
                                        'chunk_content': chunk_section
                                    })
                            
                            # Format the final research_notes with all video links
                            if video_chunks:
                                links_section = ""
                                for i, chunk in enumerate(video_chunks, 1):
                                    links_section += f"Link: {chunk['link']}\n"
                                    links_section += f"```\n{chunk['chunk_content']}\n```\n"
                                    if i < len(video_chunks):  # Add separator between chunks
                                        links_section += "\n---\n\n"
                                
                                # Replace the original research_notes with the formatted version
                                research_notes = links_section
                                
                        elif ref_type == "Google Drive Video":
                            match = re.search(r'/d/([\w-]+)', ref)
                            file_id = match.group(1) if match else ''
                            research_notes = f"Link: {ref}\nVideo_ID: {file_id}\n{research_notes}"

                course_outline_with_lo_df.loc[index, 'research_notes'] = research_notes
                progress.update()
                if progress.should_save():
                    print(f'Saving partial progress after {progress.completed_count} tasks.')
                    save_to_sheet(worksheet=course_outline_with_lo_sheet, df=course_outline_with_lo_df)

        print('All rows processed. Saving final DataFrame to sheet.')
        save_to_sheet(worksheet=course_outline_with_lo_sheet, df=course_outline_with_lo_df)

    # Run for both stages
    if outline_stage == "final":
        course_outline_with_lo_sheet, course_outline_with_lo_df = get_sheet_data_and_df(sheet, worksheet_name)
        process_rows(course_outline_with_lo_sheet, course_outline_with_lo_df)
        return
    elif outline_stage == "initial":
        course_outline_with_lo_sheet, course_outline_with_lo_df = get_sheet_data_and_df(sheet, worksheet_name)
        process_rows(course_outline_with_lo_sheet, course_outline_with_lo_df)
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
