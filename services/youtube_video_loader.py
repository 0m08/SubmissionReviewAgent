import regex as re
from urllib.parse import urlparse, parse_qs
import time
from youtube_transcript_api import YouTubeTranscriptApi
from utils.decorator_helpers import try_n_times
from langchain_core.prompts import ChatPromptTemplate
from services.llm_service import llm_with_retry, output_parser
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.documents import Document
import os
import requests
import csv
import streamlit as st
import json
from langsmith import traceable


### YT video link loader

##### Function to get video id from url

def get_video_id_from_url(url):
    """
    Extract the YouTube video ID from a given URL, handling as many known
    URL patterns as possible (watch, short links, embed, shorts, etc.).

    Args:
        url (str): The YouTube URL.

    Returns:
        str: The video ID
    """

    # Quick sanity check
    if not url:
        return None

    # Parse the URL
    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)

    # Clean up domain for easier handling; e.g. strip out "www." or "m."
    # But be aware that there are other TLDs like youtube.co.uk, etc.
    domain = parsed.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    if domain.startswith("m."):
        domain = domain[2:]

    # 1) Check the 'v' parameter (typical watch URL).
    #    Example: https://www.youtube.com/watch?v=dQw4w9WgXcQ
    if 'v' in query_params:
        return query_params['v'][0]

    # 2) Check for youtu.be short links (and possible subdomains).
    #    Example: https://youtu.be/dQw4w9WgXcQ
    #    In theory might be youtu.be/?v=xxx but that’s rare.
    if "youtu.be" in domain:
        # The path usually directly contains the video ID, e.g. /dQw4w9WgXcQ
        # Strip leading '/' from path.
        video_id = parsed.path.lstrip('/')
        if video_id:
            return video_id

    # 3) Check for /shorts/ format.
    #    Example: https://www.youtube.com/shorts/dQw4w9WgXcQ
    if '/shorts/' in parsed.path:
        # Typically: /shorts/<VIDEO_ID>
        # So the last part of the path is the ID.
        return parsed.path.split('/')[-1]

    # 4) Check for /embed/<VIDEO_ID> format.
    #    Example: https://www.youtube.com/embed/dQw4w9WgXcQ
    #    Also sometimes /v/ or /e/ can appear in old YouTube players:
    #    - https://www.youtube.com/v/dQw4w9WgXcQ
    #    - https://www.youtube.com/e/dQw4w9WgXcQ
    embed_patterns = ("/embed/", "/v/", "/e/")
    for pattern in embed_patterns:
        if pattern in parsed.path:
            return parsed.path.split('/')[-1]

    # 5) Check if there's any chance the path is just /watch
    #    but we missed query param (edge case).
    #    Sometimes YouTube might embed IDs in other query params
    #    or a parameter named 'video_id', etc. We'll do a quick scan:
    #    e.g. ?video_id=xxx or ?feature=xxx
    for param_key, param_val in query_params.items():
        # In rare cases, a param might hold an actual ID, so let's do
        # a quick check if the value might be a typical YouTube ID:
        # For simplicity, a typical ID is 11 characters (but can be longer).
        # We'll do a quick match for a base64-URL-safe pattern:
        candidate = param_val[0]
        if re.match(r'^[0-9A-Za-z_-]{10,}$', candidate):
            return candidate

    # 6) Because YouTube might have other domain TLDs (youtube.co.uk, etc.),
    #    or time-coded short link that doesn't use param `v`.
    #    Let's do a final fallback with a Regex approach on the entire URL
    #    to see if something that "looks like" a typical YouTube ID is present.
    #    This is *less* reliable, but can catch some unusual cases.
    #
    #    A typical YouTube video ID is often 11 characters long,
    #    but it can also be longer (up to 14 or more in some cases),
    #    composed of [0-9A-Za-z_-].
    #    We'll search for something that *looks like* it might be a YouTube ID.
    #    We'll guess: up to 16 characters for good measure.
    match = re.search(
        r'([0-9A-Za-z_-]{10,16})',
        parsed.path + ' ' + parsed.query
    )
    if match:
        return match.group(1)

    # If we reach here, no known pattern was recognized.
    raise Exception(f"Unable to extract video ID from URL: {url}")


##### Function to get YT Transcripts

# Function to convert time in sec to MM:SS or HH:MM:SS
def convert_time(time_in_sec, format='MM:SS'):
    if format == 'MM:SS':
        return time.strftime('%M:%S', time.gmtime(time_in_sec))
    elif format == 'HH:MM:SS':
        return time.strftime('%H:%M:%S', time.gmtime(time_in_sec))
    else:
        raise Exception('Invalid format')


# Function to convert MM:SS or HH:MM:SS into time in sec
def convert_time_to_sec(time_str):
    time_str = time_str.split(':')
    if len(time_str) == 2:
        return int(time_str[0])*60 + int(time_str[1])
    elif len(time_str) == 3:
        return int(time_str[0])*3600 + int(time_str[1])*60 + int(time_str[2])
    else:
        raise Exception('Invalid time format')


@try_n_times(3)
def get_transcript(video_id: str, return_text_only = False):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript(['en'])

        # To get subtitle in any other language (Autotranslate)
        #translated_transcript = transcript.translate('es')
        #print(translated_transcript.fetch())

        transcript =  transcript.fetch()

        # Return transcript as text if True
        if return_text_only:
          return ' '.join([item['text'] for item in transcript])

        # Check max video time
        video_duration = transcript[-1]['start']

        # Check if video duration is longer than an hour - use HH:MM:SS format in that case, else use MM:SS format
        if video_duration >= 60*60:
          format = 'HH:MM:SS'
        else:
          format = 'MM:SS'

        return [
            {
                'timestamp': convert_time(item['start'], format = format),
                'text': item['text']
            }
            for item in transcript
        ]

    except Exception as e:
        raise Exception(f"Error fetching transcript: {e}")

@traceable
@try_n_times(3)
def get_transcript_backup(video_id: str, return_text_only=False):
    """
    Fallback method using the RapidAPI endpoint.
    """
    # --- Replace these with your own values ---
    API_KEY = os.environ.get("RAPID_API_KEY")  # <--- Replace
    API_HOST = "youtube-transcript3.p.rapidapi.com"
    # ------------------------------------------

    url = "https://youtube-transcript3.p.rapidapi.com/api/transcript"
    querystring = {"videoId": video_id}
    headers = {
        "x-rapidapi-key": API_KEY,
        "x-rapidapi-host": API_HOST
    }

    try:
        response = requests.get(url, headers=headers, params=querystring)
        data = response.json()
        if not data.get('success'):
            raise Exception("Fallback API returned 'success' = False")

        transcript_data = data.get('transcript', [])
        if not transcript_data:
            raise Exception("No transcript found in fallback API response")

        # Since 'offset' and 'duration' might come as strings, we parse them:
        # Convert them to floats so we can do arithmetic for video duration, etc.
        for item in transcript_data:
            item['offset'] = float(item['offset'])
            item['duration'] = float(item['duration'])

        # Calculate total video duration from the last item’s offset + duration
        video_duration = transcript_data[-1]['offset'] + transcript_data[-1]['duration']

        if video_duration >= 3600:
            time_format = 'HH:MM:SS'
        else:
            time_format = 'MM:SS'

        if return_text_only:
            return ' '.join(item['text'] for item in transcript_data)

        # Map the fallback transcript structure to the same signature
        # as the primary get_transcript function
        results = []
        for item in transcript_data:
            results.append({
                'timestamp': convert_time(item['offset'], format=time_format),
                'text': item['text']
            })
        return results

    except Exception as e:
        raise Exception(f"Error fetching transcript via fallback method: {e}")


@st.cache_data
def load_transcripts_from_csv():
    """
    Reads the entire transcripts CSV into a dictionary {video_id -> transcript_data}.
    This is cached by Streamlit to avoid repeated I/O.
    """
    TRANSCRIPTS_CSV = "assets/video_transcripts.csv"

    if not os.path.isfile(TRANSCRIPTS_CSV):
        print("CSV File not found")
        return {}
 
    transcripts_dict = {}

    with open(TRANSCRIPTS_CSV, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        transcript_col_count = len([col for col in reader.fieldnames if col.startswith('transcript_')])
        for row in reader:
            transcripts_dict[row["video_id"]] = json.loads(
                "".join([row[f"transcript_{i}"] for i in range(transcript_col_count)])
            )
    
    return transcripts_dict

@traceable
def get_transcript_with_fallback(video_id: str, return_text_only=False):
    """
    1. Uses st.cache_data to load all transcripts from CSV into memory.
    2. Checks if 'video_id' is in the cached dict. If so, returns it immediately.
    3. Otherwise, fetches via the normal transcript or backup function,
       saves to CSV, and updates the cache by calling st.cache_data.clear().
    """

    transcripts_cache = load_transcripts_from_csv()
    
    # 1) Check in-memory cache
    if video_id in transcripts_cache:
        print(f"Video id {video_id} found in cached CSV.")
        cached_transcript = transcripts_cache[video_id]
        if len(str(cached_transcript)) > 100:
            if return_text_only:
                return " ".join(segment["text"] for segment in cached_transcript)
            print("Loaded Cached Transcript from the CSV")
            return cached_transcript
        else:
            pass

    try:
        # First attempt: official YT Transcript API
        return get_transcript(video_id, return_text_only=return_text_only)
    except Exception as e:
        # If we got here, it failed even after retries in @try_n_times
        print(f"Primary transcript fetch failed: {e}. Attempting fallback...")
        # Fallback:
        return get_transcript_backup(video_id, return_text_only=return_text_only)


##### Agents to get YT Chapters

# Define the yt chapter generator agent

generate_yt_video_chapters_prompt = """You are an AI assistant specialized in creating chapters or sections for educational YouTube videos. Your goal is to divide the video content into logical segments that will help viewers navigate the video more easily and enhance their learning experience. You will be provided with the video title and a timestamped transcript of the video.

Here's the title of the YouTube video:
<video_title>
{video_title}
</video_title>

Now, here's the timestamped transcript of the video:
<timestamped_transcript>
{timestamped_transcript}
</timestamped_transcript>

To create chapters for this educational video, follow these steps:

1. Carefully read through the entire transcript to understand the overall content and structure of the video. Treat the transcript as a literary work, considering the educational narrative and learning objectives.

2. Create an outline of the video's content, identifying major topics, themes, or shifts that would make logical breaking points for chapters. This outline will help ensure effective chapter breaks, especially for videos with complex educational content.

3. Consider the video's length when creating chapters. For shorter videos (under 10 minutes), aim for 3-5 chapters. For longer videos, you may create up to 10-15 chapters, depending on the complexity of the content.

4. Create chapters that are coherent and self-contained, each focusing on a specific subtopic or learning objective. Each chapter should have a clear goal, which you can summarize in a single bullet point.

5. Emphasize creating a structured learning journey. Each chapter should raise a question or point of interest at the beginning and address it by the end, keeping the viewer engaged and looking forward to the next chapter.

6. Use clear, concise, and descriptive titles for each chapter that accurately represent the educational content of that section.

7. Ensure chapters are evenly distributed throughout the video, avoiding concentration at the beginning or end. However, prioritize content relevance over equal-length segments.

8. Include the start time for each chapter in the format [HH:MM:SS] or [MM:SS] for shorter videos. Ensure all timestamps are accurate and within the video's total duration.

9. Avoid creating chapters that are too close together. If necessary, combine closely spaced topics into a single chapter to maintain a good learning flow.

Present your final output in the following format:

<output>

<summary>
Give a short summary of the video's content.
</summary>

<outline>
- Main topic 1
  - Subtopic 1.1
  - Subtopic 1.2
- Main topic 2
  - Subtopic 2.1
  - Subtopic 2.2
...
</outline>

<chapters>
[00:00] Chapter Title
[MM:SS] Chapter Title
[MM:SS] Chapter Title
...
</chapters>

</output>

Remember these important points:
- The first chapter should always start at [00:00].
- Ensure no timestamp exceeds the video's total duration.
- Aim for a balanced distribution of chapters while prioritizing content relevance and learning objectives.

If you're unsure about any part of the video content or have difficulty creating logical chapters, do your best to create a coherent structure based on the information available in the transcript, always keeping the educational focus in mind.
"""


@try_n_times(3)
def generate_yt_video_chapters(video_title, timestamped_transcript, messages_list = [], llm = 'groq'):
  """
  Generate chapters for a YouTube video based on its title and transcript.
  Args:
    video_title (str): The title of the YouTube video.
    timestamped_transcript (str): The transcript of the YouTube video.
    messages_list (list): A list of messages to include in the prompt.
    llm (str): The language model to use for generating the chapters.
  Returns:
    A string containing the generated chapters.
  """

  generate_yt_video_chapters_prompt_template = ChatPromptTemplate.from_messages(
      [
          ("human", generate_yt_video_chapters_prompt),
      ]
      + messages_list
  )


  # Function to extract a list of texts witin the specified xml tags
  def extract_text_in_tags(text: str):
    print(text)

    texts = [text] # Store the text within each tags as a list
    for tag in ['summary', 'outline', 'chapters']:
      # regex pattern
      pattern = f"<{tag}>\s*(.*?)\s*</{tag}>"
      match_1 = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
      if match_1:
        texts.append(match_1.group(1))
      else:
        raise Exception
        print(f"Unable to extract the text from the {tag} tags")
    return texts

  generate_yt_video_chapters_chain = generate_yt_video_chapters_prompt_template | llm_with_retry | output_parser | extract_text_in_tags

  response = generate_yt_video_chapters_chain.with_config(configurable={"llm": llm}).invoke(
      {
          "video_title": video_title,
          "timestamped_transcript": timestamped_transcript
      }
  )
  return {
      "full_response": response[0],
      "summary": response[1],
      "outline": response[2],
      "chapters": response[3]
  }


#Helper functions to get chapter chunks from transcript for each chapter

def get_timestamps_and_chapters(chapters):
    """
    Get time stamps and chapters from the generated chapters.
    Args:
        chapters (str): The generated chapters.
    Returns:
        list: A list of dictionaries containing time stamps and chapters.
    """
    # Split the chapters
    chapters = chapters.split('\n')

    # Define the regex pattern
    pattern = r"\[(\d{2}:\d{2}(?::\d{2})?)\]\s*(.*)"

    # Initialize an empty list to store the time stamps and chapters
    timestamps_and_chapters = []

    # Iterate over the chapters
    for chapter in chapters:
        # Search the pattern in the example string
        match = re.search(pattern, chapter)

        if match:
            timestamp = match.group(1)
            chapter_text = match.group(2)
        else:
            print("No match found")
            raise Exception("No match found for video chapters' timestamps")

        # Append the time stamp and chapter to the list
        timestamps_and_chapters.append(
            {
                "timestamp": timestamp,
                "chapter": chapter_text
            }
        )

    return timestamps_and_chapters


# Function to chunk the transcript and get text for each chapter chunk
def get_text_for_each_chapter(timestamped_transcript, chapters):
    """
    Get text for each chapter chunk.
    Args:
        timestamped_transcript (list): A list of dictionaries containing timestamped transcript.
        chapters (str): The generated chapters.
    Returns:
        list: A list of dictionaries containing text for each chapter chunk.
    """
    # Initialize an empty list to store the chapter chunks
    chapter_chunks = []

    # Get chapters and timestamps as a list
    timestamps_and_chapters = get_timestamps_and_chapters(chapters)

    # Iterate over the chapters
    for i, chapter_dict in enumerate(timestamps_and_chapters):
        # Get the timestamp and chapter text
        timestamp = chapter_dict["timestamp"]
        chapter_title = chapter_dict["chapter"]

        # Convert timestamp to sec
        start_time = convert_time_to_sec(timestamp)

        # Get end time
        if i < len(timestamps_and_chapters) - 1:
            end_time = convert_time_to_sec(timestamps_and_chapters[i + 1]["timestamp"])
        else:
            end_time = convert_time_to_sec("23:59:59")

        # Initialize an empty string to store the chapter chunk
        chapter_chunk = ""

        # Iterate over the timestamped transcript
        for j, transcript_dict in enumerate(timestamped_transcript):
            # Get the timestamp and transcript text
            transcript_timestamp = transcript_dict["timestamp"]
            transcript_text = transcript_dict["text"]

            # Convert timestamp to sec
            transcript_timestamp_sec = convert_time_to_sec(transcript_timestamp)

            # Check if the transcript timestamp is within the chapter chunk
            if start_time - 1 <= transcript_timestamp_sec <= end_time + 1:   # -1 and +1 to add some buffer for rounding errors
                # Append the transcript text to the chapter chunk
                chapter_chunk += str(transcript_text) + " "

        # Append the chapter chunk to the list
        chapter_chunks.append(
            {
                "timestamp": timestamp,
                "start_time": start_time,
                "end_time": end_time,
                "chapter_title": chapter_title,
                "text": chapter_chunk
            }
        )

    return chapter_chunks


def get_chapters_as_str(chapters):
    return '\n---\n'.join([
        '\n'.join(
            [f'{key}: {value}' for key, value in chapter.items()]
        )
        for chapter in chapters
    ])


#Define the yt chapter validator agent

validate_generated_chapters_prompt = """You are an AI validator agent tasked with reviewing and validating the chapters generated for an educational YouTube video. Your primary role is to ensure the accuracy of timestamps, content coverage, and logical structure of the chapters. You will be provided with the video title, and generated chapters.

Here's the information you need to review:

<video_title>
{video_title}
</video_title>

<generated_chapters>
{generated_chapters}
</generated_chapters>

Your task will be performed in three steps:

1. Summarize chapter content
2. Evaluate chapters based on key criteria
3. Provide validation result and feedback

Step 1: Summarize Chapter Content

First, create a summary of each chapter's content. Present your summaries in the following format:

<chapter_summaries>
[MM:SS] Chapter Title
Summary: [Brief summary of the chapter's content based on the transcript]

[MM:SS] Next Chapter Title
Summary: [Brief summary of the chapter's content based on the transcript]

...
</chapter_summaries>

Step 2: Evaluate Chapters

After summarizing the chapters, evaluate them based on the following criteria, in order of priority:

1. Content Coverage
2. Timestamp Accuracy
3. Logical Chapter Structure

Write your evaluation in the following format:

<evaluation>
1. Content Coverage:
   [Your detailed evaluation]

2. Timestamp Accuracy:
   [Your detailed evaluation]

3. Logical Chapter Structure:
   [Your detailed evaluation]
</evaluation>

Here are detailed descriptions of what to check for each criterion:

1. Content Coverage:
   - For each chapter, verify that the content between its timestamp and the next chapter's timestamp (or the end of the video) actually covers what the chapter title promises.
   - Check if any significant content or topics are missing from the chapter structure.

2. Timestamp Accuracy:
   - Check that the first chapter starts at [00:00].
   - Ensure that the timestamps are placed at logical break points in the content (e.g., at the beginning of a new topic or after concluding a previous one).

3. Logical Chapter Structure:
   - Ensure the chapter divisions make logical sense based on the content of the transcript.
   - Verify that each chapter covers a coherent and self-contained segment of the video.
   - Assess if the chapters create a structured learning journey that follows the video's content flow.

Step 3: Validation Result and Feedback

After completing your evaluation, provide your validation result in the following format:

<validation_result>
[APPROVED] or [NEEDS IMPROVEMENT]
</validation_result>

If the result is [APPROVED], briefly explain why the chapters meet all the criteria. Use the following format for feedback:

<feedback>
[Explain why chapters meet all criteria]
</feedback>

If the result is [NEEDS IMPROVEMENT], provide specific feedback for each issue found, referencing the criteria above. Use the following format for feedback:

<feedback>
1. [Criterion]: [Explanation of the issue]
   Suggestion: [Proposed improvement]

2. [Criterion]: [Explanation of the issue]
   Suggestion: [Proposed improvement]

...
</feedback>

Wrap your entire output within output tags.

Exception:
In some cases, the final timestamp may be set to a large number such as 23:59:59. This is acceptable as long as the chapter content is coherent and self-contained. If you encounter this, it is still okay to approve if the rest of the chapters look good.

Remember, your primary goal is to ensure that the chapters accurately represent the video's content, have correct timestamps, and provide a logical structure for the educational video. Be thorough in your review and provide constructive feedback when necessary, focusing on these key aspects.
"""

@try_n_times(3)
def validate_generated_chapters(video_title, generated_chapters, messages_list = [], llm = 'groq'):
    """
    Validate the generated chapters for an educational YouTube video.
    Args:
        video_title (str): The title of the YouTube video.
        generated_chapters (str): The generated chapters for the YouTube video.
        messages_list (list): A list of messages to include in the prompt.
        llm (str): The language model to use for generating the chapters.
    Returns:
        str: Approval or improvement feedback for the generated chapters.
    """

    validate_generated_chapters_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("human", validate_generated_chapters_prompt),
        ]
        + messages_list
    )


    # Function to extract a list of texts witin the specified xml tags
    def extract_text_in_tags(text: str):
        print(text)

        texts = [text] # Store the text within each tags as a list
        for tag in ['chapter_summaries', 'evaluation', 'validation_result', 'feedback']:
            # regex pattern
            pattern = f"<{tag}>\s*(.*?)\s*</{tag}>"
            match_1 = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match_1:
                texts.append(match_1.group(1))
            else:
                raise Exception
                print(f"Unable to extract the text from the {tag} tags")
        return texts

    validate_generated_chapters_chain = validate_generated_chapters_prompt_template | llm_with_retry | output_parser | extract_text_in_tags

    response = validate_generated_chapters_chain.with_config(configurable={"llm": llm}).invoke(
        {
            "video_title": video_title,
            "generated_chapters": generated_chapters
        }
    )
    return {
        "full_response": response[0],
        "chapter_summaries": response[1],
        "evaluation": response[2],
        "validation_result": response[3],
        "feedback": response[4]
    }



# Function to get additional metadata
@try_n_times(3)
def get_additional_metadata(video_id):
    loader = YoutubeLoader.from_youtube_url(
        f"https://www.youtube.com/watch?v={video_id}", add_video_info=True
    )
    return loader.load()[0].metadata


##### Function to get yt chapters as doc chunks
@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "chunk_videos",
    "function_name": "run_chunk_videos",
    "user_id": st.session_state.get("role", "anonymous")
})
@try_n_times(n = 3, wait = 1, backoff = "linear")
def get_yt_chapters_chunks_as_docs(video_id: str, video_title = None, timestamped_transcript = None, llm = 'gemini_2_flash'):
    """
    Get chapters for a YouTube video based on its title and transcript.
    Args:
        video_id (str): The video id of the YouTube video.
        video_title (str): The title of the YouTube video.
        timestamped_transcript (dict): The timestamped transcript of the video
        llm (str): The language model to use for generating the chapters.
    Returns:
        list: A list of chunked documents.
    """
    docs = []

    # These metadata contain other info such as video title, views, desc, author, published date, etc.
    try:
        additional_metadata = get_additional_metadata(video_id)
    except Exception as e:
        print(f"Error getting additional metadata for video id: {video_id}")
        # Populate with dummy data
        additional_metadata = {
            'title': 'Error getting additional metadata',
            'length': 86399
        }

    if video_title is None:
        video_title = additional_metadata['title']
    else:
        additional_metadata['title'] = video_title

    # Initialize variables
    max_turns = 5

    if timestamped_transcript == None:
        timestamped_transcript = get_transcript_with_fallback(video_id = video_id, return_text_only = False) #get_transcript(video_id)
    else:
        print("Using transcripts passed in to the function")
    # Check if transcript retrieved without error
    if type(timestamped_transcript) == str and timestamped_transcript.startswith('Error'):
            raise Exception(f'Error retrieving transcript for video id: {video_id}')

    timestamped_transcript_str = ',\n'.join([str(item) for item in timestamped_transcript])

    # Initialize lists to keep track of memory for each agent
    generator_agent_msg_list = []
    validator_agent_msg_list = []

    # Run the two agents in a for loop
    for i in range(max_turns):
        # Call the generator agent
        generator_response = generate_yt_video_chapters(
            video_title = video_title,
            timestamped_transcript = timestamped_transcript_str,
            messages_list = generator_agent_msg_list,
            llm = llm
        )

        # Store the very first chapters output in a variable for consistent initial prompt over the loops
        if i == 0:
            first_output_chapters_list = get_text_for_each_chapter(timestamped_transcript, generator_response['chapters'])
            first_output_chapters_str = get_chapters_as_str(first_output_chapters_list)
        else:
            chapters_list = get_text_for_each_chapter(timestamped_transcript, generator_response['chapters'])
            chapters_str = get_chapters_as_str(chapters_list)

        # Update message list
        generator_agent_msg_list.append(
            ("ai", generator_response['full_response'])
        )
        if i != 0: # Skip the first iteration
            validator_agent_msg_list.append(
                ("human", f"Here's the updated chapters from the generator agent.\n\nUpdated chapters:\n{chapters_str}")
            )


        # Call the validator agent
        validator_response = validate_generated_chapters(
            video_title = video_title,
            generated_chapters = first_output_chapters_str,
            messages_list = validator_agent_msg_list,
            llm = llm
        )

        # Check if approved
        if 'APPROVED' in validator_response['validation_result']:
            break

        # Update message list
        validator_agent_msg_list.append(
            ("ai", validator_response['full_response'])
        )
        generator_agent_msg_list.append(
            ("human", f"Here's some feedback from validation agent. Incorporate the feedback given. Make sure to reply in same format without skipping any tags.\n\nFeedback:\n{validator_response['feedback']}")
        )

    result = {
        'video_summary': generator_response['summary'],
        'video_outline': generator_response['outline'],
        'video_chapters': generator_response['chapters'],
        'chapter_summaries': validator_response['chapter_summaries'],
        'chapters_with_chunks': get_text_for_each_chapter(timestamped_transcript, generator_response['chapters'])
    }


    chapters = result['chapters_with_chunks']

    # These are some metadata generated above and saved in sheet
    video_metadata = {
        'video_summary': result['video_summary'],
        'video_chapters': result['video_chapters'],
        'chapter_summaries': result['chapter_summaries']
    }

    additional_metadata.update(video_metadata)

    # Update the end time of last chapter (since intially it was assigned 23:59:59)
    chapters[-1]['end_time'] = additional_metadata['length']

    # Get all metadata in one dict apart from chapter's text
    chapter_chunks = []

    for chapter in chapters:
        chapter_text = chapter.pop('text')
        chapter.update(additional_metadata)

        # Add timestamped source link
        base_url = f"https://www.youtube.com/watch?v={video_id}"
        start = int(chapter.get("start_time") or 0)
        end = int(chapter.get("end_time") or 0)
        if "?" in base_url:
            full_url = f"{base_url}&start={start}&end={end}"
        else:
            full_url = f"{base_url}?start={start}&end={end}"
        chapter["source"] = full_url

        chapter_chunks.append(
            Document(
                page_content = chapter_text,
                metadata = chapter
            )
        )

    docs.extend(chapter_chunks)
    return docs
