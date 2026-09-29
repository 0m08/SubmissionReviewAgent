"""
Video Reference Validation Module

This module validates video links in the Final Outline sheet using Gemini's video understanding
capabilities. It analyzes whether videos are appropriate for teaching specific learning objectives
and returns validated video clips with precise timestamps.
"""

import re
import time
import random
from typing import List, Dict, Tuple, Optional
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from langsmith import traceable

from services.sheets_service import get_sheet_data_and_df, save_to_sheet, clear_worksheet
from services.smart_progress_bar import SmartProgressBar
from services.helper_functions import get_topic_outline, get_outline_with_los
from services.youtube_video_loader import load_transcripts_from_csv, get_transcript_with_fallback
from modules.chain import Chain


def extract_video_info_from_url(url: str) -> Optional[Dict]:
    """
    Extract video ID, start time, and end time from a YouTube URL.

    Supports formats:
    - https://www.youtube.com/embed/VIDEO_ID?start=559&end=634
    - https://www.youtube.com/watch?v=VIDEO_ID&start=559&end=634
    - https://youtu.be/VIDEO_ID?start=559&end=634

    :param url: YouTube URL string
    :return: Dictionary with video_id, start_time, end_time (in seconds) or None if invalid
    """
    if not url or not isinstance(url, str):
        return None

    url = url.strip()
    if not url:
        return None

    video_id = None
    start_time = None
    end_time = None

    try:
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)

        # Extract video ID based on URL format
        if 'youtube.com/embed/' in url:
            # Format: /embed/VIDEO_ID
            path_parts = parsed.path.split('/')
            if 'embed' in path_parts:
                embed_index = path_parts.index('embed')
                if embed_index + 1 < len(path_parts):
                    video_id = path_parts[embed_index + 1]
        elif 'youtube.com/watch' in url:
            # Format: ?v=VIDEO_ID
            video_id = query_params.get('v', [None])[0]
        elif 'youtu.be/' in url:
            # Format: youtu.be/VIDEO_ID
            video_id = parsed.path.lstrip('/')

        # Extract start and end times
        start_param = query_params.get('start', [None])[0]
        end_param = query_params.get('end', [None])[0]

        if start_param:
            start_time = int(start_param)
        if end_param:
            end_time = int(end_param)

        if video_id:
            return {
                'video_id': video_id,
                'start_time': start_time,
                'end_time': end_time,
                'original_url': url
            }
    except Exception as e:
        print(f"Error parsing URL {url}: {e}")

    return None


def format_timestamp_for_gemini(seconds: int) -> str:
    """
    Convert seconds to the format expected by Gemini API (e.g., '559s').

    :param seconds: Time in seconds
    :return: Formatted timestamp string
    """
    return f"{seconds}s"


def mmss_to_seconds(mmss: str) -> int:
    """
    Convert MM:SS format timestamp to seconds.

    :param mmss: Timestamp in MM:SS format (e.g., '09:19' or '02:00')
    :return: Time in seconds
    """
    if not mmss or not isinstance(mmss, str):
        return 0

    mmss = mmss.strip()
    parts = mmss.split(':')

    try:
        if len(parts) == 2:
            # MM:SS format
            minutes = int(parts[0])
            seconds = int(parts[1])
            return minutes * 60 + seconds
        elif len(parts) == 3:
            # HH:MM:SS format (in case of longer videos)
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        else:
            # Try parsing as plain seconds
            return int(mmss)
    except ValueError:
        return 0


def build_single_video_analysis_prompt(
    topic: str,
    subtopic: str,
    learning_objective: str,
    course_outline: str,
    video_index: int,
    original_start_time: Optional[int] = None,
    original_end_time: Optional[int] = None
) -> str:
    """
    Build a detailed prompt for Gemini to analyze a single video against a learning objective.

    :param topic: The topic name
    :param subtopic: The subtopic name
    :param learning_objective: The specific learning objective to evaluate against
    :param course_outline: The full course outline for context on what's in/out of scope
    :param video_index: The index of this video in the list (for reference)
    :param original_start_time: Original start time in seconds (if provided in URL)
    :param original_end_time: Original end time in seconds (if provided in URL)
    :return: The prompt string
    """
    # Build context about the video clip boundaries
    clip_context = ""
    if original_start_time is not None or original_end_time is not None:
        start_mmss = f"{original_start_time // 60:02d}:{original_start_time % 60:02d}" if original_start_time else "00:00"
        if original_end_time:
            end_mmss = f"{original_end_time // 60:02d}:{original_end_time % 60:02d}"
            clip_context = f"\n**Note:** You are analyzing a specific clip from {start_mmss} to {end_mmss}. Your recommended timestamps should be relative to the START of the full video, not this clip."
        else:
            clip_context = f"\n**Note:** You are analyzing from {start_mmss} onwards."

    prompt = f"""You are an expert instructional designer tasked with evaluating video content for educational purposes.

## Context

**Course Outline (for understanding scope):**
{course_outline}

**Current Topic:** {topic}
**Current Subtopic:** {subtopic if subtopic else 'N/A'}
**Learning Objective:** {learning_objective}
{clip_context}

## Your Task

You have been provided with ONE video clip to analyze. Your job is to:

1. **Analyze the video clip** to understand what concepts, skills, or knowledge it teaches
2. **Evaluate relevance** to the specific learning objective above
3. **Check scope alignment** - the video content should ONLY cover what's within the scope of this learning objective. Content that covers topics from other parts of the course outline, or content that goes beyond what's needed for this specific LO, should be flagged.
4. **Determine usability** - Can this video (or a portion of it) be used DIRECTLY to teach this learning objective?
5. **Assess visual quality** - Describe the visual elements and rate how effective the visuals are for teaching

## Evaluation Criteria

The video clip is SUITABLE if:
- It directly addresses the concepts in the learning objective
- The explanation is clear and appropriate for learners
- It stays within the scope of the learning objective (doesn't include unrelated topics)
- The video quality and presentation are adequate for educational use
- The visuals help illustrate the concepts being taught

The video clip is NOT SUITABLE if:
- It covers topics outside the scope of this specific learning objective
- The content is too advanced or too basic for the learning objective
- It includes significant irrelevant information that would confuse learners
- The video quality or presentation is poor
- The visuals are distracting, irrelevant, or low quality

## Required Output Format

Analyze the video and provide your response in the following XML format.
IMPORTANT: All timestamps MUST be in MM:SS format (e.g., 01:15 for 1 minute and 15 seconds, 09:19 for 9 minutes and 19 seconds).

<analysis>
<relevance_score>[0-100, where 100 is perfectly aligned with the learning objective]</relevance_score>
<scope_alignment>[IN_SCOPE / PARTIALLY_IN_SCOPE / OUT_OF_SCOPE]</scope_alignment>
<content_summary>[2-3 sentence summary of what this video clip covers]</content_summary>
<visual_description>[Describe the visual elements used in the video: animations, diagrams, real-world footage, talking head, screen recordings, text overlays, etc. Be specific about what kinds of visuals are shown.]</visual_description>
<visual_rating>[1-5 rating where: 1=Poor visuals that detract from learning, 2=Basic/minimal visuals, 3=Adequate visuals that support the content, 4=Good visuals that enhance understanding, 5=Excellent visuals that are highly effective for teaching]</visual_rating>
<recommendation>[USE / PARTIAL_USE / DO_NOT_USE]</recommendation>
<reasoning>[Explain why this video should or should not be used, including visual quality considerations]</reasoning>
<use_video>[YES / NO - Should this video be included as a reference?]</use_video>
<start_time>[If USE or PARTIAL_USE: the start timestamp in MM:SS format for the portion to use. If DO_NOT_USE: N/A]</start_time>
<end_time>[If USE or PARTIAL_USE: the end timestamp in MM:SS format for the portion to use. If DO_NOT_USE: N/A]</end_time>
</analysis>

## Important Notes

- Be STRICT about scope alignment. It's better to recommend NOT using this video than to include one with out-of-scope content.
- If only a portion of the video is suitable, provide EXACT start and end timestamps for the usable portion.
- The final output will be used to populate a "References" column, so precision is critical.
- Timestamps should be relative to the full video duration, not relative to any clip boundaries.
- Pay close attention to the visual quality and how well the visuals support the learning objective.

Now analyze the provided video:"""

    return prompt


@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Validate Video References",
    "function_name": "analyze_single_video_with_gemini",
    "user_id": st.session_state.get("role", "anonymous"),
    "user_email": st.session_state.get("user_email", "anonymous")
})
def analyze_single_video_with_gemini(
    video_info: Dict,
    topic: str,
    subtopic: str,
    learning_objective: str,
    course_outline: str,
    video_index: int,
    model: str = 'models/gemini-2.5-flash'
) -> Optional[str]:
    """
    Use Gemini's video understanding to analyze a single video against a learning objective.

    :param video_info: Video info dictionary from extract_video_info_from_url
    :param topic: The topic name
    :param subtopic: The subtopic name
    :param learning_objective: The learning objective to evaluate against
    :param course_outline: Full course outline for scope context
    :param video_index: Index of this video in the original list
    :param model: Gemini model to use
    :return: Gemini's analysis response or None if error
    """
    if not video_info:
        return None

    try:
        client = genai.Client()

        video_id = video_info['video_id']
        start_time = video_info.get('start_time')
        end_time = video_info.get('end_time')

        # Construct the YouTube watch URL for Gemini
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        # Build video metadata if timestamps are provided
        video_metadata = None
        if start_time is not None or end_time is not None:
            metadata_kwargs = {}
            if start_time is not None:
                metadata_kwargs['start_offset'] = format_timestamp_for_gemini(start_time)
            if end_time is not None:
                metadata_kwargs['end_offset'] = format_timestamp_for_gemini(end_time)
            video_metadata = types.VideoMetadata(**metadata_kwargs)

        # Create the video part
        if video_metadata:
            video_part = types.Part(
                file_data=types.FileData(file_uri=video_url),
                video_metadata=video_metadata
            )
        else:
            video_part = types.Part(
                file_data=types.FileData(file_uri=video_url)
            )

        # Build the analysis prompt for single video
        prompt = build_single_video_analysis_prompt(
            topic=topic,
            subtopic=subtopic,
            learning_objective=learning_objective,
            course_outline=course_outline,
            video_index=video_index,
            original_start_time=start_time,
            original_end_time=end_time
        )

        print("Prompt for video", video_index, ":", prompt)

        # Create the content with video first, then prompt
        content = types.Content(
            parts=[video_part, types.Part(text=prompt)]
        )

        response = client.models.generate_content(
            model=model,
            contents=content
        )

        print(f"Gemini response for video {video_index}:", response.text)

        return response.text

    except Exception as e:
        print(f"Error analyzing video {video_index} with Gemini: {e}")
        return None


def parse_single_video_response(response: str) -> Dict:
    """
    Parse Gemini's XML response for a single video analysis.

    :param response: Gemini's response text
    :return: Dictionary with parsed evaluation
    """
    result = {
        'use_video': False,
        'start_time': None,
        'end_time': None,
        'relevance_score': 0,
        'scope_alignment': '',
        'content_summary': '',
        'visual_description': '',
        'visual_rating': 0,
        'recommendation': '',
        'reasoning': ''
    }

    if not response:
        return result

    try:
        # Extract use_video decision
        use_match = re.search(r'<use_video>([^<]+)</use_video>', response, re.IGNORECASE)
        if use_match:
            use_value = use_match.group(1).strip().upper()
            result['use_video'] = use_value == 'YES'

        # Extract start_time
        start_match = re.search(r'<start_time>([^<]+)</start_time>', response)
        if start_match:
            start_value = start_match.group(1).strip()
            if start_value.lower() not in ['n/a', 'na', 'none', '']:
                result['start_time'] = mmss_to_seconds(start_value)

        # Extract end_time
        end_match = re.search(r'<end_time>([^<]+)</end_time>', response)
        if end_match:
            end_value = end_match.group(1).strip()
            if end_value.lower() not in ['n/a', 'na', 'none', '']:
                result['end_time'] = mmss_to_seconds(end_value)

        # Extract other fields for logging/debugging
        relevance_match = re.search(r'<relevance_score>(\d+)</relevance_score>', response)
        if relevance_match:
            result['relevance_score'] = int(relevance_match.group(1))

        scope_match = re.search(r'<scope_alignment>([^<]+)</scope_alignment>', response)
        if scope_match:
            result['scope_alignment'] = scope_match.group(1).strip()

        summary_match = re.search(r'<content_summary>([^<]+)</content_summary>', response)
        if summary_match:
            result['content_summary'] = summary_match.group(1).strip()

        # Extract visual analysis fields
        visual_desc_match = re.search(r'<visual_description>([^<]+)</visual_description>', response)
        if visual_desc_match:
            result['visual_description'] = visual_desc_match.group(1).strip()

        visual_rating_match = re.search(r'<visual_rating>(\d+)</visual_rating>', response)
        if visual_rating_match:
            result['visual_rating'] = int(visual_rating_match.group(1))

        recommendation_match = re.search(r'<recommendation>([^<]+)</recommendation>', response)
        if recommendation_match:
            result['recommendation'] = recommendation_match.group(1).strip()

        reasoning_match = re.search(r'<reasoning>([^<]+)</reasoning>', response)
        if reasoning_match:
            result['reasoning'] = reasoning_match.group(1).strip()

    except Exception as e:
        print(f"Error parsing single video Gemini response: {e}")

    return result


def build_single_reference_url(video_id: str, start_time: Optional[int], end_time: Optional[int]) -> str:
    """
    Build a YouTube embed URL with timestamps for a single video.

    :param video_id: YouTube video ID
    :param start_time: Start time in seconds (optional)
    :param end_time: End time in seconds (optional)
    :return: YouTube embed URL with timestamps
    """
    if start_time is not None and end_time is not None:
        return f"https://www.youtube.com/embed/{video_id}?start={start_time}&end={end_time}"
    elif start_time is not None:
        return f"https://www.youtube.com/embed/{video_id}?start={start_time}"
    elif end_time is not None:
        return f"https://www.youtube.com/embed/{video_id}?start=0&end={end_time}"
    else:
        return f"https://www.youtube.com/embed/{video_id}"


def get_transcript_for_video(video_id: str, start_time: Optional[int] = None, end_time: Optional[int] = None) -> Optional[str]:
    """
    Load transcript for a video from cache or API, optionally filtering to a time range.
    Returns transcript with timestamps in the format: [MM:SS] text

    Includes overlapping segments:
    - For start_time: includes the last segment that starts BEFORE start_time (likely still playing)
    - For end_time: includes segments that start AT or BEFORE end_time

    :param video_id: YouTube video ID
    :param start_time: Optional start time in seconds to filter transcript
    :param end_time: Optional end time in seconds to filter transcript
    :return: Transcript text with timestamps or None if not found
    """
    try:
        # Try loading from CSV cache first
        transcripts_cache = load_transcripts_from_csv()

        if video_id in transcripts_cache:
            transcript_data = transcripts_cache[video_id]
        else:
            # Fall back to API
            transcript_data = get_transcript_with_fallback(video_id, return_text_only=False)

        if not transcript_data:
            return None

        # If transcript is already a plain string (no timestamp data), return as-is
        if isinstance(transcript_data, str):
            return transcript_data

        # First pass: convert all timestamps to seconds and store with data
        items_with_seconds = []
        for item in transcript_data:
            timestamp_str = item.get('timestamp', '0:00')
            text = item.get('text', '')

            # Convert timestamp to seconds
            parts = timestamp_str.split(':')
            if len(parts) == 2:
                item_time = int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                item_time = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            else:
                item_time = 0

            items_with_seconds.append({
                'time': item_time,
                'text': text
            })

        # If no time filtering, return all
        if start_time is None and end_time is None:
            formatted_lines = []
            for item in items_with_seconds:
                formatted_lines.append(f"[{item['time']}s] {item['text']}")
            return '\n'.join(formatted_lines) if formatted_lines else None

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

        # Filter and format transcript
        formatted_lines = []
        for i in range(start_idx, len(items_with_seconds)):
            item = items_with_seconds[i]

            # For end_time: include segments that START at or before end_time (overlap)
            if end_time is not None and item['time'] > end_time:
                break

            # Add formatted line with timestamp
            formatted_lines.append(f"[{item['time']}s] {item['text']}")

        return '\n'.join(formatted_lines) if formatted_lines else None

    except Exception as e:
        print(f"Error loading transcript for video {video_id}: {e}")
        return None


@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Validate Video References",
    "function_name": "run_final_video_selection",
    "user_id": st.session_state.get("role", "anonymous"),
    "user_email": st.session_state.get("user_email", "anonymous")
})
def run_final_video_selection(
    candidate_videos: List[Dict],
    topic: str,
    subtopic: str,
    learning_objective: str,
    course_outline: str,
    llm: str = 'gemini_2_flash'
) -> List[Dict]:
    """
    Final analysis to select the best video(s) or combination of clips to cover the learning objective.
    Uses the Chain class for LLM calls with transcript-based analysis.

    :param candidate_videos: List of dicts with video_id, visual_description, visual_rating,
                            start_time, end_time, content_summary, relevance_score
    :param topic: The topic name
    :param subtopic: The subtopic name
    :param learning_objective: The learning objective to evaluate against
    :param course_outline: Full course outline for scope context
    :param llm: LLM model to use (default: gemini_2_flash)
    :return: List of selected video dicts with final timestamps
    """
    if not candidate_videos:
        return []

    # Build video information for the prompt
    videos_info = []
    for i, video in enumerate(candidate_videos):
        video_id = video.get('video_id', '')
        start_time = video.get('start_time')
        end_time = video.get('end_time')

        # Get transcript for this video segment
        transcript = get_transcript_for_video(video_id, start_time, end_time)

        if not transcript:
            print(f"  Skipping video {video_id} - no transcript available")
            continue

        # Truncate transcript if too long (keep first 10000 chars)
        if len(transcript) > 10000:
            transcript = transcript[:10000] + "... [transcript truncated]"

        video_info = f"""
VIDEO {i + 1}:
- Video ID: {video_id}
- Suggested Clip: {start_time or 0}s to {end_time or 'end'}s
- Content Summary: {video.get('content_summary', 'N/A')}
- Visual Description: {video.get('visual_description', 'N/A')}
- Visual Rating: {video.get('visual_rating', 'N/A')}/5
- Relevance Score: {video.get('relevance_score', 'N/A')}/100

TRANSCRIPT:
{transcript}
"""
        videos_info.append({
            'index': i,
            'video_id': video_id,
            'info_text': video_info,
            'original_data': video
        })

    if not videos_info:
        print("  No videos with transcripts available for final selection")
        return []

    # Build the prompt
    videos_text = "\n---\n".join([v['info_text'] for v in videos_info])

    prompt = f"""You are an expert instructional designer selecting the best video reference(s) for a learning objective.

## Context

**Course Outline (for understanding what's in/out of scope):**
{course_outline}

**Current Topic:** {topic}
**Current Subtopic:** {subtopic if subtopic else 'N/A'}
**Learning Objective:** {learning_objective}

## Candidate Videos

The following videos have been pre-screened as potentially relevant. Each includes:
- Visual description and quality rating (1-5)
- Content summary from visual analysis
- Transcript of the relevant segment

{videos_text}

## Your Task

Analyze the transcripts and metadata to select the BEST video or combination of video clips that:

1. **Completely covers** the learning objective - all key concepts should be explained
2. **Stays in scope** - doesn't include content from other topics in the course outline
3. **Has good visual quality** - prefer videos with higher visual ratings
4. **Is concise** - shorter clips are better if they cover the same content
5. **Flows well** - if combining multiple clips, they should form a coherent sequence

## Selection Rules

- You may select ZERO videos if none adequately cover the learning objective
- You may select ONE video if it fully covers the learning objective
- You may select MULTIPLE clips (from same or different videos) if they complement each other
- For each selected clip, you MUST provide exact start and end timestamps in seconds
- Prefer fewer, more comprehensive clips over many fragmented clips

## Required Output Format

<final_selection>
<num_selected>[Number of video clips selected: 0, 1, 2, etc.]</num_selected>
<selection_reasoning>[Explain your selection - why these videos/clips were chosen or why none were suitable]</selection_reasoning>

<selected_clips>
[If num_selected > 0, list each clip in order of viewing:]
<clip>
<video_index>[1-based index from the candidate list above, e.g., 1, 2, 3]</video_index>
<start_seconds>[Start time in seconds]</start_seconds>
<end_seconds>[End time in seconds]</end_seconds>
<clip_purpose>[What part of the LO does this clip cover?]</clip_purpose>
</clip>
[Repeat for additional clips if needed]
</selected_clips>
</final_selection>

Make your selection now:"""

    # Use Chain class for LLM call
    chain = Chain(llm=llm, tags=['final_selection', 'num_selected', 'selection_reasoning', 'selected_clips'])
    chain.add_message('user', prompt)
    print("Final selection prompt:", prompt)

    try:
        response = chain.run()

        # Parse the response
        selected_videos = []

        # Extract number selected
        num_match = re.search(r'<num_selected>(\d+)</num_selected>', response.get('text', ''))
        num_selected = int(num_match.group(1)) if num_match else 0

        if num_selected == 0:
            reasoning_match = re.search(r'<selection_reasoning>([^<]+)</selection_reasoning>', response.get('text', ''))
            if reasoning_match:
                print(f"  Final selection: No videos selected - {reasoning_match.group(1).strip()[:100]}...")
            return []

        # Extract clips
        clips_section = response.get('selected_clips', '')
        clip_pattern = r'<clip>(.*?)</clip>'
        clip_matches = re.findall(clip_pattern, clips_section, re.DOTALL)

        for clip_content in clip_matches:
            clip_data = {}

            # Extract video index (1-based from LLM, convert to 0-based)
            video_index_match = re.search(r'<video_index>(\d+)</video_index>', clip_content)
            if video_index_match:
                video_index = int(video_index_match.group(1)) - 1  # Convert to 0-based
                clip_data['video_index'] = video_index

                # Look up video_id from videos_info using the index
                if 0 <= video_index < len(videos_info):
                    clip_data['video_id'] = videos_info[video_index]['video_id']
                else:
                    print(f"    Warning: Invalid video index {video_index + 1} (out of range)")
                    continue

            start_match = re.search(r'<start_seconds>(\d+)</start_seconds>', clip_content)
            if start_match:
                clip_data['start_time'] = int(start_match.group(1))

            end_match = re.search(r'<end_seconds>(\d+)</end_seconds>', clip_content)
            if end_match:
                clip_data['end_time'] = int(end_match.group(1))

            if clip_data.get('video_id'):
                selected_videos.append(clip_data)

        print(f"  Final selection: {len(selected_videos)} clip(s) selected")
        return selected_videos

    except Exception as e:
        print(f"Error in final video selection: {e}")
        # Fall back to returning all candidate videos
        return [{'video_id': v.get('video_id'), 'start_time': v.get('start_time'), 'end_time': v.get('end_time')}
                for v in candidate_videos]


@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Validate Video References",
    "function_name": "process_single_row",
    "user_id": st.session_state.get("role", "anonymous"),
    "user_email": st.session_state.get("user_email", "anonymous")
})
def process_single_row(
    row_data: Dict,
    course_outline: str,
    model: str = 'models/gemini-2.5-flash',
    llm: str = 'gemini_2_flash'
) -> Tuple[int, str]:
    """
    Process a single row to validate video references using a two-stage analysis:
    Stage 1: Analyze each video individually with Gemini (visual analysis)
    Stage 2: Use LLM with transcripts to select the best video(s)

    :param row_data: Dictionary containing row index and data
    :param course_outline: Full course outline for context
    :param model: Gemini model to use for video analysis
    :param llm: LLM model to use for final selection
    :return: Tuple of (row_index, validated_references_string)
    """
    row_index = row_data['row_index']
    topic = row_data.get('topic', '')
    subtopic = row_data.get('subtopic', '')
    learning_objective = row_data.get('learning_objective', '')
    video_links = row_data.get('video_links', '')
    youtube_links = row_data.get('youtube_videos', '')
    as_is_sources = row_data.get('as_is_sources', '')

    # Extract video IDs from as_is_sources for filtering
    # Format: "[1] https://..." - need to strip the [N] prefix
    as_is_video_ids = set()
    if as_is_sources and isinstance(as_is_sources, str):
        for source_line in as_is_sources.split('\n'):
            source_line = source_line.strip()
            if not source_line:
                continue
            # Strip the [N] prefix if present (e.g., "[1] https://..." -> "https://...")
            source_url = re.sub(r'^\[\d+\]\s*', '', source_line)
            if source_url:
                source_info = extract_video_info_from_url(source_url)
                if source_info and source_info.get('video_id'):
                    as_is_video_ids.add(source_info['video_id'])

    # Process video_links with as_is_sources filter
    filtered_video_links = []
    if video_links and isinstance(video_links, str):
        for link in video_links.split('\n'):
            link = link.strip()
            if not link:
                continue
            # Extract video ID and check if it's in as_is_sources
            link_info = extract_video_info_from_url(link)
            if link_info and link_info.get('video_id'):
                if link_info['video_id'] in as_is_video_ids:
                    filtered_video_links.append(link)
                else:
                    print(f"    Skipping video_link (not in as_is_sources): {link_info['video_id']}")

    # Combine filtered video_links with youtube_videos (no filter for youtube_videos)
    all_links = []
    all_links.extend(filtered_video_links)
    if youtube_links and isinstance(youtube_links, str):
        all_links.extend([l.strip() for l in youtube_links.split('\n') if l.strip()])

    # Remove duplicates while preserving order
    seen = set()
    unique_links = []
    for link in all_links:
        if link not in seen:
            seen.add(link)
            unique_links.append(link)

    if not unique_links:
        return (row_index, '')

    # Extract video info from URLs
    video_infos = []
    for link in unique_links:
        info = extract_video_info_from_url(link)
        if info:
            video_infos.append(info)

    if not video_infos:
        return (row_index, '')

    # ========== STAGE 1: Visual Analysis with Gemini ==========
    print(f"  Stage 1: Analyzing {len(video_infos)} video(s) with Gemini for row {row_index + 1}...")
    candidate_videos = []

    for video_index, video_info in enumerate(video_infos):
        print(f"    Analyzing video {video_index + 1}/{len(video_infos)}...")

        # Analyze single video with Gemini
        response = analyze_single_video_with_gemini(
            video_info=video_info,
            topic=topic,
            subtopic=subtopic,
            learning_objective=learning_objective,
            course_outline=course_outline,
            video_index=video_index,
            model=model
        )

        if not response:
            print(f"      No response for video {video_index + 1}")
            continue

        # Parse the single video response
        parsed = parse_single_video_response(response)

        # Collect approved videos as candidates for Stage 2
        if parsed['use_video']:
            video_id = video_info['video_id']
            candidate = {
                'video_id': video_id,
                'start_time': parsed.get('start_time'),
                'end_time': parsed.get('end_time'),
                'content_summary': parsed.get('content_summary', ''),
                'visual_description': parsed.get('visual_description', ''),
                'visual_rating': parsed.get('visual_rating', 0),
                'relevance_score': parsed.get('relevance_score', 0),
                'scope_alignment': parsed.get('scope_alignment', '')
            }
            candidate_videos.append(candidate)
            print(f"      Video {video_index + 1} PASSED Stage 1 (relevance: {parsed['relevance_score']}, visual: {parsed['visual_rating']}/5)")
        else:
            print(f"      Video {video_index + 1} REJECTED (reason: {parsed['recommendation']})")

    # If no candidates passed Stage 1, return empty
    if not candidate_videos:
        print(f"  No videos passed Stage 1 for row {row_index + 1}")
        return (row_index, '')

    # If only one candidate, skip Stage 2 and use it directly
    if len(candidate_videos) == 1:
        print(f"  Only one candidate - skipping Stage 2 for row {row_index + 1}")
        video = candidate_videos[0]
        url = build_single_reference_url(video['video_id'], video['start_time'], video['end_time'])
        return (row_index, url)

    # ========== STAGE 2: Final Selection with Transcripts ==========
    print(f"  Stage 2: Final selection from {len(candidate_videos)} candidate(s) using transcripts...")

    final_selections = run_final_video_selection(
        candidate_videos=candidate_videos,
        topic=topic,
        subtopic=subtopic,
        learning_objective=learning_objective,
        course_outline=course_outline,
        llm=llm
    )

    # Build final reference URLs
    if not final_selections:
        # Fall back to using all Stage 1 candidates
        print(f"  Stage 2 returned no selections - using all Stage 1 candidates")
        validated_urls = [
            build_single_reference_url(v['video_id'], v['start_time'], v['end_time'])
            for v in candidate_videos
        ]
    else:
        validated_urls = [
            build_single_reference_url(v['video_id'], v.get('start_time'), v.get('end_time'))
            for v in final_selections
        ]

    reference_urls = '\n'.join(validated_urls)
    return (row_index, reference_urls)


@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Validate Video References",
    "function_name": "run_validate_video_references",
    "user_id": st.session_state.get("role", "anonymous"),
    "user_email": st.session_state.get("user_email", "anonymous")
})
def run_validate_video_references(
    sheet,
    worksheet_name: str = "Final Outline",
    model: str = 'models/gemini-2.5-flash',
    llm: str = 'gemini_2_flash',
    max_workers: int = 5
):
    """
    Validate video references in the Final Outline sheet using Gemini's video understanding.

    This function uses a two-stage analysis:
    Stage 1: Uses Gemini video understanding to analyze each video visually
    Stage 2: Uses LLM with transcripts to select the best video(s) for each learning objective

    :param sheet: Google Sheets object
    :param worksheet_name: Name of the worksheet (default: "Final Outline")
    :param model: Gemini model to use for video understanding (Stage 1)
    :param llm: LLM model to use for final video selection (Stage 2)
    :param max_workers: Maximum parallel workers for processing
    :return: None
    """
    # Load the worksheet
    worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)

    if df.empty:
        raise Exception(f"The '{worksheet_name}' sheet is empty.")

    # Validate required columns exist
    required_cols = ['Topic', 'Learning Objectives']
    for col in required_cols:
        if col not in df.columns:
            raise Exception(f"Required column '{col}' not found in sheet.")

    # Ensure video columns exist
    for col in ['video_links', 'youtube_videos']:
        if col not in df.columns:
            df[col] = ''

    # Ensure References column exists
    if 'References' not in df.columns:
        df['References'] = ''

    # Get the full course outline for scope context
    # course_outline = get_topic_outline(df, use_text_labels=True)
    course_outline = get_outline_with_los(df, include_learning_objectives=True)

    # Build task list for rows that have video links but no References yet
    task_list = []
    for row_index, row in df.iterrows():
        lo = row.get('Learning Objectives', '')
        if not isinstance(lo, str) or not lo.strip():
            continue

        video_links = str(row.get('video_links', '')).strip()
        youtube_links = str(row.get('youtube_videos', '')).strip()

        # Skip if no video links to process
        if not video_links and not youtube_links:
            continue

        # Skip if References already populated (to allow incremental processing)
        existing_refs = str(row.get('References', '')).strip()
        if existing_refs:
            print(f"Skipping row {row_index + 1}: References already populated")
            continue

        task_list.append({
            'row_index': row_index,
            'topic': row.get('Topic', ''),
            'subtopic': row.get('Subtopic', ''),
            'learning_objective': lo,
            'video_links': video_links,
            'youtube_videos': youtube_links,
            'as_is_sources': str(row.get('as_is_sources', '')).strip()
        })

    total_tasks = len(task_list)
    if total_tasks == 0:
        print("No rows to process. Either no video links found or all References already populated.")
        return

    print(f"Starting video reference validation for {total_tasks} learning objectives...")

    # Process rows in parallel
    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for task in task_list:
            future = executor.submit(
                process_single_row,
                task,
                course_outline,
                model,
                llm
            )
            futures_map[future] = task

        # Track progress
        progress = SmartProgressBar(
            total_tasks=total_tasks,
            description="Validating video references",
            save_interval=5
        )

        for future in tqdm(as_completed(futures_map), total=total_tasks):
            task = futures_map[future]
            row_index = task['row_index']

            try:
                result_index, reference_urls = future.result()

                # Update the References column
                if reference_urls:
                    df.at[row_index, 'References'] = reference_urls
                    print(f"Row {row_index + 1}: Added {len(reference_urls.split(chr(10)))} validated video reference(s)")
                else:
                    print(f"Row {row_index + 1}: No suitable videos found")

                progress.update()

                # Save periodically
                if progress.should_save():
                    print(f"Saving progress after {progress.completed_count} tasks...")
                    save_to_sheet(worksheet=worksheet, df=df)

            except Exception as e:
                print(f"Error processing row {row_index + 1}: {e}")

    # Final save
    print("All rows processed. Saving final results...")
    save_to_sheet(worksheet=worksheet, df=df)
    print("Video reference validation complete.")


def delete_video_references(sheet, worksheet_name: str = "Final Outline"):
    """
    Remove the validated video references from the References column.

    :param sheet: Google Sheets object
    :param worksheet_name: Name of the worksheet
    """
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    if 'References' in df.columns:
        df['References'] = ''
        clear_worksheet(ws)
        save_to_sheet(ws, df)
        print(f"Cleared References column in '{worksheet_name}' sheet.")


def _render_video_player(url: str):
    """
    Helper function to render a single video player with timestamp info.

    :param url: YouTube embed URL with timestamps
    """
    video_info = extract_video_info_from_url(url)

    if video_info:
        video_id = video_info['video_id']
        start_time = video_info.get('start_time', 0) or 0
        end_time = video_info.get('end_time')

        # Format timestamps for display
        start_mmss = f"{start_time // 60:02d}:{start_time % 60:02d}"
        if end_time:
            end_mmss = f"{end_time // 60:02d}:{end_time % 60:02d}"
            duration = end_time - start_time
            st.caption(f"Clip: {start_mmss} - {end_mmss} (Duration: {duration}s)")
        else:
            st.caption(f"Starts at: {start_mmss}")

        # Build the embed URL
        embed_url = f"https://www.youtube.com/embed/{video_id}?start={start_time}"
        if end_time:
            embed_url += f"&end={end_time}"

        # Display the embedded video using an iframe
        iframe_html = f'''
        <iframe
            width="100%"
            height="400"
            src="{embed_url}"
            frameborder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowfullscreen>
        </iframe>
        '''
        st.markdown(iframe_html, unsafe_allow_html=True)

        # Show the raw URL for reference
        st.code(url, language=None)
    else:
        st.error(f"Could not parse video URL: {url}")


def preview_video_references(sheet, worksheet_name: str = "Final Outline"):
    """
    Display a Streamlit UI to preview videos from the References column.
    Videos are embedded with their start and end timestamps.

    :param sheet: Google Sheets object
    :param worksheet_name: Name of the worksheet
    """
    _, df = get_sheet_data_and_df(sheet, worksheet_name)

    if df.empty:
        st.warning(f"The '{worksheet_name}' sheet is empty.")
        return

    if 'References' not in df.columns:
        st.warning("No 'References' column found in the sheet.")
        return

    # Filter rows that have video references
    rows_with_videos = df[df['References'].notna() & (df['References'].str.strip() != '')]

    if rows_with_videos.empty:
        st.info("No video references found in the sheet. Run the 'Validate Video References' step first.")
        return

    st.subheader("Video Reference Preview")
    st.write(f"Found **{len(rows_with_videos)}** learning objectives with video references.")

    # Create a selectbox to choose which LO to preview
    lo_options = []
    for idx, row in rows_with_videos.iterrows():
        topic = row.get('Topic', 'Unknown Topic')
        lo = row.get('Learning Objectives', 'Unknown LO')
        # Truncate long LOs for display
        lo_display = lo[:80] + '...' if len(lo) > 80 else lo
        lo_options.append(f"Row {idx + 1}: {topic} - {lo_display}")

    selected_option = st.selectbox(
        "Select a Learning Objective to preview videos:",
        options=lo_options,
        index=0
    )

    if selected_option:
        # Extract the row index from the selection
        selected_idx = int(selected_option.split(':')[0].replace('Row ', '')) - 1
        selected_row = df.iloc[selected_idx]

        # Display LO details
        st.markdown("---")
        st.markdown(f"**Topic:** {selected_row.get('Topic', 'N/A')}")
        st.markdown(f"**Subtopic:** {selected_row.get('Subtopic', 'N/A')}")
        st.markdown(f"**Learning Objective:** {selected_row.get('Learning Objectives', 'N/A')}")
        st.markdown("---")

        # Get the video references
        references = selected_row.get('References', '')
        if references and isinstance(references, str):
            video_urls = [url.strip() for url in references.split('\n') if url.strip()]

            if video_urls:
                st.markdown(f"**{len(video_urls)} Video Reference(s):**")

                # Use tabs if multiple videos, otherwise just show directly
                if len(video_urls) > 1:
                    tabs = st.tabs([f"Video {i + 1}" for i in range(len(video_urls))])
                    for i, (tab, url) in enumerate(zip(tabs, video_urls)):
                        with tab:
                            _render_video_player(url)
                else:
                    # Single video - display directly
                    _render_video_player(video_urls[0])
            else:
                st.info("No video URLs found in References.")
        else:
            st.info("No video references for this learning objective.")


def manual_preview_video_references(sheet, worksheet_name: str = "Final Outline", skip_manual_step: bool = False):
    """
    Manual step wrapper for video preview. Returns True to mark step as complete.

    :param sheet: Google Sheets object
    :param worksheet_name: Name of the worksheet
    :param skip_manual_step: If True, skip the manual step
    :return: True to mark step as complete
    """
    if skip_manual_step:
        return True

    # The preview is shown via pre_exec_func, this just confirms the step
    return True
