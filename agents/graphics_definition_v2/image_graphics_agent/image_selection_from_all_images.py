from langsmith import traceable
import streamlit as st
from services.sheets_service import get_sheet_data_and_df, save_to_sheet, format_worksheet, clear_worksheet
from services.smart_progress_bar import SmartProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import requests
from io import BytesIO
from PIL import Image
from services.llm_service import llm_with_retry
from modules.chain import Chain
from agents.vector_store_image_search.graphics_retriever_agent import pil_to_base64_data_uri
from agents.vector_store_image_search.create_vectorstore import download_image_from_drive
from dotenv import load_dotenv
import os
import base64
from services.drive_service import login_with_service_account
from pydrive2.drive import GoogleDrive

load_dotenv()


image_selection_from_all_images_prompt = """You are an expert educational graphics curator specializing in the field of HVAC.

Your task is to review a provided set of image candidates and identify all images that are relevant to a single voiceover sentence from an educational e-learning slide. An image is considered relevant if it visually relates to, supports, or illustrates any concept, object, component, or idea mentioned or implied in the voiceover sentence.

<course_information>
Course name: {course_name}
Topic name: {topic_name}
Subtopic name: {subtopic_name}
</course_information>

<voiceover_sentence>
{vo_text}
</voiceover_sentence>

<whole_slide_context>
Slide Title: {slide_title}
Slide Content: {slide_chunk}
</whole_slide_context>

Instructions:

1. Relevance-Based Selection

- Review all provided image candidates.
- Select every image that is relevant to any part of the voiceover sentence.
- An image may be relevant even if it represents only part of the sentence or provides supportive or illustrative context.

2. Meaning-Based Judgment

- Judge relevance based on the meaning and instructional intent of the voiceover sentence.
- Use the full slide context to resolve references, pronouns, or implied meaning if needed.
- Do not select images based on general topic relevance alone.

3. Inclusion Without Redundancy

- Do not try to minimize the number of selected images.
- Include all images that meaningfully support understanding of any part of the voiceover sentence.
- Do not exclude an image solely because another image supports a similar concept.
- However, avoid selecting images that are purely redundant and do not add any new visual perspective, detail, or informational value beyond another selected image.

4. Visual Grounding

- Base all decisions on what is actually visible in each image.
- Do not rely on image titles or filenames.

The following are the complete set of available images that you need to select from:
AVAILABLE IMAGES ({num_images} images in total):
"""


image_selection_from_all_images_output_format = """

OUTPUT FORMAT:

Provide your output strictly in the following format:

<output>

<evaluation_breakdown>

<voiceover_sentence_understanding>
Briefly explain what the voiceover sentence is communicating, using slide context to resolve any references or implied meaning if needed.
</voiceover_sentence_understanding>

<image_candidate_scan>
Create a numbered list of all provided image candidates and briefly describe what you see in each of the image,
</image_candidate_scan>

<relevance_decision>
Explain which images are relevant to the voiceover sentence and why, taking all the instructions into consideration.
</relevance_decision>

</evaluation_breakdown>

<selected_images>
(List of all selected images in this exact format)
1. [Image Title] | [URL]
2. [Image Title] | [URL]
...
</selected_images>

</output>

(Ensure that you follow this exact XML format and do not add any extra text or comments outside the <output>, <evaluation_breakdown>, <selected_images> tags)
"""


def extract_drive_file_id(url):
    """
    Extract the Google Drive file ID from a URL.

    :param url: Google Drive URL
    :return: Google Drive file ID or None if not found
    """
    if not url:
        return None
    
    # Pattern for /file/d/FILE_ID/
    match = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    
    # Pattern for ?id=FILE_ID or &id=FILE_ID
    match = re.search(r'[?&]id=([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    
    # If it looks like just a file ID (no slashes, reasonable length)
    if '/' not in url and 'http' not in url.lower() and len(url) > 10:
        return url
    
    return None


def load_image_from_drive_url(url, drive, ref_id=""):
    """
    Load an image from a Google Drive URL using the Drive API.
    
    :param url: Google Drive URL
    :param drive: Google Drive instance
    :param ref_id: Reference ID for logging
    :return: PIL Image or None if loading fails
    """
    if not url or not drive:
        return None
    
    file_id = extract_drive_file_id(url)
    if not file_id:
        print(f"⚠️ Could not extract file ID from URL for {ref_id}: {url}")
        return None
    
    try:
        img = download_image_from_drive(drive, file_id)
        if img:
            print(f"✅ Loaded image from Drive for {ref_id}")
        return img
    except Exception as e:
        print(f"⚠️ Failed to load image from Drive for {ref_id}: {e}")
        return None


def get_drive_instance():
    """
    Get Google Drive instance from session state or initialize from environment.
    
    :return: Google Drive instance or None if initialization fails
    """
    # First try to get from session state
    if "drive" in st.session_state:
        return st.session_state["drive"]
    
    # Fallback: initialize from environment variables
    try:
        key_bytes = base64.b64decode(os.environ["GDRIVE_SA_B64"])
        sa_json = key_bytes.decode()
        gauth = login_with_service_account(json_str=sa_json)
        gauth.ServiceAuth()
        drive = GoogleDrive(gauth)
        return drive
    except Exception as e:
        print(f"⚠️ Could not initialize Drive from environment: {e}")
        return None


def is_drive_url(url):
    """
    Check if URL is a Google Drive URL.
    
    :param url: URL to check
    :return: True if URL is a Google Drive URL, False otherwise
    """
    return bool(re.search(r'drive\.google\.com', url, re.IGNORECASE))


def download_image_from_web_url(web_url):
    """
    Download image from a direct web URL.
    
    :param web_url: Web URL to download image from
    :return: PIL Image or None if download fails
    """
    try:
        response = requests.get(web_url, timeout=10, stream=True, allow_redirects=True)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img = img.convert("RGB")
            return img
        return None
    except Exception as e:
        print(f"⚠️  Failed to download image from web URL {web_url}: {e}")
        return None


def load_image_from_url(url, drive=None):
    """
    Load image from either Google Drive URL or web URL.
    
    :param url: Image URL (can be Google Drive or web URL)
    :param drive: Google Drive instance (optional, for Drive images)
    :return: PIL Image or None if loading fails
    """
    if not url or not url.startswith('http'):
        return None
    
    if is_drive_url(url):
        return load_image_from_drive_url(url, drive, "")
    else:
        return download_image_from_web_url(url)


def parse_urls_from_results(results_text, segment_num):
    """
    Parse titles and URLs from drive_results or web_results for a specific segment.
    
    :param results_text: The drive_results or web_results column content
    :param segment_num: Segment number to extract URLs for
    :return: List of dictionaries with 'title' and 'url' keys or empty list if no URLs found
    """
    if not results_text or results_text.strip() == "" or results_text == "nan":
        return []
    
    # Find the segment section
    segment_pattern = rf'---SEGMENT_{segment_num}---\s*\n(.*?)(?=\n---SEGMENT_|\Z)'
    match = re.search(segment_pattern, results_text, re.DOTALL)
    
    if not match:
        return []
    
    segment_content = match.group(1).strip()
    items = []
    
    # Parse each line: "Title: {title} | URL: {url}"
    for line in segment_content.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Try to parse "Title: ... | URL: ..." format
        if " | URL: " in line:
            parts = line.split(" | URL: ", 1)
            if len(parts) == 2:
                title_part = parts[0]
                url = parts[1].strip()
                
                # Extract title (remove "Title: " prefix)
                if title_part.startswith("Title: "):
                    title = title_part[7:].strip()
                else:
                    title = title_part.strip()
                
                if url:
                    items.append({"title": title, "url": url})
        # Fallback: if line is just a URL (for backward compatibility)
        elif line.startswith('http'):
            items.append({"title": "Untitled", "url": line})
    
    return items


def parse_segments_from_voiceover(voiceover_text):
    """
    Parse segments from voiceover_segment column (newline-separated).
    
    :param voiceover_text: The voiceover_segment column content
    :return: List of tuples (segment_index, segment_text) where segment_index is 1-based or empty list if no segments found
    """
    if not voiceover_text or voiceover_text.strip() == "" or voiceover_text == "nan":
        return []
    
    segments = voiceover_text.strip().split('\n')
    return [(i + 1, seg.strip()) for i, seg in enumerate(segments) if seg.strip()]


def format_selected_images_for_segment(selected_images_text):
    """
    Parse and format selected images for output to image_pool column.
    
    :param selected_images_text: Text from <selected_images> tag
    :return: List of formatted image lines: "Title | URL"
    """
    if not selected_images_text or selected_images_text.strip() == "":
        return []
    
    formatted_lines = []
    for line in selected_images_text.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Remove leading number and dot if present (format: "1. Title | URL")
        if re.match(r'^\d+\.\s*', line):
            line = re.sub(r'^\d+\.\s*', '', line)
        
        # Ensure format is "Title | URL"
        if " | " in line:
            formatted_lines.append(line)
        elif line.startswith('http'):
            # If it's just a URL, format as "Untitled | URL"
            formatted_lines.append(f"Untitled | {line}")
    
    return formatted_lines


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Image Selection from All Images",
        "function_name": "select_images_from_all_for_segment",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def select_images_from_all_for_segment(vo_text, slide_title, slide_chunk, image_urls, url_to_title, course_name, topic_name, subtopic_name, drive, llm="gemini_3_flash_thinking"):
    """
    Select relevant images from all available images for a single segment using vision model.
    
    :param vo_text: Voiceover text for the segment
    :param slide_title: Slide title
    :param slide_chunk: Full slide content
    :param image_urls: List of image URLs to analyze
    :param url_to_title: Dictionary mapping URL to title
    :param course_name: Course name
    :param topic_name: Topic name
    :param subtopic_name: Subtopic name
    :param drive: Google Drive instance
    :param llm: Language model to use
    :return: Selected images text or empty string if no images available
    """
    print(f"\n{'─'*45}")
    print(f" 🎯 Selecting images for segment")
    print(f"{'─'*45}")
    print(f"📝 VO text: \"{vo_text}\"")
    print(f"🖼️ Available images: {len(image_urls)}")
    
    if not image_urls:
        print(f"⚠️  No images available, returning empty selection")
        return ""
    
    # Build multimodal content with images
    content_parts = []
    
    # Format the prompt with variables
    prompt_text = image_selection_from_all_images_prompt.format(
        course_name=course_name,
        topic_name=topic_name,
        subtopic_name=subtopic_name,
        vo_text=vo_text,
        slide_title=slide_title,
        slide_chunk=slide_chunk,
        num_images=len(image_urls)
    )
    
    content_parts.append({
        "type": "text",
        "text": prompt_text
    })
    
    # Load and add each image
    loaded_images = []
    for i, img_url in enumerate(image_urls, 1):
        print(f"📥 Loading image {i}/{len(image_urls)}: {img_url[:50]}...")
        
        # Get title for this image
        img_title = url_to_title.get(img_url, "Untitled")
        
        pil_image = load_image_from_url(img_url, drive)
        
        if pil_image:
            # Add label text with Title and URL
            label_text = f"\n--- Image {i} of {len(image_urls)} ---\n"
            label_text += f"Title: {img_title}\n"
            label_text += f"URL: {img_url}\n"
            
            content_parts.append({
                "type": "text",
                "text": label_text
            })
            
            content_parts.append({
                "type": "image_url",
                "image_url": pil_to_base64_data_uri(pil_image)
            })
            
            loaded_images.append(pil_image)
            print(f"✅ Loaded image {i}")
        else:
            content_parts.append({
                "type": "text",
                "text": f"\n--- Image {i} of {len(image_urls)} ---\nTitle: {img_title}\nURL: {img_url}\n[Image {i} could not be loaded; evaluate based on metadata only]\n"
            })
            print(f"⚠️ Could not load image {i}: {img_title}")
    
    # Add output format instructions
    content_parts.append({
        "type": "text",
        "text": image_selection_from_all_images_output_format
    })
       
    print(f"🤖 Calling vision model with {len(loaded_images)} loaded images...")
    messages = [("user", content_parts)]
    raw_response = llm_with_retry(messages, llm_name=llm)
    
    # Extract response
    if hasattr(raw_response, "content"):
        raw_text = raw_response.content
    elif isinstance(raw_response, dict):
        raw_text = raw_response.get("content") or raw_response.get("text", "")
    else:
        raw_text = str(raw_response)
    
    # Print segment and response for debugging
    print(f"\nSegment: {vo_text}\n")
    print(f"📤 Image Selection Response from LLM:\n")
    print(raw_text)
    print(f"\n{'='*100}\n")
    
    # Parse the response
    parser_chain = Chain(llm=llm, tags=["evaluation_breakdown", "selected_images"])
    parsed = parser_chain.extract_text_in_tags(raw_text)
    selected_images_text = parsed.get("selected_images", "")
    
    if selected_images_text:
        print(f"✅ Extracted selected images ({len(selected_images_text)} chars)")
    else:
        print(f"⚠️  Failed to extract selected images from response")
    
    print(f"{'─'*45}\n")
    
    return selected_images_text


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Image Selection from All Images",
        "function_name": "process_image_selection_segment",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def process_image_selection_segment(segment_idx, vo_text, slide_title, slide_chunk, drive_results, web_results, course_name, topic_name, subtopic_name, drive, llm="gemini_3_flash_thinking"):
    """
    Process a single segment: select relevant images from all available images.
    
    :param segment_idx: Segment index (1-based)
    :param vo_text: Voiceover text for the segment
    :param slide_title: Slide title
    :param slide_chunk: Full slide content
    :param drive_results: Drive search results text
    :param web_results: Web search results text
    :param course_name: Course name
    :param topic_name: Topic name
    :param subtopic_name: Subtopic name
    :param drive: Google Drive instance
    :param llm: Language model to use
    :return: Tuple of (segment_idx, formatted_segment_text) or (segment_idx, None) if no selection generated
    """
    print(f"\n📦 Processing SEGMENT_{segment_idx}")
    
    # Get image items (with title and URL) for this segment from both drive and web results
    drive_items = parse_urls_from_results(drive_results, segment_idx)
    web_items = parse_urls_from_results(web_results, segment_idx)
    
    # Combine and deduplicate by URL
    seen_urls = set()
    all_items = []
    for item in drive_items + web_items:
        url = item.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            all_items.append(item)
    
    # Extract URLs list for the function
    all_urls = [item["url"] for item in all_items]
    # Create a mapping of URL to title for easy lookup
    url_to_title = {item["url"]: item.get("title", "Untitled") for item in all_items}
    
    print(f"🔗 Found {len(drive_items)} drive items, {len(web_items)} web items")
    print(f"📎 Total unique images: {len(all_urls)}")
    
    if not all_urls:
        print(f"⚠️ No images available for segment {segment_idx}, skipping")
        return segment_idx, None
    
    # Select images for this segment
    selected_images_text = select_images_from_all_for_segment(
        vo_text=vo_text,
        slide_title=slide_title,
        slide_chunk=slide_chunk,
        image_urls=all_urls,
        url_to_title=url_to_title,
        course_name=course_name,
        topic_name=topic_name,
        subtopic_name=subtopic_name,
        drive=drive,
        llm=llm
    )
    
    if selected_images_text:
        # Format the selected images
        formatted_images = format_selected_images_for_segment(selected_images_text)
        if formatted_images:
            # Format segment output: "---SEGMENT_N---\nTitle | URL\n..."
            segment_output = [f"---SEGMENT_{segment_idx}---"] + formatted_images
            return segment_idx, '\n'.join(segment_output)
    
    return segment_idx, None


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Image Selection from All Images",
        "function_name": "process_image_selection_row",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def process_image_selection_row(index, row, course_name, drive, llm="gemini_3_flash_thinking"):
    """
    Process a single row: select images for all segments and combine into image_pool output.
    
    :param index: Row index
    :param row: Pandas Series with row data
    :param course_name: Course name
    :param drive: Google Drive instance
    :param llm: Language model to use
    :return: Tuple of (index, image_pool_text) or (index, empty string) if no segments found
    """
    try:
        voiceover_segments = str(row.get("voiceover_segment", "")).strip()
        drive_results = str(row.get("drive_results", "")).strip()
        web_results = str(row.get("web_results", "")).strip()
        slide_chunk = str(row.get("Slide Chunk", "")).strip()
        slide_title = str(row.get("Slide Chunk Title", "")).strip()
        topic_name = str(row.get("Topic", "")).strip()
        subtopic_name = str(row.get("Subtopic", "")).strip()
        
        # Skip if required fields are empty
        if not voiceover_segments or voiceover_segments == "nan":
            return index, ""
        
        if not slide_chunk or slide_chunk == "nan":
            return index, ""
        
        # Parse segments
        segments = parse_segments_from_voiceover(voiceover_segments)
        
        if not segments:
            return index, ""
        
        print(f"\n{'═'*50}")
        print(f"📋 Processing row {index}: {len(segments)} segments")
        print(f"{'═'*50}")
        
        # Execute all segments in parallel
        with ThreadPoolExecutor(max_workers=len(segments)) as executor:
            # Submit all segments
            futures = {
                executor.submit(
                    process_image_selection_segment,
                    segment_idx,
                    vo_text,
                    slide_title,
                    slide_chunk,
                    drive_results,
                    web_results,
                    course_name,
                    topic_name,
                    subtopic_name,
                    drive,
                    llm
                ): segment_idx
                for segment_idx, vo_text in segments
            }
            
            # Collect results as they complete
            segment_results = {}
            for future in as_completed(futures):
                segment_idx = futures[future]
                try:
                    seg_idx, formatted_segment = future.result()
                    if formatted_segment:
                        segment_results[seg_idx] = formatted_segment
                except Exception as e:
                    print(f"❌ Error processing segment {segment_idx}: {e}")
        
        # Format results in order (by segment_idx)
        segment_selections = []
        for segment_idx in sorted(segment_results.keys()):
            segment_selections.append(segment_results[segment_idx])
        
        # Combine all segment selections with double newline separator
        if segment_selections:
            image_pool_text = "\n\n".join(segment_selections)
            print(f"✅ Combined {len(segment_selections)} segment selections")
            return index, image_pool_text
        else:
            print(f"⚠️  No selections generated")
            return index, ""
        
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        import traceback
        traceback.print_exc()
        return index, ""


def validate_image_pool_row(row):
    """
    Validate that image_pool matches voiceover_segment:
    - Row is not empty
    - All segments from voiceover_segment have results
    - No gaps in segment numbering (must be sequential starting from 1)
    
    :param row: Pandas Series with row data
    :return: Tuple (is_valid, error_message)
    """
    vo_segments_text = str(row.get("voiceover_segment", "")).strip()
    image_pool_text = str(row.get("image_pool", "")).strip()
    
    # Skip validation if voiceover_segment is empty
    if not vo_segments_text or vo_segments_text == "nan":
        return True, None
    
    # Count segments in voiceover_segment (split by newline)
    vo_segments = [seg.strip() for seg in vo_segments_text.split('\n') if seg.strip()]
    expected_count = len(vo_segments)
    
    # Skip validation if image_pool is empty (will be caught by retry logic)
    if not image_pool_text or image_pool_text == "nan" or image_pool_text.strip() == "":
        return False, "image_pool is empty"
    
    # Skip validation if it's an error marker
    if image_pool_text.startswith("ERROR:"):
        return False, "image_pool contains error marker"
    
    # Parse segment numbers from image_pool
    segment_pattern = r'---SEGMENT_(\d+)---'
    segment_numbers = [int(match) for match in re.findall(segment_pattern, image_pool_text)]
    
    if not segment_numbers:
        return False, "No segment markers found in image_pool"
    
    # Check count matches
    actual_count = len(segment_numbers)
    if actual_count != expected_count:
        return False, f"Segment count mismatch: expected {expected_count}, found {actual_count}"
    
    # Check for gaps (must be sequential starting from 1)
    segment_numbers_sorted = sorted(segment_numbers)
    expected_sequence = list(range(1, expected_count + 1))
    
    if segment_numbers_sorted != expected_sequence:
        missing = set(expected_sequence) - set(segment_numbers_sorted)
        extra = set(segment_numbers_sorted) - set(expected_sequence)
        error_parts = []
        if missing:
            error_parts.append(f"missing segments: {sorted(missing)}")
        if extra:
            error_parts.append(f"extra segments: {sorted(extra)}")
        return False, f"Segment numbering gap: {', '.join(error_parts)}"
    
    return True, None


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Image Selection from All Images",
        "function_name": "run_image_selection_from_all_images_for_all_rows",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def run_image_selection_from_all_images_for_all_rows(sheet, llm="gemini_3_flash_thinking", max_workers=3):
    """
    Select relevant images from all available images for all rows in the Slide Chunks sheet.
    
    :param sheet: The gspread sheet object.
    :param llm: Language model to use.
    :param max_workers: Number of parallel workers (default 3, lower due to image loading).
    :return: None if initialization fails
    """
    worksheet_name = "Slide Chunks"
    
    # Get drive instance
    drive = get_drive_instance()
    if not drive:
        print("⚠️ Drive instance not available. Some Drive images may not load.")
    
    # Fetch course info
    _, course_info_df = get_sheet_data_and_df(sheet, "Course info")
    course_name = course_info_df.loc[0, "Course Name"]
    
    # Load the worksheet and DataFrame
    worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)
    
    # Ensure image_pool column exists
    if "image_pool" not in df.columns:
        df["image_pool"] = ""
    
    # Prepare for parallel processing - only process rows that need processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks first
        for index, row in df.iterrows():
            voiceover_segments = str(row.get("voiceover_segment", "")).strip()
            image_pool = str(row.get("image_pool", "")).strip()
            
            # Skip if voiceover_segments is empty
            if not voiceover_segments or voiceover_segments == "nan":
                continue
            
            # Skip if image_pool is already filled
            if image_pool and image_pool != "nan":
                continue
            
            # Submit task for processing
            future = executor.submit(process_image_selection_row, index, row, course_name, drive, llm)
            futures_map[future] = index
        
        # If no rows to process, return early
        if not futures_map:
            print("All rows already processed or no valid voiceover segments found.")
            return
        
        # Initialize progress tracker
        total_tasks = len(futures_map)
        progress = SmartProgressBar(
            total_tasks=total_tasks,
            description="Selecting images from all available images",
            save_interval=3  # Save more frequently due to longer processing time
        )
        
        # Collect results as they complete
        for future in as_completed(futures_map):
            index = futures_map[future]
            try:
                row_index, image_pool_text = future.result()
                
                # Update dataframe
                df.at[row_index, "image_pool"] = image_pool_text
                
                # Update progress
                progress.update()
                
                # Save every 3 rows (more frequent due to longer processing)
                if progress.should_save():
                    print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
                    save_to_sheet(worksheet, df)
                    format_worksheet(worksheet)
            except Exception as e:
                print(f"Error getting result for row {index}: {e}")
                # Update dataframe with error marker so row is marked as processed
                df.at[index, "image_pool"] = f"ERROR: {str(e)}"
                progress.update()

    # Save final results before validation
    save_to_sheet(worksheet, df)
    format_worksheet(worksheet)

    # Validation and retry logic
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        # Reload dataframe to get latest state
        worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)
        
        # Validate all rows and find invalid ones
        invalid_rows = []
        for index, row in df.iterrows():
            is_valid, error_msg = validate_image_pool_row(row)
            if not is_valid:
                invalid_rows.append((index, row, error_msg))
        
        if not invalid_rows:
            break
        
        retry_count += 1
        print(f"\n⚠️ Found {len(invalid_rows)} rows with invalid image_pool. Retrying (attempt {retry_count}/{max_retries})...")
        for idx, row, error in invalid_rows[:3]:  # Show first 3 errors
            print(f"  Row {idx}: {error}")
        
        # Clear image_pool for invalid rows
        for index, row, error_msg in invalid_rows:
            df.at[index, "image_pool"] = ""
        
        # Save cleared state
        save_to_sheet(worksheet, df)
        format_worksheet(worksheet)
        
        # Retry processing invalid rows
        futures_map = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for index, row, error_msg in invalid_rows:
                future = executor.submit(process_image_selection_row, index, row, course_name, drive, llm)
                futures_map[future] = index
            
            # Collect results
            for future in as_completed(futures_map):
                index = futures_map[future]
                try:
                    row_index, image_pool_text = future.result()
                    df.at[row_index, "image_pool"] = image_pool_text
                except Exception as e:
                    print(f"Error getting result for row {index} on retry: {e}")
                    df.at[index, "image_pool"] = f"ERROR: {str(e)}"
        
        # Save after retry
        save_to_sheet(worksheet, df)
        format_worksheet(worksheet)
    
    if retry_count > 0:
        # Check final state
        worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)
        final_invalid = []
        for index, row in df.iterrows():
            is_valid, error_msg = validate_image_pool_row(row)
            if not is_valid:
                final_invalid.append((index, error_msg))
        
        if final_invalid:
            print(f"⚠️ After {retry_count} retry attempt(s), {len(final_invalid)} rows still have invalid image_pool.")
            for idx, error in final_invalid[:5]:  # Show first 5
                print(f"  Row {idx}: {error}")
        else:
            print(f"✅ All rows validated after {retry_count} retry attempt(s).")
    
    # Final save to sheet
    print('All image selections completed. Saving final DataFrame to sheet.')
    save_to_sheet(worksheet, df)
    format_worksheet(worksheet)
    print("✅ Image selection from all images complete and saved to sheet.")


def delete_image_pool(sheet):
    """
    Remove the 'image_pool' column from the specified worksheet.
    
    :param sheet: The gspread sheet object.
    :return: None
    """
    worksheet_name = "Slide Chunks"
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    if "image_pool" in df.columns:
        df = df.drop(columns=["image_pool"])
        clear_worksheet(ws)
        save_to_sheet(ws, df)
        print(f"🗑️ Deleted 'image_pool' column from '{worksheet_name}' worksheet")
    else:
        print(f"ℹ️ 'image_pool' column does not exist in '{worksheet_name}' worksheet")