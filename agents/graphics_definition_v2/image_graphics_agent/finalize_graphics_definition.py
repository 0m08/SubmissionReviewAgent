from langsmith import traceable
import streamlit as st
from services.sheets_service import get_sheet_data_and_df, save_to_sheet, format_worksheet, hide_columns_by_name, clear_worksheet
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
import json
import base64
from services.drive_service import login_with_service_account
from pydrive2.drive import GoogleDrive

load_dotenv()


generate_final_graphics_definition_prompt = """You are an expert educational graphics designer in the HVAC industry. Your task is to select the most appropriate images from the provided set to visually support a single voiceover sentence, and to produce a simple, clear graphics definition that specifies which images to show and in what order. Your decisions should prioritize visual clarity and instructional usefulness for the voiceover sentence. 

<course_information>
Course name: {course_name}
Topic name: {topic_name}
Subtopic name: {subtopic_name}
</course_information>

<voiceover_sentence>
{vo_text}
</voiceover_sentence>

<whole_slide_context>
{slide_chunk}
</whole_slide_context>

Instructions:

1. Image Selection Rules:

- Review all provided images before making any selection. Note: Image titles are for reference only and may be inaccurate - always analyze the actual visual content of each image.
- Base image selection strictly on the meaning of the voiceover sentence, not on general topic relevance.
- First, mentally break the voiceover sentence into the distinct visual elements that must be shown for the sentence to be clearly understood.
- Select images only if they are necessary to represent one of those visual elements. An image is necessary only if removing it would make the sentence harder to understand visually.
- The number of images to select must be determined solely by how many distinct visual elements are required to represent the voiceover sentence clearly so that the sentence can be easily understood.
- Do not select images that are:
  - Generic or loosely related
  - Redundant with already selected images
  - Informational but not visually required for this specific sentence

2. Clarity & Simplicity Requirements

After selecting the required images for the voiceover sentence, your next task is to generate a graphics definition that describes how the selected images should appear on screen.

- The graphics definition must be simple, clear, and easy to read.
- Describe only what is directly visible in the selected images.
- Clearly indicate the order in which the selected images should appear.
- Do not introduce new visual elements that are not present in the selected images.
- Do not describe motion, camera movements, transitions, or animations.

3. Sequence Logic

- Treat the voiceover as a single sentence that needs to be visualized from start to finish.
- Order the selected images based on how the meaning of the sentence is most clearly understood.
- When multiple images are selected, arrange them so each image introduces a new visual element in a logical progression.
- Use a sequential order by default.
- Only imply showing multiple images at the same time if the voiceover sentence clearly requires comparing or viewing two elements together.
- Keep the sequence short and focused, including only what is necessary to support the sentence.

The following are the complete set of available images.
AVAILABLE IMAGES ({num_images} images in total):
"""


generate_final_graphics_definition_output_format = """

OUTPUT FORMAT:

Provide your output strictly in this exact XML format:

<evaluation_breakdown>

This section is for the documentation of your internal reasoning and analysis. Follow the structured steps below to ground your decisions before producing the final output.

1. Voiceover Meaning:
- Explain, in your own words, what the voiceover sentence is communicating.

2. Visual Requirements for This Sentence:
- Describe what needs to be visually shown on screen for this sentence to be clearly understood.
- Think in terms of visible objects, components, diagrams, etc.

3. Image-by-Image Visual Scan:
- Review every provided image in the order given.
- For each image, carefully observe the actual visual content of the image.
- Write a short description of what you saw in each of the image.
- Do not infer or assume content based only on the image title or any other metadata, but rather based on the actual visual content of the image.

Use the following format:
   - [Image 1 Title]: Briefly describe what you visually saw in the image.
   - [Image 2 Title]: Briefly describe what you visually saw in the image.
     ...
   Continue for all the available images in the set.

4. Candidate Evaluation and Elimination:
- Compare each image against the visual requirements identified above.
- Identify and explain which of the available images clearly support the required visuals.

5. Final Image Selection Rationale:
- Based on the comparison above, identify all the images that you plan to select for this sentence.
- Explain in detail why these images best represent the required visuals.

6. Sequence Planning:
- Explain the order in which the selected images should appear to best support the voiceover sentence.
- Base this order on clarity and how the sentence is most naturally understood.

</evaluation_breakdown>

(Based on your above evaluation, provide the final graphics definition in the following format)

<graphics_definition>

<description>
Give a simple, clear description of what visuals should appear on screen for this voiceover sentence, written using only the selected images and their order of appearance.
</description>

<selection_justification>
Explain why the selected images, taken together, fully and directly support the voiceover sentence. Describe how the selected images cover all the key ideas or visual requirements expressed in the voiceover sentence,
</selection_justification>

<selected_images>
(List of all selected images in this exact format)
1. [Image Title] | [URL]
2. [Image Title] | [URL]
...
</selected_images>

</graphics_definition>
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
        print(f"⚠️ Failed to download image from web URL {web_url}: {e}")
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


def format_graphics_definition_for_sheet(vo_text, graphics_definition_xml):
    """
    Format the graphics definition XML into the readable format for the sheet.

    :param vo_text: Voiceover text for the segment
    :param graphics_definition_xml: Graphics definition XML
    :return: Formatted graphics definition text or empty string if graphics definition XML is empty
    """
    if not graphics_definition_xml or graphics_definition_xml.strip() == "":
        return ""
    
    # Parse XML tags
    parser_chain = Chain(llm="gemini_3_flash_thinking", tags=["description", "selection_justification", "selected_images"])
    parsed = parser_chain.extract_text_in_tags(graphics_definition_xml)
    
    description = parsed.get("description", "").strip()
    selection_justification = parsed.get("selection_justification", "").strip()
    selected_images_text = parsed.get("selected_images", "").strip()
    
    # Parse selected images (format: "1. Title | URL")
    images_list = []
    for line in selected_images_text.split('\n'):
        line = line.strip()
        if not line:
            continue
        # Remove leading number and dot if present
        if re.match(r'^\d+\.\s*', line):
            line = re.sub(r'^\d+\.\s*', '', line)
        images_list.append(line)
    
    # Format the output
    formatted = f"""================================================================================
SEGMENT 1
================================================================================
VO: "{vo_text}"

Description:
{description}

Selection Justification:
{selection_justification}

Images to use for this segment:"""
    
    # Add images
    for i, image_line in enumerate(images_list, 1):
        formatted += f"\n{i}. {image_line}"
    
    return formatted


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Finalize Graphics Definition",
        "function_name": "finalize_graphics_definition_for_segment",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def finalize_graphics_definition_for_segment(vo_text, slide_chunk, image_urls, url_to_title, course_name, topic_name, subtopic_name, drive, llm="gemini_3_flash_thinking"):
    """
    Finalize graphics definition for a single segment using vision model.
    
    :param vo_text: Voiceover text for the segment
    :param slide_chunk: Full slide content
    :param image_urls: List of image URLs to analyze
    :param url_to_title: Dictionary mapping URL to title
    :param course_name: Course name
    :param topic_name: Topic name
    :param subtopic_name: Subtopic name
    :param drive: Google Drive instance
    :param llm: Language model to use
    :return: Finalized graphics definition text or empty string if no images available
    """
    print(f"\n{'─'*45}")
    print(f" 🎨 Finalizing graphics for segment")
    print(f"{'─'*45}")
    print(f"📝 VO text: \"{vo_text}\"")
    print(f"🖼️ Available images: {len(image_urls)}")
    
    if not image_urls:
        print(f"⚠️  No images available, returning empty definition")
        return ""
    
    # Build multimodal content with images
    content_parts = []
    
    # Format the prompt with variables
    prompt_text = generate_final_graphics_definition_prompt.format(
        course_name=course_name,
        topic_name=topic_name,
        subtopic_name=subtopic_name,
        vo_text=vo_text,
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
            print(f"⚠️  Could not load image {i}: {img_title}")
    
    # Add output format instructions
    content_parts.append({
        "type": "text",
        "text": generate_final_graphics_definition_output_format
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
    
    # Print segment and response for debugging (matching other steps)
    print(f"\nSegment: {vo_text}\n")
    print(f"📤 Finalize Graphics Definition Response from LLM:\n")
    print(raw_text)
    print(f"\n{'='*100}\n")
    
    # Parse the response
    parser_chain = Chain(llm=llm, tags=["evaluation_breakdown", "graphics_definition"])
    parsed = parser_chain.extract_text_in_tags(raw_text)
    refined_definition = parsed.get("graphics_definition", "")
    
    if refined_definition:
        print(f"✅ Generated finalized definition ({len(refined_definition)} chars)")
    else:
        print(f"⚠️ Failed to extract graphics definition from response")
    
    print(f"{'─'*45}\n")
    
    return refined_definition


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Finalize Graphics",
        "function_name": "process_finalize_graphics_definition_segment",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def process_finalize_graphics_definition_segment(segment_idx, vo_text, slide_chunk, drive_results, web_results, course_name, topic_name, subtopic_name, drive, llm="gemini_3_flash_thinking"):
    """
    Process a single segment: finalize graphics definition.
    
    :param segment_idx: Segment index (1-based)
    :param vo_text: Voiceover text for the segment
    :param slide_chunk: Full slide content
    :param drive_results: Drive search results text
    :param web_results: Web search results text
    :param course_name: Course name
    :param topic_name: Topic name
    :param subtopic_name: Subtopic name
    :param drive: Google Drive instance
    :param llm: Language model to use
    :return: Tuple of (segment_idx, formatted_segment_text) or (segment_idx, None) if no definition generated
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
    
    # Finalize graphics for this segment
    segment_definition_xml = finalize_graphics_definition_for_segment(
        vo_text=vo_text,
        slide_chunk=slide_chunk,
        image_urls=all_urls,
        url_to_title=url_to_title,
        course_name=course_name,
        topic_name=topic_name,
        subtopic_name=subtopic_name,
        drive=drive,
        llm=llm
    )
    
    if segment_definition_xml:
        # Format the definition for the sheet
        formatted_segment = format_graphics_definition_for_sheet(vo_text, segment_definition_xml)
        if formatted_segment:
            # Replace SEGMENT 1 with actual segment number
            formatted_segment = formatted_segment.replace("SEGMENT 1", f"SEGMENT {segment_idx}")
            return segment_idx, formatted_segment
    
    return segment_idx, None


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Finalize Graphics",
        "function_name": "process_finalize_graphics_definition_row",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def process_finalize_graphics_definition_row(index, row, course_name, drive, llm="gemini_3_flash_thinking"):
    """
    Process a single row: refine graphics for all segments and combine into final definition.
    
    :param index: Row index
    :param row: Pandas Series with row data
    :param course_name: Course name
    :param drive: Google Drive instance
    :param llm: Language model to use
    :return: Tuple of (index, graphics_definition_text) or (index, empty string) if no segments found
    """
    try:
        voiceover_segments = str(row.get("voiceover_segment", "")).strip()
        drive_results = str(row.get("drive_results", "")).strip()
        web_results = str(row.get("web_results", "")).strip()
        slide_chunk = str(row.get("Slide Chunk", "")).strip()
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
                    process_finalize_graphics_definition_segment,
                    segment_idx,
                    vo_text,
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
        segment_definitions = []
        for segment_idx in sorted(segment_results.keys()):
            segment_definitions.append(segment_results[segment_idx])
        
        # Combine all segment definitions with double newline separator
        if segment_definitions:
            final_definition = "\n\n\n".join(segment_definitions)
            print(f"✅ Combined {len(segment_definitions)} segment definitions")
            return index, final_definition
        else:
            print(f"⚠️ No definitions generated")
            return index, ""
        
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        import traceback
        traceback.print_exc()
        return index, ""


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Refine Graphics",
        "function_name": "run_finalize_graphics_definition_for_all_rows",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def run_finalize_graphics_definition_for_all_rows(sheet, llm="gemini_3_flash_thinking", max_workers=3):
    """
    Refine graphics definitions for all rows in the Slide Chunks sheet.
    
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
    
    # Ensure graphics_definition column exists
    if "graphics_definition" not in df.columns:
        df["graphics_definition"] = ""
    
    # Prepare for parallel processing - only process rows that need processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks first
        for index, row in df.iterrows():
            voiceover_segments = str(row.get("voiceover_segment", "")).strip()
            graphics_definition = str(row.get("graphics_definition", "")).strip()
            
            # Skip if voiceover_segments is empty
            if not voiceover_segments or voiceover_segments == "nan":
                continue
            
            # Skip if graphics_definition is already filled
            if graphics_definition and graphics_definition != "nan":
                continue
            
            # Submit task for processing
            future = executor.submit(process_finalize_graphics_definition_row, index, row, course_name, drive, llm)
            futures_map[future] = index
        
        # If no rows to process, return early
        if not futures_map:
            print("All rows already processed or no valid voiceover segments found.")
            return
        
        # Initialize progress tracker
        total_tasks = len(futures_map)
        progress = SmartProgressBar(
            total_tasks=total_tasks,
            description="Finalizing graphics definitions",
            save_interval=3  # Save more frequently due to longer processing time
        )
        
        # Collect results as they complete
        for future in as_completed(futures_map):
            index = futures_map[future]
            try:
                row_index, graphics_definition_text = future.result()
                
                # Update dataframe
                df.at[row_index, "graphics_definition"] = graphics_definition_text
                
                # Update progress
                progress.update()
                
                # Save every 3 rows (more frequent due to longer processing)
                if progress.should_save():
                    print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
                    save_to_sheet(worksheet, df)
                    format_worksheet(worksheet)
            except Exception as e:
                print(f"Error getting result for row {index}: {e}")
                progress.update()
    
    # Final save to sheet
    print('All graphics definitions finalized. Saving final DataFrame to sheet.')
    save_to_sheet(worksheet, df)
    format_worksheet(worksheet)
    
    # Hide columns 
    columns_to_hide = ["voiceover_segment", "search_queries", "drive_results", "web_results"]
    hide_columns_by_name(worksheet, columns_to_hide, df)
    
    print("✅ Graphics definition finalization complete and saved to sheet.")


def delete_graphics_definition(sheet):
    """
    Remove the 'graphics_definition' column from the specified worksheet.
    
    :param sheet: The gspread sheet object.
    :return: None
    """
    worksheet_name = "Slide Chunks"
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    if "graphics_definition" in df.columns:
        df = df.drop(columns=["graphics_definition"])
        clear_worksheet(ws)
        save_to_sheet(ws, df)
        print(f"🗑️ Deleted 'graphics_definition' column from '{worksheet_name}' worksheet")
    else:
        print(f"ℹ️ 'graphics_definition' column does not exist in '{worksheet_name}' worksheet")



