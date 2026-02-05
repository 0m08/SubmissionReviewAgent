from modules.chain import Chain
from langsmith import traceable
import streamlit as st
from services.sheets_service import get_sheet_data_and_df, save_to_sheet, format_worksheet, clear_worksheet
from services.smart_progress_bar import SmartProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import re

load_dotenv()

# Prompt to use when we want the visual assingment to be flexible
storyboard_agent_prompt = """You are a senior instructional visual designer specializing in HVAC e-learning content. Your task is to create a visual storyboard for a slide. The storyboard should plan what visual ideas need to be shown, the order in which they appear, and how visuals transition as the slide narration progresses. This storyboard will later be used by graphics designers to select and assemble appropriate visuals for the slide.
 
These are the inputs:

<course_information>
Course name: {course_name}
Topic name: {topic_name}
Subtopic name: {subtopic_name}
</course_information>

<slide_content>
Slide Title: {slide_title}
Slide Content: {slide_content}
</slide_content>

Instructions and Guidelines:

1. Core Responsibility
   - Your responsibility is to plan the visual storytelling for the slide as it is narrated.
   - Focus on what visual ideas must appear, when they should appear, and how they transition, so the narration is easy to follow visually.
   - Only plan visual ideas that are directly implied by or necessary to support the narration; do not introduce new instructional content.

2. Break the Narration into Visual Moments
   - Read the slide content as it would be spoken aloud during narration, not as static on-screen text.
   - Identify the distinct parts of the narration where the on-screen visual should change or update.
   - Each visual moment should correspond to a specific phrase, clause, or idea in the narration.
   - Do not over-segment. Only create a new visual moment when the idea being explained clearly shifts.

3. Define the Visual Idea for Each Narration Part
   - For each part of the narration, identify the key idea being communicated and decide what must be visible on screen at that moment for clarity. Focus on what the learner needs to see to understand the narration.
   - Visual ideas may include:
      - Components or parts being referenced
      - Actions or processes being described
      - Conditions, states, or outcomes being explained
   - IMPORTANT: Use generic terms like "visual" or "visual content" when describing visual ideas. Do not use image-specific terms like "image", "photo", "picture", "2D image", etc. The visual can be either a static image or a video clip, so use neutral terminology.

4. Plan the Order and Transitions
   - Arrange visual ideas in the exact order they should appear as the narration progresses.
   - Describe how visuals should transition between narration parts.
   - Keep transitions instructionally logical and easy to follow.

5. Prioritize Instructional Clarity
   - Favor clarity and relevance over visual variety.
   - Each visual idea must directly help the learner understand the narration at that moment.
   - Avoid suggesting redundant or too many visuals that do not add instructional value.
   - IMPORTANT: You should strictly assign only 1 visual to the respective narattion parts.
   
Output:

Provide your output strictly in the following format:

<detailed_analysis>
Use this section as a reasoning scratchpad to think through the slide and plan the visual storyboard before producing the final output. It is acceptable for this section to be very detailed or lengthy if needed to arrive at a clear, instructionally sound storyboard.
</detailed_analysis>

(Based on your above analysis, provide the final output below)

<output>

<storyboard_steps>

<storyboard_step>

<narration_part>
(Exact phrase or clause from the slide narration that this visual moment aligns with)
</narration_part>

<visual_idea>
(Describe the core visual idea that should be shown on screen at this moment.)
</visual_idea>

<transition_from_previous>
(Describe how this visual moment transitions from the previous one. If this is the first visual moment, state that it is the initial visual.)   
</transition_from_previous>

</storyboard_step>

<!-- Repeat <storyboard_step> as needed, in narration order to cover all of the slide content-->

</storyboard_steps>

</output>

(Ensure that you follow this exact XML format in your output)
"""

# Prompt to use when we want only one visual idea for the entire slide
storyboard_agent_for_entire_slide_prompt = """You are a senior instructional visual designer specializing in HVAC e-learning content. Your task is to create a visual storyboard for a slide. The storyboard should plan what kind of visual needs to be shown as the entire slide is narrated. This storyboard will later be used by graphics designers to select and assemble appropriate visual for the slide.
 
These are the inputs:

<course_information>
Course name: {course_name}
Topic name: {topic_name}
Subtopic name: {subtopic_name}
</course_information>

<slide_content>
Slide Title: {slide_title}
Slide Content: {slide_content}
</slide_content>

Instructions and Guidelines:

1. Core Responsibility
   - Your responsibility is to plan the visual storyboard for the slide as it is narrated.
   - Read the slide content as it would be spoken aloud during narration, not as static on-screen text.
   - Focus on what visual must appear on the screen as the slide is narrated.
   - Only plan visual idea that is directly implied by or necessary to support the narration; do not introduce new instructional content.

2. Define the Visual Idea for the Entire Slide
   - For the entire slide, identify the key visual idea that should be shown on screen as the slide is narrated.
   - Visual idea may include:
      - Components or parts being referenced
      - Actions or processes being described
      - Conditions, states, or outcomes being explained
   - Remember, we want to assign only 1 visual for the entire slide, so the visual idea should be the most important and relevant visual that will support the entire slide.
   - IMPORTANT: Use generic terms like "visual" or "visual content" when describing the visual idea. Do not use image-specific terms like "image", "photo", "picture", "2D image", etc. The visual can be either a static image or a video clip, so use neutral terminology.

Output:

Provide your output strictly in the following format:

<detailed_analysis>
Use this section as a reasoning scratchpad to think through the slide and plan the visual storyboard before producing the final output. It is acceptable for this section to be very detailed or lengthy if needed to arrive at a clear, instructionally sound storyboard.
</detailed_analysis>

(Based on your above analysis, provide the final output below)

<output>

<slide_narattion>
(The whole slide exactly as it would be spoken aloud during narration. You sh copy this from the slide content input)
</slide_narattion>

<storyboard_for_the_slide>
(Describe the visual idea for the entire slide. Ensure that you dont use any new lines while giving this storyboard for the slide)
</storyboard_for_the_slide>

</output>

(Ensure that you follow this exact XML format in your output)   
"""


# Prompt to use when we want exactly one visual assingment for each sentence of the slide.
storyboard_agent_for_each_sentence_prompt = """You are a senior instructional visual designer specializing in HVAC e-learning content. Your task is to create a visual storyboard for a slide. The storyboard should plan what visual ideas need to be shown, the order in which they appear, and how visuals transition between sentences as the slide narration progresses. This storyboard will later be used by graphics designers to select and assemble appropriate visuals for the slide.
 
These are the inputs:

<course_information>
Course name: {course_name}
Topic name: {topic_name}
Subtopic name: {subtopic_name}
</course_information>

<slide_content>
Slide Title: {slide_title}
Slide Content: {slide_content}
</slide_content>

Instructions and Guidelines:

1. Core Responsibility
   - Your responsibility is to plan the visual storytelling for the slide as it is narrated.
   - Focus on what visual ideas must appear, when they should appear, and how they transition, so the narration is easy to follow visually.
   - Only plan visual ideas that are directly implied by or necessary to support the narration; do not introduce new instructional content.

2. Break the Narration into Visual Moments
   - Read the slide content as it would be spoken aloud during narration, not as static on-screen text.
   - Identify the distinct parts of the narration where the on-screen visual should change or update.
   - Each visual moment should correspond to a specific phrase, clause, or idea in the narration.
   - Do not over-segment. Only create a new visual moment when the idea being explained clearly shifts.

3. Define the Visual Idea for Each Narration Part
   - For each part of the narration, identify the key idea being communicated and decide what must be visible on screen at that moment for clarity. Focus on what the learner needs to see to understand the narration.
   - Visual ideas may include:
      - Components or parts being referenced
      - Actions or processes being described
      - Conditions, states, or outcomes being explained
   - IMPORTANT: Use generic terms like "visual" or "visual content" when describing visual ideas. Do not use image-specific terms like "image", "photo", "picture", "2D image", etc. The visual can be either a static image or a video clip, so use neutral terminology.

4. Plan the Order and Transitions
   - Arrange visual ideas in the exact order they should appear as the narration progresses.
   - Describe how visuals should transition between sentences.
   - Keep transitions instructionally logical and easy to follow.

5. Prioritize Instructional Clarity
   - Favor clarity and relevance over visual variety.
   - Each visual idea must directly help the learner understand the narration of the sentence at that moment.
   - IMPORTANT: You should strictly assign only 1 visual to the respective sentence.
   
Output:

Provide your output strictly in the following format:

<detailed_analysis>
Use this section as a reasoning scratchpad to think through the slide and plan the visual storyboard before producing the final output. It is acceptable for this section to be very detailed or lengthy if needed to arrive at a clear, instructionally sound storyboard.
</detailed_analysis>

(Based on your above analysis, provide the final output below)

<output>

<storyboard_steps>

<storyboard_step>

<narration_part>
(Exact sentence from the slide narration that this visual moment aligns with)
</narration_part>

<visual_idea>
(Describe the core visual idea that should be shown on screen for this slide sentence)
</visual_idea>

<transition_from_previous>
(Describe how this visual moment transitions from the previous one. If this is the first sentence of the slide, state that it is the initial visual.)
</transition_from_previous>

</storyboard_step>

<!-- Repeat <storyboard_step> as needed, in narration order to cover all of the sentences from the slide content-->

</storyboard_steps>

</output>

(Ensure that you follow this exact XML format in your output)

"""


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Storyboard Agent",
        "function_name": "generate_storyboard_for_slide",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def generate_storyboard_for_slide(course_name, topic_name, subtopic_name, slide_title, slide_content, visual_assignment_strategy="Flexible, let the agent decide", llm="gemini_3_flash_thinking"):
    """
    Generate a visual storyboard for a single slide.

    :param course_name: The course name.
    :param topic_name: The topic name.
    :param subtopic_name: The subtopic name.
    :param slide_title: The slide title.
    :param slide_content: The slide content.
    :param visual_assignment_strategy: The visual assignment strategy from the sheet.
    :param llm: The language model to use.
    :return: Tuple of (storyboard_output, strategy_type) where strategy_type indicates the format.
    """
    # Select prompt based on visual assignment strategy
    if visual_assignment_strategy == "1 Visual for the whole Slide":
        prompt_template = storyboard_agent_for_entire_slide_prompt
        strategy_type = "entire_slide"
        chain_tags = ["output", "storyboard_for_the_slide", "detailed_analysis"]
    elif visual_assignment_strategy == "1 Visual per Sentence":
        prompt_template = storyboard_agent_for_each_sentence_prompt
        strategy_type = "per_sentence"
        chain_tags = ["output", "storyboard_steps", "detailed_analysis"]
    else:  # Default: "Flexible, let the agent decide"
        prompt_template = storyboard_agent_prompt
        strategy_type = "flexible"
        chain_tags = ["output", "storyboard_steps", "detailed_analysis"]
    
    # Initialize the storyboard agent with appropriate tags
    storyboard_agent = Chain(llm=llm, tags=chain_tags)

    # # Print the formatted prompt for debugging
    # formatted_prompt = prompt_template.format(
    #     course_name=course_name,
    #     topic_name=topic_name,
    #     subtopic_name=subtopic_name,
    #     slide_title=slide_title,
    #     slide_content=slide_content
    # )
    # print("\n🔍 Storyboard Agent Prompt Being Sent to LLM:\n")
    # print(formatted_prompt)
    # print("\n" + "=" * 100 + "\n")

    # Add the user message
    storyboard_agent.add_message(
        role="user",
        content=prompt_template.format(
            course_name=course_name,
            topic_name=topic_name,
            subtopic_name=subtopic_name,
            slide_title=slide_title,
            slide_content=slide_content
        )
    )

    # Run the storyboard agent
    response = storyboard_agent.run()

    # Get the full response text 
    full_response_text = response.get("text", "")
    
    storyboard_output = response.get("output", "")
    
    print(f"\nSlide Title: {slide_title}\n")
    print(f"Visual Assignment Strategy: {visual_assignment_strategy}\n")
    print("📤 Storyboard Agent Full Response from LLM:\n")
    print(full_response_text)
    print("\n" + "=" * 100 + "\n")

    return storyboard_output, strategy_type


def format_storyboard_for_sheet(storyboard_xml, strategy_type="flexible", slide_content=""):
    """
    Format the storyboard XML into the readable format for the sheet.
    
    :param storyboard_xml: Storyboard XML from <output> tags
    :param strategy_type: Type of strategy - "entire_slide", "per_sentence", or "flexible"
    :param slide_content: The slide content
    :return: Formatted storyboard text or empty string if storyboard XML is empty
    """
    if not storyboard_xml or storyboard_xml.strip() == "":
        return ""
    
    # Handle "entire slide" format (different structure)
    if strategy_type == "entire_slide":
        # Extract storyboard_for_the_slide content
        storyboard_match = re.search(r'<storyboard_for_the_slide>(.*?)</storyboard_for_the_slide>', storyboard_xml, re.DOTALL | re.IGNORECASE)
        if storyboard_match:
            visual_idea = storyboard_match.group(1).strip()
            # Format with "When VO:" and "Visual Idea:" similar to other formats
            formatted_parts = []
            if slide_content:
                formatted_parts.append(f'When VO: "{slide_content}"')
                formatted_parts.append("")  # One line gap
            formatted_parts.append(f"Visual Idea: {visual_idea}")
            formatted_text = "\n".join(formatted_parts)
            return formatted_text
        else:
            return ""
    
    # Handle "flexible" and "per_sentence" formats (both use storyboard_steps)
    # Extract all storyboard steps
    storyboard_steps_pattern = r'<storyboard_step>(.*?)</storyboard_step>'
    storyboard_steps = re.findall(storyboard_steps_pattern, storyboard_xml, re.DOTALL | re.IGNORECASE)
    
    if not storyboard_steps:
        return ""
    
    # Build formatted output
    formatted_parts = []
    
    for step_idx, step_xml in enumerate(storyboard_steps):
        # Extract components from each storyboard step
        narration_match = re.search(r'<narration_part>(.*?)</narration_part>', step_xml, re.DOTALL | re.IGNORECASE)
        visual_idea_match = re.search(r'<visual_idea>(.*?)</visual_idea>', step_xml, re.DOTALL | re.IGNORECASE)
        transition_match = re.search(r'<transition_from_previous>(.*?)</transition_from_previous>', step_xml, re.DOTALL | re.IGNORECASE)
        
        # When VO: (with quotes)
        if narration_match:
            narration_part = narration_match.group(1).strip()
            formatted_parts.append(f'When VO: "{narration_part}"')
            formatted_parts.append("")  # One line gap
        
        # Visual Idea:
        if visual_idea_match:
            visual_idea = visual_idea_match.group(1).strip()
            formatted_parts.append(f"Visual Idea: {visual_idea}")
            formatted_parts.append("")  # One line gap
        
        
        # Add separator between steps (except after the last one)
        if step_idx < len(storyboard_steps) - 1:
            formatted_parts.append("-----------------------------------------------")
            formatted_parts.append("")  # One line gap
    
    # Join all parts with newlines
    formatted_text = "\n".join(formatted_parts)
    
    return formatted_text


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Storyboard Agent",
        "function_name": "process_storyboard_row",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def process_storyboard_row(index, row, course_name, llm="gemini_3_flash_thinking"):
    """
    Process a single row and return the storyboard output.

    :param index: The row index.
    :param row: The row data.
    :param course_name: The course name.
    :param llm: The language model to use.
    :return: Tuple of (index, storyboard_text) where storyboard_text is formatted for the sheet.
    """
    try:
        # Get row data
        topic_name = str(row.get("Topic", "")).strip()
        subtopic_name = str(row.get("Subtopic", "")).strip()
        slide_title = str(row.get("Slide Chunk Title", "")).strip()
        slide_content = str(row.get("Slide Chunk", "")).strip()
        
        # Get visual assignment strategy (default to "Flexible, let the agent decide" if not found)
        visual_assignment_strategy = str(row.get("Visual Assignment Strategy", "Flexible, let the agent decide")).strip()
        if not visual_assignment_strategy or visual_assignment_strategy == "nan":
            visual_assignment_strategy = "Flexible, let the agent decide"
        
        # Skip if slide_content is empty
        if not slide_content or slide_content == "nan":
            return index, ""
        
        # Generate storyboard
        storyboard_xml, strategy_type = generate_storyboard_for_slide(
            course_name=course_name,
            topic_name=topic_name,
            subtopic_name=subtopic_name,
            slide_title=slide_title,
            slide_content=slide_content,
            visual_assignment_strategy=visual_assignment_strategy,
            llm=llm
        )
        
        # Format for sheet
        formatted_storyboard = format_storyboard_for_sheet(storyboard_xml, strategy_type=strategy_type, slide_content=slide_content)
        
        return index, formatted_storyboard
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        return index, ""


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Storyboard Agent",
        "function_name": "run_storyboard_agent_for_all_rows",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def run_storyboard_agent_for_all_rows(sheet, llm="gemini_3_flash_thinking", max_workers=5):
    """
    Generate storyboard for all rows in the Slide Chunks sheet.

    :param sheet: The gspread sheet object.
    :param llm: The language model to use.
    :param max_workers: Number of parallel workers (default 5).
    :return: None
    """
    worksheet_name = "Slide Chunks"
    
    # Fetch course info
    _, course_info_df = get_sheet_data_and_df(sheet, "Course info")
    course_name = course_info_df.loc[0, "Course Name"]
    
    # Load the worksheet and DataFrame
    worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)

    # Ensure storyboard_planning column exists
    if "storyboard_planning" not in df.columns:
        df["storyboard_planning"] = ""

    # Prepare for parallel processing - only process rows that need processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks first
        for index, row in df.iterrows():
            slide_content = str(row.get("Slide Chunk", "")).strip()
            storyboard_planning = str(row.get("storyboard_planning", "")).strip()
            
            # Skip if slide_content is empty
            if not slide_content or slide_content == "nan":
                continue
            
            # Skip if storyboard_planning is already filled
            if storyboard_planning and storyboard_planning != "nan":
                continue
            
            # Submit task for processing
            future = executor.submit(process_storyboard_row, index, row, course_name, llm)
            futures_map[future] = index

        # If no rows to process, return early
        if not futures_map:
            print("All rows already processed or no valid slide content found.")
            return

        # Initialize progress tracker 
        total_tasks = len(futures_map)
        progress = SmartProgressBar(
            total_tasks=total_tasks,
            description="Generating storyboards",
            save_interval=5
        )

        # Collect results as they complete
        for future in as_completed(futures_map):
            index = futures_map[future]
            try:
                row_index, storyboard_text = future.result()
                
                # Update dataframe
                df.at[row_index, "storyboard_planning"] = storyboard_text
                
                # Update progress
                progress.update()
                
                # Save every 5 rows
                if progress.should_save():
                    print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
                    save_to_sheet(worksheet, df)
                    format_worksheet(worksheet)
            except Exception as e:
                print(f"Error getting result for row {index}: {e}")
                # Update dataframe with error marker so row is marked as processed
                df.at[index, "storyboard_planning"] = f"ERROR: {str(e)}"
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
        
        # Find rows that need processing (have Slide Chunk but empty storyboard_planning)
        invalid_rows = []
        for index, row in df.iterrows():
            slide_content = str(row.get("Slide Chunk", "")).strip()
            storyboard_planning = str(row.get("storyboard_planning", "")).strip()
            
            # Skip if Slide Chunk is empty
            if not slide_content or slide_content == "nan":
                continue
            
            # Check if storyboard_planning is empty or just whitespace/nan/error
            if not storyboard_planning or storyboard_planning == "nan" or storyboard_planning.strip() == "" or storyboard_planning.startswith("ERROR:"):
                invalid_rows.append((index, row))
        
        if not invalid_rows:
            break
        
        retry_count += 1
        print(f"\n⚠️ Found {len(invalid_rows)} rows with Slide Chunk but empty storyboard_planning. Retrying (attempt {retry_count}/{max_retries})...")
        
        # Clear storyboard_planning for invalid rows
        for index, row in invalid_rows:
            df.at[index, "storyboard_planning"] = ""
        
        # Save cleared state
        save_to_sheet(worksheet, df)
        format_worksheet(worksheet)
        
        # Retry processing invalid rows
        futures_map = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for index, row in invalid_rows:
                future = executor.submit(process_storyboard_row, index, row, course_name, llm)
                futures_map[future] = index
            
            # Collect results
            for future in as_completed(futures_map):
                index = futures_map[future]
                try:
                    row_index, storyboard_text = future.result()
                    df.at[row_index, "storyboard_planning"] = storyboard_text
                except Exception as e:
                    print(f"Error getting result for row {index} on retry: {e}")
                    df.at[index, "storyboard_planning"] = f"ERROR: {str(e)}"
        
        # Save after retry
        save_to_sheet(worksheet, df)
        format_worksheet(worksheet)
    
    if retry_count > 0:
        # Check final state
        worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)
        final_invalid = []
        for index, row in df.iterrows():
            slide_content = str(row.get("Slide Chunk", "")).strip()
            storyboard_planning = str(row.get("storyboard_planning", "")).strip()
            if (slide_content and slide_content != "nan" and 
                (not storyboard_planning or storyboard_planning == "nan" or storyboard_planning.strip() == "" or storyboard_planning.startswith("ERROR:"))):
                final_invalid.append(index)
        
        if final_invalid:
            print(f"⚠️ After {retry_count} retry attempt(s), {len(final_invalid)} rows still have empty storyboard_planning.")
        else:
            print(f"✅ All rows filled after {retry_count} retry attempt(s).")
    
    # Final save to sheet
    print('All slides processed. Saving final DataFrame to sheet.')
    save_to_sheet(worksheet, df)
    format_worksheet(worksheet)
    print("✅ Storyboard generation complete and saved to sheet.")


def delete_storyboard_planning(sheet):
    """
    Remove the 'storyboard_planning' column from the Slide Chunks worksheet.
    
    :param sheet: The gspread sheet object.
    :return: None
    """
    worksheet_name = "Slide Chunks"
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    if "storyboard_planning" in df.columns:
        df = df.drop(columns=["storyboard_planning"])
        clear_worksheet(ws)
        save_to_sheet(ws, df)
        print(f"🗑️ Deleted 'storyboard_planning' column from '{worksheet_name}' worksheet")
    else:
        print(f"ℹ️ 'storyboard_planning' column does not exist in '{worksheet_name}' worksheet")
