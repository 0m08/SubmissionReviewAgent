import streamlit as st
import pandas as pd
from modules.chain import Chain
from services.sheets_service import get_sheet_data_and_df, save_to_sheet, create_or_read_worksheet, format_worksheet, delete_worksheet, clear_worksheet, get_worksheet_names
from services.helper_functions import get_topic_outline, create_and_populate_columns
from services.smart_progress_bar import SmartProgressBar
import regex as re
from typing import List
from pydantic import BaseModel, Field
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.smart_progress_bar import SmartProgressBar
from services.helper_functions import compare_text_versions
import json


review_course_outline_prompt = """You are an experienced instructional designer tasked with reviewing and improving a course outline. Your goal is to provide a comprehensive analysis of the outline, identifying any issues and offering suggestions for improvement.

First, review the following information about the course:

Course Name:
<course_name>
{course_name}
</course_name>

Target Audience:
<target_audience>
{target_audience}
</target_audience>

Now, carefully study the full course outline:

<course_outline>
{course_outline}
</course_outline>

Here's the freshly added user feedback on the course outline:
<user_feedback>
{user_feedback}
</user_feedback>

Your task is to analyze this course outline thoroughly. Follow these steps:

1. Summarize your understanding of the course requirements based on the provided information.

2. For each major section of the course outline:
   a. Assess its relevance to the course objectives and target audience.
   b. Evaluate the logical flow and progression of topics.
   c. Check for any gaps in content or redundant information.
   d. Consider the depth and breadth of coverage for each topic.
   e. Provide a brief assessment of its strengths and weaknesses.
   f. If necessary, offer specific suggestions for improvement.

3. After analyzing all sections, provide an overall assessment of the course outline, including its structure, comprehensiveness, and alignment with the course objectives.

4. List key recommendations for improving the overall course outline.

As you review, consider these questions:
- Does the outline align with the course name and target audience?
- Are the sections and subsections well-organized and coherent?
- Is there a good balance between theoretical concepts and practical applications?
- Have the user comments been adequately addressed in the outline?

Refrain from giving generic suggestions such as:
- Add more practical examples and basic troubleshooting scenarios.
- Include more visual aids and simple demonstrations to help explain complex concepts
- Add hands-on activities or demonstrations where appropriate
- Include knowledge check points throughout the course
- Consider adding a glossary of basic terms
- Develop clear learning objectives for each main section

All of the above tasks are to be done while keeping the user feedback on the course outline in your mind.

Present your final review in the following format:

<course_outline_review>
<course_requirements>
[Present your understanding of the course objectives and requirements]
</course_requirements>

<overall_assessment>
[Provide a summary of your overall assessment of the course outline]
</overall_assessment>

<section_analysis>
<section_name>[Name of the section]</section_name>
<section_summary>[Summary of what's currently included in this section</section_summary>
<assessment>
[Your assessment of the section. It should include analysis of all the points mentioned i.e. relevance, logical flow, gaps / redundant info, coverage depth & breadth, strengths & weaknesses. It is okay for this section to be quite long.]
</assessment>
<suggestions>[Your suggestions for improvement, if any (optional)]</suggestions>
</section_analysis>

[Repeat the section_analysis for each major section]

<final_recommendations>
[Provide a list of key recommendations for improving the overall course outline. Be clear and verbose with the list to avoid any confusion / guesswork]
</final_recommendations>
<verdict>
["APPROVED" or "REJECTED".]
</verdict>
</course_outline_review>

Remember to be constructive in your feedback. Your goal is to help improve the course outline to better serve the target audience and meet the course objectives.

NOTE: DO NOT stop the analysis midway / skip through sections.
NOTE: You should output the entire analysis without worrying about output token limits.
NOTE: DO NOT ask questions such as "Due to the comprehensive nature of the review, I'll continue in subsequent responses. Would you like me to proceed with the next sections?".
NOTE: Instead continue to output without worrying about the output token limits.
"""


revise_outline_prompt = """Output the revised course outline within <revised_outline> tags.

Output the revised course outline in the below format:

<revised_outline>
[Topics numbered with roman numerals followed by learning objectives numbered with arabic numerals. See example below
I. Topic Name
1. Learning Objective 1
2. Learning Objective 2
...

II. Topic Name
1. Learning Objective 1
2. Learning Objective 2

...]
</revised_outline>

Don't include prefixes like Topic: or LO: or anything similar. Just output the topics and learning objectives with the corresponding roman and arabic numerals.
"""


extract_topic_subtopic_los_structured_prompt = """Your task is to extract topic, subtopics, and learning objectives in a structured manner from the below text.

<text>
{text}
</text>

1. Extraction Rules:
    - Extract the topic as the Roman numeral line (e.g., I., II.).
    - The text doesn't have subtopics. Instead use the learning objectives to infer the subtopics. Each learning objective should be mapped to a subtopic.
    - Extract the learning objectives as the Arabic numeral lines (e.g., 1., 2.).

2. Extraction Requirements:
    - Ensure all topics, subtopics, and learning objectives are fully extracted without skipping or paraphrasing.
    - Preserve the original wording and maintain the hierarchy.

3. Learning objectives Phrasing:
    - Ensure the learning objectives are stated properly. In most cases, they will, thus don't paraphrase.
    - But in cases where the learning objectives in the text are listed as words / list of words without proper context, then combine them into proper standalone statements.


Make sure to output in the proper format.
"""


def show_outline_diff(sheet):
    """
    Shows the diff between Topic Outline and Revised Outline sheets.
    
    :param sheet: The Google Sheets object
    """
    # Get data from both sheets
    _, topic_outline_df = get_sheet_data_and_df(sheet, "Topic Outline")
    _, revised_outline_df = get_sheet_data_and_df(sheet, "Revised Outline")
    
    # Convert outlines to text format
    topic_outline_text = get_topic_outline(topic_outline_df, use_text_labels=True)
    revised_outline_text = get_topic_outline(revised_outline_df, use_text_labels=True)
    
    # Compare the two outlines
    st.write("### Comparing the Original Outline with the Revised Outline: ")
    compare_text_versions(topic_outline_text, revised_outline_text, "Topic Outline", "Revised Outline")

    # Text to ask user to review the outline and either approve or reject it
    st.info("""
    **Review the outline and enter either "Approved" or "Rejected" within the `Verdict` column of the `Enhanced Outline Review` sheet.**
    - If approved, click the button to save progress and proceed to the next step.
    - If rejected, enter your feedback in the `Manual Feedback` column and click the button to get revised outline.
    """)

    return


def parse_course_outline(text, llm = "gemini_2_flash"):
    """
    Parses the text with help of LLM
    """

    class Subtopic(BaseModel):
        subtopic_name: str = Field(
            description="Name of the subtopic. This should be inferred from the learning objectives. This should be a phrase that summarizes the learning objectives."
        )
        learning_objectives: List[str] = Field(
            description="List of learning objectives for this subtopic. The list should have one learning objective only. Strip off any prefixes like '1.', '2.', 'LO', 'To', 'To be able to', etc."
        )

    class Topic(BaseModel):
        topic_name: str = Field(
            description="The main topic name. Strip off any prefixes like 'Topic', 'I.', 'II.', etc."
        )
        subtopics: List[Subtopic] = Field(
            description="All subtopics under this main topic. This should be inferred from the learning objectives. One subtopic per learning objective for this topic."
        )

    parse_course_outline_agent = Chain(llm=llm)

    parse_course_outline_agent.add_message(
        role = "user",
        content = extract_topic_subtopic_los_structured_prompt.format(
            text = text
        )
    )

    parse_course_outline_agent.structured_output = Topic

    response = parse_course_outline_agent.run()

    # Handle both Pydantic object and JSON string responses
    if isinstance(response, str):
        try:
            # Try to parse JSON string
            json_data = json.loads(response)
            # Convert JSON to a Topic object if needed
            if isinstance(json_data, dict):
                flattened_data = []
                topic_name = json_data.get("topic_name", "")
                for subtopic in json_data.get("subtopics", []):
                    flattened_data.append({
                        "Topic": topic_name,
                        "Subtopic": subtopic.get("subtopic_name", ""),
                        "Learning Objective": "\n".join([f"{i+1}. {obj}" for i, obj in enumerate(subtopic.get("learning_objectives", []))])
                    })
                return flattened_data
        except Exception as e:
            print(f"Error parsing JSON response: {e}")
            return []

    # Convert extracted data into a DataFrame (handles Pydantic object response)
    flattened_data = []
    for subtopic in response.subtopics:
        flattened_data.append({
            "Topic": response.topic_name,
            "Subtopic": subtopic.subtopic_name,
            "Learning Objective": "\n".join([f"{i+1}. {obj}" for i, obj in enumerate(subtopic.learning_objectives)])
        })
    
    return flattened_data


def parse_course_outline_for_all_topics(sheet, worksheet_name, outline_review_df, llm = "gemini_2_flash"):
    """
    This function parses the final outline, and pastes in a newly created sheet - Course Outline with LOs
    """

    course_outline_sheet,  course_outline_df = create_or_read_worksheet(sheet, worksheet_name)

    # Skip logic if already present
    if course_outline_df.shape[0] > 1:
        print("Outline sheet with LOs - Already populated")
        return

    # Retrieve outline from all outline_chunk columns
    outline_chunks = []
    outline_columns = [col for col in outline_review_df.columns if col.startswith('outline_chunk')]
    
    # Sort columns to ensure they're in the right order
    outline_columns = sorted(outline_columns)
    
    # Concatenate all chunks from the last row
    last_row_index = outline_review_df.index[-1]
    for col in outline_columns:
        chunk = outline_review_df.loc[last_row_index, col]
        if pd.notna(chunk) and chunk.strip():
            outline_chunks.append(chunk)
    
    # Combine all chunks into a single string
    last_outline_entry = "".join(outline_chunks)

    # Check if we're parsing the first row or a later row
    if last_row_index == 0:  # First row
        # Parse with Topic regex first
        topics = re.split(r"\n(?=Topic: )", last_outline_entry.strip())
        # If we only got one item, try the Roman numeral pattern instead
        if len(topics) <= 1:
            topics = re.split(r"\n(?=[IVXLCDM]+\.)", last_outline_entry.strip())
        elif not topics[0].startswith("Topic: "):
            # Handle case where the first topic might not have the prefix
            topics[0] = "Topic: " + topics[0]
    else:  # Later rows
        # Parse with Roman numeral regex first
        topics = re.split(r"\n(?=[IVXLCDM]+\.)", last_outline_entry.strip())
        # If we only got one item, try the Topic pattern instead
        if len(topics) <= 1:
            topics = re.split(r"\n(?=Topic: )", last_outline_entry.strip())
            if topics and not topics[0].startswith("Topic: "):
                # Handle case where the first topic might not have the prefix
                topics[0] = "Topic: " + topics[0]

    # Prepare for parallel processing
    futures_map = {}
    results = [None] * len(topics)  # Allocate a list for ordered results

    with ThreadPoolExecutor(max_workers = 5) as executor:
        # Submit tasks for each row
        for index, topic_content in enumerate(tqdm(topics)):
            # Submit the task
            future = executor.submit(
                parse_course_outline,
                topic_content,
                llm
            )

            # Map the Future to the index
            futures_map[future] = index

        # Collect the results as they complete
        total_tasks = len(futures_map)
        save_interval = 5  # how often to save (in number of completed tasks)

        # Initialize the progress tracker
        progress = SmartProgressBar(total_tasks = total_tasks, description = "Percent complete", save_interval = save_interval)

        # Now, pass only the futures (the keys) to as_completed:
        for future in tqdm(as_completed(futures_map), total=total_tasks):
            index = futures_map[future]
            response = future.result()
            # Collect and store results by index
            results[index] = response

            # Update progress
            progress.update()

    # Now, 'results' is a list of lists in the correct order
    final_list = []
    for item in results:
        final_list.extend(item)

    # Convert list to DataFrame
    course_outline_df = pd.DataFrame(final_list)

    save_to_sheet(worksheet = course_outline_sheet, df = course_outline_df)
    print("Outline saved to sheet")

    format_worksheet(worksheet = course_outline_sheet)

    return


def run_review_and_revise_topic_outline(sheet, course_name, target_audience, llm='gemini_2_flash', skip_manual_step = False):
    """
    Runs the review and revision process for course outlines using AI.

    :param sheet: The Google Sheets object.
    :param course_name: Name of the course.
    :param target_audience: Target audience for the course.
    :param llm: The language model to use (default: 'gemini_2_flash').
    :param skip_manual_step: Bool. If True, it will assume the outline to be approved
    :return: None
    """
    # Create st.session_state if not present for pre_exec_show_topic_outline_diff
    if 'pre_exec_show_topic_outline_diff' not in st.session_state:
        st.session_state['pre_exec_show_topic_outline_diff'] = False

    # Get the Enhanced Outline Review sheet and dataframe
    outline_review_sheet, outline_review_df = get_sheet_data_and_df(sheet, 'Enhanced Outline Review')

    # Check if already approved
    last_verdict = outline_review_df.iloc[-1]['Verdict'].strip()
    if 'approved' in last_verdict.lower() or skip_manual_step:
        print("Course Outline Approved")
        # Parse the outline and save it in new sheet
        parse_course_outline_for_all_topics(sheet = sheet, worksheet_name = "Enhanced Outline with LOs", outline_review_df = outline_review_df, llm = llm)
        return True

    # Get the last row's manual feedback
    last_manual_feedback = outline_review_df.iloc[-1]['Manual Feedback'].strip()

    # Raise an exception if the user has not provided manual feedback
    if not last_manual_feedback:
        raise Exception(
f"""## ❌ Missing Required Input

**Error**: The 'Manual Feedback' column is empty in the last row of the [Enhanced Outline Review sheet]({outline_review_sheet.url}).

### How to fix this:

1. **Option 1**: If you approve the outline
   - Enter `Approved` in the **Verdict** column

2. **Option 2**: If you want changes
   - Add your feedback in the **Manual Feedback** column

*Please complete one of these actions before continuing.*
"""
        )

    # Initialize the progress tracker
    progress = SmartProgressBar(total_tasks = 2, description = "Percent complete")
        
    # Create the list of messages
    messages = []
    for ind, row in outline_review_df.iterrows():
        # Get the values
        outline_chunks = []
        outline_columns = [col for col in outline_review_df.columns if col.startswith('outline_chunk')]
        outline_columns = sorted(outline_columns)
        
        for col in outline_columns:
            chunk = row[col] if pd.notna(row[col]) else ""
            outline_chunks.append(chunk)
        
        outline = "".join(outline_chunks)
        ai_suggestions = row['AI Suggestions']
        manual_feedback = row['Manual Feedback']

        # Add the initial prompt message
        if ind == 0:
            messages.append(
                (
                    "user",
                    review_course_outline_prompt.format(
                        course_name=course_name,
                        target_audience=target_audience,
                        course_outline=outline,
                        user_feedback=manual_feedback,
                    )
                )
            )
        else:  # AI revised outline for later rows
            messages.append(("ai", f"<revised_outline>\n{outline}\n</revised_outline>"))
            messages.append(
                (
                    "user",
                    f"Review the above outline in a similar fashion as done previously. Here is the latest user feedback on the above outline:\n{manual_feedback}.\nRemember to output within the <course_outline_review> tags in the same format."
                )
            )

        # Stop at the last row
        if ind + 1 == outline_review_df.shape[0]:
            break

        messages.append(("ai", f"<course_outline_review>\n{ai_suggestions}\n</course_outline_review>"))
        messages.append(("user", revise_outline_prompt))

    # Initiate the agent
    review_revise_agent = Chain(llm=llm)

    # Add messages to the agent
    review_revise_agent.add_messages(messages)

    # Run the agent to review
    review_revise_agent.tags = ['course_outline_review']
    reviewer_response = review_revise_agent.run()

    # Update progress
    progress.update()

    # Add AI review suggestions to df
    outline_review_df.loc[outline_review_df.shape[0] - 1, 'AI Suggestions'] = reviewer_response['course_outline_review']

    # Add revise message user prompt
    review_revise_agent.add_message(role="user", content=revise_outline_prompt)

    # Run the agent to revise
    review_revise_agent.tags = ['revised_outline']
    reviser_response = review_revise_agent.run()

    # Update progress
    progress.update()

    # Create a new row df with Turn and default empty values
    new_row_data = {
        'Turn': int(outline_review_df.iloc[-1]['Turn']) + 1,
        'Verdict': '',
        'AI Suggestions': '',
        'Manual Feedback': ''
    }
    
    # Create the new row DataFrame
    new_row_df = pd.DataFrame([new_row_data])
    
    # Use create_and_populate_columns to split the revised outline into multiple chunks
    new_row_df = create_and_populate_columns(
        df=new_row_df,
        text=reviser_response['revised_outline'],
        specific_index=0,
        col_base_name='outline_chunk',
        chunk_size=49000
    )

    # Make sure new_row_df doesn't have nan values
    new_row_df = new_row_df.fillna('')
    
    # Determine which columns should be in the result
    all_columns = outline_review_df.columns.tolist()
    
    # Make sure all necessary columns exist in new_row_df
    for col in all_columns:
        if col not in new_row_df.columns and not col.startswith('outline_chunk'):
            new_row_df[col] = ''
    
    # Add any new outline_chunk columns to all_columns if they don't exist
    for col in new_row_df.columns:
        if col.startswith('outline_chunk') and col not in all_columns:
            outline_review_df[col] = ''
            all_columns.append(col)
    
    # Concatenate the dataframes
    outline_review_df = pd.concat([outline_review_df, new_row_df], ignore_index=True)

    # Make sure outline_review_df doesn't have nan values
    outline_review_df = outline_review_df.fillna('')

    # Save to sheet
    save_to_sheet(worksheet = outline_review_sheet, df = outline_review_df)

    # Print the diff
    st.write("### Comparing the Two Most Recent Versions: ")
    
    # Get the full outline text for the two most recent versions
    previous_outline_chunks = []
    latest_outline_chunks = []
    
    for col in sorted([col for col in outline_review_df.columns if col.startswith('outline_chunk')]):
        if pd.notna(outline_review_df.iloc[-2][col]):
            previous_outline_chunks.append(outline_review_df.iloc[-2][col])
        if pd.notna(outline_review_df.iloc[-1][col]):
            latest_outline_chunks.append(outline_review_df.iloc[-1][col])
    
    previous_outline = "".join(previous_outline_chunks)
    latest_outline = "".join(latest_outline_chunks)
    
    compare_text_versions(
        previous_outline,
        latest_outline
    )

    raise Exception("""
## Outline Approval Required

**You received this error because the outline has not been approved yet.**

### What to do next:

- **If the outline looks good:** Enter `Approved` in the `Verdict` column of the last row.

- **If the outline needs changes:** Enter `Rejected` in the `Verdict` column, provide your feedback 
    in the `Manual Feedback` column, and run the agent again to generate a new outline.
""")
    # return outline_review_df


def delete_review_and_revise_outline(sheet):
    """
    Resets the outline sheet to the initial state and deletes the Course Outline with LO sheet.
    The reset state has the header row and one data row with only Turn and outline_chunk columns populated.
    :param sheet: The Google Sheets object.
    :return: None
    """
    # Check if worksheet not already deleted possibly in previous step
    sheet_names = get_worksheet_names(sheet)

    if 'Outline Review' in sheet_names:
        # Get the outline review worksheet and dataframe
        outline_review_sheet, outline_review_df = get_sheet_data_and_df(sheet, 'Outline Review')
        
        # Create a new dataframe with one row, keeping only Turn and outline_chunk values from first row
        first_row_data = {col: '' for col in outline_review_df.columns}  # Initialize all columns as empty
        first_row_data['Turn'] = outline_review_df['Turn'].iloc[0]  # Keep Turn from first row
        
        # Keep outline_chunk columns
        for col in outline_review_df.columns:
            if col.startswith('outline_chunk'):
                first_row_data[col] = outline_review_df[col].iloc[0]  # Keep outline chunks from first row
        
        new_df = pd.DataFrame([first_row_data])
        
        # Clear the worksheet
        clear_worksheet(worksheet = outline_review_sheet)

        # Save the reset dataframe back to the sheet
        save_to_sheet(worksheet = outline_review_sheet, df = new_df)
    
    # Delete the Course Outline with LOs sheet if present
    delete_worksheet(sheet = sheet, worksheet_name = 'Course Outline with LOs')

