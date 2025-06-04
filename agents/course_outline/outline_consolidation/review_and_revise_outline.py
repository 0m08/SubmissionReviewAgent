import streamlit as st
import pandas as pd
from modules.chain import Chain
from services.sheets_service import get_sheet_data_and_df, save_to_sheet, create_or_read_worksheet, format_worksheet, delete_worksheet, clear_worksheet, get_worksheet_names
from services.helper_functions import get_outline_with_los
from services.smart_progress_bar import SmartProgressBar
import regex as re
from typing import List
from pydantic import BaseModel, Field
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.smart_progress_bar import SmartProgressBar
from services.helper_functions import compare_text_versions
import xml.etree.ElementTree as ET
from langsmith import traceable


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

User Comments (made during the course outline generation phase):
<user_comments>
{user_comments}
</user_comments>

Tentative Outline (used to generate the course outline):
<tentative_outline>
{tentative_outline}
</tentative_outline>

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
- Does the outline address all key points mentioned in the tentative outline?
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


extract_topic_subtopic_los_prompt = """Your task is to extract topic, subtopics, and learning objectives in a structured manner from the below text.

<text>
{text}
</text>

1. Extraction Rules:
    - Extract the topic as the Roman numeral line (e.g., I., II.).
    - Extract the subtopic as the capital letter line (e.g., A., B.).
    - Extract the learning objectives as the Arabic numeral lines (e.g., 1., 2.).

2. Extraction Requirements:
    - Ensure all topics, subtopics, and learning objectives are fully extracted without skipping or paraphrasing.
    - Preserve the original wording and maintain the hierarchy.

3. Learning objectives Phrasing:
    - Ensure the learning objectives are stated properly. In most cases, they will, thus don't paraphrase.
    - But in cases where the learning objectives in the text are listed as words / list of words without proper context, then combine them into proper standalone statements.

Edge Cases:
- If you find more nesting at the learning objectives level eg 1a, 1b, or 1.1, 1.2, etc, then extract each of them as a separate learning objective.
- Incase the outline doesn't follow the expected format, then use your best judgement to extract the topic, subtopics and learning objectives.

---

Output Format:

1. **Root Element: `<topic>`**  
   - This is the main container of the XML document. It represents a single topic.

2. **Child Element: `<topic_name>`**  
   - Inside `<topic>`, the first element must be `<topic_name>`.  
   - Use this element to specify the name or title of the topic.

3. **Repeated Child Element: `<subtopic>`**  
   - After `<topic_name>`, there can be one or more `<subtopic>` elements.  
   - Each `<subtopic>` element describes a single subtopic within the main topic.

4. **Subtopic Name: `<subtopic_name>`**  
   - Within each `<subtopic>`, the first element is `<subtopic_name>`.  
   - Use this element to name or label the subtopic clearly.

5. **Learning Objectives: `<learning_objectives>`**  
   - Within each `<subtopic>`, the next element is `<learning_objectives>`.  
   - This is a container for the learning objectives, which describe what the reader should learn or be able to do after studying the subtopic.

6. **Learning Objective Lines: `<objective>`**  
   - Inside `<learning_objectives>`, you can list multiple learning objectives. Each objective is written inside its own `<objective>` element.  
   - Each `<objective>` must be a standalone statement. Avoid paraphrasing unless necessary, and list them as separate items to keep them clear and concise.

7. **Example Structure**  
   ```xml
   <topic>
     <topic_name>Sample Topic</topic_name>
     <subtopic>
       <subtopic_name>Introduction</subtopic_name>
       <learning_objectives>
         <objective>Understand the basics of the topic</objective>
         <objective>Become familiar with common terminology</objective>
       </learning_objectives>
     </subtopic>
     <subtopic>
       <subtopic_name>Advanced Concepts</subtopic_name>
       <learning_objectives>
         <objective>Master the advanced techniques</objective>
         <objective>Apply concepts to real-world scenarios</objective>
       </learning_objectives>
     </subtopic>
   </topic>
   ```

Following these guidelines and the schema above ensures that your XML document is valid and conforms to the specified structure.
"""


def print_course_outline_before_review(sheet, worksheet_name = 'Outline Review'):
    """
    Prints the course outline. 

    :param sheet: The sheet object from which the course outline is retrieved.
    :param worksheet_name: The name of the worksheet from which the course outline is retrieved.
    :return: None
    """

    outline_review_sheet, outline_review_df = get_sheet_data_and_df(sheet, worksheet_name)

    # Retrieve the last row in the 'outline' column
    last_outline_entry = outline_review_df['Outline'].iloc[-1]

    st.write("---")
    st.write("##### Course Outline:")
    st.code(last_outline_entry)

    return


@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Review and Revise Outline",
    "function_name": "extract_outline_data",
    "user_id": st.session_state.get("role", "anonymous")
})
def extract_outline_data(response):
    """
    Extracts topic name, subtopics, and learning objectives from the response.
    Works with both object responses and text responses containing XML.
    
    Args:
        response: The response from the LLM containing topic, subtopics, and learning objectives.
               Can be either an object with attributes or a text string with XML.
        
    Returns:
        dict: A dictionary containing:
            - topic_name (str): The name of the topic
            - subtopics (list): A list of dictionaries, each containing:
                - subtopic_name (str): The name of the subtopic
                - learning_objectives (list): A list of learning objective strings
    """
    print(response)
    # Check if response is a string (text) or an object
    if isinstance(response, str):
        # Parse XML from text
        
        # Extract the XML content if it's embedded in other text
        xml_pattern = r'<topic>.*?</topic>'
        xml_match = re.search(xml_pattern, response, re.DOTALL)
        
        if xml_match:
            xml_content = xml_match.group(0)
        else:
            xml_content = response
            
        try:
            # Parse the XML
            root = ET.fromstring(xml_content)
            
            # Extract topic name
            topic_name = root.find('topic_name').text
            
            # Extract subtopics
            subtopics = []
            for subtopic_elem in root.findall('subtopic'):
                subtopic_name = subtopic_elem.find('subtopic_name').text
                learning_objectives = []
                
                for objective_elem in subtopic_elem.find('learning_objectives').findall('objective'):
                    learning_objectives.append(objective_elem.text)
                
                subtopics.append({
                    "subtopic_name": subtopic_name,
                    "learning_objectives": learning_objectives
                })
                
            return {
                "topic_name": topic_name,
                "subtopics": subtopics
            }
            
        except Exception as e:
            # If XML parsing fails, return empty data
            print(f"Error parsing XML: {e}")
            return {
                "topic_name": "Error parsing topic",
                "subtopics": []
            }
    else:
        # Process object response
        data = {
            "topic_name": response.topic_name,
            "subtopics": []
        }
        
        for subtopic in response.subtopics:
            subtopic_data = {
                "subtopic_name": subtopic.subtopic_name,
                "learning_objectives": subtopic.learning_objectives
            }
            data["subtopics"].append(subtopic_data)
        
        return data

@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Review and Revise Outline",
    "function_name": "parse_course_outline",
    "user_id": st.session_state.get("role", "anonymous")
})
def parse_course_outline(text, llm = "gemini_2_flash"):
    """
    Parses the text with help of LLM
    """

    parse_course_outline_agent = Chain(llm=llm, use_xml_checker = True)

    parse_course_outline_agent.add_message(
        role = "user",
        content = extract_topic_subtopic_los_prompt.format(
            text = text
        )
    )

    response = parse_course_outline_agent.run()
    
    # Extract structured data from response
    outline_data = extract_outline_data(response)

    # Convert extracted data into a DataFrame
    flattened_data = []
    for subtopic in outline_data["subtopics"]:
        flattened_data.append({
            "Topic": outline_data["topic_name"],
            "Subtopic": subtopic["subtopic_name"],
            "Learning Objectives": "\n".join([f"{i+1}. {obj}" for i, obj in enumerate(subtopic["learning_objectives"])])
        })
    
    return flattened_data

@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Review and Revise Outline",
    "function_name": "parse_course_outline_for_all_topics",
    "user_id": st.session_state.get("role", "anonymous")
})
def parse_course_outline_for_all_topics(sheet, worksheet_name, outline_review_df, llm = "gemini_2_flash"):
    """
    This function parses the final outline, and pastes in a newly created sheet - Course Outline with LOs
    """

    course_outline_sheet,  course_outline_df = create_or_read_worksheet(sheet, worksheet_name)

    # Skip logic if already present
    if course_outline_df.shape[0] > 1:
        print("Outline sheet with LOs - Already populated")
        return

    # Retrieve the last row in the 'outline' column
    last_outline_entry = outline_review_df['Outline'].iloc[-1]

    # Updated Regex to Extract Topics, Subtopics, and Learning Objectives
    topics = re.split(r"\n(?=[IVXLCDM]+\.)", last_outline_entry.strip())

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

@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Review and Revise Outline",
    "function_name": "run_review_and_revise_outline",
    "user_id": st.session_state.get("role", "anonymous")
})
def run_review_and_revise_outline(sheet, course_name, target_audience, llm='gemini_2_flash', skip_manual_step = False):
    """
    Runs the review and revision process for course outlines using AI.

    :param sheet: The Google Sheets object.
    :param course_name: Name of the course.
    :param target_audience: Target audience for the course.
    :param llm: The language model to use (default: 'gemini_2_flash').
    :param skip_manual_step: Bool. If True, it will assume the outline to be approved
    :return: None
    """

    outline_review_sheet, outline_review_df = get_sheet_data_and_df(sheet, 'Outline Review')

    # Check if already approved
    last_verdict = outline_review_df.iloc[-1]['Verdict'].strip()
    if (isinstance(last_verdict, str) and 'approved' in last_verdict.lower()) or skip_manual_step:
        print("Course Outline Approved")
        # Parse the outline and save it in new sheet
        parse_course_outline_for_all_topics(sheet = sheet, worksheet_name = "Course Outline with LOs", outline_review_df = outline_review_df, llm = llm)
        return True

    # Get the last row's manual feedback
    last_manual_feedback = outline_review_df.iloc[-1]['Manual Feedback'].strip()

    # Raise an exception if the user has not provided manual feedback
    if not last_manual_feedback:
        raise Exception("Error: 'Manual Feedback' column is empty in the last row. Please add your feedback before continuing.")

    # # Pause execution until the user confirms
    # if st.button("Press to continue"):
    #     st.write("-" * 100)

    # Initialize the progress tracker
    progress = SmartProgressBar(total_tasks = 2, description = "Percent complete")
    
    revise_outline_prompt = """Output the revised course outline within <revised_outline> tags."""


    _, rough_outline_df = get_sheet_data_and_df(sheet, 'Base Outline')
    
    course_outline = get_outline_with_los(df = rough_outline_df, include_learning_objectives = True, include_prefix = False)
    # Collect User Comments
    course_objective_guidelines = '\n'.join(st.session_state['course_objective_guidelines']).strip()
    
    # print(course_objective_guidelines)
    
    # _, client_reference_df = get_sheet_data_and_df(sheet, 'Client References')
    # client_reference_comments = '\n'.join(comment for comment in client_reference_df['consolidation_comments'].to_list() if comment)
    # client_comments = '\n'.join(comment for comment in client_reference_df['Client comments'].to_list() if comment)

    _, videos_research_df = get_sheet_data_and_df(sheet, 'Videos Research')
    videos_research_comments = '\n'.join(comment for comment in videos_research_df['consolidation_comments'].to_list() if comment)

    all_user_comments = f"""Comments made by user on course objective guidelines:
{course_objective_guidelines}

# ---
# 
# Comments made by client while sharing references:
# {client_comments}
# 
# ---
# 
# Comments made by user on client references:
# {client_reference_comments}

---

Comments made by user on videos research:
{videos_research_comments}
"""

    print(all_user_comments)
    
    # Create the list of messages
    messages = []
    for ind, row in outline_review_df.iterrows():
        # Get the values
        outline = row['Outline']
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
                        user_comments=all_user_comments,
                        tentative_outline=course_outline,
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

    # Add revision as a new row
    outline_review_df = pd.concat([outline_review_df, pd.DataFrame([{
        'Turn': int(outline_review_df.iloc[-1]['Turn']) + 1,
        'Outline': reviser_response['revised_outline'],
        'Verdict': '',
        'AI Suggestions': '',
        'Manual Feedback': ''
    }])])

    # Save to sheet
    save_to_sheet(worksheet = outline_review_sheet, df = outline_review_df)

    # Print the diff
    st.write("### Comparing the Two Most Recent Versions: ")
    compare_text_versions(
        outline_review_df.iloc[-2]['Outline'],
        outline_review_df.iloc[-1]['Outline']
    )

    raise Exception("You got this error since the outline is not Approved. If the outline looks good to you, enter Approved in the `Verdict` column last row. If the outline doesn't look good, you can enter Rejected in the `Verdict` column and enter your Feedback in the `Manual Feedback` column and run the agent again to generate a new outline.")
    # return outline_review_df


def delete_review_and_revise_outline(sheet):
    """
    Resets the outline sheet to the initial state and deletes the Course Outline with LO sheet.
    The reset state has the header row and one data row with only Turn and Outline columns populated.
    :param sheet: The Google Sheets object.
    :return: None
    """
    # Check if worksheet not already deleted possibly in previous step
    sheet_names = get_worksheet_names(sheet)

    if 'Outline Review' in sheet_names:
        # Get the outline review worksheet and dataframe
        outline_review_sheet, outline_review_df = get_sheet_data_and_df(sheet, 'Outline Review')
        
        # Create a new dataframe with one row, keeping only Turn and Outline values from first row
        first_row_data = {col: '' for col in outline_review_df.columns}  # Initialize all columns as empty
        first_row_data['Turn'] = outline_review_df['Turn'].iloc[0]  # Keep Turn from first row
        first_row_data['Outline'] = outline_review_df['Outline'].iloc[0]  # Keep Outline from first row
        
        new_df = pd.DataFrame([first_row_data])
        
        # Clear the worksheet
        clear_worksheet(worksheet = outline_review_sheet)

        # Save the reset dataframe back to the sheet
        save_to_sheet(worksheet = outline_review_sheet, df = new_df)
    
    # Delete the Course Outline with LOs sheet if present
    delete_worksheet(sheet = sheet, worksheet_name = 'Course Outline with LOs')
