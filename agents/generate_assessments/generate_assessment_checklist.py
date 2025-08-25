import pandas as pd
from services.sheets_service import (
    get_sheet_data_and_df,
    clear_worksheet,
    save_to_sheet,
)
from agents.generate_assessments.chains import Chain
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.smart_progress_bar import SmartProgressBar
from langsmith import traceable
import streamlit as st



generate_checklist_prompt = """You are a Checklist Evaluation Agent tasked with rigorously evaluating the quality of assessment questions using a predefined checklist. The course name for which the assessment questions are based on is {course_name}, tailored to {target_audience}. Below is the slide content on which the assessment questions are based:

<slides>
{slides}
</slides>

These are the assessment questions:

<assessment_questions>
{assessment_questions}
</assessment_questions>

Refer to the checklist below for evaluating the assessment questions:

<checklist>
Task: {task_name}
Review Criteria:
{checklist_criteria}
</checklist>

Your task is to evaluate the quality of the assessment questions based on the checklist above, considering the Task and its associated Review Criteria. Strictly use the 'Task Name' and 'Review Criteria' exactly as they appear, without any rephrasing, grammar corrections, or paraphrasing.

For each review criterion, follow these steps:

1. Scratchpad: Think through and document your reasoning as to whether the review criterion is met. Reference the slide content, assessment questions, and the review criterion to form your reasoning. This section will be part of the final evaluation to provide transparency into your decision-making process.
2. Final Verdict: Based on your reasoning in the scratchpad, provide the verdict (Yes/No) for the review criterion.
3. If the verdict is "No," explain your reasons for giving the negative verdict in a concise manner under the "Why No" field.

Evaluation Guidelines:
- Base your evaluation of the assessment questions on the provided review criteria, ensuring alignment with the specified task and its associated review criteria. Reference the slide content, as needed, to verify aspects such as topic relevance and slide content.
- Do not alter or modify the phrasing of the task name or review criteria in any way.
- Ensure that the evaluation for every review criterion is objective and unbiased.
- Include a scratchpad field for each review criterion in the final output. This section should clearly document the thought process behind your verdict.
- Based on your evaluation in the scratchpad, provide a verdict of "Yes" or "No" inside the Verdict field for each review criterion, without adding explanations, interpretations, or additional commentary. If the review criterion is satisfied, your verdict will be "Yes" and if the review criterion is not satisfied, your verdict will be "No".
- If the verdict is "Yes," include only the fields for Task, Required Evidence, Scratchpad, and Verdict. If the verdict is "No," include an additional field, "Why no", to explain the negative verdict concisely

Reply in the following format:

Task: [Task Name]
Review Criteria: [Review Criterion Name]
Scratchpad: [Provide your reasoning here regarding whether the review criterion has been met.]
Verdict: [Yes/No]
Why no: [Provide a reason only if your verdict is "No". Omit this field entirely if your verdict is "Yes".]

[Repeat the above pattern for all review criterion under the task]
"""

def get_assessment_questions(sheet, worksheet_name):
    
    """
    This function generates the assessment checklist for a given course.
    :param sheet: The Google Sheet object.
    :param worksheet_name: The name of the worksheet.
    :param course_name: The name of the course.
    :param target_audience: The target audience for the course.
    :param llm: The language model to use.
    :return: None
    
    """
    
    _, slide_chunks_df = get_sheet_data_and_df(sheet, worksheet_name)
    
    _, assessment_df = get_sheet_data_and_df(sheet, 'Final Assessment')
    
    # Ensure topics are properly aligned in the dataframe
    assessment_df['Topic'].replace('', pd.NA, inplace=True)
    assessment_df['Topic'] = assessment_df['Topic'].ffill()

    # Extract unique topics from the Slide Chunks DataFrame
    unique_topics = pd.unique(slide_chunks_df['Topic'])

    # Prepare slides by combining all topics into one string
    slides = []
    for topic_name in unique_topics:
        topic_slides_data = slide_chunks_df[slide_chunks_df['Topic'] == topic_name]
        topic_slides = "\n".join(
            topic_slides_data['Title'] + ": " + topic_slides_data['Content']
        )
        slides.append(f"Topic: {topic_name}\n{topic_slides}")

    # Combine all topic slides into a single string
    slides = "\n\n".join(slides)

    # Prepare assessment questions using the updated DataFrame
    assessment_questions = []
    for topic_name in unique_topics:
        topic_questions = assessment_df[assessment_df['Topic'] == topic_name]

        # Format each question within the topic
        formatted_questions = []
        for _, row in topic_questions.iterrows():
            question_text = (
                f"Question type: {row['Question type']}\n"
                f"Question: {row['Question']}\n"
            )
            # Add options if applicable
            if row['Question type'] == 'multichoice':
                question_text += (
                    f"Option A: {row['Option A']}\n"
                    f"Option B: {row['Option B']}\n"
                    f"Option C: {row['Option C']}\n"
                    f"Option D: {row['Option D']}\n"
                )
            elif row['Question type'] == 'truefalse':
                question_text += (
                    f"Option A: {row['Option A']}\n"
                    f"Option B: {row['Option B']}\n"
                )
            elif row['Question type'] == 'select_all':
                question_text += (
                    f"Option A: {row['Option A']}\n"
                    f"Option B: {row['Option B']}\n"
                    f"Option C: {row['Option C']}\n"
                    f"Option D: {row['Option D']}\n"
                )
            elif row['Question type'] == 'matching':
                # Add matching pairs by splitting the correct answer into lines
                question_text += "Matching Pairs:\n"
                pairs = row['Correct Answer'].splitlines()  # Split by newline
                for pair in pairs:
                    question_text += f"  - {pair.strip()}\n"  # Ensure clean formatting

            # Add correct answer and feedback
            question_text += (
                f"Correct Answer: {row['Correct Answer']}\n"
                f"Correct Feedback: {row['Correct feedback']}\n"
                f"Incorrect Feedback: {row['Incorrect feedback']}\n"
            )
            formatted_questions.append(question_text)

        # Combine questions for the topic
        assessment_questions.append(f"Topic: {topic_name}\nQuestions:\n" + "\n\n".join(formatted_questions))

    # Combine all topics into a single string
    assessment_questions = "\n\n".join(assessment_questions)
    return assessment_questions




# Function to evaluate a checklist task
@traceable(
    metadata={
        "agent_name": "assessment",
        "step_name": "Update Assessment Checklist",
        "function_name": "generate_checklist",
        "user_id": st.session_state.get("role", "anonymous")
    }
)
def generate_checklist(course_name, target_audience, task_name, evidence_list, slides, assessment_questions, llm):
    """
    This function evaluates a single checklist task using an LLM.
    :param course_name: The name of the course.
    :param target_audience: The target audience for the course.
    :param task_name: The name of the task.
    :param evidence_list: The list of evidence required for the task.
    :param slides: The slide content for the course.
    :param assessment_questions: The assessment questions for the course.
    :param llm: The language model to use.
    :return: The evaluation response from the LLM.
    """
    # Format checklist criteria
    checklist_criteria = "\n".join([f"- {evidence}" for evidence in evidence_list])
    print(checklist_criteria)

    # Create task prompt
    task_prompt = generate_checklist_prompt.format(
        course_name=course_name,
        target_audience=target_audience,
        slides=slides,
        assessment_questions=assessment_questions,
        task_name=task_name,
        checklist_criteria=checklist_criteria
    )

    # Initialize LLM chain and run
    checklist_evaluation_agent = Chain(llm=llm)
    checklist_evaluation_agent.add_message(role='user', content=task_prompt)
    response = checklist_evaluation_agent.run()
    print(response)
    
    return response

@traceable(
    metadata={
        "agent_name": "assessment",
        "step_name": "Update Assessment Checklist",
        "function_name": "run_generate_assessment_checklist",
        "user_id": st.session_state.get("role", "anonymous")
    }
)    
def run_generate_assessment_checklist(sheet, worksheet_name, course_name, target_audience, llm="gemini_2_flash"):
    """
    This function generates the assessment checklist for a given course.
    :param sheet: The Google Sheet object.
    :param worksheet_name: The name of the worksheet.
    :param course_name: The name of the course.
    :param target_audience: The target audience for the course.
    :param llm: The language model to use.
    :return: None
    """
    
    _, slide_chunks_df = get_sheet_data_and_df(sheet, worksheet_name)
    unique_topics = pd.unique(slide_chunks_df['Topic'])

    #  Combine slides from ALL topics
    slides = "\n---\n".join(
        "Topic Name: " + row['Topic'] + "\n" +
        "Slide Title: " + row['Title'] + "\n" +
        "Slide Content: " + row['Content']
        for _, row in slide_chunks_df.iterrows()
    )

    assessment_questions = get_assessment_questions(sheet, worksheet_name)
        
    _, course_info_df = get_sheet_data_and_df(sheet, 'Course info')
    
    checklist_sheet_link = course_info_df['Checklist Link'][0]
    # gc = gspread.service_account(filename='content/service-credentials.json')
    gc = st.session_state["gc"]
    checklist_sheet = gc.open_by_url(checklist_sheet_link)
    checklist_sheet, checklist_df = get_sheet_data_and_df(checklist_sheet, 'Assessment Checklist')
    
    unique_tasks = checklist_df['Task'].unique()
    
    checklist_by_task = {}
    
    #if the llm  based output column is populated, skip processing
    if not checklist_df.empty and "LLM Based Output" in checklist_df.columns and checklist_df["LLM Based Output"].notnull().any():
        print("LLM Based Output column already has data. Skipping processing.")
    
    
    # Prepare for parallel processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        for task_name in unique_tasks:
            evidence_list = checklist_df[checklist_df['Task'] == task_name]['Review Criteria'].tolist()

            # Submit the task
            future = executor.submit(
                generate_checklist,
                course_name=course_name,
                target_audience=target_audience,
                task_name=task_name,
                evidence_list=evidence_list,
                slides=slides,
                assessment_questions=assessment_questions,
                llm=llm
            )

            # Map the Future to the task name
            futures_map[future] = task_name

        # Initialize the progress tracker
        total_tasks = len(futures_map)
        progress = SmartProgressBar(total_tasks=total_tasks, description="Percent complete:")

        # Collect the results as they complete
        for future in as_completed(futures_map):
            task_name = futures_map[future]
            response = future.result()

            # Store the response in the dictionary
            checklist_by_task[task_name] = response

            # Update progress
            progress.update()

    # Display checklist evaluations
    for task, evaluation in checklist_by_task.items():
        print(f"Task: {task}")
        print(f"Evaluation:\n{evaluation}\n")
    
    if "LLM Based Output" not in checklist_df.columns:
        checklist_df["LLM Based Output"] = ""
        
    for task_name, output in checklist_by_task.items():
        sections = output.split("\n\n")
        for section in sections:
            if "Review Criteria:" in section and "Verdict:" in section:
                required_evidence = section.split("Review Criteria:")[1].split("\n")[0].strip()
                verdict = section.split("Verdict:")[1].split("\n")[0].strip()
                mask = (checklist_df['Task'] == task_name) & (checklist_df['Review Criteria'] == required_evidence)
                checklist_df.loc[mask, "LLM Based Output"] = verdict
                print(f"Updated '{task_name}' task with '{required_evidence}' evidence: {verdict}")

    all_headers = checklist_df.columns.tolist()
    col_idx = all_headers.index("LLM Based Output") + 1

    col_letter = ""
    temp_idx = col_idx
    while temp_idx > 0:
        temp_idx, remainder = divmod(temp_idx - 1, 26)
        col_letter = chr(65 + remainder) + col_letter

    range_name = f"{col_letter}1:{col_letter}{len(checklist_df) + 1}"
    col_values = [["LLM Based Output"]] + checklist_df[["LLM Based Output"]].values.tolist()

    try:
        checklist_sheet.update(range_name=range_name, values=col_values)
        print("'LLM Based Output' column updated successfully in the Assessment Checklist Sheet!")
    except Exception as e:
        print(f"Error updating Google Sheet: {e}")

    return checklist_df


def delete_assessment_checklist(sheet, worksheet_name="Assessment Checklist"):
    """Remove the LLM Based Output column from the Assessment Checklist sheet."""
    _, course_info_df = get_sheet_data_and_df(sheet, "Course info")
    checklist_sheet_link = course_info_df["Checklist Link"][0]
    gc = st.session_state["gc"]
    checklist_sheet = gc.open_by_url(checklist_sheet_link)
    checklist_ws, checklist_df = get_sheet_data_and_df(checklist_sheet, worksheet_name)
    if "LLM Based Output" in checklist_df.columns:
        checklist_df = checklist_df.drop(columns=["LLM Based Output"])
        clear_worksheet(checklist_ws)
        save_to_sheet(checklist_ws, checklist_df)


