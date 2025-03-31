import pandas as pd
from services.sheets_service import get_sheet_data_and_df
from modules.chain import Chain
from agents.generate_assessments.checklist_sheet import get_review_checklist
from gspread_dataframe import set_with_dataframe
from gspread_formatting import CellFormat, TextFormat, format_cell_range
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.smart_progress_bar import SmartProgressBar



generate_review_checklist_prompt = """You are a Checklist Evaluation Agent tasked with rigorously assessing the quality of an assessment question based on a predefined checklist. The course name for which this assessment question is based on is {course_name}, tailored to {target_audience}. Below is the slide content on which the assessment question is based:

<slides>
{slides}
</slides>

This is the assessment question:

<assessment_question>
{assessment_question}
</assessment_question>

Refer to the checklist below to evaluate the assessment question:

<checklist>
Task: {task_name}
Review Criteria:
{checklist_criteria}
</checklist>

Your task is to evaluate the quality of the assessment question based on the checklist above, considering the Task and its associated Review Criteria. Strictly use the 'Task Name' and 'Review Criteria' exactly as they appear, without any rephrasing, grammar corrections, or paraphrasing.

For every review criterion, follow these steps:

1. Scratchpad: Think through and document your reasoning as to whether the review criterion is met. Reference the slide content, the assessment question, and the review criterion to form your reasoning. This section will be part of the final evaluation to provide transparency into your decision-making process.
2. Final Verdict: Based on your reasoning in the scratchpad, provide the verdict (Yes/No) for the review criterion.
3. If the verdict is "No", explain your reasons for giving the negative verdict in a concise manner under the "Why No" field.

Evaluation Guidelines:
- Base your evaluation of the assessment question on the provided review criteria, ensuring alignment with the specified task and its associated review criteria. Reference the slide content, as needed, to verify aspects such as topic relevance.
- Do not alter or modify the phrasing of the task name or review criteria in any way.
- Ensure that your evaluation for every review criterion is objective and unbiased.
- Include a scratchpad field for each piece of review criterion in the final output. This section should clearly document the thought process behind your verdict.
- Based on your evaluation in the scratchpad, provide a verdict of "Yes" or "No" inside the Verdict field for each piece of review criterion, without adding explanations, interpretations, or additional commentary. If the review criterion is satisfied, your verdict will be "Yes" and if the review criterion is not satisfied, your verdict will be "No".
- If the verdict is "Yes", include only the fields for Task, Required Evidence, Scratchpad, and Verdict. If the verdict is "No," include an additional field, "Why no", to explain the negative verdict concisely.

Reply in the following format:
Question Text: [Enter the Question Text]
Review Criteria: [Review Criterion Name]
Scratchpad: [Provide your reasoning here regarding whether the review criterion has been met]
Verdict: [Yes/No]
Why no: [Provide a reason only if your verdict is "No". Omit this field entirely if your verdict is "Yes"]

[Repeat the above pattern for all review criterion under the task]
"""




# Function to evaluate a checklist task for a single question
def generate_checklist_for_question(course_name, target_audience,  task_name, evidence_list, slides, question_text, llm):
    """
    This function evaluates a single question against a checklist task using an LLM.
    :param task_name: str - The name of the task
    :param evidence_list: list - A list of evidence criteria for the task
    :param slides: str - The slides for the topic
    :param question_text: str - The assessment question text
    :param llm: str - The language model to use
    :return: str - The response from the LLM
    """
    

    # Format the checklist criteria
    checklist_criteria = "\n".join([f"- {evidence}" for evidence in evidence_list])

    # Format the LLM prompt for the current task
    task_prompt = generate_review_checklist_prompt.format(
        course_name=course_name,
        target_audience=target_audience,
        slides=slides,
        assessment_question=question_text,
        task_name=task_name,
        checklist_criteria=checklist_criteria
    )

    # Initialize LLM chain and run
    checklist_evaluation_agent = Chain(llm=llm)
    checklist_evaluation_agent.add_message(role='user', content=task_prompt)
    response = checklist_evaluation_agent.run()
    print(response)

    return response

def update_review_checklist(sheet, worksheet_name, course_name, target_audience, llm):
    """
    This function updates the review checklist for all questions in the assessment.
    :param sheet: The Google Sheet object.
    :param worksheet_name: The name of the worksheet.
    :param course_name: The name of the course.
    :param target_audience: The target audience for the course.
    :param llm: The language model to use.
    :return: None
    """
    # Prepare a dictionary to store results
    review_checklist_by_task = {}
    _, checklist_df = get_review_checklist(sheet, worksheet_name)

    # Iterate through unique tasks and evaluate each question
    unique_tasks = checklist_df['Task'].unique()
    _, assessment_df = get_sheet_data_and_df(sheet, 'Final Assessment')
    
    _, slide_chunks_df = get_sheet_data_and_df(sheet, 'Slide Chunks')
    
    unique_topics = pd.unique(slide_chunks_df['Topic'])

    # Prepare slides by combining all topics into one string
    slides = []
    for topic_name in unique_topics:
        topic_slides_data = slide_chunks_df[slide_chunks_df['Topic'] == topic_name]
        topic_slides = "\n".join(
            topic_slides_data['Slide Title'] + ": " + topic_slides_data['Slide Content']
        )
        slides.append(f"Topic: {topic_name}\n{topic_slides}")

    # Combine all topic slides into a single string
    slides = "\n\n".join(slides)
    
    def evaluate_question(task_name, question_idx, question_row, evidence_list):
        question_text = (
            f"Question Type: {question_row['Question type']}\n"
            f"Question: {question_row['Question']}\n"
        )
        if question_row['Question type'] == 'multichoice':
            question_text += (
                f"Option A: {question_row['Option A']}\n"
                f"Option B: {question_row['Option B']}\n"
                f"Option C: {question_row['Option C']}\n"
                f"Option D: {question_row['Option D']}\n"
            )
        elif question_row['Question type'] == 'truefalse':
            question_text += (
                f"Option A: {question_row['Option A']}\n"
                f"Option B: {question_row['Option B']}\n"
            )
        elif question_row['Question type'] == 'matching':
            question_text += "Matching Pairs:\n"
            pairs = question_row['Correct Answer'].splitlines()
            for pair in pairs:
                question_text += f"  - {pair.strip()}\n"
        question_text += (
            f"Correct Answer: {question_row['Correct Answer']}\n"
            f"Correct Feedback: {question_row['Correct feedback']}\n"
            f"Incorrect Feedback: {question_row['Incorrect feedback']}\n"
        )

        # Evaluate the current question against the task
        response = generate_checklist_for_question(
            course_name=course_name,
            target_audience=target_audience,
            task_name=task_name,
            evidence_list=evidence_list,
            slides=slides,
            question_text=question_text,
            llm=llm
        )
        return f"Question {question_idx + 1}", response

    # Use ThreadPoolExecutor to parallelize the task
    with ThreadPoolExecutor() as executor:
        futures = []
        for task_name in unique_tasks:
            # Extract review criteria for the current task
            evidence_list = checklist_df[checklist_df['Task'] == task_name]['Review Criteria'].tolist()
            task_evaluation = {}

            for question_idx, question_row in assessment_df.iterrows():
                futures.append(executor.submit(evaluate_question, task_name, question_idx, question_row, evidence_list))

        # Collect results as they complete
        for future in as_completed(futures):
            question_id, response = future.result()
            task_evaluation[question_id] = response

        # Add the task evaluation to the main dictionary
        review_checklist_by_task[task_name] = task_evaluation

    # Display checklist evaluations
    for task, evaluation in review_checklist_by_task.items():
        print(f"Task: {task}")
        print("Evaluation:")
        for question, response in evaluation.items():
            print(f" {question}:")
            print(f"{response}\n")
            
    return review_checklist_by_task

def update_checklist_with_verdicts_preserve(checklist_df, review_checklist_by_task):
    """
    This function updates the review checklist DataFrame with the verdicts from the review checklist by task.
    :param checklist_df: The review checklist DataFrame.
    :param review_checklist_by_task: The review checklist by task.
    :return: The updated review checklist DataFrame.
    """
    # Extract tasks already in the checklist
    existing_tasks = checklist_df['Task'].tolist()

    # Identify questions to add as columns
    questions = list(next(iter(review_checklist_by_task.values())).keys())

    # Add missing question columns
    for question in questions:
        if question not in checklist_df.columns:
            checklist_df[question] = ''  # Initialize empty columns for new questions

    def update_task(task, questions_data):
        if task in existing_tasks:
            task_rows = checklist_df[checklist_df['Task'] == task]
            task_row_indices = task_rows.index.tolist()
            row_count = len(task_row_indices)

            for question, review_text in questions_data.items():
                if question in questions:
                    # Extract individual verdicts
                    verdicts = [line.split(":")[1].strip() for line in review_text.splitlines() if line.startswith("Verdict:")]

                    # Ensure rows match the number of verdicts
                    for i, verdict in enumerate(verdicts):
                        if i < row_count:  # Only populate existing rows
                            checklist_df.at[task_row_indices[i], question] = verdict
                        else:
                            break  # Stop if verdicts exceed available rows

    # Use ThreadPoolExecutor to parallelize the task
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(update_task, task, questions_data): task for task, questions_data in review_checklist_by_task.items()}

        # Ensure all tasks are completed
        for future in as_completed(futures):
            future.result()  # Ensure any exceptions are raised

    return checklist_df
    
        
    
def run_update_checklist_with_verdicts_preserve(sheet, worksheet_name, course_name, target_audience, llm):
    """
    This function updates the review checklist with the verdicts from the review checklist by task.
    :param sheet: The Google Sheet object.
    :param worksheet_name: The name of the worksheet.
    :return: None
    """
    # Initialize progress bar with 4 major steps
    progress = SmartProgressBar(total_tasks=4, description="Percent complete:", save_interval=5)
    
    # Load existing data into a DataFrame
    checklist_sheet, checklist_df = get_review_checklist(sheet, worksheet_name)
    print(checklist_df)
    progress.update()
    
    review_checklist_by_task = update_review_checklist(sheet, worksheet_name, course_name, target_audience, llm)
    print(review_checklist_by_task)
    progress.update()

    # Update the checklist with verdicts
    updated_checklist_df = update_checklist_with_verdicts_preserve(checklist_df, review_checklist_by_task)
    progress.update()

    # Write the verdict data back to the Google Sheet
    set_with_dataframe(checklist_sheet, updated_checklist_df, include_index=False, include_column_header=True)

    # Apply bold formatting to column headers
    bold_format = CellFormat(textFormat=TextFormat(bold=True))
    format_cell_range(checklist_sheet, '1:1', bold_format)
    progress.update()

    print("Review Agent Checklist updated successfully!")
    return updated_checklist_df