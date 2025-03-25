import pandas as pd
from modules.chain import Chain, extract_text_in_tags
from services.sheets_service import get_sheet_data_and_df, create_or_read_worksheet
from agents.generate_assessments.revise_assessment import revise_assessment, detect_question_type, get_question_format
from agents.generate_assessments.review_assessment import review_assessment
from agents.generate_assessments.generate_assessment_questions import run_generate_assessment_question
from tqdm import tqdm
from services.helper_functions import MultiChoiceQuestion, TrueFalseQuestion, MatchingQuestion
from agents.generate_assessments.checklist_sheet import get_review_checklist
import re
from gspread_formatting import CellFormat, TextFormat, set_column_width, set_row_height
from gspread_dataframe import set_with_dataframe
import gspread_formatting as gs
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.smart_progress_bar import SmartProgressBar


def review_and_revise_assessment_questions_with_agents(course_name, target_audience, topic_slides, assessment_question, checklist_criteria, max_turns=5, llm='gemini_flash'):
    """
    This function reviews and revises an assessment question based on a checklist criterion.
    :param assessment_question: The assessment question to review.
    :param slides: The slides for the course.
    :param checklist_criteria: The checklist criteria for review.
    :param llm: The language model to use.
    :return: The review of the assessment question.
    """
    
    question_type = detect_question_type(assessment_question)  # Function to determine question type
    question_format = get_question_format(question_type)

    for n in range(max_turns):
        # Review the assessment question
        reviewer_response = review_assessment(
            course_name=course_name,
            target_audience=target_audience,
            assessment_question=assessment_question,
            slides=topic_slides,
            checklist_criteria=checklist_criteria,
            llm=llm
        )

        # Collect feedback for the current iteration
        question_feedback_list = []
        
        
        # Loop through evaluations and extract feedback for failed criteria
        for evaluation in reviewer_response['evaluation']:
            evaluation_dict = extract_text_in_tags(
                tags=['item_name', 'analysis', 'verdict', 'feedback_summary', 'improvement_suggestions'],
                text=evaluation
            )
            # Check for failed criteria
            if 'pass' not in evaluation_dict['verdict'].lower():
                question_feedback_list.append(
                    f"Feedback summary: {evaluation_dict['feedback_summary']}\n"
                    f"Improvement suggestions: {evaluation_dict['improvement_suggestions']}"
                )

        # If length of question_feedback_list is more than 0 (which means feedback exists), then pass it to reviser agent
        if len(question_feedback_list) > 0:
            # Convert list into string
            feedback_text = "\n\n".join(question_feedback_list)

            # Pass the question and feedback to revise agent
            revised_question = revise_assessment(
                course_name=course_name,
                target_audience=target_audience,
                assessment_question=assessment_question,
                slides=topic_slides,
                reviewer_response={'text': feedback_text},
                question_format=question_format,
                llm=llm
            )

            # Update the assessment question with the revised version
            assessment_question = revised_question

        else:
            # Else, no feedback was given from the reviewer agent
            # break the loop
            break

    # Return the assessment question
    return assessment_question

def run_review_and_revise_all_questions(sheet, worksheet_name, course_name, target_audience, progress):  
    
    """
    This function reviews and revises all assessment questions based on the checklist criteria.
    :param sheet: The Google Sheet object.
    :param worksheet_name: The name of the worksheet.
    :param course_name: The name of the course.
    :param target_audience: The target audience for the course.
    :param llm: The language model to use.
    :return: None
    """
    # Initialize a dictionary to store revised assessment questions
    revised_questions_by_topic = {}
    
    _, slide_chunks_df = get_sheet_data_and_df(sheet, worksheet_name)
    
    unique_topics = pd.unique(slide_chunks_df['Topic'])
    
    questions_by_topic = run_generate_assessment_question(sheet, worksheet_name, course_name, target_audience, progress)
    
    _, review_checklist_df = get_review_checklist(sheet, 'Review Agent Checklist')
    
    # Group checklist by task column while preserving sorting
    grouped_checklist = review_checklist_df.groupby('Task', sort=False)
    
    

    
    futures_map = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        for topic in unique_topics:
            topic_slides_data = slide_chunks_df[slide_chunks_df['Topic'] == topic]
            topic_slides = "\n---\n".join(
                "Topic Name: " + topic_slides_data['Slide Title'] + "\n" + "Slide Content: " + topic_slides_data['Slide Content']
            )
            revised_questions_by_topic[topic] = []

            for assessment_question in questions_by_topic[topic]:
                for task, group in grouped_checklist:
                    checklist_criteria = group['Review Criteria'].tolist()
                    future = executor.submit(
                        review_and_revise_assessment_questions_with_agents,
                        course_name,
                        target_audience,
                        topic_slides,
                        assessment_question,
                        checklist_criteria,
                        max_turns=2,
                        llm='gemini_2_flash'
                    )
                    futures_map[future] = (topic, assessment_question)

        total_tasks = len(futures_map)

        for future in tqdm(as_completed(futures_map), total=total_tasks):
            topic, assessment_question = futures_map[future]
            revised_question = future.result()
            revised_questions_by_topic[topic].append(revised_question)
            progress.update()


            # Function to generate structured question
            def generate_structured_question(assessment_question, llm='gemini_2_flash'):
                """
                This function generates a structured question from the assessment question.
                :param assessment_question: The assessment question to structure.
                :param llm: The language model to use.
                :return: The structured assessment question.
                """
                get_structured_question_prompt = """Output the following assessment question in the proper format:
                <assessment_question>
                {assessment_question}
                </assessment_question>
                """

                # Initialize the chain
                get_structured_question_agent = Chain(llm=llm)

                # Set the appropriate Pydantic model based on question type
                if "Multiple Choice" in assessment_question:
                    get_structured_question_agent.set_structured_output(MultiChoiceQuestion)
                elif "True/False" in assessment_question:
                    get_structured_question_agent.set_structured_output(TrueFalseQuestion)
                elif "Matching" in assessment_question:
                    get_structured_question_agent.set_structured_output(MatchingQuestion)
                else:
                    raise ValueError("Unrecognized question type in the assessment question.")

                # Add the user message to the chain
                get_structured_question_agent.add_message(
                    role="user",
                    content=get_structured_question_prompt.format(assessment_question=assessment_question)
                )

                # Run the chain and get the response
                response = get_structured_question_agent.run()
                print(response)
                return response

            # Generate structured question
            try:
                response = generate_structured_question(assessment_question)

                # Append the structured question to the DataFrame
                assessment_df = pd.concat(
                    [assessment_df, pd.DataFrame(
                        [{
                            'topic': topic,
                            'question_type': response.question_type,
                            'text': response.question_text,
                            'option a': response.option_a if hasattr(response, "option_a") else "",
                            'option b': response.option_b if hasattr(response, "option_b") else "",
                            'option c': response.option_c if hasattr(response, "option_c") else "",
                            'option d': response.option_d if hasattr(response, "option_d") else "",
                            'correct answer': "\n".join(
                                f"{pair.subquestion} - {pair.answer}" for pair in response.options
                            ) if response.question_type == "matching" else response.correct_answer,
                            'correct feedback': response.correct_feedback,
                            'incorrect feedback': response.incorrect_feedback
                        }]
                    )],
                    ignore_index=True
                )

            except Exception as e:
                print(f"Error processing question for topic '{topic}': {e}")
                
    return assessment_df



def run_update_assessment_questions(sheet, worksheet_name, course_name, target_audience, llm="gemini_2_flash"):
    """
    This function generates and updates the assessment questions for a given course.
    :param sheet: The Google Sheet object.
    :param worksheet_name: The name of the worksheet.
    :param course_name: The name of the course.
    :param target_audience: The target audience for the course.
    :param llm: The language model to use.
    :return: None
    """
    
    final_assessement_sheet, _ = create_or_read_worksheet(sheet, 'Final Assessment')
    
    # Add a column named "#" before the Course Name column
    _, assessment_df = get_sheet_data_and_df(sheet, 'Final Assessment')
    
    # Define the total number of tasks and create a progress bar
    # Estimate tasks: review/revise, formatting tasks, sheet updates
    total_tasks = 2 * len(assessment_df)
    
    # Initialize the main progress bar
    progress = SmartProgressBar(total_tasks=total_tasks, description="Percent complete:", save_interval=1)
    print("Initializing assessment question generation and formatting...")
    
    #if assessment_df is populated, skip processing
    if not assessment_df.empty:
        print("Final Assessment Tab already has data. Skipping processing.")
        # Update the progress bar to completion since we're skipping
        for _ in range(total_tasks):
            progress.update()
        return assessment_df
    else:
        # Run the review and revise function - this has its own progress tracking
        print("Generating and reviewing assessment questions...")
        assessment_df = run_review_and_revise_all_questions(sheet, worksheet_name, course_name, target_audience, progress)
        
        
    
    _, course_info_df = get_sheet_data_and_df(sheet, 'Course info')
    
    if '#' not in assessment_df.columns:
        assessment_df.insert(0, '#', range(1, len(assessment_df) + 1))  # Add sequential numbers in the '#' column

    # Extract the Course Name value
    course_name_value = course_info_df['Course Name'].iloc[0]

    # Set the first row of the Course Name column to the course name value
    if not assessment_df.empty:
        assessment_df.at[0, 'Course Name'] = course_name_value
    
    progress.update()
    print(f"Basic DataFrame formatting completed ({progress.completed_count}/{progress.total_tasks}, {progress.percent_complete:.2f}%)")

    # Clear the worksheet to remove old data
    final_assessement_sheet.clear()
    
    # Reorder columns in assessment_df to match the required structure
    required_columns = [
        "#", "Course Name", "Topic", "Question type", "Question", "Option A",
        "Option B", "Option C", "Option D", "Correct Answer",
        "Correct feedback", "Incorrect feedback"
    ]

    # Ensure the DataFrame column names match exactly
    assessment_df.rename(columns={
        'topic': 'Topic',
        'question_type': 'Question type',
        'text': 'Question',
        'option a': 'Option A',
        'option b': 'Option B',
        'option c': 'Option C',
        'option d': 'Option D',
        'correct answer': 'Correct Answer',
        'correct feedback': 'Correct feedback',
        'incorrect feedback': 'Incorrect feedback'
    }, inplace=True)
    
    
    # Remove duplicate topic names by replacing them with empty strings
    assessment_df['Topic'] = assessment_df['Topic'].where(
        assessment_df['Topic'] != assessment_df['Topic'].shift()
    )
    

    # Clean True/False values in Option A and Option B columns
    for column in ['Option A', 'Option B']:
        assessment_df[column] = assessment_df[column].apply(
            lambda x: "True" if str(x).strip().lower() == "true" else
                    "False" if str(x).strip().lower() == "false" else x
        )
    
    
    # Cleanup Correct Answer field for matching question type
    if 'Question type' in assessment_df.columns and 'Correct Answer' in assessment_df.columns:
        assessment_df['Correct Answer'] = assessment_df.apply(
            lambda row: '\n'.join([line.rstrip('.;').strip() for line in row['Correct Answer'].split('\n')])
            if row['Question type'].strip().lower() == 'matching' else row['Correct Answer'],
            axis=1
        )
    
    
    # Cleanup Incorrect Feedback field for matching question type
    if 'Question type' in assessment_df.columns and 'Incorrect feedback' in assessment_df.columns:
        assessment_df['Incorrect feedback'] = assessment_df.apply(
            lambda row: '\n'.join(
                row['Incorrect feedback'].split('\n')[:-1] +
                [row['Incorrect feedback'].split('\n')[-1].rstrip(';') + '.']
            ) if row['Question type'].strip().lower() == 'matching' and row['Incorrect feedback'].strip().split('\n')[-1].endswith(';')
            else row['Incorrect feedback'],
            axis=1
        )
    
    
    # Set columns F and G (Option A and Option B for True/False) to Plain Text to prevent Google Sheets from interpreting as booleans
    plain_text_format = CellFormat(
        numberFormat={"type": "TEXT"}
    )
    gs.format_cell_range(final_assessement_sheet, 'F:G', plain_text_format)
    

    # Write DataFrame to Google Sheets with headers
    set_with_dataframe(final_assessement_sheet, assessment_df[required_columns], include_index=False, include_column_header=True)
    
    # Re-read all data from the sheet after setting text format
    data = final_assessement_sheet.get_all_values()

    # Remove trailing spaces and multiple spaces in a single pass
    cleaned_data = [
        [re.sub(r'\s{2,}', ' ', cell.strip()) if isinstance(cell, str) else cell for cell in row]
        for row in data
    ]
    

    # Update True/False values explicitly in columns F and G
    for row_idx in range(2, len(cleaned_data) + 1):
        # Directly set "True" for Column F (Option A)
        final_assessement_sheet.update_cell(row_idx, 6, "True")

        # Directly set "False" for Column G (Option B)
        final_assessement_sheet.update_cell(row_idx, 7, "False")

    # Write back the cleaned data
    final_assessement_sheet.update(cleaned_data)
    

    # Final Assessment Tab formatting

    # Set all Text size to 8
    text_size_format = CellFormat(
        textFormat=TextFormat(fontSize=8)
    )
    gs.format_cell_range(final_assessement_sheet, 'A:Z', text_size_format)

    # Clip all text
    clip_text_format = CellFormat(
        wrapStrategy='CLIP'
    )
    gs.format_cell_range(final_assessement_sheet, 'A:Z', clip_text_format)
    

    # Freeze the first row and set its height to 21
    gs.set_frozen(final_assessement_sheet, rows=1)
    set_row_height(final_assessement_sheet, '1', 21)

    # Apply bold formatting to the header row
    header_format = CellFormat(
        textFormat=TextFormat(bold=True)
    )
    gs.format_cell_range(final_assessement_sheet, 'A1:Z1', header_format)
    
    
    # Left align all text
    left_align_format = CellFormat(
        horizontalAlignment="LEFT"
    )
    gs.format_cell_range(final_assessement_sheet, 'A:Z', left_align_format)

    # Set row height for all rows except header to 30
    set_row_height(final_assessement_sheet, f'2:{len(assessment_df) + 1}', 30)
    
    

    # Set column width for "#" column to 32 and center align text
    center_align_format = CellFormat(
        horizontalAlignment="CENTER"
    )
    set_column_width(final_assessement_sheet, 'A', 32)
    gs.format_cell_range(final_assessement_sheet, 'A:A', center_align_format)

    # Set column width for all other columns to 100
    for col in range(2, len(required_columns) + 1):
        set_column_width(final_assessement_sheet, chr(64 + col), 100)
    
    print("Final Assessment Tab updated successfully!")
    
    return assessment_df     