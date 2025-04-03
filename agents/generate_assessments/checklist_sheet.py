
from pydrive2.drive import GoogleDrive
import gspread
import pandas as pd
from services.sheets_service import get_sheet_data_and_df
from agents.generate_assessments.generate_assessment_questions import run_generate_assessment_question

# Connect to the Review Agent Checklist sheet
checklist_sheet_link = "https://docs.google.com/spreadsheets/d/1O8ADTCJcwfZJwXRQEb09aXzax1b4Ll2mzagEhIdCdS0/edit?usp=sharing"

def get_review_checklist(sheet, worksheet_name):
    """
    Connects to a Google Sheet and retrieves the specified worksheet as a Pandas DataFrame.
    
    :param sheet: The gspread Sheet object.
    :param worksheet_name: Name of the worksheet to retrieve.
    :return: Tuple (review_checklist_sheet, review_checklist_df)
    """

    # Authenticate and open the Google Sheet
    gc = gspread.service_account(filename="content/service-credentials.json")
    sheet = gc.open_by_url(checklist_sheet_link)
  
   # Connect to the Review Agent Checklist sheet
    checklist_sheet = gc.open_by_url(checklist_sheet_link)
    try:
        # Load the 'Review Agent Checklist' sheet
        review_checklist_sheet = checklist_sheet.worksheet(worksheet_name)
        review_checklist_df = pd.DataFrame(review_checklist_sheet.get_all_records())

        return review_checklist_sheet, review_checklist_df

    except gspread.exceptions.WorksheetNotFound:
        print(f"Error: '{worksheet_name}' sheet not found.")
        return None, None
    
# def checklist_evaluation(sheet, worksheet_name):
#     """
#     This function evaluates the generated graphics definition against the checklist criteria for all slides in the `Graphics Definition Checklist` sheet.
#     :param sheet: The Google Sheet object.
#     :param worksheet_name: The name of the worksheet.
#     :param course_name: The name of the course.
#     :param target_audience: The target audience for the course.
#     :param llm: The language model to use.
#     :return: None
#     """

#     _, slide_chunks_df = get_sheet_data_and_df(sheet, worksheet_name)
    
#     _, review_checklist_df = get_review_checklist(sheet, worksheet_name)
    
#     # Group checklist by task column while preserving sorting
#     grouped_checklist = review_checklist_df.groupby('Task', sort=False)
    
#     # Get unique topics from Slide Chunks
#     unique_topics = pd.unique(slide_chunks_df['Topic'])
    
#     questions_by_topic = run_generate_assessment_question(sheet, 'Slide Chunks', 'gemini_2_flash')


#     for task, group in grouped_checklist:
#         print(f"Task: {task}")
#         print(group['Review Criteria'].to_list())
        
#     grouped_checklist_df = pd.DataFrame(grouped_checklist)
#     assessment_question = questions_by_topic[unique_topics[0]][0]
#     print(assessment_question)
    
#     # Filter to get slides for a particular topic
#     topic_slides_data = slide_chunks_df[slide_chunks_df['Topic'] == unique_topics[0]]
    
#     slides = "\n---\n".join(
#     "Topic Name: " + topic_slides_data['Slide Title'] + "\n" + "Slide Content: " + topic_slides_data['Slide Content']
#     )
#     print(slides)
    
    