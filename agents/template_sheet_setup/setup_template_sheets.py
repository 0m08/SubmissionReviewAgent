import os
from services.drive_service import copy_sheet_from_link
from services.sheets_service import get_sheet_data_and_df, save_to_sheet

# Template links
TEMPLATE_COURSE_SHEET_LINK = os.getenv("TEMPLATE_COURSE_SHEET_LINK", "https://docs.google.com/spreadsheets/d/1LEzp9ta8PRDR0S5Fl-jlC8NJnc44hIjr-gpfSG4rKJA/edit?usp=sharing")
TEMPLATE_CHECKLIST_SHEET_LINK = os.getenv("TEMPLATE_CHECKLIST_SHEET_LINK", "https://docs.google.com/spreadsheets/d/1m4oK0iMgG9cagTyijRDrvU5l5Lh4Ka5FeaceTqdYI9c/edit?usp=drive_link")

def setup_template_sheets(drive, gc, drive_folder_id, course_name):
    """
    Setup template sheets for a course if they don't exist.
    Args:
        drive: Authenticated PyDrive GoogleDrive object
        gc: Authenticated gspread client
        drive_folder_id (str): Google Drive folder ID
        course_name (str): Name of the course
    Returns:
        dict: Contains sheet URLs and IDs
    """
    course_sheet_name = f"AI course: {course_name}"
    checklist_sheet_name = f"AI checklist: {course_name}"
    
    # Get existing files in the folder
    file_list = drive.ListFile({
        'q': f"'{drive_folder_id}' in parents and mimeType='application/vnd.google-apps.spreadsheet' and trashed=false"
    }).GetList()
    
    course_sheet_file = None
    checklist_sheet_file = None
    
    # Check if sheets already exist
    for file in file_list:
        if file['title'] == course_sheet_name:
            course_sheet_file = file
        elif file['title'] == checklist_sheet_name:
            checklist_sheet_file = file
    
    result = {}
    
    # Create course sheet if it doesn't exist
    if not course_sheet_file:
        course_sheet_id = copy_sheet_from_link(drive, TEMPLATE_COURSE_SHEET_LINK, course_sheet_name, drive_folder_id)
        result['course_sheet_created'] = True
        result['course_sheet_id'] = course_sheet_id
        result['course_sheet_url'] = f"https://docs.google.com/spreadsheets/d/{course_sheet_id}/edit"
    else:
        result['course_sheet_created'] = False
        result['course_sheet_id'] = course_sheet_file['id']
        result['course_sheet_url'] = f"https://docs.google.com/spreadsheets/d/{course_sheet_file['id']}/edit"
    
    # Create checklist sheet if it doesn't exist
    if not checklist_sheet_file:
        checklist_sheet_id = copy_sheet_from_link(drive, TEMPLATE_CHECKLIST_SHEET_LINK, checklist_sheet_name, drive_folder_id)
        result['checklist_sheet_created'] = True
        result['checklist_sheet_id'] = checklist_sheet_id
        result['checklist_sheet_url'] = f"https://docs.google.com/spreadsheets/d/{checklist_sheet_id}/edit"
    else:
        result['checklist_sheet_created'] = False
        result['checklist_sheet_id'] = checklist_sheet_file['id']
        result['checklist_sheet_url'] = f"https://docs.google.com/spreadsheets/d/{checklist_sheet_file['id']}/edit"
    
    # Update Course Info sheet with checklist link
    try:
        course_sheet = gc.open_by_key(result['course_sheet_id'])
        course_info_sheet, course_info_df = get_sheet_data_and_df(course_sheet, 'Course info')
        
        # Update the Checklist Link column
        course_info_df['Checklist Link'] = [result['checklist_sheet_url']]
        
        # Save the updated data back to the sheet
        course_info_worksheet = course_sheet.worksheet('Course info')
        save_to_sheet(course_info_worksheet, course_info_df)
        
        result['checklist_link_updated'] = True
    except Exception as e:
        result['checklist_link_updated'] = False
        raise Exception(f"Failed to update checklist link in Course Info sheet: {str(e)}")
    
    return result