import pandas as pd
from services.sheets_service import get_sheet_data_and_df

def create_or_read_worksheet(sheet, worksheet_name):
    """
    This function creates a new worksheet if not already present. If present, it reads the sheet

    :param sheet: The sheet object.
    :param worksheet_name: The worksheet name.
    :return: worksheet, df    
    """

    # Get sheet names
    sheet_names = [worksheet.title for worksheet in sheet.worksheets()]

    if worksheet_name in sheet_names:
        worksheet, df = get_sheet_data_and_df(sheet = sheet, sheet_name = worksheet_name)
        print(f"{worksheet_name} sheet already exists")
    else:
        worksheet = sheet.add_worksheet(worksheet_name, rows = 500, cols = 20)
        df = pd.DataFrame()
        print(f"{worksheet_name} sheet created")
    
    return worksheet, df

