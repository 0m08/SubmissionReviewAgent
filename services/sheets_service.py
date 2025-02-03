import gspread
import pandas as pd


def get_sheet_data_and_df(sheet, sheet_name):
    """
    Get sheet data and convert to dataframe.
    Args:
        sheet: The sheet object.
        sheet_name: The name of the sheet.
    Returns:
        worksheet: The worksheet object.
        df: The dataframe.
    """
    worksheet = sheet.worksheet(sheet_name)
    df = pd.DataFrame(worksheet.get_all_records())
    return worksheet, df