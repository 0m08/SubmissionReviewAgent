import pandas as pd
from gspread_formatting import (
    format_cell_range,
    set_row_heights,
    Color,
    CellFormat,
    TextFormat
)
from gspread.utils import rowcol_to_a1
from utils.decorator_helpers import try_n_times

@try_n_times(n = 5, wait = 2, backoff = 'exponential')
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


@try_n_times(n = 5, wait = 2, backoff = 'exponential')
def format_worksheet(worksheet):
    """
    Apply the following formatting to the given `worksheet`:
    1) Freeze row 1
    2) Bold row 1
    3) Fill row 1 background color with a light gray
    4) All cells font size 8
    5) All cells vertical alignment = middle
    6) All cells text wrapping = clip
    7) Set rows 2+ to a fixed height of 40
    """
    
    # 1) Freeze row 1
    worksheet.freeze(rows=1, cols=0)

    # 2) Bold row 1 & 3) Light gray background for row 1
    header_format = CellFormat(
        backgroundColor=Color(0.95, 0.95, 0.95),   # Approx. #E0E0E0
        textFormat=TextFormat(bold=True)
    )
    format_cell_range(worksheet, '1', header_format)

    # 4) All cells font size 8, 5) vertical alignment = middle, 6) text wrapping = clip
    # Determine the entire worksheet range dynamically
    total_rows = worksheet.row_count
    total_cols = worksheet.col_count
    bottom_right_cell = rowcol_to_a1(total_rows, total_cols)  # e.g., "Z1000"
    full_range = f"A1:{bottom_right_cell}"

    all_cells_format = CellFormat(
        textFormat=TextFormat(fontSize=8),
        verticalAlignment='TOP',
        wrapStrategy='CLIP'
    )
    format_cell_range(worksheet, full_range, all_cells_format)

    # 7) Set rows 2 through total_rows to a fixed height of 40
    set_row_heights(worksheet, [(f'2:{total_rows}', 40)])

    print("Sheet formatting completed successfully!")


@try_n_times(n = 5, wait = 2, backoff = 'exponential')
def create_or_read_worksheet(sheet, worksheet_name, rows = 1000, cols = 20):
    """
    This function creates a new worksheet if not already present. If present, it reads the sheet

    :param sheet: The sheet object.
    :param worksheet_name: The worksheet name.
    :param rows: The number of rows to add in the sheet.
    :param cols: The number of columns to add in the sheet.
    :return: worksheet, df    
    """

    # Get sheet names
    sheet_names = [worksheet.title for worksheet in sheet.worksheets()]

    if worksheet_name in sheet_names:
        worksheet, df = get_sheet_data_and_df(sheet = sheet, sheet_name = worksheet_name)
        print(f"{worksheet_name} sheet already exists")
    else:
        worksheet = sheet.add_worksheet(worksheet_name, rows = rows, cols = cols)
        df = pd.DataFrame()
        print(f"{worksheet_name} sheet created")
    
    return worksheet, df


@try_n_times(n = 5, wait = 2, backoff = 'exponential')
def save_to_sheet(worksheet, df):
    """
    Save the datesframe to a sheet
    """
    df = df.astype(str)
    worksheet.update([df.columns.values.tolist()] + df.values.tolist())
    return


@try_n_times(n = 5, wait = 2, backoff = 'exponential')
def clear_worksheet(worksheet):
    """
    Clears the given worksheet.
    """
    worksheet.clear()


@try_n_times(n = 5, wait = 2, backoff = 'exponential')
def delete_worksheet(sheet, worksheet_name):
    """
    Delete the worksheet.
    """
    # Check if sheet name is present
    sheet_names = [worksheet.title for worksheet in sheet.worksheets()]

    if worksheet_name not in sheet_names:
        print(f"{worksheet_name} sheet does not exist")
        return

    sheet.del_worksheet(sheet.worksheet(worksheet_name))
    print(f"{worksheet_name} sheet deleted")

    return


@try_n_times(n = 5, wait = 2, backoff = 'exponential')
def get_worksheet_names(sheet):
    """
    Gets a list of names all the worksheets present in the sheet
    """
    sheet_names = [worksheet.title for worksheet in sheet.worksheets()]
    return sheet_names


@try_n_times(n = 5, wait = 2, backoff = 'exponential')
def filter_non_blank_column(worksheet, column_name, df, header_row=1):
    """
    Apply a filter to show only rows where a specific column is not blank.
    Uses a pandas DataFrame to determine the dimensions for the filter range.
    
    Args:
        worksheet: gspread worksheet object
        column_name: Name of the column to filter (e.g., "Email")
        df: pandas DataFrame containing the worksheet data
        header_row: Row number where headers are located (default: 1)
    
    Returns:
        None
    """
    # Find the column index from the column name using the worksheet headers
    headers = worksheet.row_values(header_row)
    if column_name not in headers:
        raise ValueError(f"Column '{column_name}' not found in worksheet headers")
    
    col_index = headers.index(column_name)
    
    # Determine the filter range dimensions based on the DataFrame
    num_rows = len(df) + header_row  # Add header_row to account for headers
    num_cols = len(df.columns)
    
    # Create the grid range manually
    sheet_id = worksheet.id  # Get the sheet ID
    
    # Grid ranges are zero-based in the API, but A1 notation is 1-based
    start_row_index = header_row - 1
    end_row_index = num_rows
    start_column_index = 0
    end_column_index = num_cols
    
    grid_range = {
        "sheetId": sheet_id,
        "startRowIndex": start_row_index,
        "endRowIndex": end_row_index,
        "startColumnIndex": start_column_index,
        "endColumnIndex": end_column_index
    }
    
    # Create the filter request
    request = {
        'setBasicFilter': {
            'filter': {
                'range': grid_range,
                'filterSpecs': [
                    {
                        'columnIndex': col_index,
                        'filterCriteria': {
                            'condition': {
                                'type': 'NOT_BLANK'
                            }
                        }
                    }
                ]
            }
        }
    }
    
    # Apply the filter
    worksheet.spreadsheet.batch_update({'requests': [request]})
    print(f"Filter applied successfully on column '{column_name}' for {len(df)} data rows")


@try_n_times(n = 5, wait = 2, backoff = 'exponential')
def hide_columns_by_name(worksheet, column_names, df):
    """
    Hide specific columns in a Google Sheet based on their names.
    
    Args:
        worksheet: gspread worksheet object
        column_names: List of column names to hide
        df: pandas DataFrame containing the worksheet data with column names
    
    Returns:
        None
    """
    # Find the indices of columns to hide
    column_indices = []
    for name in column_names:
        if name in df.columns:
            # Get the position of the column in the DataFrame
            col_index = df.columns.get_loc(name)
            column_indices.append(col_index)
        else:
            print(f"Warning: Column '{name}' not found in DataFrame")
    
    # Create requests for each column to hide
    requests = []
    for col_index in column_indices:
        requests.append({
            'updateDimensionProperties': {
                'range': {
                    'sheetId': worksheet.id,
                    'dimension': 'COLUMNS',
                    'startIndex': col_index,
                    'endIndex': col_index + 1
                },
                'properties': {
                    'hiddenByUser': True
                },
                'fields': 'hiddenByUser'
            }
        })
    
    if requests:
        try:
            worksheet.spreadsheet.batch_update({'requests': requests})
            print(f"Successfully hid {len(requests)} columns: {', '.join([column_names[column_indices.index(i)] for i in column_indices])}")
        except Exception as e:
            if "can't hide all the columns" in str(e).lower():
                print(f"Warning: Cannot hide requested columns as it would hide all visible columns. Skipping hide operation.")
            else:
                raise
    else:
        print("No columns were hidden")


@try_n_times(n = 5, wait = 2, backoff = 'exponential')
def resize_column_by_name(worksheet, column_name, pixel_size, wrap=None):
    """
    Resize a specific column in a Google Sheet to a given pixel width and optionally set wrapping.
    
    :param worksheet: gspread worksheet object
    :param column_name: Name of the column to resize
    :param pixel_size: Width in pixels to set for the column
    :param wrap: Text wrapping strategy ('OVERFLOW', 'CLIP', or 'WRAP')
    :return: None    
    """
    # Get all headers to find the column index
    headers = worksheet.row_values(1)
    
    if column_name not in headers:
        print(f"Warning: Column '{column_name}' not found in worksheet headers")
        return
    
    # Get the column index (0-based)
    col_index = headers.index(column_name)
    
    # Create the request to resize the column
    request = {
        'updateDimensionProperties': {
            'range': {
                'sheetId': worksheet.id,
                'dimension': 'COLUMNS',
                'startIndex': col_index,
                'endIndex': col_index + 1
            },
            'properties': {
                'pixelSize': pixel_size
            },
            'fields': 'pixelSize'
        }
    }
    
    # Apply the resize changes
    worksheet.spreadsheet.batch_update({'requests': [request]})
    print(f"Successfully resized column '{column_name}' to {pixel_size} pixels")
    
    # If wrap is specified, set the wrapping strategy
    if wrap is not None:
        if wrap not in ['OVERFLOW', 'CLIP', 'WRAP']:
            print(f"Warning: Invalid wrap value '{wrap}'. Must be 'OVERFLOW', 'CLIP', or 'WRAP'")
            return
            
        # Get total number of rows in the sheet to apply wrapping to the entire column
        total_rows = worksheet.row_count
        
        wrap_request = {
            'repeatCell': {
                'range': {
                    'sheetId': worksheet.id,
                    'startRowIndex': 0,
                    'endRowIndex': total_rows,
                    'startColumnIndex': col_index,
                    'endColumnIndex': col_index + 1
                },
                'cell': {
                    'userEnteredFormat': {
                        'wrapStrategy': wrap
                    }
                },
                'fields': 'userEnteredFormat.wrapStrategy'
            }
        }
        
        # Apply the wrap changes
        worksheet.spreadsheet.batch_update({'requests': [wrap_request]})
        print(f"Successfully set wrapping for column '{column_name}' to {wrap}")

    return

@try_n_times(n = 5, wait = 2, backoff = 'exponential')
def safe_get_sheet_data_and_df(sheet, sheet_name):
    """
    Safely get sheet data and dataframe. If the sheet does not exist, return empty DataFrame.
    
    :param sheet: gspread sheet object
    :param sheet_name: Name of the worksheet
    :return: (worksheet, DataFrame) or (None, empty DataFrame)
    """
    worksheet_names = get_worksheet_names(sheet)
    if sheet_name in worksheet_names:
        return get_sheet_data_and_df(sheet, sheet_name)
    else:
        print(f"'{sheet_name}' not found in the sheet. Skipping.")
        return None, pd.DataFrame()
    
@try_n_times(n=5, wait=2, backoff='exponential')
def hide_worksheet_by_name(sheet, worksheet_name):
    """
    Hides the specified worksheet (tab) in the Google Sheet by name.

    :param sheet: The gspread sheet object (entire Google Sheet).
    :param worksheet_name: The name of the worksheet/tab to hide.

    Returns:
        None
    """
    # Get all worksheet names
    worksheet_names = get_worksheet_names(sheet)

    # Validate worksheet exists
    if worksheet_name not in worksheet_names:
        print(f"⚠️ Worksheet '{worksheet_name}' not found. Nothing to hide.")
        return

    # Get worksheet object
    worksheet = sheet.worksheet(worksheet_name)

    # Issue the hide request
    request = {
        "updateSheetProperties": {
            "properties": {
                "sheetId": worksheet.id,
                "hidden": True
            },
            "fields": "hidden"
        }
    }

    # Apply the request
    sheet.batch_update({'requests': [request]})
    print(f" Worksheet '{worksheet_name}' has been successfully hidden.")

@try_n_times(n = 5, wait = 2, backoff = 'exponential')
def clear_all_filters(worksheet):
    """
    Removes all filters from the worksheet (clears any basic filter view).
    """
    request = {
        'clearBasicFilter': {
            'sheetId': worksheet.id
        }
    }
    worksheet.spreadsheet.batch_update({'requests': [request]})
    print(f"All filters cleared from worksheet '{worksheet.title}'")
