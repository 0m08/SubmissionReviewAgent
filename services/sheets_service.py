import pandas as pd
from gspread_formatting import (
    format_cell_range,
    set_row_heights,
    Color,
    CellFormat,
    TextFormat
)
from gspread.utils import rowcol_to_a1


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
        verticalAlignment='MIDDLE',
        wrapStrategy='CLIP'
    )
    format_cell_range(worksheet, full_range, all_cells_format)

    # 7) Set rows 2 through total_rows to a fixed height of 40
    set_row_heights(worksheet, [(f'2:{total_rows}', 40)])

    print("Sheet formatting completed successfully!")

