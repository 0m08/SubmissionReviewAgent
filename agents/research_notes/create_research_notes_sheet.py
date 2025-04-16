from services.sheets_service import get_sheet_data_and_df, create_or_read_worksheet, format_worksheet, resize_column_by_name, save_to_sheet
import pandas as pd
from services.sheets_service import format_worksheet
from services.helper_functions import create_and_populate_columns


def create_research_notes_sheet(sheet, source_worksheet_name, target_worksheet_name):
    """
    This function creates a new sheet for the research notes.

    :param sheet: The sheet object.
    :param source_worksheet_name: The source worksheet name.
    :param target_worksheet_name: The target worksheet name.
    :return: None
    """

    research_notes_sheet, research_notes_df = create_or_read_worksheet(sheet, target_worksheet_name)
    # Check if the sheet is empty    
    if research_notes_df.empty:
        print('Target sheet is empty, proceeding to populate it.')
    else:
        print('Target sheet already contains data, skipping population.')
        return

    # Read the sheet and df
    course_outline_with_lo_sheet, course_outline_with_lo_df = get_sheet_data_and_df(sheet = sheet, sheet_name = source_worksheet_name)

    # Populate the research notes by topic groups
    for topic, topic_df in course_outline_with_lo_df.groupby('Topic', sort = False):
        subtopic = '\n'.join(topic_df['Subtopic'].unique()).strip()
        learning_objectives = '\n\n'.join(topic_df['Learning Objectives'].unique()).strip()
        research_notes = '\n\n---\n\n'.join(topic_df['revised_research_notes'].unique()).strip()

        research_notes_df = pd.concat(
            [
                research_notes_df,
                pd.DataFrame(
                    {
                        'Topic': [topic],
                        'Subtopic': [subtopic],
                        'Learning Objectives': [learning_objectives],
                        'research_notes_0': [""]
                    }
                )
            ],
            ignore_index = True
        )

        research_notes_df = create_and_populate_columns(
            df = research_notes_df,
            text = research_notes,
            specific_index = research_notes_df.shape[0] - 1, # last row
            col_base_name = "research_notes",
            chunk_size = 49000,
        )

    # Save to sheet
    save_to_sheet(worksheet = research_notes_sheet, df = research_notes_df)

    # Format the sheet
    format_worksheet(research_notes_sheet)

    resize_column_by_name(worksheet = research_notes_sheet, column_name = 'Topic', pixel_size = 200, wrap = 'WRAP')
    resize_column_by_name(worksheet = research_notes_sheet, column_name = 'Subtopic', pixel_size = 150)
    resize_column_by_name(worksheet = research_notes_sheet, column_name = 'Learning Objectives', pixel_size = 300)

    return

