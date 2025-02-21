from services.sheets_service import get_sheet_data_and_df
import pandas as pd


def create_research_notes_sheet(sheet, source_worksheet_name, target_worksheet_name):
    """
    This function creates a new sheet for the research notes.

    :param sheet: The sheet object.
    :param source_worksheet_name: The source worksheet name.
    :param target_worksheet_name: The target worksheet name.
    :return: None
    """

    # Check if target sheet already exists
    # Get sheet names
    sheet_names = [worksheet.title for worksheet in sheet.worksheets()]
    # Check if target sheet already exists
    if target_worksheet_name in sheet_names:
        print('Target sheet already exists')
        return


    # Read the sheet and df
    course_outline_with_lo_sheet, course_outline_with_lo_df = get_sheet_data_and_df(sheet = sheet, sheet_name = source_worksheet_name)

    # Construct target df - research notes df
    research_notes_df = pd.DataFrame(columns = ['Topic', 'Subtopic', 'Learning Objectives', 'research_notes'])

    # Populate the research notes by topic groups
    for topic, topic_df in course_outline_with_lo_df.groupby('Topic'):
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
                        'research_notes': [research_notes]
                    }
                )
            ]
        )

    # Create sheet
    research_notes_sheet = sheet.add_worksheet(title = target_worksheet_name, rows = max(100, research_notes_df.shape[0]), cols = max(10, research_notes_df.shape[1]))
    research_notes_sheet.update([research_notes_df.columns.values.tolist()] + research_notes_df.values.tolist())

    print(f'Created sheet {target_worksheet_name}')

    return

