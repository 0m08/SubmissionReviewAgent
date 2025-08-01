import streamlit as st

import gspread

# from agents.research_notes.vector_store import load_vector_db_with_pydrive
# from services.embedding_service import get_embedding_model
from services.drive_service import drive
from services.sheets_service import get_sheet_data_and_df
# from services.helper_functions import get_outline_with_los

from agents.research_notes.retriever_agent import run_retriever_agent_for_all_rows
from agents.research_notes.generate_notes import run_research_notes_agent_for_all_rows

from dotenv import load_dotenv

#sheet_link = input('Enter sheet link: ')
#course_drive_folder_id = input('Enter drive folder id: ')

#print(len(sheet_link))


def main():
    st.title("Course Pipeline Runner")

    # Initialize session states for each step
    if 'retriever_done' not in st.session_state:
        st.session_state['retriever_done'] = False
    if 'research_done' not in st.session_state:
        st.session_state['research_done'] = False

    # User inputs
    root_folder_id = st.text_input("Enter course Drive folder ID")
    sheet_link = st.text_input("Enter Google Sheet link")

    # Button to load data
    if st.button("Load Data"):
        load_dotenv()  # Load env variables from .env
        try:
            # gc = gspread.service_account(filename='content/service-credentials.json')
            gc = st.session_state["gc"]
            sheet = gc.open_by_url(sheet_link)
            course_info_sheet, course_info_df = get_sheet_data_and_df(sheet, 'Course info')

            st.session_state['root_folder_id'] = root_folder_id
            st.session_state['sheet'] = sheet
            st.session_state['course_name'] = course_info_df['Course Name'].values[0]
            st.session_state['target_audience'] = course_info_df['Target Audience & Industry'].values[0]

            st.success("Data loaded successfully!")
        except Exception as e:
            st.error(f"Error loading data: {e}")

    # Only proceed if data is loaded
    if 'sheet' in st.session_state:
        st.subheader("1. Run the Retriever")
        if not st.session_state['retriever_done']:
            if st.button("Run Retriever"):
                try:
                    run_retriever_agent_for_all_rows(
                        root_folder_id=st.session_state['root_folder_id'],
                        drive=drive,
                        sheet=st.session_state['sheet'],
                        worksheet_name='Course Outline with LOs',
                        course_name=st.session_state['course_name'],
                        target_audience=st.session_state['target_audience'],
                        llm='gemini_2_flash'
                    )
                    st.session_state['retriever_done'] = True
                    st.success("Retriever completed!")
                except Exception as e:
                    st.error(f"Error running retriever: {e}")
        else:
            st.write("Retriever: **Done**")

        st.subheader("2. Run the Researcher")
        # Show researcher button only if the retriever step is done
        if st.session_state['retriever_done']:
            if not st.session_state['research_done']:
                if st.button("Run Researcher"):
                    try:
                        run_research_notes_agent_for_all_rows(
                            sheet=st.session_state['sheet'],
                            worksheet_name='Course Outline with LOs',
                            course_name=st.session_state['course_name'],
                            target_audience=st.session_state['target_audience'],
                            llm='gemini_2_flash'
                        )
                        st.session_state['research_done'] = True
                        st.success("Researcher completed!")
                    except Exception as e:
                        st.error(f"Error running researcher: {e}")
            else:
                st.write("Researcher: **Done**")






if __name__ == "__main__":
    main()