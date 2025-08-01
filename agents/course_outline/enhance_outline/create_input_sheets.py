import streamlit as st
import re
import pandas as pd
from services.sheets_service import (
    create_or_read_worksheet,
    get_worksheet_names,
    get_sheet_data_and_df,
    save_to_sheet,
    format_worksheet,
    resize_column_by_name,
    delete_worksheet,
)


def pre_topic_deep_research(sheet, worksheet_name):
    """
    Pre-function to display a message asking users to create a "Topic Outline" sheet.
    
    This function shows a message to the user asking them to create a "Topic Outline" sheet
    and provides a checkbox option to use existing "Course Outline with LOs" sheet as reference.
    :param sheet: The Google Sheets object.
    :param worksheet_name: The name of the worksheet to create or validate.
    :return: None
    """
    # st.write("## Topic Outline Sheet Setup")
    
    # Check if "Topic Outline" sheet already exists
    sheet_names = get_worksheet_names(sheet)
    topic_outline_exists = worksheet_name in sheet_names
    course_outline_los_exists = "Course Outline with LOs" in sheet_names
    
    st.write("---")
    st.write("**Option 1:**")
    if not topic_outline_exists:
        st.warning("⚠️ Manually create a `Topic Outline` sheet before proceeding.")
        # st.write("This sheet will be used to store the detailed topic outlines generated in this step.")        
    else:
        st.success("✅ `Topic Outline` sheet already exists.")

    # Display a sample Topic Outline structure with dummy data
    st.write("Here's how the `Topic Outline` sheet should look:")
    sample_data = {
        "Topic": ["Topic 1 Name", "Topic 1 Name", "Topic 2 Name", "..."],
        "Learning Objective": [
            "Learning Objective 1 for Topic 1 here.",
            "Learning Objective 2 for Topic 1 here.",
            "Learning Objective 1 for Topic 2 here.",
            "..."
        ]
    }
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df, hide_index = True)

    st.write("---")
    st.write("**Option 2:**")
    # Checkbox for using existing Course Outline with LOs as reference
    use_existing_outline = False
    if course_outline_los_exists:
        use_existing_outline = st.checkbox(
            "Use existing Course Outline with LOs sheet as reference", 
            value=False,
            help="Check this box to use the existing Course Outline with LOs sheet as a reference to create `Topic Outline` sheet.",
            key="use_existing_outline"
        )
        
        if use_existing_outline:
            st.info("The existing Course Outline with LOs will be used as a reference for generating detailed topic outlines.")
    else:
        st.checkbox(
            "Use existing Course Outline with LOs sheet as reference", 
            value=False,
            help="Course Outline with LOs sheet not found. Complete previous steps first.",
            key="use_existing_outline_disabled",
            disabled=True
        )
        st.warning("⚠️ No 'Course Outline with LOs' sheet found. You may need to complete previous steps first.")
    
    return


def run_create_topic_outline_sheet(sheet, worksheet_name, use_existing_outline, skip_manual_step = False):
    """
    Creates or validates the Topic Outline sheet and then creates the Topic Deep Research sheet.
    
    If use_existing_outline is True, reads the "Course Outline with LOs" sheet,
    processes it to combine topic and subtopic, and creates a new dataframe with
    each learning objective in its own row.
    
    If use_existing_outline is False, validates that the Topic Outline sheet exists
    and contains the correct columns and data.
    
    After creating or validating the Topic Outline sheet, it creates the Topic Deep Research sheet
    with unique topics from the Topic Outline sheet.
    
    :param sheet: The Google Sheets object.
    :param worksheet_name: The name of the worksheet to create or validate.
    :param use_existing_outline: Boolean indicating whether to use the existing Course Outline with LOs.
    :param skip_manual_step: Boolean indicating whether to skip manual steps.
    :return: None
    :raises: Exception if the Topic Outline sheet doesn't exist or is improperly formatted.
    """
    # Set to true if skip manual step
    if skip_manual_step:
        use_existing_outline = True
    
    # First check if Topic Outline sheet exists and if it's populated
    topic_outline_worksheet, topic_outline_df = create_or_read_worksheet(sheet, worksheet_name)
    
    # Check if the sheet is already populated with valid data
    is_populated = (not topic_outline_df.empty and 
                   "Topic" in topic_outline_df.columns and 
                   "Learning Objective" in topic_outline_df.columns and
                   not (topic_outline_df["Topic"] == "").any() and 
                   not (topic_outline_df["Learning Objective"] == "").any())
    
    # If already populated, no need to do anything
    if is_populated:
        print("Topic Outline sheet already exists and is properly populated.")
    else:
        if use_existing_outline:
            # Get the Course Outline with LOs sheet
            try:
                course_outline_worksheet, course_outline_df = get_sheet_data_and_df(sheet, "Course Outline with LOs")
                
                if course_outline_df.empty:
                    raise Exception("The 'Course Outline with LOs' sheet exists but is empty. Please complete previous steps first.")
                
                # Check if the required columns exist
                required_columns = ["Topic", "Subtopic", "Learning Objectives"]
                missing_columns = [col for col in required_columns if col not in course_outline_df.columns]
                if missing_columns:
                    raise Exception(f"The 'Course Outline with LOs' sheet is missing the required column(s): {', '.join(missing_columns)}")
                
                # Process the dataframe to combine Topic and Subtopic and split Learning Objectives
                processed_data = []
                
                for _, row in course_outline_df.iterrows():
                    topic = row["Topic"]
                    subtopic = row["Subtopic"]
                    combined_topic = f"{topic} > {subtopic}"
                    
                    # Split learning objectives by newline
                    learning_objectives = row["Learning Objectives"].split("\n")
                    
                    # Remove any numbering (e.g., "1. ", "2. ") from learning objectives
                    learning_objectives = [lo.strip() for lo in learning_objectives if lo.strip()]
                    learning_objectives = [re.sub(r"^\d+\.\s*", "", lo) for lo in learning_objectives]
                    
                    # Create a row for each learning objective
                    for lo in learning_objectives:
                        processed_data.append({
                            "Topic": combined_topic,
                            "Learning Objective": lo
                        })
                
                # Create a new dataframe
                topic_outline_df = pd.DataFrame(processed_data)
                
                # Save to the Topic Outline sheet
                save_to_sheet(worksheet=topic_outline_worksheet, df=topic_outline_df)
                
                # Format the worksheet
                format_worksheet(worksheet=topic_outline_worksheet)
                
                # Resize the columns
                resize_column_by_name(topic_outline_worksheet, "Topic", 300)
                resize_column_by_name(topic_outline_worksheet, "Learning Objective", 500)
                
                print("Topic Outline sheet created successfully from Course Outline with LOs.")
                
            except Exception as e:
                raise Exception(f"Error processing 'Course Outline with LOs' sheet: {str(e)}")
        else:
            # Validate the existing Topic Outline sheet
            try:
                # Check if the dataframe is empty
                if topic_outline_df.empty:
                    raise Exception("The 'Topic Outline' sheet exists but is empty. Please populate it with the required data.")
                
                # Check if the required columns exist
                required_columns = ["Topic", "Learning Objective"]
                missing_columns = [col for col in required_columns if col not in topic_outline_df.columns]
                if missing_columns:
                    raise Exception(f"The 'Topic Outline' sheet is missing the required column(s): {', '.join(missing_columns)}")
                
                # Check for empty rows
                if (topic_outline_df["Topic"] == "").any():
                    raise Exception("The 'Topic Outline' sheet contains empty cells in the 'Topic' column. Please ensure all Topic cells are populated.")
                if (topic_outline_df["Learning Objective"] == "").any():
                    raise Exception("The 'Topic Outline' sheet contains empty cells in the 'Learning Objective' column. Please ensure all Learning Objective cells are populated.")
                
                print("Topic Outline sheet validated successfully.")
                
            except Exception as e:
                raise Exception(f"Error validating 'Topic Outline' sheet: {str(e)}")

    # After successfully creating or validating the Topic Outline sheet, create the Topic Deep Research sheet
    try:
        # Create the Topic Deep Research sheet
        create_topic_deep_research_sheet(sheet, topic_outline_df)
        print("Topic Deep Research sheet created successfully.")
    except Exception as e:
        print(f"Warning: Failed to create Topic Deep Research sheet: {str(e)}")
        raise Exception(f"Warning: Failed to create Topic Deep Research sheet: {str(e)}")

    return True



def create_topic_deep_research_sheet(sheet, topic_outline_df=None, worksheet_name="Topic Deep Research"):
    """
    Creates a Topic Deep Research Sheet with columns: topic_query, research, learning_objectives.
    
    This function extracts unique topics from the Topic Outline sheet, preserves their order,
    and creates a new sheet with these topics as the topic_query column. The other columns
    (research and learning_objectives) are left as empty strings.
    
    :param sheet: The Google Sheets object.
    :param topic_outline_df: The dataframe containing the Topic Outline data. If None, will read from sheet.
    :param worksheet_name: The name of the worksheet to create, defaults to "Topic Deep Research".
    :return: True if successful, raises Exception otherwise.
    :raises: Exception if the Topic Outline sheet doesn't exist or is improperly formatted.
    """
    try:
        # First check if Topic Deep Research sheet already exists and is populated
        sheet_names = get_worksheet_names(sheet)
        if worksheet_name in sheet_names:
            deep_research_worksheet, deep_research_df = get_sheet_data_and_df(sheet, worksheet_name)
            
            # Check if the sheet is already populated with valid data
            is_populated = (not deep_research_df.empty and 
                           "topic_query" in deep_research_df.columns and 
                           "research" in deep_research_df.columns and 
                           "learning_objectives" in deep_research_df.columns and
                           not (deep_research_df["topic_query"] == "").all())
            
            if is_populated:
                print(f"{worksheet_name} sheet already exists and is populated.")
                return True
        
        # If topic_outline_df is not provided, read it from the sheet
        if topic_outline_df is None:
            _, topic_outline_df = get_sheet_data_and_df(sheet, "Topic Outline")
            
            if topic_outline_df.empty:
                raise Exception("The 'Topic Outline' sheet exists but is empty. Please populate it with the required data.")
            
            # Check if the required columns exist
            if "Topic" not in topic_outline_df.columns:
                raise Exception("The 'Topic Outline' sheet is missing the required 'Topic' column.")
        
        # Extract unique topics while preserving order
        topics = []
        for topic in topic_outline_df["Topic"]:
            # Only add if not already in the list
            if topic not in topics and topic.strip() != "":
                topics.append(topic)
        
        if not topics:
            raise Exception("No valid topics found in the 'Topic Outline' sheet.")
        
        # Create data for the new sheet
        deep_research_data = {
            "topic_query": topics,
            "research": [""] * len(topics),
            "learning_objectives": [""] * len(topics)
        }
        
        # Create a new dataframe
        deep_research_df = pd.DataFrame(deep_research_data)
        
        # Create or get the Topic Deep Research sheet
        deep_research_worksheet, _ = create_or_read_worksheet(sheet, worksheet_name)
        
        # Save to the Topic Deep Research sheet
        save_to_sheet(worksheet=deep_research_worksheet, df=deep_research_df)
        
        # Format the worksheet
        format_worksheet(worksheet=deep_research_worksheet)
        
        # Resize the columns
        resize_column_by_name(deep_research_worksheet, "topic_query", 300)
        resize_column_by_name(deep_research_worksheet, "research", 500)
        resize_column_by_name(deep_research_worksheet, "learning_objectives", 500)
        
        print(f"{worksheet_name} sheet created successfully with {len(topics)} unique topics.")
        return True
        
    except Exception as e:
        raise Exception(f"Error creating '{worksheet_name}' sheet: {str(e)}")


def delete_create_topic_outline(sheet, topic_outline_ws="Topic Outline", topic_deep_research_ws="Topic Deep Research"):
    """Delete the Topic Outline and Topic Deep Research worksheets."""
    delete_worksheet(sheet, topic_outline_ws)
    delete_worksheet(sheet, topic_deep_research_ws)
