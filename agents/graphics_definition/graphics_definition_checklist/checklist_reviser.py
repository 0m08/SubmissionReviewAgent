from services.sheets_service import (
    get_sheet_data_and_df,
    save_to_sheet,
    hide_columns_by_name,
    resize_column_by_name,
    format_worksheet,
    clear_worksheet,
)
from tqdm import tqdm
from agents.graphics_definition.graphics_definition_checklist.checklist_generation import generate_checklist_evaluation
import re
from modules.chain import Chain
import gspread
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.smart_progress_bar import SmartProgressBar
from langsmith import traceable
import streamlit as st


generate_revised_graphics_definition_from_checklist_evaluation_prompt = """You are a Graphics Definition Revision Agent. Your primary task is to generate a revised graphics definition by analyzing an existing graphic definition for the given slide and considering the feedback received from a checklist evaluation agent. Your overarching goal is to create a visual representation of the slide content that is clear, instructionally effective, and engaging, while carefully balancing simplicity, accuracy, and creative variation.

Below is the course information of the slide content for which you will be revising the graphics definition:
<course_information>
Course name: {course_name}
Target audience: {target_audience}
</course_information>

Here is the slide for which the graphics definition was generated:
<slide>
Slide title: {slide_title}
Slide content: {slide_content}
</slide>

Here is the graphics definition of the slide that needs revision:
<graphics_definition>
{graphics_definition}
</graphics_definition>

Here is the feedback received from the checklist evaluation agent:
<checklist_evaluation_feedback>
{feedback}
</checklist_evaluation_feedback>

Based on the feedback from the checklist evaluation agent, generate a revised graphics definition that addresses the identified issues.

Present your output strictly in the following format:

<output>

<evaluation_breakdown>
Summary: Summarize the issues identified from the feedback.

Approach to Revision: Explain your overall strategy for revising the graphics definition based on the feedback.
</evaluation_breakdown>

<checklist_revised_graphics_definition>
Provide a complete, revised graphics definition for this slide. Include all scenes from the original graphics definition, even if they were not changed. Ensure that your output strictly follows the same formatting and conventions as the original graphics definition. Your revised graphics definition must strictly follow the same scene-based format as the original, and each scene must be enclosed within "<scene>...</scene>" tags. If new scenes are added during revision (whether at the beginning or in between existing scenes), ensure that all scenes follow a sequential numbering format. The first scene must always be numbered Scene 1 (even if it's newly added), and all subsequent scenes must be numbered accordingly in sequence. If a new scene is inserted between existing scenes, adjust the numbering for all following scenes to maintain proper order. Do not use "Scene 0" or any non-sequential numbering.
Each scene should contain the following fields:
- Scene Number and Title
- Sentence(s) Depicted
- Purpose of the Scene
- Graphics Type
- Visual Elements
- Arrangement
- Presentation and Transitions
- Reusing Previous Graphics
</checklist_revised_graphics_definition>

</output>
"""

@traceable(
    metadata={
        "agent_name": "graphics_definition",
        "step_name": "Checklist Evaluation",
        "function_name": "generated_checklist_revised_graphics_definition",
        "user_id": st.session_state.get("role", "anonymous")
    }
)
def generated_checklist_revised_graphics_definition(course_name, target_audience, slide_title, slide_content, graphics_definition, feedback, llm="gemini_2_flash"):
    """
    Generate a revised graphics definition for a given slide based on checklist evaluation feedback.

    :param course_name: The name of the course.
    :param target_audience: The target audience for the course.
    :param slide_title: The title of the slide.
    :param slide_content: The content of the slide.
    :param graphics_definition: The original graphics definition for the slide.
    :param feedback: The collected feedback from failed checklist criteria.
    :param llm: The language model to use (default: "gemini_2_flash").
    :return: The revised graphics definition.
    """

    # Initialize the checklist reviser agent with the correct tag
    checklist_reviser_agent = Chain(llm=llm, tags=["checklist_revised_graphics_definition"])

    # Add the user message
    checklist_reviser_agent.add_message(
        role="user",
        content=generate_revised_graphics_definition_from_checklist_evaluation_prompt.format(
            course_name=course_name,
            target_audience=target_audience,
            slide_title=slide_title,
            slide_content=slide_content,
            graphics_definition=graphics_definition,
            feedback=feedback
        )
    )

    # Run the agent and fetch response
    response = checklist_reviser_agent.run()

    # Extract and return the revised graphics definition
    return response["checklist_revised_graphics_definition"]


@traceable(
    metadata={
        "agent_name": "graphics_definition",
        "step_name": "Checklist Evaluation",
        "function_name": "process_slide_checklist_evaluation",
        "user_id": st.session_state.get("role", "anonymous")
    }
)
def process_slide_checklist_evaluation(slide_index, row, unique_tasks, checklist_df, revised_graphics_definition, course_name, target_audience, llm):
    """
    Helper function to evaluate a single slide using the checklist criteria.
    """
    checklist_results = {}
    failed_feedback_by_slide = {}
    
    for task_name in unique_tasks:
        task_criteria_list = checklist_df[checklist_df['Task'] == task_name]['Checklist Criteria'].dropna().tolist()
        task_checklist_criteria = "\n".join([f"Criteria {i+1} - {criterion}" for i, criterion in enumerate(task_criteria_list)])

        checklist_evaluation = generate_checklist_evaluation(
            course_name=course_name,
            target_audience=target_audience,
            slide_title=row["Title"],
            slide_content=row["Content"],
            graphics_definition=revised_graphics_definition,
            task_name=task_name,
            checklist_criteria=task_checklist_criteria,
            llm=llm
        )

        if isinstance(checklist_evaluation, list):
            checklist_evaluation = "\n".join(checklist_evaluation)

        pattern = r"Final Verdict:\s*(Pass|Fail)(?:\s*Feedback:\s*(.*?))?(?:\n|$)"
        results = re.findall(pattern, checklist_evaluation, re.DOTALL)

        for (verdict, feedback) in results:
            if verdict == "Pass":
                checklist_results[task_name] = ("Pass", "")
            else:
                checklist_results[task_name] = ("Fail", feedback.strip())
                if slide_index not in failed_feedback_by_slide:
                    failed_feedback_by_slide[slide_index] = []
                failed_feedback_by_slide[slide_index].append(f"{task_name}: {feedback.strip()}")
    
    if slide_index in failed_feedback_by_slide:
        failed_feedback_text = "\n".join(failed_feedback_by_slide[slide_index])
        revised_graphics_definition = generated_checklist_revised_graphics_definition(
            course_name=course_name,
            target_audience=target_audience,
            slide_title=row["Title"],
            slide_content=row["Content"],
            graphics_definition=revised_graphics_definition,
            feedback=failed_feedback_text,
            llm=llm
        )
    
    return checklist_results, revised_graphics_definition or row["revised_graphics_definition"].strip()

@traceable(
    
    metadata={
        "agent_name": "graphics_definition",
        "step_name": "Checklist Evaluation",
        "function_name": "run_checklist_evaluation_for_all_slides",
        "user_id": st.session_state.get("role", "anonymous")
    }
)
def run_checklist_evaluation_for_all_slides(sheet, worksheet_name, course_name, target_audience, llm="gemini_2_flash"):
    """
    Iterates over each slide and its graphics definition in the main sheet, runs the checklist evaluation for each unique task,
    updates the checklist sheet with Pass/Fail results, and immediately revises the graphics definition if there are any failed criteria.
    """
    # Read the main sheet data
    slide_chunks_sheet, slide_chunks_df = get_sheet_data_and_df(sheet, worksheet_name)

    course_info_sheet, course_info_df = get_sheet_data_and_df(sheet, 'Course info')
    
    checklist_sheet_link = course_info_df['Checklist Link'][0]

    # gc = gspread.service_account(filename='content/service-credentials.json')
    gc = st.session_state["gc"]
    checklist_sheet = gc.open_by_url(checklist_sheet_link)
    
    checklist_sheet, checklist_df = get_sheet_data_and_df(checklist_sheet, 'Graphics Definition Checklist')

    unique_tasks = list(checklist_df['Task'].dropna().unique())
    slide_count = slide_chunks_df.shape[0]
    slide_columns = [f"Slide {i+1}" for i in range(slide_count)]
    
    for col in slide_columns:
        if col not in checklist_df.columns:
            checklist_df[col] = ""

    if "Graphics Definition" not in slide_chunks_df.columns:
        slide_chunks_df["Graphics Definition"] = ""

    futures_map = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        for index, row in slide_chunks_df.iterrows():
            # Check Slide Type for 'Video'
            slide_type = row.get('Slide Type', '').strip().lower()
            if slide_type == 'video':
                slide_chunks_df.at[index, 'Graphics Definition'] = 'No graphics since slide type is video'
                continue
            slide_index = index + 1
            revised_graphics_definition = row["revised_graphics_definition"].strip()
            
            future = executor.submit(
                process_slide_checklist_evaluation,
                slide_index, row, unique_tasks, checklist_df, revised_graphics_definition,
                course_name, target_audience, llm
            )
            futures_map[future] = index
    
        total_tasks = len(futures_map)
        save_interval = 5
        progress = SmartProgressBar(total_tasks=total_tasks, description="Percent complete", save_interval=save_interval)
        
        for future in tqdm(as_completed(futures_map), total=total_tasks):
            index = futures_map[future]
            checklist_results, revised_graphics_definition = future.result()
            
            for task_name, (verdict, feedback) in checklist_results.items():
                checklist_df.loc[checklist_df['Task'] == task_name, f"Slide {index+1}"] = verdict
            
            slide_chunks_df.at[index, "Graphics Definition"] = revised_graphics_definition
                
            progress.update()

            if progress.should_save():
                print(f'Saving partial progress after {progress.completed_count} tasks.')
                save_to_sheet(worksheet=checklist_sheet, df=checklist_df)
                save_to_sheet(worksheet=slide_chunks_sheet, df=slide_chunks_df)

    print('All slides processed. Saving final data.')
    save_to_sheet(worksheet=checklist_sheet, df=checklist_df)
    save_to_sheet(worksheet=slide_chunks_sheet, df=slide_chunks_df)

    # Resize the Graphics Definition column
    resize_column_by_name(slide_chunks_sheet, "Graphics Definition", 189, wrap="WRAP")

    # Format the Checklist worksheet
    format_worksheet(checklist_sheet)
    
    # Hide unnecessary columns
    column_names = [
        "Reference Description",
        "graphics_definition",
        "complexity_review",
        "missing_sentences_review",
        "accuracy_review",
        "reuse_previous_graphics_review",
        "human_review",
        "revised_graphics_definition"
    ]

    hide_columns_by_name(slide_chunks_sheet, column_names, slide_chunks_df)

    return True


def delete_graphics_definition_checklist(sheet, worksheet_name="Slide Chunks"):
    """Remove Graphics Definition column and checklist results."""
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    if "Graphics Definition" in df.columns:
        df = df.drop(columns=["Graphics Definition"])
        clear_worksheet(ws)
        save_to_sheet(ws, df)

    _, course_info_df = get_sheet_data_and_df(sheet, "Course info")
    checklist_sheet_link = course_info_df["Checklist Link"][0]
    gc = st.session_state["gc"]
    checklist_sheet = gc.open_by_url(checklist_sheet_link)
    checklist_ws, checklist_df = get_sheet_data_and_df(checklist_sheet, "Graphics Definition Checklist")
    slide_cols = [c for c in checklist_df.columns if re.match(r'^Slide \d+$', c)]
    if slide_cols:
        checklist_df = checklist_df.drop(columns=slide_cols)
        clear_worksheet(checklist_ws)
        save_to_sheet(checklist_ws, checklist_df)

