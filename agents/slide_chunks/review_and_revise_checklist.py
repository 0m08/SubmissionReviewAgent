from modules.chain import Chain
from agents.slide_chunks.format_inputs import strip_roman_numerals, strip_section_prefix
from services.sheets_service import get_sheet_data_and_df, format_worksheet, save_to_sheet
# from gspread_dataframe import set_with_dataframe
from tqdm import tqdm
import pandas as pd
import re
import gspread
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.smart_progress_bar import SmartProgressBar
import pandas as pd
import streamlit as st


generate_checklist_based_review_prompt = """We are creating structured slide content for an e-learning course. You are a Checklist Evaluation Agent tasked with rigorously assessing the quality of a slide based on a predefined checklist review criteria.

Below is the course information for which you will be doing the checklist evaluation:

<course_information>
Course name: {course_name}
Target audience: {target_audience}
</course_information>

Below is the topic and subtopic to which the slide belongs:

<topic>
Topic Name: {topic}
</topic>

<subtopic>
Subtopic Name: {subtopic}
</subtopic>

Here is the slide that needs to be evaluated:
<slide>
Slide Type: {slide_type}
Slide Title: {slide_title}
Slide Content: {slide_chunk}
</slide>

Refer to the checklist criteria below for evaluating this slide:

<checklist_criteria>
Task: {task_name}
Checklist Criteria:
{checklist_criteria}
</checklist_criteria>

For every checklist criterion, follow these steps:

1) Evaluation Breakdown:
  - Think through and document your reasoning as to whether the checklist criterion is met.
  - In this reasoning, reference the course name, target audience, topic, subtopic, slide title, and the slide content where relevant.

2) Final Verdict:
  - Based on your reasoning, provide the verdict (Pass/Fail) for the review criterion.

3) Feedback:
  - If the verdict is "Fail", explain your reasons for giving the negative verdict in a concise manner under the "Feedback" field along with suggestions of how to improve the slide based on this feedback in the same feedback field. Omit this "Feedback" field entirely if the verdict is "Pass".

Evaluation Guidelines:

  - Perform your evaluation of the slide on all the provided checklist criteria.
  - Do not alter or modify the phrasing of the checklist criteria in any way.
  - Ensure that your evaluation for every checklist criterion is objective, unbiased, and based strictly on the content of the slide in the context of the course details and target audience.
  - While suggesting improvements, avoid recommending additions that expand the slide beyond what is necessary to meet the checklist criterion. Suggestions should prioritize clarity, relevance, and instructional intent — not elaboration for its own sake.
  - When providing feedback and suggestions for a failed criterion, avoid suggesting significant content expansion. Keep in mind that the revised slide should ideally stay within approximately 10-20% of the original content length.
  - Avoid recommending generic or overly detailed explanations that can unnecessarily increase the length of the slide content
  - Include a "Evaluation Breakdown" field for every checklist criterion in the final output. This section should clearly document the thought process behind your verdict.
  - If the Final Verdict is "Pass", include only the fields for Checklist Criterion, Evaluation Breakdown, and Final Verdict in your output. If the Final Verdict is "Fail", include an additional field, "Feedback", to explain the negative verdict concisely.

Provide your output strictly in the following format:

<output>

Checklist Criterion: [Enter the Checklist Criteria text that is being evaluated as it is without any modification]
Evaluation Breakdown: [Provide your reasoning here regarding whether the checklist criterion has been met]
Final Verdict: [Enter your final verdict - Pass or Fail]
Feedback: [Provide your explanation and suggestions if your final verdict is "Fail". Omit this field entirely if your verdict is "Pass"]

[Repeat the above pattern for all the checklist criteria]

</output>
"""


def generate_checklist_evaluation(course_name, target_audience, topic, subtopic, slide_type, slide_title, slide_chunk, task_name, checklist_criteria, llm="gemini_2_flash"):
    """
    Generate checklist evaluation for a given slide chunk.

    :param course_name: The name of the course.
    :param target_audience: The target audience for the course.
    :param topic: The topic name to which the slide belongs.
    :param subtopic: The subtopic name to which the slide belongs.
    :param slide_title: The title of the slide.
    :param slide_chunk: The content of the slide chunk.
    :param task_name: The specific checklist task being evaluated.
    :param checklist_criteria: The checklist criteria used for evaluation.
    :param llm: The language model to use.
    :return: The generated checklist evaluation.
    """

    def task():
        # Initialize the checklist evaluation agent
        checklist_agent = Chain(llm=llm, tags=["output"])

        # Add the user message
        checklist_agent.add_message(
            role="user",
            content=generate_checklist_based_review_prompt.format(
                course_name=course_name,
                target_audience=target_audience,
                topic=topic,
                subtopic=subtopic,
                slide_type=slide_type,
                slide_title=slide_title,
                slide_chunk=slide_chunk,
                task_name=task_name,
                checklist_criteria=checklist_criteria
            )
        )

        # Run the agent
        response = checklist_agent.run()

        # Debug: Check the type and content of the response
        print(f"Debug: Type of response['output']: {type(response['output'])}")
        print(f"Debug: Content of response['output']: {response['output']}")

        # Join the list into a single string if response['output'] is a list
        if isinstance(response['output'], list):
            response['output'] = "\n".join(response['output'])

        return response["output"]

    with ThreadPoolExecutor() as executor:
        future = executor.submit(task)
        return future.result()
  
  

generated_checklist_revised_slide_chunks_prompt = """We are creating structured slide content for an e-learning course. You are a Slide Content Revisor Agent. Your task is to revise a slide by analyzing the original slide content and considering the feedback received from an evaluation agent for the review criteria that received a "Fail" verdict.

Below is the course information of the slide content for which you will be performing the revision:

<course_information>
Course name: {course_name}
Target audience: {target_audience}
</course_information>

Below is the topic and subtopic to which the slide belongs:

<topic>
Topic Name: {topic}
</topic>

<subtopic>
Subtopic Name: {subtopic}
</subtopic>

Here is the original slide that requires revision:

<original_slide>
Slide Type: {slide_type}
Slide title: {slide_title}
Slide content: {slide_chunk}
</original_slide>

Here is the feedback received from the evaluation agent:

<evaluation_feedback>
{feedback}
</evaluation_feedback>

Based on the feedback from the evaluation agent, generate a revised version of the slide content that effectively addresses the identified issues. 

When revising:
- Avoid significantly increasing the overall content length. The revised slide should ideally stay within approximately 10–20% of the original slide content's length.
- Avoid unnecessary elaboration or the addition of extra information that does not directly enhance or clarify the existing content, even if it is suggested in the evaluation feedback. Prioritize maintaining the original focus and instructional clarity.
- Be concise, eliminate redundancy, and preserve the instructional clarity and focus of the original slide.
- Prioritize brevity, clarity, and effectiveness over expansion.

Provide your output strictly in the following format:

<output>

<evaluation_breakdown>
Summary: Summarize the key issues mentioned in the evaluation feedback.

Approach to Revision: Explain your strategy for revising the slide content based on the feedback.
</evaluation_breakdown>

<revised_slide>
Slide Title: [If the feedback includes an issue with the slide title, provide the revised slide title. Otherwise, keep the original slide title exactly as it is.]
Slide Content: [Provide the revised slide content, ensuring that all identified issues have been addressed.]
</revised_slide>

</output>
"""


def generate_checklist_revised_slide_chunk(course_name, target_audience, topic, subtopic, slide_type, slide_title, slide_chunk, feedback, llm="gemini_2_flash"):
    """
    Generate a revised slide chunk based on checklist evaluation feedback.

    :param course_name: The name of the course.
    :param target_audience: The target audience for the course.
    :param topic: The topic name to which the slide belongs.
    :param subtopic: The subtopic name to which the slide belongs.
    :param slide_title: The title of the slide.
    :param slide_chunk: The original slide content that requires revision.
    :param feedback: The checklist evaluation feedback detailing necessary changes.
    :param llm: The language model to use (default: "gemini_2_flash").
    :return: The revised slide chunk with updated content.
    """

    def task():
        # Initialize the checklist revision agent
        checklist_reviser_agent = Chain(llm=llm, tags=["revised_slide"])

        # Add the user message
        checklist_reviser_agent.add_message(
            role="user",
            content=generated_checklist_revised_slide_chunks_prompt.format(
                course_name=course_name,
                target_audience=target_audience,
                topic=topic,
                subtopic=subtopic,
                slide_type=slide_type,
                slide_title=slide_title,
                slide_chunk=slide_chunk,
                feedback=feedback
            )
        )

        # Run the agent
        response = checklist_reviser_agent.run()

        # Extract the structured revised slide output
        return response["revised_slide"]

    with ThreadPoolExecutor() as executor:
        future = executor.submit(task)
        return future.result()
  
  # Function to ensure checklist sheet has correct slide columns

def ensure_checklist_columns(sheet,worksheet_name):
    """
    Ensures that the Checklist Sheet has 'Slide 1', 'Slide 2', ... columns in order,
    and inserts 'checklist_based_review_output' into Slide Chunks Sheet if missing.
    """

    slide_chunks_sheet, slide_chunks_df = get_sheet_data_and_df(sheet, worksheet_name)
    num_slides = len(slide_chunks_df)
    base_columns = ["Task", "Review Criteria"]
    slide_columns = [f"Slide {i + 1}" for i in range(num_slides)]
    required_columns = base_columns + slide_columns
    

    _, course_info_df = get_sheet_data_and_df(sheet, 'Course info')
    
    checklist_sheet_link = course_info_df['Checklist Link'][0]
    # gc = gspread.service_account(filename='content/service-credentials.json')
    gc = st.session_state["gc"]
    checklist_sheet = gc.open_by_url(checklist_sheet_link)

    checklist_sheet, checklist_df = get_sheet_data_and_df(checklist_sheet, 'Slide Chunks Checklist')

    # Add any missing columns
    for col in required_columns:
        if col not in checklist_df.columns:
            checklist_df[col] = ""

    # Ensure 'checklist_based_review_output' exists in Slide Chunks Sheet
    if "checklist_based_review_output" not in slide_chunks_df.columns:
        insert_at = list(slide_chunks_df.columns).index("checklist_based_slide_title")
        slide_chunks_df.insert(insert_at, "checklist_based_review_output", "")
    
    
    # Save updated sheets
    save_to_sheet(checklist_sheet, checklist_df)
    save_to_sheet(slide_chunks_sheet, slide_chunks_df)


# Helper to normalize text for accurate matching
def normalize(text):
    return " ".join(text.strip().lower().split())
def process_slide(index, row, course_name, target_audience, llm, checklist_df, slide_chunks_df):
    slide_index = index + 1
    topic = strip_roman_numerals(row["Topic"])
    subtopic = strip_section_prefix(row["Subtopic"])
    slide_type = row["Slide Type"]
    original_title = row["Slide Chunk Title"]
    base_chunk = row["learning_objectives_added_slide_chunk"]

    # Start with revised version if available
    current_title = row["checklist_based_slide_title"] or original_title
    current_chunk = row["checklist_based_slide_content"] or base_chunk

    print(f"\n🚀 Processing Slide: {slide_index} | Title: {current_title}")
    print("-" * 100)

    # Run up to 3 review-revise iterations
    for iteration in range(1, 4):
        print(f"\n🔁 Iteration {iteration} for Slide {slide_index}")
        all_failed_criteria = []
        all_pass = True

        task_names = checklist_df["Task"].unique()
        task_futures = {}

        with ThreadPoolExecutor(max_workers=5) as task_executor:
            for task_name in task_names:
                task_criteria = checklist_df.loc[checklist_df["Task"] == task_name, "Review Criteria"].dropna().tolist()
                formatted_criteria = "\n- " + "\n- ".join(task_criteria) if task_criteria else "No review criteria available."

                future = task_executor.submit(
                    generate_checklist_evaluation,
                    course_name = course_name,
                    target_audience = target_audience,
                    topic = topic,
                    subtopic = subtopic,
                    slide_type = slide_type,
                    slide_title = current_title,
                    slide_chunk = current_chunk,
                    task_name = task_name,
                    checklist_criteria = formatted_criteria,
                    llm = llm
                )
                task_futures[future] = task_name

        for future in as_completed(task_futures):
            task_name = task_futures[future]
            review_output = future.result()

            criteria_pattern = re.compile(
                r"Checklist Criterion:\s*(.*?)\s*"
                r"Evaluation Breakdown:\s*(.*?)\s*"
                r"Final Verdict:\s*(Pass|Fail)"
                r"(?:\s*Feedback:\s*(.*?))?(?:\n|$)",
                re.DOTALL
            )
            matches = criteria_pattern.findall(review_output)

            for match in matches:
                criterion, breakdown, verdict, feedback = match
                criterion = criterion.strip()
                verdict = verdict.strip()
                feedback = feedback.strip() if feedback else ""

                print(f"📝 Extracted - Criterion: {criterion}, Verdict: {verdict}, Feedback: {feedback}")

                task_mask = checklist_df["Task"] == task_name
                matched = False

                for row_idx in checklist_df[task_mask].index:
                    sheet_criterion = checklist_df.at[row_idx, "Review Criteria"]
                    if normalize(sheet_criterion) == normalize(criterion):
                        if verdict.lower() == "fail":
                            checklist_df.at[row_idx, f"Slide {slide_index}"] = f"Fail Feedback: {feedback}"
                            all_failed_criteria.append(
                                f"<criterion>\nChecklist Criterion: {criterion}\nFeedback: {feedback}\n</criterion>"
                            )
                            all_pass = False
                        else:
                            checklist_df.at[row_idx, f"Slide {slide_index}"] = "Pass"
                        matched = True
                        break

                if not matched:
                    print(f"❌ Could not match criterion to checklist sheet: {criterion}")

        # Always update current title/content at the end of each iteration
        slide_chunks_df.at[index, "checklist_based_slide_title"] = current_title
        slide_chunks_df.at[index, "checklist_based_slide_content"] = current_chunk

        if all_pass:
            slide_chunks_df.at[index, "checklist_based_review_output"] = "Pass"
            print(f"✅ All criteria passed. Review complete for Slide {slide_index}.")
            break
        else:
            combined_feedback = "\n\n".join(all_failed_criteria)
            slide_chunks_df.at[index, "checklist_based_review_output"] = combined_feedback

            print(f"⚠️ Revising Slide {slide_index} based on collected feedback")
            revised_slide = generate_checklist_revised_slide_chunk(
                course_name=course_name,
                target_audience=target_audience,
                topic=topic,
                subtopic=subtopic,
                slide_type=slide_type,
                slide_title=current_title,
                slide_chunk=current_chunk,
                feedback=slide_chunks_df.at[index, "checklist_based_review_output"],
                llm=llm
            )

            current_title = revised_slide.split("Slide Title: ")[1].split("\n")[0].strip()
            current_chunk = revised_slide.split("Slide Content: ")[1].split("</revised_slide>")[0].strip()

    slide_chunks_df.at[index, "checklist_based_slide_title"] = current_title
    slide_chunks_df.at[index, "checklist_based_slide_content"] = current_chunk


def run_checklist_review_and_revise(sheet, worksheet_name, course_name, target_audience, llm="gemini_2_flash"):
    slide_chunks_sheet, slide_chunks_df = get_sheet_data_and_df(sheet, worksheet_name)
    _, course_info_df = get_sheet_data_and_df(sheet, 'Course info')

    checklist_sheet_link = course_info_df['Checklist Link'][0]
    # gc = gspread.service_account(filename='content/service-credentials.json')
    gc = st.session_state["gc"]
    checklist_sheet = gc.open_by_url(checklist_sheet_link)
    checklist_sheet, checklist_df = get_sheet_data_and_df(checklist_sheet,  "Slide Chunks Checklist")

    required_columns = ["checklist_based_review_output", "checklist_based_slide_title", "checklist_based_slide_content"]
    for col in required_columns:
        if col not in slide_chunks_df.columns:
            slide_chunks_df[col] = ""

    save_to_sheet(slide_chunks_sheet, slide_chunks_df)
    
    ensure_checklist_columns(sheet,worksheet_name)

    futures_map = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        for index, row in slide_chunks_df.iterrows():
            if row["checklist_based_review_output"]:
                print(f"Skipping slide {index + 1}: Already reviewed.")
                continue

            future = executor.submit(process_slide, index, row, course_name, target_audience, llm, checklist_df, slide_chunks_df)
            futures_map[future] = index

        total_tasks = len(futures_map)
        progress = SmartProgressBar(total_tasks=total_tasks, description="Percent complete")

        for future in tqdm(as_completed(futures_map), total=total_tasks, desc="Processing Slides"):
            index = futures_map[future]
            try:
                future.result()

                print(f" ✅ Successfully updated slide {index + 1}.")

                save_to_sheet(checklist_sheet, checklist_df)
                save_to_sheet(slide_chunks_sheet, slide_chunks_df)

            except Exception as e:
                print(f"Error processing slide {index}: {e}")
            progress.update()

    base_columns = ["Task", "Review Criteria"]
    # Reorder columns to match required structure after processing all slides
    current_cols = list(checklist_df.columns)
    
    # Get all columns except base columns
    other_cols = [col for col in current_cols if col not in base_columns]
    
    # Sort slide columns
    slide_cols = [col for col in other_cols if col.startswith('Slide ')]
    slide_cols_sorted = sorted(slide_cols, key=lambda x: int(x.split(' ')[1]))
    
    # Get remaining non-slide columns
    remaining_cols = [col for col in other_cols if col not in slide_cols]
    
    # Create new DataFrame with correct column order
    new_df = pd.DataFrame()
    # Add base columns first
    for col in base_columns:
        new_df[col] = checklist_df[col]
    # Add sorted slide columns
    for col in slide_cols_sorted:
        new_df[col] = checklist_df[col]
    # Add remaining columns
    for col in remaining_cols:
        new_df[col] = checklist_df[col]
    
    # Replace the original DataFrame
    checklist_df = new_df

    # Save the reordered DataFrame back to the sheet
    save_to_sheet(checklist_sheet, checklist_df)


    print("\n✅ Checklist Review & Revise Process Completed 🚀")