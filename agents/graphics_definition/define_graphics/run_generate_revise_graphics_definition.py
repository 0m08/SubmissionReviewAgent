from services.sheets_service import get_sheet_data_and_df
from modules.chain import Chain
from tqdm import tqdm
import re
import streamlit as st
from services.smart_progress_bar import SmartProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed


from agents.graphics_definition.define_graphics.generate_graphics_definition import generate_graphics_definition
from agents.graphics_definition.define_graphics.generate_graphics_definition import get_previous_graphics_definition_as_str
from agents.graphics_definition.define_graphics.generate_complexity_review import generate_complexity_review
from agents.graphics_definition.define_graphics.generate_missing_sentences_review import generate_missing_sentences_review
from agents.graphics_definition.define_graphics.generate_accuracy_review import generate_accuracy_review
from agents.graphics_definition.define_graphics.generate_reuse_previous_review import generate_reuse_previous_graphics_review
from agents.graphics_definition.define_graphics.generate_reviser_output import generate_reviser_output_for_slide





reviser_agent_prompt = """ You are a Graphics Definition Revision Agent. Your primary task is to generate a revised graphics definition by analyzing an existing graphic definition for the given slide, considering both AI-generated reviews and human reviews for this graphics definition to address identified issues and incorporate the insights from the AI and human reviews. Your overarching goal is to create a visual representation of the slide content that is clear, instructionally effective, and engaging, while carefully balancing simplicity, accuracy, and creative variation.

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

Here are the AI-generated reviews:

<ai_reviews>

Complexity Review:
{complexity_review}

Missing Sentences Review:
{missing_sentences_review}

Accuracy Review:
{accuracy_review}

Opportunities to Reuse Previous Graphics Review:
{reuse_previous_graphics_review}

</ai_reviews>

Here's the human review:

<human_review>
{human_review}
</human_review>

Based on the AI reviews and human feedback, generate a revised graphics definition that addresses the identified issues while maintaining the instructional value and visual appeal of the slide. Be mindful of the existing visual style and coherence with previous slides.

Follow these guidelines for generating the revised graphics definition:

1. Prioritize Human Feedback First, AI Feedback Second:
    - If the human reviewer has provided any feedback, it must be given highest priority in the revision process.
    - Directly address any concerns or requests from the human review, even if they contradict AI feedback.
    - If the human review explicitly approves a certain aspect of the graphics definition that AI flagged, defer to the human review unless there is any critical issue.
    - If the human review does not mention an issue that AI flagged, do not assume approval—instead, evaluate the AI feedback and apply it while revising the graphics definitons.
    - AI feedback should still be considered and incorporated when necessary, but only after human feedback has been addressed.
    - If there is no human feedback, AI reviews should be followed carefully to improve the graphics definition.

2. Address AI-Identified Issues:
    - Analyze the reviews for Complexity, Missing Sentences, Accuracy and Opportunities to Reuse Previous Graphics provided by the AI.

  A) Complexity Review:
    - Check any issues identified in the Complexity review and incorporate the suggestions provided in the review to simplify the scene by reducing its complexity while maintaining its instructional value.

  B) Missing Sentences Review:
    - Check any issues identified in the Missing Sentence review and incorporate the suggestions provided in the review to ensure that all sentences are visually represented in the revised graphics definition.

  C) Accuracy Review:
    - Check any issues identified in the Accuracy review and incorporate the suggestions provided to ensure that the graphics definition correctly represents the slide content without misinterpretation or inconsistencies.

  D) Opportunities to Reuse Previous Graphics Review:
    - Check any issues identified in the Opportunities to Reuse Previous Graphics review and incorporate the suggestions provided to integrate the specified visual elements from the referenced Scene IDs, ensuring visual consistency while maintaining clarity and instructional intent.

3. Incorporate Human Feedback:
    - Address any comments from the Human Review, prioritizing specific requests and suggestions. If the human review expresses dissatisfaction with any particular aspect of the graphics definition, make changes in the graphics definition to address their concerns.

4. Maintain Instructional Value and Visual Appeal:
    - Ensure that the revised graphics definition remains instructionally sound and accurately represents the slide content.

5. Consider the Previous Graphics Definition:
    - Maintain a consistent visual style, color scheme, and conceptual flow with previous slides. If the existing graphics definition reuses elements from previous slides, continue to do so in the revised definition, unless the human reviewer or AI reviews requests a different approach.

Present your output in the following format:

<output>

<evaluation_breakdown>
- AI Review Summary:
  - Complexity Review: Summarize the key points from the Complexity Reviews across all scenes in the graphics definition. Highlight all the issues identified across scenes. If no reviews are present for Complexity, state: "No Complexity issues."
  - Missing Sentences Review: Summarize the key points from the Missing Sentences Review. Highlight all the issues identified. If no reviews are present for Missing Sentences, state: "No Missing Sentences issues."
  - Accuracy Review: Summarize the key points from the Accuracy Reviews across all scenes in the graphics definition. Highlight all the issues identified across scenes. If no reviews are present for Accuracy, state: "No Accuracy issues."
  - Reuse Previous Graphics Review: Summarize the key points from the Opportunities to Reuse Previous Graphics Review. Highlight all identified missed reuse opportunities and the suggested Scene IDs for reuse. If no reuse issues were flagged, state: "No Missed Reuse opportunities."
- Human Review Summary: Summarize the key points from the Human Review. Highlight all the issues identified across scenes. If no reviews are present from Human, state: "No issues identified by the human."

- Approach to Revision: Explain your overall strategy for revising the graphics definition based on the four AI reviews and the human review. Describe the specific changes that you will be doing and the reasoning behind those decisions. Give a detailed explaination of the approach you will be taking while revising the graphics definition based on all the reviews.

<evaluation_breakdown>

<revised_graphics_definition>
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
</revised_graphics_definition>

</output>
"""

reviser_agent_system_prompt = """You are a Graphics Definition Reviser Agent.

Here is the Reference for this slide:
<references>
{references}
</references>

Here is the list of graphics definition defined for previous slides:
<previous_graphics_definition>
{previous_graphics_definition}
<previous_graphics_definition>

When generating the graphics definition for this slide:

1. Strictly Adhere to References Over AI Feedbacks: If references are provided, they must take absolute priority over AI-generated reviews. Do not modify or revise aspects explicitly specified in the references, even if AI reviews suggest otherwise. Only make minimal adjustments to referenced elements if absolutely necessary for clarity, correctness, or feasibility. References may include:
    - General instructions or suggestions on how to represent concepts from the slide visually.
    - Descriptions of specific graphics that should be included, either in part or in full.
    - Any other directives meant to shape the graphics definition according to specific requirements.

2. Reusability:

    A) Reuse from previous slides when applicable: If any graphics from previous slides align with this slide’s content, reuse them and explicitly mention the "Scene ID" of graphics definition being reused (in part or in full):
    - If the previous graphics fully meets the needs of this slide, use it as-is.
    - If adjustments are required while reusing, adapt the graphics while maintaining its core structure and clearly specify all modifications made.

    B) Introduce new graphics when necessary: If no suitable graphics exist for reuse, create a new graphics definition that effectively represents the slide’s purpose.

    C) Ensure coherence across slides: Maintain a consistent visual style, color schemes, and conceptual flow across all slides, whether reusing or introducing new graphics.
"""


def generate_reviser_output_for_slide(course_name, target_audience, slide_title, slide_content, graphics_definition, complexity_review, missing_sentences_review, accuracy_review, reuse_previous_graphics_review, human_review, references, previous_graphics_definition = "", llm = "gemini_2_flash"):
    """
    Calls the Reviser Agent for a single slide, returning the revised graphics definition.

    :param course_name: The name of the course
    :param target_audience: The slide's target audience
    :param slide_title: The title of the slide
    :param slide_content: The content of the slide (text)
    :param graphics_definition: The original scene-based definition for this slide
    :param complexity_review: The complexity review output for this slide
    :param missing_sentences_review: The missing-sentences review output for this slide
    :param accuracy_review: The accuracy review output for this slide
    :param reuse_previous_graphics_review: The reuse previous graphics review output for this slide
    :param human_review: Human-provided feedback for this slide
    :param previous_graphics_definition:  Previous graphics definition from earlier slides
    :param references: The reference text provided for this slide
    :return: The revised graphics definition text produced by the Reviser Agent.
    """

    # Initialize the agent
    reviser_agent = Chain(llm = llm, tags=['revised_graphics_definition'])

    # Add the system message
    reviser_agent.add_message(
        role = "system",
        content = reviser_agent_system_prompt.format(
            previous_graphics_definition = previous_graphics_definition,
            references = references
        )
    )

    # Add the user message
    reviser_agent.add_message(
        role = "user",
        content = reviser_agent_prompt.format(
            course_name = course_name,
            target_audience = target_audience,
            slide_title = slide_title,
            slide_content = slide_content,
            graphics_definition = graphics_definition,
            complexity_review = complexity_review,
            missing_sentences_review = missing_sentences_review,
            accuracy_review = accuracy_review,
            reuse_previous_graphics_review = reuse_previous_graphics_review,
            human_review = human_review
        )
    )

    # Run the agent
    response = reviser_agent.run()

    return response['revised_graphics_definition']


            
def run_generate_graphics_definition(sheet, worksheet_name, course_name, target_audience, progress, llm="gemini_2_flash"):
        # Read the sheet and df
    slide_chunks_sheet, slide_chunks_df = get_sheet_data_and_df(sheet, worksheet_name)

    # Ensure required columns exist
    for col in ['graphics_definition', 'complexity_review', 'missing_sentences_review', 'accuracy_review', 'reuse_previous_graphics_review', 'human_review', 'revised_graphics_definition']:
        if col not in slide_chunks_df.columns:
            slide_chunks_df[col] = ""

    # To store all the previous graphics definition
    previous_graphics_definition = ""

    for index, row in slide_chunks_df.iterrows():
        if row['graphics_definition'].strip():
            continue

        references = row['Reference Description'].strip()
        revised_graphics_definition = row['revised_graphics_definition'].strip()

        previous_graphics_definition = get_previous_graphics_definition_as_str(
            slide_no=index + 1,
            text=revised_graphics_definition,
            existing_definition_str=previous_graphics_definition
        )

        graphics_definition = generate_graphics_definition(
            course_name=course_name,
            target_audience=target_audience,
            slide_title=row['Slide Title'],
            slide_content=row['Slide Content'],
            previous_graphics_definition=previous_graphics_definition,
            references=references,
            llm=llm
        )

        results = {'graphics_definition': graphics_definition}
        progress.update()

        def generate_review(review_func, *args):
            review_text = review_func(*args)
            verdict_match = re.search(r"<verdict>\s*(.*?)\s*</verdict>", review_text, re.DOTALL)
            verdict = verdict_match.group(1).strip().lower() if verdict_match else ""
            return review_text if verdict == "fail" else ""

        with ThreadPoolExecutor() as executor:
            future_to_review = {
                executor.submit(generate_review, generate_complexity_review, course_name, target_audience, row['Slide Title'], row['Slide Content'], graphics_definition, llm): 'complexity_review',
                executor.submit(generate_review, generate_missing_sentences_review, course_name, target_audience, row['Slide Title'], row['Slide Content'], graphics_definition, llm): 'missing_sentences_review',
                executor.submit(generate_review, generate_accuracy_review, course_name, target_audience, row['Slide Title'], row['Slide Content'], graphics_definition, llm): 'accuracy_review',
                executor.submit(generate_review, generate_reuse_previous_graphics_review, course_name, target_audience, row['Slide Title'], row['Slide Content'], graphics_definition, previous_graphics_definition, llm): 'reuse_previous_graphics_review'
            }

            for future in as_completed(future_to_review):
                review_type = future_to_review[future]
                try:
                    results[review_type] = future.result()
                except Exception as e:
                    results[review_type] = f"Error: {str(e)}"
                    progress.update()

        slide_chunks_df.loc[index, 'graphics_definition'] = results['graphics_definition']
        slide_chunks_df.loc[index, 'complexity_review'] = results['complexity_review']
        slide_chunks_df.loc[index, 'missing_sentences_review'] = results['missing_sentences_review']
        slide_chunks_df.loc[index, 'accuracy_review'] = results['accuracy_review']
        slide_chunks_df.loc[index, 'reuse_previous_graphics_review'] = results['reuse_previous_graphics_review']

        slide_chunks_df = slide_chunks_df.astype(str)
        slide_chunks_sheet.update([slide_chunks_df.columns.values.tolist()] + slide_chunks_df.values.tolist())

        print(f"Please enter manual feedback for this graphics definition: row_number - {index + 1}")
        return  # Stop execution after one row is processed

    return True


def run_revise_generated_graphics_definition(sheet, worksheet_name, course_name, target_audience, progress,  llm="gemini_2_flash"):
    """
    Process revised graphics definition for each row and generate graphics definition for the next row if needed.
    """
    slide_chunks_sheet, slide_chunks_df = get_sheet_data_and_df(sheet, worksheet_name)
    previous_graphics_definition = ""

    for index, row in slide_chunks_df.iterrows():
        # Skip processing if revised_graphics_definition is already populated
        if row['revised_graphics_definition'].strip():
            print(f"✅ Slide {index + 1}: Revised Graphics Definition already populated. Skipping.")
            continue
        references = row['Reference Description'].strip()
        graphics_definition = row['graphics_definition'].strip()
        complexity_review = row['complexity_review'].strip()
        missing_sentences_review = row['missing_sentences_review'].strip()
        accuracy_review = row['accuracy_review'].strip()
        reuse_previous_graphics_review = row['reuse_previous_graphics_review'].strip()
        human_review = row['human_review'].strip()

        # Skip the reviser agent if all review columns are empty
        if not (complexity_review or missing_sentences_review or accuracy_review or reuse_previous_graphics_review or human_review):
            print("✅ No issues detected. Skipping Reviser Agent and using original Graphics Definition.\n")
            revised_graphics_definition = graphics_definition  # Copy graphics definition directly

            # Update previous graphics definition even when reviser agent is skipped
            previous_graphics_definition = get_previous_graphics_definition_as_str(
                slide_no = index + 1,
                text = revised_graphics_definition,
                existing_definition_str = previous_graphics_definition
            )

        # slide_chunks_df.loc[index, 'revised_graphics_definition'] = revised_graphics_definition
        # slide_chunks_df = slide_chunks_df.astype(str)
        # slide_chunks_sheet.update([slide_chunks_df.columns.values.tolist()] + slide_chunks_df.values.tolist())
        # continue  # Skip reviser agent and move to the next slide

        # Run the revised graphics definition function if any review is present
        print("⏳ Generating Revised Graphics Definition\n")

        revised_graphics_definition = generate_reviser_output_for_slide(
            course_name = course_name,
            target_audience = target_audience,
            slide_title = row['Slide Title'],
            slide_content = row['Slide Content'],
            graphics_definition = graphics_definition,
            complexity_review = complexity_review,
            missing_sentences_review = missing_sentences_review,
            accuracy_review = accuracy_review,
            reuse_previous_graphics_review = reuse_previous_graphics_review,
            human_review = human_review,
            previous_graphics_definition = previous_graphics_definition,
            references = references,
            llm = llm
        )

        print("✅ Generated Revised Graphics Definition\n")
        print("-"*100)
        # progress.update()  # Update progress after generating revised graphics definition

        # Add this to the df
        slide_chunks_df.loc[index, 'revised_graphics_definition'] = revised_graphics_definition

        # Preserve the human review in the sheet before updating the DataFrame
        slide_chunks_df.loc[index, 'human_review'] = human_review

        # Convert all values to strings before updating
        slide_chunks_df = slide_chunks_df.astype(str)

        # Update Sheet with Reviser Output
        slide_chunks_sheet.update([slide_chunks_df.columns.values.tolist()] + slide_chunks_df.values.tolist())
        
        # Update previous graphics definition
        # Update previous graphics definition even when reviser agent is skipped
        previous_graphics_definition = get_previous_graphics_definition_as_str(
            slide_no = index + 1,
            text = revised_graphics_definition,
            existing_definition_str = previous_graphics_definition
        )
        
        return
    return True

def run_generate_and_revise_graphics(sheet, worksheet_name, course_name, target_audience, llm="gemini_2_flash"):
    """
    Combined function to:
    1. Generate graphics definitions if missing.
    2. Revise graphics definitions if needed.
    3. Ensure each slide is processed in order.
    
    The function stops when a graphics definition needs manual review.
    :param sheet: The sheet object.
    :param worksheet_name: The worksheet name.
    :param course_name: The course name.
    :param target_audience: The target audience.
    :param llm: The language model to use.
    :return: True if all slides have been processed, False otherwise.
    """

    # Read the sheet and dataframe
    slide_chunks_sheet, slide_chunks_df = get_sheet_data_and_df(sheet, worksheet_name)

    # Ensure required columns exist
    for col in ['graphics_definition', 'complexity_review', 'missing_sentences_review', 
                'accuracy_review', 'reuse_previous_graphics_review', 'human_review', 'revised_graphics_definition']:
        if col not in slide_chunks_df.columns:
            slide_chunks_df[col] = ""
    
    total_tasks = 6 * len(slide_chunks_df)  # 6 tasks per slide
    # progress = SmartProgressBar(total_tasks=total_tasks, description="Percent complete")
    
    if not hasattr(run_generate_and_revise_graphics, "progress"):
        run_generate_and_revise_graphics.progress = SmartProgressBar(total_tasks=total_tasks, description="Percent complete")

    progress = run_generate_and_revise_graphics.progress

    for index, row in tqdm(slide_chunks_df.iterrows(), total=total_tasks):  # Track rows, not tasks
        graphics_definition = row['graphics_definition'].strip()
        revised_graphics_definition = row['revised_graphics_definition'].strip()

        # If graphics definition exists, check if revision is needed
        if graphics_definition:
            if revised_graphics_definition:
                progress.update()
                continue  # Both exist, move to the next row

            # Run reviser function since revised definition is missing
            print(f"⏳ Revising Graphics Definition for Slide {index + 1}\n")
            run_revise_generated_graphics_definition(sheet, worksheet_name, course_name, target_audience, progress, llm)
            # progress.update()
            continue  # Move to the next row after revising

        # If graphics definition is missing, generate it and stop execution
        print(f"⏳ Generating Graphics Definition for Slide {index + 1}\n")
        run_generate_graphics_definition(sheet, worksheet_name, course_name, target_audience, progress, llm)
        # progress.update()

        # Show message and stop execution
        st.write(f"✔ Graphics Definition generated for Slide {index + 1}. Please enter review comments before continuing.(Optional)")
        return  # Stop execution to allow user to review

    print('All slides processed.')
    return True  # If all slides have been processed, return True