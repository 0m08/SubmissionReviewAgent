from modules.chain import Chain
import requests
import pandas as pd
import re
from services.sheets_service import get_sheet_data_and_df, save_to_sheet, hide_columns_by_name
import time
from tqdm import tqdm
from services.smart_progress_bar import SmartProgressBar
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from gspread_formatting import set_column_width



readability_agent_prompt = """We are creating structured slide content for an e-learning course. You are a Readability score based Revisor Agent - an expert instructional designer with a specialization in simplifying complex educational content. Your task is to revise slide content that received a low readability score, indicating that the language may be too complex for the target audience and needs to be made more accessible. You will revise the content to make it easier to understand while preserving its original meaning and instructional purpose.

<course_information>
Course name: {course_name}
Target audience: {target_audience}
</course_information>

Here are the slides that require revision. In some cases, multiple slides may appear together if they were previously merged due to low character count. You must revise each slide independently:

<slide>

{slide_content}

</slide>

Revision Guidelines: Before revising the slide content, follow these principles to ensure your revisions improve readability while preserving instructional accuracy:

1) Simplify Sentence Structure:
  - Break down long or complex sentences into shorter, clear ones.
  - Use straightforward sentence construction to avoid confusion.
  - Ensure transitions between ideas are smooth and logically connected.

2) Use Simple and Familiar Words:
  - Replace difficult or technical vocabulary with more common alternatives — unless the term is essential to the audience’s understanding.
  - Avoid jargon, abstract words, or overly academic language unless required by context.

3) Keep Content Concise and Focused:
  - Remove unnecessary words or phrases that add complexity without value.
  - Avoid repetition and excessive elaboration.

 4) Maintain the Original Meaning:
  - Do not add any new information that was not present in the original slide content — no explanations, examples, or background context beyond what is already written.
  - Do not remove any existing information from the original slide content.
  - Do not change the meaning of any sentence or idea — your revisions must only improve readability, not rephrase or reinterpret the content in a way that alters its instructional purpose.
  - If you're simplifying a sentence, you must retain 100% of its original factual meaning and intent.
  - Your role is to refine the wording, not revise the message.

5) Tailor the Language to the Target Audience:
  - Use a tone and style appropriate for the Target Audience.
  - Ensure the language is accessible for learners who may not have advanced technical or academic backgrounds.

6) Write in a Natural, Human-Sounding Style:
  - Use everyday language and a conversational tone, without sounding overly casual or informal.
  - Avoid robotic phrasing or rigid structure — aim for smooth, readable flow.

7) Ensure Consistent Length:
  - Revised content should remain similar in length to the original — do not make the slide significantly longer or shorter.
  - If a sentence must be added or split for clarity, keep the overall revision balanced.

8) Revise Each Slide Independently:
  - If two slides appear together (due to merging for AI analysis), revise them as separate slides.
  - Do not combine ideas from multiple slides. Ensure each output corresponds to one slide only.

9) Avoid Fluff or Over-Explanation:
  - Only make changes necessary to improve readability.
  - Do not expand explanations or include extra background information not present in the original.

<evaluation_breakdown>

Before revising the slides, analyze the content to identify why the readability may be low. Look for issues such as long or complex sentence structure, difficult vocabulary, or unclear phrasing. Then, explain your approach to revising the content to make it easier to read and understand — focusing on clarity, simplicity, and accessibility — while strictly preserving the original meaning and instructional intent of the slide.

</evaluation_breakdown>

Based on your above evaluation, provide the revised slide content in the following format:

<output>

<revised_slide_content>
[Enter revised slide content for Slide 1 here]
</revised_slide_content>

<revised_slide_content>
[Enter revised slide content for Slide 2 here if applicable. If there is only 1 slide, omit this entirely]
</revised_slide_content>

(Repeat the <revised_slide_content> block for each slide that appears in the input. If only one slide is present, include only one block. If multiple slides were provided in the input, revise each slide independently and provide one <revised_slide_content> block per slide in the same order as the input)

</output>
"""

def run_readability_revision(course_name, target_audience, slide_content, llm="gemini_2_flash"):
    """
    Run the Readability Score Based Revisor Agent on slide content.

    :param course_name: The name of the course.
    :param target_audience: The target audience for the course.
    :param slide_content: The full merged or single slide content that needs revision.
    :param llm: The language model to use.
    :return: Revised slide content from the agent's output.
    """

    #  Initialize the Readability Revisor Agent
    readability_agent = Chain(llm=llm, tags=["output"])

    # Format the prompt for debugging (printing)
    formatted_prompt = readability_agent_prompt.format(
        course_name=course_name,
        target_audience=target_audience,
        slide_content=slide_content
    )

    # Print the formatted prompt for debugging (just for printing purposes)
    print("\n🔹 READABILITY REVISION PROMPT BEING SENT TO LLM:\n")
    print(formatted_prompt)
    print("\n" + "=" * 100 + "\n")

    # Add the user message
    readability_agent.add_message(
        role="user",
        content=readability_agent_prompt.format(
            course_name=course_name,
            target_audience=target_audience,
            slide_content=slide_content
        )
    )

    # Run the agent
    response = readability_agent.run()

    # Extract the structured revised slide output
    return response["output"]



# Winston AI API Configuration
WINSTON_AI_URL = "https://api.gowinston.ai/v2/ai-content-detection"
WINSTON_API_KEY = os.environ.get("WINSTON_API_KEY")

# Function to check AI detection for a given slide content
def check_ai_winston(text):
    """
    Sends slide content to Winston AI and retrieves AI detection results.

    :param text: Slide content to analyze.
    :return: Dictionary with 8 AI detection results generated from the Winston.ai API
    """

    headers = {
        "Authorization": f"Bearer {WINSTON_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "language": "en",
        "version": "latest",
        "sentences": True
    }

    print(f"\n🟢 Sending request to Winston AI...")
    print(f"🔹 Text: {text}")

    try:
        response = requests.post(WINSTON_AI_URL, headers=headers, json=data)
        print(f"🟡 API Response Code: {response.status_code}")
        result = response.json()
        print(f"📝 Full API Response: {result}")

        if response.status_code == 403:
            error_message = result.get("description", "403 Forbidden")
            print(f"❌ 403 Forbidden: {error_message}")
            return {
                "Human Score": "Error",
                "Sentence-Level Scores": "Error",
                "Readability Score": "Error",
                "AI Manipulation Detected": "Error",
                "Zero-Width Space Attack": "Error",
                "Homoglyph Attack": "Error",
                "Detected Language": "Error",
                "AI Detection Error": error_message
            }

        response.raise_for_status()

        return {
            "Human Score": result.get("score", 0),
            "Sentence-Level Scores": str(result.get("sentences", {})) or "N/A",
            "Readability Score": result.get("readability_score", 0),
            "AI Manipulation Detected": "Yes" if result.get("attack_detected", {}).get("zero_width_space", False) or result.get("attack_detected", {}).get("homoglyph_attack", False) else "No",
            "Zero-Width Space Attack": result.get("attack_detected", {}).get("zero_width_space", False),
            "Homoglyph Attack": result.get("attack_detected", {}).get("homoglyph_attack", False),
            "Detected Language": result.get("language", "Unknown"),
            "AI Detection Error": ""
        }

    except requests.exceptions.RequestException as e:
        print(f"❌ API Request Failed: {e}")
        return {
            "Human Score": "Error",
            "Sentence-Level Scores": "Error",
            "Readability Score": "Error",
            "AI Manipulation Detected": "Error",
            "Zero-Width Space Attack": "Error",
            "Homoglyph Attack": "Error",
            "Detected Language": "Error",
            "AI Detection Error": str(e)
        }

# Function to format merged slides
def format_slides_for_readability_prompt(slide_contents):
    if len(slide_contents) == 1:
        return f"<slide>\n\n{slide_contents[0]}\n\n</slide>"
    return "\n\n".join([f"<slide_{i+1}>\n{slide}\n</slide_{i+1}>" for i, slide in enumerate(slide_contents)])




    


def run_ai_detection_with_readability(sheet, worksheet_name, course_name, target_audience, max_iterations=5):
    """
    Runs Winston.ai AI detection on each slide and applies Readability Reviser agent if Readability score < 50.
    Handles merged slide cases and updates results iteratively.

    :param sheet: Google Sheet object.
    :param slide_chunks_df: DataFrame of slide chunks.
    :param course_name: Name of the course.
    :param target_audience: Target audience description.
    :param max_iterations: Max allowed review-revise cycles per slide.
    :return: Updated DataFrame with all AI detection result columns and revisions.
    """

    # Ensure all required columns exist
    required_columns = [
        "Human Score", "Sentence-Level Scores", "Readability Score",
        "AI Manipulation Detected", "Zero-Width Space Attack",
        "Homoglyph Attack", "Detected Language", "AI Detection Error"
    ]
    slide_chunks_sheet, slide_chunks_df = get_sheet_data_and_df(sheet, worksheet_name)
    
    for col in required_columns:
        if col not in slide_chunks_df.columns:
            slide_chunks_df[col] = ""
    
    # Insert Slide No. as the first column
    slide_chunks_df.insert(0, "Slide No.", range(1, len(slide_chunks_df) + 1))

    save_to_sheet(slide_chunks_sheet, slide_chunks_df)

    # ✅ Set the column width of "Slide No." (column A) to 50
    set_column_width(slide_chunks_sheet, 'A', 50)

    # Freeze the first column ("Slide No.")
    slide_chunks_sheet.freeze(rows=0, cols=1)

    def process_slide(index):
        iteration = 0
        merged = False
        merged_with_previous = False

        while iteration < max_iterations:
            slide_1 = slide_chunks_df.at[index, "final_slide_content"]
            slide_2 = None

            if not slide_1 or pd.isna(slide_1):
                # return index, merged, merged_with_previous
                break

            # Merge short slides (<300 chars) with next or previous
            if iteration == 0 and len(slide_1) < 300 and index + 1 < len(slide_chunks_df):
                slide_2 = slide_chunks_df.at[index + 1, "final_slide_content"]
                if slide_2 and not pd.isna(slide_2):
                    merged = True
            elif iteration == 0 and len(slide_1) < 300 and index == len(slide_chunks_df) - 1 and index > 0:
                prev_slide = slide_chunks_df.at[index - 1, "final_slide_content"]
                if prev_slide and not pd.isna(prev_slide):
                    slide_2 = slide_1
                    slide_1 = prev_slide
                    merged = True
                    merged_with_previous = True
                    index -= 1

            # Re-fetch revised slides in later iterations if merged
            if merged and iteration > 0:
                slide_1 = slide_chunks_df.at[index, "final_slide_content"]
                slide_2 = slide_chunks_df.at[index + 1, "final_slide_content"]

            # Prepare content for AI check or readability revision
            if merged and slide_2:
                combined_text = slide_1 + " " + slide_2
                formatted_slide_input = format_slides_for_readability_prompt([slide_1, slide_2])
            else:
                combined_text = slide_1
                formatted_slide_input = format_slides_for_readability_prompt([slide_1])

            # Run Winston AI detection
            ai_result = check_ai_winston(combined_text)
            for key, value in ai_result.items():
                slide_chunks_df.at[index, key] = value

            if merged:
                # Mark Slide B as "Merged into previous row"
                slide_chunks_df.at[index + 1, "Human Score"] = "Merged into previous row"
                slide_chunks_df.at[index + 1, "AI Detection Error"] = "Merged into previous row"
                slide_chunks_df.at[index + 1, "Readability Score"] = ""
                slide_chunks_df.at[index + 1, "Sentence-Level Scores"] = ""
                slide_chunks_df.at[index + 1, "final_slide_content"] = ""

            score = ai_result.get("Readability Score", 100)
            try:
                score = float(score)
            except:
                # return index, merged, merged_with_previous
                break
            # Check readability score
            if score >= 50:
                # return index, merged, merged_with_previous  # Readability Score is acceptable
                break

            # Run Readability Reviser if Readability Score <50
            print(f"🔁 Iteration {iteration + 1}: Readability Score < 50. Running readability revision...")
            revised_output = run_readability_revision(
                course_name=course_name,
                target_audience=target_audience,
                slide_content=formatted_slide_input
            )

            # Extract and update revised slide content in the final_slide_content column
            revised_blocks = re.findall(r"<revised_slide_content>(.*?)</revised_slide_content>", revised_output, re.DOTALL)
            if not revised_blocks:
                print("⚠️ No revised content returned. Breaking.")
                # return index, merged, merged_with_previous
                break

            slide_chunks_df.at[index, "final_slide_content"] = revised_blocks[0].strip()
            if merged and len(revised_blocks) > 1:
                slide_chunks_df.at[index + 1, "final_slide_content"] = revised_blocks[1].strip()

            iteration += 1

        return index, merged, merged_with_previous

    futures_map = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        for index in range(len(slide_chunks_df)):
            future = executor.submit(process_slide, index)
            futures_map[future] = index
            
        total_tasks = len(futures_map)
        progress = SmartProgressBar(total_tasks=total_tasks, description="Percent complete")

        for future in as_completed(futures_map):
            index, merged, merged_with_previous = future.result()
            if merged:
                index += 2
            else:
                index += 1
            if merged_with_previous:
                index += 1
            progress.update()
        save_to_sheet(slide_chunks_sheet, slide_chunks_df)
        print("✅ Sheet updated with all slide results.")

    # ✅ Hide AI detection-related columns
    hide_columns_by_name(
        worksheet = slide_chunks_sheet,
        column_names = [    
        "Human Score", "Sentence-Level Scores", "Readability Score",
        "AI Manipulation Detected", "Zero-Width Space Attack",
        "Homoglyph Attack", "Detected Language", "AI Detection Error",
        "Plagiarism Score", "Total Words", "Plagiarized Words",
        "Identical Matches", "Similar Matches", "Sources Found", "Plagiarized Text",
        ],
        df = slide_chunks_df
    )

    print("\n✅ AI detection completed and columns hidden")

    return slide_chunks_df
        


