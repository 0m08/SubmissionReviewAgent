from agents.generate_assessments.chains import Chain
from services.sheets_service import (
    get_sheet_data_and_df,
    hide_worksheet_by_name,
    format_worksheet,
    create_or_read_worksheet,
    save_to_sheet,
    delete_worksheet,
)
import pandas as pd
from tqdm import tqdm
from services.smart_progress_bar import SmartProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed
from langsmith import traceable
import streamlit as st
 

generate_assessment_prompt = """As an expert Instructional Designer with extensive knowledge of the subject {course_name} and the topic {topic_name}, your task is to create tailored assessment questions for a self-paced e-learning course on {course_name}, specifically designed for {target_audience}. Please develop assessment questions based on the content provided in the following slides, which are delimited by XML tags.

<slides>
{slides}
</slides>

Before creating questions, first analyze the slide content to determine if Matching questions are appropriate for this topic:
- Identify if the topic discusses components of a system, device, or equipment
- Check if the topic presents types, categories, or classifications of something  
- Look for lists of closely related items that naturally group together
- Use this analysis to determine if matching questions would be meaningful and educational for this specific topic content

Create questions using the following approach:

ALWAYS create these question types:
- Multiple Choice: allows the selection of a single response from a pre-defined list
- True/False: a simple form of multiple choice question with just the two choices 'True' and 'False'
- Select All That Apply: allows selection of multiple correct responses from a pre-defined list

CONDITIONALLY create Matching questions ONLY if the slide content contains:
- Components of a system, device, or process
- Types or categories of related items within the same domain
- Lists of closely related concepts that naturally group together
- Items that require understanding relationships, not just definition recall

AVOID Matching questions when:
- The topic only contains unrelated definitions or concepts
- Matches would be obvious or trivial (like simple acronym definitions)
- The slide content doesn't discuss grouped or related items

Question Quantity Rules:
- Generate an appropriate number of questions based on the topic content depth and complexity
- For Select All That Apply questions: Create only 1 if your total will be less than 9 questions, or 2+ if your total will be 9 or more questions

Follow these guidelines when creating the questions.
- Ensure exhaustive and proportional coverage of the course content, including a mix of first-level and second-level questions, with thoughtful answer choices.
- Any field that does not apply to a given question type should be left empty. For instance, True/False question type will have options C and D left empty.
- For Matching question types, give the exact correct answer under the ""Correct Answer"" field. Match correctly without using Options. Use the exact option text and words being matched. Ensure that all the option text and match text for matching type questions are unique.
- For Select All That Apply questions: Use when multiple options can be simultaneously correct. Always include "(Select all that apply)" at the end of the question text. Ensure 2-3 options are correct out of 4 total options. In the ""Correct Answer"" field, list all correct option letters separated by commas (e.g., ""A, B, D""). IMPORTANT: Limit to only 1 Select All That Apply question if generating less than 9 total questions for this topic. If generating 9 or more questions, you may create 2 or more Select All That Apply questions.
- The correct feedback will be shown everytime the user answers correctly. The incorrect feedback is common feedback that will be shown anytime a user does not answer the question correctly and it should explain the correct answer.
- Correct and incorrect feedback must be provided for all question types, including matching.
- IMPORTANT: Do not reference option letters (A, B, C, D) in any feedback. Instead, refer to the actual content/text of the options. For example, say "Proper ventilation and energy efficiency are correct" instead of "A and B are correct".
- For True/False questions, if the correct answer is True, always start the correct and incorrect feedback with "Correct! This statement is true..." and "Incorrect! This statement is true..." respectively. If the correct answer is False, always start the correct and incorrect feedback with "Correct! This statement is false..." and "Incorrect! This statement is false..." respectively. Always use lowercase "true" and "false" in this starting statement.
- Creation of Matching questions is optional and should be created only if the topic content naturally contains related items.

Reply in the following output format:

<output>

<question>
Question no: 1
Question Type: [Question type here]
Question: [Question text here]

Option A: [Option A text here]
Option B: [Option B text here]
Option C: [Option C text here. Leave blank for True/False questions]
Option D: [Option D text here. Leave blank for True/False questions]

Correct Answer: [Correct Option letter eg. ""C""]

Correct feedback: [Feedback for correct answer. For True/False questions: If correct answer is True, start with "Correct! This statement is true..." If correct answer is False, start with "Correct! This statement is false..." For other question types, start with "Correct!" and explain why the chosen option content is correct without mentioning option letters.]

Incorrect feedback: [Feedback for incorrect answer. For True/False questions: If correct answer is True, start with "Incorrect! This statement is true..." If correct answer is False, start with "Incorrect! This statement is false..." For other question types, start with "Incorrect!" and explain the correct answer content without mentioning option letters.]

</question>

<question>
Question no: 2
Question Type: [Question type here]
Question: [Question text here]

Option A: [Option A text here]
Option B: [Option B text here]
Option C: [Option C text here. Leave blank for True/False questions]
Option D: [Option D text here. Leave blank for True/False questions]

Correct Answer: [Correct Option letter eg. ""B""]

Correct feedback: [Feedback for correct answer. For True/False questions: If correct answer is True, start with "Correct! This statement is true..." If correct answer is False, start with "Correct! This statement is false..." For other question types, start with "Correct!" and explain why the chosen option content is correct without mentioning option letters.]

Incorrect feedback: [Feedback for incorrect answer. For True/False questions: If correct answer is True, start with "Incorrect! This statement is true..." If correct answer is False, start with "Incorrect! This statement is false..." For other question types, start with "Incorrect!" and explain the correct answer content without mentioning option letters.]

</question>

<question>
Question no: 3
Question Type: Select All That Apply
Question: [Question text here] (Select all that apply)

Option A: [Option A text here]
Option B: [Option B text here]
Option C: [Option C text here]
Option D: [Option D text here]

Correct Answer: [List all correct option letters separated by commas, e.g., "A, B, D"]

Correct feedback: Correct! [Acknowledge the multiple correct answers by referring to their actual content/text, not option letters. For example: "Proper ventilation, energy efficiency, and comfort control are indeed the key benefits because..." instead of "A, B, and C are correct because..."]

Incorrect feedback: Incorrect! [Explain all the correct answers by referring to their actual content/text, not option letters. For example: "The correct answers are proper ventilation, energy efficiency, and comfort control because..." instead of "A, B, and C are correct because..."]

</question>

<question>
Question no: 4
Question Type: Matching (This question type is optional)
Question: [Question text here]

Option A: [Option A text here. Leave blank if not applicable]
Option B: [Option B text here. Leave blank if not applicable]
Option C: [Option C text here. Leave blank if not applicable]
Option D: [Option D text here. Leave blank if not applicable]

Match 1: [Match 1 text here. Leave blank if not applicable]
Match 2: [Match 2 text here. Leave blank if not applicable]
Match 3: [Match 3 text here. Leave blank if not applicable]
Match 4: [Match 4 text here. Leave blank if not applicable]

Correct Answer:
Ensure that you give the actual Option text and Match text and not the option number and match number. Provide the matches in this format where applicable:
[Option A text here] - [Correct Match text here];
[Option B text here] - [Correct Match text here];
[Option C text here] - [Correct Match text here];
[Option D text here] - [Correct Match text here].

Correct feedback: Correct! You have correctly matched all the options with their respective choices.

Incorrect feedback: Incorrect! These are the correct match choices:
[Option A text here] - [Correct Match text here]; [Option B text here] - [Correct Match text here]; [Option C text here] - [Correct Match text here]; [Option D text here] - [Correct Match text here].
Ensure you provide all the exact option texts and their corresponding correct match texts without using option letters (A, B, C, D) in the feedback.
</question>

[Repeat such pattern for all the questions]

</output>
"""

# Function to generate assessment question
@traceable(
    metadata={
        "agent_name": "assessment",
        "step_name": "Generate Assessment Questions",
        "function_name": "generate_assessment_question",
        "user_id": st.session_state.get("role", "anonymous")
    }
)
def generate_assessment_question(course_name, target_audience, topic_name, slides, llm = 'gemini_2_flash'):
    """
    This function generates assessment questions for a given course.
    :param course_name: The name of the course.
    :param target_audience: The target audience for the course.
    :param topic_name: The name of the topic.
    :param slides: The slides for the topic.
    :param llm: The language model to use.
    :return: str - The generated assessment question.
    """
    generate_assessment_agent = Chain(llm=llm, tags=['question'])

    generate_assessment_agent.add_message(
        role='user',
        content=generate_assessment_prompt.format(
            course_name=course_name,
            topic_name=topic_name,
            target_audience=target_audience,
            slides=slides
        )
    )

    response = generate_assessment_agent.run()

    return response['question']

@traceable(
    metadata={
        "agent_name": "assessment",
        "step_name": "Generate Assessment Questions",
        "function_name": "run_generate_assessment_question",
        "user_id": st.session_state.get("role", "anonymous")
    }
)
def run_generate_assessment_question(sheet, worksheet_name, course_name, target_audience, llm = 'gemini_2_flash'):
    """
    This function generates assessment questions for a given course.
    :param sheet: The Google Sheet object.
    :param worksheet_name: The name of the worksheet.
    :param course_name: The name of the course.
    :param target_audience: The target audience for the course.
    :return: dict - The generated assessment questions by topic.
    """
    _, slide_chunks_df = get_sheet_data_and_df(sheet, worksheet_name)
    questions_by_topic = {}
    
    # Access or create the "Assessment questions" worksheet
    assessment_questions_sheet, assessment_questions_df = create_or_read_worksheet(sheet, "Assessment questions")
    
    # if the assessment questions sheet is already populated, skip processing
    if 'topic' not in assessment_questions_df.columns:
        assessment_questions_df['topic'] = ''
        assessment_questions_df['questions'] = ''
    else:
        # Check if the assessment questions sheet is already populated then skip processing
        if not assessment_questions_df.empty:
            print("Assessment questions sheet is already populated. Skipping processing.")
            return questions_by_topic    

    unique_topics = pd.unique(slide_chunks_df['Topic'])

    futures_map = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        for topic_name in unique_topics:
            topic_slides_data = slide_chunks_df[slide_chunks_df['Topic'] == topic_name]
            slides = "\n".join(
                topic_slides_data['Title'] + ": " + topic_slides_data['Content']
            )

            future = executor.submit(
                generate_assessment_question,
                course_name,
                target_audience,
                topic_name,
                slides,
                llm
            )

            futures_map[future] = topic_name

        total_tasks = len(futures_map)
        progress = SmartProgressBar(total_tasks=total_tasks, description="Percent complete", save_interval=5)

        for future in tqdm(as_completed(futures_map), total=total_tasks):
            topic_name = futures_map[future]
            response = future.result()
            questions_by_topic[topic_name] = response
            progress.update()

    # Convert the questions_by_topic dictionary to a DataFrame
    assessment_questions_df = pd.DataFrame((questions_by_topic.items()), columns=['topic', 'questions'])
    
    # Save the DataFrame to the 'Assessment questions' worksheet
    save_to_sheet(assessment_questions_sheet, assessment_questions_df)

    # Format the worksheet after saving
    format_worksheet(assessment_questions_sheet)

    #  Hide the worksheet after population
    hide_worksheet_by_name(sheet, "Assessment questions")

    return questions_by_topic


def delete_assessment_questions(sheet, worksheet_name="Assessment questions"):
    """Delete the Assessment questions worksheet."""
    delete_worksheet(sheet, worksheet_name)

