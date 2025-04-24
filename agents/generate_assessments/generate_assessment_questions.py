from agents.generate_assessments.chains import Chain
from services.sheets_service import get_sheet_data_and_df, format_worksheet,create_or_read_worksheet, save_to_sheet
import pandas as pd
from tqdm import tqdm
from services.smart_progress_bar import SmartProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed

 

generate_assessment_prompt = """As an expert Instructional Designer with extensive knowledge of the subject {course_name} and the topic {topic_name}, your task is to create tailored assessment questions for a self-paced e-learning course on {course_name}, specifically designed for {target_audience}. Please develop assessment questions based on the content provided in the following slides, which are delimited by XML tags.

<slides>
{slides}
</slides>

Create questions from the following question types, delimited by XML tags:
<assessment_questions_types>
Question Type, Description
Multiple Choice: allows the selection of a single response from a pre-defined list.
True/False: a simple form of multiple choice question with just the two choices 'True' and 'False'.
Matching: the answer to each of a number of sub-questions must be selected from a list of possibilities.
</assessment_question_types>

Follow these guidelines when creating the questions.
- Ensure exhaustive and proportional coverage of the course content, including a mix of first-level and second-level questions, with thoughtful answer choices.
- Any field that does not apply to a given question type should be left empty. For instance, True/False question type will have options C and D left empty.
- For Matching question types, give the exact correct answer under the ""Correct Answer"" field. Match correctly without using Options. Use the exact option text and words being matched. Ensure that all the option text and match text for matching type questions are unique.
- The correct feedback will be shown everytime the user answers correctly. The incorrect feedback is common feedback that will be shown anytime a user does not answer the question correctly and it should explain the correct answer.
- Correct and incorrect feedback must be provided for all question types, including matching.
- For Multiple Choice and True/False questions, simply indicate the correct answer (e.g., ""B"") in the ""Correct Answer"" field.

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

Correct feedback: [Feedback for correct answer. Always start this feedback with this word - Correct!]

Incorrect feedback: [Feedback for incorrect answer. Always start this feedback with this word - Incorrect!]

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

Correct feedback: [Feedback for correct answer. Always start this feedback with this word - Correct!]

Incorrect feedback: [Feedback for incorrect answer. Always start this feedback with this word - Incorrect!]

</question>

<question>
Question no: 3
Question Type: Matching
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
A. [Option A text here] - [Correct Match text here]; B. [Option B text here] - [Correct Match text here]; C. [Option C text here] - [Correct Match text here]; D. [Option D text here] - [Correct Match text here].
Ensure you provide all the exact option texts and their corresponding correct match texts-not just the option or match numbers.
</question>

[Repeat such pattern for all the questions]

</output>
"""

# Function to generate assessment question
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
                topic_slides_data['final_slide_title'] + ": " + topic_slides_data['final_slide_content']
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
    
    return questions_by_topic