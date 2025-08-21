from agents.generate_assessments.chains import Chain
import re
from langsmith import traceable
import streamlit as st




revise_assessment_prompt = """You are an expert revision assistant tasked with revising an original assessment question for the course titled {course_name}, tailored to the {target_audience}, based on feedback received from a reviewing agent. Below is the slide content on which the original assessment question is based on:

<topic_slides>
{topic_slides}
</topic_slides>

This is the Original Assessment Question:
<original_question>
{original_question}
</original_question>

This is the feedback received from the reviewing agent for this Original Assessment Question:
<reviewer_feedback>
{reviewer_feedback}
</reviewer_feedback>

Follow the steps below to revise the assessment question:

1. Carefully analyze the reviewer feedback to identify specific areas for improvement.
2. Revise the assessment question strictly according to the reviewer feedback. Only address the issues raised — do not introduce unrelated changes. Reference the slide content where necessary to ensure the revised question aligns with the course material. Strictly avoid altering the question type while revising the question.
3. Provide the final revised question in the required question format.

Make sure to reply in the following output format:

<analysis>
Provide a brief explanation of your understanding of the reviewer feedback and the key areas you addressed during the revision.
</analysis>

<revised_question>
{question_format}
</revised_question>
(Strictly follow this question format while generating the revised question)
"""

# Question formats
multichoice_question_format = """
<question>
Question no: [Question number here]
Question Type: Multiple Choice
Question: [Question text here]

Option A: [Option A text here]
Option B: [Option B text here]
Option C: [Option C text here]
Option D: [Option D text here]

Correct Answer: [Correct Option letter eg. "B"]

Correct feedback: Correct! [Feedback for correct answer. Don't give information about slide content references while giving this feedback.]
Incorrect feedback: Incorrect! [Feedback for any incorrect answer. Don't give information about slide content references while giving this feedback.]

</question>
"""

truefalse_question_format = """
<question>

Question no: [Question number here]
Question Type: True/False
Question: [Question text here]

Option A: True
Option B: False

Correct Answer: [Correct Option letter eg. "A"]

Correct feedback: Correct! [If correct answer is True, start with "This statement is true." If correct answer is False, start with "This statement is false." Then provide feedback for correct answer. Don't give information about slide content references while giving this feedback.]
Incorrect feedback: Incorrect! [If correct answer is True, start with "This statement is true." If correct answer is False, start with "This statement is false." Then provide feedback for any incorrect answer. Don't give information about slide content references while giving this feedback.]

</question>
"""

matching_question_format = """
<question>

Question no: [Question number here]
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
[Ensure that you give the all the exact option text and its corresponding correct answer text and not the option number and its corresponding correct match number. Provide the matches in this format where applicable:
[Exact Option A text here] - [Exact Correct Answer text for Option A here. Leave blank if not applicable]\n
[Exact Option B text here] - [Exact Correct Answer text for Option B here. Leave blank if not applicable]\n
[Exact Option C text here] - [Exact Correct Answer text for Option C here. Leave blank if not applicable]\n
[Exact Option D text here] - [Exact Correct Answer text for Option D here. Leave blank if not applicable]

Correct feedback: Correct! You have correctly matched all the options with their respective choices.

Incorrect feedback: Incorrect! These are the correct match choices: A. Exact Option A text here - Exact Correct Answer text for Option A here. Leave blank if not applicable; B. Exact Option B text here - Exact Correct Answer text for Option B here. Leave blank if not applicable; C. Exact Option C text here - Exact Correct Answer text for Option C here. Leave blank if not applicable; D. Exact Option D text here - Exact Correct Answer text for Option D here. Leave blank if not applicable.
[Ensure you provide all the exact option texts and their corresponding correct answer texts, not just the option or match numbers.]
</question>
"""

select_all_question_format = """
<question>

Question no: [Question number here]
Question Type: Select All That Apply
Question: [Question text here]

Option A: [Option A text here]
Option B: [Option B text here]
Option C: [Option C text here]
Option D: [Option D text here]

Correct Answer: [List all correct option letters separated by commas, e.g., "A, B, D"]

Correct feedback: Correct! [Acknowledge the multiple correct answers in your feedback. Don't give information about slide content references while giving this feedback.]

Incorrect feedback: Incorrect! [Explain all the correct answers and why they are correct. Don't give information about slide content references while giving this feedback.]

</question>
"""

# Function to detect the type of an assessment question
def detect_question_type(assessment_question):
    """
    Detect the question type based on the "Question Type" field.

    Args:
        assessment_question (str): The assessment question text.

    Returns:
        str: The detected question type (Multiple Choice, True/False, Select All That Apply, Matching).
    """
    # Extract "Question Type:" from the question
    pattern = r"Question Type:\s*(.+)"
    match = re.search(pattern, assessment_question, re.IGNORECASE)

    if match:
        question_type = match.group(1).strip()
        # Normalize the extracted question type
        if question_type.lower() in {"multiple choice", "multiple-choice"}:
            return "Multiple Choice"
        elif question_type.lower() in {"true/false", "true-false"}:
            return "True/False"
        elif question_type.lower() in {"select all that apply", "select-all-that-apply"}:
            return "Select All That Apply"
        elif question_type.lower() == "matching":
            return "Matching"
        else:
            raise ValueError(f"Unrecognized question type: {question_type}")

    # Raise an error if "Question Type:" is missing
    raise ValueError("The 'Question Type:' field is missing from the provided question.")

# Function to get the correct output format dynamically
def get_question_format(question_type):
    if question_type == "Multiple Choice":
        return multichoice_question_format
    elif question_type == "True/False":
        return truefalse_question_format
    elif question_type == "Select All That Apply":
        return select_all_question_format
    elif question_type == "Matching":
        return matching_question_format
    else:
        raise ValueError("Invalid question type provided!")
    

# Function to revise assessment question
@traceable(
    metadata={
        "agent_name": "assessment",
        "step_name": "Review and Revise all Assessment Questions",
        "function_name": "revise_assessment",
        "user_id": st.session_state.get("role", "anonymous")
    }
)
def revise_assessment(course_name, target_audience, assessment_question, slides, reviewer_response, question_format, llm='gemini_2_flash'):
    """
    Revise an assessment question based on reviewer feedback.
    :param course_name: The name of the course.
    :param target_audience: The target audience for the course.
    :param assessment_question: The original assessment question.
    :param slides: The slides for the topic.
    :param reviewer_response: The feedback received from the reviewing agent.
    :param question_format: The format for the revised question.
    :param llm: The language model to use.
    :return: The revised assessment question.
    """
    revise_assessment_agent = Chain(llm=llm, tags=['analysis', 'revised_question'])

    revise_assessment_agent.add_message(
        role="user",
        content=revise_assessment_prompt.format(
            course_name=course_name,
            target_audience=target_audience,
            topic_slides=slides,
            original_question=assessment_question,
            reviewer_feedback=reviewer_response["text"],
            question_format=question_format
        )
    )

    reviser_response = revise_assessment_agent.run()
    return reviser_response['revised_question'][0]




