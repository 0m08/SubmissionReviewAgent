from pydantic import BaseModel, Field
from typing import Literal, List 
#Pydantic function for structured output

#Multiple Choice Question
class MultiChoiceQuestion(BaseModel):
    question_type: str = Field(default="multichoice", description="The type of the question. Always set to 'multichoice' for multiple-choice questions.")
    question_text: str = Field(description="The question text.")
    option_a: str = Field(description="Text for Option A")
    option_b: str = Field(description="Text for Option B")
    option_c: str = Field(description="Text for Option C")
    option_d: str = Field(description="Text for Option D")
    correct_answer: Literal["A", "B", "C", "D"] = Field(description="The label of the correct answer, must be one of 'A', 'B', 'C', 'D'. This corresponds to the correct answer.")
    correct_feedback: str = Field(description="Correct feedback for the answer. Should be positive and reinforce the correct choice.")
    incorrect_feedback: str = Field(description="Incorrect feedback for the answer. Should briefly explain why the correct answer is correct without mentioning slide content explicitly.")

# True/False Question
class TrueFalseQuestion(BaseModel):
    question_type: str = Field(default="truefalse", description="The type of the question. Always set to 'truefalse' for true/false questions.")
    question_text: str = Field(description="The question text.")
    option_a: str = Field(default="True", description="Option A for True.")
    option_b: str = Field(default="False", description="Option B for False.")
    correct_answer: Literal["A", "B"] = Field(description="The label of the correct answer, must be one of 'A', 'B'. This corresponds to the correct answer.")
    correct_feedback: str = Field(description="Correct feedback for the answer. Should be positive and reinforce the correct choice.")
    incorrect_feedback: str = Field(description="Incorrect feedback for the answer. Should briefly explain why the correct answer is correct without mentioning slide content explicitly.")

# Matching Question
class MatchingPair(BaseModel):
    """Represents a subquestion-answer pair for matching-type questions."""
    subquestion: str = Field(description="The option text. Should be a term, phrase, or statement to be matched. It must correspond directly to the provided option in the question. The subquestion text must not contain the option numbers, letters, or additional formatting. It should only include the option text.")
    answer: str = Field(description="The correct match text. This should represent the related term, phrase, or explanation that matches the subquestion. It must correspond directly to the correct match text. The answer text must not contain any option numbers, letters, or additional formatting. It should only include the match text.")

class MatchingQuestion(BaseModel):
    question_type: str = Field(default="matching", description="The type of the question. Always set to 'matching' for matching-type questions.")
    question_text: str = Field(description="The question text. It should instruct the learner to match subquestions with their correct answers.")
    options: List[MatchingPair] = Field(
        description="A list of subquestion-answer pairs for the matching question. Each pair must have:"
                    "- The subquestion: A term, phrase, or statement that corresponds to the provided option text."
                    "- The answer: The correct match text."
                    "Strictly maintain the order inside the list such that the subquestion is always the option text and the answer is always the correct match text"

                    "Example:"
                    "Question: Match the humidity term with its description:"

                    "Option A: Relative Humidity"
                    "Option B: Absolute Humidity"
                    "Option C: Dew Point"
                    "Option D: Saturation"

                    "Match 1: The temperature at which the air becomes saturated with water vapor."
                    "Match 2: The amount of water vapor present in the air compared to the maximum it could hold at that temperature."
                    "Match 3: The state where air is holding the maximum amount of water vapor it can at a given temperature"
                    "Match 4: The mass of water vapor in a given volume of air."

                    "MatchingPair(subquestion='Relative Humidity', answer='The amount of water vapor present in the air compared to the maximum it could hold at that temperature'),"
                    "MatchingPair(subquestion='Absolute Humidity', answer='The mass of water vapor in a given volume of air'),"
                    "MatchingPair(subquestion='Dew Point', answer='The temperature at which the air becomes saturated with water vapor'),"
                    "MatchingPair(subquestion='Saturation', answer='The state where air is holding the maximum amount of water vapor it can at a given temperature')"
    )
    correct_answer: str = Field(
        description="The correct matches provided as a single string, formatted as: "
                    "Exact Option A text here - Exact Correct Answer text for Option A here\n"
                    "Exact Option B text here - Exact Correct Answer text for Option B here\n"
                    "Exact Option C text here - Exact Correct Answer text for Option C here\n"
                    "Exact Option D text here - Exact Correct Answer text for Option D here"

                    "Strictly ensure that you give all the actual option texts and their corresponding correct answer texts. It must not have the option numbers and their corresponding correct match numbers."

                    "Example:"
                    "Low Humidity - Comfortable for most people\n"
                    "Moderate Humidity - Generally comfortable, slight stickiness possible\n"
                    "High Humidity - Discomfort due to excessive moisture in the air\n"
                    "Very High Humidity - Oppressive and can lead to heat-related illnesses"
    )
    correct_feedback: str = Field(
        description="Correct feedback for the matching question. Should only say this-'Correct! You have correctly matched all the options with their respective matches.' "
    )
    incorrect_feedback: str = Field(
        description="Incorrect feedback for the matching question. Should say this-'Incorrect! These are the correct matches:' and then give all the actual option texts and its corresponding correct answer texts."
                    "Here's how it should look like - "
                    "Incorrect! These are the correct matches: A. Exact Option A text here - Exact Correct Answer text for Option A here; B. Exact Option B text here - Exact Correct Answer text for Option B here; C. Exact Option C text here - Exact Correct Answer text for Option C here; D. Exact Option D text here - Exact Correct Answer text for Option D here."

                    "Example:"
                    "Incorrect! These are the correct matches: A. Low Humidity - Comfortable for most people; B. Moderate Humidity - Generally comfortable, slight stickiness possible; C. High Humidity - Discomfort due to excessive moisture in the air; D. Very High Humidity - Oppressive and can lead to heat-related illnesses."
    )