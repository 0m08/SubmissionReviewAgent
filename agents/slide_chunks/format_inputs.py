import re
from pydantic import BaseModel, Field
from typing import List 

#  Function to strip Roman numerals from the topic
def strip_roman_numerals(text):
    return re.sub(r"^[IVXLCDM]+\.\s*", "", text).strip()

# Function to strip "Section X:" from the subtopic
def strip_section_prefix(text):
    return re.sub(r"^Section\s+([IVXLCDM]+|\d+)\b[:\s-]*", "", text, flags=re.IGNORECASE).strip()

def extract_failed_criteria(evaluation_output):
    """
    Extract only the failed review criteria blocks from the evaluation output.

    :param evaluation_output: The full structured review output containing all criteria.
    :return: A string containing only the failed criteria blocks, or an empty string if all criteria pass.
    """
    criteria_pattern = re.compile(
        r"<criterion>\s*Review Criterion: (.*?)\s*Feedback: (.*?)\s*Verdict: (Pass|Fail)\s*Suggestion: (.*?)\s*</criterion>",
        re.DOTALL
    )

    failed_criteria_blocks = []
    matches = criteria_pattern.findall(evaluation_output)

    for criterion, feedback, verdict, suggestion in matches:
        if verdict.strip().lower() == "fail":
            failed_criteria_blocks.append(f"<criterion>\nReview Criterion: {criterion}\nFeedback: {feedback.strip()}\nVerdict: {verdict}\nSuggestion: {suggestion.strip()}\n</criterion>")

    return "\n\n".join(failed_criteria_blocks) if failed_criteria_blocks else ""



# Pydantic object to get structured output
class Slide(BaseModel):
    slide_title: str = Field(description="The title of the slide.")
    slide_content: str = Field(description="The content/body of the slide.")

class Subtopic(BaseModel):
    # The name of the subtopic can be the same as its transition slide's title
    subtopic_name: str = Field(description="Name of the subtopic.")
    transition_slide: Slide = Field(description="Transition slide for this subtopic.")
    content_slides: List[Slide] = Field(
        description="List of content slides under this subtopic."
    )
    summary_slide: Slide = Field(description="Summary slide for this subtopic.")