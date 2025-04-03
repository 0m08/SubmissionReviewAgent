from modules.chain import Chain
from tqdm import tqdm
from services.sheets_service import get_sheet_data_and_df, save_to_sheet
import pandas as pd
from services.smart_progress_bar import SmartProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.helper_functions import get_topic_outline


add_los_to_outline_prompt = """Your task is to incorporate potentially missing LOs into the course outline.

Here's the info you will be working with:
<course_info>
Course Name: {course_name}
Target Audience: {target_audience}

Course Outline: 
{course_outline}
</course_info>

Here's the set of potential learning objectives to incorporate:
<potential_learning_objectives>
{potential_learning_objectives}
</potential_learning_objectives>

Each LO has be categorized properly. Incorporate the LOs that are not already covered and are relevant.

Output in the following format:
<objective_analysis>
[Place analysis of the potential learning objectives here]
<objective_analysis>
<revised_outline>
[Revised Outline in the same format as the original course outline]
<revised_outline>

Make sure to reply in the proper format.
"""


def add_los_to_outline(course_name, target_audience, course_outline, potential_learning_objectives, llm = "gemini_2_flash"):
    """
    Incorporates missing learning objectives into the course outline.
    
    :param course_name (str): Name of the course.
    :param target_audience (str): Target audience for the course.
    :param course_outline (str): Course outline in text format.
    :param potential_learning_objectives (str): Potential learning objectives to incorporate.
    :param llm (str): Language model to use (default: 'gemini_2_flash').
    :returns: str: Revised course outline with missing learning objectives incorporated.
    """
    agent = Chain(llm = llm, tags = ["revised_outline"])

    agent.add_message(
        role = "user", content = add_los_to_outline_prompt.format(
            course_name = course_name,
            target_audience = target_audience,
            course_outline = course_outline,
            potential_learning_objectives = potential_learning_objectives
        )
    )

    response = agent.run()

    return response["revised_outline"]


