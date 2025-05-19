from modules.chain import Chain
from langsmith import traceable
import streamlit as st

# Initialize the checklist sheet using the provided link
# checklist_sheet = gc.open_by_url(checklist_sheet_link)

# Load the checklist sheet data
# checklist_sheet, checklist_df = get_sheet_data_and_df(checklist_sheet, 'Graphics Definition Checklist')

generate_checklist_prompt = """You are a Checklist Evaluation Agent tasked with rigorously assessing the quality of a generated graphics definition based on a predefined checklist.

Below is the course information for which you will be doing the checklist evaluation:

<course_information>
Course name: {course_name}
Target audience: {target_audience}
</course_information>

Here is the slide for which the graphics definition was generated:
<slide>
Slide title: {slide_title}
Slide content:
{slide_content}
</slide>

Here is the graphics definition that was generated:
<graphics_definition>
{graphics_definition}
</graphics_definition>

Refer to the checklist criteria below for evaluating the generated graphics definition:

<checklist_criteria>
Task: {task_name}
Checklist Criteria:
{checklist_criteria}
</checklist_criteria>

For every checklist criterion, follow these steps:

1. Evaluation Breakdown: Think through and document your reasoning as to whether the checklist criterion is met. Reference the course name, target audience, slide content and the graphics definition to form your reasoning.
2. Final Verdict: Based on your reasoning, provide the verdict (Pass/Fail) for the review criterion.
3. Feedback: If the verdict is "Fail", explain your reasons for giving the negative verdict in a concise manner under the Feedback field.

Evaluation Guidelines:

- Do your evaluation of the graphics definition on all the provided checklist criteria.
- Do not alter or modify the phrasing of the checklist criteria in any way.
- Ensure that your evaluation for every checklist criterion is objective and unbiased.
- Include a "Evaluation Breakdown" field for every checklist criterion in the final output. This section should clearly document the thought process behind your verdict.
- Based on your evaluation, provide a verdict of "Pass" or "Fail" inside the "Final Verdict" field for every checklist criterion, without adding explanations, interpretations, or additional commentary. If the checklist criterion is satisfied, your Final Verdict will be "Pass" and if the checklist criterion is not satisfied, your Final Verdict will be "Fail".
- If the Final Verdict is "Pass", include only the fields for Slide Title, Checklist Criterion, Evaluation Breakdown, and Final Verdict in your output. If the Final Verdict is "Fail", include an additional field, "Feedback", to explain the negative verdict concisely.

Output Format:

Provide your output strictly in the following format:

<checklist_evaluation>

<output>
Slide Title: [Enter the Slide Title text as it is]
Checklist Criterion: [Enter the Checklist Criterion text that is being evaluated as it is without any modification]
Evaluation Breakdown: [Provide your reasoning here regarding whether the checklist criterion has been met]
Final Verdict: [Enter your final verdict- Pass or Fail]
Feedback: [Provide your explaination if your final verdict is "Fail". Omit this field entirely if your verdict is "Pass"]

[Repeat the above pattern for all the checklist criteria]
</output>

</checklist_evaluation>
"""

@traceable(
    name="generate_checklist_evaluation",
    metadata={
        "agent_name": "graphics_definition",
        "step_name": "Checklist Evaluation",
        "function_name": "generate_checklist_evaluation",
        "user_id": st.session_state.get("role", "anonymous")
    }
)
def generate_checklist_evaluation(course_name, target_audience, slide_title, slide_content, graphics_definition, task_name, checklist_criteria, llm = "gemini_2_flash"):
    """
    Generate checklist evaluation for a given slide's graphics definition.

    :param course_name: The name of the course.
    :param target_audience: The target audience for the course.
    :param slide_title: The title of the slide.
    :param slide_content: The content of the slide.
    :param graphics_definition: The revised graphics definition for the slide.
    :param checklist_criteria: The checklist criteria used for evaluation.
    :param llm: The language model to use (default: "gemini_2_flash").
    :return: The generated checklist evaluation.
    """

    # Initialize the agent
    checklist_agent = Chain(llm = llm, tags = ["checklist_evaluation"])

    # Add the user message
    checklist_agent.add_message(
        role = "user",
        content = generate_checklist_prompt.format(
            course_name = course_name,
            target_audience = target_audience,
            slide_title = slide_title,
            slide_content = slide_content,
            graphics_definition = graphics_definition,
            task_name = task_name,
            checklist_criteria = checklist_criteria
        )
    )

    # Run the agent
    response = checklist_agent.run()

    return response["checklist_evaluation"]