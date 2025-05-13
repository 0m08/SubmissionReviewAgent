from modules.chain import Chain
from concurrent.futures import ThreadPoolExecutor
from langsmith import traceable
import streamlit as st

generate_accuracy_review_prompt = """You are a Graphics Definition Review Agent. Your task is to analyze the provided graphics definition for a slide and evaluate whether it accurately represents the slide content. The goal is to ensure that all technical details, equipment, tools, measurements, and concepts are visualized correctly and without any misrepresentation.

Below is the course information for which you will be performing the accuracy review:
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

Follow the below guidelines while conducting the accuracy review:

1. Scene-Based Accuracy Evaluation:
    Each scene in the graphics definition must be reviewed separately, but the final verdict will be for the entire slide:

    A) Verification of Technical Accuracy:

    - Verify that all technical equipment, instruments, devices, tools, and measurement systems, etc. (e.g., gauges, sensors, meters, scales, thermometers, etc.) are accurately depicted in terms of structure, function, and intended use. Ensure that they are correctly labeled and that no incorrect or misleading representations are present in the graphics.
    - If any technical equipment, instrument, device, tool, or measurement system, etc. has multiple variations, confirm that the correct type is represented based on the context of the course, target audience, and slide content.
    - If there is ambiguity about which version should be used, flag it for further clarification rather than assuming a generic depiction.

    B) Alignment with Slide Content:

    - Ensure that all visual elements align with the instructional message conveyed in the slide content.
    - The graphics should represent the key concepts accurately without altering or oversimplifying the meaning.
    - Verify that all symbols, units, measurements, etc., used in the visuals are correct, properly labeled, and appropriate for the context. Ensure that no incorrect, inconsistent, or misleading representations are present.

2. Evaluation Breakdown:
    Before providing the final verdict, conduct a structured analysis to ensure a comprehensive evaluation of accuracy. This breakdown should cover the following aspects:

    - Technical Equipment and Tools Check: Ensure that all technical equipment, instruments, devices, tools, and measurement systems, etc. (e.g., gauges, meters, thermometers, etc.) are correctly depicted in terms of structure, function, and intended use. Verify that labels are accurate and representations are not misleading.
    - Slide Content Alignment: Confirm that the graphics align with the instructional message of the slide and accurately represent key concepts without oversimplification or misinterpretation.
    - Symbols, Units, and Measurements Validation: Check that all symbols, units, numerical values, labels, etc. are correct, properly formatted, and contextually appropriate.
    - Ambiguity and Clarifications Needed: Identify any unclear visual elements or cases where the correct variation of a tool or device is uncertain. Flag these for further clarification instead of assuming a generic depiction.

3. Final Instructions:
    Before making your verdict of Pass or Fail, carefully review the entire graphics definition to ensure that it accurately represents the slide content. The evaluation should be conducted for all scenes together to ensure coherence, consistency, and accuracy across the full slide. Use the following criteria to guide your evaluation:

    - Carefully analyze all scenes within the graphics definition together before making a judgment on overall accuracy.
    - If all visual elements across the entire graphics definition accurately represent the slide content, provide justification of how all the visual elements are accurate in the feedback section. In the suggestion section, state: "No changes needed" and give a verdict of "Pass".
    - If any inaccuracies are found in any part of the graphics definition, give your justification of the inaccuracy in the feedback section. In the suggestion section, give specific recommendations on how to correct the inaccuracies while maintaining clarity and instructional alignment, and strictly give a verdict of "Fail".

Present your output in the following format:

<output>

<evaluation_breakdown>

- Technical Equipment and Tools Check: Ensure that all technical equipment, instruments, devices, tools, and measurement systems, etc. (e.g., gauges, meters, thermometers, etc.) are correctly depicted in terms of structure, function, and intended use. Verify that labels are accurate and representations are not misleading.
- Slide Content Alignment: Confirm that the graphics align with the instructional message of the slide and accurately represent key concepts without oversimplification or misinterpretation.
- Symbols, Units, and Measurements Validation: Check that all symbols, units, numerical values, labels, etc. are correct, properly formatted, and contextually appropriate.
- Ambiguity and Clarifications Needed: Identify any unclear visual elements or cases where the correct variation of a tool or device is uncertain. Flag these for further clarification instead of assuming a generic depiction.

</evaluation_breakdown>

<accuracy_review>

<feedback>
If any inaccuracies are found, specify which elements are incorrect and why they need correction. If all the elements are accurate, provide justification for how each element is deemed accurate.
</feedback>

<verdict>
[Pass/Fail based on your analysis and evaluation]
</verdict>

<suggestion>
If Fail, provide specific recommendations on how to correct the inaccuracies while maintaining clarity and instructional alignment. If Pass, only state: "No changes needed."
</suggestion>

</accuracy_review>

</output>
"""


@traceable(
    metadata={
        "agent_name": "graphics_definition",
        "step_name": "Graphics Definition Generation",
        "function_name": "generate_accuracy_review",
        "user_id": st.session_state.get("role", "anonymous")
    }
)
def generate_accuracy_review(course_name, target_audience, slide_title, slide_content, graphics_definition, llm="gemini_2_flash"):
    """
    Generate accuracy review for a given slide.

    :param course_name: The name of the course.
    :param target_audience: The target audience for the course.
    :param slide_title: The title of the slide.
    :param slide_content: The content of the slide.
    :param graphics_definition: The generated graphics definition for the slide.
    :param llm: The language model to use.
    :return: The accuracy review output.
    """
    
    def task():
        generate_accuracy_review_agent = Chain(llm=llm, tags=['output'])
        generate_accuracy_review_agent.add_message(
            role="user",
            content=generate_accuracy_review_prompt.format(
                course_name=course_name,
                target_audience=target_audience,
                slide_title=slide_title,
                slide_content=slide_content,
                graphics_definition=graphics_definition
            )
        )
        response = generate_accuracy_review_agent.run()
        return response['output']
    with ThreadPoolExecutor() as executor:
        future = executor.submit(task)
        return future.result()
