from modules.chain import Chain
from concurrent.futures import ThreadPoolExecutor
from langsmith import traceable
import streamlit as st

generate_complexity_review_prompt = """You are a Graphics Definition Review Agent. Your task is to analyze the provided graphics definition for a slide and evaluate its complexity. The goal is to ensure that the visualization is effective, easy to create, and not overly complex while maintaining clarity and instructional value.

Below is the course information for which you will be reviewing the graphics definition:
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

Follow the below guidelines while doing your review for complexity of the graphics definition:

1. Scene-Based Complexity Evaluation:
    Each scene in the graphics definition must be reviewed separately, but the final verdict will be for the entire slide. Consider the following aspects when evaluating complexity:

    - Use of Animations and Transitions: If the graphics definition includes animations, evaluate whether they enhance comprehension and engagement while remaining practical to implement. Animations should only be used when they meaningfully contribute to the instructional message. If an animation effectively conveys the intended meaning and is reasonable to create, it is acceptable. However, if a simpler animation or transition such as fade-in, slide-in, or scaling effects can achieve the same instructional impact without compromising clarity, it should be suggested as an alternative. If a scene relies on animations that are overly complex or require detailed object manipulation, you should try to identify a simpler approach, if possible, that maintains instructional effectiveness while ensuring ease of implementation. If no animations are used in a scene, no changes are required.
    - Balance Between Detail and Simplicity: The graphical representation should be detailed enough to support comprehension but not so intricate that it complicates the design process. Determine whether the level of detail aligns with the target audience and instructional objectives.

2. Evaluation Breakdown:
    Before providing the final verdict, conduct a structured evaluation of the graphics definition for all scenes in the slide. This breakdown should assess the entire visual representation based on the following aspects:
    - Animation vs Transition Evaluation: If animations are included in any scenes, assess whether they are necessary or if simpler transition effects could be used to achieve the same effect with less complexity.
    - Opportunities for Simplification: Identify specific ways in which the visual representation of the slide can be refined or simplified without losing instructional effectiveness.

3. Final Instructions:
    Before making your Pass/Fail verdict, carefully review the graphics definition to determine whether it is appropriately designed or overly complex. Use the following criteria to guide your evaluation:

    - Carefully analyze each scene within the graphics definition and identify any unnecessary complexity.
    - If the complexity for all the scenes is justified and maintains instructional effectiveness, provide your justification of how the scenes are not complex in the feedback section. In the suggestion section, state: "No changes needed" and give a verdict of "Pass".
    - If any of the scenes includes complex animations, or redundant graphics, provide your justification of how the scenes are complex and how they can be simplified in the feedback section. Provide specific suggestions for simplifying the visuals while ensuring clarity and instructional alignment and strictly give a verdict of "Fail".

Present your output in the following format:

<output>

<evaluation_breakdown>

- Key Observations: Summarize major complexity issues, if any.
- Animation vs Transition Evaluation: If the graphics definition includes animations, evaluate whether they are necessary, effective, and could be simplified. If no animations are used, state "No animations are used."
- Opportunities for Simplification: Suggest ways to reduce complexity while maintaining clarity.

</evaluation_breakdown>

<complexity_review>

<feedback>
If the scenes are complex, provide a detailed assessment of what aspects are too complex and need revision. If all the scenes are simple, explain why they are easy to implement and well-structured.
</feedback>

<verdict>
[Pass/Fail based on your analysis and evaluation]
</verdict>

<suggestion>
If the verdict is Fail, provide specific suggestions on how to simplify the scenes while maintaining instructional value. If Pass, state "No changes needed."
</suggestion>

</complexity_review>

</output>
"""

@traceable(
    metadata={
        "agent_name": "graphics_definition",
        "step_name": "Graphics Definition Generation",
        "function_name": "generate_complexity_review",
        "user_id": st.session_state.get("role", "anonymous")
    }
)
def generate_complexity_review(course_name, target_audience, slide_title, slide_content, graphics_definition, llm="gemini_2_flash"):
    """
    Generate complexity review for a given slide.

    :param course_name: The name of the course.
    :param target_audience: The target audience for the course.
    :param slide_title: The title of the slide.
    :param slide_content: The content of the slide.
    :param graphics_definition: The generated graphics definition for the slide.
    :param llm: The language model to be used.
    :return: The complexity review output.
    """
    
    def task():
        generate_complexity_review_agent = Chain(llm=llm, tags=['output'])
        generate_complexity_review_agent.add_message(
            role="user",
            content=generate_complexity_review_prompt.format(
                course_name=course_name,
                target_audience=target_audience,
                slide_title=slide_title,
                slide_content=slide_content,
                graphics_definition=graphics_definition
            )
        )
        response = generate_complexity_review_agent.run()
        return response['output']
    
    with ThreadPoolExecutor() as executor:
        future = executor.submit(task)
        return future.result()