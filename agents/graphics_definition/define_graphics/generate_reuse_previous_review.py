from modules.chain import Chain
from concurrent.futures import ThreadPoolExecutor
from langsmith import traceable
import streamlit as st

generate_reuse_previous_graphics_review_prompt = """You are a Graphics Definition Review Agent. Your task is to critically analyze the graphics definition generated for a slide and determine whether any opportunities to reuse previously defined visual elements from earlier slides' graphics definitions were overlooked. Your objective is to identify relevant and beneficial opportunities to reuse graphics from earlier slides to enhance visual consistency and instructional alignment, only where it makes sense and does not compromise clarity or effectiveness.

Below is the course information for which you will be performing the review of graphics reuse opportunities:
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

Carefully analyze the graphics definition by comparing it against the previous slides' graphics definitions. Do not assume anything that is not explicitly stated in the previous graphics definitions. Only flag reuse opportunities if a relevant Scene ID from a previous slide contains visual elements that should have been reused or adapted. Reuse is not mandatory, so do not force it. Your goal is simply to identify meaningful reuse opportunities — only if they exist — that could enhance consistency and efficiency without compromising clarity or instructional effectiveness. Follow the below guidelines while conducting your review:

1. Identify Potential Reuse Opportunities:

    - Compare the current slide’s graphics definition with previous slides' definitions.
    - Look for specific visual elements - such as icons, diagrams, animations, layouts, etc. - that could have been reused or adapted instead of creating entirely new visuals.
    - Do not suggest reusing complete scenes or visual elements if the slide content is entirely different — only flag opportunities where visual elements could have been reused.
    - If no suitable or beneficial reuse can be found, do not force reuse. It is acceptable to conclude that no reuse opportunities exist for the slide.

2. Assess Consistency Across Slides:

    - Determine whether similar concepts were visualized differently in this slide compared to earlier slides without a clear instructional or visual benefit to doing things differently.
    - If a previously established visual structure (e.g., a commonly used symbol set, a standard diagram style, or a recurring animation approach) was not maintained where it logically could have been, consider whether consistent reuse might have helped. If so, mention it — otherwise, reuse is not required.
    - Ensure that visual elements remain cohesive across slides, avoiding unnecessary variations in style when consistency would clearly improve comprehension.

3. Evaluation Breakdown:
    Before providing the final verdict, conduct a structured analysis of the graphics definition to ensure an informed evaluation. This breakdown must cover the following aspects:

    - Missed Reuse Opportunities: Identify any visual elements from previous slides' graphics definitions that could have been reused but were not, only if their reuse would have been relevant and beneficial. Explicitly reference the Scene ID from which they should have been reused.
    - Justification for New Visuals: Assess whether the newly created visuals were necessary due to unique slide content. If a new visual was introduced where reuse was possible, explain why an existing element would have been a better choice.
    - Impact on Visual Consistency: Determine whether the absence of reuse has led to inconsistencies across slides. Identify whether maintaining previously established visual elements would have enhanced coherence and comprehension.
    - Adaptation vs. Direct Reuse: If a previous visual element could have been modified instead of creating an entirely new one, only highlight this as a missed opportunity if that adaptation would have clearly improved visual consistency or efficiency without reducing clarity or instructional effectiveness.

4. Final Instructions:
    Before making your Pass/Fail verdict, carefully review the graphics definition and determine whether any reuse opportunities were missed. Use the following criteria to guide your evaluation:

    - If there are no previous graphics definitions available, automatically provide a verdict of "Pass" with the suggestion "No changes needed".
    - If previous graphics definitions exist but none are relevant or beneficial for reuse, it is acceptable to conclude that no reuse opportunities exist. In this case, provide a verdict of "Pass" with the suggestion "No changes needed."
    - If all relevant reuse opportunities were correctly identified and applied, or if reuse was not possible due to the uniqueness of the slide content, provide justification in the feedback section, explaining why reuse was either effectively implemented or not applicable. In the suggestion section, state: "No changes needed" and give a verdict of "Pass".
    - If reuse opportunities were missed, provide a detailed explanation in the feedback section, specifying which visual elements should have been reused (e.g., icons, diagrams, animations, layouts), the Scene ID from which they should have been reused, and how they could have been incorporated. Clearly explain why their reuse would improve visual consistency and efficiency. In the suggestion section, provide specific recommendations on how to revise the graphics definition to incorporate the missed reuse opportunities and strictly give a verdict of "Fail".

Present your output in the following format:

<output>

<evaluation_breakdown>

- Missed Reuse Opportunities: Identify any reusable elements that were overlooked.
- Justification for New Visuals: Assess whether the newly created visuals were necessary.
- Impact on Visual Consistency: Determine whether the lack of reuse caused inconsistencies.
- Adaptation vs. Direct Reuse: Highlight cases where previous graphics could have been modified instead of replaced.

</evaluation_breakdown>

<reuse_graphics_review>

<feedback>
Provide a detailed assessment of whether any reuse opportunities were missed or if reuse was applied effectively.
</feedback>

<verdict>
[Pass/Fail based on your analysis and evaluation]
</verdict>

<suggestion>
If Fail, provide specific recommendations on how to incorporate reusable elements while ensuring clarity and consistency. If Pass, state "No changes needed."
</suggestion>

</reuse_graphics_review>

</output>
"""


generate_reuse_previous_graphics_review_system_prompt = """ You are a Graphics Opportunity Review Agent. Your task is to analyze the provided graphics definition for a slide and determine if any opportunities to reuse graphics from previous slides were missed.

Here is the list of graphics definitions from previous slides:
<previous_graphics_definition>
{previous_graphics_definition}
</previous_graphics_definition>

Refer to these previous graphics definitions for any visual elements that could potentially be reused. Do not assume any information beyond what is explicitly stated in these previous slides' graphics definition. Only flag reuse opportunities if any relevant scene from these previous definitions contains elements that align with the current slide’s content. If none are relevant or beneficial, do not force reuse.
"""

@traceable(
    metadata={
        "agent_name": "graphics_definition",
        "step_name": "Graphics Definition Generation",
        "function_name": "generate_reuse_previous_graphics_review",
        "user_id": st.session_state.get("role", "anonymous")
    }
)
def generate_reuse_previous_graphics_review(course_name, target_audience, slide_title, slide_content, graphics_definition, previous_graphics_definition, llm="gemini_2_flash"):
    """
    Generate reuse graphics opportunity review for a given slide.

    :param course_name: The name of the course.
    :param target_audience: The target audience.
    :param slide_title: The title of the slide.
    :param slide_content: The content of the slide.
    :param graphics_definition: The graphics definition generated for the slide.
    :param previous_graphics_definition: The graphics definitions from previous slides.
    :param llm: The language model to use.
    :return: The reuse graphics review output.
    """
    
    def task():
        reuse_previous_graphics_review_agent = Chain(llm=llm, tags=['output'])
        reuse_previous_graphics_review_agent.add_message(
            role="system",
            content=generate_reuse_previous_graphics_review_system_prompt.format(
                previous_graphics_definition=previous_graphics_definition
            )
        )
        reuse_previous_graphics_review_agent.add_message(
            role="user",
            content=generate_reuse_previous_graphics_review_prompt.format(
                course_name=course_name,
                target_audience=target_audience,
                slide_title=slide_title,
                slide_content=slide_content,
                graphics_definition=graphics_definition
            )
        )
        response = reuse_previous_graphics_review_agent.run()
        return response['output']
    

    with ThreadPoolExecutor() as executor:
        future = executor.submit(task)
        return future.result()
