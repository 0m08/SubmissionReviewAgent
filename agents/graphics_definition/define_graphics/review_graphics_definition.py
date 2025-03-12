from modules.chain import Chain
from concurrent.futures import ThreadPoolExecutor

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



generate_missing_sentences_review_prompt = """You are a Graphics Definition Review Agent. Your task is to analyze the provided graphics definition for a slide and evaluate whether all sentences from the slide content are fully represented in the visuals. The goal is to ensure that every sentence from the slide content is visually represented in the graphics definition, without any omissions.

Below is the course information for which you will be performing the missing sentences review:
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

Follow the below guidelines while conducting the missing sentences review:

1. Full sentence coverage across all scenes:
    Ensure that every sentence from the slide content is visually represented across the entire graphics definition, not just within individual scenes. Before marking any sentence as missing, review all scenes together to determine if it is represented somewhere. A sentence should only be marked as missing if it is completely absent from all scenes. When analyzing missing sentences, consider the following aspects:

    - Identifying Missing Sentences: Ensure that every sentence from the slide content is explicitly represented in the graphics definition. If a sentence is fully missing, flag it.
    - Handling Transition, Introductory, and Summary Sentences: Even if a sentence does not introduce new information, if it serves as an introduction, transition, or summary, it must still be represented visually. If these sentences are completely missing, suggest minimal visuals such as a title, simple icon, or transition effect to maintain context.

2. Evaluation Breakdown:
    Before providing the final verdict, conduct a structured analysis of the graphics definition to ensure an informed evaluation. This breakdown must cover the following aspects:

    - Missing Sentences Check: Identify whether all sentences from the slide content are visually represented.
    - Handling Introductory, Transition, and Summarizing Sentences: If the slide content contains introductory, transition, or summarizing sentences, ensure they are visualized appropriately.
    - Partial Representation Issues: Flag any sentences that are only partially depicted.

3. Final Instructions:
    Before making your verdict of pass or fail, carefully review the graphics definition as a whole, ensuring that every sentence is accounted for across all scenes. Use the following criteria to guide your evaluation:

    - Review all scenes together before determining if a sentence is missing. A sentence should only be considered missing if it is completely absent across all scenes. If a sentence is depicted in another scene, do not flag it as missing. Instead, confirm that it is visually represented somewhere in the graphics definition.
    - If all sentences are fully represented, give a verdict of Pass and state: "No changes needed"
    - The review should focus only on evaluating whether all sentences are visualized and should not suggest entirely new visuals.
    - If any sentence is not visually represented across scenes in the graphics definition, give a verdict of Fail and provide specific suggestions for incorporating the missing sentences into the graphics definition while ensuring clarity and instructional alignment.

Present your output in the following format:

<output>

<evaluation_breakdown>

- Missing Sentences Check: Identify whether all sentences from the slide content are visually represented.
- Handling Introductory and Transition Sentences: Ensure they are visualized appropriately.
- Partial Representation Issues: Flag any sentences that are only partially depicted.
- Visual Relevance and Order: Identify if any sentence’s visual appears too early or too late.

</evaluation_breakdown>

<missing_sentences_review>

<feedback>
If any sentence is missing, specify which one. If all sentences are represented, state "No feedback"
</feedback>

<verdict>
[Pass/Fail based on your analysis and evaluation]
</verdict>

<suggestion>
If Fail, provide specific recommendations on how to incorporate the missing sentences while maintaining instructional clarity. If Pass, state "No changes needed."
</suggestion>

</missing_sentences_review>

</output>
"""



def generate_missing_sentences_review(course_name, target_audience, slide_title, slide_content, graphics_definition, llm="gemini_2_flash"):
    """
    Generate missing sentences review for a given slide.

    :param course_name: The name of the course.
    :param target_audience: The target audience.
    :param slide_title: The title of the slide.
    :param slide_content: The content of the slide.
    :param graphics_definition: The graphics definition generated for the slide.
    :param llm: The language model to use.
    :return: The missing sentences review output.
    """
    
    def task():
        missing_sentences_review_agent = Chain(llm=llm, tags=['output'])
        missing_sentences_review_agent.add_message(
            role="user",
            content=generate_missing_sentences_review_prompt.format(
                course_name=course_name,
                target_audience=target_audience,
                slide_title=slide_title,
                slide_content=slide_content,
                graphics_definition=graphics_definition
            )
        )
        response = missing_sentences_review_agent.run()
        return response['output']
    
    with ThreadPoolExecutor() as executor:
        future = executor.submit(task)
        return future.result()



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



generate_reuse_previous_graphics_review_prompt = """You are a Graphics Definition Review Agent. Your task is to critically analyze the graphics definition generated for a slide and determine whether any opportunities to reuse previously defined visual elements from earlier slides' graphics definitions were overlooked. Your objective is to maximize the reuse of graphics wherever possible to enhance visual consistency and instructional alignment while maintaining clarity and effectiveness in conveying the slide content.

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

Carefully analyze the graphics definition by comparing it against the previous slides' graphics definitions. Do not assume anything that is not explicitly stated in the previous graphics definitions. Only flag reuse opportunities if a relevant Scene ID from a previous slide contains visual elements that should have been reused or adapted. Your goal is not to force reuse but to identify opportunities where previously defined visual components can be reused to enhance consistency and efficiency while maintaining clarity and instructional effectiveness. Follow the below guidelines while conducting your review:

1. Identify Potential Reuse Opportunities:

    - Compare the current slide’s graphics definition with previous slides' definitions.
    - Look for specific visual elements - such as icons, diagrams, animations, layouts, etc. - that could have been reused or adapted instead of creating entirely new visuals.
    - Do not suggest reusing complete scenes or visual elements if the slide content is entirely different — only flag opportunities where visual elements could have been reused.

2. Assess Consistency Across Slides:

    - Determine whether similar concepts were visualized differently in this slide compared to earlier slides without a strong reason for deviation.
    - If a previously established visual structure (e.g., a commonly used symbol set, a standard diagram style, or a recurring animation approach) was not maintained where it logically could have been, highlight it as a missed opportunity for reuse.
    - Ensure that visual elements remain cohesive across slides, avoiding unnecessary variations in style when consistency would improve comprehension.

3. Evaluation Breakdown:
    Before providing the final verdict, conduct a structured analysis of the graphics definition to ensure an informed evaluation. This breakdown must cover the following aspects:

    - Missed Reuse Opportunities: Identify any visual elements from previous slides' graphics definitions that could have been reused but were not. Explicitly reference the Scene ID from which they should have been reused.
    - Justification for New Visuals: Assess whether the newly created visuals were necessary due to unique slide content. If a new visual was introduced where reuse was possible, explain why an existing element would have been a better choice.
    - Impact on Visual Consistency: Determine whether the absence of reuse has led to inconsistencies across slides. Identify whether maintaining previously established visual elements would have enhanced coherence and comprehension.
    - Adaptation vs. Direct Reuse: If a previous visual element could have been modified instead of creating an entirely new one, highlight this as a missed opportunity for reuse with adaptation.

4. Final Instructions:
    Before making your Pass/Fail verdict, carefully review the graphics definition and determine whether any reuse opportunities were missed. Use the following criteria to guide your evaluation:
    - If there are no previous graphics definitions available, automatically provide a verdict of "Pass" with the suggestion "No changes needed".
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

Refer to these previous graphics definitions for any visual elements that could potentially be reused. Do not assume any information beyond what is explicitly stated in these previous slides' graphics definition. Only flag reuse opportunities if any relevant scene from these previous definitions contains elements that align with the current slide’s content.
"""


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


