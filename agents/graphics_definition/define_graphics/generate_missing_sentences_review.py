from modules.chain import Chain
from concurrent.futures import ThreadPoolExecutor

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