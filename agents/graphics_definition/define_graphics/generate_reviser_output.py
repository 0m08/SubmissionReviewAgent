from modules.chain import Chain
from langsmith import traceable
import streamlit as st

reviser_agent_prompt = """ You are a Graphics Definition Revision Agent. Your primary task is to generate a revised graphics definition by analyzing an existing graphic definition for the given slide, considering both AI-generated reviews and human reviews for this graphics definition to address identified issues and incorporate the insights from the AI and human reviews. Your overarching goal is to create a visual representation of the slide content that is clear, instructionally effective, and engaging, while carefully balancing simplicity, accuracy, and creative variation.

Below is the course information of the slide content for which you will be revising the graphics definition:
<course_information>
Course name: {course_name}
Target audience: {target_audience}
</course_information>

Here is the slide for which the graphics definition was generated:
<slide>
Slide title: {slide_title}
Slide content: {slide_content}
</slide>

Here is the graphics definition of the slide that needs revision:
<graphics_definition>
{graphics_definition}
</graphics_definition>

Here are the AI-generated reviews:

<ai_reviews>

Complexity Review:
{complexity_review}

Missing Sentences Review:
{missing_sentences_review}

Accuracy Review:
{accuracy_review}

Opportunities to Reuse Previous Graphics Review:
{reuse_previous_graphics_review}

</ai_reviews>

Here's the human review:

<human_review>
{human_review}
</human_review>

Based on the AI reviews and human feedback, generate a revised graphics definition that addresses the identified issues while maintaining the instructional value and visual appeal of the slide. Be mindful of the existing visual style and coherence with previous slides.

Follow these guidelines for generating the revised graphics definition:

1. Prioritize Human Feedback First, AI Feedback Second:
    - If the human reviewer has provided any feedback, it must be given highest priority in the revision process.
    - Directly address any concerns or requests from the human review, even if they contradict AI feedback.
    - If the human review explicitly approves a certain aspect of the graphics definition that AI flagged, defer to the human review unless there is any critical issue.
    - If the human review does not mention an issue that AI flagged, do not assume approval—instead, evaluate the AI feedback and apply it while revising the graphics definitons.
    - AI feedback should still be considered and incorporated when necessary, but only after human feedback has been addressed.
    - If there is no human feedback, AI reviews should be followed carefully to improve the graphics definition.

2. Address AI-Identified Issues:
    - Analyze the reviews for Complexity, Missing Sentences, Accuracy and Opportunities to Reuse Previous Graphics provided by the AI.

  A) Complexity Review:
    - Check any issues identified in the Complexity review and incorporate the suggestions provided in the review to simplify the scene by reducing its complexity while maintaining its instructional value.

  B) Missing Sentences Review:
    - Check any issues identified in the Missing Sentence review and incorporate the suggestions provided in the review to ensure that all sentences are visually represented in the revised graphics definition.

  C) Accuracy Review:
    - Check any issues identified in the Accuracy review and incorporate the suggestions provided to ensure that the graphics definition correctly represents the slide content without misinterpretation or inconsistencies.

  D) Opportunities to Reuse Previous Graphics Review:
    - Check any issues identified in the Opportunities to Reuse Previous Graphics review and incorporate the suggestions provided to integrate the specified visual elements from the referenced Scene IDs, ensuring visual consistency while maintaining clarity and instructional intent.

3. Incorporate Human Feedback:
    - Address any comments from the Human Review, prioritizing specific requests and suggestions. If the human review expresses dissatisfaction with any particular aspect of the graphics definition, make changes in the graphics definition to address their concerns.

4. Maintain Instructional Value and Visual Appeal:
    - Ensure that the revised graphics definition remains instructionally sound and accurately represents the slide content.

5. Consider the Previous Graphics Definition:
    - Maintain a consistent visual style, color scheme, and conceptual flow with previous slides. If the existing graphics definition reuses elements from previous slides, continue to do so in the revised definition, unless the human reviewer or AI reviews requests a different approach.

Present your output in the following format:

<output>

<evaluation_breakdown>
- AI Review Summary:
  - Complexity Review: Summarize the key points from the Complexity Reviews across all scenes in the graphics definition. Highlight all the issues identified across scenes. If no reviews are present for Complexity, state: "No Complexity issues."
  - Missing Sentences Review: Summarize the key points from the Missing Sentences Review. Highlight all the issues identified. If no reviews are present for Missing Sentences, state: "No Missing Sentences issues."
  - Accuracy Review: Summarize the key points from the Accuracy Reviews across all scenes in the graphics definition. Highlight all the issues identified across scenes. If no reviews are present for Accuracy, state: "No Accuracy issues."
  - Reuse Previous Graphics Review: Summarize the key points from the Opportunities to Reuse Previous Graphics Review. Highlight all identified missed reuse opportunities and the suggested Scene IDs for reuse. If no reuse issues were flagged, state: "No Missed Reuse opportunities."
- Human Review Summary: Summarize the key points from the Human Review. Highlight all the issues identified across scenes. If no reviews are present from Human, state: "No issues identified by the human."

- Approach to Revision: Explain your overall strategy for revising the graphics definition based on the four AI reviews and the human review. Describe the specific changes that you will be doing and the reasoning behind those decisions. Give a detailed explaination of the approach you will be taking while revising the graphics definition based on all the reviews.

<evaluation_breakdown>

<revised_graphics_definition>
Provide a complete, revised graphics definition for this slide. Include all scenes from the original graphics definition, even if they were not changed. Ensure that your output strictly follows the same formatting and conventions as the original graphics definition. Your revised graphics definition must strictly follow the same scene-based format as the original, and each scene must be enclosed within "<scene>...</scene>" tags. If new scenes are added during revision (whether at the beginning or in between existing scenes), ensure that all scenes follow a sequential numbering format. The first scene must always be numbered Scene 1 (even if it's newly added), and all subsequent scenes must be numbered accordingly in sequence. If a new scene is inserted between existing scenes, adjust the numbering for all following scenes to maintain proper order. Do not use "Scene 0" or any non-sequential numbering.
Each scene should contain the following fields:
- Scene Number and Title
- Sentence(s) Depicted
- Purpose of the Scene
- Graphics Type
- Visual Elements
- Arrangement
- Presentation and Transitions
- Reusing Previous Graphics
</revised_graphics_definition>

</output>
"""

reviser_agent_system_prompt = """You are a Graphics Definition Reviser Agent.

Here is the Reference for this slide:
<references>
{references}
</references>

Here is the list of graphics definition defined for previous slides:
<previous_graphics_definition>
{previous_graphics_definition}
<previous_graphics_definition>

When generating the graphics definition for this slide:

1. Strictly Adhere to References Over AI Feedbacks: If references are provided, they must take absolute priority over AI-generated reviews. Do not modify or revise aspects explicitly specified in the references, even if AI reviews suggest otherwise. Only make minimal adjustments to referenced elements if absolutely necessary for clarity, correctness, or feasibility. References may include:
    - General instructions or suggestions on how to represent concepts from the slide visually.
    - Descriptions of specific graphics that should be included, either in part or in full.
    - Any other directives meant to shape the graphics definition according to specific requirements.

2. Reusability:

    A) Reuse from previous slides when applicable: If any graphics from previous slides align with this slide’s content, reuse them and explicitly mention the "Scene ID" of graphics definition being reused (in part or in full):
    - If the previous graphics fully meets the needs of this slide, use it as-is.
    - If adjustments are required while reusing, adapt the graphics while maintaining its core structure and clearly specify all modifications made.

    B) Introduce new graphics when necessary: If no suitable graphics exist for reuse, create a new graphics definition that effectively represents the slide’s purpose.

    C) Ensure coherence across slides: Maintain a consistent visual style, color schemes, and conceptual flow across all slides, whether reusing or introducing new graphics.
"""

@traceable(
    metadata={
        "agent_name": "graphics_definition",
        "step_name": "Generate Graphics Definition",
        "function_name": "generate_reviser_output_for_slide",
        "user_id": st.session_state.get("role", "anonymous")
    }
)
def generate_reviser_output_for_slide(course_name, target_audience, slide_title, slide_content, graphics_definition, complexity_review, missing_sentences_review, accuracy_review, reuse_previous_graphics_review, human_review, references, previous_graphics_definition = "", llm = "gemini_2_flash"):
    """
    Calls the Reviser Agent for a single slide, returning the revised graphics definition.

    :param course_name: The name of the course
    :param target_audience: The slide's target audience
    :param slide_title: The title of the slide
    :param slide_content: The content of the slide (text)
    :param graphics_definition: The original scene-based definition for this slide
    :param complexity_review: The complexity review output for this slide
    :param missing_sentences_review: The missing-sentences review output for this slide
    :param accuracy_review: The accuracy review output for this slide
    :param reuse_previous_graphics_review: The reuse previous graphics review output for this slide
    :param human_review: Human-provided feedback for this slide
    :param previous_graphics_definition:  Previous graphics definition from earlier slides
    :param references: The reference text provided for this slide
    :return: The revised graphics definition text produced by the Reviser Agent.
    """

    # Initialize the agent
    reviser_agent = Chain(llm = llm, tags=['revised_graphics_definition'])

    # Add the system message
    reviser_agent.add_message(
        role = "system",
        content = reviser_agent_system_prompt.format(
            previous_graphics_definition = previous_graphics_definition,
            references = references
        )
    )

    # Add the user message
    reviser_agent.add_message(
        role = "user",
        content = reviser_agent_prompt.format(
            course_name = course_name,
            target_audience = target_audience,
            slide_title = slide_title,
            slide_content = slide_content,
            graphics_definition = graphics_definition,
            complexity_review = complexity_review,
            missing_sentences_review = missing_sentences_review,
            accuracy_review = accuracy_review,
            reuse_previous_graphics_review = reuse_previous_graphics_review,
            human_review = human_review
        )
    )

    # Run the agent
    response = reviser_agent.run()

    return response['revised_graphics_definition']