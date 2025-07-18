# from modules import Chain
import re
from modules.chain import Chain
from langsmith import traceable
import streamlit as st


generate_graphics_definition_prompt = """You are a Graphics Definition Agent. Your task is to create precise and structured graphics definition for a given slide, ensuring each sentence is visually represented in a clear, instructional, and engaging manner.

Below is the course information for which you will be creating the graphics definition:
<course_information>
Course name: {course_name}
Target audience: {target_audience}
</course_information>

Here's the current slide we need to focus on:
<slide>
Slide title: {slide_title}
Slide content:
{slide_content}
</slide>

Follow the below Guidelines:

1. Scene-Based Graphics Definition:
    Define graphics on a scene-by-scene basis, ensuring that each sentence from the slide content is represented exactly once. Each scene must cover only the sentence(s) assigned to it, without overlapping or repeating content across scenes. If a sentence is already visualized in one scene, it must not appear again in another scene. No sentence should be left out—every sentence in the slide content must be represented explicitly. If multiple sentences logically belong to the same visual representation, they should be grouped within a single scene while maintaining clarity in their depiction. Each scene should focus on capturing the full instructional meaning of the sentence rather than individual words while maintaining alignment with the intended learning objectives. The graphics definition for a scene must be directly relevant to all the sentence content it depicts, ensuring that no visual elements are unrelated or misaligned with the instructional intent. Scenes must transition smoothly, forming a logical and cohesive narrative that enhances comprehension and engagement. Each scene must contain the following components:

    - Scene Number and Title: A unique number and a concise title summarizing the visual concept of the scene.
    - Sentence(s) Depicted: The sentence(s) from the slide content that the scene visualizes.
    - Purpose of the Scene: A clear explanation of what this scene represents and what it aims to convey.
    - Graphics Type: Specify the type of graphics (e.g., image, diagram, chart, icon, animation, illustration, infographics, etc.)
    - Visual Elements: A detailed explanation of all graphical components used in the scene (e.g., icons, diagrams, illustrations, animations, etc).
    - Arrangement: Describe the spatial positioning of the elements, their hierarchy, and how they interact with one another. Describe any layering, alignment, or grouping details.
    - Presentation and Transitions: Describe how and when each visual element will appear, animate, or transition on the slide in sync with the narration of the sentence. Clearly specify which graphics appear at which part of the sentence, ensuring a precise connection between narration and visuals. Maintain a smooth storytelling experience by continuing existing graphics where appropriate or introducing new ones that naturally extend the narrative.
    - Reusing Previous Graphics: If applicable, reference any visuals from earlier scenes that can be reused. Specify if modifications or adjustments are needed for consistency and coherence.

2. Evaluation Breakdown:
    Before generating the graphics definition, start with a structured analysis in the evaluation breakdown field. The purpose of this analysis is to evaluate the slide content and establish a clear foundation for the graphics definition. The graphics definition must align directly with the concepts, visual elements, and transitions outlined in your analysis in the evaluation breakdown field. The following key aspects must be addressed in this analysis:

    - Read and Understand the Slide content thoroughly: Ensure you grasp the key instructional message being conveyed in the slide content.
    - Core Concept to Visualize: Identify the main idea or takeaway that the visuals should reinforce.
    - Segment the Content into Logical Scenes: Break the content into logical scenes for clarity and engagement.
    - Sentence-to-Scene Mapping: Assign every sentence to a scene, ensuring all sentences are fully represented without omission or duplication. If multiple sentences logically belong to the same scene, they should be grouped together within that scene while maintaining clarity of the graphics definition.
    - Brainstorm Multiple Possible Visual Representations for every Scene: Generate at least 2-3 different ways to visually depict each scene. Consider various approaches, including icons, diagrams, animations, and other graphics.
    - Evaluate and select the Most Effective and Feasible Ideas: Compare the brainstormed ideas for each scene. Identify the strengths and weaknesses of each option. Select the best idea based on clarity, engagement, ease of creation, and alignment with instructional goals of the scene. Justify why the chosen idea is the most effective representation.
    - Determine which Visual Elements to Use: Define the best type of graphics (icons, diagrams, illustrations, animations, etc.) that align with the content.
    - Plan scene transitions and animations: Outline how elements should appear, transition, or build up dynamically on the slide.
    - Ensure Visual Consistency: Maintain consistency across scenes by reusing elements when applicable.

3. Additional Guidelines:
    To ensure clarity, consistency, and instructional value, follow these additional guidelines when defining graphics:

    - Ensure that every sentence in the slide content is strictly and explicitly represented in the graphics definition, including introductory statements, summary sentences, and guiding questions. Even if a sentence does not introduce a new concept but serves as an introduction, transition, or question, it must still be visually represented in a meaningful way.
    - Where appropriate, reuse visual elements to maintain a consistent structure across scenes. Maintain a smooth narrative flow, ensuring that learners intuitively understand how concepts connect.
    - Graphics should enhance learning while keeping the slide visually clear and engaging.
    - Keep definition concise while providing enough detail for accurate implementation.
    - Use clear and simple language in definition so that even a novice graphics designer can accurately create the graphics.
    - Tailor the definition to the target audience, ensuring the graphics are appropriate for their level of understanding and learning needs.

Present your output in the following format:

<evaluation_breakdown>
[Outline your analysis and scene planning here. Your analysis should cover:
- Key instructional message
- Core concept to visualize
- Logical scene segmentation
- Sentence-to-Scene Mapping
- Brainstormed visual ideas (at least 3 per scene)
- Selection of the most effective idea per scene
- Visual elements, transitions, and consistency planning]
</evaluation_breakdown>

<graphics_definition>

<scene>
Scene [Insert scene number] : [Scene Title]

Sentence(s) Depicted: [Include the full sentence(s) from the slide content that the scene visualizes]

Purpose of the Scene: [Briefly explain what the scene represents and what it conveys]

Graphics Type: [Specify the type of graphics used for the scene]

Visual Elements: [Explain all graphical components used in the scene]

Arrangement: [Describe how these elements are positioned and interact]

Presentation and Transitions: [Explain how and when each element appears, animates, or transitions in sync with the narration of the sentence. Clearly specify which graphics appear at which part of the sentence, ensuring an exact match between narration and visuals]

Reusing Previous Graphics: [Specify any reused graphics from earlier scenes and mention modifications if needed]
</scene>

[Repeat the scene blocks for all the scenes you have planned...]

</graphics_definition>

"""


generate_graphics_definition_prompt_examples = """Examples:
Use these below examples as a reference for the structure, style and detail of your graphics definition.

<examples>

<example>

<slide_content>
Let's talk about temperature. It's something we deal with every day, especially in the HVAC industry. Temperature is simply a measure of how hot or cold something is. But there's more to it than that. At a molecular level, temperature tells us about the average speed of particles in a substance. You can think of it as the energy level of these molecules. Understanding temperature is crucial for managing comfort in any space.
</slide_content>

<evaluation_breakdown>

Key Instructional Message:
This slide introduces temperature as a concept and its relevance in HVAC, transitioning from a basic definition to its molecular understanding and finally to its practical application in managing comfort.

Core Concept to Visualize:
- Temperature measures how hot or cold something is.
- It is related to the energy level and movement of molecules.
- Understanding temperature is essential for HVAC professionals in maintaining comfort in any space.

Logical Scene Segmentation:
Scene 1: Introduction to Temperature – Establish the concept of temperature and its daily relevance in HVAC.
Scene 2: Molecular Understanding of Temperature – Explain temperature in terms of molecular motion and energy levels.
Scene 3: Importance of Temperature Control – Highlight how temperature management affects comfort.

Sentence-to-Scene Mapping:
Scene 1: "Let's talk about temperature. It's something we deal with every day, especially in the HVAC industry. Temperature is simply a measure of how hot or cold something is. But there's more to it than that."
Scene 2: "At a molecular level, temperature tells us about the average speed of particles in a substance. You can think of it as the energy level of these molecules."
Scene 3: "Understanding temperature is crucial for managing comfort in any space."

Brainstormed Visual Ideas:
- Scene 1: Introduction to Temperature
    Idea 1: A thermometer illustration with a rising red column along with illustrations of temperature in HVAC industry.
    Idea 2: A digital thermostat displaying an indoor temperature.
    Idea 3: A scene of different weather conditions representing temperature variations.

- Scene 2: Molecular Understanding of Temperature
    Idea 1: A heatmap visualization showing energy distribution at different temperatures.
    Idea 2: A magnifying glass zooming into a surface to reveal molecular movement.
    Idea 3: A comparative diagram showing fast and slow-moving molecules.

- Scene 3: Importance of Temperature Control
    Idea 1: A family sitting comfortably in a temperature-controlled room.
    Idea 2: A side-by-side comparison of comfortable and uncomfortable indoor environments.
    Idea 3: A thermostat adjusting room temperature dynamically.

Evaluation and Selection of Most Effective Ideas:
- Scene 1 (Selected: Idea 1): A thermometer illustration effectively introduces the concept.
- Scene 2 (Selected: Idea 3): A comparative diagram illustrates molecular motion clearly.
- Scene 3 (Selected: Idea 1): A family in a controlled indoor space highlights temperature management’s impact.

Visual Elements and Transitions:
- Scene 1: A thermometer illustration appears first.
- Scene 2: A molecular motion diagram fades in.
- Scene 3: A living space with an air conditioner and comfortable occupants appears.

- Consistency Planning: A red-blue gradient differentiates hot and cold temperatures. The thermometer graphics remains consistent in the first scene.

</evaluation_breakdown>

<graphics_definition>

<scene>
Scene 1: Introduction to Temperature

Sentence(s) Depicted: "Let's talk about temperature. It's something we deal with every day, especially in the HVAC industry. Temperature is simply a measure of how hot or cold something is. But there's more to it than that."

Purpose of the Scene:  Introduce the concept of temperature using a familiar visual representation.

Graphics Type: Composite Illustration

Visual Elements:
- Thermometer: A vertical thermometer with a red liquid column.
- HVAC Industry Icons:
  - A digital thermostat showing temperature control.
  - An HVAC technician working on an AC unit.
  - A sun icon (yellow) representing heat.
  - A snowflake icon (blue) representing cold.
- Temperature Scale: A gradient bar transitioning from blue (cold) to red (hot).
- Magnifying Glass: A magnifying glass appearing to emphasize deeper exploration.

Arrangement:
- The thermometer appears first at the center.
- The HVAC-related icons (thermostat, technician, sun, and snowflake) fade in, positioned around the thermometer.
- The temperature scale slides in from the top and positions itself below the thermometer.
- The magnifying glass enters from the right and hovers over the thermometer.

Presentation and Transitions:
- Thermometer fades in at the center of the slide at the start(during "Let's talk about temperature.").
- HVAC-related icons fade in around the thermometer (during "It's something we deal with every day, especially in the HVAC industry.").
- HVAC-related icons fade out and temperature scale slides in from the top and settles below the thermometer (during "Temperature is simply a measure of how hot or cold something is.").
- Magnifying glass appears from the right and overlays the thermometer (during "But there's more to it than that.").

Reusing Previous Graphics:
- No reuse.
</scene>

<scene>
Scene 2: Molecular Understanding of Temperature

Sentence(s) Depicted: "At a molecular level, temperature tells us about the average speed of particles in a substance. You can think of it as the energy level of these molecules."

Purpose of the Scene: Illustrate the relationship between temperature and molecular motion, emphasizing that higher temperature corresponds to increased molecular movement.

Graphics Type: Comparative Diagram

Visual Elements:
- Two square boxes representing different temperature conditions:
  - Left Box (Cooler Conditions): Light blue background, blue molecules grouped closely together with small black arrows.
  - Right Box (Warmer Conditions): Light red background, red molecules spaced farther apart with longer black arrows.

Arrangement:
- The two boxes are aligned horizontally, side by side.

Presentation and Transitions:
- The two molecular boxes fade in together as the sentence begins.

Reusing Previous Graphics:
- No reuse .
</scene>

<scene>
Scene 3: Importance of Temperature Control

Sentence(s) Depicted: "Understanding temperature is crucial for managing comfort in any space."

Purpose of the Scene:  Show how temperature control impacts comfort.

Graphics Type: Illustration

Visual Elements:
- A family sitting comfortably in a living room.
- An air conditioner mounted overhead emitting cool air.

Arrangement:
- The family is centrally placed.
- The air conditioner is positioned above them with airflow motion lines.

Presentation and Transitions:
- The illustration fades in as the narration starts.

Reusing Previous Graphics:
- No reuse.
</scene>

</graphics_definition>

</example>

<example>

<slide_content>
To really grasp temperature, imagine you could see molecules. In warmer things, these molecules would be zipping around faster. In cooler things, they'd be moving more slowly. Let's use water as an example. In an 80-degree glass of water, the molecules are bouncing around more quickly than in a 60-degree glass. This molecular motion is what we're really measuring when we talk about temperature.
</slide_content>

<evaluation_breakdown>

Key Instructional Message:
This slide builds on the audience's understanding of temperature by introducing molecular motion as the core concept being measured. It uses relatable examples, such as glasses of water at different temperatures, to visually and conceptually explain the connection between temperature and molecular behavior.

Core Concept to Visualize:
- Temperature is a measure of molecular motion.
- Warmer molecules move faster, cooler molecules move slower.
- Water serves as a real-world example of molecular motion.

Logical Scene Segmentation:
Scene 1: Molecular Motion and Temperature Differences – Establishes the relationship between temperature and molecular motion, comparing how molecules behave at different temperatures.
Scene 2: Molecular Motion in Water – Introduce water as the medium for temperature comparison and show molecular motion differences in warm vs. cool water.

Sentence-to-Scene Mapping:
Scene 1: "To really grasp temperature, imagine you could see molecules. In warmer things, these molecules would be zipping around faster. In cooler things, they'd be moving more slowly."
Scene 2: "Let's use water as an example. In an 80-degree glass of water, the molecules are bouncing around more quickly than in a 60-degree glass. This molecular motion is what we're really measuring when we talk about temperature."

Brainstormed Visual Ideas:
- Scene 1: Molecular Motion and Temperature Differences
    Idea 1: A thermometer illustration that transitions into a magnified molecular view.
    Idea 2: A split-screen with a thermometer on one side and animated molecules on the other.
    Idea 3: A temperature gradient bar shifting from blue to red, symbolizing heat levels affecting molecular motion.

- Scene 2: Molecular Motion in Water
    Idea 1: A bar graph dynamically linked to molecular movement speed.
    Idea 2: A close-up of water molecules before specifying temperature differences.
    Idea 3: Two glasses of water with molecular animations inside, one at 80°F and one at 60°F.

Evaluation and Selection of Most Effective Ideas:
- Scene 1 (Selected: Idea 1): The thermometer transitioning into molecular visualization effectively introduces the concept.
- Scene 2 (Selected: Idea 3): Two glasses of water with animated molecules illustrate how molecular motion differs based on temperature.

Visual Elements and Transitions:
- Scene 1 A thermometer illustration appears first, then transitions into a magnified molecular view with animated molecules.
- Scene 2: Two identical glasses of water appear, and molecular motion animations begin inside each.

- Consistency Planning: A consistent color scheme will be applied — blue for cooler molecules and red for warmer molecules. Molecular motion will follow the same animation principles across scenes, ensuring clarity. Repeatable elements like molecules, water glasses, and thermometers will be used and adapted across scenes to create a seamless instructional flow.

</evaluation_breakdown>

<graphics_definition>

<scene>
Scene 1: Molecular Motion and Temperature Differences

Sentence(s) Depicted: "To really grasp temperature, imagine you could see molecules. In warmer things, these molecules would be zipping around faster. In cooler things, they'd be moving more slowly."

Purpose of the Scene: Establish the fundamental relationship between temperature and molecular motion.

Graphics Type: Illustration and Animated diagram.

Visual Elements:
- A vertical thermometer with a red liquid column, positioned centrally.
- Two square boxes:
  - Left box (blue): Represents cooler conditions, with slowly moving blue molecules.
  - Right box (red): Represents warmer conditions, with rapidly moving red molecules.

Arrangement:
- The thermometer is centered on the slide initially.
- As the narration progresses, the thermometer disappears, and two molecular representation boxes appear side by side.
- Both boxes are of equal size and aligned horizontally.

Presentation and Transitions:
- - The thermometer appears first.
- As the narration reaches "imagine," the thermometer fades out, and the two molecular representation boxes fade in simultaneously to replace the thermometer.
- As the narration reaches "zipping around faster," molecules inside the red box start moving rapidly in random directions.
- As the narration reaches "moving more slowly," molecules inside the blue box start moving in a slow, gradual motion.
- The animation is continuous throughout the scene, reinforcing the contrast in molecular motion.

Reusing Previous Graphics:
- No reuse.
</scene>

<scene>
Scene 2: Molecular Motion in Water

Sentence(s) Depicted: "Let's use water as an example. In an 80-degree glass of water, the molecules are bouncing around more quickly than in a 60-degree glass. This molecular motion is what we're really measuring when we talk about temperature."

Purpose of the Scene: Introduce water as the medium for temperature comparison and show molecular motion differences in warm vs. cool water.

Graphics Type: Animated Illustration

Visual Elements:
- Two smaller molecular representation boxes with animations from Scene 2, placed inside each glass.
  - Left glass (60°F): Contains a scaled-down blue molecular box, with blue molecules moving slowly in random directions.
  - Right glass (80°F): Contains a scaled-down red molecular box, with red molecules moving rapidly in random directions.
- Temperature labels:
  - "60°F" positioned below the left glass.
  - "80°F" positioned below the right glass.

Arrangement:
- The two glasses remain side by side, aligned horizontally.
- The molecular boxes are resized and positioned inside each respective glass.
- The temperature labels are placed below each glass.

Presentation and Transitions:
- The glasses fade in first as the narration starts, replacing the molecular boxes from the previous scene.
- The molecular boxes fade in and appear inside the respective glasses as the narration mentions "In an 80-degree glass of water, the molecules are bouncing around more quickly than in a 60-degree glass."
- The temperature labels fade in at the same time as the molecular boxes.
- Molecules begin moving immediately once the boxes appear, visually illustrating the motion as it is described.

Reusing Previous Graphics:
- Reuses the molecular motion boxes from Scene 2, with reduced size and repositioning inside the glasses.
</scene>

</graphics_definition>

</example>

</examples>
"""

# System prompt for graphics definition
generate_graphics_definition_system_prompt = """You are a Graphics Definition Agent.

Here is the Reference for this slide:
<references>
{references}
</references>

Here is the list of graphics definition defined for previous slides:
<previous_graphics_definition>
{previous_graphics_definition}
<previous_graphics_definition>

When generating the graphics definition for this slide:

1. Incorporate References when provided: If references are available, use them as guide while creating the graphics definition for the slide. You must strictly follow the provided references as closely as possible. The references may include:
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
        "step_name": "Graphics Definition Generation",
        "function_name": "generate_graphics_definition",
        "user_id": st.session_state.get("role", "anonymous")
    }
)
def generate_graphics_definition(course_name, target_audience, slide_title, slide_content, previous_graphics_definition, references, llm = "gemini_2_flash"):
    """
    Generate graphics definition for a given slide.

    :param course_name: The name of the course.
    :param target_audience: The target audience for the course.
    :param slide_title: The title of the slide.
    :param slide_content: The content of the slide.
    :param previous_graphics_definition: The graphics definition from previous slides.
    :param references: The reference text provided for the slide.
    :return: The generated graphics definition.
    """

    # Initialize the agent
    generate_graphics_definition_agent = Chain(llm = llm, tags = ['graphics_definition'])

    # Add the system message
    generate_graphics_definition_agent.add_message(
        role = "system",
        content = generate_graphics_definition_system_prompt.format(
            previous_graphics_definition = previous_graphics_definition,
            references = references
        )
    )

    # Add the user message
    generate_graphics_definition_agent.add_message(
        role = "user",
        content = generate_graphics_definition_prompt.format(
            course_name = course_name,
            target_audience = target_audience,
            slide_title = slide_title,
            slide_content = slide_content
        ) + generate_graphics_definition_prompt_examples
    )

    # Run the agent
    response = generate_graphics_definition_agent.run()

    return response['graphics_definition']

def extract_text_from_scene_tags(text):
    tag = "scene"
    pattern = f"<{tag}>\s*(.*?)\s*</{tag}>"
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
    if matches:
        return matches
    else:
        raise ValueError(f"No matches found for tag: {tag}")
    

def extract_visual_elements_and_arrangement(text):
    """
    Extracts the 'Visual Elements', 'Arrangement' and 'Reusing Previous Graphics' sections from the script text.

    :param text: The entire scene description as a string.
    :return: A tuple (visual_elements_str, arrangement_str, reuse_previous_graphics_str). If a section isn't found,
             its corresponding string will be empty.
    """

    # Regex to capture text after 'Visual Elements:' up to (but not including) the next heading that starts at the beginning of a line and ends with a colon.
    visual_pattern = re.compile(
        r'^Visual Elements:\s*(.*?)^(?=[A-Z][^:\n]*?:)',
        re.DOTALL | re.MULTILINE
    )

    # Regex to capture text after 'Arrangement:' up to the next heading.
    arrangement_pattern = re.compile(
        r'^Arrangement:\s*(.*?)^(?=[A-Z][^:\n]*?:)',
        re.DOTALL | re.MULTILINE
    )

    # Regex to capture the "Reusing Previous Graphics:" up to the next heading or the end of the scene.
    reuse_graphics_pattern = re.compile(
        r'^Reusing Previous Graphics:\s*([\s\S]*?)(?=\n[A-Z][^:\n]*?:|</scene>|$)',
        re.DOTALL | re.MULTILINE
    )

    # Search for matches
    visual_match = visual_pattern.search(text)
    arrangement_match = arrangement_pattern.search(text)
    reuse_graphics_match = reuse_graphics_pattern.search(text)

    # Extract and strip results
    visual_text = visual_match.group(1).strip() if visual_match else ""
    arrangement_text = arrangement_match.group(1).strip() if arrangement_match else ""
    reuse_previous_graphics_text = reuse_graphics_match.group(1).strip() if reuse_graphics_match else ""

    return visual_text, arrangement_text, reuse_previous_graphics_text


def get_previous_graphics_definition_as_str(slide_no, text, existing_definition_str = ""):
    """
    This function creates and returns the previous graphics definition as a string

    :param slide_no: The slide number
    :param text: The text to be processed
    :param existing_definition_str: The existing definition string
    :return: The updated existing definition string
    """

    # If the text is empty or doesn't contain <scene>, return the existing definition str
    if not text or "<scene>" not in text:
        return existing_definition_str

    # Extract the scenes as a list
    scenes = extract_text_from_scene_tags(text)

    # Loop over the list to get the visual elements, arrangement and reusing previous graphics for each scene
    for scene_no, scene in enumerate(scenes):

        # Extract the visual element and arrangement for current scene
        visual_elements, arrangement, reuse_previous_graphics = extract_visual_elements_and_arrangement(scene)

        # Add scene id > slide no + scene no
        existing_definition_str += f"Scene ID: {slide_no}_{scene_no + 1}\n\n"

        # Add the visual element and arrangement
        existing_definition_str += f"Visual Elements:\n{visual_elements}\n\n"
        existing_definition_str += f"Arrangement:\n{arrangement}\n\n"
        existing_definition_str += f"Reusing Previous Graphics:\n{reuse_previous_graphics}\n\n"

        existing_definition_str += "---\n\n"

    return existing_definition_str.strip().strip('---')

