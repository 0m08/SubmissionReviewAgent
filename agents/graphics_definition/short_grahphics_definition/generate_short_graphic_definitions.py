from modules.chain import Chain
from tqdm import tqdm
from services.sheets_service import get_sheet_data_and_df, save_to_sheet, clear_worksheet
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.smart_progress_bar import SmartProgressBar
from langsmith import traceable


generate_short_graphic_definitions_prompt = """Your task is to generate short graphic definitions from the provided long graphic definitions.

Refer to the following examples for guidance on how to create short graphic definitions:
<examples>
{examples}
</examples>

Simple notes to follow for generating short graphic definitions:
- Remove tags
- Remove scene
- Change 'Sentence(s) Depicted:' to 'Sentence:'
- Remove all visual elements
- Remove the Arrangement section
- Remove the title 'Presentation and Transitions:', but keep the items
- Keep 'Reusing Previous Graphics:' only if previous graphics are used


Here's the long graphic definitions to focus on:
<long_graphic_definitions>
{long_graphic_definitions}
</long_graphic_definitions>

Your task is to analyze the long graphic definitions and generate short graphic definitions.
Output your response in the following format:
<short_graphic_definitions>
[Place your short graphic definitions with these tags.]
</short_graphic_definitions>

Your short graphic definitions should be concise, clear, and capture the essence of the long graphic definitions.
"""


generate_short_graphic_definitions_examples = """
<example>
Example 1:

<scene>
Scene 1: Acknowledge AC Voltage Dominance

Sentence(s) Depicted: "While AC voltage is the primary power source for many HVAC components, "

Purpose of the Scene:  Illustrate that AC voltage is currently the primary power source in many HVAC systems.

Graphics Type: Diagram

Visual Elements:
HVAC System Diagram: A simplified static diagram of a typical HVAC system, including:
    - AC Power Source (e.g., a standard electrical outlet).
    - AC Compressor.
    - AC Fan Motor.
    - Control Circuit: Reuse "Control Circuit Icon" from Scene 10_4.
AC Voltage Symbol: An AC voltage symbol (a sine wave) next to each component powered by AC.
Static Arrows: Arrows indicating the flow of AC voltage within the system.
Labels: Labels for each component and arrow.

Arrangement:
The HVAC system diagram is positioned in the center of the screen.
Components are clearly visible within the diagram.
Arrows show the directional flow of AC voltage.
AC voltage symbols are placed near each component powered by AC.

Presentation and Transitions:
The entire HVAC system diagram fades in at the start of the sentence.
As the narration mentions "AC voltage," the AC voltage symbols next to each component fade in.
The arrows indicating AC voltage flow also fade in as the narration progresses.

Reusing Previous Graphics:
Reuses the HVAC System Icon from Scene 9_2, but adapts it to show AC voltage symbols.
Reuses the "Control Circuit Icon" from Scene 10_4.
</scene>

<scene>
Scene 2: Transition to DC Voltage Importance

Sentence(s) Depicted: "let's explore the reasons behind the growing importance of DC voltage."

Purpose of the Scene: Introduce the concept that DC voltage is becoming increasingly important in HVAC systems.

Graphics Type: Diagram

Visual Elements:
Futuristic HVAC System Diagram: A more advanced HVAC system diagram, building upon the previous one, and including:
    - Solar Panels (representing DC power source).
    - Battery Storage (representing DC power storage).
    - Variable Speed Compressor.
    - DC Fan Motor.
    - Control Circuit: Reuse "Control Circuit Icon" from Scene 10_4.
DC Voltage Symbol: A DC voltage symbol (a straight line over a dashed line) next to components powered by DC.
Arrows: Arrows indicating the flow of DC voltage within the system.
Labels: Labels for each component and arrow.
A magnifying glass icon appearing over the DC components.

Arrangement:
The futuristic HVAC system diagram is positioned in the center of the screen.
Components are clearly visible within the diagram.
Arrows show the directional flow of DC voltage.
DC voltage symbols are placed near each component powered by DC.
A magnifying glass icon appears over the DC components.

Presentation and Transitions:
The AC HVAC system diagram from Scene 1 fades out.
The futuristic DC-integrated HVAC system diagram fades in, replacing the previous diagram.
As the narration mentions "growing importance of DC voltage," the DC voltage symbols fade in.
Arrows indicating DC voltage flow also fade in.
The magnifying glass appears, focusing on the DC components.

Reusing Previous Graphics:
Reuses the HVAC System Icon from Scene 10_2, but adapts it to highlight DC voltage symbols and includes a magnifying glass.
Reuses the "Control Circuit Icon" from Scene 10_4.
</scene>

This can be:
Sentence: "While AC voltage is the primary power source for many HVAC components, "
Purpose of the Scene:  Illustrate that AC voltage is currently the primary power source in many HVAC systems.
Graphics Type: Diagram
The entire HVAC system diagram fades in at the start of the sentence.
As the narration mentions "AC voltage," the AC voltage symbols next to each component fade in.
The arrows indicating AC voltage flow also fade in as the narration progresses.
Reusing Previous Graphics:
Reuses the HVAC System Icon from Scene 9_2, but adapts it to show AC voltage symbols.
Reuses the "Control Circuit Icon" from Scene 10_4.

Sentence: "let's explore the reasons behind the growing importance of DC voltage."
Purpose of the Scene: Introduce the concept that DC voltage is becoming increasingly important in HVAC systems.
Graphics Type: Diagram
The AC HVAC system diagram from Scene 1 fades out.
The futuristic DC-integrated HVAC system diagram fades in, replacing the previous diagram.
As the narration mentions "growing importance of DC voltage," the DC voltage symbols fade in.
Arrows indicating DC voltage flow also fade in.
The magnifying glass appears, focusing on the DC components.
Reusing Previous Graphics:
Reuses the HVAC System Icon from Scene 10_2, but adapts it to highlight DC voltage symbols and includes a magnifying glass.
Reuses the "Control Circuit Icon" from Scene 10_4.
</example>

<example>
Example 2:

<scene>
Scene 1: Introduction to DC Voltage Applications

Sentence(s) Depicted: "Now, let's examine specific examples of how DC voltage is utilized in various components and systems within HVAC."

Purpose of the Scene: Introduce the concept of DC voltage applications in HVAC systems, setting the stage for specific examples.

Graphics Type: System Diagram

Visual Elements:
HVAC System Diagram: A simplified diagram of a typical HVAC system, including components such as:
    - Thermostat
    - Control Board (with a miniaturized representation of the control circuit from Scene 15_2)
    - Sensors
    - Actuators
    - Motors (small DC motors)
DC Voltage Symbol: Reusing the "DC Voltage Symbol" from Scene 14_1.
Highlight Effect: A fade-in effect to highlight specific components in the diagram.
Connecting Lines: Dashed lines to indicate the flow of DC voltage.

Arrangement:
The HVAC system diagram is centrally placed on the slide.
The DC Voltage Symbol appears near each highlighted component.
Dashed lines connect the DC Voltage Symbol to the highlighted components.
The miniaturized control circuit diagram is integrated within the control board component.

Presentation and Transitions:
The HVAC system diagram appears first.
As the narration begins, the thermostat component is highlighted with a fade-in effect.
The DC Voltage Symbol appears near the thermostat.
Dashed lines connect the DC Voltage Symbol to the thermostat.
After a brief pause, the next component (e.g., control board) is highlighted with a fade-in effect, and the process repeats. The miniaturized control circuit diagram is visible within the control board.
This continues for each component mentioned (sensors, actuators, small DC motors).

Reusing Previous Graphics:
Reusing the "DC Voltage Symbol" from Scene 14_1.
Reusing and adapting the simplified control circuit diagram from Scene 15_2, miniaturized and integrated into the control board component.
</scene>

This can be:

Sentence: "Now, let's examine specific examples of how DC voltage is utilized in various components and systems within HVAC."
Purpose of the Scene: Introduce the concept of DC voltage applications in HVAC systems, setting the stage for specific examples.
Graphics Type: System Diagram
The HVAC system diagram appears first.
As the narration begins, the thermostat component is highlighted with a fade-in effect.
The DC Voltage Symbol appears near the thermostat.
Dashed lines connect the DC Voltage Symbol to the thermostat.
After a brief pause, the next component (e.g., control board) is highlighted with a fade-in effect, and the process repeats. The miniaturized control circuit diagram is visible within the control board.
This continues for each component mentioned (sensors, actuators, small DC motors).
Reusing Previous Graphics:
Reusing the "DC Voltage Symbol" from Scene 14_1.
Reusing and adapting the simplified control circuit diagram from Scene 15_2, miniaturized and integrated into the control board component.
</example>
"""


@traceable(metadata={
    "agent_name": "graphics_definition",
    "step_name": "Generate Short Graphic Definitions",
    "function_name": "generate_short_graphic_definitions",
    "user_id": st.session_state.get("role", "anonymous")
})
def generate_short_graphic_definitions(long_graphic_definitions, llm = 'gemini_2_flash'):
    """
    This function generates short graphic definitions from long graphic definitions.

    :param long_graphic_definitions: The long graphic definitions.
    :param llm: The language model to use.
    :return: The short graphic definitions.
    """
    generate_short_graphic_definitions_agent = Chain(llm = llm, tags = ['short_graphic_definitions'])

    generate_short_graphic_definitions_agent.add_message(
        role = 'user',
        content = generate_short_graphic_definitions_prompt.format(
            examples = generate_short_graphic_definitions_examples,
            long_graphic_definitions = long_graphic_definitions
        )
    )

    response = generate_short_graphic_definitions_agent.run()

    return response['short_graphic_definitions']


@traceable(metadata={
    "agent_name": "graphics_definition",
    "step_name": "Run Generate Short Graphic Definitions",
    "function_name": "run_generate_short_graphic_definitions",
    "user_id": st.session_state.get("role", "anonymous")
})
def run_generate_short_graphic_definitions(sheet, worksheet_name, llm = 'gemini_2_flash'):
    """
    This function runs the generate short graphic definitions agent for all rows in the sheet.

    :param sheet: The sheet object.
    :param worksheet_name: The worksheet name.
    :param llm: The language model to use.
    :return: None
    """

    # Read the sheet and df
    slide_chunks_sheet, slide_chunks_df = get_sheet_data_and_df(sheet, worksheet_name)

    # Create short_graphics_definitions col if not already present
    if 'short_graphics_definitions' not in slide_chunks_df.columns:
        slide_chunks_df['short_graphics_definitions'] = ''

    # Check if this step is already done by checking all rows of short_graphics_definitions column
    if slide_chunks_df['short_graphics_definitions'].str.strip().all():
        print('Short graphics definitions already generated for all rows. Skipping this step.')
        return

    # Prepare for parallel processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit tasks for each row
        for index, row in slide_chunks_df.iterrows():

            # Skip if short graphic definitions already populated
            if row['short_graphics_definitions'] != '':
                print(f'Skipping row {index}. Already populated')
                continue
            
            # Submit the task
            future = executor.submit(
                generate_short_graphic_definitions,
                long_graphic_definitions = row['Graphics Definition'],
                llm = llm
            )

            # Map the Future to the index
            futures_map[future] = index

        # Collect the results as they complete
        total_tasks = len(futures_map)
        save_interval = 5  # how often to save (in number of completed tasks)

        # Initialize the progress tracker
        progress = SmartProgressBar(total_tasks = total_tasks, description = "Percent complete", save_interval = save_interval)

        # Now, pass only the futures (the keys) to as_completed:
        for future in tqdm(as_completed(futures_map), total=total_tasks):
            index = futures_map[future]  # retrieve the index
            short_graphic_definition = future.result()

            # Update the df row with analysis
            slide_chunks_df.loc[index, 'short_graphics_definitions'] = short_graphic_definition

            # Update progress
            progress.update()

            # Check if we should save
            if progress.should_save():
                print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
                save_to_sheet(worksheet = slide_chunks_sheet, df = slide_chunks_df)

    # Final save to sheet after all tasks
    print('All rows processed. Saving final DataFrame to sheet.')
    save_to_sheet(worksheet = slide_chunks_sheet, df = slide_chunks_df)

    return


def delete_short_graphic_definitions(sheet, worksheet_name="Slide Chunks"):
    """Remove the short_graphics_definitions column from the worksheet."""
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    if "short_graphics_definitions" in df.columns:
        df = df.drop(columns=["short_graphics_definitions"])
        clear_worksheet(ws)
        save_to_sheet(ws, df)


