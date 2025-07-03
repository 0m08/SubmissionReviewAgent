from modules.chain import Chain
from tqdm import tqdm
from services.sheets_service import (
    get_sheet_data_and_df,
    save_to_sheet,
    filter_non_blank_column,
    clear_worksheet,
    delete_worksheet,
    clear_all_filters,
)
import json
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from services.helper_functions import get_outline_with_los, validate_column_values
from services.smart_progress_bar import SmartProgressBar
from langsmith import traceable


get_relevant_chunks_prompt = """You are tasked with identifying sections of a video that can be used as-is for a given course. Your goal is to analyze the video chapter summaries and determine which parts, if any, align well with the course content and are suitable for the target audience. Be conservative in your selections and only suggest sections that properly fit the course intent.

First, review the course information:

Course Name: {course_name}
Target Audience: {target_audience}

Course Outline:
<course_outline>
{course_outline}
</course_outline>

Now, examine the video information:

Video Title: {video_title}

Video Chapter Summaries:
<chapter_summaries>
{chapter_summaries}
</chapter_summaries>

To analyze the video and identify relevant sections:

1. Carefully read through the entire outline and video chapter summaries.
2. Compare the content of the video to the course outline and objectives.
3. Identify sections that directly relate to specific points in the course outline.
4. Avoid selecting sections that only provide supplementary or tangential information wrt the course outline.

Output in the following format:

<scratchpad>
[Place to think step by step and show your thinking.
Its okay for this section to be quite long as long as you end up with proper classification.
Analyze each section wrt to the course outline. Ask questions to self and self reflect.]
</scratchpad>

<video_relevance>
[Fully Relevant / Partially Relevant / Irrelevant]
</video_relevance>

<proposed_chapters_to_include>
[If video is marked as fully or partially relevant, list timestamps and relevant chapters in following format: Timestamp in square brackets followed by chapter title. If multiple relevant chapters, insert them in new lines. If video is irrelevant, leave this section blank.]
</proposed_chapters_to_include>

Important considerations:
- Only include relevant sections if it is a 100% match and exactly covers one or more concepts in the outline.
- DO NOT include sections that don't fit the outline 100%.
- I REPEAT AGAIN, only identify relevant sections, these sections can be used directly, don't contain the information in any other context.
- Most of the times, you will not be able to find relevant sections and that is acceptable.

Remember, your goal is to identify only those sections that align closely with the course content and are immediately usable without modification. Avoid suggesting sections that would require significant editing or additional explanation to fit the course.
"""


relevant_chunk_example = {
'course_name': "Basics of Electricity and Magnetism",

'target_audience': "Entry-level HVAC technicians",

'course_outline': """Topic: Electrons in Electricity
  Subtopic: Fundamental electrical concepts
  Subtopic: Electrical charge
  Subtopic: Conductors
  Subtopic: Insulators
  Subtopic: Dielectrics

Topic: Electrical Basic Terms
  Subtopic: Voltage
  Subtopic: Amps - milliamps, microamps, etc.
  Subtopic: Resistance - Ohm's, megohms, etc.
  Subtopic: Power - watts

Topic: Magnetism
  Subtopic: Magnetic principles in electricity
  Subtopic: Magnetic components - coils
  Subtopic: Magnetic components - transformer""",

'video_title': "HVAC Myth Busting #2 - Volts Go Down, Amps Go Up",
'chapter_summaries': """[00:00] Intro and Myth Introduction
Summary: The video starts with an introduction, a bit of banter, and sets the stage for discussing common HVAC myths, specifically focusing on a recent Facebook poll about voltage, amps, and heat strips.

[00:57] Myth Busting Time: Voltage, Amps, and Heat Strips
Summary: The presenter introduces the myth-busting topic of what happens when voltage changes, differentiating between resistive loads like heat strips and inductive loads like motors. He explains the basic concepts of resistance, inductive reactance, and impedance, and argues that understanding real-world application of Ohm's law is more important than just the formula.

[04:07] The Mistake: A Real-World Example with Heat Strips
Summary: The presenter discusses a real-world mistake he made involving a heat strip restring kit and uses a heat strip data tag to illustrate how decreasing voltage decreases both amps and watts. He emphasizes that resistance stays relatively constant in a heat strip.

[06:00] Voltage Drop on Resistive Loads: Amps and Watts
Summary: The presenter explains why decreasing voltage decreases both amps and watts, emphasizing that watts do not stay fixed when voltage changes. He shares a personal story about cutting a heat strip, which increased the amperage, and explains that reducing resistance increases current if voltage is fixed.

[08:07] The Truth About Resistance and Amperage
Summary: The presenter reinforces the relationship between resistance and amperage, stating that reducing resistance increases amperage and vice-versa, and officially "busts" the myth for resistive loads: decreasing voltage decreases both wattage and amperage.

[09:56] Heat Strip Experiment: Proving the Point
Summary: The presenter demonstrates the effect of reduced voltage on amperage using a heat strip connected to a DC power supply, proving the point that reduced voltage results in reduced amperage. He then reiterates the importance of understanding what variable is being solved for in Ohm's law.

[10:58] Understanding Ohm's Law in Practice
Summary: The presenter explains the practical application of Ohm's Law, highlighting that while the formula is simple, real-world situations are more complex because changing one variable often changes others. He uses the example of a light bulb to show that resistance changes as it heats up.

[13:46] Light Bulbs and Real-World Resistance
Summary:  The presenter continues the discussion on light bulbs as an example of the complexities of Ohm's law in real life. He explains that the resistance of a light bulb changes when it heats up, and the math only works if you calculate the resistance based on the working voltage and amperage, not the cold resistance.

[14:55] Inductive Loads: The Complexity of Motors
Summary: The presenter introduces inductive loads by discussing the complexities of motors and how their behavior is different than resistive loads. He uses a compressor as an example, explaining that the winding resistance doesn't tell the whole story because of inductive reactance.

[16:11] Inductive Reactance and Motor Behavior
Summary: The presenter explains inductive reactance and its effect on motor current, clarifying that it's the magnetic resistance that opposes current once the motor reaches speed. He explains that more physical resistance to a motor (slip) results in higher current due to less inductive reactance. He transitions to discussing how voltage changes impact motors.

[17:37] What Happens to Motors When Voltage Drops?
Summary: The presenter explains what happens to a motor when voltage drops, explaining that the amperage also goes down, but not as fast as the voltage. He explains that this causes the motor to run less efficiently and generate more heat, ultimately potentially stalling.

[19:56] Practical Implications of Incorrect Voltage
Summary: The presenter emphasizes that motors should not be run at voltages lower than their design parameters, explaining that this can reduce capacity and efficiency. He mentions that some manufacturers provide charts showing the impact of running equipment at lower voltages.

[20:47] Conclusion: Understanding Electrical Variables
Summary: The presenter concludes by emphasizing that in electrical math, changing one variable often changes others. He addresses the common misconception that lower voltage always leads to higher amperage, explaining that motors are designed differently to achieve the same horsepower at different voltages, and highlights the difference between PSC and ECM motors.

[22:57] Q&A: Power Surges, Capacitors, and More
Summary: The presenter opens the chat for a Q&A session, answering questions about power surges, surge protectors, capacitor sizing, and other electrical topics.""",

'output': """<scratchpad>
Step-by-step Analysis and Reasoning:

1. **Review of Course Outline**
   - The course is designed for entry-level HVAC technicians and covers foundational concepts in electricity and magnetism. Specifically:
     - **Electrons in Electricity** (fundamental electrical concepts, electrical charge, conductors, insulators, dielectrics)
     - **Electrical Basic Terms** (voltage, amps, resistance, power)
     - **Magnetism** (magnetic principles, coils, transformers)

2. **Check Each Video Chapter Against the Course Outline**
   - The primary question: “Does this chapter provide a standalone, 100% relevant explanation of these foundational topics at an introductory level and in a manner that can be directly inserted into the course?”

   **[00:00] Intro and Myth Introduction**
   - Focuses on banter, setting up the myth-busting topic (volts and amps in HVAC).
   - Does not provide fundamental, structured, or stand-alone coverage of course topics (like purely defining voltage, current, resistance, or magnetism).
   - **Conclusion**: Not sufficiently focused on an introductory explanation; too much informal intro and not purely educational for the course.

   **[00:57] Myth Busting Time: Voltage, Amps, and Heat Strips**
   - Talks about the difference between resistive loads and inductive loads (motors), referencing real-world scenarios (heat strips).
   - May seem related to “Electrical Basic Terms” (voltage, current, resistance), but it does so in a context that presumes some prior HVAC knowledge (e.g., heat strips, inductive reactance in motors).
   - The style is less about fundamentals and more about correcting a misconception in the field.
   - **Conclusion**: Partially covers relevant terms but not at an entry-level, standalone format.

   **[04:07] The Mistake: A Real-World Example with Heat Strips**
   - Discusses a mistake made with a heat strip restring kit, referencing data tags, decreases in voltage/amps, etc.
   - He uses this to illustrate practical knowledge about how watts and current change with voltage in resistive loads.
   - This is advanced, real-world HVAC troubleshooting rather than a basic introduction to the concept of resistance or voltage.
   - **Conclusion**: Partially touches on “voltage,” “amps,” and “power,” but again it’s not a direct fundamental explanation suitable for entry-level theory lessons.

   **[06:00] Voltage Drop on Resistive Loads: Amps and Watts**
   - Explains why reducing voltage lowers amperage and wattage, referencing personal anecdotes about cutting a heat strip.
   - Technically covers voltage, current, resistance, and power relationships, but all within a heat strip context.
   - Could be conceptually related to “Electrical Basic Terms,” but the emphasis is more on applying those terms in a specialized HVAC scenario rather than providing a self-contained educational module.
   - **Conclusion**: Partially relevant but does not present the basics alone; it’s an applied scenario.

   **[08:07] The Truth About Resistance and Amperage**
   - Focuses on how changing resistance changes amperage, busting the myth that decreased voltage = increased amperage for resistive loads.
   - Again, covers core relationships, but specifically from the angle of busting a misconception in HVAC.
   - **Conclusion**: Partially relevant but lacks the universal, entry-level structure required by the course outline.

   **[09:56] Heat Strip Experiment: Proving the Point**
   - Demonstrates reduced voltage leads to reduced amperage on a DC power supply with a heat strip.
   - Emphasizes “real-world” demonstration rather than fundamental, from-scratch definitions or theory.
   - **Conclusion**: Still specialized and not a purely educational segment on fundamental definitions.

   **[10:58] Understanding Ohm’s Law in Practice**
   - Talks about Ohm’s law, but underscores complexity in real-world (changing resistance with temperature, etc.).
   - This goes beyond the simpler, fundamental coverage that might be found in a basic course section on Ohm’s law.
   - **Conclusion**: Could conceptually align with “fundamental electrical concepts,” but it’s heavily geared toward advanced practical scenarios.

   **[13:46] Light Bulbs and Real-World Resistance**
   - Example of how a light bulb’s resistance changes as it heats up, further illustrating complexities.
   - Again, interesting advanced phenomenon but not a straightforward introduction.
   - **Conclusion**: Not suitable for direct insertion as a basic-level segment.

   **[14:55] Inductive Loads: The Complexity of Motors**
   - Moves into inductive reactance and motor windings, referencing compressors.
   - Potentially relevant to “Magnetism” or “Magnetic components – coils,” but the discussion is at a more advanced practical level.
   - **Conclusion**: Not a clean match for a fundamental introduction to magnetism or coils.

   **[16:11] Inductive Reactance and Motor Behavior**
   - Delves deeper into inductive reactance, slip in motors, motor current draw.
   - All advanced material for entry-level technicians wanting fundamental definitions (the course specifically lists simpler topics like “Magnetic principles in electricity,” not in-depth slip or inductive reactance calculations).
   - **Conclusion**: Too specialized for direct use.

   **[17:37] What Happens to Motors When Voltage Drops?**
   - Continues advanced motor-related discussion.
   - Not providing a foundational theory lesson—rather, an experienced technician’s viewpoint on real-world voltage drops.
   - **Conclusion**: Not a direct fit for a fundamental course piece.

   **[19:56] Practical Implications of Incorrect Voltage**
   - Continues advanced motor/equipment performance discussion.
   - Not a basic introduction or “101-level” concept.
   - **Conclusion**: Not directly matching the course’s fundamental aims.

   **[20:47] Conclusion: Understanding Electrical Variables**
   - Summarizes the complexities of changing one electrical variable in real-world HVAC scenarios.
   - Mentions motors designed for different voltages, PSC vs. ECM motors.
   - **Conclusion**: More advanced than a purely fundamental approach, not a direct basic explanation for novices.

   **[22:57] Q&A: Power Surges, Capacitors, and More**
   - Q&A session about various advanced or tangential topics, e.g., surge protectors, capacitor sizing.
   - Does not match the course outline’s fundamental coverage.
   - **Conclusion**: Not a match for the foundational topics in the course.

3. **Overall Match Assessment**
   - The video heavily focuses on real-world HVAC troubleshooting and myth-busting rather than purely foundational definitions or lessons.
   - While there is mention of voltage, amperage, resistance, power, and inductive/magnetic concepts, all are framed at a more advanced real-world application level.
   - The course needs “as-is” content that focuses on fundamental definitions and explanations for beginners.
   - Given the video’s context (HVAC myth-busting), these chapters would almost certainly require additional explanation or editing to align with an entry-level course.
   - Therefore, even though the video addresses some relevant terms (voltage, amps, resistance, etc.), it does not do so in a self-contained, purely introductory way.

4. **Final Conclusion**
   - The video is **Partially Relevant** (it touches on relevant electrical concepts), but none of the chapters can be used “as is” without modification in an entry-level fundamentals course.

</scratchpad>

<video_relevance>
Partially Relevant
</video_relevance>

<proposed_chapters_to_include>

</proposed_chapters_to_include>"""
}


@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Identify Relevant Chunks",
    "function_name": "get_relevant_chunks",
    "user_id": st.session_state.get("role", "anonymous")
})
def get_relevant_chunks(course_name, target_audience, course_outline, video_title, chapter_summaries, llm='gemini_2_flash'):
    """
    This function identifies relevant sections of a video based on its relevance to a course outline.

    :param course_name: The course name.
    :param target_audience: The target audience.
    :param course_outline: The outline of the course.
    :param video_title (str): The title of the video.
    :param chapter_summaries (str): The chapter summaries of the video.
    :return dict: The classification of the video.
    """
    get_relevant_chunks_agent = Chain(llm=llm, tags=['scratchpad', 'video_relevance', 'proposed_chapters_to_include'])

    get_relevant_chunks_agent.add_message(
        role="user",
        content=get_relevant_chunks_prompt.format(
            course_name=relevant_chunk_example['course_name'],
            target_audience=relevant_chunk_example['target_audience'],
            course_outline=relevant_chunk_example['course_outline'],
            video_title=relevant_chunk_example['video_title'],
            chapter_summaries=relevant_chunk_example['chapter_summaries']
        )
    )

    get_relevant_chunks_agent.add_message(
        role="ai",
        content=relevant_chunk_example['output']
    )

    get_relevant_chunks_agent.add_message(
        role="user",
        content=get_relevant_chunks_prompt.format(
            course_name=course_name,
            target_audience=target_audience,
            course_outline=course_outline,
            video_title=video_title,
            chapter_summaries=chapter_summaries
        )
    )

    response = get_relevant_chunks_agent.run()

    return response


def run_get_relevant_chunks(sheet, worksheet_name, course_name, target_audience, llm='gemini_2_flash'):
    """
    This function gets relevant chunks for a given video

    :param sheet: The sheet object.
    :param worksheet_name: The worksheet name.
    :param course_name: The name of the course.
    :param target_audience: The target audience of the course.
    :param llm: The language model to use.
    :return: None
    """
    videos_research_sheet, videos_research_df = get_sheet_data_and_df(sheet, worksheet_name)
    video_chunks_sheet, video_chunks_df = get_sheet_data_and_df(sheet, 'Video Chunks')
    rough_outline_sheet, rough_outline_df = get_sheet_data_and_df(sheet = sheet, sheet_name = "Base Outline")

    # Get the course outline
    course_outline = get_outline_with_los(
        df = rough_outline_df,
        include_learning_objectives = True
    )

    if 'chapter_summaries' not in videos_research_df.columns:
        print("Chapter summaries already created")
        for ind, row in videos_research_df.iterrows():
            video_id = row['video_id']
            if video_id in video_chunks_df['video_id'].values:
                metadata = json.loads(video_chunks_df[video_chunks_df['video_id'] == video_id].iloc[0]['metadata'])
                videos_research_df.loc[ind, 'chapter_summaries'] = metadata['chapter_summaries']


    if 'video_relevance' not in videos_research_df.columns:
        videos_research_df[['video_scratchpad', 'video_relevance', 'proposed_chapters_to_include', 'Manual Review', 'Used for']] = ''
    elif videos_research_df['video_relevance'].notna().any():
        print("Relevant chunks already created - Skip this step")
        return
  
    # Prepare for parallel processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers = 5) as executor:
        # Submit tasks for each row
        for index, row in videos_research_df.iterrows():
            
            if 'irrelevant' in row['video_verdict'].lower() or row['video_verdict'] == '' or row['video_relevance']:
                continue
            
            # Submit the task
            future = executor.submit(
                get_relevant_chunks, 
                course_name, 
                target_audience, 
                course_outline, 
                row['title'], 
                row['chapter_summaries'], 
                llm
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
            index = futures_map[future]
            response = future.result()

            # Update the df
            videos_research_df.loc[index, ['video_scratchpad', 'video_relevance', 'proposed_chapters_to_include']] = [response['scratchpad'], response['video_relevance'], response['proposed_chapters_to_include']]

            # Update progress
            progress.update()

            # Check if we should save
            if progress.should_save():
                print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
                save_to_sheet(worksheet = videos_research_sheet, df = videos_research_df)

    # Final save to sheet after all tasks
    print('All rows processed. Saving final DataFrame to sheet.')
    save_to_sheet(worksheet = videos_research_sheet, df = videos_research_df)

    # Apply filter on sheet to only show rows with relevant chunks
    filter_non_blank_column(videos_research_sheet, column_name = 'proposed_chapters_to_include', df = videos_research_df)

    return


def manual_input_mark_relevant_videos(sheet, worksheet_name, skip_manual_step = False):
    """
    Checks whether the user has properly added manual inputs for marking relevant videos

    :param sheet: The Google Sheets object.
    :param worksheet_name: The name of the worksheet.
    :param skip_manual_step: Bool - Fill the relevant rows with "Skipped - Default - Yes" if True
    :return: True if both columns are properly populated, raises an error otherwise.    
    """

    # Get the sheet and DataFrame
    video_research_sheet, videos_research_df = get_sheet_data_and_df(sheet, worksheet_name)

    if skip_manual_step:
        print("Skipping Manual Review - Mark Relevant Videos")
        mask = videos_research_df["proposed_chapters_to_include"] != ""
        # Set col values
        videos_research_df.loc[mask, "Manual Review"] = "Skipped - Default - Yes"
        videos_research_df.loc[mask, "Used for"] = "Skipped - Default - Just Content"
        # Save changes to sheet
        save_to_sheet(worksheet = video_research_sheet, df = videos_research_df)
        
        return True

    # Check if user has properly added inputs - for both Manual Review and User for columns
    validate_column_values(
        df = videos_research_df,
        filter_column = "proposed_chapters_to_include",
        validation_column = "Manual Review",
        valid_values = ["Yes", "No"],
        case_sensitive = False,
        require_populated = False
    )

    # Ensure atleast one row should be filled
    if "Yes" not in videos_research_df[videos_research_df["proposed_chapters_to_include"] != ""]["Manual Review"].values:
        raise ValueError(f"No videos marked as `Yes`. Ensure to mark atleast one or more videos as `Yes` within the `Manual Review` column in the `Videos Research` sheet - {video_research_sheet.url}")

    validate_column_values(
        df = videos_research_df,
        filter_column = "proposed_chapters_to_include",
        validation_column = "Used for",
        valid_values = ["Just Content", "As A Video"],
        case_sensitive = False,
        require_populated = False
    )

    return True


def delete_relevant_chunks(sheet, worksheet_name="Videos Research"):
    """Remove columns related to relevant chunk identification and clear filters."""
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    # Clear all filters before modifying columns
    clear_all_filters(ws)
    cols = [
        "video_scratchpad",
        "video_relevance",
        "proposed_chapters_to_include",
        "Manual Review",
        "Used for",
    ]
    cols = [c for c in cols if c in df.columns]
    if cols:
        df = df.drop(columns=cols)
        clear_worksheet(ws)
        save_to_sheet(ws, df)


def delete_mark_relevant_videos(sheet, worksheet_name="Videos Research"):
    """Clear manual review columns without dropping them."""
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    for col in ["Manual Review", "Used for"]:
        if col in df.columns:
            df[col] = ""
    save_to_sheet(ws, df)

