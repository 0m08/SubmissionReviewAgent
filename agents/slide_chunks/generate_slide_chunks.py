from services.sheets_service import get_sheet_data_and_df, save_to_sheet, clear_worksheet, delete_worksheet, get_worksheet_names
from modules.chain import Chain
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
from services.smart_progress_bar import SmartProgressBar
from langsmith import traceable
import re


slide_chunks_generation_prompt = """We are developing structured instructional slides for an E-learning course. You are a Slide Chunking Agent responsible for converting research-based content into slide-sized instructional chunks. Your task is to segment the given research notes into a sequence of clear, pedagogically sound slide units that align with the subtopic's learning objectives, maintain a coherent instructional flow, and reflect the broader topic context.

Below is the course information for which the research notes were generated:

<course_information>
Course name: {course_name}
Target audience: {target_audience}
</course_information>

Below are the topic and subtopic to which the research notes belong:

<topic>
{topic}
</topic>

<subtopic>
{subtopic}
</subtopic>

Below are the full research notes for this subtopic. Each research note is grouped under the specific learning objective it is intended to support:

<research_notes>
{research_notes}
</research_notes>

Use the following instructions to break down the research notes into
clear, slide-ready units aligned with the given subtopic:

<instructions>

1. Understand the Structure of the Research Notes:

Before chunking the content into slides, carefully examine the format of
each research note. Each block of research content is provided alongside
its associated learning objective and may follow one of the two formats
described below:

a. Document-Derived Format
- This format is structured like an article or summary with clear sections, bullet points, or numbered sub-points.
- It may include explanations, facts, cause-effect relationships, or short direct quotes from documents.
- Paragraphs may be logically organized around key instructional points.

b. Transcript-Derived Format
- This format consists of line-by-line transcript segments extracted from one or more videos, with each line prefixed by a timestamp in seconds (e.g., 266, 268).
- The tone could be conversational and informal, with natural pauses, incomplete sentences, or casual phrasing typical of spoken language.
- You may encounter transcript segments from multiple different videos within the same learning objective, each with their own Video_Id and timestamp ranges.

You may encounter either or both formats within a subtopic. Your job is to identify the format for each section and respect that format's tone, structure, and pacing while chunking into slides.

2. Understand What Constitutes a Slide Chunk:

The structure and style of slide chunks will depend on whether the
source content is document-derived or transcript-derived. Follow the
corresponding guidance based on the format of the research notes.

a. For Document-Derived Research Notes
- A slide chunk is a self-contained instructional unit designed to fit on a single E-learning slide.
- It should convey one clear, teachable idea or a set of closely related points with sufficient depth and context.
- The content should be substantial enough to provide meaningful instruction, typically around 90-120 words per slide, ensuring sufficient depth without becoming too dense.
- Each slide chunk should be pedagogically meaningful---either introducing, explaining, or elaborating on a specific concept with enough detail to be truly instructive.
- Paraphrase the source content into clear, accessible, and instructional language that builds understanding progressively.

b. For Timestamped Transcript-Derived Research Notes
- A slide chunk is a meaningful sub-segment of the original transcript that maintains a natural conversational flow.
- Do not rewrite or paraphrase the transcript content.
- Instead, your task is to divide the transcript into natural, logically flowing chunks based on shifts in ideas or instructional steps.
- Each chunk should include enough transcript content to provide substantial instruction, typically around 45-75 seconds, to preserve the natural speaking rhythm.
- Create multiple slide chunks from the same video if the video content is long. Each resulting video chunk that you create should have its own Start and End timestamps and should not be more than 75-90 seconds long.

- For each chunk, return:
  i) A descriptive but concise Slide Title
  ii) The original Video Id from which the chunk is extracted
  iii) The Start and End timestamps for the chunk
  iv) The original Transcript lines along with their timestamps (do not edit the content).

3. Ensure Alignment with the Learning Objective:
- Every slide chunk you generate must support the provided learning objective.
- Do not include tangential information. Before creating each slide, ask: "Does this idea help the learner achieve the learning objective?"
- If a portion of the research is interesting but not directly relevant, exclude it.
- If multiple segments support the same instructional point, group or merge them into a single coherent chunk.
- For transcript-derived notes, include every line in order without omission.

4. Do Not Add Unverified or External Information:
- You must only use the content present in the provided <research_notes> block.
- Do not invent, speculate, or introduce additional facts or analogies unless they are already implied or explicitly stated in the source content.
- Your role is to restructure existing content, not to expand it with new material.

5. Ensure Logical Flow and Instructional Storytelling:

a. For Document-Derived Research Notes
- **Adopt a Professional, Mentor-Like Tone:** The language should be clear, direct, and empowering, as if a senior technician is guiding a new team member through a real-time walkthrough. Avoid overly casual slang while maintaining approachability. Never instruct the learner to "call a professional" or "consult a technician" - you are teaching them to become that professional.
- **Use Action-Oriented Verbs:** Use strong, hands-on verbs that reflect on-the-job tasks. Prefer words like "spot," "check," or "find" over "identify" or "assess."
- **Simplify Technical Language:** Actively rewrite complex technical descriptions into clear, direct statements that are easy to visualize and understand. Break down industry jargon into accessible terms.
- **Build a Narrative:** Weave the points into a story. Follow a natural progression from **problem -> process -> consequence.** Show *how* and *why* things happen. Connect the steps to the technician's workflow.
- **Strategic Question Placement:** Use questions sparingly and strategically throughout the content, not just at slide beginnings. Mix questions with direct, hands-on instructions to create variety and avoid predictable patterns.
- **Provide Visual Guidance:** "Walk" the learner through the process with your words. Describe what they will see, where they should look, and what to watch out for.
- **Frame as Real-Time Walkthrough:** Present content as if the learner is performing the task right now. Use present-tense, action-oriented language that makes them feel they are actively engaged in the process.
- **Use Analogies for New Terms:** When introducing new technical terminology, consider using analogies or comparisons to familiar concepts to enhance understanding.
- **Eliminate All Redundancy:** Remove filler words and avoid restating concepts, even when phrased differently. Every sentence must add new value or advance the instruction.

b. For Timestamped Transcript-Derived Research Notes
- Maintain the original speaker sequence and line order.
- Your job is to segment the transcript into logical, meaningful chunks that reflect a natural flow of ideas as spoken. Allow for longer chunks to preserve conversational flow.

6. Follow the Prescribed Slide Structure and Types:

For each subtopic, you must generate exactly three types of slides: a
Transition Slide, multiple Content Slides, and a Summary Slide.

a. Transition Slide (Exactly One)
- This is the first slide. It introduces what the learner will be doing in the subtopic.
- Review all research notes to identify the main instructional points.
- Since learning objectives already set expectations, avoid preview-style introductions starting with "In this section..." Instead, create engaging hooks that grab attention and fit the topic naturally.
- **Hook Variety Guidelines:**
  * Avoid overused patterns: "Imagine you're...", "Ever done this?", "Have you ever wondered...", "Think about a time...", "Picture this..."
  * Use diverse hook types: surprising facts/statistics, targeted questions, scenario statements, real-life on-the-job situations, clear value statements, or practical use cases
  * Ensure hooks are relevant to the topic, match learner context (HVAC/electrical/on-site scenarios), use believable job situations, and highlight problems the lesson will solve
  * Connect to customer impact, system reliability, or safety when possible
- Write the slide in your own instructional language using fresh, varied approaches.
- The tone should be clear and audience-friendly.

b. Content Slides (One or More)
- These are the main instructional slides.
- **Chunk by Task, Not by Fact:** Group information based on how a technician works. If a tech looks at a component's appearance and location simultaneously, teach it that way. Merge slides that support a single, unified task.
- Each slide chunk should be substantial enough to provide meaningful instruction (around 90-120 words for document-derived content).
- Ensure each slide contributes to the learning objective with enough depth.
- **Craft Engaging Titles:** Titles should be clear, concise, grab attention, and sound practical or answer a "why this matters" question.
- **Vary Slide Title Formats:** Use a mix of styles to improve engagement and prevent repetitive structure across slides. Rotate between:
  * **Question format** (e.g., "What's Causing That Capacitor to Overheat?")
  * **Action-oriented** (e.g., "Stop Shorts Before They Start")
  * **Direct, attention-grabbing statements** (e.g., "Three Signs Your Wiring Needs Attention")
  * **Problem-solution framing** (e.g., "Loose Connections = System Failures")
  * **Practical or location-focused** (e.g., "Where Wires Fail Most Often")
- Avoid using the same title format across multiple slides in a row.
- **Structure Comparisons Clearly:** For slides comparing multiple items or concepts, frame the comparison clearly at the beginning to establish context.

c. Summary Slide (Exactly One)
- This is the final slide. It provides a concise, memorable recap.
- **Do not use bullet points.**
- The summary should be concise and laser-focused. Highlight only the absolute key takeaway and its practical application. Avoid rehashing content already covered.
- Frame it around how the technician will *use* the skills they just learned. Do not start with "In this section, we covered..."

7.  Slide Output Format:
    All slide outputs must follow a structured format.


    a. For Document-Derived Slides:
       Use the following format:


       Title: [Slide title]
       Slide Type: [Transition / Content / Summary]
       Content: [Detailed instructional content in natural language]


    b. For Timestamped Transcript-Derived Slides:
       Use the following format:


       Title: [Slide title]
       Slide Type: [Video]
       Video_Id: [Video ID from the input]
       Start: [Start timestamp]
       End: [End timestamp]
       Transcript:
         - '[timestamp]': [line of transcript]
         - '[timestamp]': [line of transcript]
         ...


8.  Ensure Full Coverage of All Input:
    The final slide set must comprehensively reflect all research notes provided for the subtopic. Avoid skipping content unless it’s clearly off-topic or redundant.


</instructions>


Provide your output strictly in the following format:

<output>

<evaluation_breakdown>
Use this section to plan how you will chunk the input research notes into slides. This is your internal reasoning space.

You must address the following points:

1. Subtopic Understanding: Summarize what the learner is expected to understand or do.
2. Source Format Analysis: Identify the format of the research notes.
3. Slide Structure Plan: Describe your plan for the Transition, Content, and Summary slides.
4. Chunking and Coverage Strategy: Explain how you will ensure all objectives and notes are covered.
</evaluation_breakdown>

<slides>
Provide all slides in sequence, starting with a Transition Slide, followed by one or more Content Slides, and ending with a Summary Slide. Ensure each slide is separated by a full blank line.
</slides>

</output>

IMPORTANT: Your response must end with the closing </output> tag. Do not include any text after the </output> tag.
"""


slide_chunks_generation_few_shot_prompt = """Follow the example below to understand how to generate a complete and well-structured set of slides based on a subtopic's learning objectives and research notes. Note - This is only a demonstration. Do not copy or reuse slide content from this example - even if the course name, topic, subtopic, or learning objectives appear similar. Always generate output based solely on the actual input and the research notes provided.

<input>

<course_information>
Course name: Condenser Maintenance
Target audience: Entry-level HVAC technicians
</course_information>

<topic>
Preparation for Fan Removal
</topic>

<subtopic>
Preparation
</subtopic>

<research_notes>
... [same research notes as provided in the previous prompt] ...
</research_notes>

</input>

<output>

<slides>

Title: Getting Ready for a Safe Fan Removal
Slide Type: Transition
Content: Before you even think about touching that fan, there are a few key prep steps we need to walk through. In this section, you'll learn how to be certain the power is off, not just guess. We'll also cover the right way to take photos of your setup so you have a roadmap for putting it all back together, and finally, how to give the fan blade a good inspection to spot any problems before they start. Getting this prep work right is what separates a smooth job from a frustrating one.

Title: Setting Up the Voltmeter and Initial Safety Checks
Slide Type: Video
Video_Id: sI_569t7HmE
Start: 2
End: 42
Transcript:
- '2': All right. And now that we got
- '3': everything off, again, before we start
- '6': messing with anything, we are going to
- '8': verify with a voltmeter. So, first
- '12': going to go ahead and turn it to volts
- '13': DC. That's the little straight line with
- '16': the little dashes on it. And again, it
- '18': just bounces around a little bit between
- '20': a few volts. But, uh, we're going to go
- '22': throw it on our battery bus bars here
- '24': between the positive and negative.
- '30': saying I got 12 mill volts, which is
- '33': essentially
- '34': nothing. We're going to check our
- '36': battery terminals up
- '38': here. I've got 45 m
- '42': volts. We're going to check our PV

Title: Completing Voltage Verification Across All System Components
Slide Type: Video
Video_Id: sI_569t7HmE
Start: 42
End: 92
Transcript:
- '42': volts. We're going to check our PV
- '47': [Music]
- '49': conductors. I got zero
- '53': volts.
- '55': Going to check all of our
- '58': MPPPTs. Zero
- '64': volts. This one's got 12 mill
- '69': [Music]
- '71': volts. Let's go ahead and check uh you
- '74': know ground to
- '77': neutral zero volts.
- '80': Ground the
- '84': hot 7 mill volts. Ground the
- '88': negative a few
- '90': milli volts. I think we're safe to say
- '92': we're good to work on this.

Title: Why You Need to Document Your Work?
Slide Type: Content
Content: So, why do we need to stop and take pictures before we start wrenching? Because HVAC systems can have tricky wiring, and the fan blade has to sit just right in its housing for the system to move air correctly. If you don't record the setup, you might cross a wire during reassembly, leading to electrical problems, or install the fan at the wrong height, which kills the unit's cooling power. Taking a few minutes to document everything now with clear photos and notes will save you from major headaches and hours of troubleshooting later. It's your roadmap for getting the job done right.

Title: Spotting Fan Blade Problems Before You Start
Slide Type: Content
Content: Before you do any work, it's time to play detective with the fan blade. A good inspection now can save you from a callback later. So, what are you looking for? First, give each blade a close look for any obvious physical damage---things like bends, warping, or even small cracks. A bent blade won't move air efficiently, and a cracked one is a serious safety risk because it could fly apart during operation. Next, check for rust or corrosion, which can weaken the blade. Finally, grab the blade and gently check for any looseness in the rivets or bolts holding it to the hub. Any of these issues could be the root cause of a noise complaint or performance problem.

Title: Making the Call: When to Replace a Fan Blade
Slide Type: Content
Content: After your inspection, you have to make a judgment call: does this blade need to be replaced? If you spot any major cracks, fractures, or significant corrosion, the decision is easy---replace it immediately. There's no fixing that kind of damage. For minor issues, like small bends or light surface rust, you have to consider the impact. Will this cause a vibration? Is it affecting airflow? A damaged blade can reduce cooling efficiency and put extra strain on the motor. When you're in doubt, the safest and most professional move is always to replace the blade. It's better to explain the need for a new part to the customer now than to get a call back for a bigger failure down the line.

Title: Your Key Takeaway
Slide Type: Summary
Content: Remember, taking a few extra minutes before you start the real work to check for power, document your setup, and inspect the parts isn't just about following steps---it's about controlling the job so you can work safely and avoid problems before they happen.

</slides>

</output>

"""


@traceable(
    metadata={
        "agent_name": "slide_chunks",
        "step_name": "Generate Slide Chunks from the Research Notes",
        "function_name": "generate_slide_chunks_from_research_notes",
        "user_id": st.session_state.get("role", "anonymous")
    }
)
def generate_slide_chunks_from_research_notes(course_name, target_audience, topic, subtopic, research_notes, llm="gemini_2_flash"):
    """
    Generate slide chunks for a subtopic based on research notes using a slide chunking agent.

    :param course_name: Name of the course.
    :param target_audience: Intended audience of the course.
    :param topic: Topic name to which the subtopic belongs.
    :param subtopic: Subtopic name.
    :param research_notes: Full structured research notes for the subtopic.
    :param llm: Language model to use.
    :return: The full <output> block containing from the agent response.
    """
    # Initialize the agent
    slide_chunking_agent = Chain(llm = llm, tags = ['output'])

    # Format the prompt for debugging (printing)
    formatted_prompt = slide_chunks_generation_prompt.format(
            course_name = course_name,
            target_audience = target_audience,
            topic = topic,
            subtopic = subtopic,
            research_notes = research_notes
        ) + slide_chunks_generation_few_shot_prompt

    # Print the formatted prompt for debugging
    print("\n🔍 Slide Chunk Prompt Being Sent to LLM:\n")
    print(formatted_prompt)
    print("\n" + "=" * 100 + "\n")

    # Add prompt with formatted inputs
    slide_chunking_agent.add_message(
        role = "user",
        content = slide_chunks_generation_prompt.format(
            course_name = course_name,
            target_audience = target_audience,
            topic = topic,
            subtopic = subtopic,
            research_notes = research_notes
        ) + slide_chunks_generation_few_shot_prompt
    )

    # Run the agent
    response = slide_chunking_agent.run()

    return response['output']


@traceable(
    metadata={
        "agent_name": "slide_chunks",
        "step_name": "Generate Slide Chunks from the Research Notes",
        "function_name": "generate_slide_chunks_from_research_notes_for_all_subtopics",
        "user_id": st.session_state.get("role", "anonymous")
    }
)
def generate_slide_chunks_from_research_notes_for_all_subtopics(sheet, sheet_name, llm="gemini_2_flash", max_workers=5):
    """
    Generates the slide chunks for each unique subtopic in the Final Outline sheet
    
    :param sheet: The gspread sheet object.
    :param sheet_name: The worksheet name.
    :param llm: The language model to use .
    :param max_workers: Number of parallel workers (default 5).
    :return: None
    """

    # Fetch course info
    _, course_info_df = get_sheet_data_and_df(sheet, "Course info")
    course_name = course_info_df.loc[0, "Course Name"]
    target_audience = course_info_df.loc[0, "Target Audience & Industry"]

    # Load the worksheet and DataFrame
    worksheet, df = get_sheet_data_and_df(sheet, sheet_name)

    # Ensure slide_chunks column exists and is empty
    if "slide_chunks" not in df.columns:
        df["slide_chunks"] = ""
  
    # Identify unique Topic-Subtopic combinations in order of appearance
    subtopic_first_indices = df.drop_duplicates(["Topic", "Subtopic"], keep="first").index.tolist()
    subtopic_names = [f"{df.loc[idx, 'Topic']}-{df.loc[idx, 'Subtopic']}" for idx in subtopic_first_indices]

    # For each unique subtopic, gather all rows for that subtopic (in order)
    subtopic_to_rows = {}
    for idx, subtopic_key in zip(subtopic_first_indices, subtopic_names):
        topic, subtopic = subtopic_key.split("-")
        subtopic_to_rows[subtopic_key] = df[(df["Topic"] == topic) & (df["Subtopic"] == subtopic)]

    # For each unique subtopic, get topic and subtopic from first row
    subtopic_to_topic = {key: key.split("-")[0] for key in subtopic_names}
    subtopic_to_subtopic = {key: key.split("-")[1] for key in subtopic_names}

    # Helper to construct research_notes string for a subtopic
    def construct_research_notes(rows):
        blocks = []
        for _, row in rows.iterrows():
            lo = row["Learning Objectives"]
            rn = row["research_notes"]
            blocks.append(f"<learning_objective>\n\nLearning Objective: {lo}\n\n<research_note>\n\n{rn}\n\n</research_note>\n\n</learning_objective>")
        return "\n\n".join(blocks)

    # Function to run the agent and extract <slides> for a subtopic
    def process_subtopic(subtopic_key):
        topic = subtopic_to_topic[subtopic_key]
        subtopic_val = subtopic_to_subtopic[subtopic_key]
        research_notes = construct_research_notes(subtopic_to_rows[subtopic_key])
        output = generate_slide_chunks_from_research_notes(
            course_name, target_audience, topic, subtopic_val, research_notes, llm=llm
        )
        # Extract <slides>...</slides>
        match = re.search(r"<slides>(.*?)</slides>", output, re.DOTALL)
        slides_text = match.group(1).strip() if match else ""
        return slides_text

    # Parallel processing for all unique subtopics
    results = [None] * len(subtopic_names)
    progress = SmartProgressBar(total_tasks=len(subtopic_names), description="Generating slide chunks", save_interval=5)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(process_subtopic, subtopic_key): idx for idx, subtopic_key in enumerate(subtopic_names)}
        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            slides_text = future.result()
            results[idx] = slides_text
            completed += 1
            progress.update()
            if progress.should_save():
                # Save partial progress to sheet
                # Place results in the first row of each subtopic, blank elsewhere
                df["slide_chunks"] = ""
                for i, subtopic_idx in enumerate(subtopic_first_indices):
                    df.at[subtopic_idx, "slide_chunks"] = results[i] if results[i] is not None else ""
                save_to_sheet(worksheet, df)
    # Fill the slide_chunks column for first row of each unique subtopic, blank for the rest
    df["slide_chunks"] = ""
    for i, subtopic_idx in enumerate(subtopic_first_indices):
        df.at[subtopic_idx, "slide_chunks"] = results[i] if results[i] is not None else ""
    # Save final DataFrame to sheet
    save_to_sheet(worksheet, df)
    print("Slide chunk generation complete and saved to sheet.")


def delete_slide_chunks_generation(sheet, worksheet_name="Final Outline"):
    """
    Remove the 'slide_chunks' column from the specified worksheet (default 'Final Outline').
    :param sheet: The gspread sheet object.
    :param worksheet_name: The worksheet name (default 'Final Outline').
    :return: None
    """
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    if "slide_chunks" in df.columns:
        df = df.drop(columns=["slide_chunks"])
        clear_worksheet(ws)
        save_to_sheet(ws, df)
    
    # Delete the checklist backup sheet if it exists
    backup_name = "Backup Slide Chunks Sheet for Delete step of Slide Chunks Checklist"
    sheet_names = get_worksheet_names(sheet)
    if backup_name in sheet_names:
        delete_worksheet(sheet, backup_name)
        print(f"🗑️ Deleted backup sheet '{backup_name}' when deleting slide chunks generation")
