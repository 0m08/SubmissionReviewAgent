from services.sheets_service import get_sheet_data_and_df, save_to_sheet, clear_worksheet
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

Use the following instructions to break down the research notes into clear, slide-ready units aligned with the given subtopic:

<instructions>

1. Understand the Structure of the Research Notes:
  Before chunking the content into slides, carefully examine the format of each research note. Each block of research content is provided alongside its associated learning objective and may follow one of the two formats described below:

    a. Document-Derived Format
      - This format is structured like an article or summary with clear sections, bullet points, or numbered sub-points.
      - It may include explanations, facts, cause-effect relationships, or short direct quotes from documents.
      - Paragraphs may be logically organized around key instructional points.

    b. Transcript-Derived Format
      - This format consists of line-by-line transcript segments extracted from a video, with each line prefixed by a timestamp in seconds (e.g., 266, 268).
      - The tone could be conversational and informal, with natural pauses, incomplete sentences, or casual phrasing typical of spoken language.

  You may encounter either or both formats within a subtopic. Your job is to identify the format for each section and respect that format's tone, structure, and pacing while chunking into slides.

2. Understand What Constitutes a Slide Chunk:
  The structure and style of slide chunks will depend on whether the source content is document-derived or transcript-derived. Follow the corresponding guidance based on the format of the research notes.

    a. For Document-Derived Research Notes
      - A slide chunk is a self-contained instructional unit designed to fit on a single E-learning slide.
      - It should convey one clear, teachable idea or closely related set of points.
      - The content in a chunk should be concise enough to be narrated within approximately 30 seconds, which typically translates to 70–80 words per slide based on a natural speaking rate.
      - Each slide chunk should be pedagogically meaningful — either introducing, explaining, elaborating, or summarizing a specific concept.
      - Paraphrase the source content into clear, accessible, and instructional language suitable for slide narration.

    b. For Timestamped Transcript-Derived Research Notes
      - A slide chunk is a meaningful sub-segment of the original transcript.
      - Do not rewrite or paraphrase the transcript content.
      - Instead, your task is to divide the transcript into a sequence of natural, logically flowing chunks based on shifts in ideas, instructional steps, or topic transitions.
      - Each chunk should include only as much transcript content as can reasonably be narrated within approximately 30 seconds, which typically translates to 70–80 words per slide based on a natural speaking rate.
      - For each chunk, return:
          i) A descriptive but concise Slide Title
          ii) The original Video Id
          iii) The Start and End timestamps for the chunk
          iv) The original Transcript lines along with the timestamps (do not edit the content)
      - The chunking should enhance learning by making the transcript easier to follow and aligned with instructional pacing.

3. Ensure Alignment with the Learning Objective:
  - Every slide chunk you generate must support the provided learning objective.
  - Do not include tangential information or background that does not meaningfully contribute to helping the learner achieve that learning objective.
  - Before creating each slide chunk, ask: “Does this idea help the learner progress toward the learning objective?”
  - If a portion of the research notes is interesting but not directly relevant, it should be excluded from the final output.
  - If multiple segments support the same instructional point, group or merge them into a single coherent chunk rather than repeating the same idea across multiple slides.
  - For transcript-derived research notes:
      - You must include every transcript line in your output — do not skip or omit any content.
      - Maintain the original timestamp order — do not rearrange or combine non-consecutive lines.
      - Divide the transcript into sequential, pedagogically meaningful chunks based on clear shifts in ideas or focus, while ensuring that the entire transcript is covered exactly once, in order.

4. Do Not Add Unverified or External Information:
  - You must only use the content present in the provided <research_notes> block.
  - Do not invent, speculate, or introduce additional facts, examples, analogies, or elaborations unless they are already implied or explicitly stated in the source content.
  - Avoid making assumptions beyond what the original research note conveys - even if the information seems obvious.
  - Your role is to restructure and segment existing content into slide-friendly units, not to enhance or expand it with new material.
  - Paraphrasing is encouraged when working with document-style research notes to improve clarity and instructional tone — but do not change the original meaning or introduce new information.
  - For transcript-derived notes, do not paraphrase. Instead, preserve the exact wording of each transcript line within its assigned chunk. Your job is to segment the transcript logically — not to rewrite it.

5. Ensure Logical Flow and Instructional Storytelling:

  a. For Document-Derived Research Notes
     If the research note is in document-derived format, organize the slides to create a clear, coherent instructional narrative.
      - The sequence of slides should feel like a smooth progression of ideas — each slide building naturally on the previous one.
      - The overall structure should follow an instructional flow that starts with introducing the concept, then explaining or elaborating on its components, and finally summarizing all the concepts covered
      - The content should feel cohesive and interconnected — not like a list of disconnected facts. Use transitions or bridge statements between slides if needed to help maintain continuity.
      - Use instructional language that sounds natural, engaging, and learner-friendly. The tone should be clear and slightly conversational, as if guiding a learner through a concept step-by-step.
      - Aim to subtly emulate a story-like flow, especially when the content lends itself to real-world scenarios, challenges, cause-effect relationships, or sequential processes.
      - Avoid robotic, overly mechanical, or template-like phrasing. The content should feel like it was written by a knowledgeable human instructor, not auto-generated.
      - Do not use overly formal or academic language . Write as though you are explaining to a motivated learner with no prior knowledge, while still respecting their intelligence and considering the target audience.

  b. For Timestamped Transcript-Derived Research Notes
     If the research note is in timestamped transcript format, do not paraphrase, restructure, or rewrite the content.
      - Maintain the original speaker sequence and line order.
      - Your job is to segment the transcript into logical, meaningful chunks that reflect a natural flow of ideas as spoken — without altering the language or inserting bridging text.

6. Follow the Prescribed Slide Structure and Types:
  For each subtopic, you must generate exactly three types of slides: a Transition Slide, multiple Content Slides, and a Summary Slide. The slides should collectively represent and organize the research notes for all learning objectives within that subtopic.

    a. Transition Slide (Exactly One)
       This is the first slide of the set. It introduces the learner to what they will be learning in this subtopic. You must create only one transition slide for the given subtopic, based on all the research notes provided, which may include both document-derived and timestamped transcript formats.

       Instructions:
         - Review the full set of research notes across all learning objectives within the subtopic.
         - Identify the main instructional points that will actually be covered in the content slides (i.e., the core teaching material).
         - The transition slide should only preview the ideas and concepts that will be covered in the upcoming content slides — do not include unrelated or extraneous information.
         - Write the slide in your own instructional language — do not copy or reuse lines from the transcript or document text.
         - Keep the tone clear, target audience-friendly, and appropriate for introducing the section.

       All Transition Slides must: 
         - Begin with a phrase like “In this section, we will…” or a similar preview-style opener.
         - Use concise, introductory instructional language appropriate for the target audience.
         - Be returned in the sentence-based slide format (i.e., no timestamps), even if some input research notes were in transcript format.

    b. Content Slides (One or More)
       These are the main instructional slides that break down the research notes into coherent, focused units. You will generate as many content slides as needed to cover the material, while respecting chunk size limits and instructional clarity. The content slides should represent all research notes provided for the subtopic — which may include a mix of document-style notes and timestamped transcript segments.

       Instructions:
         - Read through all the research notes provided across all learning objectives within the subtopic.
         - Break the content into logical, pedagogically sound chunks — each chunk should focus on a single core idea or tightly related set of points.
         - Each slide chunk must be short enough to be narrated within approximately 30 seconds (about 70–80 words), whether derived from document notes or timestamped transcripts.
         - Ensure that each content slide meaningfully contributes to the associated learning objective.

          i) For Document-Derived Research Notes:
            - Paraphrase the content into clear, concise instructional language — avoid copying directly.
            - Use short sentences and bullet points if necessary, but ensure the content flows naturally as a slide explanation.
            - Do not introduce any new information beyond what is present in the research notes.
            - There is no fixed limit to the number of content slides you may generate. You must ensure that all meaningful points from the research notes are reflected in the slides — even if it requires many short, focused slides. It's acceptable (and encouraged) to break the material into multiple pedagogically sound chunks to preserve clarity, pacing, and instructional value. All relevant content should be represented in the slides, even if some points receive only brief or partial coverage. Avoid skipping any portion of the research notes unless it is clearly redundant, irrelevant to the learning objective, or off-topic. 
          
          ii) For Timestamped Transcript-Derived Notes:
            - Do not paraphrase, summarize, or rewrite the transcript lines.
            - Your job is to segment the transcript into meaningful chunks using the existing timestamps.
            - Keep each chunk within a reasonable duration (roughly 30 seconds).
            - Maintain the sequence of transcript lines as given; do not reorder or omit any lines.
            - For each chunk, specify the corresponding Start, End, and include the exact transcript lines within that range.

       All Content Slides must:
          - Include a distinct and meaningful slide title, derived from the content it covers or the learning objective it supports.
          - Use "Slide Type: Content" for document-derived content, and "Slide Type: Video" for transcript-based content.
          - Clearly differentiate one slide from another - avoid redundant or overly similar titles.
          - Maintain consistent formatting according to the source type (document-style or transcript-style).

    c. Summary Slide (Exactly One)
      This is the final slide of the set. It provides a concise recap of the key instructional points covered in the content slides. You must create only one summary slide for the given subtopic, using your own instructional language to synthesize the main takeaways. This slide should reflect all the concepts actually covered in the content slides — not general ideas from the original research notes.

       Instructions:
         - Review the full set of content slides generated for the subtopic.
         - Identify the most important points or takeaways actually covered across those slides - not what was in the original research notes.
         - Do not repeat slide content verbatim or reuse full sentences. Rephrase and condense the ideas in a learner-friendly, summary-style manner.
         - Exclude any ideas that were not explicitly covered in the content slides — the summary should only reflect the material that was actually delivered in the content slides.

       All Summary Slides must:
         - Begin with a phrase like “In this section, we covered:” or a similar reflective opener.
         - Present the takeaways in a short, clear list — using either bullet points or numbered format.
         - Be returned in sentence-based format (i.e., no timestamps), even if some input research notes were in transcript format.
         - Maintain a neutral, instructional tone appropriate for E-learning — avoid overly casual or formal language.
         - Ensure the summary is meaningful on its own, providing a helpful closing recap for the learner.

7. Slide Title Guidelines:
  - Every slide, regardless of its type or source format, must include a clear, distinct, and meaningful Title.
  - Titles can be derived from either the slide content itself or the associated learning objective.
  - Do not repeat or reuse the same title across multiple slides.
  - Titles should be short, concise and instructional. While some slide types (like introductions or summaries) may use broad framing, content slide titles must clearly convey the specific idea or teaching point being addressed.

8. Slide Output Format:
  All slide outputs must follow a structured, source-sensitive format based on the type of research note the slide was derived from.

    a. For Document-Derived Slides:
      Use the following format:

      Title: [Slide title]
      Slide Type: [Transition / Content / Summary]
      Content: [Paraphrased instructional content in natural language]

      Example:

      Title: Introduction to Preventive Maintenance  
      Slide Type: Transition  
      Content: In this section, we will explore what preventive maintenance is, why it matters, and how it helps reduce unexpected equipment failures.

      Title: What is Preventive Maintenance?  
      Slide Type: Content  
      Content: Preventive maintenance involves regularly scheduled inspections and adjustments to avoid unexpected equipment failure.

      Title: Benefits of Preventive Maintenance  
      Slide Type: Content  
      Content: Regular maintenance improves system reliability, extends equipment life, and reduces costly repairs.

      Title: Key Takeaways  
      Slide Type: Summary  
      Content: In this section, we covered:  
      1. Preventive maintenance involves scheduled inspections.  
      2. It improves reliability and reduces breakdowns.  
      3. It helps minimize long-term costs.

    b. For Timestamped Transcript-Derived Slides:
      Use the following format:

      Title: [Slide title]  
      Slide Type: [Transition / Video / Summary]  
      Video_Id: [Video ID from the input]  
      Start: [Start timestamp]  
      End: [End timestamp]  
      Transcript:  
        - '[timestamp]': [line of transcript]  
        - '[timestamp]': [line of transcript]  
      ...

      Example:

      Title: Preparing for Disconnection  
      Slide Type: Transition  
      Content: In this section, we will walk through the process of safely disconnecting an HVAC unit before maintenance.

      Title: Step 1. Identify the power source  
      Slide Type: Video  
      Video_Id: bgUGUEYtNbA  
      Start: 477  
      End: 490  
      Transcript:  
        - '477': The first step is to identify the power source ...  
        - '480': Locate the power supply to the unit ...  
        - '483': Ensure it is accessible before continuing ...  

      Title: Step 2. Switch off the power  
      Slide Type: Video  
      Video_Id: bgUGUEYtNbA  
      Start: 493  
      End: 520  
      Transcript:  
        - '493': Switch off the power by disconnecting ...  
        - '497': Make sure there is no voltage present ...  

      Title: Key Takeaways  
      Slide Type: Summary  
      Content: In this section, we covered:  
      1. Identify the power source before working.  
      2. Turn off the power completely to ensure safety.

  Note: The Transition and Summary slides — even when generated from transcript-derived input — must still be written in natural instructional language and follow the document-style output format (i.e., do not include timestamps).

9. Ensure Full Coverage of All Input:
  - The final slide set must comprehensively reflect all research notes provided for the subtopic.
  - For document-derived notes: Avoid skipping content unless it’s clearly off-topic or redundant.
  - For transcript-derived notes: Do not skip any transcript lines — include every line once, in order, without omission.

</instructions>

Provide your output strictly in the following format:

<output>

<evaluation_breakdown>
Use this section to plan how you will chunk the input research notes into slides. This is your internal reasoning space — do not begin writing the final slide outputs yet. Think carefully before proceeding.

You must address the following points:

1. Subtopic Understanding 
   Summarize what the learner is expected to understand or achieve from this subtopic, based on the learning objectives and research notes.

2. Source Format Analysis 
   Identify which research notes are document-derived vs. transcript-derived. Briefly describe the structure and tone of each. Note if both formats are present.

3. Slide Structure Plan 
   Describe your overall plan for:
   - The Transition Slide: What key ideas will be previewed?
   - The Content Slides: How many content slides do you expect to generate? What instructional chunks or themes will they follow?
   - The Summary Slide: What main takeaways will be highlighted?

4. Chunking and Coverage Strategy 
   Explain how you will:
   - Ensure all learning objectives and research notes are covered.
   - Avoid skipping or blending unrelated segments.
   - Preserve order for transcript-derived content.
   - Maintain logical flow and instructional clarity.

You may also use this space to document any additional observations, reflections, or considerations — based on the <instructions> section above — that will help you generate a stronger, more instructionally aligned slide sequence and its content.

</evaluation_breakdown>

(Based on your above evaluation, provide the slides)

<slides>
Provide all slides in sequence for this subtopic, starting with a Transition Slide, followed by one or more Content Slides, and ending with a Summary Slide. 
Ensure the output strictly follows the slide formats described above in Instruction no. 8. Ensure that each slide chunk is separated by a full blank line to maintain clarity and visual distinction in the output.
</slides>

</output>

Note: Strictly remember to always enclose your entire output inside the <output> .... </output> tags.
"""


slide_chunks_generation_few_shot_prompt = """Follow the example below to understand how to generate a complete and well-structured set of slides based on a subtopic’s learning objectives and research notes. Note - This is only a demonstration. Do not copy or reuse slide content from this example - even if the course name, topic, subtopic, or learning objectives appear similar. Always generate output based solely on the actual input and the research notes provided.

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

<learning_objective>

Learning Objective: Pull the disconnect and confirm power is off with a voltmeter as a safety measure.

<research_note>

Link: https://youtube.com/shorts/sI_569t7HmE?si=lCKEGi8R9hYbZbim&start=2&end=92
Video_Id: sI_569t7HmE
Start: 2
End: 92
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

</research_note>

</learning_objective>

<learning_objective>

Learning Objective: Document the existing setup by taking photos of the wiring before disconnecting. Note the fan blade height within the shroud.

<research_note>

1. **Importance of Documentation Before Disassembly**
      1.1. Why Documenting is Crucial: Before you start disconnecting anything, it's vital to document the existing setup. This is because HVAC systems can have complex wiring, and the exact placement of the fan blade within the shroud is critical for proper airflow. Accurate documentation ensures that you can reassemble everything correctly, avoiding potential electrical issues or performance problems.
      1.2. Methods of Documentation: The two most effective methods are taking photos and making notes. Photos provide a visual record of the wiring and component placement, while notes can capture specific details or measurements that might not be clear in a photo.

   2. **Photographing the Wiring**
      2.1. Taking Clear Photos: When photographing the wiring, ensure the photos are clear and well-lit. Take multiple photos from different angles to capture all the wiring connections, including the wires connected to the contactor, capacitor, and any control boards.
      2.2. What to Capture in Photos: Focus on capturing the colors of the wires and where they connect. According to the document, "Before disconnecting any wires, make note of their locations or take photos. This can help when wiring the replacement motor." Note any labels or markings on the wires or terminals. These photos will serve as a reference when you're wiring the new motor.

   3. **Noting Fan Blade Height**
      3.1. Importance of Correct Fan Blade Height: The height of the fan blade within the shroud is crucial for optimal airflow. If the fan blade is too high or too low, it can reduce the system's cooling efficiency.
      3.2. How to Measure/Note Fan Blade Height: Before removing the fan blade, carefully observe its position within the shroud. Measure the distance from the top of the shroud to the top of the fan blade. You can also take a photo showing the fan blade's position relative to the shroud. The document states, "Install the inspected or new fan blade, verifying the fan is in the same location when it is placed in the shroud," reinforcing the importance of knowing the original location.

</research_note>

</learning_objective>

<learning_objective>

Learning Objective: Perform an initial inspection: Inspect the fan blade for damage (bends, damaged rivets, corrosion). Replace the blade if damaged.

<research_note>

1.  **Initial Inspection of the Condenser Fan Blade**
        1.  **Purpose of the initial inspection:** The initial inspection aims to identify any existing damage or potential problems with the condenser fan blade *before* beginning any maintenance or repair work. This proactive approach can prevent further damage to the HVAC system and ensure the technician is prepared with the correct replacement parts, if needed. As stated in the documents, "Identifying signs of a malfunctioning component in an HVAC system can help homeowners prevent costly repairs and maintain optimal performance..."
        2.  **When to perform the inspection:** An initial inspection should be performed:
            *   As part of any scheduled HVAC maintenance. "Regular maintenance is crucial for assessing the condition of the fan blade."
            *   When troubleshooting HVAC system issues such as unusual noises, slow spinning, or overheating.
            *   Before starting any work on the condenser fan or motor.
            *   Periodically as a general check-up, in addition to annual inspections.

    2.  **Types of Damage to Look For**
        1.  **Visual cues: bends, warping, cracks, fractures:** Carefully examine the fan blades for any visible signs of physical damage. This includes bends, warping, cracks, and fractures. "Inspect the fan blades for any signs of damage, such as bending or warping. If you notice any bent blades, consider replacing them as bent blades can disrupt airflow and reduce efficiency." A fractured or cracked blade is a severe issue that requires immediate attention, as "A fractured or cracked blade can split, flinging metal that could cause serious injury to anyone nearby or damage to your AC unit."
        2.  **Material degradation: corrosion, wear:** Look for signs of corrosion (rust) or general wear on the fan blades. "Wheels...Wear or corrosion..." Corrosion can weaken the blade material, making it more susceptible to failure.
        3.  **Fastener issues: loose or damaged rivets/bolts:** Check the rivets or bolts that attach the fan blades to the hub for any signs of looseness or damage. "Wheels...Loose rivets or bolts." Loose fasteners can cause the blades to vibrate or even detach during operation.
        4.  **Audible cues: Unusual noises:** Note any unusual noises coming from the condenser fan during operation, such as howling, rattling, or wobbling. "When a blade is damaged, it will produce a howling or metallic wobbling or rattling noise. A visual inspection can help to confirm that the blade is the issue." These sounds can indicate a damaged or unbalanced fan blade.

    3.  **Assessing Damage and Determining Replacement**
        1.  **Severity of damage:** Assess the severity of any identified damage. Minor bends or surface corrosion might not require immediate replacement, but should be monitored. Major cracks, fractures, or significant corrosion warrant immediate replacement.
        2.  **Impact on performance and safety:** Consider how the damage impacts the fan's performance and the overall safety of the HVAC system. Damaged blades can reduce airflow, leading to decreased cooling efficiency and potential overheating. Fractured blades pose a safety hazard.
        3.  **Decision to replace:** If the fan blade exhibits significant damage that affects its performance or poses a safety risk, it should be replaced. "Check all fastenings and impeller blades for cracks or damage - Replace as necessary." It is always best to err on the side of caution and replace a questionable blade to prevent further damage or injury.

</research_note>

</learning_objective>

</research_notes>

</input>

<output>

<slides>

Title: Getting Ready for Fan Removal
Slide Type: Transition
Content: In this section, we will learn how to properly prepare before removing a condenser fan. We’ll walk through verifying power is off, documenting the fan setup, and inspecting the fan blade for any signs of damage or wear.

Title: Setting the Voltmeter and Testing Battery Terminals
Slide Type: Video
Video_Id: sI_569t7HmE
Start: 2
End: 38
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

Title: Confirming Zero Voltage on System Circuits
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

Title: Why Documentation Matters
Slide Type: Content
Content: Before disconnecting any components, it’s critical to document the current setup. HVAC systems have complex wiring and fan blade positioning, both of which must be restored accurately. Documentation helps prevent wiring mistakes and airflow issues during reassembly.

Title: How to Document Wiring and Blade Position
Slide Type: Content
Content: Take clear, well-lit photos of all wiring connections from multiple angles, including wires connected to the contactor, capacitor, and control boards. Note wire colors and terminal positions. Also, record the fan blade’s height in the shroud by measuring or photographing its position before removal.

Title: Performing an Initial Inspection
Slide Type: Content
Content: Always start with a fan blade inspection before maintenance. Look for visual damage like bending, warping, or cracking. Also check for corrosion and loose rivets or bolts, as these can reduce efficiency or pose safety risks.

Title: Identifying Damage and Deciding on Replacement
Slide Type: Content
Content: If the fan blade is significantly damaged — such as cracked or corroded — it should be replaced immediately. Even minor issues should be closely monitored. Always consider how the damage might impact airflow, safety, and overall HVAC system performance.

Title: Key Takeaways
Slide Type: Summary
Content: In this section, we covered:
1. How to verify that power is safely off using a voltmeter.
2. Why documenting wiring and fan blade height is essential for accurate reassembly.
3. How to inspect the fan blade for damage and when to replace it for safety and efficiency.

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
  
    # Identify unique subtopics in order of appearance
    subtopic_first_indices = df.drop_duplicates("Subtopic", keep="first").index.tolist()
    subtopic_names = df.loc[subtopic_first_indices, "Subtopic"].tolist()

    # For each unique subtopic, gather all rows for that subtopic (in order)
    subtopic_to_rows = {subtopic: df[df["Subtopic"] == subtopic] for subtopic in subtopic_names}

    # For each unique subtopic, get topic and subtopic from first row
    subtopic_to_topic = {subtopic: rows.iloc[0]["Topic"] for subtopic, rows in subtopic_to_rows.items()}
    subtopic_to_subtopic = {subtopic: rows.iloc[0]["Subtopic"] for subtopic, rows in subtopic_to_rows.items()}

    # Helper to construct research_notes string for a subtopic
    def construct_research_notes(rows):
        blocks = []
        for _, row in rows.iterrows():
            lo = row["Learning Objectives"]
            rn = row["research_notes"]
            blocks.append(f"<learning_objective>\n\nLearning Objective: {lo}\n\n<research_note>\n\n{rn}\n\n</research_note>\n\n</learning_objective>")
        return "\n\n".join(blocks)

    # Function to run the agent and extract <slides> for a subtopic
    def process_subtopic(subtopic):
        topic = subtopic_to_topic[subtopic]
        subtopic_val = subtopic_to_subtopic[subtopic]
        research_notes = construct_research_notes(subtopic_to_rows[subtopic])
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
        future_to_idx = {executor.submit(process_subtopic, subtopic): idx for idx, subtopic in enumerate(subtopic_names)}
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
    from services.sheets_service import get_sheet_data_and_df, clear_worksheet, save_to_sheet
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    if "slide_chunks" in df.columns:
        df = df.drop(columns=["slide_chunks"])
        clear_worksheet(ws)
        save_to_sheet(ws, df)

