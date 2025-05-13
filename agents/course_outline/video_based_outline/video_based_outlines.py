from modules.chain import Chain
from tqdm import tqdm
from services.youtube_video_loader import get_transcript
from services.sheets_service import get_sheet_data_and_df, save_to_sheet
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.helper_functions import get_outline_with_los, validate_column_values
import json
import streamlit as st
from services.smart_progress_bar import SmartProgressBar
from langsmith import traceable


generate_outline_from_video_transcript_prompt = """You are tasked with creating an outline (or a section of an outline) for a course based on the content of a YouTube video. Your goal is to generate a structured outline that accurately reflects the concepts covered in the video while aligning with the course objectives.

You will be provided with the following information:

<course_info>
Course Name: {course_name}

Target Audience: {target_audience}

Tentative Outline:
{tentative_outline}
</course_info>

<video_info>
Video Title: {video_title}

Video Transcript: {video_transcript}

List of Concepts to Include:
{list_of_concepts_to_include}
</video_info>

Follow these steps to create the outline:

1. Carefully review the course name, target audience, and tentative outline to understand the context and objectives of the course.

2. Analyze the video information, including the title, transcript, and list of concepts that should be included. Pay special attention to concepts that are explained in detail, not just mentioned in passing.

3. Compare the video content with the tentative outline to identify relevant topics and subtopics that align with the course objectives.

4. Create an outline that includes only concepts that are "properly" covered within the video. "Properly" means that the concept is explained, not just mentioned as a passing statement.

5. Organize the outline in a hierarchical structure with main topics and subtopics. Ensure that the structure is logical and flows well.

6. Include only information that is actually present in the video content. Do not add concepts or explanations that are not covered in the video, even if they might seem relevant to the course.

7. If the video does not cover all the topics in the tentative outline, focus on creating a partial outline based on the available content.

Present your outline in the following format:

<outline>
1. Main Topic
   1.1 Subtopic
   1.2 Subtopic
      1.2.1 Sub-subtopic
2. Main Topic
   2.1 Subtopic
   2.2 Subtopic
</outline>

Remember:
- Only include concepts that are explained in detail in the video.
- Align the outline with the course objectives and target audience.
- Do not add information that is not present in the video content.
- If the video doesn't cover all aspects of the tentative outline, it's okay to have a partial outline.
- Be detailed with the outline. Use sentences if necessary to explain exactly what is to be covered such that there is no guesswork down the line on what material to include.

If you need to explain your thought process or justify your choices, use <contemplator> tags before presenting the final outline. It is okay for this section to be quite long.
"""


generate_outline_from_video_transcript_example = {
    'course_name': 'Basics of Electricity and Magnetism',
    'target_audience': 'Entry-level HVAC technicians',
    'tentative_outline': """Topic: Electrons in Electricity
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
  Subtopic: Magnetic components - transformer
""",
    'video_title': 'Analogies for Magnetism and Electricity w/ Ty Branaman',
    'video_transcript': """hey thanks for watching in this video my good friend ty branaman is back and he's giving some great analogies and just different concepts for understanding electricity and magnetism better this is a really great opportunity for you to share this video with somebody who maybe struggles with the basic principles of electricity and magnetism as many newer technicians and apprentices do i've put a link down in the description to ty's channel i would strongly suggest you subscribe to ty's channel he has a lot of really great training and education on there so here we go ty brandon talking about the basics of magnetism and electricity [Music] what happens if i send a magnet through a coil of wire why do we end up happening you create electricity you create electricity yes essentially we create electricity if i have a magnet and here's a magnet i'm sending it through a coil of wire so if i send a magnet through a coil of wire we generate electricity we can see that happening because we have the light and it's actually storing a little bit but that's how we generate electricity and i didn't bring my other device but we have two magnets and we send coils of wire turning through that magnetic field and we end up doing what creating electricity now what happens though when i send electricity through a coil of wire you create a magnet create a magnet yes so if we send electricity through the coil we create a magnet and if i send a magnet through a coil of wire i get electricity electricity generator does that make sense so this solenoid if i send 24 volts to this solenoid what happens magnet are you just that you're assuming now you're assuming that i'm right and i'm not always right let's plug it in and see right somebody want to press that button oh jeez no he's kidding go ahead yes what happens what happened it's a magnet now release it we have an electromagnet right this is electrician's simplest form that right there is key though it's so very important our electromagnet we know that's an electron magnet but let's think a little bit more about a motor what's really cool about a motor is it doesn't just have one electromagnet here we have how many different electromagnets six correct six we have six different electromagnets and what's cool is let's say this one is north and this one's south because it's altered and current going and these are in series we're going to be moving the electromagnetic field around as we go and eventually we're going to get here and they switch so this one is north it's now south this one's south that's down north and we're going to keep rotating that electromagnet around now if i put another magnet inside or if i induce a magnetic field inside what would be happening to my rotor would it be turning following it yes yes it would this is how we have motors i'm making electromagnets with electricity and then i have my rotor that i'm turning through here and i can make motors work inversely generations that same way let's say i have a wind turbine or water flowing through dams and i'm using my magnet and i'm turning this lever forcing a magnet through these windings what's happening creating electricity good electricity i'm generating electricity so i can use the magnet and force electricity out or i can force electricity in and make the motor turn electricity and magnetism go hand in hand they are very closely connected and we talked about magnetic fields earlier here you can see the magnetic field on a simple north and south with alternating current we're changing directions 60 times a second we still have this magnetic field in the side now we have that clamp meter when we have this clamp meter it does many things i check voltage or ohms using the leads but what do i use this part for so amperage amperage i'm checking amperage but does it touch a wire when i'm checking amperage what is it touching the magnetic field around the wire the magnetic field around the wire that's right so as amperage flows faster the magnetic field gets stronger has anybody been in a welding shop before fantastic you go to a welding shop people are grinding there's all this uh well essentially there's this stuff right here all over the floor what's cool is if you see the leads where they're welding there's lots of amperage going through there and it creates a very strong magnetic field and all of this these shavings in the floor will be perpendicular to the wire well this is working the same essential way as i test and there's electricity going through a wire it's creating that magnetic field this right here is reading that magnetic field choosing some cool math and it's giving you a number giving you the amperage isn't that amazing i think that's awesome i can check the speed of electron flow through this wire without actually in touching that wire at all so if i send electricity through now how can i increase the magnetic field you load the wire around yes loop the wire around and guess what we do on these motors i loop the wire around it magnifies the electromagnetic field it makes it stronger that's why they do in the the transformers it's exactly right transformers also the solenoid we have over there does the same thing here's another solenoid and what we've done is we've wrapped the wire around it around and around and around and around and we've increased that magnetic field the same thing here with another set of wire we've increased that magnetic field and we're going to talk about transformers today too so here's my electromagnets here another traumatic electromagnetic electromagnetic electromagnet so this motor these are called poles so here i have one two three four five six pole motor now what would turn faster a six pole motor or an eight pole motor everybody agree only one person's answering is if only one person answered and everybody agree with him no no no oh not i i like that answer but i don't have to agree with him i don't like it yeah yes oh now you're changing it no no i said yes it goes faster for sure everybody agree that's how you be where is that let's bring that office chair out here let's see what's going to happen somebody want to roll that office chat here so we're going to test this theory out we are all going to be involved in what it takes to make a motor work all right let's count the poles one two three four five six seven eight we got an eight pole motor so we can't bypass anybody i'm an electromagnetic field and i'm gonna move and you're in that electromagnetic field now you move with him you gotta have a full grip on it and then you fold it to hit pull to him good and then you good keep it try to keep it in the center you can't bypass burt okay so this is an eight pole motor the magnetic field the next person you gotta have full control you can't just pass them up see how it slid through let's go a little bit full control good full control eight pole motor so far so good you're the rotor we're the stator we're stationary he's rotating so far so good sure all right now let's take away two people so let's have okay you you're out then you're extra across from them everybody will spread back out now how many polls we have now one two three four six six good math don't trust me with numbers let's try it again all right is it turning faster oh oh the the rotor says it's turning faster but i don't believe the rotor let's try this a little bit more back let's take two people away all right let's all separate out now let's try again now we got a four pole motor right you ready so i turn it to you all right all right is it turning faster yeah so the fewer number of poles the faster that motor's gonna turn but i don't know let's take away two more people just shoot me we're a two pole motor now let's see how this two pole motor is going to work you ready is it turning faster sir we don't want a puker notice right so on a motor which is going to make that motor turn faster more poles or fewer pulls fewer pulls the fewer number of poles stays the same the fewer number pulls the faster that it turns the more number of poles the slower that it turns as each and every one does that make sense so far so good so if you actually look inside of motors you're going to have a start winding and a run winding you're going to count one so if you count the run winding you can actually count the number of poles you can do a little bit of scary math stuff and you actually find out how fast that motor turns which would turn faster my six pole or this two pole the two poles now there's some variances we have some other things to determine such as frequency also slip and voltage is going to be a factor net as well but this is going to understand that magnetic field that magnetic field is moving around so it doesn't change the speed of the magnetic field to have more poles in it but it still has to go from top to bottom every time like just as fast it has to go to top to bottom of now this pole now this pole now this pole all the way around right because i have fewer poles i can go all the way around faster yeah then you lose power or you have a bigger variation oh power though that's a great question power's going to be dependent on something else does anybody know the formula for power or just give me another word for power let's start there what's another word for power what yes my buddy watts right watts is another word for power so for power how do we get the formula for watts times uh okay i remember the whole thing as voltage time amperage yes volts times amps equals watts what volts times equals what's now this is a true form of power so confidence volts times equals watts so this is really going to go have a factor in our resistance how much resistance we have very low resistance we're going to have faster amp flow which means we're going to be using more watts with more power so the number of poles doesn't necessarily determine how strong that motor is going to be it's also going to have to do with the resistance of that so let's bring this to the next thing we're going to be talking about less resistance faster amperage more wattage big thanks to tai for doing this and for everything that he brings to our trade it's always great to collaborate with other people who care about our trade and who want to see it do better and tai is definitely one of those people again look down in the description we have a link to his channel and i would ask that you subscribe to his channel let you follow everything that he has going on thanks for watching we'll catch you on the next video thanks for watching our video if you enjoyed it and got something out of it if you wouldn't mind hitting the thumbs up button to like the video subscribe to the channel and click the notifications bell to be notified when new videos come out hvac school is far more than a youtube channel you can find out more by going to hvacrschool.com which is our website and hub for all of our content including tech tips videos podcasts and so much more you can also subscribe to the podcast on any podcast app of your choosing you can also join our facebook group if you want to weigh in on the conversation yourself thanks again for watching [Music] you""",
    'list_of_concepts_to_include': """[00:42] Generating Electricity with Magnets
[01:18] Creating Magnets with Electricity
[04:59] Increasing Magnetic Field Strength
[09:20] Power: Voltage Times Amperage""",
    'output': """<contemplator>

**Step-by-Step Reasoning**

1. **Review Course Context**
   - The course is called “Basics of Electricity and Magnetism,” aimed at entry-level HVAC technicians.
   - The tentative outline includes topics like fundamental electrical concepts, voltage, resistance, power, magnetism, coils, transformers, etc.

2. **Analyze the Video Content**
   - The video focuses on demonstrating how magnets can create electricity, how electricity can create magnets, the relationship between electricity and magnetism in motors (poles, speed, etc.), and how to strengthen the magnetic field by looping wires.
   - It also covers power as “volts times amps equals watts” and briefly mentions resistance in that context.
   - Other concepts from the tentative outline (such as conductors, insulators, dielectrics, in-depth coverage of voltage or resistance, etc.) are not explained in detail in the video, so they should not be included.

3. **Match Video Content to the Tentative Outline**
   - “Electrons in Electricity” (Fundamental concepts, charge, conductors, insulators, dielectrics) is *not* covered in detail in the video.
   - “Electrical Basic Terms” (Voltage, Amps, Resistance, Power) is *partially* covered. The video explains how to measure amperage (clamp meter and magnetic fields) and defines power as \( P = V \times I \). However, it does *not* give a full treatment of voltage or resistance in the classical sense.
   - “Magnetism” (Magnetic principles in electricity, coils, transformers) *is* explained extensively, particularly around motors, solenoids, and the concept of creating/generating magnetic fields.

4. **Outline Construction**
   - Include only topics the video explains in detail:
     - Generating electricity using magnets (electromagnetic induction).
     - Creating magnets using electricity (electromagnets, solenoids).
     - Motor principles using magnetic poles (more poles = slower speed, fewer poles = faster speed).
     - Strengthening the magnetic field (increasing wire loops, examples with clamp meter).
     - Power as \( \text{{Watts}} = \text{{Volts}} \times \text{{Amps}} \) and its dependency on current/resistance.

5. **Final Check**
   - Present the final outline in a hierarchical structure, focusing on the above concepts.
   - Do not add any details not present in the video transcript.
   - The outline reflects what was “properly” explained in the video and aligns with the course goals for HVAC technicians (understanding practical, real-world electricity and magnetism fundamentals).

</contemplator>

<outline>
1. Generating Electricity with Magnets
   1.1 **Electromagnetic Induction**
       - Demonstration of sending a magnet through a coil of wire to generate electricity.
       - Explanation of how movement within a magnetic field induces current in the coil.
   1.2 **Real-World Examples**
       - Wind turbines and water turbines forcing magnets through coils to create electrical power.

2. Creating Magnets with Electricity
   2.1 **Electromagnets and Solenoids**
       - Demonstration of applying voltage to a coil to produce a magnetic field.
       - Example with a 24 V solenoid showing how electricity flowing through coils creates a strong, temporary magnet.
   2.2 **Everyday Applications**
       - Motors and solenoids in HVAC systems rely on electromagnetic principles.

3. Relationship Between Electricity and Magnetism in Motors
   3.1 **Poles and Motor Speed**
       - Demonstration showing how multiple electromagnetic poles in a stator cause a rotor to spin.
       - Explanation: fewer poles = higher speed, more poles = lower speed.
   3.2 **Visual Demonstration with Participants**
       - Office chair exercise illustrating how each “pole” attracts the rotor in sequence, causing rotation.

4. Increasing Magnetic Field Strength
   4.1 **Looping the Wire**
       - Explanation that increasing the number of wire turns amplifies the magnetic field.
   4.2 **Measuring Current and Magnetic Fields**
       - Use of a clamp meter to detect the magnetic field around a conductor carrying current.
       - Welding shop example: how shavings arrange themselves in response to a strong magnetic field.

5. Understanding Power (Watts)
   5.1 **Power Formula**
       - Definition of real power as \( \text{{Watts}} = \text{{Volts}} \times \text{{Amps}} \).
   5.2 **Influence of Resistance**
       - Brief mention that lower resistance allows higher current, thus higher watts (more power).
       - Understanding how voltage, current, and resistance together affect motor performance.
</outline>"""
}

@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Generate Video Based Outlines",
    "function_name": "generate_outline_from_video_transcript",
    "user_id": st.session_state.get("role", "anonymous")
})
def generate_outline_from_video_transcript(course_name, target_audience, course_outline, video_title, transcript, list_of_concepts_to_include, example = generate_outline_from_video_transcript_example, llm = 'gemini_2_flash'):
    """
    This function generates an outline for a course based on a video transcript.
    Args:
        video_title (str): The title of the video.
        transcript (str): The transcript of the video.
        list_of_concepts_to_include (str): The list of concepts to include in the outline.
        example (dict): The example to use for the agent.
    Returns:
        str: The generated outline.
    """
    generate_outline_from_video_transcript_agent = Chain( llm = llm, tags = ['contemplator', 'outline'])

    # Setting a one shot example
    generate_outline_from_video_transcript_agent.add_message(
        role = "user",
        content = generate_outline_from_video_transcript_prompt.format(
            course_name = example['course_name'],
            target_audience = example['target_audience'],
            tentative_outline = example['tentative_outline'],
            video_title = example['video_title'],
            video_transcript = example['video_transcript'],
            list_of_concepts_to_include = example['list_of_concepts_to_include']
        )
    )
    generate_outline_from_video_transcript_agent.add_message(
        role = "ai",
        content = example['output']
    )

    # Actual user message
    generate_outline_from_video_transcript_agent.add_message(
        role = "user",
        content = generate_outline_from_video_transcript_prompt.format(
            course_name = course_name,
            target_audience = target_audience,
            tentative_outline = course_outline,
            video_title = video_title,
            video_transcript = transcript,
            list_of_concepts_to_include = list_of_concepts_to_include
        )
    )

    response = generate_outline_from_video_transcript_agent.run()

    return response


def run_generate_video_based_outline(sheet, worksheet_name, course_name, target_audience, llm = 'gemini_2_flash'):
    """
    This function generated video based outlines for all the rows marked manually
    :param sheet: The sheet object.
    :param worksheet_name: The worksheet name.
    :param course_name: The course name.
    :param target_audience: The target audience.
    :param llm: The language model to use.
    :return: None
    """

    # Read the sheet and df
    videos_research_sheet, videos_research_df = get_sheet_data_and_df(sheet, worksheet_name)
    rough_outline_sheet, rough_outline_df = get_sheet_data_and_df(sheet = sheet, sheet_name = "Rough Outline")

    if 'outline_contemplator' not in videos_research_df.columns:
        videos_research_df['outline_contemplator'] = ''
        videos_research_df['outline'] = ''
        videos_research_df['consolidation_comments'] = ''

    # Check if column b is populated (not empty string and not NaN) for all rows where column a is "Yes"
    mask = videos_research_df['Manual Review'].str.contains("yes", case=False, na=False)
    all_rows_populated = ((videos_research_df.loc[mask, 'outline'].notna()) & (videos_research_df.loc[mask, 'outline'] != "")).all()
    if all_rows_populated == True:
        print("Video based outlines already generated for all marked rows. Skipping this step.")
        return

    # Get the course outline
    course_outline = get_outline_with_los(
        df = rough_outline_df,
        include_learning_objectives = False
    )

    # Get video_transcript column count
    video_transcript_col_count = len([col for col in videos_research_df.columns if 'video_transcript_' in col])

    # Prepare for parallel processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers = 5) as executor:
        # Submit tasks for each row
        for index, row in videos_research_df.iterrows():
            
            # Skip if video not marked as Yes in Manual Review
            if 'yes' not in row['Manual Review'].lower():
                continue
            
            # Skip is already populated
            if row['outline'] != '':
                continue

            # Get the transcript
            timestamped_transcript = json.loads(
                ''.join(
                    [row[f'video_transcript_{i}'] for i in range(video_transcript_col_count)]
                )
            )
            video_transcript = ' '.join([item['text'] for item in timestamped_transcript])

            # Submit the task
            future = executor.submit(
                generate_outline_from_video_transcript,
                course_name,
                target_audience,
                course_outline,
                row['title'],
                video_transcript,
                row['proposed_chapters_to_include'],
                generate_outline_from_video_transcript_example,
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
            index = futures_map[future]  # retrieve the index
            response = future.result()

            # Update the df
            videos_research_df.loc[index, 'outline_contemplator'] = response['contemplator']
            videos_research_df.loc[index, 'outline'] = response['outline']

            # Update progress
            progress.update()

            # Check if we should save
            if progress.should_save():
                print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
                save_to_sheet(worksheet = videos_research_sheet, df = videos_research_df)

    # Final save to sheet after all tasks
    print('All rows processed. Saving final DataFrame to sheet.')
    save_to_sheet(worksheet = videos_research_sheet, df = videos_research_df)

    return


generate_video_outline_consolidation_comments_prompt = """You are tasked to add "consolidation comments" for a list of partial outlines generated from video transcripts. You are overtaking this task from a human thus, you need to ensure to achieve similar / better performance.

The course outline was generated based on the following course info:
<course_info>
Course name: {course_name}
Target audience: {target_audience}
Initial (tentative) outline: {tentative_outline}
</course_info>

The following thought process was carried to generate the partial outline:
<partial_outline>
Video title: {video_title}
Agent thoughts: 
{video_scratchpad}

Video relevance: {video_relevance}

Proposed chapters to include:
{proposed_chapters_to_include}

Partial outline: 
{partial_outline}
</partial_outline>

Task: Analyze the partial outline based on the all the information shared above. Your task is to provide consolidation comments for this outline. 

Think of consolidation comments ranging from a simple "looks good" to a couple of sentences (eg. remove this that, avoid brand names, etc.) based on the current case.

These consolidation comments will be given for a list of about 15-20 potentially overlapping outlines. Currently, you are providing comments for one such partial outline.

Make sure to output in the following format:
<inputs_analysis>
[Your understanding of the inputs and the course requirements.]
</inputs_analysis>
<outline_analysis>
[Your analysis of the partial outline based on the inputs. What looks good. What doesn't look good, etc.]
</outline_analysis>
<consolidation_comments>
[Your consolidation comments for this outline]
</consolidation_comments>

Remember: Since your are reviewing a partial outline, focus on the information available at hand and don't worry about missing information. Thus, your comments should mostly be of the type "looks good" or "make deletetions" or "comments on structure", etc. Never suggest "add these".
"""


@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Manual Review - Video Outline Consolidation Comments",
    "function_name": "generate_video_outline_consolidation_comments",
    "user_id": st.session_state.get("role", "anonymous")
})
def generate_video_outline_consolidation_comments(course_name, target_audience, course_outline, video_title, video_scratchpad, video_relevance, proposed_chapters_to_include, partial_outline, llm = "gemini_2_flash"):
    """
    Generates consolidation comments for video based outlines
    """
    generate_consolidation_comments_agent = Chain(llm = llm, tags = ['consolidation_comments'])

    generate_consolidation_comments_agent.add_message(
        role = "user",
        content = generate_video_outline_consolidation_comments_prompt.format(
            course_name = course_name,
            target_audience = target_audience,
            tentative_outline = course_outline,
            video_title = video_title,
            video_scratchpad = video_scratchpad,
            video_relevance = video_relevance,
            proposed_chapters_to_include = proposed_chapters_to_include,
            partial_outline = partial_outline,
        )
    )

    response = generate_consolidation_comments_agent.run()

    return response['consolidation_comments']


def run_generate_video_outline_consolidation_comments(sheet, worksheet_name, course_name, target_audience, llm = "gemini_2_flash"):
    """
    Run the generate consolidation comments for all video based outlines.
    :param sheet: The sheet object.
    :param worksheet_name: The worksheet name.
    :param course_name: The course name.
    :param target_audience: The target audience.
    :param llm: The language model to use.
    :return: None
    """

    # Get the sheet and DataFrame
    videos_research_sheet, videos_research_df = get_sheet_data_and_df(sheet, worksheet_name)
    rough_outline_sheet, rough_outline_df = get_sheet_data_and_df(sheet = sheet, sheet_name = "Rough Outline")

    # Get the course outline
    course_outline = get_outline_with_los(
        df = rough_outline_df,
        include_learning_objectives = False
    )

    # Prepare for parallel processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers = 5) as executor:
        # Submit tasks for each row
        for index, row in videos_research_df.iterrows():
            
            # Skip if already populated
            if row['consolidation_comments'] != '':
                continue

            # Skip for blank outlines
            if row['outline'] == '':
                continue

            # Submit the task
            future = executor.submit(
                generate_video_outline_consolidation_comments,
                course_name,
                target_audience,
                course_outline,
                row['title'],
                row['video_scratchpad'],
                row['video_relevance'],
                row['proposed_chapters_to_include'],
                row['outline'],
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
            index = futures_map[future]  # retrieve the index
            consolidation_comments = future.result()

            # Update the df
            videos_research_df.loc[index, 'consolidation_comments'] = consolidation_comments

            # Update progress
            progress.update()

            # Check if we should save
            if progress.should_save():
                print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
                save_to_sheet(worksheet = videos_research_sheet, df = videos_research_df)

    # Final save to sheet after all tasks
    print('All rows processed. Saving final DataFrame to sheet.')
    save_to_sheet(worksheet = videos_research_sheet, df = videos_research_df)

    return


def manual_input_video_outline_consolidation_comments(sheet, worksheet_name, course_name, target_audience, skip_manual_step = False, llm = "gemini_2_flash"):
    """
    Checks whether the user has properly added comments for in the outline_consolidation column for video based outlines.

    :param sheet: The Google Sheets object.
    :param worksheet_name: The name of the worksheet.
    :param skip_manual_step: Bool. If True, the column can be left blank without any validation errors
    :return: True if column is properly populated, raises an error otherwise.     
    """

    if skip_manual_step:
        run_generate_video_outline_consolidation_comments(
            sheet = sheet,
            worksheet_name = worksheet_name,
            course_name = course_name,
            target_audience = target_audience,
            llm = llm
        )
        return True

    # Get the sheet and DataFrame
    videos_research_sheet, videos_research_df = get_sheet_data_and_df(sheet, worksheet_name)

    # Check if user has properly added inputs - for consolidation_comments columns
    validate_column_values(
        df = videos_research_df,
        filter_column = "outline",
        validation_column = "consolidation_comments",
        require_populated = True
    )

    return True

