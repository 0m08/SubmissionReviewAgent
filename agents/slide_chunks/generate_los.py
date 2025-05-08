import pandas as pd
from modules.chain import Chain
from tqdm import tqdm
from agents.slide_chunks.format_inputs import strip_section_prefix
from services.sheets_service import get_sheet_data_and_df
from gspread_dataframe import set_with_dataframe
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.smart_progress_bar import SmartProgressBar


generate_learning_objectives_slide_prompt = """We are creating structured slide content for an e-learning course. You are an expert instructional designer and e-learning content creator, crafting educational materials with the fluency and adaptability of a world-class writer — producing content indistinguishable from human authorship. In this role, you are serving as a Learning Objectives Generator Agent, responsible for creating a structured Learning Objectives slide. Your task is to analyze all slide content for a given topic and generate a concise, well-structured list of learning objectives that accurately represent the key takeaways from the topic.

Below is the course information for which the learning objectives are being generated:

<course_information>
Course name: {course_name}
Target audience: {target_audience}
</course_information>

Below is the topic for which you will generate the learning objectives:

<topic>
Topic Name: {topic}
</topic>

Below are the Slide Chunks containing structured content for this topic:

<slide_chunks>
{slide_chunks}
</slide_chunks>

Each topic consists of multiple subtopics, and each subtopic contains several slide chunks. Each slide chunk has a Slide Type, Slide Chunk Title, and Slide Chunk Content. Your task is to critically assess the relationship between these subtopics and their respective slide chunks, ensuring that the generated learning objectives accurately reflect the key takeaways from the topic. The learning objectives must be well-structured, instructionally cohesive, and logically connected to the overall topic.

Follow these guidelines carefully while generating the content for the Learning Objectives slide:

1. Extract Key Takeaways from Slide Chunks:
   - Analyze the slide chunks for the given topic and extract the most important learning points.
   - Ensure that the learning objectives accurately reflect the core knowledge conveyed in the slides.

2. Ensure Learning Objectives Are Clear and Concise:
   - Each learning objective must be short, specific, and actionable.
   - Use simple and direct language, avoiding unnecessary complexity.
   - Ensure that each objective provides a meaningful takeaway for the learner.

3. Cover All Key Concepts Across Subtopics:
   - The generated learning objectives should summarize the entire topic, not just individual subtopics.
   - Identify common themes across subtopics and ensure they are represented.
   - Ensure that no critical concept is left out.

4. Use Action - Oriented Language:
   - Start each learning objective with a strong action verb that describes what the learner will actively do (e.g., "Use tools to troubleshoot" instead of "Explain tools used to troubleshoot"). Ensure the objective focuses on the learner's action, not what the instructor explains.
   - Example verbs: Define, Identify, Describe, Demonstrate, List, Recognize, Apply, Understand, Differentiate, etc.
   - Avoid vague terms like "Know about" or "Be familiar with" — focus on measurable outcomes.

5. Avoid Repetitive or Formulaic Sentence Patterns:
   - Do not structure all learning objectives in the exact same way (e.g., "Define X," "Define Y," "Define Z").
   - Vary the phrasing naturally to mimic how a human would avoid monotony — while still starting with action verbs.
   - Use natural transitions, slight rewording, or variation in verbs to keep the list engaging and fluid.

6. Ensure Objectives Feel Targeted and Personal, Not Generic:
   - Avoid generic phrasing like "Understand important concepts" or "Learn the key points."
   - The objective should feel specific, purposeful, and written with intention — like it was created for a real learner.
   - Make sure each point delivers value and feels meant for someone to read and use, not just to check a box.

7. Prioritize Clarity Over Overly Formal or Stiff Language:
   - Use plain, precise words that are accessible to the target audience.
   - Avoid needlessly formal or academic phrasing that can feel distant or robotic.
   - The tone should feel instructive and professional — not like a corporate template or machine-generated list.

8. Follow the Standard Learning Objectives Format:
   - Start the learning objectives with a natural, human-like introductory phrase that clearly tells the learner what they will gain from the topic. Use phrasing that feels conversational yet professional, such as:
     "After completing this topic, you’ll be able to..."
     "Once you’ve finished this section, you’ll know how to..."
     "This topic will help you learn how to..."
     Choose a lead-in that suits the tone of the content and supports a more engaging learning experience.
   - This phrase should be followed by a numbered list of learning objectives.

9. Ensure an Appropriate Number of Learning Objectives:
   - The topic must have between 3 and 5 learning objectives, with fewer being preferred whenever possible.
   - Avoid listing too few objectives (which may not fully cover the topic) or too many (which may overwhelm the learners).

Present your output strictly in the following format:

<output>

<evaluation_breakdown>

(Before giving your output, document your thought process for generating the learning objectives in the following format)

Key Concepts: [List key concepts covered in the topic, extracted from slide chunks.]
Subtopic Relationships: [Describe how the subtopics are connected and contribute to the overall topic.]
Learning Objectives Planning: [Analyze the extracted key concepts and subtopic relationships to determine how they should be translated into well-structured learning objectives. Outline the logical flow and instructional approach for presenting the objectives.]

</evaluation_breakdown>

Based on your above evaluation, give the learning objectives in the following format

<learning_objectives>
By the end of this topic, you will be able to:
(Provide a numbered list of the learning objectives that you generated)
</learning_objectives>

</output>
"""


generate_learning_objectives_slide_prompt_examples = """Examples:
Use these below examples as a reference for the structure, style and detail of your learning objectives slide content.

<examples>

<example>

<course_information>
Course name: Comfort
Target audience: Entry-level HVAC technicians
</course_information>

<topic>
Topic name: Humidity
</topic>

<slide_chunks>

<subtopic_1>

<subtopic_name>
Subtopic Name: Basics of Air and Humidity
</subtopic_name>

<slide_chunk>

Slide Chunk 1:
Slide Type: Transition Slide
Slide Chunk Title: Introduction to Humidity
Slide Chunk Content: Humidity is all around us, even if we can't see it. It's a key factor in determining how comfortable we feel, whether we're indoors or outdoors.

</slide_chunk>

<slide_chunk>

Slide Chunk 2:
Slide Type: Content Slide
Slide Chunk Title: Air Has Weight and Takes Up Space
Slide Chunk Content: Before we dive into humidity, let's talk about air itself. Although it might seem like nothing, air actually has weight and takes up space. At sea level, the atmosphere exerts a pressure of about 14.7 pounds per square inch on us from all directions. This might sound surprising, but it's what makes up our atmosphere.

</slide_chunk>

<slide_chunk>

Slide Chunk 3:
Slide Type: Content Slide
Slide Chunk Title: Constituents of Air
Slide Chunk Content: Air is composed of several elements:
- Nitrogen
- Oxygen
- Carbon dioxide
- Argon
- Various trace gases
- And importantly for our discussion, water vapor

The amount of water vapor in the air can vary significantly, and this variation is what we refer to as humidity.

</slide_chunk>

<slide_chunk>

Slide Chunk 4:
Slide Type: Summary Slide
Slide Chunk Title: Summary - Air and Its Components
Slide Chunk Content: In this section, we've learned that:
- Air has weight and takes up space
- Air exerts pressure on us (14.7 psi at sea level)
- Air is composed of various gases, and water vapor
- The amount of water vapor in air varies and is what we call humidity

</slide_chunk>

</subtopic_1>

<subtopic_2>

<subtopic_name>
Subtopic Name: Understanding Humidity
</subtopic_name>

<slide_chunk>

Slide Chunk 1:
Slide Type: Transition Slide
Slide Chunk Title: Understanding Humidity
Slide Chunk Content: Now that we understand the basics of air, let's dive deeper into the concept of humidity and how we measure it.

</slide_chunk>

<slide_chunk>

Slide Chunk 2:
Slide Type: Content Slide
Slide Chunk Title: Relative Humidity
Slide Chunk Content: When we talk about humidity in everyday life, we're usually referring to relative humidity. But what does this mean?
Relative Humidity is the amount of water vapor in the air compared to the maximum amount the air could hold at that temperature, expressed as a percentage.
For example, if at a particular moment, if the air is holding half the moisture than it's capacity to hold at that temperature, the relative humidity, at that moment, is 50%.
When the relative humidity reaches 100%, the air is saturated — it's holding as much water vapor as it possibly can at that temperature.
Understanding relative humidity helps us gauge how close the air is to saturation, which affects our comfort and the likelihood of condensation."

</slide_chunk>

<slide_chunk>

Slide Chunk 3:
Slide Type: Content Slide
Slide Chunk Title: Dew Point
Slide Chunk Content: At the point where air reaches 100% relative humidity, any additional cooling or moisture will cause water vapor to condense into liquid water. This temperature is known as the dew point.
The dew point is the temperature at which air becomes saturated with water vapor and therefore that water vapor begins to condense.

</slide_chunk>

<slide_chunk>

Slide Chunk 4:
Slide Type: Content Slide
Slide Chunk Title: Dew Point - Example
Slide Chunk Content: You've probably experienced the dew point without realizing it. When you see:
Clouds forming in the sky
Dew on the grass in the morning
Water droplets forming on a cold glass of water
Fog on your windows

These are all examples of air reaching its dew point. The water vapor in the air has cooled to the point where it can no longer remain as vapor and condenses into liquid water.
When we say that relative humidity is 100%, we mean that the air is saturated. At this point of saturation, any additional cooling or addition of moisture will cause condensation, forming dew, fog, or precipitation.

</slide_chunk>

<slide_chunk>

Slide Chunk 5:
Slide Type: Content Slide
Slide Chunk Title: Absolute Humidity vs Relative Humidity
Slide Chunk Content: When discussing humidity, it's crucial to understand the difference between absolute humidity and relative humidity:
Absolute Humidity:
Measures the exact amount of water vapor present in a given volume of air.
Typically expressed in grains per pound of dry air.
Doesn't change just because the air gets warmer or cooler.
Relative Humidity:
The amount of water vapor present in the air compared to the maximum it could hold at that temperature.
Expressed as a percentage.
Changes with temperature, even if the absolute amount of moisture remains the same.
Both measures are important:
Absolute humidity tells us the actual amount of moisture in the air.
Relative humidity helps us understand how close the air is to saturation, which affects comfort and condensation.
When discussing humidity, it's crucial to understand the difference between absolute humidity and relative humidity.
Absolute humidity refers to the exact amount of water vapor in a specific volume of air.
It's typically measured in grains per pound of dry air.
What’s key here is that absolute humidity remains constant, regardless of whether the air gets warmer or cooler.
On the other hand, relative humidity is a measure of how much water vapor is in the air compared to the maximum amount it could hold at that specific temperature.
It's expressed as a percentage.
Unlike absolute humidity, relative humidity changes with temperature, even if the actual moisture level in the air stays the same.
Both of these measures play a crucial role.
Absolute humidity gives us the actual moisture content in the air, while relative humidity tells us how close the air is to saturation, which directly impacts comfort and the potential for condensation.

</slide_chunk>

<slide_chunk>

Slide Chunk 6:
Slide Type: Content Slide
Slide Chunk Title: Relative Humidity Depends on Temperature
Slide Chunk Content: Here's a key point to remember: the amount of water vapor air can hold depends greatly on its temperature. Warmer air can hold more moisture than cooler air.
Think about a cup of coffee or tea:
When it's cold, sugar doesn't dissolve easily.
Heat it up, and the liquid can hold much more sugar.
Air works similarly with water vapor.
Cooler air holds less moisture and reaches 100% relative humidity with less water vapor.
Warmer air holds more moisture and requires more water vapor to reach 100% relative humidity.
So, relative humidity depends on temperature.

</slide_chunk>

<slide_chunk>

Slide Chunk 7:
Slide Type: Content Slide
Slide Chunk Title: Relative Humidity Depends on Temperature - Example
Slide Chunk Content: Let's use a practical example to understand this better. Imagine it's a cool morning, and the air inside your house is at 60°F (15.5°C) with 60% relative humidity. As the day warms up, the indoor temperature rises to 80°F (26.7°C), but no moisture is added or removed from the air. What happens to the relative humidity? It actually decreases! Even though the absolute amount of moisture hasn't changed, the warmer air can hold more moisture, so the relative humidity has decreased.

</slide_chunk>

<slide_chunk>

Slide Chunk 8:
Slide Type: Summary Slide
Slide Chunk Title: Summary: Understanding Humidity Measures
Slide Chunk Content: Alright, let's quickly go over what we've learned about humidity:
- We talked about relative humidity - that's how much moisture is in the air compared to how much it could hold.
- We learned about dew point - that's when the air is holding all the moisture it can.
- We looked at the difference between relative humidity and the total amount of moisture in the air.
- And we saw how temperature really changes things - hot air can hold way more moisture than cold air.

</slide_chunk>

</subtopic_2>

<subtopic_3>

<subtopic_name>
Subtopic Name: Humidity's Impact on Comfort and IAQ
</subtopic_name>

<slide_chunk>

Slide Chunk 1:
Slide Type: Transition Slide
Slide Chunk Title: Humidity's Impact
Slide Chunk Content: Now that we understand how to measure and describe humidity, let's explore how it affects our comfort and indoor environments.

</slide_chunk>

<slide_chunk>

Slide Chunk 2:
Slide Type: Content Slide
Slide Chunk Title: Humidity and Comfort
Slide Chunk Content: Now, let's talk about why humidity matters for comfort. Our bodies rely on evaporation to regulate temperature. When we're hot, we sweat, and as that sweat evaporates, it cools us down.
High relative humidity makes this process less effective. The air is already holding a lot of moisture, so our sweat doesn't evaporate as readily. This is why a hot, humid day can feel so uncomfortable because our body's natural cooling system isn't working efficiently.
For optimal comfort, we generally aim for relative humidity between 30% and 60%. In humid climates, around 50% is ideal, while in drier climates, 35 to 40% can be more comfortable.

</slide_chunk>

<slide_chunk>

Slide Chunk 3:
Slide Type: Content Slide
Slide Chunk Title: Humidity and Indoor Air Quality
Slide Chunk Content: Humidity doesn't just affect how we feel - it also impacts indoor air quality. When relative humidity is between 40-50%, we avoid many of the problems associated with air that's too dry or too moist.
Very dry air can irritate our mucous membranes, while very humid air can promote the growth of bacteria, mold, and fungi. High humidity can also lead to condensation on walls and around vents, potentially causing damage and creating unhealthy conditions.

</slide_chunk>

<slide_chunk>

Slide Chunk 4:
Slide Type: Content Slide
Slide Chunk Title: Controlling Humidity
Slide Chunk Content: In the HVAC industry, controlling humidity is a major concern. Air conditioners play a big role in this. As air passes over the cold evaporator coil in an AC unit, it cools down. If it cools enough to reach the dew point, moisture in the air condenses on the coil and is drained away, effectively dehumidifying the air.
However, it's a bit of a balancing act. As we cool the air, we're also reducing its capacity to hold moisture. So the air leaving an AC unit often has high relative humidity.

</slide_chunk>

<slide_chunk>

Slide Chunk 5:
Slide Type: Summary Slide
Slide Chunk Title: Key Takeaways
Slide Chunk Content: Remember these key concepts about humidity:
1. Humidity is the amount of water vapor in the air
2. We measure humidity in two ways: absolute humidity (total moisture content) and relative humidity (percentage of maximum possible moisture at a given temperature)
3. Warmer air can hold more moisture than cooler air
4. Relative humidity greatly affects our comfort and indoor air quality
5. Optimal relative humidity for comfort is generally between 30% and 60%
6. HVAC systems, especially air conditioners, play a crucial role in controlling indoor humidity
7. Understanding and managing humidity is essential for creating comfortable and healthy indoor environments

</slide_chunk>

</subtopic_3>

</slide_chunks>

<output>

<learning_objectives>

After completing this topic, you’ll be able to:
1. Understand what air is made of and how its weight affects HVAC systems.
2. Tell the difference between absolute and relative humidity.
3. Explain how temperature changes impact the moisture-holding ability of air.
4. See how HVAC systems help keep indoor humidity levels balanced and comfortable.

</learning_objectives>

</output>

</example>

<example>

<course_information>
Course name: Taking Temperature & Humidity Measurement
Target audience: Entry-level HVAC technicians
</course_information>

<topic>
Topic name: Basic Measurement Terminology
</topic>

<slide_chunks>

<subtopic_1>

<subtopic_name>
Subtopic Name: Qualitative & Quantitative Measurements
</subtopic_name>

<slide_chunk>

Slide Chunk 1:
Slide Type: Transition Slide
Slide Chunk Title: Introduction to Data Types
Slide Chunk Content: Let’s take a closer look at the two types of data: qualitative and quantitative.

</slide_chunk>

<slide_chunk>

Slide Chunk 2:
Slide Type: Content Slide
Slide Chunk Title: Identifying Useful Data
Slide Chunk Content: Let’s start with a quick question. Which of the following statements provides useful data?
My pipe clamp read 95 degrees Fahrenheit on the liquid line.
It felt muggy when I walked inside that house.
Which one did you choose?

</slide_chunk>

<slide_chunk>

Slide Chunk 3:
Slide Type: Content Slide
Slide Chunk Title: Both Statements Provide Useful Data
Slide Chunk Content: If you chose the first statement, you’re correct! But if you selected the second one, you’re also right. Both statements offer valuable information about the system, albeit in different forms.

</slide_chunk>

<slide_chunk>

Slide Chunk 4:
Slide Type: Content Slide
Slide Chunk Title: Quantitative Data Explained
Slide Chunk Content: The first statement, mentioning 95 degrees Fahrenheit, represents quantitative data. This type of data uses numbers to quantify measurements like temperature, humidity, and pressure, typically obtained through precise tools.

</slide_chunk>

<slide_chunk>

Slide Chunk 5:
Slide Type: Content Slide
Slide Chunk Title: Quantitative Data Explained
Slide Chunk Content: On the other hand, the second statement about feeling muggy is qualitative data. Qualitative data involves observations and descriptions without attaching numerical values.

</slide_chunk>

<slide_chunk>

Slide Chunk 6:
Slide Type: Content Slide
Slide Chunk Title: Tools Provide Quantitative Data
Slide Chunk Content: The tools you buy are going to give you quantitative data by measuring parameters like temperature, humidity, and pressure. These precise measurements are essential for diagnosing and maintaining HVAC systems effectively.

</slide_chunk>

<slide_chunk>

Slide Chunk 7:
Slide Type: Content Slide
Slide Chunk Title: Gathering Qualitative Data
Slide Chunk Content: You can gather qualitative data by using your hand to feel for airflow and thinking about how the temperature feels on your skin is a great example. A lot of customers’ comfort complaints fall into this category. Customers who feel muggy or say that their upstairs rooms are warmer than downstairs are providing qualitative data.

</slide_chunk>

<slide_chunk>

Slide Chunk 8:
Slide Type: Content Slide
Slide Chunk Title: Role of Qualitative Data in Diagnostics
Slide Chunk Content: Qualitative data is especially useful when trying to get a general idea of system performance. You may try to get a feel for airflow coming out of the vents or condenser or look for oil spots on the evaporator coil. These qualitative observations help build a comprehensive understanding of the entire system.

</slide_chunk>

<slide_chunk>

Slide Chunk 9:
Slide Type: Summary Slide
Slide Chunk Title: Key Takeaways: Qualitative & Quantitative Measurements
Slide Chunk Content: In summary, both qualitative and quantitative data are essential for diagnosing and maintaining HVAC systems. Qualitative data offers valuable observations, while quantitative data provides precise measurements. Utilizing both types of data effectively leads to more accurate and efficient system assessments.

</slide_chunk>

</subtopic_1>

<subtopic_2>

<subtopic_name>
Subtopic Name: Accuracy vs. Precision vs. Resolution
</subtopic_name>

<slide_chunk>

Slide Chunk 1:
Slide Type: Transition Slide
Slide Chunk Title: Accuracy vs. Precision vs. Resolution
Slide Chunk Content: Here, we'll delve into three critical concepts in measurement: accuracy, precision, and resolution.

</slide_chunk>

<slide_chunk>

Slide Chunk 2:
Slide Type: Content Slide
Slide Chunk Title: Limitations of Qualitative Data
Slide Chunk Content: While the qualitative data is useful for those initial visual inspections before you pull out your probes, it’s not going to be a reliable tool for the detailed aspects of diagnosis. For example, if you put your hand on a suction line and feel that it's cool, what are the odds that you’d nail the exact temperature to the tenth of a degree Fahrenheit? Realistically, you’ll most likely guess a few degrees off.

</slide_chunk>

<slide_chunk>

Slide Chunk 3:
Slide Type: Content Slide
Slide Chunk Title: Understanding Accuracy
Slide Chunk Content: Most qualitative data will be less accurate than measurements from calibrated test instruments. Accuracy refers to how close a reading is to the actual value. Think of it like an arrow hitting the bullseye; an accurate shot hits the center more reliably than one that lands on the outer bands. We often measure accuracy by comparing a tool to a standard or to other tools of the same type.

</slide_chunk>

<slide_chunk>

Slide Chunk 4:
Slide Type: Content Slide
Slide Chunk Title: Understanding Precision
Slide Chunk Content: Accuracy often gets confused with precision. Precision is the ability of a test instrument to produce consistent results that are close together. Let’s revisit our archery example: even if an archer fails to hit the bullseye, they can still be precise if they hit the same spot on the outer band several times in a row. However, if they hit the bullseye several times in a row, that’s both accurate and precise.

</slide_chunk>

<slide_chunk>

Slide Chunk 5:
Slide Type: Content Slide
Slide Chunk Title: Understanding Resolution
Slide Chunk Content: Accuracy is also often confused with resolution. Resolution is the smallest detectable change that a test instrument can read. For example, a thermostat’s temperature sensor might read to the nearest degree, whereas a digital oral thermometer or a pipe clamp might read to the nearest tenth of a degree. These tools have different resolutions.

</slide_chunk>

<slide_chunk>

Slide Chunk 6:
Slide Type: Summary Slide
Slide Chunk Title: Key Takeaways: Accuracy, Precision, and Resolution
Slide Chunk Content: To summarize, accuracy refers to how close a measurement is to the actual value, precision indicates the consistency of measurements, and resolution defines the smallest change that can be detected by an instrument. Understanding these concepts ensures that you choose the right tools for accurate and reliable diagnostics.

</slide_chunk>

</subtopic_2>

<subtopic_3>

<subtopic_name>
Subtopic Name: Absolute & Differential Measurements
</subtopic_name>

<slide_chunk>

Slide Chunk 1:
Slide Type: Transition Slide
Slide Chunk Title: Absolute & Differential Measurements
Slide Chunk Content: Let us understand absolute and differential measurements.

</slide_chunk>

<slide_chunk>

Slide Chunk 2:
Slide Type: Content Slide
Slide Chunk Title: Understanding Differential Measurements
Slide Chunk Content: Insights like static pressure drop describe the difference between multiple points, which is known as a differential measurement. For instance, measuring the static pressure at one point in the ductwork gives you data about the pressure being exerted in all directions. However, this doesn't provide the same insight as measuring two different points and noticing a large drop over a component like a filter.

</slide_chunk>

<slide_chunk>

Slide Chunk 3:
Slide Type: Content Slide
Slide Chunk Title: Impact of Differential Pressure
Slide Chunk Content: If you measure a large differential over a filter, it indicates that your filter is too restrictive and negatively impacting the HVAC equipment’s performance, either because it’s dirty or by design.

</slide_chunk>

<slide_chunk>

Slide Chunk 4:
Slide Type: Content Slide
Slide Chunk Title: Voltage Drop Measurements
Slide Chunk Content: The same is true of your meter leads when you measure voltage drop; you’re measuring the difference between two points, and that’s exactly why you’ll read 0 voltage across a closed switch. There will be live power, but there is no difference in the voltage or electrical 'pressure' across a switch. If the switch is open, the electrical 'pressure' is cut off where the circuit is incomplete, so we could read 240 volts from one point to another across the open switch.

</slide_chunk>

<slide_chunk>

Slide Chunk 5:
Slide Type: Content Slide
Slide Chunk Title: Delta T (ΔT) as Differential Measurement
Slide Chunk Content: Delta T (ΔT) is another differential measurement; you’re measuring the temperature split across the coil. Plus, delta is used to note a 'change' or 'difference' in scientific equations.

</slide_chunk>

<slide_chunk>

Slide Chunk 6:
Slide Type: Content Slide
Slide Chunk Title: Absolute Measurements
Slide Chunk Content: Compare those types of measurements to something like your liquid line temperature or compressor amps. You’re measuring only one point, so you’re getting an absolute measurement. You may compare the data to a standard, but you’re not measuring the difference between points A and B on a system.

</slide_chunk>

<slide_chunk>

Slide Chunk 7:
Slide Type: Summary Slide
Slide Chunk Title: Key Takeaways: Absolute & Differential Measurements
Slide Chunk Content: In summary, understanding absolute and differential measurements is essential for effective HVAC diagnostics. Absolute measurements provide data from single points, while differential measurements highlight differences between two points. Additionally, choosing the right tool with appropriate accuracy, precision, and resolution ensures reliable and efficient system assessments.

</slide_chunk>

</subtopic_3>

</slide_chunks>

<output>

<learning_objectives>

After completing this topic, you’ll be able to:
1. Spot the difference between qualitative and quantitative measurements during HVAC diagnostics.
2. Explain how accuracy, precision, and resolution affect your choice of diagnostic tools.
3. Describe when to use absolute vs. differential measurements to evaluate HVAC system performance.

</learning_objectives>

</output>

</example>

</examples>
"""



def generate_learning_objectives_slide(course_name, target_audience, topic, slide_chunks, llm="gemini_2_flash"):
    """
    Generate a Learning Objectives slide for a given topic.

    :param course_name: The name of the course.
    :param target_audience: The target audience for the course.
    :param topic: The topic for which learning objectives are generated.
    :param slide_chunks: The structured slide chunks content for the topic.
    :param llm: The language model to use.
    :return: The generated learning objectives slide content.
    """

    # Initialize the agent
    learning_objectives_agent = Chain(llm=llm, tags=['learning_objectives'])

    # Add the user message
    learning_objectives_agent.add_message(
        role="user",
        content=generate_learning_objectives_slide_prompt.format(
            course_name=course_name,
            target_audience=target_audience,
            topic=topic,
            slide_chunks=slide_chunks
        ) + generate_learning_objectives_slide_prompt_examples
    )

    # Run the agent
    response = learning_objectives_agent.run()

    return response['learning_objectives']


# Function to format slide chunks for input (Topic-Level)
def format_slide_chunks_for_learning_objective(topic_df):
    """
    Formats slide chunks by grouping them under their respective subtopics
    to match the required input structure for generating learning objectives.

    :param topic_df: The DataFrame containing all slide chunks for a topic.
    :return: A formatted string representing the slide chunks grouped by subtopics.
    """
    formatted_chunks = []
    subtopic_groups = topic_df.groupby("Subtopic", sort=False)  # Group slide chunks by subtopic

    for subtopic_idx, (subtopic, subtopic_df) in enumerate(subtopic_groups, start=1):
        formatted_subtopic = f"""
<subtopic_{subtopic_idx}>
<subtopic_name>
Subtopic Name: {strip_section_prefix(subtopic)}
</subtopic_name>
        """.strip()

        subtopic_chunks = []
        for chunk_idx, (_, row) in enumerate(subtopic_df.iterrows(), start=1):
            subtopic_chunks.append(f"""
<slide_chunk>
Slide Chunk {chunk_idx}:
Slide Type: {row['Slide Type']}
Slide Chunk Title: {strip_section_prefix(row['Slide Chunk Title'])}
Slide Chunk Content: {row['Slide Chunk']}
</slide_chunk>
            """.strip())

        formatted_subtopic += "\n\n" + "\n\n".join(subtopic_chunks) + f"\n</subtopic_{subtopic_idx}>"
        formatted_chunks.append(formatted_subtopic)

    return "\n\n".join(formatted_chunks)



def process_topic(topic, slide_chunks_df, course_name, target_audience, llm):
    """
    Process a single topic to generate learning objectives and update the DataFrame.

    :param topic: The topic to process.
    :param slide_chunks_df: The DataFrame containing slide chunks.
    :param course_name: Name of the course.
    :param target_audience: Target audience of the course.
    :param llm: The language model to use.
    :return: The topic and the updated DataFrame for that topic.
    """
    topic_df = slide_chunks_df[slide_chunks_df["Topic"] == topic].copy()  # Get topic-specific slides

    print(f"\n🚀 Processing Topic: {topic}")
    print("-" * 100)

    # Format slide chunks for the Learning Objectives Agent
    formatted_slide_chunks = format_slide_chunks_for_learning_objective(topic_df)

    # Generate Learning Objectives
    print("⏳ Running Learning Objectives Agent...")
    learning_objectives = generate_learning_objectives_slide(
        course_name, target_audience, topic, formatted_slide_chunks, llm
    )
    print("✅ Learning Objectives Generated")
    
    if isinstance(learning_objectives, list):
        learning_objectives = "\n".join(learning_objectives)

    # Create a new row for the Learning Objectives Slide
    learning_objectives_row = {
        "Topic": topic,
        "Subtopic": "",  # Leave blank
        "Slide Type": "Learning Objectives Slide",
        "Slide Chunk Title": "Learning Objectives",
        **{col: "" for col in slide_chunks_df.columns if col not in ["Topic", "Subtopic", "Slide Type", "Slide Chunk Title", "slide_chunk"]},  # Keep other columns empty
        "learning_objectives_added_slide_chunk": learning_objectives  # Store the generated learning objectives
    }

    # Insert Learning Objectives row at the top of the topic
    topic_df = pd.concat([pd.DataFrame([learning_objectives_row]), topic_df], ignore_index=True)

    # Ensure all existing slide rows have correct "slide_chunk" values
    topic_df.loc[1:, "learning_objectives_added_slide_chunk"] = topic_df.loc[1:, "Slide Chunk"]

    return topic, topic_df

def run_learning_objectives_agent(sheet, worksheet_name, course_name, target_audience, llm="gemini_2_flash"):
    """
    Runs the Learning Objectives Agent for each topic, updates slide_chunks_df, and writes back to Google Sheets.

    :param sheet: The Google Sheets object.
    :param worksheet_name: The name of the worksheet.
    :param course_name: Name of the course.
    :param target_audience: Target audience of the course.
    :param llm: The language model to use.
    """

    # Load Slide Chunks Data
    slide_chunks_sheet, slide_chunks_df = get_sheet_data_and_df(sheet, worksheet_name)

    # If learning_objectives in slide_chunks_df.columns skip processing
    if "learning_objectives_added_slide_chunk" in slide_chunks_df.columns and not slide_chunks_df["learning_objectives_added_slide_chunk"].isnull().all():
        print("Learning Objectives already processed. Skipping.")
        return

    # Ensure "slide_chunk" column exists (new column to store Learning Objectives)
    if "learning_objectives_added_slide_chunk" not in slide_chunks_df.columns:
        slide_chunks_df["learning_objectives_added_slide_chunk"] = ""

    # Process each unique topic
    unique_topics = slide_chunks_df["Topic"].unique()

    # Prepare for parallel processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        for topic in unique_topics:
            # Submit the task for each topic
            future = executor.submit(process_topic, topic, slide_chunks_df, course_name, target_audience, llm)
            futures_map[future] = topic
            
        total_tasks = len(futures_map)
        progress = SmartProgressBar(total_tasks=total_tasks, description="Percent complete:")

        # Temporary dict to collect updated topic data
        topic_data_map = {}

        # Collect results as they complete
        for future in tqdm(as_completed(futures_map)):
            topic, updated_topic_df = future.result()
            topic_data_map[topic] = updated_topic_df
            progress.update()

        # Reconstruct slide_chunks_df in original topic order
        ordered_topic_dfs = [topic_data_map[topic] for topic in unique_topics if topic in topic_data_map]
        slide_chunks_df = pd.concat(ordered_topic_dfs, ignore_index=True)

        # Write once to Google Sheets after all processing
        set_with_dataframe(slide_chunks_sheet, slide_chunks_df)
        print("\n✅ All Topics Processed and Updated in Google Sheets 🚀\n")