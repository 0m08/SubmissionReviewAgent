from modules.chain import Chain
from agents.slide_chunks.format_inputs import strip_roman_numerals, strip_section_prefix, extract_failed_criteria
from services.sheets_service import get_sheet_data_and_df, save_to_sheet
# from gspread_dataframe import set_with_dataframe
from tqdm import tqdm
from services.smart_progress_bar import SmartProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed




generate_humanlike_review_prompt = """ We are creating structured slide content for an e-learning course. You are an expert instructional designer and e-learning content creator, crafting educational materials with the fluency and adaptability of a world-class writer — producing content indistinguishable from human authorship. In this role, you are serving as a Slide Content Review Agent, responsible for evaluating whether the generated slide content sounds natural, human-written, and engaging - rather than AI-generated, robotic, or overly mechanical. Your task is to analyze the slide content and determine if it meets the required quality standards. The review should focus on eliminating unnatural phrasing, improving sentence structure, and ensuring the content aligns with human-like writing patterns, as well as other human-like qualities, while maintaining clarity and instructional effectiveness. Be extremely vigilant for subtle signs of AI-like construction in the slide content, even if it appears superficially polished but lacks depth, variation, or a genuinely human voice. Your task is not only to detect obvious robotic phrasing, but also to identify mild signals of artificiality and recommend revisions to improve human-likeness.

Below is the course information for which you will be reviewing the slide content:

<course_information>
Course Name: {course_name}
Target Audience: {target_audience}
</course_information>

Below is the topic and subtopic to which the slide belongs:

<topic>
Topic Name: {topic}
</topic>

<subtopic>
Subtopic Name: {subtopic}
</subtopic>

Below is the slide that needs to be reviewed:

<slide>

Slide Type:
{slide_type}

Slide Title:
{slide_title}

Slide Content:
{slide_content}

</slide>

A) Review Criteria: You must evaluate this slide chunk based on the following predefined Review Criteria.

  1) Sentence Structure and Readability:
    - Are sentences clear, well-structured, and easy to follow?
    - Does the content use short, structured, and natural sentences rather than long, dense paragraphs?
    - Are long ideas broken into multiple sentences for better readability?
    - Do transitions between sentences feel natural and fluid, or are they abrupt, overly segmented, or disconnected?

  2) Natural and Human-Like Tone:
    - Does the slide content sound like it was written by a human instructional designer rather than AI-generated?
    - Does the slide content sound overly formal or stiff for the given target audience?
    - Are there opportunities to slightly relax the tone without becoming unprofessional?
    - Is the structure natural or too rigid, lacking a smooth progression of ideas?
    - Does the tone feel conversational and approachable, or does it come across as overly mechanical or impersonal?
    - Are there natural opportunities to use contractions (e.g., "you're", "it's", "don't") to improve flow and conversational tone — and if so, are they appropriately used to make the content feel more human and less robotic?

  3) Engagement and Natural Flow:
    - Does the content engage the reader with a human tone, or does it feel flat and mechanical?
    - Does the slide chunk read in a way that feels effortless and clear, or does it feel artificial?
    - Are the key points presented in a simple, approachable manner, or do they seem obscured by unnecessary complexity?

  4) Avoiding AI-Like Overuse of Passive Voice:
    - Is the content written in an active voice where appropriate?
    - Does it avoid the overuse of passive voice, which can make sentences sound stiff or unnatural?

  5) Perplexity and Burstiness:
    - Perplexity: Does the writing introduce variation in word choice, phrasing, and sentence structure to make it feel naturally written?
    - Burstiness: Does the text contain a mix of short and long sentences instead of uniform, AI-like sentence lengths?

  6) Conciseness and Clear Communication:
    - Is the content concise, yet informative, avoiding unnecessary filler or overly verbose explanations?
    - Does it strike the right balance between being clear and being detailed?
    - Does the content use accessible, easy-to-understand, everyday language where possible — even when explaining technical concepts — and avoid unnecessary jargon or overly formal vocabulary?

  7) Missed Opportunities for Direct Address and Audience Engagement:
    - Are there appropriate opportunities to use direct addresses (e.g., “you,” “your”) that would help the content feel more personal, relatable, or human — but the content fails to take advantage of them?
    - Are there opportunities to use rhetorical questions, simple calls to action, or conversational cues that would strengthen audience connection — but they are missed?
    - Does the current point of view (e.g., passive or overly formal third-person) create unnecessary distance when a more engaging tone would be suitable for the subject and target audience?

  8) Awkward Collocations and Expressions:
    - Does the slide content contain any unnatural or awkward combinations of words that feel “off” to a fluent reader?
    - Are there phrasings or sentence constructions that seem overly literal, robotic, or unlikely to be used by a human?
    - Does the content miss opportunities to use more natural expressions, idioms, or word pairings that would feel smoother or more fluent?

  9) Authenticity and Personalization:
    - Does the slide content include subtle personal touches, authentic anecdotes, relatable experiences, or minor references that align well with human communication patterns — where appropriate and where there are opportunities to do so?
    - Does it completely avoid personalized details or references, wherever possible making it sound generic or impersonal?
    - Are there natural opportunities for minor personalization or authenticity that the content neglects?

  10) Predictability and Cliched Phrasing:
    - Does the content use overly common, repetitive, or formulaic phrases that reduce its perceived originality?
    - Does the content miss opportunities to use fresher, more dynamic wording that sounds more human?

  11) Colloquial Expressions and Idioms:
   - Does the content naturally incorporate light, conversational expressions or idiomatic language — only where appropriate and contextually fitting?
   - If such expressions are used, do they enhance the content’s clarity, engagement, or human-likeness without sounding forced or unprofessional?
   - If not used, are there clear missed opportunities where their inclusion would have made the content more relatable or fluid?

B) Evaluation Guidelines: Before assigning a Pass or Fail verdict, carefully review the slide content against each of the predefined review criteria. Your evaluation should ensure that the slide content feels naturally written by a human instructional designer, avoiding AI-generated patterns, robotic phrasing, or unnatural structure. Use the following approach to guide your evaluation:

    - Be highly critical and strict in your assessment. Do not overlook any issues. If the slide content does not fully meet a criterion, strictly assign a "Fail" verdict for that criterion and provide specific, actionable suggestions for improvement.
    - Evaluate each review criterion separately to ensure a thorough and precise assessment.
    - If the slide content fully meets a specific criterion, provide a brief justification confirming its effectiveness in the feedback section. In the suggestion section, state "No changes needed" and assign a verdict of "Pass" for that criterion.
    - If the slide content fails a specific criterion, clearly explain the issue in the feedback section. In the suggestion section, provide direct, actionable recommendations to address the problem while maintaining the instructional alignment and strictly assign a "Fail" verdict for that criterion.
    - Strictly ensure that all your feedback and suggestions focus on refining the existing slide content - do not introduce new information unnecessarily or alter its original intent. The content itself should remain unchanged; your task is only to improve its language, tone, and flow to make it feel more human-written and natural while preserving its meaning.
    - Strictly avoid suggesting rhetorical or formulaic openings like “Ever wondered…”, “Ever notice…”, “Ever been…”, “Have you ever…”, “Imagine…”, or similar patterns. These expressions are frequently overused by AI models and can make multiple slides sound repetitive and unnatural.
    - While making suggestions for failed criteria, you must consider the total impact across all failed items. Your suggestions across all failed criteria should strictly result in no more than a 10–20% increase in the original slide content’s length. You are strictly responsible for ensuring that your collective suggestions across all failed criteria do not exceed a total 10–20% increase in content length. If adhering to this limit requires omitting or deferring less essential improvements, you must do so. Breaching this threshold is not permitted, regardless of the number of failed criteria.
    - When suggesting improvements, do not expand or lengthen the original slide content beyond what is necessary for human-like improvement. Only propose adding new sentences or phrases if they are absolutely essential to enhance clarity, tone, or human-likeness - and even then, keep such additions minimal and purposeful. The revised content should remain similar in length to the original. Avoid suggestions that would make the content substantially longer. The goal is to maintain concise, tightly focused, and instructionally efficient communication while improving human-likeness.
    - Avoid proposing additions or elaborations unless they are essential for improving human-likeness. Be selective and concise in your recommendations.
    - Ensure that while improving human-likeness, the writing continues to function as instructional content - delivering clear, purposeful learning without becoming overly casual, vague, or drifting off-topic. The goal is to humanize the writing without diluting its instructional function.


Present your output strictly in the following format:

<output>

<evaluation_breakdown>

(Before assigning a verdict, carefully analyze the slide content holistically. Document your thought process and analysis for each criterion in the following format)

- Sentence Structure and Readability: Provide your detailed analysis explaining whether the slide content meets this criterion.
- Natural and Human-Like Tone: Provide your detailed analysis explaining whether the slide content meets this criterion.
- Engagement and Natural Flow: Provide your detailed analysis explaining whether the slide content meets this criterion.
- Avoiding AI-Like Overuse of Passive Voice: Provide your detailed analysis explaining whether the slide content meets this criterion.
- Perplexity and Burstiness: Provide your detailed analysis explaining whether the slide content meets this criterion.
- Conciseness and Clear Communication: Provide your detailed analysis explaining whether the slide content meets this criterion.
- Missed Opportunities for Direct Address and Audience Engagement: Provide your detailed analysis explaining whether the slide content meets this criterion.
- Awkward Collocations and Expressions: Provide your detailed analysis explaining whether the slide content meets this criterion.
- Authenticity and Personalization: Provide your detailed analysis explaining whether the slide content meets this criterion.
- Predictability and Cliched Phrasing: Provide your detailed analysis explaining whether the slide content meets this criterion.
- Colloquial Expressions and Idioms: Provide your detailed analysis explaining whether the slide content meets this criterion.

</evaluation_breakdown>

Based on your above evaluation, give your final verdict for each review criterion in the following format:

<evaluation_output>

<criterion>
Review Criterion: Sentence Structure and Readability
Feedback: Provide your feedback for this criterion. If the slide content passes, confirm that the sentence structure is clear, well-organized, improves readability, and transitions smoothly between sentences. If it fails, describe specific issues related to sentence complexity, awkward phrasing, unnatural flow, or lack of cohesion between sentences — including abrupt, overly segmented, or disconnected transitions.
Verdict: [Pass/Fail based on your evaluation]
Suggestion: Provide specific suggestions to improve clarity, sentence structure, and the natural flow of ideas between sentences. If the criterion passes, state: "No changes needed."
</criterion>

<criterion>
Review Criterion: Natural and Human-Like Tone
Feedback: Provide your feedback for this criterion. If the slide content passes, confirm that it sounds like it was written by a human instructional designer, with an appropriate tone for the target audience. If it fails, describe any robotic, overly formal, unnatural, or impersonal phrasing — especially if the tone lacks warmth, conversational flow, or sounds overly mechanical.
Verdict: [Pass/Fail based on your evaluation]
Suggestion: Provide specific suggestions to improve the tone and make it sound more natural, human-like, and conversational where appropriate - including using contractions or informal phrasing when it improves flow and feels appropriate for the context. If the criterion passes, state: "No changes needed."
</criterion>

<criterion>
Review Criterion: Engagement and Natural Flow
Feedback: Provide your feedback for this criterion. If the slide content passes, confirm that it is engaging, easy to read, and flows naturally like human-written content. If it fails, describe any instances where the writing feels mechanical, disjointed, overly rigid, or lacks smooth progression, making it difficult to follow. Highlight if key points are obscured by unnecessary complexity or if the content feels flat and unengaging.
Verdict: [Pass/Fail based on your evaluation]
Suggestion: Suggest ways to improve the content’s engagement and logical flow while ensuring an effortless reading experience. If the criterion passes, state: "No changes needed."
</criterion>

<criterion>
Review Criterion: Avoiding AI-Like Overuse of Passive Voice
Feedback: Provide your feedback for this criterion. If the slide content passes, confirm that active voice is appropriately used and passive voice is not excessive. If it fails, describe cases where overuse of passive voice makes sentences stiff or unclear.
Verdict: [Pass/Fail based on your evaluation]
Suggestion: Recommend rewording sentences into active voice where necessary to enhance clarity and readability. If the criterion passes, state: "No changes needed."
</criterion>

<criterion>
Review Criterion: Perplexity and Burstiness
Feedback: Provide your feedback for this criterion. If the slide content passes, confirm that it has a natural variation in sentence structure, word choice, and phrasing. If it fails, describe where the content feels repetitive, uniform, or AI-like in sentence patterns.
Verdict: [Pass/Fail based on your evaluation]
Suggestion: Suggest improvements to introduce more variation in word choice and sentence structures, ensuring a more dynamic and human-like writing style. If the criterion passes, state: "No changes needed."
</criterion>

<criterion>
Review Criterion: Conciseness and Clear Communication
Feedback: Provide your feedback for this criterion. If the slide content passes, confirm that it is concise, free from unnecessary filler, and effectively communicates key points. Also note whether the language is easy to understand, avoids overly formal phrasing, and keeps technical explanations accessible to the target audience. If it fails, describe where the content is too verbose, redundant, unclear, or uses complex language that may feel unnatural or difficult for learners.
Verdict: [Pass/Fail based on your evaluation]
Suggestion: Recommend removing unnecessary words, simplifying explanations, and using clearer, more accessible vocabulary. If needed, suggest rewording overly formal or jargon-heavy sections to make them easier to follow while preserving key details. If the criterion passes, state: "No changes needed."
</criterion>

<criterion>
Review Criterion: Missed Opportunities for Direct Address and Audience Engagement
Feedback: Provide your feedback for this criterion. If the slide content passes, confirm that it appropriately uses direct audience engagement or that there were no suitable opportunities to do so. If it fails, describe any missed opportunities where using second-person tone, rhetorical questions, or audience cues could have made the content feel more human and connected.
Verdict: [Pass/Fail based on your evaluation]
Suggestion: Suggest specific ways the content could incorporate direct address or subtle audience engagement — only if such additions would feel natural, suitable for the context, preserve the original meaning of the slide content and beneficial for improving human-likeness. If the criterion passes, state: "No changes needed."
</criterion>

<criterion>
Review Criterion: Awkward Collocations and Expressions
Feedback: Provide your feedback for this criterion. If the slide content passes, confirm that all word choices and phrasing sound fluent and natural, with no awkward or robotic expressions. If it fails, highlight any unnatural collocations or phrases that disrupt readability or sound AI-generated.
Verdict: [Pass/Fail based on your evaluation]
Suggestion: Provide specific suggestions to improve fluency by replacing awkward phrases with more natural alternatives. Only make suggestions that preserve the original meaning of the slide content. If the criterion passes, state: "No changes needed."
</criterion>

<criterion>
Review Criterion: Authenticity and Personalization
Feedback: Provide your feedback for this criterion. If the slide content passes, confirm that it includes subtle personal touches, relatable cues, or authentic phrasing that enhance its human-like quality — or that there were no suitable opportunities to include such personalization. If it fails, explain how the content feels overly generic, impersonal, or disconnected, and point out any opportunities where relatable references or authentic cues could have been integrated naturally.
Verdict: [Pass/Fail based on your evaluation]
Suggestion: Suggest subtle improvements for adding light personal touches, relatable context, or authentic expressions that would make the content feel more naturally human. Only suggest changes if they feel appropriate for the context and the target audience. Only make suggestions that preserve the original meaning of the slide content. If the criterion passes, state: "No changes needed."
</criterion>

<criterion>
Review Criterion: Predictability and Clichéd Phrasing
Feedback: Provide your feedback for this criterion. If the slide content passes, confirm that it avoids overly predictable or formulaic language and uses varied, fresh, and dynamic phrasing. If it fails, describe any use of repetitive, templated, or overly familiar phrasing that reduces the originality or human-like quality of the writing.
Verdict: [Pass/Fail based on your evaluation]
Suggestion: Suggest alternative phrasing that feels more natural, dynamic, or original without altering the meaning or instructional intent. Focus on replacing clichés or predictable sentence patterns with more expressive and human-like language. If the criterion passes, state: "No changes needed."
</criterion>

<criterion>
Review Criterion: Colloquial Expressions and Idioms
Feedback: Provide your feedback for this criterion. If the slide content passes, confirm that idiomatic or conversational expressions are used appropriately and naturally where suitable — or that none were used because there were no clear opportunities. If the content fails, describe any missed opportunities where such expressions could have made the content more relatable, or highlight any instances where they were used awkwardly or out of context.
Verdict: [Pass/Fail based on your evaluation]
Suggestion: Suggest using a light, conversational or idiomatic phrase only if it would feel natural, helpful, and appropriate for the context. If the criterion passes, state: "No changes needed."
</criterion>

</evaluation_output>

</output>
"""





def generate_humanlike_review(course_name, target_audience, topic, subtopic, slide_type, slide_title, slide_content, llm="gemini_2_flash"):
    """
    Generate a human-likeness review for a slide content based on predefined review criteria.

    :param course_name: The name of the course.
    :param target_audience: The target audience for the course.
    :param topic: The topic to which the slide belongs.
    :param subtopic: The subtopic to which the slide belongs.
    :param slide_type: The type of the slide
    :param slide_title: The title of the slide.
    :param slide_content: The content of the slide.
    :param llm: The language model to use.
    :return: The structured review output.
    """

    # Initialize the review agent
    review_agent = Chain(llm=llm, tags=["evaluation_output"])

    # Add the user message
    review_agent.add_message(
        role="user",
        content=generate_humanlike_review_prompt.format(
            course_name=course_name,
            target_audience=target_audience,
            topic=topic,
            subtopic=subtopic,
            slide_type=slide_type,
            slide_title=slide_title,
            slide_content=slide_content,
        )
    )

    # Run the review agent
    response = review_agent.run()

    return response["evaluation_output"]



generate_humanlike_revise_prompt = """We are creating structured slide content for an e-learning course. You are an expert instructional designer and e-learning content creator, crafting educational materials with the fluency and adaptability of a world-class writer — producing content indistinguishable from human authorship. In this role, you are serving as a Slide Content Revisor Agent, responsible for refining slide content that was flagged as AI-generated, robotic, or overly mechanical. Your task is to carefully revise the slide content based on the feedback and suggestions received from a review agent. Your revisions should specifically address the issues identified in the review agent's evaluation while ensuring that all other content remains unchanged. You must prioritize only the most essential improvements, ensuring that the revised content remains concise and does not exceed a total length increase of 10–20%.

Below is the course information for which you will be revising the slide content:

<course_information>
Course Name: {course_name}
Target Audience: {target_audience}
</course_information>

Below is the topic and subtopic to which the slide belongs:

<topic>
Topic Name: {topic}
</topic>

<subtopic>
Subtopic Name: {subtopic}
</subtopic>

Below is the original slide content that requires revision:

<slide>

Slide Type:
{slide_type}

Slide Title:
{slide_title}

Slide Content:
{slide_content}

</slide>

Below are the specific review criteria that failed and require revision:

<failed_criteria>
{failed_criteria}
</failed_criteria>

You are tasked with carefully revising the slide content based on the feedback and suggestions provided for each failed criterion, ensuring that the content sounds natural, human-written, and engaging while maintaining its original meaning and instructional intent.

Revision Guidelines: Before revising the slide content, follow these principles to ensure the revisions align with the required quality standards:

1) Apply Only Necessary Changes:
  - Modify the slide content only in response to the specific feedback and suggestions provided in the failed criteria.
  - Do not change or remove any part of the content that was not flagged for revision.

2) Preserve Original Meaning and Intent:
  - Strictly ensure that the instructional message, factual accuracy, and core meaning of the slide content remain unchanged while making the revisions.
  - Do not reinterpret or reframe the content in a way that alters its original intent.

3) Ensure Seamless Integration:
  - Revised content should blend naturally with the surrounding text, maintaining a smooth and logical flow.
  - Avoid making revisions that feel disjointed or inconsistent with the rest of the slide content.

4) Maintain Conciseness and Length Constraint:
  - Strictly ensure that your revised slide content does not exceed a total increase of 10–20% in length compared to the original.
  - Even if multiple suggestions were given, you must prioritize only the most essential and high-impact changes.
  - If adhering to this limit requires skipping or scaling down certain suggestions, you must do so.
  - You are strictly responsible for ensuring the overall revised content remains within this threshold — this limit must not be breached.
  - Do not blindly implement all suggestions. Always assess the value and impact of each one before deciding to include it.

5) Maintain Conciseness and Avoid Unnecessary Expansion:
  - Keep the revised content as concise and focused as the original.
  - Avoid expanding or lengthening the slide content unnecessarily.
  - Add new sentences or phrases only when they are absolutely essential to improve clarity, tone, or natural flow — and even then, keep additions minimal and purposeful.
  - Do not elaborate unnecessarily or over-explain in ways that dilute instructional efficiency.
  - All revisions must prioritize tight, efficient communication that aligns with human-like but instructionally sound writing.
  - The revised slide content should remain close in length to the original — avoid making it substantially longer under any condition.
  - Clarity, tone, and flow improvements should never come at the cost of length discipline.

6) Avoid Overused Rhetorical Openings:
  - Strictly avoid beginning the slide with rhetorical or formulaic openings like “Ever wondered…”, “Ever been…”, “Ever notice…”, “Have you ever…”, “Imagine…”, or similar patterns — even if suggested in the review feedback and suggestions.
  - These openings are commonly overused by AI models and reduce the naturalness and variety of the slide content.

7) Revised Slide Content:
  - Give only the revised slide content in your output and not any other details like slide title, slide type, etc.

Present your output strictly in the following format:

<output>

<evaluation_breakdown>

(Before making revisions, carefully analyze the feedback and suggestions provided. While planning your revisions, consider the total length impact across all criteria. Your combined changes should not increase the overall slide content length by more than 10–20%. Prioritize only the most essential and high-impact suggestions. Omit or scale down lower-priority improvements if needed to stay within this limit. Document your thought process for refining the slide content for the failed criteria in the following format)

- [Insert failed criterion name]: Provide your detailed thought process and approach for revising this aspect of the slide content.

- [Insert failed criterion name]: Provide your detailed thought process and approach for revising this aspect of the slide content.

Repeat this pattern for each failed criterion.

<prioritization_strategy>

Explain how you plan to prioritize which suggestions to implement in order to stay within the 10–20% total length increase limit. Mention which lower-priority suggestions you plan to skip or scale back, and justify your decisions to maintain brevity while still achieving human-likeness improvements.

</prioritization_strategy>

</evaluation_breakdown>

Based on your above evaluation, provide the revised slide content in the following format:

<revised_slide_content>
[Insert the revised version of the slide content, ensuring all necessary improvements have been made while keeping the original information intact. Strictly ensure that you only give the revised slide content here and nothing else.]
</revised_slide_content>

</output>
"""


generate_humanlike_revise_few_shot_prompt = """Use the following examples to guide how you revise slide content to make it sound more human-written — natural, fluent, and engaging — while preserving the original meaning and instructional intent.

These examples demonstrate the style and tone that your revisions should aim for. They show how to make slide content feel more natural and less AI-generated, without expanding the length unnecessarily or changing the underlying message.

Each example contains:
1. The original Slide type and its Slide content that exhibited AI-like patterns or tone.
2. The revised version that improves human-likeness while keeping the content concise and instructionally focused.

Along with using the review agent’s feedback and suggestions, follow the patterns and revision style demonstrated in these examples to improve human-likeness. Your revisions should follow a similar pattern exhibited in these examples — subtle but effective improvements that make the content feel authentic, human-like, easy to read, and natural for a human learner.

Review the following examples to understand how human-like improvements are applied to slide content:

<examples>

<example>

<original_slide_content>

Slide Type:
Learning Objectives Slide

Slide Content:
By the end of this topic, you will be able to:
1. Define cubic feet, PSI, and vacuum, and explain their relevance to airflow in HVAC systems.
2. Differentiate between CFM, SCFM, and ACFM, and explain why each is important for airflow measurement.
3. Explain how temperature, humidity, and altitude affect air density and ACFM in HVAC systems.
4. Distinguish between total system airflow and point measurements, and explain Bernoulli's Principle and its applications in HVAC systems.

</original_slide_content>

<revised_slide_content>

After finishing this section, you'll know how to:
1. Explain what cubic foot, PSI, and vacuum mean and why they matter for airflow in HVAC systems.
2. Tell the difference between CFM, SCFM, and ACFM and describe how people use them to measure airflow.
3. Spot how temperature, humidity, and altitude change air density and ACFM in HVAC systems.
4. Talk about total system airflow compared to point measurements, and how Bernoulli's Principle relates to airflow in HVAC systems.

</revised_slide_content>

</example>

<example>

<original_slide_content>

Slide Type:
Learning Objectives Slide

Slide Content:
By the end of this topic, you will be able to:
1. Differentiate between axial and centrifugal fans/blowers and identify their common applications.
2. Explain how static pressure affects airflow and motor load in both axial and centrifugal fan/blower systems.
3. Describe the importance of motor load windows and the consequences of operating outside of them.
4. Explain the relationship between CFM, blower wheel selection, and static pressure when replacing or specifying blower components.

</original_slide_content>

<revised_slide_content>

Once you finish this section, you'll know how to:
1. Tell the difference between axial and centrifugal fans/blowers and spot where they're used.
2. Understand how static pressure affects airflow and motor load in both axial and centrifugal fan/blower setups.
3. Grasp why motor load windows matter and what happens when you operate outside of them.
4. Get the link between CFM, blower wheel choice, and static pressure when you're swapping out or picking blower parts.

</revised_slide_content>

</example>

<example>

<original_slide_content>

Slide Type:
Learning Objectives Slide

Slide Content:
By the end of this topic, you will be able to:
1. Define adjustable pulleys and explain their benefits in controlling airflow in HVAC systems.
2. Describe the mechanical principles by which adjustable pulleys control fan speed and airflow (CFM).
3. Perform the steps to safely adjust an adjustable pulley, including belt tensioning and amperage monitoring.
4. Recognize the potential impact of pulley adjustments on air balance and motor load.

</original_slide_content>

<revised_slide_content>

After completing this section, you'll be able to:
1. Explain what adjustable pulleys are and how they help control airflow in HVAC systems.
2. Understand the mechanics behind how adjustable pulleys manage fan speed and airflow (CFM).
3. Perform the steps to adjust a pulley, including how to tension the belt and monitor amperage.
4. Understand how adjusting pulleys can affect air balance and motor load.

</revised_slide_content>

</example>

<example>

<original_slide_content>

Slide Type: Content Slide

Slide Content:
Be aware that adjusting sheaves can affect the overall air balance of the system. Changing the airflow in one area can impact other zones, potentially leading to comfort issues or inefficient operation. Consider the impact on sensible heat ratios and fresh air intake. *Avoid altering sheaves unless you understand the system's air balance.* If unsure, consult with a more experienced technician or supervisor.

</original_slide_content>

<revised_slide_content>

Sheave adjustments can influence the system's general air balance, so be aware. Modifying the airflow in one area can affect other zones, which can lead to comfort issues or inefficient operation. Think of the effects on fresh air intake and sensible heat ratios. Steer clear of changing sheaves unless you know the air balance of the system*. If not sure, ask a more seasoned technician or supervisor.

</revised_slide_content>

</example>

<example>

<original_slide_content>

Slide Type: Content Slide

Slide Content:
High static pressure typically leads to reduced airflow. This puts a strain on the blower motor, potentially shortening its lifespan and increasing energy consumption.

</original_slide_content>

<revised_slide_content>

Usually, high stationary pressure results in lower airflow. This leads to strain on your blower motor, possibly reducing its lifetime and raising energy consumption.

</revised_slide_content>

</example>

<example>

<original_slide_content>

Slide Type: Content Slide

Slide Content:
When addressing airflow complaints, consider the throw requirements of the room. If a room has a greater distance to cover, a higher air velocity may be necessary to achieve adequate circulation. However, remember that increasing air velocity also increases noise, so it's a balancing act.

</original_slide_content>

<revised_slide_content>

Take the room's throw requirements into account when resolving complaints about airflow. A higher air velocity might be required to achieve proper circulation in a room with a larger distance to cover. But keep in mind that it's a balancing act, because rising air velocity also raises noise.

</revised_slide_content>

</example>

<example>

<original_slide_content>

Slide Type: Content Slide

Slide Content:

Now that you know the basics of fans and blowers, let's dive into how static pressure – think of it as the resistance to airflow – impacts how they work. Understanding this is super important for when you're diagnosing problems and trying to get the best performance out of a system.

</original_slide_content>

<revised_slide_content>

We have established a basic understanding of fans and blowers; now it is time to learn about static pressure, which, simply stated, is the resistance to airflow, and how it influences fans and blowers. Understanding this information is really important when you are troubleshooting issues and trying to optimize a system's performance.

</revised_slide_content>

</example>

<example>

<original_slide_content>

Slide Type: Summary Slide

Slide Content:
In summary,
* CFM depends on both the blower wheel and housing.
* Increased static pressure reduces CFM.
* Specify both CFM and static pressure when ordering replacements.
* Matching wheel characteristics is important in replacements.

</original_slide_content>

<revised_slide_content>

Let's summarize what we learned in this section,
* CFM is dependent on the blower wheel and housing.
* CFM decreases with increasing static pressure.
* When ordering replacements, make sure to include both static and CFM pressure.
* It is important to match the wheel characteristics when ordering replacements.

</revised_slide_content>

</example>

<example>

<original_slide_content>

Slide Type: Summary Slide

In summary, remember to:
* Use manometers or Magnehelic gauges to measure static pressure.
* Always zero the gauge before you take any readings.
* Place probes correctly on the return and supply sides.
* Use static pressure tips to minimize turbulence.

</original_slide_content>

<revised_slide_content>

In conclusion, keep in mind to:
* Measure static pressure using manometers or Magnehelic gauges.
* Prior to taking any measurements, always zero the gauge.
* Properly position the probes on the supply and return sides.
* To reduce turbulence, use static pressure tips.

</revised_slide_content>

</example>

</examples>
"""


def generate_humanlike_revise(course_name, target_audience, topic, subtopic, slide_type, slide_title, slide_content, failed_criteria, llm="gemini_2_flash"):
    """
    Generate a human-like revision for a slide content based on failed review criteria.

    :param course_name: The name of the course.
    :param target_audience: The target audience for the course.
    :param topic: The topic to which the slide belongs.
    :param subtopic: The subtopic to which the slide belongs.
    :param slide_type: The type of the slide.
    :param slide_title: The title of the slide.
    :param slide_content: The content of the slide (before revision).
    :param failed_criteria: The failed review criteria and their feedback.
    :param llm: The language model to use.
    :return: The structured revised slide content.
    """

    # Initialize the reviser agent
    revise_agent = Chain(llm=llm, tags=["revised_slide_content"])

    # Add the user message
    revise_agent.add_message(
        role="user",
        content=generate_humanlike_revise_prompt.format(
            course_name=course_name,
            target_audience=target_audience,
            topic=topic,
            subtopic=subtopic,
            slide_type=slide_type,
            slide_title=slide_title,
            slide_content=slide_content,
            failed_criteria=failed_criteria,
        ) + generate_humanlike_revise_few_shot_prompt
    )

    # Run the reviser agent
    response = revise_agent.run()

    return response["revised_slide_content"]




def process_slide(index, row, course_name, target_audience, llm, max_iterations):
    """
    Process a single slide for AI Detection Review & Revise.

    :param index: The index of the slide in the DataFrame.
    :param row: The row data of the slide.
    :param course_name: The name of the course.
    :param target_audience: The target audience for the course.
    :param llm: The language model to use.
    :param max_iterations: Maximum number of Review-Revise loops allowed per slide.
    :return: The final slide content and title.
    """
    topic = strip_roman_numerals(row["Topic"])
    subtopic = strip_section_prefix(row["Subtopic"])
    slide_type = row["Slide Type"]
    slide_title = strip_section_prefix(row["checklist_based_slide_title"])
    slide_content = row["checklist_based_slide_content"]  # First iteration uses this

    iteration = 0
    while iteration < max_iterations:
        # Run Review Agent
        review_output = generate_humanlike_review(
            course_name=course_name,
            target_audience=target_audience,
            topic=topic,
            subtopic=subtopic,
            slide_type=slide_type,
            slide_title=slide_title,
            slide_content=slide_content,
            llm=llm
        )

        # Extract failed criteria
        failed_criteria = extract_failed_criteria(review_output)

        if not failed_criteria:
            break

        # Run Revisor Agent
        revised_slide_content = generate_humanlike_revise(
            course_name=course_name,
            target_audience=target_audience,
            topic=topic,
            subtopic=subtopic,
            slide_type=slide_type,
            slide_title=slide_title,
            slide_content=slide_content,
            failed_criteria=failed_criteria,
            llm=llm
        )

        # Update for next iteration
        slide_content = revised_slide_content
        iteration += 1

    return slide_content, row["checklist_based_slide_title"]
    
def run_ai_detection_review_revise(sheet, worksheet_name, course_name, target_audience, llm="gemini_2_flash", max_iterations=3):
    """
    Runs the AI Detection Review & Revise Workflow for all slides in the Google Sheet.

    :param sheet: The Google Sheets object.
    :param course_name: The name of the course.
    :param target_audience: The target audience for the course.
    :param llm: The language model to use (default: gemini_2_flash).
    :param max_iterations: Maximum number of Review-Revise loops allowed per slide.
    """

    # Load Slide Chunks Data
    slide_chunks_sheet, slide_chunks_df = get_sheet_data_and_df(sheet, worksheet_name)

    # Ensure "final_slide_title" and "final_slide_content" columns exist
    if "final_slide_title" not in slide_chunks_df.columns:
        final_title_index = slide_chunks_df.columns.get_loc("checklist_based_slide_content") + 1
        slide_chunks_df.insert(final_title_index, "final_slide_title", "")

    if "final_slide_content" not in slide_chunks_df.columns:
        final_content_index = final_title_index + 1
        slide_chunks_df.insert(final_content_index, "final_slide_content", "")

    # Skip the process if the columns are already filled 
    if slide_chunks_df["final_slide_title"].astype(str).str.strip().ne("").all() and slide_chunks_df["final_slide_content"].astype(str).str.strip().ne("").all():
      print("All slides already processed. Skipping the review and revise process")
      return


    # Prepare for parallel processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit tasks for each slide
        for index, row in slide_chunks_df.iterrows():
            future = executor.submit(process_slide, index, row, course_name, target_audience, llm, max_iterations)
            futures_map[future] = index

        # Collect the results as they complete
        total_tasks = len(futures_map)
        save_interval = 5  # how often to save (in number of completed tasks)

        # Initialize the progress tracker
        progress = SmartProgressBar(total_tasks=total_tasks, description="Percent complete", save_interval = save_interval)

        # Now, pass only the futures (the keys) to as_completed:
        for future in tqdm(as_completed(futures_map)):
            index = futures_map[future]  # retrieve the index
            slide_content, final_slide_title = future.result()

            # Update the df with the final slide content and title
            slide_chunks_df.at[index, "final_slide_content"] = slide_content
            slide_chunks_df.at[index, "final_slide_title"] = final_slide_title

            # Update progress
            progress.update()

            # Check if we should save
            if progress.should_save():
                print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
                save_to_sheet(slide_chunks_sheet, slide_chunks_df)

    # Final save to sheet after all tasks
    print('All slides processed. Saving final DataFrame to sheet.')
    save_to_sheet(slide_chunks_sheet, slide_chunks_df)

    print("\n✅ AI Detection Review & Revise Process Completed 🚀")