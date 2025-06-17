from modules.chain import Chain
from services.sheets_service import (
    get_sheet_data_and_df,
    save_to_sheet,
    clear_worksheet,
)
from services.smart_progress_bar import SmartProgressBar
import streamlit as st
import pandas as pd
import re
from langsmith import traceable


research_checklist_reviewer_prompt = """You are a checklist-based review agent. Your task is to evaluate the structured research-based content generated for an E-learning course using the predefined checklist criteria provided. Each entry corresponds to a topic and includes structured research-based content that covers its subtopics and learning objectives. Your job is to carefully assess whether the content meets the instructional, structural, and quality standards outlined in the checklist.

Below is the course information for which the content was generated:

<course_information>
Course name: {course_name}
Target audience: {target_audience}
</course_information>

Here is the scope of this review:

<scope>
{scope} 
</scope>

Here is the research-based content to review. Each topic and subtopic section is assigned a unique identifier in the format t_X_s_Y, where X is the topic number and Y is the subtopic number.

<research_based_content>
{section_notes}
</research_based_content>

Here is the checklist task and all its associated review criteria:

<checklist_criteria>
{checklist_criteria}
</checklist_criteria>

Follow the instructions below to evaluate the research-based content accurately using the provided checklist criteria:

A) Scope-Based Evaluation Guidelines:

- Each review criterion has a defined scope that determines how to apply the review criteria. Use the scope provided to guide your evaluation:

a) Topic: If the scope is "Topic", you will be given only one topic. Review the subtopics, learning objectives, and the corresponding research-based content provided for that topic. Apply each review criterion to this topic-level content.
b) Course: If the scope is "course", you will be given all topics in the course along with their associated subtopics, learning objectives, and research-based content. Review the entire course holistically, and apply each review criterion across the full set of topics and sections.

- When referring to any part of the content in your evaluation or output , always reference the corresponding unique section identifier (e.g., t_1_s_2) to precisely indicate the relevant subtopic.

B) Understanding the Structure of the Research-Based Content:

- The research-based content is organized using a structured tagging format to represent instructional slides. Each topic includes its subtopics, learning objectives, and the content itself, formatted as follows:

a) <transition_slide>: Introduces the subtopic with a brief overview. Includes:
"Slide Title" and "Slide Content" field.
b) <content_slides>: Contains one or more <slide> blocks with detailed instructional material. Each <slide> includes:
"Slide Title" and "Slide Content" field.
c) <summary_slide>: Provides a summary of the key points covered in the subtopic. Includes:
"Slide Title" and "Slide Content" field.

- Each subtopic block (consisting of the three tags above) is separated from the next by a horizontal divider marked with "---"
- Each subtopic block begins with a unique identifier in the format `t_{topic_number}_s_{subtopic_number}`. This ID helps locate specific subtopic sections. Use this ID when referring to or giving feedback about specific sections (e.g., t_1_s_2).
- If the scope is "topic", you will receive one block wrapped in <topic> ... </topic>, representing a single topic with its structured research-based content.
- If the scope is "course", you will receive multiple topic blocks, each wrapped in <topic_1> ... </topic_1>, <topic_2> ... </topic_2>, etc. Each topic block follows the same structure described above.

C) Interpreting and Applying the Checklist Task and its Review Criteria:

- The checklist task may contain one or more review criteria.
- Evaluate each review criterion separately. Do not combine them into a single statement or summary judgment.
- Assess whether the research-based content satisfies each review criterion based on the provided scope.
- Always use the exact wording of the task name and its review criteria. Do not paraphrase or modify them in any way during analysis or output.

D) Assigning Verdicts and Giving Feedback:

- For each review criterion, assign a verdict of either "Pass" or "Fail" based on whether the research-based content meets the specified requirement.
- Regardless of the verdict, provide a justification that briefly explains why you reached that decision.
- If the verdict is "Fail", provide a feedback field that:
  a) Clearly identifies which part of the content did not meet the review criterion and briefly explains why. Be sure to include the unique identifier IDs (e.g., t_2_s_3) to precisely indicate where the issues occur.
  b) Describes how the issue can be addressed or improved in a way that aligns with the original instructional intent.
- Avoids suggesting major overhauls or unrelated additions — the improvement should stay within the scope and purpose of the original content.
- Feedback should be specific and aligned directly with the failed criterion. Avoid vague, overly general, or speculative suggestions.
- Ensure that your feedback is practicable and actionable — it should offer clear, specific guidance that can realistically be implemented to improve the content.

E) Evaluation Breakdown Section (Internal Reasoning and Analysis):

Use this section to evaluate the research-based content step by step before producing your final output. This is your internal workspace for interpreting the checklist criteria, analyzing the content, and documenting your reasoning.
Follow the steps below for each review criterion:

1) Understand the criterion: 
- Read the review criterion carefully and interpret what it is asking.

2) Apply the criterion to the appropriate content:
- Use the provided scope ("topic" or "course") to guide your evaluation:
  a) If the scope is "topic", the input will include only one topic with its subtopics, learning objectives, and structured content. Apply the review criterion to this topic's content alone.
  b) If the scope is "course", the input will include all topics in the course, each with its associated subtopics, learning objectives, and structured content. Evaluate the criterion across the entire course.

3) Analyze the content:
- Determine whether the content meets or fails the review criterion.

4) Document your reasoning:
- Clearly explain why the content satisfies or does not satisfy the criterion.
- If it fails, briefly describe what the issue is.

5) Prepare for feedback if applicable:
- If the content fails the review criterion, think about how the issue could be fixed in a way that aligns with the original instructional intent.
- Do not suggest unrelated content. Think in terms of practical, actionable, specific, and instructionally consistent improvements.

Provide your output strictly in the following format:

<output>

<evaluation_breakdown>

<analysis_of_criterion_1>

Checklist Task: (Enter the checklist task name exactly as given)  
Review Criterion 1: (Enter the review criterion 1 exactly as given)

1. Understanding the Criterion: (Interpret what the review criterion is evaluating)
2. Scope-Based Application: (Specify what content is being evaluated based on the scope and your understanding of what it means)
3. Analysis: (Evaluate whether the content meets the criterion, citing observations from the input)
4. Reasoning: (Briefly explain the rationale behind your decision)
5. Feedback Planning: If your verdict is “Fail" note how the issue could be addressed and what you plan to include in the "Feedback" field for improving the content.

</analysis_of_criterion_1>

<analysis_of_criterion_2>

Checklist Task: (Enter the checklist task name exactly as given)  
Review Criterion 2: (Enter the review criterion 2 exactly as given)

1. Understanding the Criterion: (Interpret what the review criterion is evaluating)
2. Scope-Based Application: (Specify what content is being evaluated based on the scope and your understanding of what it means)
3. Analysis: (Evaluate whether the content meets the criterion, citing observations from the input)
4. Reasoning: (Briefly explain the rationale behind your decision)
5. Feedback Planning: If your verdict is “Fail" note how the issue could be addressed and what you plan to include in the "Feedback" field for improving the content.

</analysis_of_criterion_2>

.........

(Repeat the above "analysis of criterion" block for each review criterion in the checklist task)

</evaluation_breakdown>

Based on your above evaluation, provide the final output for each review criterion in the following format:

<final_output>

<criterion_1>

Task: [Enter the task name exactly as it is]

Review Criterion 1: [Enter the review criterion 1 as it is]

Analysis: [Summarize your evaluation of the content with respect to this review criterion and the specified scope] 

Verdict: [Pass/Fail]

Justification: [Give a brief justification for your verdict]

Feedback: [Only include if verdict is “Fail”; clearly mention the unique identifier IDs (e.g., t_1_s_4) and describe how to address the issue based on your planning]

</criterion_1>

<criterion_2>

Task: [Enter the task name exactly as it is]

Review Criterion 2: [Enter the review criterion 2 as it is]

Analysis: [Summarize your evaluation of the content with respect to this review criterion and the specified scope] 

Verdict: [Pass/Fail]

Justification: [Give a brief justification for your verdict]

Feedback: [Only include if verdict is “Fail”; clearly mention the unique identifier IDs (e.g., t_1_s_4) and describe how to address the issue based on your planning]

</criterion_2>

.........

(Repeat this criterion block for each criterion under the task)

</final_output>

</output>

Note: Strictly remember to always enclose your entire output inside the <output> .... </output> tags.
"""


def run_research_checklist_review(course_name, target_audience, scope, section_notes, checklist_criteria, llm="gemini_2_flash"):
    """Run the Research-Based Content Checklist Review Agent."""
    review_agent = Chain(llm=llm, tags=["output"])
    formatted_prompt = research_checklist_reviewer_prompt.format(
        course_name=course_name,
        target_audience=target_audience,
        scope=scope,
        section_notes=section_notes,
        checklist_criteria=checklist_criteria,
    )
    print("\n🔍 RESEARCH CHECKLIST REVIEW PROMPT BEING SENT TO LLM:\n")
    print(formatted_prompt)
    print("\n" + "=" * 100 + "\n")
    review_agent.add_message(
        role="user",
        content=research_checklist_reviewer_prompt.format(
            course_name=course_name,
            target_audience=target_audience,
            scope=scope,
            section_notes=section_notes,
            checklist_criteria=checklist_criteria,
        ),
    )
    response = review_agent.run()
    return response["output"]


research_checklist_reviser_prompt = """You are a revision agent responsible for improving structured research-based content for an E-learning course based on checklist-based review feedback. You will receive the course context, the original research notes (covering either a single topic or the full course, depending on the scope), the specific checklist task and review criterion that failed, and reviewer feedback describing what issues were found and what needs to be changed. Your job is to revise only the necessary parts of the research-based content based on the given feedback. All revisions must align with the course’s learning objectives and should be suitable for the target audience.

Below is the course information for which the research-based content was generated:

<course_information>
Course name: {course_name}
Target audience: {target_audience}
</course_information> 

Here is the scope of this revision:

<scope>
{scope}
</scope>

Here is the original research-based content, of which you must revise only the specific sections referenced in the feedback:

<original_research_based_content>
{original_section_notes}
</original_research_based_content>

Below is the checklist task, the failed review criteria, and the corresponding reviewer feedback. This feedback specifies what content failed, why, and how it should be improved. Each piece of feedback will also mention the unique subtopic ID(s) (e.g., t_2_s_3) that need revision:

<checklist_task>
{checklist_task}
</checklist_task>
Follow the below instructions carefully while doing the revisions:

A) Scope-Based Revision Guidelines:

- Each checklist review criterion comes with a defined scope that determines how much of the research-based content needs revision.

a) Topic scope: You will receive only one topic. Revise only the specific subtopic sections (identified by IDs like t_1_s_2) mentioned in the feedback for that topic.
b) Course scope: You will receive all topics and their structured research-based content. Revise only the specific subtopic sections (e.g., t_2_s_4, t_3_s_1) within the corresponding topics as identified in the feedback.

-  Do not make changes to any other sections of the input. Leave all unrelated content unchanged.

B) Understanding the Structure of the Research-Based Content:

- The research-based content is formatted using a consistent structure that organizes instructional material for each subtopic within a topic. You must understand this structure to correctly identify which parts need revision based on the feedback.
- Each topic consists of one or more subtopics, and each subtopic follows this structure:

a) <transition_slide>: Introduces the subtopic with a brief overview. Includes:
"Slide Title" and "Slide Content" field.
b) <content_slides>: Contains one or more <slide> blocks with detailed instructional material. Each <slide> includes:
"Slide Title" and "Slide Content" field.
c) <summary_slide>: Summarizes key points covered in the subtopic. Includes:
"Slide Title" and "Slide Content" field.

- Each complete subtopic block — consisting of these three tagged parts — is separated from the next by a horizontal divider marked with "---".
- Each subtopic section starts with a unique identifier in the format t_{topic_number}_s_{subtopic_number}. For example:
"t_1_s_2" refers to Topic 1, Subtopic 2.
- These identifiers are referenced in the feedback to specify exactly which subtopic(s) need revision. You must use them to locate the affected sections and apply changes only to those.
- Depending on the scope (topic or course), the input format will vary:

a) For topic-level input, you’ll receive one <topic>...</topic> block.
b) For course-level input, you’ll receive multiple topic blocks: <topic_1>...</topic_1>, <topic_2>...</topic_2>, etc.

- Use these wrappers to identify where each topic begins and ends, and apply the revision only to the subtopic sections explicitly flagged in the feedback (e.g., t_1_s_2, t_2_s_3).

C) Interpreting and Applying Reviewer Feedback:

- Each checklist task includes one or more review criteria that failed. For each failed criterion, you will be given:
a) The exact review criterion
b) The scope of evaluation (topic or course)
c) The feedback describing what needs to be changed
d) The identifier(s) of the subtopic sections that need revision (e.g., t_1_s_2, t_3_s_1)

- Your job is to use this feedback to locate the specified topic and its subtopic section(s) and revise their content as instructed. Do not attempt to reinterpret, rephrase, or generalize the feedback.
- If multiple criteria fail, revise all the affected sections mentioned. Keep track of each feedback block separately — each criterion may require a different type of revision.
- Do not make any edits beyond what is specified in the feedback. Only the sections that are flagged should be revised.
- If the feedback specifies a change within only certain slides of a subtopic section, revise only those slides. All other slides and structural elements (such as <transition_slide>, <summary_slide>, or other <slide> blocks) in that section must remain exactly as they were. Do not rewrite or alter content that was not flagged.
- Only include the revised subtopic sections in your final output. Each revised section should include all its slides, but only the flagged parts should be changed. Everything else must be preserved exactly as-is. Do not include any unmodified subtopic sections in the output.

D) Revision Guidelines

- All revisions must maintain instructional clarity and remain suitable for the target audience.
- Preserve the original instructional intent unless the feedback explicitly asks for a change in meaning.
- Use clear, concise, and neutral language appropriate for E-learning materials.
- Avoid adding unrelated information, extra examples, or off-topic elaboration unless the feedback directly asks for it.
- If a revision affects structure (e.g., merging or splitting slide content), ensure that the resulting section still fits the format of:
<transition_slide>, <content_slides>, <summary_slide>, with proper “Slide Title” and “Slide Content” fields inside each.
- If the feedback is unclear or vague, follow the most direct and literal interpretation. Do not infer new changes beyond what is written.
 
E) Output Format:

- Your final output must include only the revised subtopic sections that were explicitly mentioned in the feedback.
- You must return your output in a structured format, enclosed within <output>...</output> tags.
- First, include an <evaluation_breakdown> section that documents your interpretation and revision plan for each failed criterion.
- Then, include a <revised_research_notes> section that contains only the subtopic sections that were modified.
- Within each revised subtopic section, include all structural components (<transition_slide>, <content_slides>, <summary_slide>), but only revise the specific parts flagged in the feedback. Leave all unmentioned parts unchanged.
- Do not include any subtopic section that was not revised.
- Do not output the entire original content — only the revised sections.

F) Evaluation Breakdown Section (Internal Reasoning and Planning):

Before producing your final revised output, use this section to think through each failed review criterion and plan your changes. This is your internal reasoning space. For each failed criterion, follow these steps:

1) Understand the Criterion: Read the review criterion carefully. Identify what instructional standard it evaluates and what kind of issue it points to.

2) Interpret the Feedback: Read the feedback and determine what exactly needs to change. Identify the affected subtopic section(s) using the provided IDs (e.g., t_1_s_2) and understand whether the feedback refers to a slide title, slide content, or an entire structural block.

3) Plan the Revision: Based on the feedback and scope, decide how to revise the identified section. If the feedback references specific slides, plan to change only those slides. Leave the rest of the section unchanged.
 
Document this reasoning step for each failed criterion in the <evaluation_breakdown> section of your output.

Provide your output strictly in the following format:

<output>

<evaluation_breakdown>

<criterion_1>

Checklist Task: [Enter the checklist task name exactly as given]

Review Criterion 1: [Enter the review criterion 1 exactly as given]

Scope: [topic/course]

Affected Section(s): [List the IDs of the subtopic sections to revise, e.g., t_1_s_2, t_2_s_4]

Feedback Summary: [Copy the feedback exactly as provided for this criterion]

Revision Plan: [Briefly describe your understanding of the issue and how you plan to revise the affected section(s). If specific slides are mentioned, state which ones will be revised and how.]

</criterion_1>

<criterion_2>

Checklist Task: [Enter the checklist task name exactly as given]

Review Criterion 2: [Enter the review criterion 2 exactly as given]

Scope: [topic/course]

Affected Section(s): [List the IDs of the subtopic sections to revise, e.g., t_3_s_1]

Feedback Summary: [Copy the feedback exactly as provided for this criterion]

Revision Plan: [Briefly describe your understanding of the issue and how you plan to revise the affected section(s). If specific slides are mentioned, state which ones will be revised and how.]

</criterion_2>

......

(Repeat this block for each failed review criterion in the checklist task)

</evaluation_breakdown>

Based on your above evaluation, provide your output of the revised research-based content below. Include only the subtopic sections that were actually modified. Keep the structure and tags intact, and preserve all unchanged content within those sections.

<final_output>

<revised_section_1>

ID: (Enter the subtopic ID that is being revised, e.g., t_1_s_2)

<transition_slide>
Slide Title: (Enter the original or revised transition slide title — revise only if instructed)
Slide Content: (Enter the original or revised transition slide content — revise only if instructed)
</transition_slide>

<content_slides>
<slide>
Slide Title: (Enter the original or revised content slide title — revise only if instructed)
Slide Content: (Enter the original or revised content slide content — revise only if instructed)
</slide>
<!-- Include any additional slides that are part of this section and need to remain unchanged or are modified -->
</content_slides>

<summary_slide>
Slide Title: (Enter the original or revised summary slide title — revise only if instructed)
Slide Content: (Enter the original or revised summary slide content — revise only if instructed)
</summary_slide>

</revised_section_1>

<revised_section_2>

ID: (Enter the subtopic ID that is being revised, e.g., t_3_s_1)

<transition_slide>
Slide Title: (Enter the original or revised transition slide title — revise only if instructed)
Slide Content: (Enter the original or revised transition slide content — revise only if instructed)
</transition_slide>

<content_slides>
<slide>
Slide Title: (Enter the original or revised content slide title — revise only if instructed)
Slide Content: (Enter the original or revised content slide content — revise only if instructed)
</slide>
<!-- Include any additional slides that are part of this section and need to remain unchanged or are modified -->
</content_slides>

<summary_slide>
Slide Title: (Enter the original or revised summary slide title — revise only if instructed)
Slide Content: (Enter the original or revised summary slide content — revise only if instructed)
</summary_slide>
</revised_section_2>

......

(Repeat this block for each revised section)

</final_output>

</output>

Note: Strictly remember to always enclose your entire output inside the <output> .... </output> tags.
"""


def run_research_checklist_revise(course_name, target_audience, scope, original_section_notes, checklist_task, llm="gemini_2_flash"):
    """Run the Research-Based Content Revision Agent using checklist review feedback."""
    reviser_agent = Chain(llm=llm, tags=["output"])
    formatted_prompt = research_checklist_reviser_prompt.format(
        course_name=course_name,
        target_audience=target_audience,
        scope=scope,
        original_section_notes=original_section_notes,
        checklist_task=checklist_task,
    )
    print("\n🔧 RESEARCH CHECKLIST REVISER PROMPT BEING SENT TO LLM:\n")
    print(formatted_prompt)
    print("\n" + "=" * 100 + "\n")
    reviser_agent.add_message(
        role="user",
        content=research_checklist_reviser_prompt.format(
            course_name=course_name,
            target_audience=target_audience,
            scope=scope,
            original_section_notes=original_section_notes,
            checklist_task=checklist_task,
        ),
    )
    response = reviser_agent.run()
    return response["output"]


def _parse_review_output(review_output):
    pattern = re.compile(r"<criterion_\d+>(.*?)</criterion_\d+>", re.DOTALL)
    failed = []
    for block in pattern.findall(review_output):
        verdict_match = re.search(r"Verdict:\s*(.*)", block)
        verdict = verdict_match.group(1).strip() if verdict_match else "Pass"
        if verdict.lower() == "fail":
            task_name = re.search(r"Task:\s*(.*)", block).group(1).strip()
            crit_match = re.search(r"Review Criterion(?: \d+)?:\s*(.*)", block)
            review_crit = crit_match.group(1).strip() if crit_match else ""
            feedback_match = re.search(r"Feedback:\s*(.*)", block, re.DOTALL)
            feedback = feedback_match.group(1).strip() if feedback_match else ""
            failed.append({"task_name": task_name, "review_criteria": review_crit, "feedback": feedback})
    return failed


def _parse_reviser_output(reviser_output):
    id_to_content = {}
    pattern = re.compile(r"<revised_section_\d+>(.*?)</revised_section_\d+>", re.DOTALL)
    for block in pattern.findall(reviser_output):
        id_match = re.search(r"ID:\s*(t_\d+_s_\d+)", block)
        if not id_match:
            continue
        sec_id = id_match.group(1).strip()
        content = re.sub(r"ID:\s*t_\d+_s_\d+\n?", "", block).strip()
        id_to_content[sec_id] = content
    return id_to_content


def _build_topic_input(index, row, notes):
    sub_blocks = [b.strip() for b in notes.split("\n\n---\n\n")]
    sections = []
    for i, block in enumerate(sub_blocks, 1):
        sections.append(f"<t_{index}_s_{i}>\n{block}\n</t_{index}_s_{i}>")
    return sub_blocks, f"<topic>\n<topic_title>{row['Topic']}</topic_title>\n<subtopics>{row['Subtopic']}</subtopics>\n<learning_objectives>{row['Learning Objectives']}</learning_objectives>\n" + "\n\n---\n\n".join(sections) + "\n</topic>"


def _build_course_input(df):
    course_text = ""
    id_to_block = {}
    counts = {}
    for idx, row in df.iterrows():
        notes = row['checklist_revised_section_notes'] or row['section_notes']
        sub_blocks = [b.strip() for b in notes.split("\n\n---\n\n")]
        counts[idx] = len(sub_blocks)
        blocks = []
        for i, block in enumerate(sub_blocks, 1):
            sec_id = f"t_{idx+1}_s_{i}"
            id_to_block[sec_id] = block
            blocks.append(f"<{sec_id}>\n{block}\n</{sec_id}>")
        course_text += f"<topic_{idx+1}>\n<topic_title>{row['Topic']}</topic_title>\n<subtopics>{row['Subtopic']}</subtopics>\n<learning_objectives>{row['Learning Objectives']}</learning_objectives>\n" + "\n\n---\n\n".join(blocks) + f"\n</topic_{idx+1}>\n"
    return course_text, id_to_block, counts


@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Checklist Based Review and Revise Agents",
    "function_name": "run_research_review_revise_checklist",
    "user_id": st.session_state.get("role", "anonymous")
})
def run_research_review_revise_checklist(sheet, worksheet_name, llm="gemini_2_flash"):
    research_notes_sheet, research_notes_df = get_sheet_data_and_df(sheet, worksheet_name)
    _, course_info_df = get_sheet_data_and_df(sheet, "Course info")
    course_name = course_info_df['Course Name'][0]
    target_audience = course_info_df['Target Audience & Industry'][0]
    checklist_sheet_link = course_info_df['Checklist Link'][0]
    gc = st.session_state["gc"]
    checklist_sheet = gc.open_by_url(checklist_sheet_link)
    checklist_sheet, checklist_df = get_sheet_data_and_df(checklist_sheet, "Research Notes Checklist")

    if 'checklist_revised_section_notes' not in research_notes_df.columns:
        research_notes_df['checklist_revised_section_notes'] = research_notes_df['section_notes']

    unique_tasks = checklist_df['Task'].dropna().unique()
    progress = SmartProgressBar(total_tasks=len(unique_tasks), description="Percent complete", save_interval=1)

    for task in unique_tasks:
        task_df = checklist_df[checklist_df['Task'] == task]
        scope = task_df['Scope'].iloc[0].strip().lower()

        checklist_criteria = ""
        for _, r in task_df.iterrows():
            checklist_criteria += (
                f"Task: {r['Task']}\n"
                f"Review Criterion: {r['Review Criteria']}\n"
                f"Scope: {r['Scope']}\n\n"
            )

        if scope == 'topic':
            for idx, row in research_notes_df.iterrows():
                notes = research_notes_df.at[idx, 'checklist_revised_section_notes']
                blocks, formatted_input = _build_topic_input(idx+1, row, notes)
                review_output = run_research_checklist_review(course_name, target_audience, 'topic', formatted_input, checklist_criteria, llm)
                failed = _parse_review_output(review_output)
                if failed:
                    failed_block = ""
                    for i, f in enumerate(failed, 1):
                        failed_block += f"<criterion_{i}>\nTask: {f['task_name']}\nReview Criterion {i}: {f['review_criteria']}\nFeedback: {f['feedback']}\n</criterion_{i}>\n\n"
                    reviser_output = run_research_checklist_revise(course_name, target_audience, 'topic', formatted_input, failed_block, llm)
                    id_map = _parse_reviser_output(reviser_output)
                    new_blocks = []
                    for i, block in enumerate(blocks, 1):
                        sec_id = f"t_{idx+1}_s_{i}"
                        new_blocks.append(id_map.get(sec_id, block))
                    research_notes_df.at[idx, 'checklist_revised_section_notes'] = "\n\n---\n\n".join(new_blocks)
                else:
                    research_notes_df.at[idx, 'checklist_revised_section_notes'] = notes
        else:
            course_input, id_to_block, counts = _build_course_input(research_notes_df)
            review_output = run_research_checklist_review(course_name, target_audience, 'course', course_input, checklist_criteria, llm)
            failed = _parse_review_output(review_output)
            id_map = {}
            if failed:
                failed_block = ""
                for i, f in enumerate(failed, 1):
                    failed_block += f"<criterion_{i}>\nTask: {f['task_name']}\nReview Criterion {i}: {f['review_criteria']}\nFeedback: {f['feedback']}\n</criterion_{i}>\n\n"
                reviser_output = run_research_checklist_revise(course_name, target_audience, 'course', course_input, failed_block, llm)
                id_map = _parse_reviser_output(reviser_output)
            for idx, row in research_notes_df.iterrows():
                blocks = []
                for i in range(1, counts[idx]+1):
                    sec_id = f"t_{idx+1}_s_{i}"
                    blocks.append(id_map.get(sec_id, id_to_block[sec_id]))
                research_notes_df.at[idx, 'checklist_revised_section_notes'] = "\n\n---\n\n".join(blocks)

        save_to_sheet(research_notes_sheet, research_notes_df)
        progress.update()

    print("\n✅ Checklist Review & Revise Process Completed 🚀")
    save_to_sheet(research_notes_sheet, research_notes_df)


def delete_research_checklist_review_revise(sheet, worksheet_name="Research Notes"):
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    cols = ["checklist_revised_section_notes"]
    cols = [c for c in cols if c in df.columns]
    if cols:
        df = df.drop(columns=cols)
        clear_worksheet(ws)
        save_to_sheet(ws, df)
