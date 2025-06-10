from modules.chain import Chain
from services.sheets_service import get_sheet_data_and_df, format_worksheet, save_to_sheet, hide_columns_by_name
# from gspread_dataframe import set_with_dataframe
from tqdm import tqdm
import pandas as pd
import re
import gspread
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.smart_progress_bar import SmartProgressBar
import pandas as pd
from langsmith import traceable
import streamlit as st

outline_checklist_review_prompt = """ You are a checklist-based review agent. A course outline for an E-learning course was created. Your task is to review the topics, subtopics, and learning objectives from this generated course outline using the predefined review criteria provided.
For each checklist task and its review criterion, determine whether the outline content satisfies the requirement. If any issue is found, you will mark it as a failure and apply the predefined operation specified for that criterion — such as deleting, modifying, splitting, or merging content. Do not invent or suggest a different operation. Your goal is to ensure that the course outline remains clear, relevant, and instructionally sound.

Below is the course information for which the outline was generated:

<course_information>
Course name: {course_name}
Target audience: {target_audience}
</course_information>

Here is the course outline to review:

<course_outline>
{outline_entry}
</course_outline>

Here is the checklist task and its corresponding review criteria you need to use for evaluation:

<checklist_criteria>
{checklist_criteria}
</checklist_criteria>

A) Scope-based checklist application guidelines

Each review criterion has its own scope which indicates whether you will be reviewing topics, subtopics, the entire outline or learning objectives. Here is how to apply the checklist task and its review criterion based on the scope of evaluation:

- Outline: Apply the checklist across the entire course outline. Review all the topics, subtopics, and learning objectives together as needed.
- Topic: Review all topics in the outline together. For each topic, examine all blocks where it appears to understand its full context and usage. Apply the checklist criteria by comparing topics against each other and their associated subtopics and learning objectives.
- Subtopic: Review all subtopics in the outline together. For each subtopic, examine all blocks where it appears to understand its full context and usage. Apply the checklist criteria by comparing subtopics against each other and their parent topics and learning objectives.
- Learning Objective: Review all learning objectives in the outline together. For each learning objective, examine its block to understand its context. Apply the checklist criteria by comparing learning objectives against each other and their associated topics and subtopics.

B) Multiple review criteria under a single checklist task

- Each checklist task may contain one or more review criteria.
- Evaluate the outline content against each criterion individually.
- Do not treat the group of criteria as a single combined statement—assess each one on its own.
- Strictly preserve the wording of the task name and each review criterion. Do not rephrase, paraphrase, or alter them in any way.

C) How to assign verdicts and recommend operations

- Each checklist task contains multiple individual review criteria.
- Evaluate the course outline content against each criterion separately.
- For every review criterion, assign a verdict of either "Pass" or "Fail" and provide a brief justification for your verdict.

If the criterion fails, use the predefined operation already provided for that review criterion. Do not invent or suggest a new operation. The possible predefined operations are:

a) delete → for removing unnecessary, out-of-scope, or redundant content

b) modify → for improving clarity, completeness, or phrasing

c) split → for breaking complex or multi-part content into simpler entries

d) merge → for combining overly brief or overlapping entries

Only if the verdict is Fail, include a short feedback statement that clearly explains what the issue is and how the predefined operation should be applied. Do not alter the operation — simply explain what content needs to be revised and how.

So your output for each review criterion should strictly contain the following fields:
Task, Review Criterion, Analysis, Verdict, Justification, Operation (if failed), Feedback (if failed)

D) Evaluation Breakdown

Use this section to reason step by step before producing the final output. You must evaluate the course outline entry against each review criterion individually, using the provided scope to determine how to apply the checklist. First, interpret what the scope means for this task. Then, go through each criterion and assess whether the content satisfies it.

For each review criterion:

- Analyze whether the outline content meets or fails the criterion.
- Explain your reasoning clearly in natural language.
- If the criterion is failed, explain clearly why it fails, and how applying the provided predefined operation will address the issue effectively.
- Refer to the course name, target audience, and context as needed to make better judgments.
- This is not the final output - treat this as your internal analysis based on which you will generate the final output and verdicts for each of the review criteria.

Provide your output strictly in the following format:

<output>

<evaluation_breakdown>

Task name: [Enter the task name as it is]

Scope: [Briefly describe how you interpret the given scope and how it affects your evaluation]

Review Criterion 1: [Enter the review criterion 1 as it is]

Analysis: [Explain your reasoning for this criterion — whether the outline passes or fails, and why. Also explain how applying the provided predefined operation will address the issue effectively]

Review Criterion 2: [Enter the review criterion 2 as it is]

Analysis: [Explain your reasoning for this criterion — whether the outline passes or fails, and why. Also explain how applying the provided predefined operation will address the issue effectively]

......

(Repeat this "Review Criterion" + "Analysis" block for each criterion under the task)

</evaluation_breakdown>

Based on your above evaluation, provide the final output for each review criterion in the following format:

<final_output>

<criterion_1>

Task: [Enter the task name exactly as it is]

Review Criterion 1: [Enter the review criterion 1 as it is]

Analysis: [Summarize your evaluation of the outline content with respect to this review criterion and the specified scope]  

Verdict: [Pass / Fail] 

Justification: [Briefly explain why you reached this verdict]  

Operation: [This field should be included only if the verdict is "Fail". Copy only the given operation for this criterion as it is]

Feedback: [This field should appear only if the verdict is "Fail". Briefly explain what the issue is and how to apply the given operation to resolve it]

</criterion_1>

<criterion_2>

Task: [Enter the task name exactly as it is]

Review Criterion 2: [Enter the review criterion 2 as it is]

Analysis: [Summarize your evaluation of the outline content with respect to this review criterion and the specified scope]  

Verdict: [Pass / Fail] 

Justification: [Briefly explain why you reached this verdict]  

Operation: [This field should be included only if the verdict is "Fail". Copy only the given operation for this criterion as it is]

Feedback: [This field should appear only if the verdict is "Fail". Briefly explain what the issue is and how to apply the given operation to resolve it]

</criterion_2>

......

(Repeat this criterion block for each criterion under the task)

</final_output>

</output>
"""

@traceable(
    metadata={
        "agent_name": "course_outline",
        "step_name": "Checklist Based Review and Revise Agents",
        "function_name": "run_outline_checklist_review",
        "user_id": st.session_state.get("role", "anonymous")
    }
)
def run_outline_checklist_review(course_name, target_audience, outline_entry, checklist_criteria, llm="gemini_2_flash"):
    """
    Run the Outline Checklist Review Agent on the course outline.

    :param course_name: Name of the course.
    :param target_audience: Intended audience for the course.
    :param outline_entry: The outline entry to evaluate (formatted block of topic/subtopic/LO lines).
    :param checklist_criteria: The checklist task and its associated criteria.
    :param llm: The language model to use.
    :return: Structured checklist review output from the agent.
    """

    # Initialize the Review Agent
    review_agent = Chain(llm=llm, tags=["output"])

    # Format the prompt for debugging (printing)
    formatted_prompt = outline_checklist_review_prompt.format(
        course_name=course_name,
        target_audience=target_audience,
        outline_entry=outline_entry,
        checklist_criteria=checklist_criteria
    )

    # Print the formatted prompt for debugging
    print("\n🔍 OUTLINE CHECKLIST REVIEW PROMPT BEING SENT TO LLM:\n")
    print(formatted_prompt)
    print("\n" + "=" * 100 + "\n")

    # Add the user message
    review_agent.add_message(
        role="user",
        content=outline_checklist_review_prompt.format(
            course_name=course_name,
            target_audience=target_audience,
            outline_entry=outline_entry,
            checklist_criteria=checklist_criteria
        )
    )

    # Run the agent
    response = review_agent.run()

    # Return output from the LLM
    return response["output"]


outline_checklist_reviser_prompt = """ A course outline for an E-learning course was created. You are a revision agent responsible for improving course outline entries based on checklist-based review feedback. You will receive the course context, the original outline entry, the specific checklist task and review criterion that failed, the recommended operation (e.g., delete, modify, split, or merge), and reviewer feedback describing the issue and what exactly should be done to fix it. Your job is to generate a revised version of the outline using the suggested operation and feedback, while preserving the instructional clarity, alignment, and intent of the course. All revisions should be appropriate for the target audience and consistent with the rest of the outline structure.

Below is the course information for which the outline was generated:

<course_information>
Course name: {course_name}
Target audience: {target_audience}
</course_information>

Below is the full course outline. The issue you need to revise is contained within this outline. Each entry includes a Topic, Subtopic, and Learning Objective. Use the feedback to identify and revise only the relevant parts.

<original_outline>  
{original_outline}  
</original_outline>

Here is the checklist task and the review criteria that were not satisfied. This includes the specific standard(s) the content failed to meet, along with the recommended operation and feedback on how to fix the issue:

<checklist_task>
Task: {task_name}
Review Criteria:
{review_criteria}
</checklist_task>

Follow the below instructions carefully while doing the revision:

A) Scope-Based Revision Guidelines

Each review criterion has its own scope which indicates whether you will be revising topics, subtopics, the entire outline or learning objectives. Here is how to apply the revision based on the scope of evaluation:

- Outline: Revise one or more entries from the course outline as needed. These may include multiple Topic, Subtopic, and Learning Objective line(s). Apply revisions wherever necessary based on the feedback.
- Topic: Review all topics in the outline together and revise as needed. When revising a topic, examine all blocks where it appears to understand its full context and usage. Make revisions by comparing topics against each other and their associated subtopics and learning objectives.
- Subtopic: Review all subtopics in the outline together and revise as needed. When revising a subtopic, examine all blocks where it appears to understand its full context and usage. Make revisions by comparing subtopics against each other and their parent topics and learning objectives.
- Learning Objective: Review all learning objectives in the outline together and revise as needed. When revising a learning objective, examine its block to understand its context. Make revisions by comparing learning objectives against each other and their associated topics and subtopics.

B) Operation Guidelines

Each checklist task includes a recommended operation based on the type of issue found. Apply the operation exactly as specified in the reviewer feedback:

- Delete: Remove the identified entry or part of an entry.
- Modify: Rewrite the existing line(s) as instructed.
- Split: Break a single entry into multiple clearer parts.
- Merge: Combine related entries into a single, unified entry.

Do not make additional edits beyond what is described in the feedback.

C) Use of Reviewer Feedback

Use the reviewer feedback exactly as provided. It describes:

- What the issue is
- What needs to be changed
- What operation to apply

Do not reinterpret, generalize, or invent additional issues. Revise only the part of the outline that is mentioned in the feedback, and follow the instructions closely. Do not alter or rewrite any other entries in the outline. Leave all unrelated content - anything not referenced in the feedback - strictly unchanged.

D) Revision Requirements

All revisions in the outline must:

- Maintain instructional clarity - the content should be relevant to the course and suitable for the target audience.
- Preserve the intent of the original entry unless feedback explicitly calls for a change in meaning.
- Avoid introducing unrelated ideas or terminology not present in the original outline or the feedback.
- Use concise, neutral, and instructional language — suitable for E-learning contexts.

The goal is to improve the entry based on the feedback while ensuring consistency with the rest of the outline.

Strictly give your output in the following format:

<output>

<evaluation_breakdown>

<criterion_1>

Task: [Enter the task name exactly as it is]

Review Criterion 1: [Enter the review criterion 1 as it is]

Feedback: [Copy the given reviewer feedback for this criterion exactly]

Scope: [Copy the given scope for this criterion]

Operation: [Copy the given operation for this criterion - delete / modify / split / merge]

Revision Plan: [Explain your reasoning and the approach you will take to revise the outline based on the given feedback]

</criterion_1>

<criterion_2>

Task: [Enter the task name exactly as it is]

Review Criterion 2: [Enter the review criterion 2 as it is]

Feedback: [Copy the given reviewer feedback for this criterion exactly]

Scope: [Copy the given scope for this criterion]

Operation: [Copy the given operation for this criterion - delete / modify / split / merge]

Revision Plan: [Explain your reasoning and the approach you will take to revise the outline based on the given feedback]

</criterion_2>

......

(Repeat this criterion block for each review criterion under the task)

</evaluation_breakdown>

<final_outline>

[Insert the complete revised course outline below. Keep all original entries exactly as they are, except for the ones that require changes based on the reviewer feedback. Apply revisions only where instructed. Each block must follow this structure, with exactly one Learning Objective per block:

<block_1>

Topic: ...
Subtopic: ...
Learning Objective: ...

</block_1>

<block_2>

Topic: ...
Subtopic: ...
Learning Objective: ...

</block_2>

......

(Repeat this block format for each Learning Objective in the full revised outline)

</final_outline>

</output>
"""

@traceable(
    metadata={
        "agent_name": "course_outline",
        "step_name": "Checklist Based Review and Revise Agents",
        "function_name": "run_outline_checklist_reviser",
        "user_id": st.session_state.get("role", "anonymous")
    }
)
def run_outline_checklist_reviser(course_name, target_audience, original_outline, task_name, review_criteria, llm="gemini_2_flash"):
    """
    Run the Outline Checklist Reviser Agent on the course outline using reviewer feedback.

    :param course_name: Name of the course.
    :param target_audience: Intended audience for the course.
    :param scope: Scope of the revision (e.g., Outline, Topic, Subtopic, Learning Objective).
    :param original_outline: The full original course outline.
    :param task_name: The checklist task name under which one or more criteria failed.
    :param review_criteria: Combined failed review criteria text (including operation and feedback).
    :param llm: The language model to use.
    :return: Structured revised course outline from the agent.
    """

    # Initialize the Reviser Agent
    reviser_agent = Chain(llm=llm, tags=["output"])

    # Format the prompt for debugging (printing)
    formatted_prompt = outline_checklist_reviser_prompt.format(
        course_name=course_name,
        target_audience=target_audience,
        original_outline=original_outline,
        task_name=task_name,
        review_criteria=review_criteria
    )

    # Print the formatted prompt for debugging
    print("\n🛠️ OUTLINE CHECKLIST REVISION PROMPT BEING SENT TO LLM:\n")
    print(formatted_prompt)
    print("\n" + "=" * 100 + "\n")

    # Add the user message
    reviser_agent.add_message(
        role="user",
        content=outline_checklist_reviser_prompt.format(
            course_name=course_name,
            target_audience=target_audience,
            original_outline=original_outline,
            task_name=task_name,
            review_criteria=review_criteria
        )
    )

    # Run the agent
    response = reviser_agent.run()

    # Return output from the LLM
    return response["output"]

@traceable(
    metadata={
        "agent_name": "course_outline",
        "step_name": "Checklist Based Review and Revise Agents",
        "function_name": "run_outline_checklist_review_and_revise",
        "user_id": st.session_state.get("role", "anonymous")
    }
)
def run_outline_checklist_review_and_revise(sheet, worksheet_name, llm="gemini_2_flash"):
    """
    Run a review-and-revise loop on a course outline using checklist tasks.
    For each unique checklist task, this function:
      - Reviews the outline based on scope and checklist criteria
      - Revises it based on failed verdicts
      - Saves updated revised values back to the sheet after each task

    :param sheet: Main Google Sheet object
    :param worksheet_name: Name of the worksheet containing the course outline
    :param llm: Language model to use for both agents
    """
    outline_sheet, outline_df = get_sheet_data_and_df(sheet, worksheet_name)
    course_info_sheet, course_info_df = get_sheet_data_and_df(sheet, 'Course info')

    checklist_sheet_link = course_info_df['Checklist Link'][0]
    gc = st.session_state["gc"]
    checklist_sheet = gc.open_by_url(checklist_sheet_link)
    checklist_sheet, checklist_df = get_sheet_data_and_df(checklist_sheet, 'Course Outline Checklist')

    course_name = course_info_df['Course Name'][0]
    target_audience = course_info_df['Target Audience & Industry'][0]

    for col in ['Revised Topic', 'Revised Subtopic', 'Revised Learning Objectives']:
        if col not in outline_df.columns:
            outline_df[col] = ""

    unique_tasks = checklist_df['Task'].unique()

    # Initialize the progress tracker
    progress = SmartProgressBar(total_tasks=len(unique_tasks), description="Percent complete", save_interval=1)

    for task in unique_tasks:
        task_criteria_df = checklist_df[checklist_df['Task'] == task]

        checklist_criteria = ""
        for _, row in task_criteria_df.iterrows():
            checklist_criteria += (
                f"Task: {row['Task']}\n"
                f"Review Criterion: {row['Review Criteria']}\n"
                f"Scope: {row['Scope']}\n"
                f"Operation: {row['Operation']}\n\n"
            )

        outline_entry = ""
        for i, row in outline_df.iterrows():
            topic = row['Revised Topic'] if row['Revised Topic'] else row['Topic']
            subtopic = row['Revised Subtopic'] if row['Revised Subtopic'] else row['Subtopic']
            lo = row['Revised Learning Objectives'] if row['Revised Learning Objectives'] else row['Learning Objectives']
            outline_entry += f"Topic: {topic}\nSubtopic: {subtopic}\nLearning Objective: {lo}\n\n"

        review_output = run_outline_checklist_review(
            course_name=course_name,
            target_audience=target_audience,
            outline_entry=outline_entry,
            checklist_criteria=checklist_criteria,
            llm=llm
        )

        failed_criteria_blocks = re.findall(r"<criterion_\d+>(.*?)</criterion_\d+>", review_output, re.DOTALL)
        failed_criteria_info = []

        for block in failed_criteria_blocks:
            if re.search(r"Verdict:\s*Fail", block):
                task_name = re.search(r"Task:\s*(.*)", block).group(1).strip()
                review_criterion_match = re.search(r"(?:Review )?Criterion(?:\s+\d+)?:\s*(.*)", block)
                if not review_criterion_match:
                    print(f"⚠️ Could not find Review Criterion in block: {block}")
                    continue
                review_criterion = review_criterion_match.group(1).strip()
                feedback = re.search(r"Feedback:\s*(.*)", block, re.DOTALL).group(1).strip()
                operation = re.search(r"Operation:\s*(.*)", block).group(1).strip()
                failed_criteria_info.append({
                    "task_name": task_name,
                    "review_criteria": review_criterion,
                    "operation": operation,
                    "feedback": feedback
                })

        if not failed_criteria_info:
            print(f"✅ All criteria passed for task: {task}")
            continue

        # Store the outline_entry that was used for review before clearing revised columns
        reviser_input = outline_entry

        # Clear revised columns before processing reviser output since we have failed criteria
        for i in range(len(outline_df)):
            outline_df.at[i, 'Revised Topic'] = ""
            outline_df.at[i, 'Revised Subtopic'] = ""
            outline_df.at[i, 'Revised Learning Objectives'] = ""

        formatted_failed_criteria = ""
        for i, crit in enumerate(failed_criteria_info, 1):
            matching_row = task_criteria_df[task_criteria_df['Review Criteria'] == crit['review_criteria']].iloc[0]
            scope = matching_row['Scope']
            operation = matching_row['Operation']

            formatted_failed_criteria += f"<criterion_{i}>\n"
            formatted_failed_criteria += f"Task: {crit['task_name']}\n"
            formatted_failed_criteria += f"Review Criterion {i}: {crit['review_criteria']}\n"
            formatted_failed_criteria += f"Feedback: {crit['feedback']}\n"
            formatted_failed_criteria += f"Scope: {scope}\n"
            formatted_failed_criteria += f"Operation: {operation}\n"
            formatted_failed_criteria += f"</criterion_{i}>\n\n"

        reviser_output = run_outline_checklist_reviser(
            course_name=course_name,
            target_audience=target_audience,
            original_outline=reviser_input,  # Use the stored input that was used for review
            task_name=task,
            review_criteria=formatted_failed_criteria,
            llm=llm
        )

        final_outline_match = re.search(r"<final_outline>(.*?)</final_outline>", reviser_output, re.DOTALL)
        if not final_outline_match:
            print(f"⚠️ No final outline found for task: {task}")
            continue

        final_outline_text = final_outline_match.group(1)
        revised_blocks = re.findall(r"<block_\d+>(.*?)</block_\d+>", final_outline_text, re.DOTALL)

        revised_entries = []
        for block in revised_blocks:
            topic = re.search(r"Topic:\s*(.*)", block).group(1).strip()
            subtopic = re.search(r"Subtopic:\s*(.*)", block).group(1).strip()
            lo = re.search(r"Learning Objective:\s*(.*)", block).group(1).strip()
            revised_entries.append((topic, subtopic, lo))

        for i, (topic, subtopic, lo) in enumerate(revised_entries):
            outline_df.at[i, 'Revised Topic'] = topic
            outline_df.at[i, 'Revised Subtopic'] = subtopic
            outline_df.at[i, 'Revised Learning Objectives'] = lo

        save_to_sheet(outline_sheet, outline_df)
        print(f"✅ Revisions applied and saved for task: {task}")
        
        # Update progress after each task
        progress.update()