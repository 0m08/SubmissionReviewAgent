from services.sheets_service import get_sheet_data_and_df, save_to_sheet, clear_worksheet
from modules.chain import Chain
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
from services.smart_progress_bar import SmartProgressBar
from langsmith import traceable
from tqdm import tqdm
import re

subtopic_context_aware_review_prompt = """We are creating structured research-based content for an E-learning course. You are a Review Agent responsible for evaluating the quality of the given research note based on a defined set of evaluation criteria.

Below is the course information for which the research notes were generated:

<course_information>
Course name: {course_name}
Target audience: {target_audience}
</course_information>

Below is the topic and subtopic to which the research notes belongs:

<topic>
{topic}
</topic>

<subtopic>
{subtopic}
</subtopic>

Below is the learning objective that the given research note is intended to support:

<learning_objective>
{learning_objective}
</learning_objective>

Here are all the research notes generated for this subtopic:

<all_research_notes>
{all_research_notes}
</all_research_notes>

Here is the specific research note that you need to evaluate:

<research_note>
{research_note}
</research_note>

Follow the instructions below to carefully evaluate the given research note against the specified quality criteria:

1. Understand the Structure and Format of the Research Note:
Before applying any evaluation criteria, begin by identifying the structural format of the given research note. There are two main formats you may encounter:

  a. Document-Derived Format:
    - The content is written in a structured, outline-style format with clear section headers and numbered points.
    - It often includes labeled main ideas and sub-points, with supporting information such as definitions, explanations, or direct quotations from source documents.
    - Quotes may include document references in parentheses, e.g., (Doc 5).

  b. Transcript-Derived Format:
    - The content appears as a sequence of transcript lines, each prefixed with a timestamp in seconds (e.g., '266', '268').
    - It reflects spoken language and may contain natural pauses, hesitations, or informal phrasing.
    - The research note in this case is directly extracted from a video transcript and has not been paraphrased or rewritten.

2. Treat Document-Based and Transcript-Based Notes Equally:
  - The research note may be based on documents or timestamped transcripts. Regardless of the source, your review approach should remain the same.
  - Regardless of the source, follow the same review process for evaluating the research note.
  - If the research note appears to be transcript-based, review it the same way you would a document-based note — without bias toward its format.

3. Focus on One Research Note at a Time:
  - Only evaluate the single given research note in this task — do not evaluate the other research notes listed under <all_research_notes>.
  - However, use the other notes as context to avoid redundancy and ensure the given note contributes unique and necessary value within the subtopic.

Use only the following review criteria for carrying out your evaluation:

1. Alignment with the Learning Objective:
  - The content clearly and directly supports the specific learning objective.
  - The information is not too broad, vague, or unrelated to the objective’s intent.

2. Instructional Relevance and Usefulness:
  - The content adds meaningful instructional value for the learner — it introduces, explains, or elaborates on ideas that are useful for mastering the learning objective.
  - It avoids generic filler, tangents, or excessive repetition of ideas already covered in other research notes in <all_research_notes>.

3. Structural Coherence and Focus:
  - The note presents a focused, coherent idea or set of related points rather than a scattered or disjointed collection of statements.
  - It maintains internal consistency and has a logical flow — even if brief.

4. Appropriate Level of Detail:
  - The note contains a balanced amount of detail — enough to be educational, but not overloaded with unnecessary specifics or trivial information.
  - It avoids being too shallow (e.g., only stating obvious facts) or too dense with overly technical details (unless suitable for the audience).

5. Redundancy and Duplication Across Research Notes:
  - The research note does not unnecessarily repeat content already covered in other research notes under <all_research_notes>.
  - Some minor overlap (1 or 2 sentences) is acceptable if it is instructionally necessary - for example, when similar subtopics or learning objectives require brief reiteration of key concepts for clarity or completeness.
  - However, the research note should not contain substantial duplication, where the majority of its content simply restates what has already been established in earlier notes.
  - Each research note should make a unique contribution to the learner’s understanding of the subtopic.

Output Requirements:

For the given research note, you must evaluate it individually against each of the above specified review criteria. For each criterion, provide the following fields:
  - Review Criterion: Restate the full name of the review criterion (as listed).
  - Verdict: Either Pass or Fail, based strictly on whether the research note satisfies the criterion.
  - Justification: A clear and concise explanation of why you gave this verdict. If the verdict is Pass, briefly state what the note does well. If the verdict is Fail, explain exactly what the issue is and how the note falls short.
  - Feedback: Only required if the verdict is Fail. Provide actionable, constructive feedback that explains what needs to change or improve in the research note to meet this criterion. Be specific and helpful.

Give your output strictly in the following format:

<output>

<evaluation_breakdown>

Review Criterion 1: [Enter the review criterion 1 as it is]

Analysis: [Explain your analysis and reasoning for this criterion — whether the research note passes or fails, and why]

Review Criterion 2: [Enter the review criterion 2 as it is]

Analysis: [Explain your analysis and reasoning for this criterion — whether the research note passes or fails, and why]

......

(Repeat this "Review Criterion" + "Analysis" block for each of the review criterion)

</evaluation_breakdown>

<final_output>

<criterion_1>
Review Criterion: [Enter the review criterion 1 as it is]  
Verdict: [Pass/Fail]  
Justification: Justify why the research note passes or fails this criterion.  
Feedback: [Only include if Verdict is Fail. Give clear, specific suggestions for improvement]
</criterion_1>

<criterion_2>
Review Criterion: [Enter the review criterion 2 as it is]  
Verdict: [Pass/Fail]  
Justification: Justify why the research note passes or fails this criterion.  
Feedback: [Only include if Verdict is Fail. Give clear, specific suggestions for improvement]
</criterion_2>

.....

(Repeat this criterion block for each of the review criterion) 

</final_output>

</output>

Note: Strictly remember to always enclose your entire output inside the <output> .... </output> tags.
"""



@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Subtopic Context-Aware Review and Revise",
    "function_name": "run_subtopic_context_aware_review",
    "user_id": st.session_state.get("role", "anonymous")
})
def run_subtopic_context_aware_review(course_name, target_audience, topic, subtopic, learning_objective, all_research_notes, research_note, llm="gemini_2_flash"):
    """
    Run the Subtopic Context-Aware Review Agent on the given research note.

    :param course_name: Name of the course.
    :param target_audience: Intended audience for the course.
    :param topic: Topic to which the research note belongs.
    :param subtopic: Subtopic under which this research note is grouped.
    :param learning_objective: The specific learning objective the research note is intended to support.
    :param all_research_notes: All other research notes generated for the same subtopic.
    :param research_note: The specific research note to review.
    :param llm: The language model to use.
    :return: Structured checklist review output from the agent.
    """


    # Prompt template should be defined elsewhere
    review_agent = Chain(llm = llm, tags = ["final_output"])

    # Format the prompt for debugging (printing)
    formatted_prompt = subtopic_context_aware_review_prompt.format(
            course_name = course_name,
            target_audience = target_audience,
            topic = topic,
            subtopic = subtopic,
            learning_objective = learning_objective,
            all_research_notes = all_research_notes,
            research_note = research_note
        )

    # Print the formatted prompt for debugging
    print("\n🔍 REVIEW PROMPT BEING SENT TO LLM:\n")
    print(formatted_prompt)
    print("\n" + "=" * 100 + "\n")

    # Add the user message
    review_agent.add_message(
        role="user",
        content=subtopic_context_aware_review_prompt.format(
            course_name = course_name,
            target_audience = target_audience,
            topic = topic,
            subtopic = subtopic,
            learning_objective = learning_objective,
            all_research_notes = all_research_notes,
            research_note = research_note
        )
    )

    # Run the agent
    response = review_agent.run()

    return response["final_output"]


subtopic_context_aware_reviser_prompt = """We are creating structured research-based content for an E-learning course. You are a Revision Agent responsible for improving a given research note based on the feedback received. You will receive the course context, the topic and subtopic to which the research note belongs, the specific learning objective it is intended to support, all other research notes for that subtopic, and the review feedback indicating which quality criterion the research note failed and why. Your task is to revise only the given research note to address the reviewer’s feedback. Make precise, minimal edits that fully resolve the issue while preserving instructional clarity and consistency with the rest of the subtopic.

Below is the course information for which the research notes were generated:

<course_information>
Course name: {course_name}
Target audience: {target_audience}
</course_information>

Below is the topic and subtopic to which the research notes belongs:

<topic>
{topic}
</topic>

<subtopic>
{subtopic}
</subtopic>

Below is the learning objective that the given research note is intended to support:

<learning_objective>
{learning_objective}
</learning_objective>

Here are all the research notes generated for this subtopic:

<all_research_notes>
{all_research_notes}
</all_research_notes>

Here is the specific research note that you need to evaluate:

<research_note>
{research_note}
</research_note>

Here is the block(s) of failed review criteria and reviewer feedback that must be addressed through revision:

<failed_criteria_block>
{failed_criteria_block}
</failed_criteria_block>

Follow the instructions below to carefully revise the given research note against the specified failed review criteria:

1. Understand the Structure and Format of the Research Note

Before making any revisions, identify the structural format of the given <research_note>. You may encounter either of the following:

a. Document-Derived Format:

- The content is written in a structured, outline-style format with clear section headers and numbered points.
- It often includes labeled main ideas and sub-points, with supporting information such as definitions, explanations, or direct quotations from source documents.
- Quotes may include document references in parentheses, e.g., (Doc 5).

b. Transcript-Derived Format

- The content appears as a sequence of transcript lines, each prefixed with a timestamp in seconds (e.g., '266', '268').
- It reflects spoken language and may contain natural pauses, hesitations, or informal phrasing.
- The research note in this case is directly extracted from a video transcript and has not been paraphrased or rewritten.

2. Preserve the Original Format of the Research Note
- Your revised output must strictly match the format of the original <research_note>.
- If the original research note is in document-derived format, your revision must retain the same structured, outline-style format with clear sections, numbered points, and quotations (if present).
- If the research note is in transcript-derived format, your revision must remain in timestamped transcript style — preserve the timestamps and line structure as-is.
- Do not rewrite transcript-based notes into paragraph or outline format. Similarly, do not convert structured notes into transcripts.

3. Revise Only Based on Failed Review Criteria
- You will be given one or more failed review criteria along with reviewer justification and feedback. These are the only issues that require revision.
- Do not make any edits beyond what the failed criteria and feedback explicitly specify.
- Do not reinterpret the intent of the original content unless the feedback clearly calls for it.

4. Preserve Instructional Value and Tone
- Maintain the educational tone and instructional intent of the original research note.
- Ensure that any changes continue to support the given learning objective in a clear and meaningful way.
- Use language that is concise, neutral, and appropriate for the target audience.
- Do not inject personal opinions, exaggerated claims, or off-topic elaboration.

Give your output strictly in the following format:

<evaluation_breakdown>

<criterion_1>

Review Criterion 1: [Enter the failed review criterion 1 as it is]

Feedback: [Copy the given reviewer feedback for this criterion exactly as it is]

Revision Plan: [Explain your reasoning and the approach you will take to revise the research note based on the given feedback]

</criterion_1>

<criterion_2>

Review Criterion 2: [Enter the failed review criterion 2 as it is]

Feedback: [Copy the given reviewer feedback for this criterion exactly as it is]

Revision Plan: [Explain your reasoning and the approach you will take to revise the research note based on the given feedback]

</criterion_2>

......

(Repeat this criterion block for each of the given failed review criterion)

</evaluation_breakdown>

<revised_research_note>
Return the entire research note with the necessary revisions applied based on the failed criterion. Do not return only the edited portions — include the full research note in its original format (document-style or timestamped), with only the required edits made.
</revised_research_note>

</output>

Note: Strictly remember to always enclose your entire output inside the <output> .... </output> tags.
"""


@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Subtopic Context-Aware Review and Revise",
    "function_name": "run_subtopic_context_aware_reviser",
    "user_id": st.session_state.get("role", "anonymous")
})
def run_subtopic_context_aware_reviser( course_name, target_audience, topic, subtopic, learning_objective, all_research_notes, research_note, failed_criteria_block, llm="gemini_2_flash"):
    """
    Run the Subtopic-Level Context-Aware Reviser agent on the given research note.

    :param course_name: Name of the course.
    :param target_audience: Intended audience for the course.
    :param topic: Topic to which the research note belongs.
    :param subtopic: Subtopic to which the research note belongs.
    :param learning_objective: The learning objective supported by this research note.
    :param all_research_notes: All research notes under this subtopic.
    :param research_note: The specific research note to revise.
    :param failed_criteria_block: The reviewer's feedback for failed criteria.
    :param llm: The language model to use.
    :return: Structured revised research note output from the agent.
    """

    # Initialize the Revision Agent
    reviser_agent = Chain(llm = llm, tags = ["revised_research_note"])

    # Add the user message
    reviser_agent.add_message(
        role="user",
        content=subtopic_context_aware_reviser_prompt.format(
            course_name = course_name,
            target_audience = target_audience,
            topic = topic,
            subtopic = subtopic,
            learning_objective = learning_objective,
            all_research_notes = all_research_notes,
            research_note = research_note,
            failed_criteria_block = failed_criteria_block
        )
    )

    # Run the agent
    response = reviser_agent.run()

    # Return output from the LLM
    return response["revised_research_note"]


@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Subtopic Context-Aware Review and Revise",
    "function_name": "run_subtopic_context_aware_review_and_revise_for_all_rows",
    "user_id": st.session_state.get("role", "anonymous")
})
def run_subtopic_context_aware_review_and_revise_for_all_rows(sheet, worksheet_name, llm="gemini_2_flash", max_workers=5):
    """
    Runs the subtopic-context-aware review and revise workflow for all rows in the Final Outline sheet.es the sheet in-place. Uses parallelization and a Streamlit progress bar.
    
    :param sheet: The main Google Sheet object.
    :param worksheet_name: The name of the worksheet containing the research notes.
    :param llm: The language model to use for both review and revise agents (default is 'gemini_2_flash').
    :param max_workers: The maximum number of parallel threads to use (default is 5).
    :return: None       
    """

    # Fetch course info (single row)
    _, course_info_df = get_sheet_data_and_df(sheet, "Course info")
    course_name = course_info_df.loc[0, "Course Name"]
    target_audience = course_info_df.loc[0, "Target Audience & Industry"]

    # Load Final Outline sheet
    worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)

    # Ensure columns exist
    if "subtopic_context_aware_review" not in df.columns:
        df["subtopic_context_aware_review"] = ""
    if "revised_research_notes" not in df.columns:
        df["revised_research_notes"] = ""

    # Identify unique subtopics and their first topic
    subtopic_groups = df.groupby("Subtopic", sort=False)
    subtopic_to_topic = df.drop_duplicates("Subtopic").set_index("Subtopic")["Topic"].to_dict()

    # Prepare for parallel processing
    futures_map = {}
    total_tasks = len(df)
    save_interval = 5
    progress = SmartProgressBar(total_tasks=total_tasks, description="Percent complete", save_interval=save_interval)

    def format_all_research_notes(subtopic_rows):
        blocks = []
        for idx, (_, row) in enumerate(subtopic_rows.iterrows(), 1):
            lo = row["Learning Objectives"]
            note = row["research_notes"]
            blocks.append(f"<learning_objective_{idx}_research_note>\nLearning Objective: {lo}\n\nResearch Note:\n{note}\n</learning_objective_{idx}_research_note>")
        return "\n\n".join(blocks)

    def parse_failed_criteria_blocks(review_output):
        # Extract <criterion_n> blocks with Fail verdict
        pattern = r"(<criterion_\d+>.*?Verdict:\s*Fail.*?</criterion_\d+>)"
        return "\n\n".join(re.findall(pattern, review_output, re.DOTALL))

    def process_row(index, row, subtopic_rows):
        # Prepare inputs
        topic = subtopic_to_topic[row["Subtopic"]]
        subtopic = row["Subtopic"]
        learning_objective = row["Learning Objectives"]
        research_note = row["research_notes"]
        all_research_notes = format_all_research_notes(subtopic_rows)
        
        # --- REVIEW AGENT ---
        print(f"Running review agent")
        review_output = run_subtopic_context_aware_review(
            course_name, target_audience, topic, subtopic, learning_objective, all_research_notes, research_note, llm=llm
        )
        # Save review output
        failed_criteria_block = parse_failed_criteria_blocks(review_output)

        # --- REVISER AGENT ---
        if failed_criteria_block.strip():
            print(f"Running reviser agent")
            revised_note = run_subtopic_context_aware_reviser(
                course_name, target_audience, topic, subtopic, learning_objective, all_research_notes, research_note, failed_criteria_block, llm=llm
            )
        else:
            revised_note = research_note
        return {"review": review_output, "revised": revised_note}

    # Submit all tasks
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for subtopic, subtopic_rows in subtopic_groups:
            subtopic_rows = subtopic_rows.copy()
            for index, row in subtopic_rows.iterrows():
                future = executor.submit(process_row, index, row, subtopic_rows)
                futures_map[future] = index

        completed = 0
        for future in as_completed(futures_map):
            index = futures_map[future]
            result = future.result()
            df.at[index, "subtopic_context_aware_review"] = result["review"]
            df.at[index, "revised_research_notes"] = result["revised"]
            completed += 1
            progress.update()
            if progress.should_save():
                print(f"Saving partial progress to sheet after {completed} tasks completed.")
                save_to_sheet(worksheet, df)

    print("All rows processed. Saving final DataFrame to sheet.")
    save_to_sheet(worksheet, df)
    print("Sheet update complete.")


def delete_subtopic_context_aware_review_and_revise(sheet, worksheet_name="Final Outline"):
    """Remove subtopic context-aware review and revise columns from the worksheet."""
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    cols = [
        "subtopic_context_aware_review",
        "revised_research_notes",
    ]
    cols = [c for c in cols if c in df.columns]
    if cols:
        df = df.drop(columns=cols)
        clear_worksheet(ws)
        save_to_sheet(ws, df)

