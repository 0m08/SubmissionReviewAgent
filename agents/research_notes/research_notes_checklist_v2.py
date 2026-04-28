"""
Research Notes Checklist v2 - Parallel Review with Aggregation

Experimental version that changes the review-revise pipeline:

OLD FLOW (v1 - sequential per task):
    For each (task, scope) sequentially:
        review → revise → next task

NEW FLOW (v2 - parallel reviews + aggregated revision):
    1. ALL task reviews run in parallel (each with its own scope/criteria)
    2. An aggregator LLM combines all review results into unified feedback
    3. A single reviser pass addresses all aggregated feedback at once
    4. For re-review iterations, diffs are sent back to each task reviewer in parallel
"""
from modules.chain import Chain
from services.sheets_service import (
    get_sheet_data_and_df,
    save_to_sheet,
    clear_worksheet,
    delete_worksheet,
    format_worksheet,
    create_or_read_worksheet,
    hide_worksheet_by_name,
    hide_columns_by_name,
)
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.helper_functions import get_outline_with_los, validate_column_values
from services.smart_progress_bar import SmartProgressBar
from langsmith import traceable

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage, HumanMessage

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain.chat_models import init_chat_model

from pydantic import BaseModel, Field
from langgraph.prebuilt import InjectedState
from langchain.agents import AgentState
import pandas as pd
from services.crud_text_block_tools import create_block, read_blocks, update_block, delete_block, search_web, stop, str_replace, preview_image
from services.helper_functions import iterate_scope
from agents.research_notes.inline_image_placement import extract_inline_image_links
from typing import Any, Callable, Dict, Hashable, Iterable, List, Tuple
import difflib


# ============================================================
# Tool Mapping
# ============================================================

TOOL_NAME_TO_FUNCTION = {
    "Create": create_block,
    "Read": read_blocks,
    "Update": update_block,
    "Delete": delete_block,
    "Search": search_web,
    "Stop": stop,
    "StrReplace": str_replace,
    "PreviewImage": preview_image,
}


# ============================================================
# Utility Functions
# ============================================================

def is_failed_items_empty(failed_items: str, min_length: int = 50) -> bool:
    """
    Check if the failed_items content is effectively empty.

    The reviewer agent sometimes fills <failed_items> with placeholder text like:
    - "None"
    - "<!-- No failed items -->"
    - "N/A"
    - Empty whitespace

    This function returns True if the content should be treated as empty.

    Logic:
    1. Empty or whitespace-only → empty
    2. Content shorter than min_length chars → likely placeholder, treated as empty
       (Genuine failed items need criteria name + feedback, requiring 50+ chars)

    :param failed_items: The failed items string from the reviewer.
    :param min_length: Minimum length threshold - content shorter than this is considered empty.
                       Default 50 chars since genuine feedback needs item + explanation.
    :return: True if effectively empty, False otherwise.
    """
    if not failed_items:
        return True

    # Strip whitespace
    stripped = failed_items.strip()

    if not stripped:
        return True

    # Check length threshold - genuine failed items need criteria name + feedback
    if len(stripped) < min_length:
        return True

    return False


def restore_dropped_image_links(original_df: pd.DataFrame, revised_df: pd.DataFrame) -> str:
    """
    Compare image links per block between the pre-revision and post-revision DataFrames.
    For blocks present in both, append any `![](url)` links the reviser dropped back to
    the end of the revised block text. For blocks the reviser removed outright, log the
    lost URLs (no auto-restore of the block itself).

    Mutates ``revised_df`` in place. Returns a human-readable report string listing what
    was restored or warned about; empty string if nothing was dropped.

    :param original_df: DataFrame before revision.
    :param revised_df: DataFrame after revision.
    :return: Report string describing restorations and warnings.
    """
    report_lines = []
    original_ids = set(original_df.index.tolist())
    revised_ids = set(revised_df.index.tolist())

    # Blocks present in both: restore any dropped links
    for block_id in sorted(original_ids & revised_ids):
        pre_text = str(original_df.at[block_id, 'block text'])
        post_text = str(revised_df.at[block_id, 'block text'])
        pre_links = set(extract_inline_image_links(pre_text))
        post_links = set(extract_inline_image_links(post_text))
        missing = pre_links - post_links
        if not missing:
            continue
        print(f"⚠️ Reviser dropped image links from block {block_id}: {sorted(missing)}")
        restored = post_text.rstrip()
        for url in sorted(missing):
            restored += f"\n\n![]({url})"
        revised_df.at[block_id, 'block text'] = restored
        report_lines.append(
            f"Block {block_id}: restored {len(missing)} dropped image link(s): {sorted(missing)}"
        )

    # Blocks removed entirely: warn but do not restore the block
    for block_id in sorted(original_ids - revised_ids):
        pre_text = str(original_df.at[block_id, 'block text'])
        pre_links = set(extract_inline_image_links(pre_text))
        if pre_links:
            print(
                f"⚠️ Reviser deleted block {block_id} which contained image links: "
                f"{sorted(pre_links)} (not auto-restored)"
            )
            report_lines.append(
                f"Block {block_id}: DELETED by reviser; lost image links: {sorted(pre_links)}"
            )

    return "\n".join(report_lines)


def compute_blockwise_diffs(original_df: pd.DataFrame, revised_df: pd.DataFrame) -> str:
    """
    Compute blockwise diffs between original and revised DataFrames.
    Returns a formatted string showing what changed in each block.

    :param original_df: DataFrame before revision
    :param revised_df: DataFrame after revision
    :return: Formatted string with diffs for each changed block
    """
    diffs = []

    # Get all block IDs from both DataFrames
    original_ids = set(original_df.index.tolist())
    revised_ids = set(revised_df.index.tolist())

    # Deleted blocks
    deleted_ids = original_ids - revised_ids
    for block_id in sorted(deleted_ids):
        diffs.append(f"### Block {block_id}: DELETED\n```\n{original_df.at[block_id, 'block text'][:500]}...\n```\n")

    # New blocks
    new_ids = revised_ids - original_ids
    for block_id in sorted(new_ids):
        diffs.append(f"### Block {block_id}: ADDED\n```\n{revised_df.at[block_id, 'block text']}\n```\n")

    # Modified blocks (exist in both)
    common_ids = original_ids & revised_ids
    for block_id in sorted(common_ids):
        original_text = str(original_df.at[block_id, 'block text'])
        revised_text = str(revised_df.at[block_id, 'block text'])

        if original_text != revised_text:
            # Generate unified diff
            original_lines = original_text.splitlines(keepends=True)
            revised_lines = revised_text.splitlines(keepends=True)

            diff = difflib.unified_diff(
                original_lines,
                revised_lines,
                fromfile=f'Block {block_id} (before)',
                tofile=f'Block {block_id} (after)',
                lineterm=''
            )
            diff_text = ''.join(diff)

            if diff_text.strip():
                diffs.append(f"### Block {block_id}: MODIFIED\n```diff\n{diff_text}\n```\n")

    if not diffs:
        return "No changes detected."

    return "\n".join(diffs)


# ============================================================
# Prompts
# ============================================================

research_notes_checklist_prompt = """Assume the role of a checklist agent tasked with evaluating the following blocks of research notes for the given list of checklist criteria. These research notes are later used to create slide content for the course.

These research notes are created for the following:
<course_info>
Course Name: {course_name}

Target Audience: {target_audience}

Course Objective Guidelines: {course_objective_guidelines}

Course Background: {course_background}
</course_info>

Here's the research notes to evaluate:
<research_notes>
{research_notes}
</research_notes>

Here are the checklist criteria to evaluate, along with examples for each:
<checklist_criteria>
{criteria_with_examples}
</checklist_criteria>

Critical Video-Based Research Notes Handling:
Before proceeding with evaluation, you must identify which blocks contain video-based content by looking for this specific format pattern in the "####**Research Notes:**" sections:
- Line starting with "Link:" followed by a youtube video link
- Line starting with "Video_Id:"
- Line starting with "Start:"
- Line starting with "End:"
- A "Transcript:" section with timestamped entries in the format "- '[number]': [text]"

For blocks that contain this video-based format, you must:
1. Immediately skip evaluation for those specific blocks
2. Do NOT provide any feedback or suggestions for those blocks

For blocks that do not contain video-based format, proceed with the evaluation.

This is a strict requirement - video-based research notes blocks must never be evaluated, but normal format blocks should be evaluated.

Critical Evaluation Instructions (only for non-video-based research notes):
- Be Extremely Strict in your evaluation. Do not pass any criteria unless they are fully and completely satisfied.
- Read each review criteria text WORD BY WORD and understand exactly what it is asking for.
- Do not make assumptions or interpret criteria loosely. Follow the exact wording and requirements stated in each criteria.
- Each criteria has specific requirements that must be met - evaluate against those exact requirements, not general best practices.
- Do not let overall quality of the content influence your judgment - focus solely on whether each specific criteria requirement is met.
- Strictly evaluate every single criteria provided in the checklist. Do not skip or miss any criteria.
- Inline image links: Any markdown image links (`![alt](url)`) present in the research notes are intentional and must be preserved. Do not flag their presence as a defect and do not propose removing or rewriting them. Image-link repositioning is handled by a dedicated pipeline step.

Make sure to output in the following format:
<analysis>
[Analysis of the research notes based on the checklist criteria. It is okay for the analysis to be quite long for accurate evaluation.]
</analysis>

<passed_items>
[List of checklist items that passed. Each item should be the exact review criteria text as it appears in the checklist input along with the criteria name.]
</passed_items>

<failed_items>
[List of checklist items that failed (if any, else leave blank). Format each failed item as follows:
Item: [exact review criteria text as it appears in the checklist input along with the criteria name]
Feedback: [specific detailed feedback on why it failed and what needs to be improved or corrected]

Item: [next failed review criteria text along with the criteria name]
Feedback: [feedback for that criteria]

...and so on for all failed items]
</failed_items>

NOTES:
- The analysis should be thorough and cover all aspects of the checklist and all of the criteria.
- The passed items should only include those that fully meet the criteria.
- Feedback of the failed items should include all necessary information for the author to understand what needs to be fixed.
- When listing passed or failed items, always use the complete criteria text along with the criteria name.
- For video-based research notes block, do not provide any feedback or suggestions.
- Refer to the examples provided within each criteria to understand the evaluation approach.
"""


research_notes_reviser_prompt = """You are tasked with revising the following block(s) of research notes based on the feedback provided in the checklist evaluation. These research notes are later used to create slide content for the course.

This research notes is created for the following course:
<course_info>
Course Name: {course_name}

Target Audience: {target_audience}

Course Objective Guidelines: {course_objective_guidelines}

Course Background: {course_background}
</course_info>

Here are the research notes to revise:
<research_notes>
{research_notes}
</research_notes>

Here is the detailed analysis from the review agent that evaluated the research notes:
<review_analysis>
{review_analysis}
</review_analysis>

Here is the checklist feedback with specific failed items to address:
<checklist_feedback>
{checklist_feedback}
</checklist_feedback>

Here are the review criteria with their corrective operations and examples:
<criteria_with_corrective_operations>
{criteria_with_ops_and_examples}
</criteria_with_corrective_operations>

Follow these Paraphrasing Rules when editing the research notes:
<paraphrasing_rules>
1. Use plain language, not corporate jargon
2. Sound like a real technician, not an AI
3. Keep all factual information accurate (never add or remove facts)
4. Use active voice and short sentences
5. Explain trade terminology when first used
6. Be conversational but professional
7. No buzzwords like "optimal," "leverage," "utilize," "facilitate"
8. No hype or fluff - every sentence adds value

Examples:
BAD: "Technicians must de-energize the system to mitigate hazards prior to engaging with components."
GOOD: "Shut the power off before you touch anything—it keeps you safe."

BAD: "Ensure optimal airflow parameters."
GOOD: "Good airflow keeps the system running right."

BAD: "Apply pookie to the joints."
GOOD: "Apply mastic—also called pookie by tradesman—to seal the joints."
</paraphrasing_rules>

CRITICAL FORMAT REQUIREMENTS:
When using the CRUD tools, the output you return must strictly maintain the exact format structure as of the original research notes that you are revising:

1. Each block must start with: "###Block ID: [number]"
2. Followed by: "####**Topic:**" section with the topic content
3. Followed by: "####**Subtopic:**" section with the subtopic content
4. Followed by: "####**Learning Objective:**" section with the learning objective content
5. Followed by: "####**Research Notes:**" section with the research notes content

Example of correct format:
###Block ID: 1
####**Topic:**
HVAC Fundamentals
####**Subtopic:**
Refrigeration Cycle
####**Learning Objective:**
Understand the basic refrigeration cycle
####**Research Notes:**
The refrigeration cycle consists of four main components...

IMPORTANT: Never change or remove these headers (Block ID, Topic, Subtopic, Learning Objective, Research Notes). Only modify the content after the header as required for the revision. The format is essential for the system to function properly.

Inline image links: Any markdown image links (`![alt](url)`) present in the research notes must be preserved. Do not delete, rewrite, or modify the URL or alt text of an existing image link during unrelated edits. Image-link repositioning is handled by a dedicated pipeline step; treat every `![](...)` you see as load-bearing content to keep intact.

NOTES:
- Make use of the given set of CRUD block text tools to make the necessary revisions.
- These CRUD tools allow you to create, read, update, and delete blocks of text from the above research notes as needed.
- You can only work with one block of text at a time. Each block of text is a separate entity identified by a unique ID - Block ID.
- To implement some of the feedback, you may need to make edits to multiple blocks of text.

Critical Video-Based Research Notes Handling:
Before proceeding with any revisions, you must identify which blocks contain video-based content by looking for this specific format pattern in the "####**Research Notes:**" sections:
- Line starting with "Link:" followed by a youtube video link
- Line starting with "Video_Id:"
- Line starting with "Start:"
- Line starting with "End:"
- A "Transcript:" section with timestamped entries in the format "- '[number]': [text]"

For blocks that contain this video-based format, you must:
1. Immediately skip all revision activities for those specific blocks
2. Keep those blocks exactly as they are without any modifications
3. Do NOT use any CRUD tools to make changes to those blocks
4. Do NOT apply any feedback or corrections to those blocks
5. Do NOT make any edits whatsoever to those blocks

For blocks that do not contain video-based format, proceed with normal revision using the feedback and CRUD tools. This is a strict requirement - video-based research notes blocks must never be modified, but normal format blocks should be revised as required by the feedback.

Corrective Operations and Examples:
Each criteria above includes its corrective operation and an example. Match each failed criteria from the checklist feedback with its corresponding corrective operation. Only apply the corrective operations for criteria that actually failed - ignore those for criteria that passed. Use the examples as reference to understand the approach to revising the research notes.

IMPORTANT - Completion Requirement:
After you have completed ALL necessary revisions to address the failed checklist items, you MUST call the `stop` tool to signal that your work is complete. Provide a brief summary of what was accomplished in the reason parameter. Do not stop until all failed items have been addressed.
"""


aggregator_review_prompt = """You are an aggregator reviewer. Multiple independent review agents have evaluated the same set of research notes, each focusing on different quality criteria. The reviser agent will ONLY see your output — it will NOT see the individual reviews. Therefore, your report must be comprehensive and preserve all actionable detail from the original feedback.

These research notes are created for the following:
<course_info>
Course Name: {course_name}

Target Audience: {target_audience}

Course Objective Guidelines: {course_objective_guidelines}

Course Background: {course_background}
</course_info>

Here are the research notes that were evaluated:
<research_notes>
{research_notes}
</research_notes>

Here are the individual review results from each reviewer:
<individual_results>
{individual_results}
</individual_results>

Instructions:
1. Read through all individual reviews carefully.
2. Cross-reference each reviewer's findings against the actual research notes above to verify accuracy.
3. If two reviewers give conflicting assessments on the same block or criteria, use the research notes to determine which assessment is correct.
4. PRESERVE the original feedback text from each reviewer as-is. Do NOT summarize, rephrase, or shorten feedback. The individual reviewers have written feedback in a format that is easy for the reviser to follow — keep that format intact.
5. Deduplicate only when two reviewers flag the exact same issue on the exact same block. In that case, keep the more detailed feedback of the two.
6. Do NOT remove failures that were identified by a reviewer, unless another reviewer's conflicting assessment is demonstrably correct based on the actual research notes.
7. If while cross-referencing you spot additional issues that no reviewer caught, you may add them as new failures — but clearly mark them as "[Aggregator-identified]".
8. Inline image links: Any markdown image links (`![alt](url)`) present in the research notes must be preserved. Do not synthesize or retain feedback that asks the reviser to remove or rewrite existing image links. Image-link repositioning is handled by a dedicated pipeline step.

Output your consolidated report:

<analysis>
[Unified analysis combining insights from all reviewers. Organize by block or theme for clarity. Note any conflicts between reviewers and how you resolved them based on the actual research notes. It is okay for the analysis to be quite long for accuracy and clarity.]
</analysis>

<failed_items>
[Consolidated list of ALL failed items organized by block. The reviser works one block at a time, so group all feedback for each block together. If no items failed across any reviewer, leave this section blank.]
</failed_items>
"""


# ============================================================
# Core Agent Functions (unchanged from v1)
# ============================================================

@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Checklist Based Review and Revise Agents (v2)",
    "function_name": "run_research_notes_checklist_agent",
    "user_id": st.session_state.get("role", "anonymous")
})
def run_research_notes_checklist_agent(course_name, target_audience, course_objective_guidelines, course_background, research_notes, criteria_with_examples, llm = "gemini_3_flash", message_history=None, blockwise_diffs=None):
    """
    Run the research notes checklist agent with the provided parameters.
    :param course_name: Name of the course for which the research notes are created.
    :param target_audience: Target audience for the course.
    :param course_objective_guidelines: Guidelines for the course objectives.
    :param course_background: Background information about the course.
    :param research_notes: The block of research notes to evaluate.
    :param criteria_with_examples: The checklist criteria with their examples to evaluate the research notes against.
    :param llm: The language model to use for the agent.
    :param message_history: Optional list of previous messages (tuples) to continue the conversation.
    :param blockwise_diffs: Optional string containing blockwise diffs from reviser changes (for subsequent iterations).
    :return: A tuple of (analysis, failed_items, updated_message_history).
    """
    research_notes_checklist_agent = Chain(llm = llm, tags = ["analysis", "passed_items", "failed_items"])

    # If we have message history, add all previous messages first
    if message_history:
        research_notes_checklist_agent.add_messages(message_history)
        # For subsequent iterations, send a follow-up message with diffs
        follow_up_content = f"""The reviser agent has made changes based on your previous feedback. Here are the CHANGES made (blockwise diffs):

<blockwise_diffs>
{blockwise_diffs}
</blockwise_diffs>

Please re-evaluate the research notes based on these changes. Check if the previous issues have been addressed and identify any remaining or new issues.

Output your evaluation in the same format:
<analysis>
[Your analysis focusing on whether the changes address the previous feedback]
</analysis>

<passed_items>
[List of checklist items that now pass]
</passed_items>

<failed_items>
[List of checklist items that still fail or new failures, with feedback (if any, else leave blank)]
</failed_items>"""
        research_notes_checklist_agent.add_message(role="user", content=follow_up_content)
    else:
        # First iteration - send the full initial prompt
        research_notes_checklist_agent.add_message(
            role = "user",
            content = research_notes_checklist_prompt.format(
                course_name = course_name,
                target_audience = target_audience,
                course_objective_guidelines = course_objective_guidelines,
                course_background = course_background,
                research_notes = research_notes,
                criteria_with_examples = criteria_with_examples,
            )
        )

    response = research_notes_checklist_agent.run()

    # Return updated message history for subsequent iterations
    updated_history = research_notes_checklist_agent.messages_list

    return response["analysis"], response["failed_items"], updated_history


@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Checklist Based Review and Revise Agents (v2)",
    "function_name": "run_reviser_agent",
    "user_id": st.session_state.get("role", "anonymous")
})
def run_reviser_agent(research_notes, review_analysis, checklist_feedback, criteria_with_ops_and_examples, df, course_name, target_audience, course_objective_guidelines, course_background, tools=None, llm = "gemini_3_flash", message_history=None, blockwise_diffs=None):
    """
    Run the reviser agent with the provided parameters.
    :param research_notes: The block of research notes to revise.
    :param review_analysis: The detailed analysis from the review agent explaining evaluation reasoning.
    :param checklist_feedback: The feedback from the checklist evaluation to consider for revisions.
    :param criteria_with_ops_and_examples: All criteria with their corrective operations and examples for the reviser to reference.
    :param df: The dataframe containing the research notes.
    :param course_name: Name of the course for which the research notes are created.
    :param target_audience: Target audience for the course.
    :param course_objective_guidelines: Guidelines for the course objectives.
    :param course_background: Background information about the course.
    :param tools: List of tool functions to make available to the agent. Defaults to all CRUD tools.
    :param llm: The language model to use for the agent.
    :param message_history: Optional list of previous messages to continue the conversation.
    :param blockwise_diffs: Optional string containing blockwise diffs from your previous changes (for context in subsequent iterations).
    :return: A tuple of (revised_dataframe, updated_message_history).
    """
    # Default to all CRUD tools if none specified
    if tools is None:
        tools = [create_block, read_blocks, update_block, delete_block, str_replace, stop]
    print(f"\n🔄 STARTING REVISER AGENT for {len(df)} blocks")
    print(f"📝 Checklist feedback length: {len(checklist_feedback)} chars")
    print("=" * 80)

    rate_limiter = InMemoryRateLimiter(
        requests_per_second=1,  # <-- Super slow! We can only make a request once every 10 seconds!!
        check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
        max_bucket_size=10,  # Controls the maximum burst size.
    )

    from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model = "gemini-3-flash-preview",
        thinking_level = "high",
        include_thoughts = True,
        rate_limiter = rate_limiter,
        max_retries = 20,
    )

    class BufferState(AgentState):
        """
        Short-term state for the agent execution.
        Only one field is needed: the shared dataframe instance.
        """
        df: pd.DataFrame = Field(default_factory=pd.DataFrame)

    @wrap_tool_call
    def handle_tool_errors(request, handler):
        """Handle tool execution errors with custom messages."""
        try:
            return handler(request)
        except Exception as e:
            # Return a custom error message to the model
            return ToolMessage(
                content=f"Tool error: Please check your input and try again.\n({str(e)})",
                tool_call_id=request.tool_call["id"]
            )

    print(f"🔧 Tools available to reviser: {[t.name for t in tools]}")

    graph = create_agent(
        model=llm,
        tools=tools,
        state_schema=BufferState,
        middleware=[handle_tool_errors],
    )

    if message_history:
        # Continue from previous conversation - just append new follow-up message
        follow_up_content = f"""The reviewer has re-evaluated the research notes after your previous revisions. Here's what you changed:

<your_previous_changes>
{blockwise_diffs}
</your_previous_changes>

The reviewer found the following remaining issues:

<new_review_analysis>
{review_analysis}
</new_review_analysis>

<remaining_failed_items>
{checklist_feedback}
</remaining_failed_items>

Please address these remaining issues using the CRUD tools. Remember to call the `stop` tool when you have completed all revisions."""
        messages = message_history + [HumanMessage(content=follow_up_content)]
    else:
        # First iteration - send the full initial prompt
        messages = [HumanMessage(content=research_notes_reviser_prompt.format(
            research_notes=research_notes,
            review_analysis=review_analysis,
            checklist_feedback=checklist_feedback,
            criteria_with_ops_and_examples=criteria_with_ops_and_examples,
            course_name=course_name,
            target_audience=target_audience,
            course_objective_guidelines=course_objective_guidelines,
            course_background=course_background,))]

    state = {
        "messages": messages,
        "df": df,        # one buffer for the whole session
    }

    final_state = graph.invoke(
        state,
        {"recursion_limit": 50}
    )

    print("=" * 80)
    print("✅ REVISER AGENT COMPLETED")
    print(f"📊 Final DataFrame shape: {final_state['df'].shape}")
    print("=" * 80)

    # Return messages directly to preserve thought signatures and all metadata
    return final_state["df"], final_state["messages"]


# ============================================================
# NEW: Parallel Review & Aggregation Functions
# ============================================================

@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Checklist Based Review and Revise Agents (v2)",
    "function_name": "_review_single_slice",
    "user_id": st.session_state.get("role", "anonymous")
})
def _review_single_slice(
    key, df_slice, course_name, target_audience,
    course_objective_guidelines, course_background,
    criteria_with_examples_str, reviewer_llm, slice_history, blockwise_diffs
):
    """
    Review a single scope slice. Designed to be called from a thread pool.

    :param key: The scope slice key (e.g., topic name, row index, "ALL")
    :param df_slice: DataFrame slice for this scope
    :param course_name: Name of the course
    :param target_audience: Target audience
    :param course_objective_guidelines: Course objective guidelines
    :param course_background: Course background
    :param criteria_with_examples_str: Criteria with examples for the review agent
    :param reviewer_llm: LLM to use for review
    :param slice_history: Message history for this slice from previous iterations
    :param blockwise_diffs: Diffs from previous revision (for re-review context)
    :return: Tuple of (key, analysis, failed_items, history)
    """
    research_notes_str = "\n\n---\n\n".join(df_slice["block text"].tolist())

    analysis, failed_items, history = run_research_notes_checklist_agent(
        course_name=course_name,
        target_audience=target_audience,
        course_objective_guidelines=course_objective_guidelines,
        course_background=course_background,
        research_notes=research_notes_str,
        criteria_with_examples=criteria_with_examples_str,
        llm=reviewer_llm,
        message_history=slice_history,
        blockwise_diffs=blockwise_diffs,
    )

    return key, analysis, failed_items, history


@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Checklist Based Review and Revise Agents (v2)",
    "function_name": "run_single_task_review",
    "user_id": st.session_state.get("role", "anonymous")
})
def run_single_task_review(
    task_key, scope, scope_to_selector, research_notes_df,
    criteria_with_examples_str, course_name, target_audience,
    course_objective_guidelines, course_background,
    reviewer_llm="gemini_3_flash", reviewer_histories=None, blockwise_diffs=None
):
    """
    Run review for a single task across all its scope slices.
    Each task may slice the data differently based on its scope (Global, Topic, etc.).
    Scope slices are processed in parallel using a thread pool.

    :param task_key: String identifier for this task (e.g., "Task Name (Scope)")
    :param scope: The scope level for slicing (Global (full output), Topic, Subtopic, Learning Objective)
    :param scope_to_selector: Mapping from scope to DataFrame selector
    :param research_notes_df: The full DataFrame of research notes
    :param criteria_with_examples_str: Criteria with examples for the review agent
    :param course_name: Name of the course
    :param target_audience: Target audience
    :param course_objective_guidelines: Course objective guidelines
    :param course_background: Course background
    :param reviewer_llm: LLM to use for review
    :param reviewer_histories: Dict of {slice_key: message_history} from previous iterations
    :param blockwise_diffs: Diffs from previous revision (for re-review context)
    :return: Tuple of (task_key, combined_analysis, combined_failed_items, updated_histories)
    """
    print(f"📋 Starting review for task '{task_key}' with scope '{scope}'")

    updated_histories = {}

    if reviewer_histories is None:
        reviewer_histories = {}

    # Collect all non-empty scope slices
    slices = [
        (key, df_slice)
        for key, df_slice in iterate_scope(scope, research_notes_df, scope_to_selector)
        if not df_slice.empty
    ]

    if not slices:
        print(f"⚠️ No non-empty slices for task '{task_key}'")
        return task_key, "No analysis generated.", "", {}

    print(f"🚀 Launching {len(slices)} parallel scope-slice reviews for task '{task_key}'...")

    # Run all scope slices in parallel, preserving original order
    # Use index-keyed dict so results can be reassembled in submission order
    slice_results = [None] * len(slices)
    with ThreadPoolExecutor(max_workers=min(5, len(slices))) as executor:
        futures_map = {}
        for idx, (key, df_slice) in enumerate(slices):
            slice_history = reviewer_histories.get(key)
            future = executor.submit(
                _review_single_slice,
                key=key,
                df_slice=df_slice,
                course_name=course_name,
                target_audience=target_audience,
                course_objective_guidelines=course_objective_guidelines,
                course_background=course_background,
                criteria_with_examples_str=criteria_with_examples_str,
                reviewer_llm=reviewer_llm,
                slice_history=slice_history,
                blockwise_diffs=blockwise_diffs,
            )
            futures_map[future] = (idx, key)

        for future in as_completed(futures_map):
            idx, slice_key = futures_map[future]
            try:
                slice_results[idx] = future.result()
            except Exception as e:
                print(f"❌ Error reviewing slice '{slice_key}' for task '{task_key}': {e}")
                slice_results[idx] = (slice_key, f"Error during review: {e}", "", None)

    # Assemble results (order matches original iterate_scope order)
    all_analyses = []
    all_failed_items = []
    for key, analysis, failed_items, history in slice_results:
        all_analyses.append(f"[Scope: {key}]\n{analysis}")
        if not is_failed_items_empty(failed_items):
            all_failed_items.append(f"[Scope: {key}]\n{failed_items}")
        if history is not None:
            updated_histories[key] = history

    combined_analysis = "\n\n---\n\n".join(all_analyses) if all_analyses else "No analysis generated."
    combined_failed = "\n\n---\n\n".join(all_failed_items) if all_failed_items else ""

    print(f"✅ Review for task '{task_key}': {len(all_analyses)} slices analyzed, {len(all_failed_items)} with failures")

    return task_key, combined_analysis, combined_failed, updated_histories


@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Checklist Based Review and Revise Agents (v2)",
    "function_name": "run_all_task_reviews_parallel",
    "user_id": st.session_state.get("role", "anonymous")
})
def run_all_task_reviews_parallel(
    task_configs, research_notes_df, scope_to_selector,
    course_name, target_audience, course_objective_guidelines, course_background,
    reviewer_llm="gemini_3_flash", all_reviewer_histories=None, blockwise_diffs=None
):
    """
    Run all task reviews in parallel using a thread pool.

    :param task_configs: List of dicts, each with keys: task_key, scope, criteria_with_examples_str
    :param research_notes_df: The full DataFrame of research notes
    :param scope_to_selector: Mapping from scope to DataFrame selector
    :param course_name: Name of the course
    :param target_audience: Target audience
    :param course_objective_guidelines: Course objective guidelines
    :param course_background: Course background
    :param reviewer_llm: LLM to use for review
    :param all_reviewer_histories: Dict of {task_key: {slice_key: history}} from previous iterations
    :param blockwise_diffs: Diffs from previous revision (for re-review context)
    :return: List of (task_key, analysis, failed_items, histories) tuples
    """
    if all_reviewer_histories is None:
        all_reviewer_histories = {}

    if not task_configs:
        print("No task configs to process")
        return []

    print(f"🚀 Launching {len(task_configs)} parallel task reviews...")

    results = [None] * len(task_configs)
    futures_map = {}
    with ThreadPoolExecutor(max_workers=min(5, len(task_configs))) as executor:
        for idx, config in enumerate(task_configs):
            task_key = config['task_key']
            future = executor.submit(
                run_single_task_review,
                task_key=task_key,
                scope=config['scope'],
                scope_to_selector=scope_to_selector,
                research_notes_df=research_notes_df,
                criteria_with_examples_str=config['criteria_with_examples_str'],
                course_name=course_name,
                target_audience=target_audience,
                course_objective_guidelines=course_objective_guidelines,
                course_background=course_background,
                reviewer_llm=reviewer_llm,
                reviewer_histories=all_reviewer_histories.get(task_key),
                blockwise_diffs=blockwise_diffs,
            )
            futures_map[future] = (idx, task_key)

    for future in as_completed(futures_map):
        idx, task_key = futures_map[future]
        try:
            results[idx] = future.result()
            print(f"✅ Parallel review completed for task '{task_key}'")
        except Exception as e:
            print(f"❌ Error in parallel review for task '{task_key}': {e}")
            results[idx] = (task_key, f"Error during review: {e}", "", {})

    print(f"🏁 All {len(results)} parallel task reviews completed")
    return results


@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Checklist Based Review and Revise Agents (v2)",
    "function_name": "aggregate_review_results",
    "user_id": st.session_state.get("role", "anonymous")
})
def aggregate_review_results(review_results, research_notes_df, course_name, target_audience, course_objective_guidelines, course_background, llm="gemini_3_flash"):
    """
    Aggregate results from multiple parallel review tasks into a single unified report.
    Uses an LLM to intelligently merge and deduplicate findings from different reviewers.
    The aggregator receives full course context and research notes so it can resolve
    conflicting assessments between reviewers by checking the actual content.

    Short-circuits:
    - If no failures across all tasks, returns combined analysis with empty failed_items.
    - If only one task produced results, returns that task's results directly.

    :param review_results: List of (task_key, analysis, failed_items, histories) tuples
    :param research_notes_df: The full DataFrame of research notes (for context)
    :param course_name: Name of the course
    :param target_audience: Target audience for the course
    :param course_objective_guidelines: Course objective guidelines
    :param course_background: Course background information
    :param llm: LLM to use for aggregation
    :return: Tuple of (aggregated_analysis, aggregated_failed_items)
    """
    # Quick check - if no failures, skip LLM aggregation
    has_failures = any(
        not is_failed_items_empty(failed)
        for _, _, failed, _ in review_results
    )

    if not has_failures:
        combined_analysis = "\n\n".join(
            f"**Task: {task_key}**\n{analysis}"
            for task_key, analysis, _, _ in review_results
        )
        print("✅ Aggregator: No failures found across any task - skipping LLM aggregation")
        return combined_analysis, ""

    # If only one task returned results, return it directly (no aggregation needed)
    if len(review_results) == 1:
        _, analysis, failed, _ = review_results[0]
        print("✅ Aggregator: Single task - returning results directly")
        return analysis, failed

    # Build individual results for the aggregator prompt
    individual_results = []
    for i, (task_key, analysis, failed_items, _) in enumerate(review_results, 1):
        individual_results.append(
            f"<review_{i}>\n"
            f"Task: {task_key}\n\n"
            f"Analysis:\n{analysis}\n\n"
            f"Failed Items:\n{failed_items if not is_failed_items_empty(failed_items) else 'None - all criteria passed for this task'}\n"
            f"</review_{i}>"
        )

    individual_results_str = "\n\n".join(individual_results)

    print(f"🔄 Aggregator: Combining results from {len(review_results)} tasks via LLM...")

    # Build research notes string for context
    research_notes_str = "\n\n---\n\n".join(research_notes_df["block text"].tolist())

    # Use LLM to aggregate
    aggregator = Chain(llm=llm, tags=["analysis", "failed_items"])
    aggregator.add_message(
        role="user",
        content=aggregator_review_prompt.format(
            course_name=course_name,
            target_audience=target_audience,
            course_objective_guidelines=course_objective_guidelines,
            course_background=course_background,
            research_notes=research_notes_str,
            individual_results=individual_results_str,
        )
    )
    response = aggregator.run()

    print("✅ Aggregator: LLM aggregation complete")
    return response["analysis"], response["failed_items"]


# ============================================================
# Main Orchestration (rewritten for v2 parallel flow)
# ============================================================

@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Checklist Based Review and Revise Agents (v2)",
    "function_name": "run_research_notes_checklist_and_reviser",
    "user_id": st.session_state.get("role", "anonymous")
})
def run_research_notes_checklist_and_reviser(sheet, course_name, target_audience, checklist_sheet_link, gc, llm = "gemini_3_flash", reviewer_llm = "gemini_3_flash"):
    """
    Run the research notes checklist and reviser pipeline using v2 parallel architecture.

    V2 Flow:
    1. Prepare all task configs from the checklist sheet
    2. Iteration loop:
       a. Run ALL task reviews in parallel (each task uses its own scope for slicing)
       b. Aggregate all review results via LLM into unified feedback
       c. If no failures, exit early
       d. Run a single reviser pass on the full DataFrame with aggregated feedback
       e. Compute diffs for next iteration's re-review context

    :param sheet: Google Sheet object for the course.
    :param course_name: Name of the course.
    :param target_audience: Target audience for the course.
    :param checklist_sheet_link: Link to the Google Sheet containing the checklist.
    :param gc: Google Sheets client instance.
    :param llm: The language model to use for the reviser agent.
    :param reviewer_llm: The language model to use for the review agents.
    :return: None
    """

    # ── Load checklist sheet ──
    checklist_sheet = gc.open_by_url(checklist_sheet_link)
    checklist_worksheet, checklist_df = get_sheet_data_and_df(sheet=checklist_sheet, sheet_name="Research Notes Checklist")

    # ── Load course info ──
    course_info_worksheet, course_info_df = get_sheet_data_and_df(sheet=sheet, sheet_name="Course info")

    course_objective_guidelines = ""
    course_background = ""

    if not course_info_df.empty:
        if "Course Objective Guidelines" in course_info_df.columns:
            course_objective_guidelines = str(course_info_df["Course Objective Guidelines"].iloc[0]) if not pd.isna(course_info_df["Course Objective Guidelines"].iloc[0]) else ""
        if "Course Background" in course_info_df.columns:
            course_background = str(course_info_df["Course Background"].iloc[0]) if not pd.isna(course_info_df["Course Background"].iloc[0]) else ""

    # ── Load research notes ──
    research_notes_sheet, research_notes_df = get_sheet_data_and_df(sheet=sheet, sheet_name="Final Outline")

    # ── Create backup for delete step restore ──
    backup_ws_name = "Backup Final Outline Sheet for Delete step of Research Notes Checklist"
    print(f"📋 Creating backup sheet '{backup_ws_name}' for delete step functionality...")
    backup_ws, _ = create_or_read_worksheet(sheet, backup_ws_name)
    clear_worksheet(backup_ws)
    save_to_sheet(backup_ws, research_notes_df)
    format_worksheet(backup_ws)
    try:
        hide_worksheet_by_name(sheet, backup_ws_name)
        print(f"✅ Backup sheet created and hidden successfully")
    except Exception as e:
        print(f"⚠️ Warning: Could not hide backup sheet: {e}")

    # ── Prepare DataFrame ──
    if "order" not in research_notes_df.columns:
        research_notes_df["order"] = research_notes_df.index.astype(float)

    if 'block text' not in research_notes_df.columns:
        research_notes_df['block text'] = ''

    for index, row in research_notes_df.iterrows():
        if pd.isna(research_notes_df.at[index, 'block text']) or research_notes_df.at[index, 'block text'] == '':
            research_notes_str = ""
            research_notes_str += f"###Block ID: {index}\n"
            research_notes_str += f"####**Topic:**\n{row['Topic']}\n"
            research_notes_str += f"####**Subtopic:**\n{row['Subtopic']}\n"
            research_notes_str += f"####**Learning Objective:**\n{row['Learning Objectives']}\n"
            research_notes_str += f"####**Research Notes:**\n{row['research_notes']}\n"
            research_notes_df.at[index, 'block text'] = research_notes_str

    scope_to_selector = {
        "Global (full output)": [],          # whole frame
        "Topic":  ["Topic"],                 # group by Topic col
        "Subtopic": ["Topic", "Subtopic"],   # group by both
        "Learning Objective": "__row__",     # one row each
    }

    # ── Prepare all task configs upfront ──
    task_scope_groups = list(checklist_df.groupby(["Task", "Scope"], sort=False))

    task_configs = []
    all_criteria_with_ops_strs = []
    all_reviser_tools = {}  # {tool.name: tool} - dict to deduplicate (StructuredTool is unhashable)
    global_max_iterations = 1

    print(f"\n{'='*80}")
    print(f"📊 Preparing {len(task_scope_groups)} task configs for parallel review")
    print(f"{'='*80}")

    for (task, scope), grp in task_scope_groups:
        print(f"  Task: {task}, Scope: {scope}")

        # Extract criteria data
        criteria_list = grp["Review Criteria"].tolist()
        criteria_names = grp["Criteria Name"].tolist()
        review_examples_list = grp["Review Agent Examples"].tolist()
        corrective_ops_list = grp["Corrective Operations"].tolist()
        reviser_examples_list = grp["Reviser Agent Examples"].tolist()

        # Extract iteration count with default of 1
        iteration_count_values = grp["Iteration Count"].tolist() if "Iteration Count" in grp.columns else [1]
        try:
            max_iterations = int(iteration_count_values[0]) if iteration_count_values else 1
            if max_iterations < 1:
                max_iterations = 1
        except (ValueError, TypeError):
            max_iterations = 1

        global_max_iterations = max(global_max_iterations, max_iterations)

        # Extract and map tools from "Review Agent Tools" column
        tools_list = grp["Review Agent Tools"].tolist() if "Review Agent Tools" in grp.columns else []
        unique_tool_names = set()
        for tools_str in tools_list:
            if pd.notna(tools_str) and tools_str:
                tool_names = [t.strip() for t in str(tools_str).split(",")]
                unique_tool_names.update(tool_names)

        reviser_tools = []
        for tool_name in unique_tool_names:
            if tool_name in TOOL_NAME_TO_FUNCTION:
                reviser_tools.append(TOOL_NAME_TO_FUNCTION[tool_name])
            else:
                print(f"⚠️ Unknown tool name in checklist: {tool_name}")

        if not reviser_tools:
            reviser_tools = [create_block, read_blocks, update_block, delete_block, str_replace, search_web, stop]
        if stop not in reviser_tools:
            reviser_tools.append(stop)

        # Accumulate tools for the combined reviser
        for t in reviser_tools:
            all_reviser_tools[t.name] = t

        # Build criteria + examples for review agent
        criteria_with_examples = []
        for i, (criteria_name, criteria, example) in enumerate(zip(criteria_names, criteria_list, review_examples_list), 1):
            criteria_with_examples.append(
                f"<criteria_{i}>\n"
                f"Review Criteria name: {criteria_name}\n"
                f"Review Criteria: {criteria}\n\n"
                f"<example>\n{example}\n</example>\n"
                f"</criteria_{i}>"
            )
        criteria_with_examples_str = "\n\n".join(criteria_with_examples)

        # Build criteria + corrective ops + examples for reviser agent
        criteria_with_ops_and_examples = []
        for i, (criteria_name, criteria, corrective_ops, reviser_example) in enumerate(zip(criteria_names, criteria_list, corrective_ops_list, reviser_examples_list), 1):
            criteria_with_ops_and_examples.append(
                f"<criteria_{i}>\n"
                f"Review Criteria name: {criteria_name}\n"
                f"Review Criteria: {criteria}\n\n"
                f"Corrective Operation: {corrective_ops}\n\n"
                f"<example>\n{reviser_example}\n</example>\n"
                f"</criteria_{i}>"
            )
        criteria_with_ops_and_examples_str = "\n\n".join(criteria_with_ops_and_examples)
        all_criteria_with_ops_strs.append(criteria_with_ops_and_examples_str)

        task_key = f"{task} ({scope})"
        task_configs.append({
            'task_key': task_key,
            'scope': scope,
            'criteria_with_examples_str': criteria_with_examples_str,
        })

        print(f"    Criteria: {len(criteria_list)}, Tools: {[t.name for t in reviser_tools]}, Max iter: {max_iterations}")

    # Combined criteria and tools for the single reviser
    combined_criteria_with_ops = "\n\n".join(all_criteria_with_ops_strs)
    combined_reviser_tools = list(all_reviser_tools.values())
    if stop.name not in all_reviser_tools:
        combined_reviser_tools.append(stop)

    print(f"\n📊 Global max iterations: {global_max_iterations}")
    print(f"🔧 Combined reviser tools: {[t.name for t in combined_reviser_tools]}")

    # ── Parallel Review → Aggregate → Revise Loop ──
    all_reviewer_histories = {}  # {task_key: {slice_key: history}}
    reviser_history = None
    last_diffs = None

    for iteration in range(1, global_max_iterations + 1):
        print(f"\n{'='*80}")
        print(f"🔄 GLOBAL ITERATION {iteration}/{global_max_iterations}")
        print(f"{'='*80}")

        # ── Phase 1: Run all task reviews in parallel ──
        print(f"\n📋 Phase 1: Parallel reviews for {len(task_configs)} tasks...")
        review_results = run_all_task_reviews_parallel(
            task_configs=task_configs,
            research_notes_df=research_notes_df,
            scope_to_selector=scope_to_selector,
            course_name=course_name,
            target_audience=target_audience,
            course_objective_guidelines=course_objective_guidelines,
            course_background=course_background,
            reviewer_llm=reviewer_llm,
            all_reviewer_histories=all_reviewer_histories,
            blockwise_diffs=last_diffs,
        )

        # Update reviewer histories for next iteration
        for task_key, _, _, histories in review_results:
            all_reviewer_histories[task_key] = histories

        # ── Phase 2: Aggregate review results ──
        print(f"\n🔗 Phase 2: Aggregating {len(review_results)} review results...")
        aggregated_analysis, aggregated_failed = aggregate_review_results(
            review_results,
            research_notes_df=research_notes_df,
            course_name=course_name,
            target_audience=target_audience,
            course_objective_guidelines=course_objective_guidelines,
            course_background=course_background,
            llm=reviewer_llm,
        )

        print(f"📝 Aggregated failed items length: {len(aggregated_failed)} chars")
        if aggregated_failed:
            print(f"📝 Aggregated failed items preview: {aggregated_failed[:300]}...")

        # ── Phase 3: Check if all criteria passed ──
        if is_failed_items_empty(aggregated_failed):
            print(f"\n✅ All criteria passed at iteration {iteration} - exiting early")
            break

        # ── Phase 4: Run single reviser with aggregated feedback ──
        print(f"\n🔧 Phase 4: Running reviser with aggregated feedback...")
        df_before_revision = research_notes_df.copy()
        research_notes_str = "\n\n---\n\n".join(research_notes_df["block text"].tolist())

        research_notes_df, reviser_history = run_reviser_agent(
            research_notes=research_notes_str,
            review_analysis=aggregated_analysis,
            checklist_feedback=aggregated_failed,
            criteria_with_ops_and_examples=combined_criteria_with_ops,
            df=research_notes_df,
            course_name=course_name,
            target_audience=target_audience,
            course_objective_guidelines=course_objective_guidelines,
            course_background=course_background,
            tools=combined_reviser_tools,
            llm=llm,
            message_history=reviser_history,
            blockwise_diffs=last_diffs,
        )

        # ── Phase 4b: Restore any image links the reviser dropped ──
        image_guard_report = restore_dropped_image_links(df_before_revision, research_notes_df)
        if image_guard_report:
            print(f"\n🖼️ Image-link guard restored dropped links:\n{image_guard_report}")

        # ── Phase 5: Compute diffs for next iteration ──
        last_diffs = compute_blockwise_diffs(df_before_revision, research_notes_df)
        if image_guard_report:
            last_diffs = (
                f"{last_diffs}\n\n### Image-link guard (automatic restorations)\n"
                f"```\n{image_guard_report}\n```\n"
            )
        print(f"\n📊 Iteration {iteration} complete. DataFrame shape: {research_notes_df.shape}")
        if last_diffs != "No changes detected.":
            print(f"📝 Diffs preview: {last_diffs[:300]}...")

    # ── Post-processing: Validate and save ──
    print("\n🔍 Validating and correcting block text format...")
    corrected_count = 0
    for index, row in research_notes_df.iterrows():
        block_text = str(row.get('block text', '')).strip()

        has_correct_format = (
            'Block ID' in block_text or
            '####**Topic:**' in block_text
        )

        if not has_correct_format and block_text != '':
            research_notes_str = ""
            research_notes_str += f"###Block ID: {index}\n"
            research_notes_str += f"####**Topic:**\n{row['Topic']}\n"
            research_notes_str += f"####**Subtopic:**\n{row['Subtopic']}\n"
            research_notes_str += f"####**Learning Objective:**\n{row['Learning Objectives']}\n"
            research_notes_str += f"####**Research Notes:**\n{block_text}\n"

            research_notes_df.loc[index, 'block text'] = research_notes_str
            corrected_count += 1
            print(f"✅ Row {index} corrected successfully")
        elif not has_correct_format and block_text == '':
            print(f"⚠️ Row {index}: Empty block text, skipping")
        else:
            print(f"✅ Row {index}: Format is correct, no action needed")

    print(f"✅ Corrected format for {corrected_count} rows")

    # Clear the worksheet first to handle row deletions properly
    research_notes_sheet.clear()

    # Save to sheet
    save_to_sheet(worksheet=research_notes_sheet, df=research_notes_df)

    # Parse the updated block_text column back to individual columns
    print("\n🔄 Parsing updated block_text content back to individual columns...")
    research_notes_df = parse_block_text_to_columns(research_notes_df)

    column_renames = {}
    if 'order' in research_notes_df.columns:
        column_renames['order'] = 'rn_order'
    if 'block text' in research_notes_df.columns:
        column_renames['block text'] = 'rn_block text'

    if column_renames:
        research_notes_df = research_notes_df.rename(columns=column_renames)

    save_to_sheet(worksheet=research_notes_sheet, df=research_notes_df)

    # Hide working columns
    columns_to_hide = []
    if 'rn_order' in research_notes_df.columns:
        columns_to_hide.append('rn_order')
    if 'rn_block text' in research_notes_df.columns:
        columns_to_hide.append('rn_block text')

    if columns_to_hide:
        hide_columns_by_name(research_notes_sheet, columns_to_hide, research_notes_df)

    print("✅ Block text parsing completed. Individual columns updated with revised content.")

    return


# ============================================================
# Block Text Parsing (unchanged from v1)
# ============================================================

class BlockTextContent(BaseModel):
    topic: str = Field(description="The topic extracted from the block text.")
    subtopic: str = Field(description="The subtopic extracted from the block text.")
    learning_objective: str = Field(description="The learning objective extracted from the block text.")
    research_notes: str = Field(description="The research notes content extracted from the block text.")

block_text_parsing_prompt = """You are an expert parser for research notes block text. Given a block of text in a structured format, extract the following fields:

- topic: The value after '####**Topic:**' (extract only the topic text, not the header)
- subtopic: The value after '####**Subtopic:**' (extract only the subtopic text, not the header)
- learning_objective: The value after '####**Learning Objective:**' (extract only the learning objective text, not the header)
- research_notes: The value after '####**Research Notes:**' (extract only the research notes content, not the header)

Return the output as a structured object with these fields. Do not add or infer any information. Only extract what is present in the block. Preserve all formatting, line breaks, and structure in the research_notes field.

Block:
{block}
"""


@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Checklist Based Review and Revise Agents (v2)",
    "function_name": "parse_block_text_row",
    "user_id": st.session_state.get("role", "anonymous")
})
def parse_block_text_row(block_text_cell, index):
    """
    Parses a single block_text cell to extract Topic, Subtopic, Learning Objective, and Research Notes.
    :param block_text_cell: The raw text from the block_text column.
    :param index: The row index (for debugging).
    :return: Dict with parsed content or None if parsing fails.
    """
    try:
        # Use LLM + Pydantic to parse and validate
        agent = Chain(llm="gemini_2_flash")
        agent.add_message(
            role="user",
            content=block_text_parsing_prompt.format(block=block_text_cell)
        )
        agent.structured_output = BlockTextContent
        response = agent.run()
        parsed = response.model_dump()
        return parsed
    except Exception as e:
        print(f"Failed to parse block_text for row {index}: {e}")
        return None


@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Checklist Based Review and Revise Agents (v2)",
    "function_name": "parse_block_text_to_columns",
    "user_id": st.session_state.get("role", "anonymous")
})
def parse_block_text_to_columns(df, max_workers=5):
    """
    Parses the block_text column and updates the individual Topic, Subtopic, Learning Objectives, and research_notes columns.
    :param df: The DataFrame to update.
    :param max_workers: Number of parallel workers (default 5).
    :return: The updated DataFrame.
    """
    # Check if block_text column exists
    if 'block text' not in df.columns:
        print("No 'block text' column found. Nothing to parse.")
        return df

    # Prepare for parallel processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, row in df.iterrows():
            block_text_cell = str(row.get("block text", "")).strip()
            if not block_text_cell:
                continue  # Skip empty block_text
            future = executor.submit(parse_block_text_row, block_text_cell, index)
            futures_map[future] = index

        total_tasks = len(futures_map)
        progress = SmartProgressBar(total_tasks=total_tasks, description="Parsing block text", save_interval=5)

        for future in as_completed(futures_map):
            index = futures_map[future]
            try:
                parsed_content = future.result()
                if parsed_content:
                    # Update the DataFrame with parsed content
                    df.loc[index, 'Topic'] = parsed_content['topic']
                    df.loc[index, 'Subtopic'] = parsed_content['subtopic']
                    df.loc[index, 'Learning Objectives'] = parsed_content['learning_objective']
                    df.loc[index, 'research_notes'] = parsed_content['research_notes']
                    print(f"Updated row {index} with parsed content")
            except Exception as e:
                print(f"Error parsing row {index}: {e}")
            progress.update()

    print(f"Successfully parsed block_text column and updated individual columns.")

    return df


# ============================================================
# Delete / Restore (unchanged from v1)
# ============================================================

def delete_research_notes_checklist_and_reviser(sheet, worksheet_name="Final Outline"):
    """Restore the Final Outline from the hidden backup created for the research notes checklist step,
    then delete the backup sheet.
    """
    backup_ws_name = "Backup Final Outline Sheet for Delete step of Research Notes Checklist"

    # Load current Final Outline and the backup
    try:
        final_ws, _ = get_sheet_data_and_df(sheet, worksheet_name)
    except Exception:
        # If Final Outline doesn't exist yet, nothing to restore into
        print(f"⚠️ Final Outline sheet '{worksheet_name}' not found. Nothing to restore.")
        return

    try:
        backup_ws, backup_df = get_sheet_data_and_df(sheet, backup_ws_name)
        print(f"✅ Found backup sheet with {len(backup_df)} rows")
    except Exception:
        # No backup exists; nothing to do
        print(f"⚠️ No backup worksheet found for research notes checklist delete step. Skipping restore.")
        return

    # Restore: overwrite Final Outline with backup
    print(f"🔄 Restoring Final Outline from backup...")
    clear_worksheet(final_ws)
    if not backup_df.empty:
        save_to_sheet(final_ws, backup_df)

    # Delete backup sheet
    print(f"🗑️ Cleaning up backup sheet...")
    try:
        delete_worksheet(sheet, backup_ws_name)
        print(f"✅ Backup sheet '{backup_ws_name}' deleted successfully")
    except Exception as e:
        print(f"⚠️ Warning: Could not delete backup sheet: {e}")

    return
