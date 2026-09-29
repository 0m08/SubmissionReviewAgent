"""
Slide Chunks Checklist v2 - Parallel Review with Aggregation

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
    safe_get_sheet_data_and_df,
    get_worksheet_names,
)
import re
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.helper_functions import iterate_scope
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
from services.crud_text_block_tools import create_block, read_blocks, update_block, delete_block, search_web, stop, str_replace
from typing import Any, Callable, Dict, Hashable, Iterable, List, Tuple
import difflib
import services.crud_text_block_tools as crud_tools


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


def get_context_for_slice(df_slice, outline_df):
    """
    Fetch context (topic, subtopic, learning objectives, research notes) from the
    Final Outline sheet for the topics/subtopics present in the given df_slice.

    :param df_slice: DataFrame slice of slide chunks being evaluated.
    :param outline_df: DataFrame from the Final Outline sheet.
    :return: Formatted context string.
    """
    if outline_df.empty or df_slice.empty:
        return "Context unavailable."

    # Handle column name variations in slide chunks
    slice_topic_col = 'topic' if 'topic' in df_slice.columns else 'Topic'
    slice_subtopic_col = 'subtopic' if 'subtopic' in df_slice.columns else 'Subtopic'

    # Handle column name variations in outline
    outline_topic_col = 'Topic' if 'Topic' in outline_df.columns else 'topic'
    outline_subtopic_col = 'Subtopic' if 'Subtopic' in outline_df.columns else 'subtopic'
    lo_col = 'Learning Objectives' if 'Learning Objectives' in outline_df.columns else 'learning_objectives'
    rn_col = 'research_notes' if 'research_notes' in outline_df.columns else 'Research Notes'

    if lo_col not in outline_df.columns:
        return "Learning Objectives column not found in Final Outline."

    # Get unique topics and subtopics from the slice
    topics = df_slice[slice_topic_col].unique() if slice_topic_col in df_slice.columns else []
    subtopics = df_slice[slice_subtopic_col].unique() if slice_subtopic_col in df_slice.columns else []

    context_parts = []
    for topic in topics:
        if pd.isna(topic):
            continue
        for subtopic in subtopics:
            if pd.isna(subtopic):
                continue
            match = outline_df[
                (outline_df[outline_topic_col] == topic) &
                (outline_df[outline_subtopic_col] == subtopic)
            ]
            if not match.empty:
                for _, row in match.iterrows():
                    lo = str(row.get(lo_col, '')).strip() if pd.notna(row.get(lo_col)) else ''
                    rn = str(row.get(rn_col, '')).strip() if pd.notna(row.get(rn_col)) else ''
                    part = f"Topic: {topic}\nSubtopic: {subtopic}"
                    if lo:
                        part += f"\nLearning Objective: {lo}"
                    if rn:
                        part += f"\nResearch Notes: {rn}"
                    context_parts.append(part)

    if not context_parts:
        # Fallback: try to match by topic only
        for topic in topics:
            if pd.isna(topic):
                continue
            match = outline_df[outline_df[outline_topic_col] == topic]
            if not match.empty:
                for _, row in match.iterrows():
                    sub = str(row.get(outline_subtopic_col, '')).strip()
                    lo = str(row.get(lo_col, '')).strip() if pd.notna(row.get(lo_col)) else ''
                    rn = str(row.get(rn_col, '')).strip() if pd.notna(row.get(rn_col)) else ''
                    part = f"Topic: {topic}\nSubtopic: {sub}"
                    if lo:
                        part += f"\nLearning Objective: {lo}"
                    if rn:
                        part += f"\nResearch Notes: {rn}"
                    context_parts.append(part)

    if not context_parts:
        return "No matching context found in Final Outline for this slice."

    return "\n\n---\n\n".join(context_parts)


# ============================================================
# Prompts
# ============================================================

slide_chunks_checklist_prompt = """Assume the role of a checklist agent tasked with evaluating the following block(s) of slide chunks for the given list of checklist criteria.

These slide chunks are created for the following:
<course_info>
Course Name: {course_name}
Target Audience: {target_audience}
</course_info>

CONTEXT (Topic, Subtopic, Learning Objectives, and Research Notes):
The slide chunks you are evaluating should align with this context:
<context>
{context}
</context>

Here's the slide chunks to evaluate:
<slide_chunks>
{slide_chunks}
</slide_chunks>

Here are the checklist criteria to evaluate, along with examples for each:
<checklist_criteria>
{criteria_with_examples}
</checklist_criteria>

Critical Evaluation Instructions:
- Be Extremely Strict in your evaluation. Do not pass any criteria unless they are fully and completely satisfied.
- Read each review criteria text WORD BY WORD and understand exactly what it is asking for.
- Do not make assumptions or interpret criteria loosely. Follow the exact wording and requirements stated in each criteria.
- Each criteria has specific requirements that must be met - evaluate against those exact requirements, not general best practices.
- Do not let overall quality of the content influence your judgment - focus solely on whether each specific criteria requirement is met.
- Strictly evaluate every single criteria provided in the checklist. Do not skip or miss any criteria.
- Inline image links: Markdown image links in the slide content — in either `[alt](url)` or `![alt](url)` form — are intentional and handled by a dedicated upstream pipeline step. Do not flag their presence, format, placement, or alt text as a defect under any criteria. Do not ask the reviser to remove, rewrite, reformat, or reposition them.

Make sure to output in the following format:
<analysis>
[Analysis of the slide chunks based on the checklist criteria AND context alignment. It is okay for the analysis to be quite long for accurate evaluation.]
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
- Feedback of the failed items should include all necessary information for the reviser agent to understand what needs to be fixed to better align with the context.
- When listing passed or failed items, always use the complete criteria text along with the criteria name.
- Refer to the examples provided within each criteria to understand the evaluation approach.
"""


slide_chunks_reviser_prompt = """You are tasked with revising the following block(s) of slide chunks based on the feedback provided in the checklist evaluation.

These slide chunks are created for the following:
<course_info>
Course Name: {course_name}
Target Audience: {target_audience}
</course_info>

CONTEXT (Topic, Subtopic, Learning Objectives, and Research Notes):
The slide chunks should align with this context:
<context>
{context}
</context>

Here are the slide chunks to revise:
<slide_chunks>
{slide_chunks}
</slide_chunks>

Here is the detailed analysis from the review agent that evaluated the slide chunks:
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

CRITICAL FORMAT REQUIREMENTS:
When using the CRUD tools, the output you return must strictly maintain the exact format structure as of the original slide chunks that you are revising:

1. Each block must start with: "###Block ID: [number]"
2. Followed by: "####**Topic:**" section with the topic content
3. Followed by: "####**Subtopic:**" section with the subtopic content
4. Followed by: "####**Slide Chunk:**" section with the slide chunk content

Example of correct format:
###Block ID: 1
####**Topic:**
HVAC Fundamentals
####**Subtopic:**
Refrigeration Cycle
####**Slide Chunk:**
Title: Understanding the Basic Refrigeration Cycle
Slide Type: Transition
Content: The refrigeration cycle consists of four main components...

IMPORTANT: Never change or remove these headers (Block ID, Topic, Subtopic, Slide Chunk) from the block text. Only modify the content after the header as required for the revision. The format is essential for the system to function properly.

Inline image links: Any markdown image links in the slide Content — in either `[alt](url)` or `![alt](url)` form — must be preserved. Do not delete, rewrite, drop, merge, reposition, or modify the URL or alt text of any existing image link during your revisions, even if the checklist feedback asks for unrelated edits to the surrounding text. Keep each link in the exact position (same surrounding sentence) and exact format (same `[` vs `!` prefix) it currently has. Image-link placement is handled by a dedicated upstream pipeline step — treat every `[...](...)` / `![...](...)` you see as load-bearing content to preserve verbatim.

NOTES:
- Make use of the given set of CRUD block text tools to make the necessary revisions.
- If creating new slide using create_blocks, ensure to follow the correct format and maintain the headers required for revision.
- The first slide of each topic must always be a "Learning Objectives" slide. Do not create or insert any type of slide before the first "Learning Objectives" slide of a topic. Also, the last slide of the topic must be a "Topic Summary" slide.
- The first slide of each sub-topic must always be a "Transition" slide. Do not create or insert any "Content" slide before the first "Transition" slide of a sub-topic. Also, the last slide of the sub-topic must be a "Summary" slide.
- These CRUD tools allow you to create, read, update, and delete blocks of text from the above slide chunk as needed.
- You can only work with one block of text at a time. Each block of text is a separate entity identified by a unique ID - Block ID.
- To implement some of the feedback, you may need to make edits to multiple blocks of text.

Corrective Operations and Examples:
Each criteria above includes its corrective operation and an example. Match each failed criteria from the checklist feedback with its corresponding corrective operation. Only apply the corrective operations for criteria that actually failed - ignore those for criteria that passed. Use the examples as reference to understand the approach to revising the slide chunks.

IMPORTANT - Completion Requirement:
After you have completed ALL necessary revisions to address the failed checklist items, you MUST call the `stop` tool to signal that your work is complete. Provide a brief summary of what was accomplished in the reason parameter. Do not stop until all failed items have been addressed.
"""


aggregator_review_prompt = """You are an aggregator reviewer. Multiple independent review agents have evaluated the same set of slide chunks, each focusing on different quality criteria. The reviser agent will ONLY see your output — it will NOT see the individual reviews. Therefore, your report must be comprehensive and preserve all actionable detail from the original feedback.

These slide chunks are created for the following:
<course_info>
Course Name: {course_name}
Target Audience: {target_audience}
</course_info>

Here are the slide chunks that were evaluated:
<slide_chunks>
{slide_chunks}
</slide_chunks>

Here are the individual review results from each reviewer:
<individual_results>
{individual_results}
</individual_results>

Instructions:
1. Read through all individual reviews carefully.
2. Cross-reference each reviewer's findings against the actual slide chunks above to verify accuracy.
3. If two reviewers give conflicting assessments on the same block or criteria, use the slide chunks to determine which assessment is correct.
4. PRESERVE the original feedback text from each reviewer as-is. Do NOT summarize, rephrase, or shorten feedback. The individual reviewers have written feedback in a format that is easy for the reviser to follow — keep that format intact.
5. Deduplicate only when two reviewers flag the exact same issue on the exact same block. In that case, keep the more detailed feedback of the two.
6. Do NOT remove failures that were identified by a reviewer, unless another reviewer's conflicting assessment is demonstrably correct based on the actual slide chunks.
7. If while cross-referencing you spot additional issues that no reviewer caught, you may add them as new failures — but clearly mark them as "[Aggregator-identified]".
8. Inline image links: Markdown image links in the slide content (`[alt](url)` or `![alt](url)`) are intentional and handled by a dedicated upstream pipeline step. Drop any reviewer failures or feedback that ask the reviser to remove, rewrite, reposition, or reformat image links, since acting on such feedback would break the upstream placement work.

Output your consolidated report:

<analysis>
[Unified analysis combining insights from all reviewers. Organize by block or theme for clarity. Note any conflicts between reviewers and how you resolved them based on the actual slide chunks. It is okay for the analysis to be quite long for accuracy and clarity.]
</analysis>

<failed_items>
[Consolidated list of ALL failed items organized by block. The reviser works one block at a time, so group all feedback for each block together. If no items failed across any reviewer, leave this section blank.]
</failed_items>
"""


# ============================================================
# Core Agent Functions
# ============================================================

@traceable(metadata={
    "agent_name": "slide_chunks",
    "step_name": "Checklist Based Review and Revise Agents (v2)",
    "function_name": "run_slide_chunks_checklist_agent",
    "user_id": st.session_state.get("role", "anonymous"),
    "user_email": st.session_state.get("user_email", "anonymous")
})
def run_slide_chunks_checklist_agent(course_name, target_audience, context, slide_chunks, criteria_with_examples, llm="gemini_3_flash", message_history=None, blockwise_diffs=None):
    """
    Run the slide chunks checklist agent with the provided parameters.
    :param course_name: Name of the course for which the slide chunks are created.
    :param target_audience: Target audience for the course.
    :param context: Context string with topic, subtopic, LO, and research notes from Final Outline.
    :param slide_chunks: The block of slide chunks to evaluate.
    :param criteria_with_examples: The checklist criteria with their examples to evaluate the slide chunks against.
    :param llm: The language model to use for the agent.
    :param message_history: Optional list of previous messages (tuples) to continue the conversation.
    :param blockwise_diffs: Optional string containing blockwise diffs from reviser changes (for subsequent iterations).
    :return: A tuple of (analysis, failed_items, updated_message_history).
    """
    slide_chunks_checklist_agent = Chain(llm=llm, tags=["analysis", "passed_items", "failed_items"])

    # If we have message history, add all previous messages first
    if message_history:
        slide_chunks_checklist_agent.add_messages(message_history)
        # For subsequent iterations, send a follow-up message with diffs
        follow_up_content = f"""The reviser agent has made changes based on your previous feedback. Here are the CHANGES made (blockwise diffs):

<blockwise_diffs>
{blockwise_diffs}
</blockwise_diffs>

Please re-evaluate the slide chunks based on these changes. Check if the previous issues have been addressed and identify any remaining or new issues.

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
        slide_chunks_checklist_agent.add_message(role="user", content=follow_up_content)
    else:
        # First iteration - send the full initial prompt
        slide_chunks_checklist_agent.add_message(
            role="user",
            content=slide_chunks_checklist_prompt.format(
                course_name=course_name,
                target_audience=target_audience,
                context=context,
                slide_chunks=slide_chunks,
                criteria_with_examples=criteria_with_examples,
            )
        )

    response = slide_chunks_checklist_agent.run()

    # Return updated message history for subsequent iterations
    updated_history = slide_chunks_checklist_agent.messages_list

    return response["analysis"], response["failed_items"], updated_history


@traceable(metadata={
    "agent_name": "slide_chunks",
    "step_name": "Checklist Based Review and Revise Agents (v2)",
    "function_name": "run_reviser_agent",
    "user_id": st.session_state.get("role", "anonymous"),
    "user_email": st.session_state.get("user_email", "anonymous")
})
def run_reviser_agent(slide_chunks, review_analysis, checklist_feedback, criteria_with_ops_and_examples, df, course_name, target_audience, context, tools=None, llm="gemini_3_flash", message_history=None, blockwise_diffs=None):
    """
    Run the reviser agent with the provided parameters.
    :param slide_chunks: The block of slide chunks to revise.
    :param review_analysis: The detailed analysis from the review agent explaining evaluation reasoning.
    :param checklist_feedback: The feedback from the checklist evaluation to consider for revisions.
    :param criteria_with_ops_and_examples: All criteria with their corrective operations and examples for the reviser to reference.
    :param df: The dataframe containing the slide chunks.
    :param course_name: Name of the course for which the slide chunks are created.
    :param target_audience: Target audience for the course.
    :param context: Context string with topic, subtopic, LO, and research notes from Final Outline.
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
        requests_per_second=1,
        check_every_n_seconds=0.1,
        max_bucket_size=10,
    )

    from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        thinking_level="high",
        include_thoughts=True,
        rate_limiter=rate_limiter,
        max_retries=20,
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
        follow_up_content = f"""The reviewer has re-evaluated the slide chunks after your previous revisions. Here's what you changed:

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
        messages = [HumanMessage(content=slide_chunks_reviser_prompt.format(
            slide_chunks=slide_chunks,
            review_analysis=review_analysis,
            checklist_feedback=checklist_feedback,
            criteria_with_ops_and_examples=criteria_with_ops_and_examples,
            course_name=course_name,
            target_audience=target_audience,
            context=context,))]

    state = {
        "messages": messages,
        "df": df,
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
# Parallel Review & Aggregation Functions
# ============================================================

@traceable(metadata={
    "agent_name": "slide_chunks",
    "step_name": "Checklist Based Review and Revise Agents (v2)",
    "function_name": "_review_single_slice",
    "user_id": st.session_state.get("role", "anonymous"),
    "user_email": st.session_state.get("user_email", "anonymous")
})
def _review_single_slice(
    key, df_slice, course_name, target_audience,
    context, criteria_with_examples_str,
    reviewer_llm, slice_history, blockwise_diffs
):
    """
    Review a single scope slice. Designed to be called from a thread pool.

    :param key: The scope slice key (e.g., topic name, row index, "ALL")
    :param df_slice: DataFrame slice for this scope
    :param course_name: Name of the course
    :param target_audience: Target audience
    :param context: Context string with topic/subtopic/LO/research notes
    :param criteria_with_examples_str: Criteria with examples for the review agent
    :param reviewer_llm: LLM to use for review
    :param slice_history: Message history for this slice from previous iterations
    :param blockwise_diffs: Diffs from previous revision (for re-review context)
    :return: Tuple of (key, analysis, failed_items, history)
    """
    slide_chunks_str = "\n\n---\n\n".join(df_slice["block text"].tolist())

    analysis, failed_items, history = run_slide_chunks_checklist_agent(
        course_name=course_name,
        target_audience=target_audience,
        context=context,
        slide_chunks=slide_chunks_str,
        criteria_with_examples=criteria_with_examples_str,
        llm=reviewer_llm,
        message_history=slice_history,
        blockwise_diffs=blockwise_diffs,
    )

    return key, analysis, failed_items, history


@traceable(metadata={
    "agent_name": "slide_chunks",
    "step_name": "Checklist Based Review and Revise Agents (v2)",
    "function_name": "run_single_task_review",
    "user_id": st.session_state.get("role", "anonymous"),
    "user_email": st.session_state.get("user_email", "anonymous")
})
def run_single_task_review(
    task_key, scope, scope_to_selector, slide_chunks_df,
    criteria_with_examples_str, course_name, target_audience,
    outline_df, reviewer_llm="gemini_3_flash", reviewer_histories=None, blockwise_diffs=None
):
    """
    Run review for a single task across all its scope slices.
    Each task may slice the data differently based on its scope (Global, Topic, etc.).
    Scope slices are processed in parallel using a thread pool.

    :param task_key: String identifier for this task (e.g., "Task Name (Scope)")
    :param scope: The scope level for slicing (Global (full output), Topic, Subtopic, Learning Objective)
    :param scope_to_selector: Mapping from scope to DataFrame selector
    :param slide_chunks_df: The full DataFrame of slide chunks
    :param criteria_with_examples_str: Criteria with examples for the review agent
    :param course_name: Name of the course
    :param target_audience: Target audience
    :param outline_df: DataFrame from the Final Outline sheet for context
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
        for key, df_slice in iterate_scope(scope, slide_chunks_df, scope_to_selector)
        if not df_slice.empty
    ]

    if not slices:
        print(f"⚠️ No non-empty slices for task '{task_key}'")
        return task_key, "No analysis generated.", "", {}

    print(f"🚀 Launching {len(slices)} parallel scope-slice reviews for task '{task_key}'...")

    # Run all scope slices in parallel, preserving original order
    slice_results = [None] * len(slices)
    with ThreadPoolExecutor(max_workers=min(5, len(slices))) as executor:
        futures_map = {}
        for idx, (key, df_slice) in enumerate(slices):
            slice_history = reviewer_histories.get(key)
            # Get context for this specific slice
            context = get_context_for_slice(df_slice, outline_df)
            future = executor.submit(
                _review_single_slice,
                key=key,
                df_slice=df_slice,
                course_name=course_name,
                target_audience=target_audience,
                context=context,
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
    "agent_name": "slide_chunks",
    "step_name": "Checklist Based Review and Revise Agents (v2)",
    "function_name": "run_all_task_reviews_parallel",
    "user_id": st.session_state.get("role", "anonymous"),
    "user_email": st.session_state.get("user_email", "anonymous")
})
def run_all_task_reviews_parallel(
    task_configs, slide_chunks_df, scope_to_selector,
    course_name, target_audience, outline_df,
    reviewer_llm="gemini_3_flash", all_reviewer_histories=None, blockwise_diffs=None
):
    """
    Run all task reviews in parallel using a thread pool.

    :param task_configs: List of dicts, each with keys: task_key, scope, criteria_with_examples_str
    :param slide_chunks_df: The full DataFrame of slide chunks
    :param scope_to_selector: Mapping from scope to DataFrame selector
    :param course_name: Name of the course
    :param target_audience: Target audience
    :param outline_df: DataFrame from the Final Outline sheet for context
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
                slide_chunks_df=slide_chunks_df,
                criteria_with_examples_str=config['criteria_with_examples_str'],
                course_name=course_name,
                target_audience=target_audience,
                outline_df=outline_df,
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
    "agent_name": "slide_chunks",
    "step_name": "Checklist Based Review and Revise Agents (v2)",
    "function_name": "aggregate_review_results",
    "user_id": st.session_state.get("role", "anonymous"),
    "user_email": st.session_state.get("user_email", "anonymous")
})
def aggregate_review_results(review_results, slide_chunks_df, course_name, target_audience, llm="gemini_3_flash"):
    """
    Aggregate results from multiple parallel review tasks into a single unified report.
    Uses an LLM to intelligently merge and deduplicate findings from different reviewers.

    Short-circuits:
    - If no failures across all tasks, returns combined analysis with empty failed_items.
    - If only one task produced results, returns that task's results directly.

    :param review_results: List of (task_key, analysis, failed_items, histories) tuples
    :param slide_chunks_df: The full DataFrame of slide chunks (for context)
    :param course_name: Name of the course
    :param target_audience: Target audience for the course
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

    # Build slide chunks string for context
    slide_chunks_str = "\n\n---\n\n".join(slide_chunks_df["block text"].tolist())

    # Use LLM to aggregate
    aggregator = Chain(llm=llm, tags=["analysis", "failed_items"])
    aggregator.add_message(
        role="user",
        content=aggregator_review_prompt.format(
            course_name=course_name,
            target_audience=target_audience,
            slide_chunks=slide_chunks_str,
            individual_results=individual_results_str,
        )
    )
    response = aggregator.run()

    print("✅ Aggregator: LLM aggregation complete")
    return response["analysis"], response["failed_items"]


# ============================================================
# Main Orchestration (v2 parallel flow)
# ============================================================

@traceable(metadata={
    "agent_name": "slide_chunks",
    "step_name": "Checklist Based Review and Revise Agents (v2)",
    "function_name": "run_slide_chunks_checklist_and_reviser",
    "user_id": st.session_state.get("role", "anonymous"),
    "user_email": st.session_state.get("user_email", "anonymous")
})
def run_slide_chunks_checklist_and_reviser(sheet, course_name, target_audience, checklist_sheet_link, gc, llm="gemini_3_flash", reviewer_llm="gemini_3_flash"):
    """
    Run the slide chunks checklist and reviser pipeline using v2 parallel architecture.

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
    checklist_worksheet, checklist_df = get_sheet_data_and_df(sheet=checklist_sheet, sheet_name="Slide Chunks Checklist")

    # ── Load slide chunks sheet ──
    slide_chunks_sheet, slide_chunks_df = get_sheet_data_and_df(sheet=sheet, sheet_name="Slide Chunks")

    # ── Load Final Outline for context ──
    _, outline_df = safe_get_sheet_data_and_df(sheet, "Final Outline")
    if outline_df is None or outline_df.empty:
        print("⚠️ Final Outline sheet not found or empty. Context will be limited.")
        outline_df = pd.DataFrame()

    # ── Create backup for delete step restore ──
    backup_ws_name = "Backup Slide Chunks Sheet for Delete step of Slide Chunks Checklist"
    print(f"📋 Creating backup sheet '{backup_ws_name}' for delete step functionality...")
    backup_ws, _ = create_or_read_worksheet(sheet, backup_ws_name)
    clear_worksheet(backup_ws)
    save_to_sheet(backup_ws, slide_chunks_df)
    format_worksheet(backup_ws)
    try:
        hide_worksheet_by_name(sheet, backup_ws_name)
        print(f"✅ Backup sheet created and hidden successfully")
    except Exception as e:
        print(f"⚠️ Warning: Could not hide backup sheet: {e}")

    # ── Prepare DataFrame ──
    if "order" not in slide_chunks_df.columns:
        slide_chunks_df["order"] = slide_chunks_df.index.astype(float) + 1

    if 'block text' not in slide_chunks_df.columns:
        slide_chunks_df['block text'] = ''

    for index, row in slide_chunks_df.iterrows():
        if pd.isna(slide_chunks_df.at[index, 'block text']) or slide_chunks_df.at[index, 'block text'] == '':
            slide_chunks_str = ""
            slide_chunks_str += f"###Block ID: {index}\n"
            slide_chunks_str += f"####**Topic:**\n{row.get('topic', row.get('Topic', ''))}\n"
            slide_chunks_str += f"####**Subtopic:**\n{row.get('subtopic', row.get('Subtopic', ''))}\n"

            slide_type = row.get('Slide Type', row.get('slide_type', ''))
            slide_chunk_title = row.get('Slide Chunk Title', row.get('slide_chunk_title', ''))
            slide_chunk = row.get('Slide Chunk', row.get('slide_chunk', ''))

            slide_content = ""
            if slide_type:
                slide_content += f"Slide Type: {slide_type}\n"
            if slide_chunk_title:
                slide_content += f"Title: {slide_chunk_title}\n"
            if slide_chunk:
                slide_content += f"Content: {slide_chunk}\n"
            slide_chunks_str += f"####**Slide Chunk:**\n{slide_content}\n"

            slide_chunks_df.at[index, 'block text'] = slide_chunks_str

    scope_to_selector = {
        "Global (full output)": [],
        "Topic": ["topic"] if "topic" in slide_chunks_df.columns else ["Topic"],
        "Subtopic": (["topic", "subtopic"] if "topic" in slide_chunks_df.columns else ["Topic", "Subtopic"]),
        "Learning Objective": "__row__",
    }

    # ── Prepare all task configs upfront ──
    task_scope_groups = list(checklist_df.groupby(["Task", "Scope"], sort=False))

    task_configs = []
    all_criteria_with_ops_strs = []
    all_reviser_tools = {}  # {tool.name: tool} - dict to deduplicate
    global_max_iterations = 1

    print(f"\n{'='*80}")
    print(f"📊 Preparing {len(task_scope_groups)} task configs for parallel review")
    print(f"{'='*80}")

    for (task, scope), grp in task_scope_groups:
        print(f"  Task: {task}, Scope: {scope}")

        # Extract criteria data
        criteria_list = grp["Review Criteria"].tolist()
        criteria_names = grp["Criteria Name"].tolist() if "Criteria Name" in grp.columns else [f"Criteria {i+1}" for i in range(len(criteria_list))]
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

    # ── Topic-wise Parallel Review → Aggregate → Revise Loop ──
    topic_col = 'topic' if 'topic' in slide_chunks_df.columns else 'Topic'
    unique_topics = slide_chunks_df[topic_col].unique().tolist()

    print(f"\n📊 Processing {len(unique_topics)} topics in parallel for topic-wise review-aggregate-revise pipeline")

    def _process_single_topic(topic, topic_idx, total_topics):
        """
        Run the full review → aggregate → revise loop for a single topic.
        Designed to be called from a thread pool.

        :param topic: The topic name to process.
        :param topic_idx: 1-based index of this topic (for logging).
        :param total_topics: Total number of topics (for logging).
        :return: The revised DataFrame for this topic.
        """
        topic_df = slide_chunks_df[slide_chunks_df[topic_col] == topic].copy()

        print(f"\n{'='*80}")
        print(f"📚 TOPIC {topic_idx}/{total_topics}: {topic} ({len(topic_df)} blocks)")
        print(f"{'='*80}")

        topic_reviewer_histories = {}
        topic_reviser_history = None
        topic_last_diffs = None

        for iteration in range(1, global_max_iterations + 1):
            print(f"\n  🔄 [{topic}] Iteration {iteration}/{global_max_iterations}")

            # ── Phase 1: Run all task reviews in parallel for this topic ──
            print(f"  📋 [{topic}] Phase 1: Parallel reviews for {len(task_configs)} tasks...")
            review_results = run_all_task_reviews_parallel(
                task_configs=task_configs,
                slide_chunks_df=topic_df,
                scope_to_selector=scope_to_selector,
                course_name=course_name,
                target_audience=target_audience,
                outline_df=outline_df,
                reviewer_llm=reviewer_llm,
                all_reviewer_histories=topic_reviewer_histories,
                blockwise_diffs=topic_last_diffs,
            )

            # Update reviewer histories for next iteration
            for task_key, _, _, histories in review_results:
                topic_reviewer_histories[task_key] = histories

            # ── Phase 2: Aggregate review results for this topic ──
            print(f"  🔗 [{topic}] Phase 2: Aggregating {len(review_results)} review results...")
            aggregated_analysis, aggregated_failed = aggregate_review_results(
                review_results,
                slide_chunks_df=topic_df,
                course_name=course_name,
                target_audience=target_audience,
                llm=reviewer_llm,
            )

            print(f"  📝 [{topic}] Aggregated failed items length: {len(aggregated_failed)} chars")
            if aggregated_failed:
                print(f"  📝 [{topic}] Aggregated failed items preview: {aggregated_failed[:300]}...")

            # ── Phase 3: Check if all criteria passed ──
            if is_failed_items_empty(aggregated_failed):
                print(f"\n  ✅ [{topic}] All criteria passed at iteration {iteration} - exiting early")
                break

            # ── Phase 4: Run single reviser with aggregated feedback for this topic ──
            print(f"  🔧 [{topic}] Phase 4: Running reviser with aggregated feedback...")
            df_before_revision = topic_df.copy()
            slide_chunks_str = "\n\n---\n\n".join(topic_df["block text"].tolist())

            # Get context for this topic
            topic_context = get_context_for_slice(topic_df, outline_df)

            topic_df, topic_reviser_history = run_reviser_agent(
                slide_chunks=slide_chunks_str,
                review_analysis=aggregated_analysis,
                checklist_feedback=aggregated_failed,
                criteria_with_ops_and_examples=combined_criteria_with_ops,
                df=topic_df,
                course_name=course_name,
                target_audience=target_audience,
                context=topic_context,
                tools=combined_reviser_tools,
                llm=llm,
                message_history=topic_reviser_history,
                blockwise_diffs=topic_last_diffs,
            )

            # ── Phase 5: Compute diffs for next iteration ──
            topic_last_diffs = compute_blockwise_diffs(df_before_revision, topic_df)
            print(f"  📊 [{topic}] Iteration {iteration} complete. DataFrame shape: {topic_df.shape}")
            if topic_last_diffs != "No changes detected.":
                print(f"  📝 [{topic}] Diffs preview: {topic_last_diffs[:300]}...")

        return topic_df

    # Run all topics in parallel
    total_topics = len(unique_topics)
    revised_topic_dfs = [None] * total_topics

    with ThreadPoolExecutor(max_workers=min(5, total_topics)) as executor:
        futures_map = {}
        for idx, topic in enumerate(unique_topics):
            future = executor.submit(
                _process_single_topic,
                topic=topic,
                topic_idx=idx + 1,
                total_topics=total_topics,
            )
            futures_map[future] = (idx, topic)

        for future in as_completed(futures_map):
            idx, topic = futures_map[future]
            try:
                revised_topic_dfs[idx] = future.result()
                print(f"✅ Topic '{topic}' processing completed")
            except Exception as e:
                print(f"❌ Error processing topic '{topic}': {e}")
                # Fall back to the original unrevised data for this topic
                revised_topic_dfs[idx] = slide_chunks_df[slide_chunks_df[topic_col] == topic].copy()

    # Combine all revised topic DataFrames back together
    slide_chunks_df = pd.concat(revised_topic_dfs)

    # ── Post-processing: Validate and save ──
    print("\n🔍 Validating and correcting block text format...")
    corrected_count = 0
    for index, row in slide_chunks_df.iterrows():
        block_text = str(row.get('block text', '')).strip()

        has_correct_format = (
            'Block ID' in block_text or
            '####**Topic:**' in block_text
        )

        if not has_correct_format and block_text != '':
            slide_chunks_str = ""
            slide_chunks_str += f"###Block ID: {index}\n"
            slide_chunks_str += f"####**Topic:**\n{row.get('topic', row.get('Topic', ''))}\n"
            slide_chunks_str += f"####**Subtopic:**\n{row.get('subtopic', row.get('Subtopic', ''))}\n"
            slide_chunks_str += f"####**Slide Chunk:**\n{block_text}\n"

            slide_chunks_df.loc[index, 'block text'] = slide_chunks_str
            corrected_count += 1
            print(f"✅ Row {index} corrected successfully")
        elif not has_correct_format and block_text == '':
            print(f"⚠️ Row {index}: Empty block text, skipping")
        else:
            print(f"✅ Row {index}: Format is correct, no action needed")

    print(f"✅ Corrected format for {corrected_count} rows")

    # Order DataFrame by 'order' column before parsing
    if 'order' in slide_chunks_df.columns:
        slide_chunks_df = slide_chunks_df.sort_values('order')
        print("Ordered DataFrame by 'order' column before parsing")

    # Clear the worksheet first to handle row deletions properly
    clear_worksheet(slide_chunks_sheet)

    # Save to sheet
    save_to_sheet(worksheet=slide_chunks_sheet, df=slide_chunks_df)

    # Parse the updated block_text column back to individual columns
    print("\n🔄 Parsing updated block_text content back to individual columns...")
    slide_chunks_df = parse_block_text_to_columns(slide_chunks_df)

    # Parse slide_chunks into individual slide components
    print("\n🔄 Parsing slide_chunks into individual slide components...")
    slide_chunks_df = parse_slide_chunks_to_components(slide_chunks_df)

    # Delete the slide_chunks column after parsing is complete
    if 'slide_chunks' in slide_chunks_df.columns:
        slide_chunks_df = slide_chunks_df.drop(columns=['slide_chunks'])

    save_to_sheet(worksheet=slide_chunks_sheet, df=slide_chunks_df)

    # Hide the utility columns that users don't need to see
    columns_to_hide = ['order', 'block text']
    try:
        hide_columns_by_name(slide_chunks_sheet, columns_to_hide, slide_chunks_df)
        print(f"👁️ Hidden columns: {', '.join(columns_to_hide)}")
    except Exception as e:
        print(f"⚠️ Could not hide columns: {e}")

    print("✅ Block text parsing completed. Individual columns updated with revised content.")
    print("✅ Slide chunks parsing completed. Slide Type, Slide Chunk Title, and Slide Chunk columns updated.")

    # Reset global block ID counter for next run
    crud_tools._global_block_id_counter = None

    return


# ============================================================
# Block Text Parsing
# ============================================================

class BlockTextContent(BaseModel):
    topic: str = Field(description="The topic extracted from the block text.")
    subtopic: str = Field(description="The subtopic extracted from the block text.")
    slide_chunks: str = Field(description="The slide chunks content extracted from the block text.")

block_text_parsing_prompt = """You are an expert parser for slide chunks block text. Given a block of text in a structured format, extract the following fields:

- topic: The value after '####**Topic:**' (extract only the topic text, not the header)
- subtopic: The value after '####**Subtopic:**' (extract only the subtopic text, not the header)
- slide_chunks: The value after '####**Slide Chunk:**' or '####**Slide Chunks:**' (extract only the slide chunks content, not the header)

Return the output as a structured object with these fields. Do not add or infer any information. Only extract what is present in the block. Preserve all formatting, line breaks, and structure in the slide_chunks field.

Block:
{block}
"""


@traceable(metadata={
    "agent_name": "slide_chunks",
    "step_name": "Checklist Based Review and Revise Agents (v2)",
    "function_name": "parse_block_text_row",
    "user_id": st.session_state.get("role", "anonymous"),
    "user_email": st.session_state.get("user_email", "anonymous")
})
def parse_block_text_row(block_text_cell, index):
    """
    Parses a single block_text cell to extract Topic, Subtopic, and Slide Chunks.
    :param block_text_cell: The raw text from the block_text column.
    :param index: The row index (for debugging).
    :return: Dict with parsed content or None if parsing fails.
    """
    try:
        # Use LLM + Pydantic to parse and validate
        agent = Chain(llm="gemini_3_flash")
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
    "agent_name": "slide_chunks",
    "step_name": "Checklist Based Review and Revise Agents (v2)",
    "function_name": "parse_block_text_to_columns",
    "user_id": st.session_state.get("role", "anonymous"),
    "user_email": st.session_state.get("user_email", "anonymous")
})
def parse_block_text_to_columns(df, max_workers=5):
    """
    Parses the block_text column and updates the individual Topic, Subtopic, and slide_chunks columns.
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
                    df.loc[index, 'slide_chunks'] = parsed_content['slide_chunks']
                    print(f"Updated row {index} with parsed content")
            except Exception as e:
                print(f"Error parsing row {index}: {e}")
            progress.update()

    print(f"Successfully parsed block_text column and updated individual columns.")

    return df


def parse_slide_chunks_to_components(df, max_workers=5):
    """
    Parse the slide_chunks column into individual slide components: Slide Type, Slide Chunk Title, and Slide Chunk
    :param df: The DataFrame to update.
    :param max_workers: Number of parallel workers (default 5).
    :return: The updated DataFrame.
    """
    print("🔄 Starting slide chunks component parsing...")

    # Add the target columns if they don't exist
    target_columns = ['Slide Type', 'Slide Chunk Title', 'Slide Chunk']
    for col in target_columns:
        if col not in df.columns:
            df[col] = ''

    # Check if slide_chunks column exists
    if 'slide_chunks' not in df.columns:
        print("No 'slide_chunks' column found. Nothing to parse.")
        return df

    def parse_single_slide_chunk(slide_chunks_content, index):
        """Parse a single slide_chunks content into components"""
        try:
            if pd.isna(slide_chunks_content) or str(slide_chunks_content).strip() == '':
                return {'Slide Type': '', 'Slide Chunk Title': '', 'Slide Chunk': ''}

            content = str(slide_chunks_content).strip()

            # Initialize result
            result = {'Slide Type': '', 'Slide Chunk Title': '', 'Slide Chunk': ''}

            # Split content by lines for parsing
            lines = content.split('\n')
            current_section = None
            current_content = []

            for line in lines:
                line = line.strip()

                # Check for slide type indicators
                if line.lower().startswith('slide type:') or line.lower().startswith('type:'):
                    result['Slide Type'] = line.split(':', 1)[1].strip() if ':' in line else ''
                    current_section = 'type'

                # Check for title indicators
                elif line.lower().startswith('title:') or line.lower().startswith('slide title:'):
                    result['Slide Chunk Title'] = line.split(':', 1)[1].strip() if ':' in line else ''
                    current_section = 'title'

                # Check for content indicators
                elif line.lower().startswith('content:') or line.lower().startswith('slide content:'):
                    content_text = line.split(':', 1)[1].strip() if ':' in line else ''
                    current_content = [content_text] if content_text else []
                    current_section = 'content'

                # Continue collecting content
                elif current_section == 'content' and line:
                    current_content.append(line)

                # If no specific section, try to extract patterns
                elif not current_section and line:
                    if ':' in line:
                        key_part = line.split(':', 1)[0].strip().lower()
                        value_part = line.split(':', 1)[1].strip()

                        if 'type' in key_part:
                            result['Slide Type'] = value_part
                        elif 'title' in key_part:
                            result['Slide Chunk Title'] = value_part
                        elif 'content' in key_part:
                            current_content = [value_part] if value_part else []
                            current_section = 'content'

            # Join content lines
            if current_content:
                result['Slide Chunk'] = '\n'.join(current_content)

            # If no structured format found, put everything in Slide Chunk
            if not any(result.values()) and content:
                result['Slide Chunk'] = content

            return result

        except Exception as e:
            print(f"Error parsing slide chunks for row {index}: {e}")
            return {'Slide Type': '', 'Slide Chunk Title': '', 'Slide Chunk': str(slide_chunks_content)}

    # Process each row
    total_rows = len(df)
    progress = SmartProgressBar(total_tasks=total_rows, description="Parsing slide components", save_interval=5)

    for index, row in df.iterrows():
        slide_chunks_content = row.get('slide_chunks', '')
        parsed_components = parse_single_slide_chunk(slide_chunks_content, index)

        # Update the dataframe
        for col, value in parsed_components.items():
            df.loc[index, col] = value

        progress.update()

    print(f"✅ Successfully parsed slide_chunks into individual components for {total_rows} rows.")
    return df


# ============================================================
# Delete / Restore
# ============================================================

def delete_slide_chunks_checklist(sheet):
    """
    Delete function for the Slide Chunks Checklist step.
    Restores the 'Slide Chunks' sheet from backup and deletes the backup.
    """
    backup_ws_name = "Backup Slide Chunks Sheet for Delete step of Slide Chunks Checklist"

    # Load current Slide Chunks and the backup
    try:
        slide_chunks_ws, _ = get_sheet_data_and_df(sheet, "Slide Chunks")
    except Exception:
        print(f"⚠️ Slide Chunks sheet not found. Nothing to restore.")
        return

    try:
        backup_ws, backup_df = get_sheet_data_and_df(sheet, backup_ws_name)
        print(f"✅ Found backup sheet with {len(backup_df)} rows")
    except Exception:
        print(f"⚠️ No backup worksheet found for slide chunks checklist delete step. Skipping restore.")
        return

    # Restore: overwrite Slide Chunks with backup
    print(f"🔄 Restoring Slide Chunks from backup...")
    clear_worksheet(slide_chunks_ws)
    if not backup_df.empty:
        save_to_sheet(slide_chunks_ws, backup_df)

    # Delete backup sheet
    print(f"🗑️ Cleaning up backup sheet...")
    try:
        delete_worksheet(sheet, backup_ws_name)
        print(f"✅ Backup sheet '{backup_ws_name}' deleted successfully")
    except Exception as e:
        print(f"⚠️ Warning: Could not delete backup sheet: {e}")

    return
