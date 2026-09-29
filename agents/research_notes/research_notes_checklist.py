from modules.chain import Chain
from tqdm import tqdm
from services.sheets_service import (
    get_sheet_data_and_df,
    save_to_sheet,
    filter_non_blank_column,
    clear_worksheet,
    delete_worksheet,
    clear_all_filters,
    format_worksheet,
    create_or_read_worksheet,
    hide_worksheet_by_name,
    hide_columns_by_name,
)
import json
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from services.helper_functions import get_outline_with_los, validate_column_values
from services.smart_progress_bar import SmartProgressBar
from langsmith import traceable

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage, HumanMessage

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain.chat_models import init_chat_model


from pydantic import Field
# from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langchain.agents import AgentState
import pandas as pd
from services.crud_text_block_tools import create_block, read_blocks, update_block, delete_block, search_web, stop, str_replace
from services.helper_functions import iterate_scope

# Mapping from checklist tool names to actual tool functions
TOOL_NAME_TO_FUNCTION = {
    "Create": create_block,
    "Read": read_blocks,
    "Update": update_block,
    "Delete": delete_block,
    "Search": search_web,
    "Stop": stop,
    "StrReplace": str_replace,
}

import pandas as pd
from typing import Any, Callable, Dict, Hashable, Iterable, List, Tuple
from pydantic import BaseModel, Field
import difflib


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


@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Checklist Based Review and Revise Agents",
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


@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Checklist Based Review and Revise Agents",
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

    # llm = init_chat_model(
    #     # "google_genai:gemini-2.5-flash",
    #     "openai:gpt-5-mini",
    #     rate_limiter = rate_limiter,
    #     max_retries = 20,
    #     reasoning_effort="high",
    # )

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


@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Checklist Based Review and Revise Agents",
    "function_name": "process_single_rn_slice",
    "user_id": st.session_state.get("role", "anonymous")
})
def process_single_rn_slice(key, df_slice, criteria_with_examples_str, criteria_with_ops_and_examples_str, course_name, target_audience, course_objective_guidelines, course_background, tools, llm, max_iterations=1, reviewer_llm = "gemini_3_flash"):
    """
    Process a single research notes slice: review → revise (if needed) → return result
    This function runs in parallel for scope-based processing.
    Supports multiple review-revise iterations until all criteria pass or max iterations reached.
    Message history is maintained across iterations for context continuity.

    :param key: The key identifying this slice (from iterate_scope)
    :param df_slice: The DataFrame slice to process
    :param criteria_with_examples_str: Formatted criteria with examples string for review agent
    :param criteria_with_ops_and_examples_str: Formatted criteria with operations and examples for reviser agent
    :param course_name: Name of the course
    :param target_audience: Target audience for the course
    :param course_objective_guidelines: Guidelines for course objectives
    :param course_background: Background information about the course
    :param tools: List of tool functions to make available to the reviser agent
    :param llm: The language model to use
    :param max_iterations: Maximum number of review-revise cycles (default 1)
    :return: The revised DataFrame slice (or original if no changes needed)
    """
    print(f"🔄 Processing slice '{key}' with {len(df_slice)} rows (max {max_iterations} iterations)")

    if df_slice.empty:
        print(f"Empty slice '{key}', returning original")
        return df_slice

    # Initialize message histories for both agents
    reviewer_history = None
    reviser_history = None
    # Track diffs from previous revision for context
    last_revision_diffs = None

    for iteration in range(1, max_iterations + 1):
        print(f"🔄 Iteration {iteration}/{max_iterations} for slice '{key}'")

        # Build research notes string from current slice
        research_notes_str = "\n\n---\n\n".join(df_slice["block text"].tolist())

        # Run the research notes checklist agent (with history and diffs for continuity)
        analysis, failed_items, reviewer_history = run_research_notes_checklist_agent(
            course_name=course_name,
            target_audience=target_audience,
            course_objective_guidelines=course_objective_guidelines,
            course_background=course_background,
            research_notes=research_notes_str,
            criteria_with_examples=criteria_with_examples_str,
            llm=reviewer_llm,
            message_history=reviewer_history,
            blockwise_diffs=last_revision_diffs
        )

        print(f"Failed items for slice '{key}' (iteration {iteration}): {failed_items}")

        if is_failed_items_empty(failed_items):
            print(f"✅ All criteria passed at iteration {iteration} for slice '{key}'")
            break  # Exit early - all passed

        # Save state before revision for diff computation
        df_before_revision = df_slice.copy()

        # Run the reviser agent with the failed items and analysis (with history and diffs for continuity)
        df_slice, reviser_history = run_reviser_agent(
            research_notes=research_notes_str,
            review_analysis=analysis,
            checklist_feedback=failed_items,
            criteria_with_ops_and_examples=criteria_with_ops_and_examples_str,
            df=df_slice,
            course_name=course_name,
            target_audience=target_audience,
            course_objective_guidelines=course_objective_guidelines,
            course_background=course_background,
            tools=tools,
            llm=llm,
            message_history=reviser_history,
            blockwise_diffs=last_revision_diffs
        )

        # Compute diffs for next iteration
        last_revision_diffs = compute_blockwise_diffs(df_before_revision, df_slice)
        print(f"Revised research notes for slice '{key}' (iteration {iteration}): shape {df_slice.shape}")

    return df_slice


@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Checklist Based Review and Revise Agents",
    "function_name": "process_rn_scope_slices_parallel",
    "user_id": st.session_state.get("role", "anonymous")
})
def process_rn_scope_slices_parallel(scope, research_notes_df, scope_to_selector, criteria_with_examples_str, criteria_with_ops_and_examples_str, course_name, target_audience, course_objective_guidelines, course_background, tools, llm, max_iterations=1):
    """
    Process all research notes slices within a scope in parallel.

    :param scope: The scope to process (Topic, Subtopic, Learning Objective)
    :param research_notes_df: The main DataFrame
    :param scope_to_selector: Scope selector mapping
    :param criteria_with_examples_str: Formatted criteria with examples string for review agent
    :param criteria_with_ops_and_examples_str: Formatted criteria with operations and examples for reviser agent
    :param course_name: Name of the course
    :param target_audience: Target audience for the course
    :param course_objective_guidelines: Guidelines for course objectives
    :param course_background: Background information about the course
    :param tools: List of tool functions to make available to the reviser agent
    :param llm: The language model to use
    :param max_iterations: Maximum number of review-revise cycles per slice (default 1)
    :return: List of (original_slice, revised_slice) tuples
    """
    print(f"🚀 Starting parallel processing for scope: {scope}")
    
    # Collect all slices for this scope
    all_slices = list(iterate_scope(scope, research_notes_df, scope_to_selector))
    
    if not all_slices:
        print(f"No slices found for scope {scope}")
        return []
    
    print(f"Found {len(all_slices)} slices for scope {scope}")
    
    # Prepare for parallel processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        
        # Submit all slices for parallel processing
        for key, df_slice in all_slices:
            if df_slice.empty:
                print(f"Skipping empty slice '{key}'")
                continue
                
            future = executor.submit(
                process_single_rn_slice,
                key, df_slice, criteria_with_examples_str,
                criteria_with_ops_and_examples_str,
                course_name, target_audience, course_objective_guidelines,
                course_background, tools, llm, max_iterations
            )
            futures_map[future] = (key, df_slice)
        
        # Collect results
        total_tasks = len(futures_map)
        
        if total_tasks == 0:
            print(f"No valid slices to process for scope {scope}")
            return []
            
        print(f"Processing {total_tasks} slices in parallel for scope {scope}")
        
        results = []
        for future in as_completed(futures_map):
            key, original_slice = futures_map[future]
            try:
                revised_slice = future.result()
                results.append((original_slice, revised_slice))
                print(f"✅ Completed processing slice '{key}'")
            except Exception as e:
                print(f"❌ Error processing slice '{key}': {e}")
                # In case of error, use original slice
                results.append((original_slice, original_slice))
    
    print(f"🏁 Completed parallel processing for scope {scope}: {len(results)} results")
    return results


@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Checklist Based Review and Revise Agents",
    "function_name": "run_research_notes_checklist_and_reviser",
    "user_id": st.session_state.get("role", "anonymous")
})
def run_research_notes_checklist_and_reviser(sheet, course_name, target_audience, checklist_sheet_link, gc, llm = "gemini_3_flash", reviewer_llm = "gemini_3_flash"):
    """
    Run the research notes checklist agent and reviser agent with the provided parameters.
    :param course_name: Name of the course for which the research notes are created.
    :param target_audience: Target audience for the course.
    :param checklist_sheet_link: Link to the Google Sheet containing the checklist.
    :param gc: Google Sheets client instance.
    :param llm: The language model to use for the agents.
    :return: None
    """

    # Load the checklist sheet
    # checklist_sheet_link = st.session_state["checklist_sheet_link"]
    # gc = st.session_state["gc"]
    checklist_sheet = gc.open_by_url(checklist_sheet_link)
    checklist_worksheet, checklist_df = get_sheet_data_and_df(sheet = checklist_sheet, sheet_name = "Research Notes Checklist")

    # Load the Course info sheet to get course objective guidelines and course background
    course_info_worksheet, course_info_df = get_sheet_data_and_df(sheet = sheet, sheet_name = "Course info")
    
    # Extract course objective guidelines and course background
    course_objective_guidelines = ""
    course_background = ""
    
    if not course_info_df.empty:
        if "Course Objective Guidelines" in course_info_df.columns:
            course_objective_guidelines = str(course_info_df["Course Objective Guidelines"].iloc[0]) if not pd.isna(course_info_df["Course Objective Guidelines"].iloc[0]) else ""
        if "Course Background" in course_info_df.columns:
            course_background = str(course_info_df["Course Background"].iloc[0]) if not pd.isna(course_info_df["Course Background"].iloc[0]) else ""

    # Load the research notes sheet
    research_notes_sheet, research_notes_df = get_sheet_data_and_df(sheet = sheet, sheet_name = "Final Outline")

    # Create/overwrite a hidden backup sheet to support Delete Step restore
    backup_ws_name = "Backup Final Outline Sheet for Delete step of Research Notes Checklist"
    print(f"📋 Creating backup sheet '{backup_ws_name}' for delete step functionality...")
    backup_ws, _ = create_or_read_worksheet(sheet, backup_ws_name)
    clear_worksheet(backup_ws)
    save_to_sheet(backup_ws, research_notes_df)
    format_worksheet(backup_ws)
    # Hide the backup worksheet 
    try:
        hide_worksheet_by_name(sheet, backup_ws_name)
        print(f"✅ Backup sheet created and hidden successfully")
    except Exception as e:
        print(f"⚠️ Warning: Could not hide backup sheet: {e}")

    if "order" not in research_notes_df.columns:
        research_notes_df["order"] = research_notes_df.index.astype(float)

    # Ensure block text column exists and all rows have block text
    if 'block text' not in research_notes_df.columns:
        research_notes_df['block text'] = ''
    
    # Generate block text for all rows that don't have it
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


    # Group rows so all criteria of one Task/Scope are checked together
    task_scope_groups = list(checklist_df.groupby(["Task", "Scope"], sort=False))
    total_task_scopes = len(task_scope_groups)
    
    progress = SmartProgressBar(total_tasks=total_task_scopes, save_interval=1)
    
    for (task, scope), grp in task_scope_groups:
        
        print(f"Running checklist for Task: {task}, Scope: {scope}")

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
        print(f"📊 Max iterations for this task/scope: {max_iterations}")

        # Extract and map tools from "Review Agent Tools" column
        tools_list = grp["Review Agent Tools"].tolist() if "Review Agent Tools" in grp.columns else []
        # Collect unique tools across all criteria in this group
        unique_tool_names = set()
        for tools_str in tools_list:
            if pd.notna(tools_str) and tools_str:
                tool_names = [t.strip() for t in str(tools_str).split(",")]
                unique_tool_names.update(tool_names)
        # Map tool names to functions
        reviser_tools = []
        for tool_name in unique_tool_names:
            if tool_name in TOOL_NAME_TO_FUNCTION:
                reviser_tools.append(TOOL_NAME_TO_FUNCTION[tool_name])
            else:
                print(f"⚠️ Unknown tool name in checklist: {tool_name}")
        # Default to all tools if none specified
        if not reviser_tools:
            reviser_tools = [create_block, read_blocks, update_block, delete_block, str_replace, search_web, stop]
        # Always ensure stop tool is included so agent can signal completion
        if stop not in reviser_tools:
            reviser_tools.append(stop)
        print(f"🔧 Tools for this task/scope: {[t.name for t in reviser_tools]}")

        # Build merged criteria + examples for checklist agent
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

        # Build merged criteria + corrective ops + examples for reviser agent
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

        # Choose processing method based on scope
        if scope == "Global (full output)":
            print(f"Using SEQUENTIAL processing for Global scope (max {max_iterations} iterations)")

            # Keep existing sequential logic for Global scope
            for key, df_slice in iterate_scope(scope, research_notes_df, scope_to_selector):

                print(f"Processing {key} with {len(df_slice)} rows")

                if df_slice.empty:
                    print(f"No data for {key}, skipping...")
                    continue

                # Initialize message histories for both agents
                reviewer_history = None
                reviser_history = None
                # Track diffs from previous revision for context
                last_revision_diffs = None

                # Iteration loop for review-revise cycles
                for iteration in range(1, max_iterations + 1):
                    print(f"🔄 Iteration {iteration}/{max_iterations} for {key}")

                    research_notes_str = "\n\n---\n\n".join(df_slice["block text"].tolist())

                    # Run the research notes checklist agent (with history and diffs for continuity)
                    analysis, failed_items, reviewer_history = run_research_notes_checklist_agent(
                        course_name=course_name,
                        target_audience=target_audience,
                        course_objective_guidelines=course_objective_guidelines,
                        course_background=course_background,
                        research_notes=research_notes_str,
                        criteria_with_examples=criteria_with_examples_str,
                        llm=reviewer_llm,
                        message_history=reviewer_history,
                        blockwise_diffs=last_revision_diffs
                    )

                    print(f"Failed items for {key} (iteration {iteration}): {failed_items}")

                    if is_failed_items_empty(failed_items):
                        print(f"✅ All criteria passed at iteration {iteration} for {key}")
                        break  # Exit early - all passed

                    # Save state before revision for diff computation
                    df_before_revision = df_slice.copy()

                    # Run the reviser agent with the failed items, analysis, and criteria with corrective operations (with history and diffs for continuity)
                    df_slice, reviser_history = run_reviser_agent(
                        research_notes=research_notes_str,
                        review_analysis=analysis,
                        checklist_feedback=failed_items,
                        criteria_with_ops_and_examples=criteria_with_ops_and_examples_str,
                        df=df_slice,
                        course_name=course_name,
                        target_audience=target_audience,
                        course_objective_guidelines=course_objective_guidelines,
                        course_background=course_background,
                        tools=reviser_tools,
                        llm=llm,
                        message_history=reviser_history,
                        blockwise_diffs=last_revision_diffs
                    )

                    # Compute diffs for next iteration
                    last_revision_diffs = compute_blockwise_diffs(df_before_revision, df_slice)
                    print(f"Revised research notes for {key} (iteration {iteration}): shape {df_slice.shape}")

                # Merge the final slice back into the main DataFrame (after all iterations)
                # Remove the original slice from main DataFrame - but only indices that still exist
                indices_to_drop = [idx for idx in df_slice.index if idx in research_notes_df.index]
                if indices_to_drop:
                    research_notes_df = research_notes_df.drop(indices_to_drop)

                # Add the revised slice back to main DataFrame
                research_notes_df = pd.concat([research_notes_df, df_slice], ignore_index=False)

                # Sort by the existing numeric 'order' column to maintain stable order
                research_notes_df = research_notes_df.sort_values("order", kind="stable")
                
        else:
            print(f"Using PARALLEL processing for {scope} scope (max {max_iterations} iterations per slice)")

            # Use new parallel processing for Topic, Subtopic, Learning Objective scopes
            slice_results = process_rn_scope_slices_parallel(
                scope, research_notes_df, scope_to_selector,
                criteria_with_examples_str, criteria_with_ops_and_examples_str,
                course_name, target_audience,
                course_objective_guidelines, course_background, reviser_tools, llm,
                max_iterations=max_iterations
            )
            
            # Merge all parallel results back to main DataFrame
            print(f"Merging {len(slice_results)} parallel results back to main DataFrame")
            for original_slice, revised_slice in slice_results:
                # Use existing safe merge logic for each result
                indices_to_drop = [idx for idx in original_slice.index if idx in research_notes_df.index]
                if indices_to_drop:
                    research_notes_df = research_notes_df.drop(indices_to_drop)
                
                # Add the revised slice back to main DataFrame
                research_notes_df = pd.concat([research_notes_df, revised_slice], ignore_index=False)
                
                # Sort by the existing numeric 'order' column to maintain stable order
                research_notes_df = research_notes_df.sort_values("order", kind="stable")
            
            print(f"Completed merging parallel results for {scope} scope")
        
        progress.update()
        
    # Validate and correct block text format before parsing
    print("\n🔍 Validating and correcting block text format...")
    corrected_count = 0
    for index, row in research_notes_df.iterrows():
        block_text = str(row.get('block text', '')).strip()
        
        # Check if format is correct
        has_correct_format = (
            'Block ID' in block_text or 
            '####**Topic:**' in block_text
        )
        
        if not has_correct_format and block_text != '':
            # Regenerate block text with correct format
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
    save_to_sheet(worksheet = research_notes_sheet, df = research_notes_df)
    
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
    
    save_to_sheet(worksheet = research_notes_sheet, df = research_notes_df)
    
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


# Block Text Parsing for Research Notes
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
    "step_name": "Checklist Based Review and Revise Agents",
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
    "step_name": "Checklist Based Review and Revise Agents",
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