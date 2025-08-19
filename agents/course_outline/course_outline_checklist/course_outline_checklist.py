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
)
import json
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from services.helper_functions import get_outline_with_los, validate_column_values
from services.smart_progress_bar import SmartProgressBar
from langsmith import traceable

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain.chat_models import init_chat_model


from pydantic import Field
# from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.prebuilt.chat_agent_executor import AgentState
import pandas as pd
from services.crud_text_block_tools import create_block, read_blocks, update_block, delete_block 
from services.helper_functions import iterate_scope

import pandas as pd
from typing import Any, Callable, Dict, Hashable, Iterable, List, Tuple
from pydantic import BaseModel, Field


course_outline_checklist_prompt = """Assume the role of a checklist agent tasked with evaluating the following blocks of course outline for the given list of checklist criteria. This course outline is later used to create research notes and slide content for the course.

This course outline is created for the following course:
<course_info>
Course Name: {course_name}

Target Audience: {target_audience}

Course Objective Guidelines: {course_objective_guidelines}

Course Background: {course_background}
</course_info>

Here's the course outline to evaluate:
<course_outline>
{course_outline}
</course_outline>

Course Outline Structure:
The course outline is structured in blocks where each block contains a Topic, Subtopic, and Learning Objective. Topics and subtopics may appear multiple times across different blocks, but each block has a unique Learning Objective. This repetition is intentional - it allows for multiple specific learning objectives under the same topic/subtopic structure. When evaluating, keep in mind this relationship between related blocks that share the same topic or subtopic.

Here are the checklist criteria to evaluate:
<checklist>
{checklist}
</checklist>

Make sure to output in the following format:
<analysis>
[Analysis of the course outline based on the checklist criteria. It is okay for the analysis to be quite long for accurate evaluation.]
</analysis>

<passed_items>
[List of checklist items that passed. Each item should be the exact review criteria text as it appears in the checklist input along with the criteria name.]
</passed_items>

<failed_items>
[List of checklist items that failed. Format each failed item as follows:
Item: [exact review criteria text as it appears in the checklist input along with the criteria name]
Feedback: [specific detailed feedback on why it failed and what needs to be improved or corrected]

Item: [next failed review criteria text along with the criteria name]
Feedback: [feedback for that criteria]

...and so on for all failed items]
</failed_items>

Feedback Quality Requirements:
- The feedback must be crystal-clear, specific, and directly actionable. Avoid vague or generic statements.
- Explicitly state what to change and where: reference exact Block ID(s) and whether the change is to Topic, Subtopic, or Learning Objective text.
- When proposing edits, provide concrete replacement wording text; do not just describe high-level issues.
- If content should be removed, name the specific Block ID(s) to delete. If reordering is needed, specify the precise new order.
- Your feedback must be as easy to understand and implement as possible.

Notes:
- The analysis should be thorough and cover all aspects of the checklist and all of the criteria.
- The passed items should only include those that fully meet the criteria.
- Feedback of the failed items should include all necessary information for the author to understand what needs to be fixed.
- When listing passed or failed items, always use the complete criteria text along with the criteria name.

Follow the examples below to understand how to evaluate each review criteria:

<examples>
{examples}
</examples>
"""


@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Checklist Based Review and Revise Agents",
    "function_name": "run_course_outline_checklist_agent",
    "user_id": st.session_state.get("role", "anonymous")
})
def run_course_outline_checklist_agent(course_name, target_audience, course_objective_guidelines, course_background, course_outline, checklist, examples, llm = "gemini_2_5_flash"):
    """
    Run the course outline checklist agent with the provided parameters.
    :param course_name: Name of the course for which the course outline is created.
    :param target_audience: Target audience for the course.
    :param course_objective_guidelines: Guidelines for the course objectives.
    :param course_background: Background information about the course.
    :param course_outline: The block of course outline to evaluate.
    :param checklist: The checklist criteria to evaluate the course outline against.
    :param examples: The examples to guide evaluation of the checklist criteria.
    :param llm: The language model to use for the agent.
    :return: A string of all the failed checklist items with feedback. 
    """
    print(f"🔍 COURSE OUTLINE CHECKLIST AGENT using LLM: {llm}")
    course_outline_checklist_agent = Chain(llm = llm, tags = ["failed_items"])

    # # Format the prompt for debugging (printing)
    # formatted_prompt = course_outline_checklist_prompt.format(
    #     course_name = course_name,
    #     target_audience = target_audience,
    #     course_objective_guidelines = course_objective_guidelines,
    #     course_background = course_background,
    #     course_outline = course_outline,
    #     checklist = checklist,
    #     examples = examples,
    # )

    # # Print the formatted prompt for debugging
    # print("\n🔍 COURSE OUTLINE CHECKLIST PROMPT BEING SENT TO LLM:\n")
    # print(formatted_prompt)
    # print("\n" + "=" * 100 + "\n")

    course_outline_checklist_agent.add_message(
        role = "user",
        content = course_outline_checklist_prompt.format(
            course_name = course_name,
            target_audience = target_audience,
            course_objective_guidelines = course_objective_guidelines,
            course_background = course_background,
            course_outline = course_outline,
            checklist = checklist,
            examples = examples,
        )
    )

    response = course_outline_checklist_agent.run()

    return response["failed_items"]


course_outline_reviser_prompt = """You are tasked with revising the following block(s) of course outline based on the feedback provided in the checklist evaluation. This course outline is later used to create research notes and slide content for the course.

This course outline is created for the following course:
<course_info>
Course Name: {course_name}

Target Audience: {target_audience}

Course Objective Guidelines: {course_objective_guidelines}

Course Background: {course_background}
</course_info>

Here are the course outline block(s) to revise:
<course_outline>
{course_outline}
</course_outline>

Course Outline Structure:
The course outline is structured in blocks where each block contains a Topic, Subtopic, and Learning Objective. Topics and subtopics may appear multiple times across different blocks, but each block has a unique Learning Objective. This repetition is intentional - it allows for multiple specific learning objectives under the same topic/subtopic structure. When revising, keep in mind this relationship between related blocks that share the same topic or subtopic.

Here is the checklist feedback to consider:
<checklist_feedback>
{checklist_feedback}
</checklist_feedback>

Here are the review criteria with their corrective operations:
<criteria_with_corrective_operations>
{criteria_with_ops}
</criteria_with_corrective_operations>

Here are the revision examples that you can use as reference to understand the approach to revising the course outline:
<revision_examples>
{reviser_examples}
</revision_examples>

Critical Format Requirements:
When using the CRUD tools, the output you return must strictly maintain the exact format structure as of the original course outline that you are revising:

1. Each block must start with: "###Block ID: [number]"
2. Followed by: "####**Topic:**" section with the topic content
3. Followed by: "####**Subtopic:**" section with the subtopic content  
4. Followed by: "####**Learning Objective:**" section with the learning objective content

Example of correct format:
###Block ID: 1
####**Topic:**
HVAC Fundamentals
####**Subtopic:**
Refrigeration Cycle
####**Learning Objective:**
Understand the basic refrigeration cycle

Important: Never change or remove these headers (Block ID, Topic, Subtopic, Learning Objective). Only modify the content after the header as required for the revision. The format is essential for the system to function properly.

Notes:
- Make use of the given set of CRUD block text tools to make the necessary revisions. 
- These CRUD tools allow you to create, read, update, and delete blocks of text from the above course outline as needed.
- You can only work with one block of text at a time. Each block of text is a separate entity identified by a unique ID - Block ID.
- To implement some of the feedback, you may need to make edits to multiple blocks of text.

Corrective Operations:
These are specific instructions for fixing failed criteria items. Match each failed criteria from the checklist feedback above with its corresponding corrective operation in the criteria section below. Only apply the corrective operations for criteria that actually failed - ignore corrective operations for criteria that passed the evaluation. Use these corrective operations in combination with the feedback to make the necessary revisions.

Revision Examples:
These examples show how to apply the corrections to the course outline. Use these examples as reference to understand the approach to revising the course outline. Only refer the examples of the criteria that failed.
"""


@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Checklist Based Review and Revise Agents",
    "function_name": "run_course_outline_reviser_agent",
    "user_id": st.session_state.get("role", "anonymous")
})
def run_course_outline_reviser_agent(course_outline, checklist_feedback, criteria_with_ops, reviser_examples, df, course_name, target_audience, course_objective_guidelines, course_background, llm = "gemini_2_flash"):
    """
    Run the course outline reviser agent with the provided parameters.
    :param course_outline: The block of course outline to revise.
    :param checklist_feedback: The feedback from the checklist evaluation to consider for revisions.
    :param criteria_with_ops: All criteria with their corrective operations for the reviser to reference.
    :param reviser_examples: Examples showing how to apply corrections for each criteria.
    :param df: The dataframe containing the course outline.
    :param course_name: Name of the course for which the course outline is created.
    :param target_audience: Target audience for the course.
    :param course_objective_guidelines: Guidelines for the course objectives.
    :param course_background: Background information about the course.
    :param llm: The language model to use for the agent.
    :return: The revised block of course outline as a dataframe.
    """
    print(f"\n🔄 STARTING COURSE OUTLINE REVISER AGENT for {len(df)} blocks")
    print(f"🤖 REVISER AGENT using LLM: {llm}")
    print(f"📝 Checklist feedback length: {len(checklist_feedback)} chars")
    print("=" * 80)

    rate_limiter = InMemoryRateLimiter(
        requests_per_second=1,  # <-- Super slow! We can only make a request once every 10 seconds!!
        check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
        max_bucket_size=10,  # Controls the maximum burst size.
    )

    # Map LLM names to init_chat_model format
    llm_mapping = {
        "gpt5_thinking": "openai:gpt-5",  # Use gpt-5 with reasoning_effort="high"
        "gemini_2_5_flash": "google_genai:gemini-2.5-flash",
        "gemini_2_5_pro": "google_genai:gemini-2.5-pro",  
        "sonnet_4_thinking": "anthropic:claude-sonnet-4-20250514",  # Claude Sonnet 4 with thinking
    }
    
    model_name = llm_mapping.get(llm, "google_genai:gemini-2.5-flash")  # Default fallback
    print(f"🔗 Mapped to init_chat_model: {model_name}")
    
    # Add reasoning_effort parameter for thinking models
    init_params = {
        "rate_limiter": rate_limiter,
        "max_retries": 20,
    }
    
    if llm == "gpt5_thinking":
        init_params["reasoning_effort"] = "high"
    
    llm_instance = init_chat_model(model_name, **init_params)

    class BufferState(AgentState):
        """
        Short-term state for the agent execution.
        Only one field is needed: the shared dataframe instance.
        """
        df: pd.DataFrame = Field(default_factory=pd.DataFrame)


    graph = create_react_agent(
        model=llm_instance,
        tools=[create_block, read_blocks, update_block, delete_block],
        state_schema=BufferState,     # <— includes the dataframe
    )

    # # Format the prompt for debugging (printing)
    # formatted_prompt = course_outline_reviser_prompt.format(
    #     course_outline=course_outline,
    #     checklist_feedback=checklist_feedback,
    #     criteria_with_ops=criteria_with_ops,
    #     reviser_examples=reviser_examples,
    #     course_name=course_name,
    #     target_audience=target_audience,
    #     course_objective_guidelines=course_objective_guidelines,
    #     course_background=course_background,
    # )

    # # Print the formatted prompt for debugging
    # print("\n🔍 COURSE OUTLINE REVISER PROMPT BEING SENT TO LLM:\n")
    # print(formatted_prompt)
    # print("\n" + "=" * 100 + "\n")

    state = {
        "messages": [{"role": "user", "content": course_outline_reviser_prompt.format(
            course_outline=course_outline,
            checklist_feedback=checklist_feedback,
            criteria_with_ops=criteria_with_ops,
            reviser_examples=reviser_examples,
            course_name=course_name,
            target_audience=target_audience,
            course_objective_guidelines=course_objective_guidelines,
            course_background=course_background,)}],
        "df": df,        # one buffer for the whole session
    }

    final_state = graph.invoke(
        state,
        {"recursion_limit": 50}
    )

    print("=" * 80)
    print("✅ COURSE OUTLINE REVISER AGENT COMPLETED")
    print(f"📊 Final DataFrame shape: {final_state['df'].shape}")
    print("=" * 80)

    return final_state["df"]


@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Checklist Based Review and Revise Agents",
    "function_name": "process_single_co_slice",
    "user_id": st.session_state.get("role", "anonymous")
})
def process_single_co_slice(key, df_slice, criteria_str, examples_str, criteria_ops_str, reviser_examples_str, course_name, target_audience, course_objective_guidelines, course_background, llm):
    """
    Process a single slice: review → revise (if needed) → return result
    This function runs in parallel for scope-based processing.
    
    :param key: The key identifying this slice (from iterate_scope)
    :param df_slice: The DataFrame slice to process
    :param criteria_str: Formatted criteria string for review agent
    :param examples_str: Formatted examples string for review agent
    :param criteria_ops_str: Formatted criteria with operations for reviser agent
    :param reviser_examples_str: Formatted reviser examples string
    :param course_name: Name of the course
    :param target_audience: Target audience for the course
    :param course_objective_guidelines: Guidelines for course objectives
    :param course_background: Background information about the course
    :param llm: The language model to use
    :return: The revised DataFrame slice (or original if no changes needed)
    """
    print(f"🔄 Processing slice '{key}' with {len(df_slice)} rows in parallel")
    
    if df_slice.empty:
        print(f"Empty slice '{key}', returning original")
        return df_slice
    
    # Build course outline string from slice
    course_outline_str = "\n\n---\n\n".join(df_slice["block text"].tolist())
    
    # Run the course outline checklist agent
    failed_items = run_course_outline_checklist_agent(
        course_name=course_name,
        target_audience=target_audience,
        course_objective_guidelines=course_objective_guidelines,
        course_background=course_background,
        course_outline=course_outline_str,
        checklist=criteria_str,
        examples=examples_str,
        llm=llm
    )
    
    print(f"Failed items for slice '{key}': {failed_items}")
    
    if not failed_items:
        print(f"All items passed for slice '{key}'")
        return df_slice  # No changes needed
    
    # Run the reviser agent with the failed items
    revised_course_outline_df = run_course_outline_reviser_agent(
        course_outline=course_outline_str,
        checklist_feedback=failed_items,
        criteria_with_ops=criteria_ops_str,
        reviser_examples=reviser_examples_str,
        df=df_slice,
        course_name=course_name,
        target_audience=target_audience,
        course_objective_guidelines=course_objective_guidelines,
        course_background=course_background,
        llm=llm
    )
    
    print(f"Revised course outline for slice '{key}': shape {revised_course_outline_df.shape}")
    return revised_course_outline_df


@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Checklist Based Review and Revise Agents",
    "function_name": "process_co_scope_slices_parallel",
    "user_id": st.session_state.get("role", "anonymous")
})
def process_co_scope_slices_parallel(scope, course_outline_df, scope_to_selector, criteria_str, examples_str, criteria_ops_str, reviser_examples_str, course_name, target_audience, course_objective_guidelines, course_background, llm):
    """
    Process all slices within a scope in parallel.
    
    :param scope: The scope to process (Topic, Subtopic, Learning Objective)
    :param course_outline_df: The main DataFrame
    :param scope_to_selector: Scope selector mapping
    :param criteria_str: Formatted criteria string for review agent
    :param examples_str: Formatted examples string for review agent
    :param criteria_ops_str: Formatted criteria with operations for reviser agent
    :param reviser_examples_str: Formatted reviser examples string
    :param course_name: Name of the course
    :param target_audience: Target audience for the course
    :param course_objective_guidelines: Guidelines for course objectives
    :param course_background: Background information about the course
    :param llm: The language model to use
    :return: List of (original_slice, revised_slice) tuples
    """
    print(f"🚀 Starting parallel processing for scope: {scope}")
    
    # Collect all slices for this scope
    all_slices = list(iterate_scope(scope, course_outline_df, scope_to_selector))
    
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
                process_single_co_slice,
                key, df_slice, criteria_str, examples_str, 
                criteria_ops_str, reviser_examples_str,
                course_name, target_audience, course_objective_guidelines, 
                course_background, llm
            )
            futures_map[future] = (key, df_slice)
        
        # Collect results with progress tracking
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
    "agent_name": "course_outline",
    "step_name": "Checklist Based Review and Revise Agents",
    "function_name": "run_course_outline_checklist_and_reviser",
    "user_id": st.session_state.get("role", "anonymous")
})
def run_course_outline_checklist_and_reviser(sheet, course_name, target_audience, checklist_sheet_link, gc, llm = "gemini_2_5_flash"):
    """
    Run the course outline checklist agent and reviser agent with the provided parameters.
    :param course_name: Name of the course for which the course outline is created.
    :param target_audience: Target audience for the course.
    :param checklist_sheet_link: Link to the Google Sheet containing the checklist.
    :param gc: Google Sheets client instance.
    :param llm: The language model to use for the agents.
    :return: None
    """

    # Load the checklist sheet
    checklist_sheet = gc.open_by_url(checklist_sheet_link)
    checklist_worksheet, checklist_df = get_sheet_data_and_df(sheet = checklist_sheet, sheet_name = "Course Outline Checklist")

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

    # Load the course outline sheet
    course_outline_sheet, course_outline_df = get_sheet_data_and_df(sheet = sheet, sheet_name = "Final Outline")

    # Create/overwrite a hidden backup sheet to support Delete Step restore
    backup_ws_name = "Backup Final Outline Sheet for Delete step of Course Outline Checklist"
    print(f"📋 Creating backup sheet '{backup_ws_name}' for delete step functionality...")
    backup_ws, _ = create_or_read_worksheet(sheet, backup_ws_name)
    clear_worksheet(backup_ws)
    save_to_sheet(backup_ws, course_outline_df)
    format_worksheet(backup_ws)
    # Hide the backup worksheet 
    try:
        hide_worksheet_by_name(sheet, backup_ws_name)
        print(f"✅ Backup sheet created and hidden successfully")
    except Exception as e:
        print(f"⚠️ Warning: Could not hide backup sheet: {e}")

    if "order" not in course_outline_df.columns:
        course_outline_df["order"] = course_outline_df.index.astype(float)

    # Ensure block text column exists and all rows have block text
    if 'block text' not in course_outline_df.columns:
        course_outline_df['block text'] = ''
    
    # Generate block text for all rows that don't have it
    for index, row in course_outline_df.iterrows():
        if pd.isna(course_outline_df.at[index, 'block text']) or course_outline_df.at[index, 'block text'] == '':
            course_outline_str = ""
            course_outline_str += f"###Block ID: {index}\n"
            course_outline_str += f"####**Topic:**\n{row['Topic']}\n"
            course_outline_str += f"####**Subtopic:**\n{row['Subtopic']}\n"
            course_outline_str += f"####**Learning Objective:**\n{row['Learning Objectives']}\n"

            course_outline_df.at[index, 'block text'] = course_outline_str


    scope_to_selector = {
        "Global (full output)": [],          # whole frame
        "Topic":  ["Topic"],                 # group by Topic col
        "Subtopic": ["Topic", "Subtopic"],   # group by both
        "Learning Objective": "__row__",     # one row each
    }


    # Group rows so all criteria of one Task/Scope are checked together
    task_scope_groups = list(checklist_df.groupby(["Task", "Scope"], sort=False))
    total_task_scopes = len(task_scope_groups)
    
    progress = SmartProgressBar(total_tasks=total_task_scopes * 2, save_interval=1)  # *2 for two loops
    
    #T wo-loop structure for review and revise
    for loop_num in range(1, 3):  # Loop 1 and Loop 2
        print(f"\n{'='*80}")
        print(f"🔄 STARTING REVIEW-REVISE LOOP {loop_num}")
        print(f"{'='*80}")
        
        for (task, scope), grp in task_scope_groups:
            
            print(f"Loop {loop_num} - Running checklist for Task: {task}, Scope: {scope}")

            # Extract criteria and examples together
            criteria_list = grp["Review Criteria"].tolist()
            criteria_names = grp["Criteria Name"].tolist()
            examples_list = grp["Review Agent Examples"].tolist()
            
            # Build structured criteria format
            criteria_with_separators = []
            for i, (criteria_name, criteria) in enumerate(zip(criteria_names, criteria_list), 1):
                criteria_with_separators.append(f"<criteria_{i}>\nReview Criteria name: {criteria_name}\nReview Criteria: {criteria}\n</criteria_{i}>")
            
            criteria_str = "\n\n".join(criteria_with_separators)
            
            # Build examples with structured format for each criteria
            examples_with_separators = []
            for i, (criteria_name, criteria, example) in enumerate(zip(criteria_names, criteria_list, examples_list)):
                criteria_num = i + 1
                examples_with_separators.append(f"<criteria_{criteria_num}>\n\nReview Criteria name: {criteria_name}\nReview Criteria: {criteria}\n\n<example>\n\n{example}\n\n</example>\n\n</criteria_{criteria_num}>")
            examples_str = "\n\n".join(examples_with_separators)
            
            # Build criteria with corrective operations for reviser agent
            criteria_with_ops = []
            for _, row in grp.iterrows():
                criteria_name = row["Criteria Name"]
                criteria = row["Review Criteria"]
                corrective_ops = row["Corrective Operations"]
                criteria_with_ops.append(f"<criteria>\n\nReview Criteria name: {criteria_name}\nReview Criteria: {criteria}\n\nCorrective Operation: {corrective_ops}\n\n</criteria>")
            
            criteria_ops_str = "\n\n".join(criteria_with_ops)
            
            # Build examples with structured format for reviser agent
            reviser_examples_with_separators = []
            for i, (criteria_name, criteria, corrective_ops, reviser_example) in enumerate(zip(criteria_names, criteria_list, grp["Corrective Operations"].tolist(), grp["Reviser Agent Examples"].tolist())):
                criteria_num = i + 1
                reviser_examples_with_separators.append(f"<criteria_{criteria_num}>\n\nReview Criteria name: {criteria_name}\nReview Criteria: {criteria}\n\nCorrective Operation: {corrective_ops}\n\n<example>\n\n{reviser_example}\n\n</example>\n\n</criteria_{criteria_num}>")
            
            reviser_examples_str = "\n\n".join(reviser_examples_with_separators)

            # Choose processing method based on scope
            if scope == "Global (full output)":
                print(f"Loop {loop_num} - Using SEQUENTIAL processing for Global scope")
                
                for key, df_slice in iterate_scope(scope, course_outline_df, scope_to_selector):

                    print(f"Loop {loop_num} - Processing {key} with {len(df_slice)} rows")

                    if df_slice.empty:
                        print(f"No data for {key}, skipping...")
                        continue

                    course_outline_str = "\n\n---\n\n".join(df_slice["block text"].tolist())

                    # Run the course outline checklist agent
                    failed_items = run_course_outline_checklist_agent(
                        course_name=course_name,
                        target_audience=target_audience,
                        course_objective_guidelines=course_objective_guidelines,
                        course_background=course_background,
                        course_outline=course_outline_str,
                        checklist=criteria_str,
                        examples=examples_str,
                        llm=llm
                    )

                    print(f"Failed items for {key}: {failed_items}")

                    if not failed_items:
                        print(f"All items passed for {key}.")
                        continue

                    # Run the reviser agent with the failed items and criteria with corrective operations
                    revised_course_outline_df = run_course_outline_reviser_agent(
                        course_outline=course_outline_str,
                        checklist_feedback=failed_items,
                        criteria_with_ops=criteria_ops_str,
                        reviser_examples=reviser_examples_str,
                        df=df_slice,
                        course_name=course_name,
                        target_audience=target_audience,
                        course_objective_guidelines=course_objective_guidelines,
                        course_background=course_background,
                        llm=llm
                    )

                    print(f"Revised course outline for {key}: {revised_course_outline_df}")

                    # Merge the revised slice back into the main DataFrame
                    # Remove the original slice and replace with the revised version
                    # Remove the original slice from main DataFrame - but only indices that still exist
                    indices_to_drop = [idx for idx in df_slice.index if idx in course_outline_df.index]
                    if indices_to_drop:
                        course_outline_df = course_outline_df.drop(indices_to_drop)
                    
                    # Add the revised slice back to main DataFrame
                    course_outline_df = pd.concat([course_outline_df, revised_course_outline_df], ignore_index=False)
                    
                    # Sort by the existing numeric 'order' column
                    course_outline_df = course_outline_df.sort_values("order", kind="stable")
                    
            else:
                print(f"Loop {loop_num} - Using PARALLEL processing for {scope} scope")
                
                # Use parallel processing for Topic, Subtopic, Learning Objective scopes
                slice_results = process_co_scope_slices_parallel(
                    scope, course_outline_df, scope_to_selector, 
                    criteria_str, examples_str, criteria_ops_str, 
                    reviser_examples_str, course_name, target_audience, 
                    course_objective_guidelines, course_background, llm
                )
                
                # Merge all parallel results back to main DataFrame
                print(f"Loop {loop_num} - Merging {len(slice_results)} parallel results back to main DataFrame")
                for original_slice, revised_slice in slice_results:
                    # Use existing safe merge logic for each result
                    indices_to_drop = [idx for idx in original_slice.index if idx in course_outline_df.index]
                    if indices_to_drop:
                        course_outline_df = course_outline_df.drop(indices_to_drop)
                    
                    # Add the revised slice back to main DataFrame
                    course_outline_df = pd.concat([course_outline_df, revised_slice], ignore_index=False)
                    
                    # Sort by the existing numeric 'order' column
                    course_outline_df = course_outline_df.sort_values("order", kind="stable")
                
                print(f"Loop {loop_num} - Completed merging parallel results for {scope} scope")

            progress.update()
        
        # Inter-loop processing after Loop 1
        if loop_num == 1:
            print(f"\n🔄 COMPLETING LOOP 1 - PARSING AND CLEANUP")
            print("="*80)
            
            # Parse the updated block text column back to individual columns after Loop 1
            print("📝 Parsing Loop 1 block text content back to individual columns...")
            course_outline_df = parse_co_block_text_to_columns(course_outline_df)
            
            # Clean up: Remove order and block text columns before Loop 2
            columns_to_drop = []
            if 'order' in course_outline_df.columns:
                columns_to_drop.append('order')
                print("🗑️ Removing 'order' column before Loop 2")
            if 'block text' in course_outline_df.columns:
                columns_to_drop.append('block text')
                print("🗑️ Removing 'block text' column before Loop 2")
            
            if columns_to_drop:
                course_outline_df = course_outline_df.drop(columns=columns_to_drop)
            
            # Update sheet with Loop 1 results (parsed individual columns, no order/block text)
            print("💾 Updating sheet with Loop 1 parsed results...")
            course_outline_sheet.clear()
            save_to_sheet(worksheet=course_outline_sheet, df=course_outline_df)
            
            # Prepare for Loop 2: Regenerate order and block text columns
            print("🔄 Preparing for Loop 2: Regenerating order and block text...")
            
            # Regenerate order column
            course_outline_df["order"] = course_outline_df.index.astype(float)
            
            # Regenerate block text for all rows
            if 'block text' not in course_outline_df.columns:
                course_outline_df['block text'] = ''
            
            for index, row in course_outline_df.iterrows():
                course_outline_str = ""
                course_outline_str += f"###Block ID: {index}\n"
                course_outline_str += f"####**Topic:**\n{row['Topic']}\n"
                course_outline_str += f"####**Subtopic:**\n{row['Subtopic']}\n"
                course_outline_str += f"####**Learning Objective:**\n{row['Learning Objectives']}\n"
                course_outline_df.at[index, 'block text'] = course_outline_str
            
            print(f"✅ Ready for Loop 2 with {len(course_outline_df)} rows")
            print("="*80)
            
    # Final processing
    print(f"\n🏁 BOTH LOOPS COMPLETED - FINAL PROCESSING")
    print("="*80)
    
    # Validate and correct block text format before parsing
    print("\n🔍 Validating and correcting block text format...")
    corrected_count = 0
    for index, row in course_outline_df.iterrows():
        block_text = str(row.get('block text', '')).strip()
        
        # Check if format is correct
        has_correct_format = (
            'Block ID' in block_text or 
            '####**Topic:**' in block_text
        )
        
        if not has_correct_format and block_text != '':
            # Regenerate block text with correct format
            course_outline_str = ""
            course_outline_str += f"###Block ID: {index}\n"
            course_outline_str += f"####**Topic:**\n{row['Topic']}\n"
            course_outline_str += f"####**Subtopic:**\n{row['Subtopic']}\n"
            course_outline_str += f"####**Learning Objective:**\n{row['Learning Objectives']}\n"
            
            course_outline_df.loc[index, 'block text'] = course_outline_str
            corrected_count += 1
            print(f"✅ Row {index} corrected successfully")
        elif not has_correct_format and block_text == '':
            print(f"⚠️ Row {index}: Empty block text, skipping")
        else:
            print(f"✅ Row {index}: Format is correct, no action needed")
    
    print(f"✅ Corrected format for {corrected_count} rows")
    
    # Clear the worksheet first to handle row deletions properly
    course_outline_sheet.clear()
    
    # Save to sheet
    save_to_sheet(worksheet = course_outline_sheet, df = course_outline_df)
    
    # Parse the updated block text column back to individual columns
    print("\n🔄 Parsing updated block text content back to individual columns...")
    course_outline_df = parse_co_block_text_to_columns(course_outline_df)

    column_renames = {}
    if 'order' in course_outline_df.columns:
        column_renames['order'] = 'co_order'
    if 'block text' in course_outline_df.columns:
        column_renames['block text'] = 'co_block text'

    if column_renames:
        course_outline_df = course_outline_df.rename(columns=column_renames)
    
    save_to_sheet(worksheet = course_outline_sheet, df = course_outline_df)
    format_worksheet(worksheet = course_outline_sheet)
    print("✅ Block text parsing completed. Individual columns updated with revised content.")
   
    return


# Block Text Parsing for Course Outline
class CourseOutlineBlockContent(BaseModel):
    topic: str = Field(description="The topic extracted from the block text.")
    subtopic: str = Field(description="The subtopic extracted from the block text.")
    learning_objective: str = Field(description="The learning objective extracted from the block text.")

co_block_text_parsing_prompt = """You are an expert parser for course outline block text. Given a block of text in a structured format, extract the following fields:

- topic: The value after '####**Topic:**' (extract only the topic text, not the header)
- subtopic: The value after '####**Subtopic:**' (extract only the subtopic text, not the header)
- learning_objective: The value after '####**Learning Objective:**' (extract only the learning objective text, not the header)

Return the output as a structured object with these fields. Do not add or infer any information. Only extract what is present in the block.

Block:
{block}
"""


@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Checklist Based Review and Revise Agents",
    "function_name": "parse_co_block_text_row",
    "user_id": st.session_state.get("role", "anonymous")
})
def parse_co_block_text_row(block_text_cell, index):
    """
    Parses a single block text cell to extract Topic, Subtopic, and Learning Objective.
    :param block_text_cell: The raw text from the block text column.
    :param index: The row index (for debugging).
    :return: Dict with parsed content or None if parsing fails.
    """
    try:
        # Use LLM + Pydantic to parse and validate
        agent = Chain(llm="gemini_2_flash")
        agent.add_message(
            role="user",
            content=co_block_text_parsing_prompt.format(block=block_text_cell)
        )
        agent.structured_output = CourseOutlineBlockContent
        response = agent.run()
        parsed = response.model_dump()
        return parsed
    except Exception as e:
        print(f"Failed to parse block text for row {index}: {e}")
        return None


@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Checklist Based Review and Revise Agents",
    "function_name": "parse_co_block_text_to_columns",
    "user_id": st.session_state.get("role", "anonymous")
})
def parse_co_block_text_to_columns(df, max_workers=5):
    """
    Parses the block text column and updates the individual Topic, Subtopic, and Learning Objectives columns.
    :param df: The DataFrame to update.
    :param max_workers: Number of parallel workers (default 5).
    :return: The updated DataFrame.
    """
    # Check if block text column exists
    if 'block text' not in df.columns:
        print("No 'block text' column found. Nothing to parse.")
        return df

    # Prepare for parallel processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, row in df.iterrows():
            block_text_cell = str(row.get("block text", "")).strip()
            if not block_text_cell:
                continue  # Skip empty block text
            future = executor.submit(parse_co_block_text_row, block_text_cell, index)
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
                    print(f"Updated row {index} with parsed content")
            except Exception as e:
                print(f"Error parsing row {index}: {e}")
            progress.update()

    print(f"Successfully parsed block text column and updated individual columns.")
    
    return df



def delete_course_outline_checklist_and_reviser(sheet, worksheet_name="Final Outline"):
    """Restore the Final Outline from the hidden backup created for the checklist step,
    then delete the backup sheet.
    """
    backup_ws_name = "Backup Final Outline Sheet for Delete step of Course Outline Checklist"

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
        print(f"⚠️ No backup worksheet found for checklist delete step. Skipping restore.")
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


