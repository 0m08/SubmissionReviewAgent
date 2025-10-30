from modules.chain import Chain
from tqdm import tqdm
import re
from services.sheets_service import (
    get_sheet_data_and_df,
    save_to_sheet,
    filter_non_blank_column,
    clear_worksheet,
    delete_worksheet,
    clear_all_filters,
    hide_columns_by_name,
    get_worksheet_names,
    safe_get_sheet_data_and_df,
    hide_worksheet_by_name,
    create_or_read_worksheet,
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


from typing import Annotated, List, Optional, Tuple
from pydantic import Field
# from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.prebuilt.chat_agent_executor import AgentState
import pandas as pd
from services.crud_text_block_tools import create_block, read_blocks, update_block, delete_block
import pandas as pd
from typing import Any, Callable, Dict, Hashable, Iterable, List, Tuple
from pydantic import BaseModel, Field
from services.helper_functions import iterate_scope
import services.crud_text_block_tools as crud_tools
    

@traceable(metadata={
    "agent_name": "slide chunks",
    "step_name": "Slide Chunks Checklist Review and Revise",
    "function_name": "process_single_slide_chunk_slice",
    "user_id": st.session_state.get("role", "anonymous")
})
def process_single_slide_chunk_slice(key, df_slice, criteria_str, examples_str, criteria_ops_str, reviser_examples_str, course_name, target_audience, sheet, gc, global_max_index, llm):
    """
    Process a single topic slice: review → revise (if needed) → return result
    This function runs in parallel for scope-based processing.
    """
    print(f"🔄 Processing topic slice '{key}' with {len(df_slice)} rows in parallel")
    if df_slice.empty:
        print(f"Empty slice '{key}', returning original")
        return key, None, df_slice, global_max_index
    
    slide_chunks_str = "\n\n---\n\n".join(df_slice["block text"].tolist())
    learning_objective = get_learning_objectives("Topic", df_slice, sheet, gc)
    print(f"📚 Learning objectives for topic '{key}': {learning_objective[:150]}...")
    failed_items = run_slide_chunks_checklist_agent(
        course_name=course_name,
        target_audience=target_audience,
        slide_chunks=slide_chunks_str,
        checklist=criteria_str,
        examples=examples_str,
        learning_objective=learning_objective,
        llm=llm
    )
    print(f"Failed items for topic '{key}': {failed_items}")
    if not failed_items:
        print(f"All items passed for topic '{key}'.")
        return key, None, df_slice, global_max_index
    
    revised_slide_chunks_df = run_reviser_agent(
        slide_chunks=slide_chunks_str,
        checklist_feedback=failed_items,
        criteria_with_ops=criteria_ops_str,
        reviser_examples=reviser_examples_str,
        df=df_slice,
        learning_objective=learning_objective,
        global_max_index=global_max_index,
        llm=llm
    )
    print(f"Revised slide chunks for topic '{key}': {revised_slide_chunks_df}")
    
    # --- Handle new blocks and assign temporary IDs ---
    current_max_index = global_max_index
    new_indices = [idx for idx in revised_slide_chunks_df.index if idx not in df_slice.index]
    temp_id_mapping = {}  # Track temporary IDs for new blocks
    
    for idx in new_indices:
        # Assign a temporary unique block ID (will be finalized in main loop)
        current_max_index += 1
        temp_new_id = current_max_index
        revised_slide_chunks_df = revised_slide_chunks_df.rename(index={idx: temp_new_id})
        temp_id_mapping[idx] = temp_new_id
        print(f"🆔 Assigned temporary ID {temp_new_id} to new block (was {idx})")
    
    # --- Fix block IDs in text for all blocks ---
    revised_slide_chunks_df = fix_block_ids_in_text(revised_slide_chunks_df)
    
    # --- Fill topic/subtopic for all blocks if NaN ---
    topic_col = 'topic' if 'topic' in revised_slide_chunks_df.columns else 'Topic'
    subtopic_col = 'subtopic' if 'subtopic' in revised_slide_chunks_df.columns else 'Subtopic'
    
    # Ensure columns exist
    if topic_col not in revised_slide_chunks_df.columns:
        revised_slide_chunks_df[topic_col] = ''
    if subtopic_col not in revised_slide_chunks_df.columns:
        revised_slide_chunks_df[subtopic_col] = ''
    
    for idx in revised_slide_chunks_df.index:
        if 'block text' in revised_slide_chunks_df.columns:
            block_text = str(revised_slide_chunks_df.loc[idx, 'block text'])
            
            # Check if topic/subtopic are missing or NaN and extract from block text
            current_topic = revised_slide_chunks_df.loc[idx, topic_col]
            current_subtopic = revised_slide_chunks_df.loc[idx, subtopic_col]
            
            if pd.isna(current_topic) or current_topic == '' or current_topic == 'nan':
                topic_match = re.search(r'####\*\*Topic:\*\*\s*\n(.+?)(?=\n####|\n\n|\Z)', block_text, re.DOTALL)
                topic_value = topic_match.group(1).strip() if topic_match else ''
                if topic_value:
                    revised_slide_chunks_df.loc[idx, topic_col] = topic_value
                    print(f"📝 Filled topic for block {idx}: '{topic_value}'")
            
            if pd.isna(current_subtopic) or current_subtopic == '' or current_subtopic == 'nan':
                subtopic_match = re.search(r'####\*\*Subtopic:\*\*\s*\n(.+?)(?=\n####|\n\n|\Z)', block_text, re.DOTALL)
                subtopic_value = subtopic_match.group(1).strip() if subtopic_match else ''
                if subtopic_value:
                    revised_slide_chunks_df.loc[idx, subtopic_col] = subtopic_value
                    print(f"📝 Filled subtopic for block {idx}: '{subtopic_value}'")
        
        # --- Track and remove deleted blocks ---
        deleted_indices = set(df_slice.index) - set(revised_slide_chunks_df.index)
        if deleted_indices:
            print(f"🗑️ Removing deleted blocks: {deleted_indices}")
            revised_slide_chunks_df = revised_slide_chunks_df.drop(index=deleted_indices)
            revised_slide_chunks_df = revised_slide_chunks_df[~revised_slide_chunks_df.index.isin(deleted_indices)]
    
    print(f"✅ Processed slice '{key}': fixed block IDs, filled topic/subtopic fields")
    
    return key, revised_slide_chunks_df, df_slice, current_max_index

def process_slide_chunk_scope_slices_parallel(scope, slide_chunks_df, scope_to_selector, criteria_str, examples_str, criteria_ops_str, reviser_examples_str, course_name, target_audience, sheet, gc, global_max_index, llm):
    """
    Process all topic slices within a scope in parallel.
    """
    print(f"🚀 Starting parallel processing for scope: {scope}")
    all_slices = list(iterate_scope(scope, slide_chunks_df, scope_to_selector))
    if not all_slices:
        print(f"No slices found for scope {scope}")
        return []
    print(f"Found {len(all_slices)} slices for scope {scope}")
    
    # Ensure parent relationship is set for all slices to enable full DataFrame context in CRUD tools
    enhanced_slices = []
    for key, df_slice in all_slices:
        df_slice_with_parent = ensure_parent_relationship(df_slice, slide_chunks_df)
        enhanced_slices.append((key, df_slice_with_parent))
    
    args_list = [(key, df_slice, criteria_str, examples_str, criteria_ops_str, reviser_examples_str, course_name, target_audience, sheet, gc, global_max_index, llm) for key, df_slice in enhanced_slices]
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_single_slide_chunk_slice, *args) for args in args_list]
        for future in as_completed(futures):
            results.append(future.result())
    print(f"🏁 Completed parallel processing for scope {scope}: {len(results)} results")
    return results

def get_learning_objectives(scope, df_slice, sheet, gc):
    """
    Fetch learning objectives based on scope. Handles all scopes in one clean function.
    
    :param scope: The scope (Global, Topic, Subtopic, Learning Objective)
    :param df_slice: The DataFrame slice being evaluated
    :param sheet: The Google Sheet object (passed from main function)
    :param gc: Google Sheets client (passed from main function)
    :return: Formatted learning objectives string
    """
    try:
        # Get the Final Outline sheet from the same workbook
        _, outline_df = safe_get_sheet_data_and_df(sheet, "Final Outline")
        if outline_df.empty:
            print("⚠️ Final Outline sheet not found or empty")
            return f"Learning objectives unavailable for scope '{scope}'. Ensure content is educationally sound and aligned with course goals."
        
        # Handle column name variations
        topic_col = 'Topic' if 'Topic' in outline_df.columns else 'topic'
        subtopic_col = 'Subtopic' if 'Subtopic' in outline_df.columns else 'subtopic'
        learning_obj_col = 'Learning Objectives' if 'Learning Objectives' in outline_df.columns else 'learning_objectives'
        
        if learning_obj_col not in outline_df.columns:
            return f"Learning Objectives column not found. Ensure content supports educational goals for scope '{scope}'."
        
        # Handle different scopes
        if scope == "Subtopic" and not df_slice.empty:
            # Single subtopic
            slice_topic_col = 'topic' if 'topic' in df_slice.columns else 'Topic'
            slice_subtopic_col = 'subtopic' if 'subtopic' in df_slice.columns else 'Subtopic'
            
            if slice_topic_col in df_slice.columns and slice_subtopic_col in df_slice.columns:
                topic = df_slice.iloc[0][slice_topic_col]
                subtopic = df_slice.iloc[0][slice_subtopic_col]
                
                match = outline_df[(outline_df[topic_col] == topic) & (outline_df[subtopic_col] == subtopic)]
                if not match.empty:
                    los = [row[learning_obj_col].strip() for _, row in match.iterrows() if pd.notna(row[learning_obj_col])]
                    if los:
                        return "Learning Objectives:\n" + "\n".join(f"• {lo}" for lo in los)
        
        elif scope == "Topic" and not df_slice.empty:
            # Multiple subtopics under one topic
            slice_topic_col = 'topic' if 'topic' in df_slice.columns else 'Topic'
            slice_subtopic_col = 'subtopic' if 'subtopic' in df_slice.columns else 'Subtopic'
            
            if slice_topic_col in df_slice.columns:
                topic = df_slice.iloc[0][slice_topic_col]
                subtopics = df_slice[slice_subtopic_col].unique()
                
                objectives = []
                for subtopic in subtopics:
                    if pd.notna(subtopic):
                        match = outline_df[(outline_df[topic_col] == topic) & (outline_df[subtopic_col] == subtopic)]
                        if not match.empty:
                            subtopic_los = [row[learning_obj_col].strip() for _, row in match.iterrows() if pd.notna(row[learning_obj_col])]
                            objectives.extend([f"• {subtopic}: {lo}" for lo in subtopic_los])
                
                if objectives:
                    return f"Learning Objectives for '{topic}':\n" + "\n".join(objectives) + ""
        
        elif scope == "Global (full output)":
            # Collect all learning objectives grouped by topic (preserve original order)
            # Filter rows that have topic, subtopic and a learning objective
            filtered = outline_df[[topic_col, subtopic_col, learning_obj_col]].dropna()

            if filtered.empty:
                return f"No learning objectives found in the outline for scope '{scope}'."

            objectives_by_topic = {}
            # groupby with sort=False preserves first-seen order of topics
            for topic, group in filtered.groupby(topic_col, sort=False):
                objs = []
                for _, row in group.iterrows():
                    sub = row[subtopic_col]
                    lo = str(row[learning_obj_col]).strip()
                    if lo:
                        objs.append(f"  • {sub}: {lo}")
                if objs:
                    objectives_by_topic[topic] = objs

            if objectives_by_topic:
                result_lines = ["All Course Learning Objectives:"]
                for topic, objs in objectives_by_topic.items():
                    result_lines.append("")
                    result_lines.append(f"{topic}:")
                    result_lines.extend(objs)

                result_lines.append("")
                result_lines.append("Ensure content supports these comprehensive learning objectives.")
                return "\n".join(result_lines)
        
        # Fallback for any scope
        return f"No specific learning objectives found for scope '{scope}'. Ensure content is educationally sound and aligned with course goals."
        
    except Exception as e:
        print(f"❌ Error fetching learning objectives: {e}")
        return f"Error retrieving learning objectives for scope '{scope}'. Ensure content supports educational goals."

slide_chunks_checklist_prompt = """Assume the role of a checklist agent tasked with evaluating the following block(s) of slide chunks for the given list of checklist criteria.

These slide chunks are created for the following:
<course_info>
Course Name: {course_name}
Target Audience: {target_audience}
</course_info>

LEARNING OBJECTIVE CONTEXT:
The slide chunks you are evaluating should align with this specific learning objective:
<learning_objective>
{learning_objective}
</learning_objective>

Here's the slide chunks to evaluate:
<slide_chunks>
{slide_chunks}
</slide_chunks>

Here are the checklist criteria to evaluate:
<checklist>
{checklist}
</checklist>

Make sure to output in the following format:
<analysis>
[Analysis of the slide chunks based on the checklist criteria AND learning objective alignment. It is okay for the analysis to be quite long for accurate evaluation.]
</analysis>

<passed_items>
[List of checklist items that passed. Each item should be a single line with the item name.]
</passed_items>

<failed_items>
[List of checklist items that failed. Each item should be a single line with the item name followed by feedback in a new line on why it failed.
Item: State the checklist item.
Feedback: Provide specific feedback on why it failed and what needs to be improved or corrected.
...
]
</failed_items>

NOTES:
- The analysis should be thorough and cover all aspects of the checklist and all of the criteria.
- The passed items should only include those that fully meet the criteria.
- Feedback of the failed items should include all necessary information for the reviser agent to understand what needs to be fixed to better align with the learning objective.

Follow the examples below to understand how to evaluate each review criteria:

<examples>
{examples}
</examples>
"""


@traceable(metadata={
    "agent_name": "slide chunks",
    "step_name": "Slide Chunks Checklist Review and Revise",
    "function_name": "run_slide_chunks_checklist_agent",
    "user_id": st.session_state.get("role", "anonymous")
})
def run_slide_chunks_checklist_agent(course_name, target_audience, slide_chunks, checklist, examples, learning_objective, llm = "gemini_2_flash"):
    """
    Run the slide chunks checklist agent with the provided parameters.
    :param course_name: Name of the course for which the slide chunks are created.
    :param target_audience: Target audience for the course.
    :param slide_chunks: The block of slide chunks to evaluate.
    :param checklist: The checklist criteria to evaluate the slide chunks against.
    :param examples: The examples to guide evaluation of the checklist criteria.
    :param learning_objective: The learning objective that the slide chunks should align with.
    :param llm: The language model to use for the agent.
    :return: A string of all the failed checklist items with feedback. 
    """
    slide_chunks_checklist_agent = Chain(llm = llm, tags = ["analysis", "passed_items", "failed_items"])

    slide_chunks_checklist_agent.add_message(
        role = "user",
        content = slide_chunks_checklist_prompt.format(
            course_name = course_name,
            target_audience = target_audience,
            slide_chunks = slide_chunks,
            checklist = checklist,
            examples = examples,
            learning_objective = learning_objective,
        )
    )

    response = slide_chunks_checklist_agent.run()

    return response["failed_items"]


slide_chunks_reviser_prompt = """You are tasked with revising the following block of slide chunks based on the feedback provided in the checklist evaluation.

Here are the slide chunks to revise:
<slide_chunks>
{slide_chunks}
</slide_chunks>

Here is the learning objective that the slide chunks should align with:
<learning_objective>
{learning_objective}
</learning_objective>

Here is the checklist feedback to consider:
<checklist_feedback>
{checklist_feedback}
</checklist_feedback>

Here are the review criteria with their corrective operations:
<criteria_with_corrective_operations>
{criteria_with_ops}
</criteria_with_corrective_operations>

Here are the revision examples that you can use as reference to understand the approach to revising the slide chunks:
<revision_examples>
{reviser_examples}
</revision_examples>

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

NOTES:
- Make use of the given set of CRUD block text tools to make the necessary revisions. 
- If creating new slide using create_blocks, ensure to follow the correct format and maintain the headers required for revision.
- The first slide of each topic must always be a "Learning Objectives" slide. Do not create or insert any type of slide before the first "Learning Objectives" slide of a topic. Also, the last slide of the topic must be a "Topic Summary" slide.
- The first slide of each sub-topic must always be a "Transition" slide. Do not create or insert any "Content" slide before the first "Transition" slide of a sub-topic. Also, the last slide of the sub-topic must be a "Summary" slide.
- These CRUD tools allow you to create, read, update, and delete blocks of text from the above slide chunk as needed.
- You can only work with one block of text at a time. Each block of text is a separate entity identified by a unique ID - Block ID.
- To implement some of the feedback, you may need to make edits to multiple blocks of text.

Corrective Operations:
These are specific instructions for fixing failed criteria items. Match each failed criteria from the checklist feedback above with its corresponding corrective operation in the criteria section below. Only apply the corrective operations for criteria that actually failed - ignore corrective operations for criteria that passed the evaluation. Use these corrective operations in combination with the feedback to make the necessary revisions.

Revision Examples:
These examples show how to apply the corrections to the slides. Use these examples as reference to understand the approach to revising the research notes. Only refer the examples of the criteria that failed.
"""


@traceable(metadata={
    "agent_name": "slide chunks",
    "step_name": "Slide Chunks Checklist Review and Revise",
    "function_name": "run_reviser_agent",
    "user_id": st.session_state.get("role", "anonymous")
})
def run_reviser_agent(slide_chunks, checklist_feedback, criteria_with_ops, reviser_examples, df, learning_objective, global_max_index=None, llm = "gemini_2_flash"):
    """
    Run the reviser agent with the provided parameters.
    :param slide_chunks: The block of slide chunks to revise.
    :param checklist_feedback: The feedback from the checklist evaluation to consider for revisions.
    :param criteria_with_ops: All criteria with their corrective operations for the reviser to reference.
    :param reviser_examples: Examples showing how to apply corrections for each criteria.
    :param df: The dataframe containing the slide chunks.
    :param global_max_index: The global maximum index from the full dataset for new block creation.
    :param llm: The language model to use for the agent.
    :return: The revised block of slide chunks as a dataframe.
    """
    print(f"\n🔄 STARTING REVISER AGENT for {len(df)} blocks")
    print(f"📝 Checklist feedback length: {len(checklist_feedback)} chars")
    print("=" * 80)

    # Store global max index in DataFrame attributes for CRUD tools to access
    if global_max_index is not None:
        df.attrs['global_max_index'] = global_max_index
        # print(f"✅ Set global_max_index in df.attrs: {global_max_index}")
        
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=1,  # <-- Super slow! We can only make a request once every 10 seconds!!
        check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
        max_bucket_size=10,  # Controls the maximum burst size.
    )

    llm = init_chat_model(
        # "google_genai:gemini-2.5-flash",
        "openai:gpt-5-mini",
        rate_limiter = rate_limiter,
        max_retries = 20,
    )

    class BufferState(AgentState):
        """
        Short-term state for the agent execution.
        Only one field is needed: the shared dataframe instance.
        """
        df: pd.DataFrame = Field(default_factory=pd.DataFrame)


    graph = create_react_agent(
        model=llm,
        tools=[create_block, read_blocks, update_block, delete_block],
        state_schema=BufferState,     # <— includes the dataframe
        # prompt=slide_chunks_reviser_prompt,
        # checkpointer=checkpointer,
    )

    # # Format the prompt for debugging (printing)
    # formatted_prompt = slide_chunks_reviser_prompt.format(
    #     slide_chunks=slide_chunks,
    #     checklist_feedback=checklist_feedback,
    #     criteria_with_ops=criteria_with_ops,
    #     reviser_examples=reviser_examples,
    # )

    # # Print the formatted prompt for debugging
    # print("\n🔍 RESEARCH NOTES REVISER PROMPT BEING SENT TO LLM:\n")
    # print(formatted_prompt)
    # print("\n" + "=" * 100 + "\n")

    state = {
        "messages": [{"role": "user", "content": slide_chunks_reviser_prompt.format(
            slide_chunks=slide_chunks,
            checklist_feedback=checklist_feedback,
            criteria_with_ops=criteria_with_ops,
            reviser_examples=reviser_examples,
            learning_objective=learning_objective,)}],
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

    return final_state["df"]

def fix_block_ids_in_text(df):
    """
    Ensures that Block ID in the text content matches the DataFrame index.
    """
    print("🔧 Fixing Block IDs in text content...")
    fixed_count = 0
    
    for idx in df.index:
        if 'block text' in df.columns and pd.notna(df.loc[idx, 'block text']):
            block_text = str(df.loc[idx, 'block text'])
            
            # Check if Block ID exists and matches index
            pattern = r"###Block ID:\s*(.+?)(?:\n|$)"
            match = re.search(pattern, block_text)
            
            if match:
                current_id = match.group(1).strip()
                if current_id != str(idx):
                    # Fix the Block ID
                    corrected_text = re.sub(r"###Block ID:\s*.+?(?=\n|$)", f"###Block ID: {idx}", block_text)
                    df.loc[idx, 'block text'] = corrected_text
                    fixed_count += 1
                    print(f"🔧 Fixed Block ID: '{current_id}' -> '{idx}' for index {idx}")
            else:
                # Add missing Block ID
                corrected_text = f"###Block ID: {idx}\n{block_text}"
                df.loc[idx, 'block text'] = corrected_text
                fixed_count += 1
                print(f"🔧 Added missing Block ID: {idx}")
    
    print(f"✅ Fixed Block IDs in {fixed_count} text blocks")
    return df


def ensure_parent_relationship(df_slice, parent_df):
    """
    Ensure that DataFrame slices maintain reference to parent DataFrame.
    This is crucial for the CRUD tools to access the full context when assigning order values.
    """
    if hasattr(df_slice, '_parent'):
        return df_slice
    
    # Set parent reference for CRUD tools to access full DataFrame context
    df_slice._parent = parent_df
    return df_slice


@traceable(
    metadata={
        "agent_name": "slide_chunks",
        "step_name": "Evaluate and Revise Slide Chunks using Checklist",
        "function_name": "run_slide_chunks_checklist_and_reviser",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def run_slide_chunks_checklist_and_reviser(sheet, course_name, target_audience, checklist_sheet_link, gc, llm = "gemini_2_flash"):
    """
    Run the slide chunks checklist agent and reviser agent with the provided parameters.
    :param course_name: Name of the course for which the slide chunks are created.
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
    checklist_worksheet, checklist_df = get_sheet_data_and_df(sheet = checklist_sheet, sheet_name = "Slide Chunks Checklist")

    # Load the slide chunks sheet
    slide_chunks_sheet, slide_chunks_df = get_sheet_data_and_df(sheet = sheet, sheet_name = "Slide Chunks")

    print(slide_chunks_df)
    if "order" not in slide_chunks_df.columns:
        # Ensure order starts from 1 (not 0)
        slide_chunks_df["order"] = slide_chunks_df.index.astype(float) + 1

    # Ensure block text column exists and all rows have block text
    if 'block text' not in slide_chunks_df.columns:
        slide_chunks_df['block text'] = ''

    # Store the original global max index for new block ID assignment
    global_max_index = slide_chunks_df.index.max()
    print(f"🔍 Global max index at start: {global_max_index}")
    
    # Generate block text for all rows that don't have it
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
            slide_chunks_str += f"####**Slide Chunks:**\n{slide_content}\n"

            slide_chunks_df.at[index, 'block text'] = slide_chunks_str

    # print(slide_chunks_df['block text'])
    scope_to_selector = {
        "Global (full output)": [],          # whole frame
        "Topic":  ["topic"] if "topic" in slide_chunks_df.columns else ["Topic"],                 # group by Topic col
        "Subtopic": (["topic", "subtopic"] if "topic" in slide_chunks_df.columns else ["Topic", "Subtopic"]),   # group by both
        "Learning Objective": "__row__",     # one row each
    }


    # Group rows so all criteria of one Task/Scope are checked together
    task_scope_groups = list(checklist_df.groupby(["Task", "Scope"], sort=False))
    total_task_scopes = len(task_scope_groups)
    
    progress = SmartProgressBar(total_tasks=total_task_scopes, save_interval=1)
    
    for (task, scope), grp in task_scope_groups:
        print(f"Running checklist for Task: {task}, Scope: {scope}")
        
        # Extract criteria and examples together
        criteria_list = grp["Review Criteria"].tolist()
        examples_list = grp["Review Agent Examples"].tolist()
        criteria_str = "\n".join(criteria_list)
        
        # Build examples with structured format for each criteria
        examples_with_separators = []
        for i, (criteria, example) in enumerate(zip(criteria_list, examples_list)):
            criteria_num = i + 1
            examples_with_separators.append(f"<criteria_{criteria_num}>\n\nReview Criteria: {criteria}\n\n<example>\n\n{example}\n\n</example>\n\n</criteria_{criteria_num}>")
        examples_str = "\n\n".join(examples_with_separators)
        
        # Build criteria with corrective operations for reviser agent
        criteria_with_ops = []
        for _, row in grp.iterrows():
            criteria = row["Review Criteria"]
            corrective_ops = row["Corrective Operations"]
            criteria_with_ops.append(f"<criteria>\n\nReview Criteria: {criteria}\n\nCorrective Operation: {corrective_ops}\n\n</criteria>")
        criteria_ops_str = "\n\n".join(criteria_with_ops)
        
        # Build examples with structured format for reviser agent
        reviser_examples_with_separators = []
        for i, (criteria, corrective_ops, reviser_example) in enumerate(zip(criteria_list, grp["Corrective Operations"].tolist(), grp["Reviser Agent Examples"].tolist())):
            criteria_num = i + 1
            reviser_examples_with_separators.append(f"<criteria_{criteria_num}>\n\nReview Criteria: {criteria}\n\nCorrective Operation: {corrective_ops}\n\n<example>\n\n{reviser_example}\n\n</example>\n\n</criteria_{criteria_num}>")
        reviser_examples_str = "\n\n".join(reviser_examples_with_separators)

        # Parallelize all topic slices for this checklist task
        results = process_slide_chunk_scope_slices_parallel(
            scope=scope,
            slide_chunks_df=slide_chunks_df,
            scope_to_selector=scope_to_selector,
            criteria_str=criteria_str,
            examples_str=examples_str,
            criteria_ops_str=criteria_ops_str,
            reviser_examples_str=reviser_examples_str,
            course_name=course_name,
            target_audience=target_audience,
            sheet=sheet,
            gc=gc,
            global_max_index=global_max_index,
            llm=llm
        )

        # Merge results back into main DataFrame
        for key, revised_slide_chunks_df, df_slice, updated_max_index in results:
            if revised_slide_chunks_df is None:
                continue

            # Update global_max_index based on parallel processing
            global_max_index = max(global_max_index, updated_max_index)

            # --- Handle new blocks (IDs already assigned in parallel process) ---
            new_indices = [idx for idx in revised_slide_chunks_df.index if idx not in slide_chunks_df.index]
            for idx in new_indices:
                # Assign a unique global block ID
                global_max_index += 1
                revised_slide_chunks_df = revised_slide_chunks_df.rename(index={idx: global_max_index})
                idx = global_max_index  # Use the new index

                # Fill topic/subtopic for new block
                if 'block text' in revised_slide_chunks_df.columns:
                    block_text = str(revised_slide_chunks_df.loc[idx, 'block text'])
                    topic_match = re.search(r'####\*\*Topic:\*\*\s*\n(.+?)(?=\n####|\n\n|\Z)', block_text, re.DOTALL)
                    subtopic_match = re.search(r'####\*\*Subtopic:\*\*\s*\n(.+?)(?=\n####|\n\n|\Z)', block_text, re.DOTALL)
                    topic_value = topic_match.group(1).strip() if topic_match else ''
                    subtopic_value = subtopic_match.group(1).strip() if subtopic_match else ''
                    topic_col = 'topic' if 'topic' in slide_chunks_df.columns else 'Topic'
                    subtopic_col = 'subtopic' if 'subtopic' in slide_chunks_df.columns else 'Subtopic'
                    if topic_col not in revised_slide_chunks_df.columns:
                        revised_slide_chunks_df[topic_col] = ''
                    if subtopic_col not in revised_slide_chunks_df.columns:
                        revised_slide_chunks_df[subtopic_col] = ''
                    revised_slide_chunks_df.loc[idx, topic_col] = topic_value
                    revised_slide_chunks_df.loc[idx, subtopic_col] = subtopic_value
                    print(f"✅ Populated new block {idx}: topic='{topic_value}', subtopic='{subtopic_value}'")

            # --- Update existing blocks ---
            existing_indices = [idx for idx in revised_slide_chunks_df.index if idx in slide_chunks_df.index]
            for idx in existing_indices:
                for col in revised_slide_chunks_df.columns:
                    if col in slide_chunks_df.columns:
                        slide_chunks_df.loc[idx, col] = revised_slide_chunks_df.loc[idx, col]

            # --- Append new blocks ---
            new_blocks_df = revised_slide_chunks_df.loc[[idx for idx in revised_slide_chunks_df.index if idx not in slide_chunks_df.index]]
            if not new_blocks_df.empty:
                slide_chunks_df = pd.concat([slide_chunks_df, new_blocks_df], ignore_index=False, sort=False)
                print(f"Appended {len(new_blocks_df)} new blocks")

            slide_chunks_df = slide_chunks_df.sort_index()
            print(f"Main DataFrame shape after merge: {slide_chunks_df.shape}")

            # Check for deleted blocks
            deleted_indices = set(df_slice.index) - set(revised_slide_chunks_df.index)
            if deleted_indices:
                print(f"🗑️ Removing deleted blocks from main DataFrame: {deleted_indices}")
                slide_chunks_df = slide_chunks_df.drop(index=deleted_indices, errors='ignore')

            # Note: Block IDs already fixed in parallel process
        progress.update()
        
    # Validate and correct block text format before parsing
    print("\n🔍 Validating and correcting block text format...")
    corrected_count = 0
    for index, row in slide_chunks_df.iterrows():
        block_text = str(row.get('block text', '')).strip()
        
        # Check if format is correct
        has_correct_format = (
            'Block ID' in block_text or 
            '####**Topic:**' in block_text
        )
        
        if not has_correct_format and block_text != '':
            # Regenerate block text with correct format
            slide_chunks_str = ""
            slide_chunks_str += f"###Block ID: {index}\n"
            slide_chunks_str += f"####**Topic:**\n{row['Topic']}\n"
            slide_chunks_str += f"####**Subtopic:**\n{row['Subtopic']}\n"
            slide_chunks_str += f"####**Slide Chunks:**\n{block_text}\n"

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
    save_to_sheet(worksheet = slide_chunks_sheet, df = slide_chunks_df)
    
    # Parse the updated block_text column back to individual columns (optional final step)
    print("\n🔄 Parsing updated block_text content back to individual columns...")
    slide_chunks_df = parse_block_text_to_columns(slide_chunks_df)
    
    # Parse slide_chunks into individual slide components
    print("\n🔄 Parsing slide_chunks into individual slide components...")
    slide_chunks_df = parse_slide_chunks_to_components(slide_chunks_df)
    
    # Delete the slide_chunks column after parsing is complete
    if 'slide_chunks' in slide_chunks_df.columns:
        slide_chunks_df = slide_chunks_df.drop(columns=['slide_chunks'])
        # print("🗑️ Deleted slide_chunks column after parsing completion.")
   
    save_to_sheet(worksheet = slide_chunks_sheet, df = slide_chunks_df)

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


# Block Text Parsing 
class BlockTextContent(BaseModel):
    topic: str = Field(description="The topic extracted from the block text.")
    subtopic: str = Field(description="The subtopic extracted from the block text.")
    learning_objective: str = Field(description="The learning objective extracted from the block text.")
    slide_chunks: str = Field(description="The slide chunks content extracted from the block text.")

block_text_parsing_prompt = """You are an expert parser for slide chunks block text. Given a block of text in a structured format, extract the following fields:

- topic: The value after '####**Topic:**' (extract only the topic text, not the header)
- subtopic: The value after '####**Subtopic:**' (extract only the subtopic text, not the header)
- slide_chunks: The value after '####**Slide Chunks:**' (extract only the slide chunks content, not the header)

Return the output as a structured object with these fields. Do not add or infer any information. Only extract what is present in the block. Preserve all formatting, line breaks, and structure in the slide_chunks field.

Block:
{block}
"""

def parse_block_text_row(block_text_cell, index):
    """
    Parses a single block_text cell to extract Topic, Subtopic, Learning Objective, and Slide Chunks.
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

@traceable(
    metadata={
        "agent_name": "slide_chunks",
        "step_name": "Parse Block Text to Individual Columns",
        "function_name": "parse_block_text_to_columns",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def parse_block_text_to_columns(df, max_workers=5):
    """
    Parses the block_text column and updates the individual Topic, Subtopic, Learning Objectives, and slide_chunks columns.
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
                    # Look for common patterns like "Title: Something" or "Type: Something"
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

def create_backup_slide_chunks(sheet):
    """
    Create a backup of the 'Slide Chunks' sheet before running the checklist.
    """
    backup_name = "Backup Slide Chunks Sheet for Delete step of Slide Chunks Checklist"
    
    # Check if we just performed a delete operation - if so, don't create new backup
    if st.session_state.get("just_deleted_checklist", False):
        st.session_state["just_deleted_checklist"] = False
        print(f"Skipping backup creation - just restored from backup")
        return
    
    # Check if backup already exists, if yes, delete it first
    sheet_names = get_worksheet_names(sheet)
    if backup_name in sheet_names:
        delete_worksheet(sheet, backup_name)
    
    # Get the original worksheet
    try:
        original_ws = sheet.worksheet("Slide Chunks")
    except Exception as e:
        print(f"Error: 'Slide Chunks' sheet not found: {e}")
        return
    
    # Duplicate the worksheet
    backup_ws = original_ws.duplicate(new_sheet_name=backup_name)
    
    # Hide the backup sheet
    hide_worksheet_by_name(sheet, backup_name)
    
    print(f"✅ Created and hid backup sheet: '{backup_name}'")

def delete_slide_chunks_checklist(sheet):
    """
    Delete function for the Slide Chunks Checklist step.
    Restores the 'Slide Chunks' sheet from backup and deletes the backup.
    """
    backup_name = "Backup Slide Chunks Sheet for Delete step of Slide Chunks Checklist"
    
    # Check if backup exists
    sheet_names = get_worksheet_names(sheet)
    if backup_name not in sheet_names:
        print(f"Backup sheet '{backup_name}' not found. Cannot restore.")
        return
    
    try:
        # Get worksheets
        slide_chunks_ws = sheet.worksheet("Slide Chunks")
        backup_ws = sheet.worksheet(backup_name)
        
        # Get data from backup
        backup_data = backup_ws.get_all_values()
        
        # Clear the current Slide Chunks sheet
        clear_worksheet(slide_chunks_ws)
        
        # Copy data from backup to Slide Chunks
        if backup_data:
            slide_chunks_ws.update(backup_data)
        
        # Delete the backup sheet
        delete_worksheet(sheet, backup_name)
        
        # Set flag to prevent immediate recreation of backup
        st.session_state["just_deleted_checklist"] = True
        
        print(f"✅ Restored 'Slide Chunks' from backup and deleted '{backup_name}'")
        
    except Exception as e:
        print(f"Error during delete operation: {e}")