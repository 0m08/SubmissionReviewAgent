from modules.chain import Chain
from tqdm import tqdm
from services.sheets_service import (
    get_sheet_data_and_df,
    save_to_sheet,
    filter_non_blank_column,
    clear_worksheet,
    delete_worksheet,
    clear_all_filters,
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

# ---------------------------------------------------------------
# 1.  Generic iterator that yields exactly the slice required
# ---------------------------------------------------------------
def iterate_scope(
    raw_scope: str,
    data_df: pd.DataFrame,
    scope_to_selector: Dict[str, Any],
    scope_parser: Callable[[str], str] = lambda s: s,
) -> Iterable[Tuple[Hashable, pd.DataFrame]]:
    """
    Parameters
    ----------
    raw_scope : the literal text from checklist_df["Scope"]
    data_df   : the dataframe being reviewed
    scope_to_selector :
        Dict mapping a *canonical scope key* (e.g. "Global", "Topic") to one of:
        • [] or () or None       → treat as *Global* (whole df in one go)
        • "__row__"              → iterate row‑by‑row
        • list/tuple of columns  → groupby those columns
        • callable(df) -> iterator[(key, slice)] for anything advanced
    scope_parser :
        Converts the raw text ("Global (full output)") to the canonical key
        used in `scope_to_selector`  – default is `first word only`.

    Yields
    ------
    (key, df_slice) pairs for each review pass.
    """
    scope_key = scope_parser(raw_scope)

    if scope_key not in scope_to_selector:
        raise ValueError(f"Scope '{scope_key}' not found in scope_to_selector")

    selector = scope_to_selector[scope_key]

    # -------- dispatch selector type --------
    if selector in (None, [], ()):
        yield "ALL", data_df

    elif selector == "__row__":
        for idx, row in data_df.iterrows():
            yield idx, row.to_frame().T  # keep slice a DataFrame

    elif callable(selector):
        # Your own function can do anything it likes
        yield from selector(data_df)

    else:
        # Assume list/tuple → groupby
        group_cols = list(selector)
        for key, grp in data_df.groupby(group_cols, dropna=False, sort=False):
            yield key, grp


research_notes_checklist_prompt = """Assume the role of a checklist agent tasked with evaluating the following block(s) of research notes for the given list of checklist criteria.

These research notes are created for the following:
<course_info>
Course Name: {course_name}
Target Audience: {target_audience}
</course_info>

Here's the research notes to evaluate:
<research_notes>
{research_notes}
</research_notes>

Here are the checklist criteria to evaluate:
<checklist>
{checklist}
</checklist>

Make sure to output in the following format:
<analysis>
[Analysis of the research notes based on the checklist criteria. It is okay for the analysis to be quite long for accurate evaluation.]
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
- Feedback of the failed items should include all necessary information for the author to understand what needs to be fixed.

Follow the examples below to understand how to evaluate each review criteria:

<examples>
{examples}
</examples>
"""


def run_research_notes_checklist_agent(course_name, target_audience, research_notes, checklist, examples, llm = "gemini_2_flash"):
    """
    Run the research notes checklist agent with the provided parameters.
    :param course_name: Name of the course for which the research notes are created.
    :param target_audience: Target audience for the course.
    :param research_notes: The block of research notes to evaluate.
    :param checklist: The checklist criteria to evaluate the research notes against.
    :param examples: The examples to guide evaluation of the checklist criteria.
    :param llm: The language model to use for the agent.
    :return: A string of all the failed checklist items with feedback. 
    """
    research_notes_checklist_agent = Chain(llm = llm, tags = ["analysis", "passed_items", "failed_items"])

    # # Format the prompt for debugging (printing)
    # formatted_prompt = research_notes_checklist_prompt.format(
    #     course_name = course_name,
    #     target_audience = target_audience,
    #     research_notes = research_notes,
    #     checklist = checklist,
    #     examples = examples,
    # )

    # # Print the formatted prompt for debugging
    # print("\n🔍 RESEARCH NOTES CHECKLIST PROMPT BEING SENT TO LLM:\n")
    # print(formatted_prompt)
    # print("\n" + "=" * 100 + "\n")

    research_notes_checklist_agent.add_message(
        role = "user",
        content = research_notes_checklist_prompt.format(
            course_name = course_name,
            target_audience = target_audience,
            research_notes = research_notes,
            checklist = checklist,
            examples = examples,
        )
    )

    response = research_notes_checklist_agent.run()

    return response["failed_items"]


research_notes_reviser_prompt = """You are tasked with revising the following block of research notes based on the feedback provided in the checklist evaluation.

Here are the research notes to revise:
<research_notes>
{research_notes}
</research_notes>

Here is the checklist feedback to consider:
<checklist_feedback>
{checklist_feedback}
</checklist_feedback>

Here are the review criteria with their corrective operations and revision examples:
<criteria_with_ops_and_examples>
{criteria_with_ops_and_examples}
</criteria_with_ops_and_examples>

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

IMPORTANT: Never change or remove these headers. Only modify the content after the header as required for the revision. The format is essential for the system to function properly.

NOTES:
- Make use of the given set of CRUD block text tools to make the necessary revisions. 
- These CRUD tools allow you to create, read, update, and delete blocks of text from the above research notes as needed.
- You can only work with one block of text at a time. Each block of text is a separate entity identified by a unique ID - Block ID.
- To implement some of the feedback, you may need to make edits to multiple blocks of text.

Corrective Examples and Revision Examples:
These are specific instructions and examples for fixing failed criteria items. Match each failed criteria from the checklist feedback above with its corresponding corrective operation and revision example below. Only apply the corrective operations for criteria that actually failed - ignore corrective operations for criteria that passed the evaluation. Use these corrective operations and revision examples in combination with the feedback to make the necessary revisions. The revision examples provide reference patterns for applying the feedback to research notes. Use the checklist feedback as your primary guide, and study these examples to understand the general approach and methodology for making the required changes, then apply similar principles to the research notes you are revising.
"""



def run_reviser_agent(research_notes, checklist_feedback, criteria_with_ops_and_examples, df, llm = "gemini_2_flash"):
    """
    Run the reviser agent with the provided parameters.
    :param research_notes: The block of research notes to revise.
    :param checklist_feedback: The feedback from the checklist evaluation to consider for revisions.
    :param criteria_with_ops: All criteria with their corrective operations for the reviser to reference.
    :param df: The dataframe containing the research notes.
    :param llm: The language model to use for the agent.
    :return: The revised block of research notes as a dataframe.
    """

    rate_limiter = InMemoryRateLimiter(
        requests_per_second=1,  # <-- Super slow! We can only make a request once every 10 seconds!!
        check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
        max_bucket_size=10,  # Controls the maximum burst size.
    )

    llm = init_chat_model(
        "google_genai:gemini-2.5-flash",
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
        # prompt=research_notes_reviser_prompt,
        # checkpointer=checkpointer,
    )

    # # Format the prompt for debugging (printing)
    # formatted_prompt = research_notes_reviser_prompt.format(
    #     research_notes=research_notes,
    #     checklist_feedback=checklist_feedback,
    #     criteria_with_ops_and_examples=criteria_with_ops_and_examples,
    # )

    # # Print the formatted prompt for debugging
    # print("\n🔍 RESEARCH NOTES REVISER PROMPT BEING SENT TO LLM:\n")
    # print(formatted_prompt)
    # print("\n" + "=" * 100 + "\n")

    state = {
        "messages": [{"role": "user", "content": research_notes_reviser_prompt.format(
            research_notes=research_notes,
            checklist_feedback=checklist_feedback,
            criteria_with_ops_and_examples=criteria_with_ops_and_examples,)}],
        "df": df,        # one buffer for the whole session
    }

    final_state = graph.invoke(
        state,
        {"recursion_limit": 50}
    )

    return final_state["df"]


def run_research_notes_checklist_and_reviser(sheet, course_name, target_audience, checklist_sheet_link, gc, llm = "gemini_2_flash"):
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

    # Load the research notes sheet
    research_notes_sheet, research_notes_df = get_sheet_data_and_df(sheet = sheet, sheet_name = "Final Outline")

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
        "Global (full output)": [],                        # whole frame
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

        # Extract criteria and examples together
        criteria_list = grp["Review Criteria"].tolist()
        examples_list = grp["Review Agent Examples"].tolist()
        
        criteria_str = "\n".join(criteria_list)
        
        # Build examples with separators between criteria
        examples_with_separators = []
        for i, (criteria, example) in enumerate(zip(criteria_list, examples_list)):
            if i > 0:
                examples_with_separators.append("------")
            examples_with_separators.append(f"Review Criteria: {criteria}\n\n{example}")
        examples_str = "\n\n".join(examples_with_separators)
        
        # Build criteria with corrective operations and reviser examples for reviser agent
        criteria_with_ops_and_examples = []
        for _, row in grp.iterrows():
            criteria = row["Review Criteria"]
            corrective_ops = row["Corrective Operations"]
            reviser_example = row["Reviser Agent Examples"]
            criteria_with_ops_and_examples.append(f"Review Criteria: {criteria}\n\nCorrective Operation: {corrective_ops}\n\nRevision Example:\n\n{reviser_example}")
        
        criteria_ops_and_examples_str = "\n\n------\n\n".join(criteria_with_ops_and_examples)

        for key, df_slice in iterate_scope(scope, research_notes_df, scope_to_selector):

            print(f"Processing {key} with {len(df_slice)} rows")

            if df_slice.empty:
                print(f"No data for {key}, skipping...")
                continue

            research_notes_str = "\n\n---\n\n".join(df_slice["block text"].tolist())

            # Run the research notes checklist agent
            failed_items = run_research_notes_checklist_agent(
                course_name=course_name,
                target_audience=target_audience,
                research_notes=research_notes_str,
                checklist=criteria_str,
                examples=examples_str,
                llm=llm
            )

            print(f"Failed items for {key}: {failed_items}")

            if not failed_items:
                print(f"All items passed for {key}.")
                continue

            # Run the reviser agent with the failed items and criteria with corrective operations
            revised_research_notes_df = run_reviser_agent(
                research_notes=research_notes_str,
                checklist_feedback=failed_items,
                criteria_with_ops_and_examples=criteria_ops_and_examples_str,
                df=df_slice,
                llm=llm
            )

            print(f"Revised research notes for {key}: {revised_research_notes_df}")

            # Merge the revised slice back into the main DataFrame
            # Use the original slice indices to ensure correct mapping
            for idx in df_slice.index:
                if idx in revised_research_notes_df.index:
                    research_notes_df.loc[idx, 'block text'] = revised_research_notes_df.loc[idx, 'block text']
                    print(f"Updated block text for index {idx}")
                else:
                    print(f"Warning: Index {idx} not found in revised DataFrame")

            # print(revised_research_notes_df)
        
        progress.update()
        
    # Validate and correct block text format before parsing
    print("\n🔍 Validating and correcting block text format...")
    corrected_count = 0
    for index, row in research_notes_df.iterrows():
        block_text = str(row.get('block text', '')).strip()
        
        # Check if format is correct
        has_correct_format = 'Block ID' in block_text
        
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
            print(f"  Corrected format for row {index}")
    
    print(f"✅ Corrected format for {corrected_count} rows")
    
    # Parse the updated block_text column back to individual columns
    print("\n🔄 Parsing updated block_text content back to individual columns...")
    research_notes_df = parse_block_text_to_columns(research_notes_df)
    print("✅ Block text parsing completed. Individual columns updated with revised content.")
    
    # Save to sheet
    save_to_sheet(worksheet = research_notes_sheet, df = research_notes_df)

    return



# Define the checklist prompt and code to run it for a given task - run outline checklist

## Prompt for outline checklist - output should be a pass or fail + feedback for each checklist item

## Prompt for reviser agent - output will be revised block of text based on the checklist feedback

## Reviser agent will have access to the CRUD block text tools, will have same input text as the outline checklist agent.

# Block Text Parsing functionality
from pydantic import BaseModel, Field

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
    
    # # Remove rows with empty block text
    # initial_count = len(df)
    
    # # Debug: Check column names and data
    # print(f"🔍 DataFrame columns: {list(df.columns)}")
    # print(f"🔍 DataFrame shape before cleaning: {df.shape}")
    
    # # Check if 'block text' column exists
    # if 'block text' not in df.columns:
    #     print("❌ 'block text' column not found!")
    #     return df
    
    # # More robust empty detection - handle NaN, empty strings, whitespace-only strings
    # def is_empty_block_text(value):
    #     # Handle NaN/None values
    #     if pd.isna(value) or value is None:
    #         return True
    #     # Handle empty strings or whitespace-only strings
    #     if isinstance(value, str):
    #         return value.strip() == ''
    #     # Handle other types that might be considered empty
    #     if value == '' or value == 'nan' or value == 'None':
    #         return True
    #     return False
    
    # # Debug: Check the last few rows specifically
    # print(f"🔍 Last 5 rows of 'block text' column:")
    # for i in range(max(0, len(df)-5), len(df)):
    #     if i < len(df):
    #         block_text_value = df.iloc[i]['block text']
    #         print(f"  Row {i}: '{repr(block_text_value)}' (type: {type(block_text_value)})")
    #         # Also check if our empty detection function thinks it's empty
    #         is_empty = is_empty_block_text(block_text_value)
    #         print(f"    -> is_empty_block_text() returns: {is_empty}")
    
    # # Debug: Check what we're finding
    # empty_rows = df[df['block text'].apply(is_empty_block_text)]
    # if not empty_rows.empty:
    #     print(f"🔍 Found {len(empty_rows)} rows with empty block text:")
    #     for idx, row in empty_rows.iterrows():
    #         print(f"  Row {idx}: block_text = '{repr(row['block text'])}'")
    # else:
    #     print("🔍 No empty rows found - this might be the issue!")
    
    # # Apply the filter
    # df = df[~df['block text'].apply(is_empty_block_text)]
    # removed_count = initial_count - len(df)
    
    # if removed_count > 0:
    #     print(f"🧹 Removed {removed_count} rows with empty block text.")
    #     # Reset index to ensure clean sequential numbering
    #     df = df.reset_index(drop=True)
    # else:
    #     print("✅ No rows with empty block text found to remove.")
    
    # print(f"🔍 DataFrame shape after cleaning: {df.shape}")
    
    # return df