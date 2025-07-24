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
- The analysis should be thorough and cover all aspects of the checklist.
- The passed items should only include those that fully meet the criteria.
- Feedback of the failed items should include all necessary information for the author to understand what needs to be fixed.
"""


def run_research_notes_checklist_agent(course_name, target_audience, research_notes, checklist, llm = "gemini_2_flash"):
    """
    Run the research notes checklist agent with the provided parameters.
    :param course_name: Name of the course for which the research notes are created.
    :param target_audience: Target audience for the course.
    :param research_notes: The block of research notes to evaluate.
    :param checklist: The checklist criteria to evaluate the research notes against.
    :param llm: The language model to use for the agent.
    :return: A string of all the failed checklist items with feedback. 
    """
    research_notes_checklist_agent = Chain(llm = llm, tags = ["analysis", "passed_items", "failed_items"])

    research_notes_checklist_agent.add_message(
        role = "user",
        content = research_notes_checklist_prompt.format(
            course_name = course_name,
            target_audience = target_audience,
            research_notes = research_notes,
            checklist = checklist,
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

NOTES:
- Make use of the given set of CRUD block text tools to make the necessary revisions. 
- These CRUD tools allow you to create, read, update, and delete blocks of text from the above research notes as needed.
- You can only work with one block of text at a time. Each block of text is a separate entity identified by a unique ID - Block ID.
- To implement some of the feedback, you may need to make edits to multiple blocks of text.
"""



def run_reviser_agent(research_notes, checklist_feedback, df, llm = "gemini_2_flash"):
    """
    Run the reviser agent with the provided parameters.
    :param research_notes: The block of research notes to revise.
    :param checklist_feedback: The feedback from the checklist evaluation to consider for revisions.
    :param df: The dataframe containing the research notes.
    :param llm: The language model to use for the agent.
    :return: The revised block of research notes as a dataframe.
    """

    rate_limiter = InMemoryRateLimiter(
        requests_per_second=0.1,  # <-- Super slow! We can only make a request once every 10 seconds!!
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

    state = {
        "messages": [{"role": "user", "content": research_notes_reviser_prompt.format(
            research_notes=research_notes,
            checklist_feedback=checklist_feedback,)}],
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

    if 'block text' not in research_notes_df.columns:
        for index, row in research_notes_df.iterrows():
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
    for (task, scope), grp in tqdm(checklist_df.groupby(["Task", "Scope"], sort = False)):
        
        print(f"Running checklist for Task: {task}, Scope: {scope}")

        criteria_list = grp["Review Criteria"].tolist()

        criteria_str = "\n".join(criteria_list)

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
                llm=llm
            )

            print(f"Failed items for {key}: {failed_items}")

            if not failed_items:
                print(f"All items passed for {key}.")
                continue

            # Run the reviser agent with the failed items as feedback
            revised_research_notes_df = run_reviser_agent(
                research_notes=research_notes_str,
                checklist_feedback=failed_items,
                df=df_slice,
                llm=llm
            )

            print(f"Revised research notes for {key}: {revised_research_notes_df}")

            # print(revised_research_notes_df)
        
    # Save to sheet
    save_to_sheet(worksheet = research_notes_sheet, df = research_notes_df)

    return



# Define the checklist prompt and code to run it for a given task - run outline checklist

## Prompt for outline checklist - output should be a pass or fail + feedback for each checklist item

## Prompt for reviser agent - output will be revised block of text based on the checklist feedback

## Reviser agent will have access to the CRUD block text tools, will have same input text as the outline checklist agent.