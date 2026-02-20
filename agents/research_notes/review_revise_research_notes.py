from modules.chain import Chain
from langsmith import traceable
import streamlit as st
import difflib
import pandas as pd
from typing import Annotated

from langchain.tools import tool
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage, HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langgraph.prebuilt import InjectedState
from pydantic import Field


# ─────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────
class ResearchNotesState(AgentState):
    """
    State for the research notes reviser agent.
    Uses a dict wrapper for research_notes so tools can mutate it in-place
    (strings are immutable; dicts are mutable references — same pattern as
    the DataFrame in crud_text_block_tools.py).
    """
    research_notes: dict = Field(default_factory=lambda: {"text": ""})
    row_id: int = Field(default=-1)


# ─────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────
def is_failed_items_empty(failed_items: str, min_length: int = 50) -> bool:
    """
    Check if the failed_items content is effectively empty.

    The reviewer agent sometimes fills <failed_items> with placeholder text like
    "None", "N/A", or empty whitespace. Genuine feedback needs criteria name +
    explanation (50+ chars).

    :param failed_items: The failed items string from the reviewer.
    :param min_length: Minimum length threshold.
    :return: True if effectively empty, False otherwise.
    """
    if not failed_items:
        return True
    stripped = failed_items.strip()
    if not stripped:
        return True
    if len(stripped) < min_length:
        return True
    return False


def compute_text_diff(original: str, revised: str) -> str:
    """
    Compute a unified diff between two research notes strings.

    :param original: Research notes before revision.
    :param revised: Research notes after revision.
    :return: Formatted diff string.
    """
    original_lines = original.splitlines(keepends=True)
    revised_lines = revised.splitlines(keepends=True)

    diff = difflib.unified_diff(
        original_lines,
        revised_lines,
        fromfile="Research Notes (before)",
        tofile="Research Notes (after)",
        lineterm=""
    )
    diff_text = "".join(diff)

    if not diff_text.strip():
        return "No changes detected."

    return diff_text


# ─────────────────────────────────────────────────────────────────────────
# TOOL FACTORIES  (tools that need external dependencies via closure)
# ─────────────────────────────────────────────────────────────────────────
def create_lookup_fallback_context_tool(sheet):
    """Factory: creates a lookup_fallback_context tool with sheet access."""

    @tool("lookup_fallback_context", parse_docstring=True)
    def lookup_fallback_context(
        row_id: Annotated[int, InjectedState("row_id")],
        intent: str,
    ) -> str:
        """Look up the fallback context from the Context Column Backup sheet.

        Use this tool FIRST when information is missing from the research notes.
        It retrieves the original source context documents that were used to
        generate the research notes for this row.

        Args:
            intent: Description of what information you are looking for.

        Returns:
            The fallback context content for the current row.
        """
        print(f"🔧 TOOL USED: lookup_fallback_context | row_id: {row_id} | intent: {intent}")
        try:
            backup_ws = sheet.worksheet("Context Column Backup")
            backup_df = pd.DataFrame(backup_ws.get_all_records())

            # Find the row matching our row_id
            backup_df["row_index"] = backup_df["row_index"].astype(int)
            matching_rows = backup_df[backup_df["row_index"] == row_id]

            if matching_rows.empty:
                return f"No fallback context found for row {row_id}."

            row = matching_rows.iloc[0]
            context_parts = []
            for col in sorted(row.index):
                if col.startswith("context_") and str(row[col]).strip():
                    context_parts.append(str(row[col]))

            if not context_parts:
                return f"Fallback context columns are empty for row {row_id}."

            return f"Fallback context for row {row_id}:\n\n{''.join(context_parts)}"
        except Exception as e:
            return f"Error reading fallback context: {str(e)}"

    return lookup_fallback_context


def create_search_vectorstore_tool(compression_retriever):
    """Factory: creates a search_vectorstore tool with the compression retriever."""

    @tool("search_vectorstore", parse_docstring=True)
    def search_vectorstore(
        query: str,
        intent: str,
    ) -> str:
        """Search the course document vectorstore for relevant information.

        Use this tool when lookup_fallback_context did not provide the
        information you need. It searches across all course documents.

        Args:
            query: The search query to find relevant information.
            intent: Description of what information you are looking for.

        Returns:
            Retrieved document content relevant to the query.
        """
        print(f"🔧 TOOL USED: search_vectorstore | query: {query} | intent: {intent}")
        try:
            docs = compression_retriever.invoke(query)
            if not docs:
                return f"No relevant documents found for query: {query}"

            results = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "Unknown")
                results.append(f"--- Document {i} (Source: {source}) ---\n{doc.page_content}\n")

            return "\n".join(results)
        except Exception as e:
            return f"Error searching vectorstore: {str(e)}"

    return search_vectorstore


def create_search_web_tool(web_search_retriever):
    """Factory: creates a search_web tool using the Exa web search retriever."""

    @tool("search_web", parse_docstring=True)
    def search_web(
        query: str,
        intent: str,
    ) -> str:
        """Search the web using Exa Search for relevant information.

        Use this tool when vectorstore search did not provide good results.

        Args:
            query: The search query.
            intent: Description of what information you are looking for.

        Returns:
            Web search results relevant to the query.
        """
        print(f"🔧 TOOL USED: search_web | query: {query} | intent: {intent}")
        try:
            docs = web_search_retriever.invoke(query)
            if not docs:
                return f"No web results found for query: {query}"

            results = []
            for i, doc in enumerate(docs, 1):
                url = doc.metadata.get("url", "Unknown")
                title = doc.metadata.get("title", "No title")
                results.append(f"--- Result {i}: {title} ({url}) ---\n{doc.page_content[:2000]}\n")

            return "\n".join(results)
        except Exception as e:
            return f"Error searching web: {str(e)}"

    return search_web


# ─────────────────────────────────────────────────────────────────────────
# STANDALONE TOOLS
# ─────────────────────────────────────────────────────────────────────────
@tool("ask_gemini_with_google", parse_docstring=True)
def ask_gemini_with_google(
    query: str,
    intent: str,
) -> str:
    """Search using Google Search with Gemini grounding as a last resort.

    Use this tool only when all other search tools (lookup_fallback_context,
    search_vectorstore, search_web) have failed to provide the needed information.

    Args:
        query: The search query to find information about.
        intent: Description of what information you are looking for.

    Returns:
        The answer based on Google Search results via Gemini.
    """
    print(f"🔧 TOOL USED: ask_gemini_with_google | query: {query} | intent: {intent}")

    from google import genai
    from google.genai import types

    client = genai.Client()

    grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )
    url_context_tool = types.Tool(
        url_context=types.UrlContext()
    )

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True
        ),
        tools=[grounding_tool, url_context_tool]
    )

    focused_prompt = f"""Search Query: {query}
Intent: {intent}

Instructions: Based on the search results, provide a concise and specific answer that directly addresses the query and intent above.
- Focus only on information relevant to the query
- Avoid unnecessary details, tangents, or general background information
- Be factual and precise
- Keep the response brief and actionable"""

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=focused_prompt,
        config=config,
    )

    if response and response.text:
        return f"🔍 Search results for '{query}':\n\n{response.text}"
    else:
        return f"⚠️ No results found for query: {query}"


@tool("str_replace", parse_docstring=True)
def rn_str_replace(
    research_notes: Annotated[dict, InjectedState("research_notes")],
    old_text: str,
    new_text: str,
    intent: str,
) -> str:
    """Replace a specific substring within the research notes with precision.

    This tool performs exact string matching and replacement. The old_text must
    match the research notes content exactly, including all whitespace and
    indentation. Must appear exactly once.

    Args:
        old_text: The exact text to find and replace. Must match exactly,
                  including whitespace. Must appear exactly once.
        new_text: The text to replace old_text with. Must differ from old_text.
                  Use an empty string to delete the matched text.
        intent: A field that describes the intent of the change.

    Returns:
        A confirmation message.

    Raises:
        ValueError: If old_text is not found, equals new_text, or appears multiple times.
    """
    print(
        f"🔧 TOOL USED: str_replace | "
        f"old_text_length: {len(old_text)} chars | new_text_length: {len(new_text)} chars | "
        f"intent: {intent}"
    )

    current_text = research_notes["text"]

    if old_text == new_text:
        raise ValueError("old_text and new_text are identical. No replacement needed.")

    match_count = current_text.count(old_text)

    if match_count == 0:
        raise ValueError(
            "old_text not found in the research notes. "
            "Ensure the text matches exactly, including whitespace and indentation."
        )

    if match_count > 1:
        raise ValueError(
            f"old_text appears {match_count} times in the research notes. "
            "Include more surrounding context to uniquely identify the target location."
        )

    research_notes["text"] = current_text.replace(old_text, new_text, 1)
    return "✏️ Replaced text in research notes."


@tool("full_overwrite", parse_docstring=True)
def full_overwrite(
    research_notes: Annotated[dict, InjectedState("research_notes")],
    new_text: str,
    intent: str,
) -> str:
    """Fully overwrite the current research notes with new content.

    Use this tool when the research notes need substantial restructuring
    that cannot be achieved with targeted str_replace edits.

    Args:
        new_text: The complete new research notes to replace all current content.
        intent: Description of why a full overwrite is needed.

    Returns:
        A confirmation message with the new length.
    """
    print(f"🔧 TOOL USED: full_overwrite | new_text_length: {len(new_text)} chars | intent: {intent}")
    research_notes["text"] = new_text
    return f"✅ Research notes fully overwritten ({len(new_text)} chars)."


@tool("stop", parse_docstring=True, return_direct=True)
def rn_stop(
    reason: str,
) -> str:
    """Signal that all revisions are complete and the agent should stop.

    Call this tool when you have finished making all necessary revisions
    to the research notes.

    Args:
        reason: A brief explanation of what was accomplished.

    Returns:
        A confirmation message.
    """
    print(f"🛑 TOOL USED: stop | reason: {reason}")
    return f"✅ Agent completed. Reason: {reason}"


# ─────────────────────────────────────────────────────────────────────────
# TOOL BUILDER
# ─────────────────────────────────────────────────────────────────────────
def _build_tools(sheet, compression_retriever=None, web_search_retriever=None):
    """Assemble the tools list for the reviser agent."""
    tools = [create_lookup_fallback_context_tool(sheet)]

    if compression_retriever:
        tools.append(create_search_vectorstore_tool(compression_retriever))

    if web_search_retriever:
        tools.append(create_search_web_tool(web_search_retriever))

    tools.append(ask_gemini_with_google)
    tools.append(rn_str_replace)
    tools.append(full_overwrite)
    tools.append(rn_stop)

    return tools


# ─────────────────────────────────────────────────────────────────────────
# REVIEWER  (Chain-based, same pattern as research_notes_checklist.py)
# ─────────────────────────────────────────────────────────────────────────
review_research_notes_prompt = """You are an expert reviewer evaluating research notes for a specific subtopic within an E-learning course.

Course Name: {course_name}
Target Audience: {target_audience}

Full Course Outline with Learning Objectives:
<full_course_outline>
{full_course_outline}
</full_course_outline>

Subtopic and Learning Objectives:
<subtopic_and_los>
{subtopic_and_los}
</subtopic_and_los>

Research Notes to Review:
<research_notes>
{research_notes}
</research_notes>

Evaluate the research notes strictly against these THREE criteria:

1. **Out of Scope Content**: Does the content contain information that does not belong for the given subtopic and learning objectives? This includes tangential information, content that belongs to other subtopics in the course outline, or irrelevant details that distract from the focused learning objectives.

2. **Missing Information**: Is there important information missing from the research notes that should be present to adequately cover the subtopic and learning objectives? Consider whether key concepts, definitions, examples, or explanations that are essential for the target audience are absent.

3. **Poor Flow**: Do the notes flow poorly? This includes bad structure, illogical ordering of information, abrupt transitions between ideas, repetitive content, or lack of coherent organization that would make it difficult for the target audience to follow.

Important rules:
- Evaluate strictly against the criteria above, not general best practices.
- Be specific in your feedback — cite exact passages or describe exact gaps.
- If a criterion is fully satisfied, do not list it in failed_items.

Output your evaluation in this exact format:

<analysis>
Provide a thorough analysis of the research notes against all three criteria. Explain your reasoning for each criterion.
</analysis>

<failed_items>
For each criterion that failed, provide:

Item: [Criterion name — one of: Out of Scope Content, Missing Information, Poor Flow]
Feedback: [Specific, actionable feedback explaining what is wrong and what needs to change]

If ALL criteria pass, leave this section empty.
</failed_items>
"""

review_research_notes_followup_prompt = """The reviser has made changes to the research notes. Here are the diffs from the last revision:

<text_diffs>
{text_diffs}
</text_diffs>

Here are the updated research notes after revision:
<research_notes>
{research_notes}
</research_notes>

Re-evaluate the research notes against the same three criteria (Out of Scope Content, Missing Information, Poor Flow). Check whether previous issues have been addressed and identify any remaining or new issues.

<analysis>
[Your analysis here]
</analysis>

<failed_items>
[Failed items, or empty if all pass]
</failed_items>
"""


@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Research Notes Reviewer",
    "function_name": "review_research_notes",
    "user_id": st.session_state.get("role", "anonymous")
})
def review_research_notes(
    course_name, target_audience, course_outline, subtopic_and_los,
    research_notes, llm="gemini_3_flash",
    message_history=None, text_diffs=None
):
    """
    Review research notes for quality issues.

    :param course_name: Course name.
    :param target_audience: Target audience.
    :param course_outline: Full course outline with LOs.
    :param subtopic_and_los: Subtopic + Learning Objectives string.
    :param research_notes: The research notes text to review.
    :param llm: LLM to use for the reviewer.
    :param message_history: Previous message history for continuity.
    :param text_diffs: Diffs from the last revision (for subsequent iterations).
    :return: (analysis, failed_items, updated_message_history)
    """
    reviewer = Chain(llm=llm, tags=["analysis", "failed_items"])

    if message_history:
        # Continue from previous conversation
        reviewer.add_messages(message_history)
        reviewer.add_message(
            role="user",
            content=review_research_notes_followup_prompt.format(
                text_diffs=text_diffs,
                research_notes=research_notes,
            )
        )
    else:
        # First iteration — full prompt
        reviewer.add_message(
            role="user",
            content=review_research_notes_prompt.format(
                course_name=course_name,
                target_audience=target_audience,
                full_course_outline=course_outline,
                subtopic_and_los=subtopic_and_los,
                research_notes=research_notes,
            )
        )

    response = reviewer.run()
    updated_history = reviewer.messages_list

    return response["analysis"], response["failed_items"], updated_history


# ─────────────────────────────────────────────────────────────────────────
# REVISER  (LangGraph agent, same pattern as research_notes_checklist.py)
# ─────────────────────────────────────────────────────────────────────────
rn_reviser_prompt = """You are an expert educational content reviser. Your task is to revise research notes based on review feedback.

Course Name: {course_name}
Target Audience: {target_audience}

Course Outline with Learning Objectives:
<full_course_outline>
{full_course_outline}
</full_course_outline>

Subtopic and Learning Objectives:
<subtopic_and_los>
{subtopic_and_los}
</subtopic_and_los>

Current Research Notes:
<research_notes>
{research_notes}
</research_notes>

Review Analysis:
<review_analysis>
{review_analysis}
</review_analysis>

Failed Items to Address:
<failed_items>
{failed_items}
</failed_items>

Instructions:
- Address each failed item using the available tools.
- For MISSING INFORMATION, follow this tool priority order:
  1. First try `lookup_fallback_context` to check the original context documents.
  2. If that doesn't have what you need, try `search_vectorstore`.
  3. If vectorstore doesn't help, try `search_web`.
  4. As a last resort, use `ask_gemini_with_google`.
- For OUT OF SCOPE CONTENT: Use `str_replace` to remove or replace the out-of-scope content. Replace removed content with relevant material if appropriate, or simply delete it.
- For POOR FLOW: Use `str_replace` for targeted fixes (reordering sentences, improving transitions). Use `full_overwrite` only when the entire structure needs to be reorganized.
- Use `str_replace` for targeted, precise edits. Use `full_overwrite` only when major restructuring is truly required.
- When adding new information from search tools, integrate it naturally into the existing notes.
- After all revisions are complete, call `stop` to signal completion.

Follow these Paraphrasing Rules in all revisions:
1. Use plain language, not corporate jargon
2. Sound like a real technician, not an AI
3. Keep all factual information accurate (never add or remove facts)
4. Use active voice and short sentences
5. Explain trade terminology when first used
6. Be conversational but professional
7. No buzzwords like "optimal," "leverage," "utilize," "facilitate"
8. No hype or fluff — every sentence adds value
"""


@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Research Notes Reviser",
    "function_name": "run_rn_reviser_agent",
    "user_id": st.session_state.get("role", "anonymous")
})
def run_rn_reviser_agent(
    research_notes, review_analysis, failed_items,
    course_name, target_audience, course_outline, subtopic_and_los,
    row_id, tools, llm="gemini_3_flash",
    message_history=None, text_diffs=None
):
    """
    Run the reviser agent to fix issues identified by the reviewer.

    :param research_notes: Current research notes text.
    :param review_analysis: Analysis from the reviewer.
    :param failed_items: Failed criteria feedback from the reviewer.
    :param course_name: Course name.
    :param target_audience: Target audience.
    :param course_outline: Full course outline with LOs.
    :param subtopic_and_los: Subtopic + Learning Objectives string.
    :param row_id: DataFrame row index for fallback context lookup.
    :param tools: List of tool functions for the agent.
    :param llm: LLM identifier (unused — hardcoded to Gemini for agent).
    :param message_history: Previous messages for continuity.
    :param text_diffs: Diffs from previous revision for context.
    :return: (revised_research_notes_string, updated_message_history)
    """
    from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

    print(f"\n🔄 STARTING REVISER AGENT for row {row_id}")
    print(f"📝 Failed items length: {len(failed_items)} chars")
    print("=" * 80)

    rate_limiter = InMemoryRateLimiter(
        requests_per_second=1,
        check_every_n_seconds=0.1,
        max_bucket_size=10,
    )

    llm_instance = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        thinking_level="high",
        include_thoughts=True,
        rate_limiter=rate_limiter,
        max_retries=20,
    )

    @wrap_tool_call
    def handle_tool_errors(request, handler):
        """Handle tool execution errors with custom messages."""
        try:
            return handler(request)
        except Exception as e:
            return ToolMessage(
                content=f"Tool error: Please check your input and try again.\n({str(e)})",
                tool_call_id=request.tool_call["id"]
            )

    print(f"🔧 Tools available to reviser: {[t.name for t in tools]}")

    graph = create_agent(
        model=llm_instance,
        tools=tools,
        state_schema=ResearchNotesState,
        middleware=[handle_tool_errors],
    )

    if message_history:
        # Continue from previous conversation — append follow-up
        follow_up_content = f"""The reviewer has re-evaluated the research notes after your previous revisions. Here's what you changed:

<your_previous_changes>
{text_diffs}
</your_previous_changes>

The reviewer found the following remaining issues:

<new_review_analysis>
{review_analysis}
</new_review_analysis>

<remaining_failed_items>
{failed_items}
</remaining_failed_items>

Please address these remaining issues using the available tools. Remember to call the `stop` tool when you have completed all revisions."""
        messages = message_history + [HumanMessage(content=follow_up_content)]
    else:
        # First iteration — send the full initial prompt
        messages = [HumanMessage(content=rn_reviser_prompt.format(
            research_notes=research_notes,
            review_analysis=review_analysis,
            failed_items=failed_items,
            course_name=course_name,
            target_audience=target_audience,
            full_course_outline=course_outline,
            subtopic_and_los=subtopic_and_los,
        ))]

    state = {
        "messages": messages,
        "research_notes": {"text": research_notes},
        "row_id": row_id,
    }

    final_state = graph.invoke(
        state,
        {"recursion_limit": 50}
    )

    print("=" * 80)
    print("✅ REVISER AGENT COMPLETED")
    print(f"📊 Final research notes length: {len(final_state['research_notes']['text'])} chars")
    print("=" * 80)

    return final_state["research_notes"]["text"], final_state["messages"]


# ─────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────
@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Researcher (Review-Revise)",
    "function_name": "generate_review_revise_research_notes",
    "user_id": st.session_state.get("role", "anonymous")
})
def generate_review_revise_research_notes(
    course_name, target_audience, course_outline, subtopic_and_los,
    relevant_documents, row_id, sheet, compression_retriever=None,
    web_search_retriever=None, llm="gemini_3_flash", max_iterations=3,
    reviewer_llm="gemini_3_flash"
):
    """
    Generate research notes with a review-revise quality loop.

    Calls the existing generate_research_notes for initial generation, then
    runs a review-revise loop (up to max_iterations) to improve quality.

    :param course_name: Course name.
    :param target_audience: Target audience.
    :param course_outline: Full course outline with LOs.
    :param subtopic_and_los: Subtopic + Learning Objectives string.
    :param relevant_documents: Context documents for initial generation.
    :param row_id: DataFrame row index (for fallback context lookup).
    :param sheet: Google Sheet object (for fallback context lookup).
    :param compression_retriever: Optional ContextualCompressionRetriever for vectorstore search.
    :param web_search_retriever: Optional web search retriever (ExaSearch).
    :param llm: LLM for initial generation.
    :param max_iterations: Max review-revise cycles (default 3).
    :param reviewer_llm: LLM for the reviewer.
    :return: Final research notes string.
    """
    # Step 1: Generate initial research notes (lazy import to avoid circular dependency)
    from agents.research_notes.generate_notes import generate_research_notes
    print(f"📝 Generating initial research notes for row {row_id}...")
    research_notes = generate_research_notes(
        course_name=course_name,
        target_audience=target_audience,
        course_outline=course_outline,
        subtopic_and_los=subtopic_and_los,
        relevant_documents=relevant_documents,
        llm=llm,
    )
    print(f"✅ Initial research notes generated ({len(research_notes)} chars)")

    # Step 2: Build tools for the reviser agent
    tools = _build_tools(sheet, compression_retriever, web_search_retriever)

    # Step 3: Review-Revise loop
    reviewer_history = None
    reviser_history = None
    last_text_diffs = None

    for iteration in range(1, max_iterations + 1):
        print(f"\n🔄 Review-Revise iteration {iteration}/{max_iterations} for row {row_id}")

        # Run reviewer
        analysis, failed_items, reviewer_history = review_research_notes(
            course_name=course_name,
            target_audience=target_audience,
            course_outline=course_outline,
            subtopic_and_los=subtopic_and_los,
            research_notes=research_notes,
            llm=reviewer_llm,
            message_history=reviewer_history,
            text_diffs=last_text_diffs,
        )

        print(f"📋 Review result (iteration {iteration}): failed_items length = {len(failed_items)} chars")

        # Check if all criteria pass
        if is_failed_items_empty(failed_items):
            print(f"✅ All criteria passed at iteration {iteration} for row {row_id}")
            break

        # Save notes before revision for diff computation
        notes_before = research_notes

        # Run reviser agent
        research_notes, reviser_history = run_rn_reviser_agent(
            research_notes=research_notes,
            review_analysis=analysis,
            failed_items=failed_items,
            course_name=course_name,
            target_audience=target_audience,
            course_outline=course_outline,
            subtopic_and_los=subtopic_and_los,
            row_id=row_id,
            tools=tools,
            message_history=reviser_history,
            text_diffs=last_text_diffs,
        )

        # Compute diff for next iteration
        last_text_diffs = compute_text_diff(notes_before, research_notes)
        print(f"📊 Revision {iteration} complete. Diff length: {len(last_text_diffs)} chars")

    return research_notes
