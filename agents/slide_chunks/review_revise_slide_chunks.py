from modules.chain import Chain
from langsmith import traceable
import streamlit as st
import difflib
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
class SlideChunksState(AgentState):
    """
    State for the slide chunks reviser agent.
    Uses a dict wrapper so tools can mutate the text in-place.
    """
    slide_chunks: dict = Field(default_factory=lambda: {"text": ""})


# ─────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────
def is_failed_items_empty(failed_items: str, min_length: int = 50) -> bool:
    """
    Check if the failed_items content is effectively empty.

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
    Compute a unified diff between two slide chunks strings.

    :param original: Slide chunks before revision.
    :param revised: Slide chunks after revision.
    :return: Formatted diff string.
    """
    original_lines = original.splitlines(keepends=True)
    revised_lines = revised.splitlines(keepends=True)

    diff = difflib.unified_diff(
        original_lines,
        revised_lines,
        fromfile="Slide Chunks (before)",
        tofile="Slide Chunks (after)",
        lineterm=""
    )
    diff_text = "".join(diff)

    if not diff_text.strip():
        return "No changes detected."

    return diff_text


def _build_research_notes_context(subtopics_data):
    """
    Build a formatted string of all research notes from the subtopics data
    for the reviewer to cross-reference against slide chunks.

    :param subtopics_data: List of dicts with keys: subtopic, learning_objectives, research_notes.
    :return: Formatted string with all research notes organized by subtopic.
    """
    blocks = []
    for item in subtopics_data:
        subtopic = item["subtopic"]
        notes = item["research_notes"]
        if notes:
            notes_block = "\n\n".join(notes)
            blocks.append(
                f"Subtopic: {subtopic}\n"
                f"Research Notes:\n{notes_block}"
            )
    return "\n\n---\n\n".join(blocks)


# ─────────────────────────────────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────────────────────────────────
@tool("str_replace", parse_docstring=True)
def sc_str_replace(
    slide_chunks: Annotated[dict, InjectedState("slide_chunks")],
    old_text: str,
    new_text: str,
    intent: str,
) -> str:
    """Replace a specific substring within the slide chunks with precision.

    This tool performs exact string matching and replacement. The old_text must
    match the slide chunks content exactly, including all whitespace and
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

    current_text = slide_chunks["text"]

    if old_text == new_text:
        raise ValueError("old_text and new_text are identical. No replacement needed.")

    match_count = current_text.count(old_text)

    if match_count == 0:
        raise ValueError(
            "old_text not found in the slide chunks. "
            "Ensure the text matches exactly, including whitespace and indentation."
        )

    if match_count > 1:
        raise ValueError(
            f"old_text appears {match_count} times in the slide chunks. "
            "Include more surrounding context to uniquely identify the target location."
        )

    slide_chunks["text"] = current_text.replace(old_text, new_text, 1)
    return "✏️ Replaced text in slide chunks."


@tool("full_overwrite", parse_docstring=True)
def sc_full_overwrite(
    slide_chunks: Annotated[dict, InjectedState("slide_chunks")],
    new_text: str,
    intent: str,
) -> str:
    """Fully overwrite the current slide chunks with new content.

    Use this tool when the slide chunks need substantial restructuring
    that cannot be achieved with targeted str_replace edits.

    Args:
        new_text: The complete new slide chunks to replace all current content.
        intent: Description of why a full overwrite is needed.

    Returns:
        A confirmation message with the new length.
    """
    print(f"🔧 TOOL USED: full_overwrite | new_text_length: {len(new_text)} chars | intent: {intent}")
    slide_chunks["text"] = new_text
    return f"✅ Slide chunks fully overwritten ({len(new_text)} chars)."


@tool("stop", parse_docstring=True, return_direct=True)
def sc_stop(
    reason: str,
) -> str:
    """Signal that all revisions are complete and the agent should stop.

    Call this tool when you have finished making all necessary revisions
    to the slide chunks.

    Args:
        reason: A brief explanation of what was accomplished.

    Returns:
        A confirmation message.
    """
    print(f"🛑 TOOL USED: stop | reason: {reason}")
    return f"✅ Agent completed. Reason: {reason}"


# ─────────────────────────────────────────────────────────────────────────
# REVIEWER
# ─────────────────────────────────────────────────────────────────────────
review_slide_chunks_prompt_old = """You are an expert reviewer evaluating slide chunks generated for a topic within an E-learning course.

Course Name: {course_name}
Target Audience: {target_audience}

Topic: {topic}

The research notes that were used to generate these slide chunks:
<research_notes>
{research_notes_context}
</research_notes>

Slide Chunks to Review:
<slide_chunks>
{slide_chunks}
</slide_chunks>

Evaluate the slide chunks strictly against these FOUR criteria:

1. **Slide Title Naming**: Are the slide titles short, concise, and logically consistent? Specifically check:
   - Titles should be clear labels, not full sentences or questions.
   - Titles should follow a logical naming theme based on the topic, subtopic, and learning objectives. For example, if a sequence of slides covers individual components, the titles should clearly name each component — consistency is good when it reflects the content structure.
   - Do NOT flag titles as repetitive just because they follow a similar pattern (e.g., "The High-Pressure Gauge", "The Low-Pressure Gauge"). Consistent patterns are correct when the content logically calls for it.
   - Only flag titles that are genuinely unclear, misleading, overly long, or that fail to describe their slide's content.
   - Content should not rely on the title to make sense — the slide must be understandable even if the title is hidden.
   - Titles should not assume they will be read aloud by the narrator.

2. **Missing Data from Research Notes**: Is there important information from the research notes that is missing from the slide chunks? Cross-reference the research notes against the slide content to find:
   - Key facts, measurements, specifications, or procedural steps that were dropped.
   - Definitions or explanations present in the research notes but absent from the slides.
   - Important examples or details that should have been included.
   - Note: Minor omissions of truly redundant or tangential details are acceptable.

3. **New Invented Data**: Do the slide chunks contain any information that is NOT present in the research notes? Check for:
   - Facts, statistics, or measurements not found in the research notes.
   - Examples, analogies, or metaphors that were invented rather than drawn from the source.
   - Claims or statements that go beyond what the research notes contain.
   - Note: Minor rephrasing for clarity is acceptable — this criterion targets substantive additions of new information.

4. **Transition Slide Redundancy**: Does the transition slide repeat or heavily overlap with the content of the slides that immediately follow it (especially the first content slide)? Check for:
   - The transition slide restating facts, details, or explanations that appear in the very next slide.
   - The transition hook containing content that should only appear in the body slides.
   - Note: The transition slide should introduce or tease the topic — not deliver the actual instructional content. A brief thematic connection is fine; duplicating specific details is not.

Important rules:
- Evaluate strictly against the criteria above, not general best practices.
- Be specific in your feedback — cite exact slide titles, passages, or describe exact gaps.
- If a criterion is fully satisfied, do not list it in failed_items.

Output your evaluation in this exact format:

<analysis>
Provide a thorough analysis of the slide chunks against all four criteria. Explain your reasoning for each criterion.
</analysis>

<failed_items>
For each criterion that failed, provide:

Item: [Criterion name — one of: Slide Title Naming, Missing Data from Research Notes, New Invented Data, Transition Slide Redundancy]
Feedback: [Specific, actionable feedback explaining what is wrong and what needs to change]

If ALL criteria pass, leave this section empty.
</failed_items>
"""

review_slide_chunks_prompt = """You are an expert reviewer evaluating slide chunks generated for a topic within an E-learning course.

Course Name: {course_name}
Target Audience: {target_audience}
Topic: {topic}

The research notes that were used to generate these slide chunks:
<research_notes>
{research_notes_context}
</research_notes>

Slide Chunks to Review:
<slide_chunks>
{slide_chunks}
</slide_chunks>

Evaluate the slide chunks against the criteria below. Work through them methodically — each addresses a distinct and common failure mode in slide chunk generation.

---

## STRUCTURAL CRITERIA

### 1. Transition Slide Scope
Transition slides should orient the learner, not teach the lesson. Check that:
- The transition introduces the subtopic's theme or relevance without delivering core teaching content.
- Key facts, diagnostic rules, procedural details, or specific conclusions that belong on content slides are NOT already stated in the transition.
- The transition does not name punchlines that a later content slide is supposed to teach (e.g., listing all three possible diagnoses before the diagnostic slide covers them).
- The transition is brief (1-2 sentences) and serves as a teaser or hook, not a mini-lecture.

Failure pattern: When a transition front-loads the lesson, the corresponding content slide becomes redundant because the learner already heard the main point.

### 2. Transition Slide Coverage
Every subtopic should have a transition slide, not just the first one. Check that:
- Each distinct subtopic present in the research notes has its own transition slide.
- No subtopic begins abruptly with a content slide when a shift in theme or skill area is occurring.

Failure pattern: Later subtopics often lack transitions, creating jarring jumps when the focus shifts (e.g., from gauge reading to safety practices, or from zeroing to field calibration).

### 3. Inter-Slide Redundancy
No two slides should deliver the same core information. Check that:
- No pair of slides (including transition + content pairs) repeats the same fact, definition, or explanation.
- If a concept appears briefly in one slide and is elaborated in another, the first mention is clearly a setup (not a full explanation) and the second adds genuinely new depth.
- When the same concept appears across two different learning objectives in the research notes, the corresponding slides differentiate what each adds rather than restating the same definition.

Failure pattern: Back-to-back slides paraphrase the same point, or a transition slide and its following content slide cover identical ground.

### 4. Connective Flow
Slides within a subtopic should form a coherent narrative thread. Check that:
- When a subtopic spans multiple content slides, each slide logically leads into the next.
- Thematic shifts within a single subtopic (e.g., from diagnostic interpretation to safety practices) are acknowledged rather than left as abrupt pivots with no bridging.

Failure pattern: Individually well-written slides read like disconnected paragraphs rather than a flowing lesson when taken in sequence.

---

## ALIGNMENT & ATTRIBUTION CRITERIA

### 5. Subtopic Attribution
Each slide must be labeled under the correct subtopic. Check that:
- The subtopic label on each slide matches the subtopic of the research note block its content is drawn from.
- If a slide bridges two subtopics, it is attributed to the one where the majority of its content originates.
- Subtopic names are spelled and capitalized consistently across all slides. If the source research notes contain a typo (e.g., "Guages"), the slides should use the corrected form consistently.

Failure pattern: A slide about calibration is labeled under the "Zeroing" subtopic, or the same subtopic name is spelled two different ways across slides.

---

## SUMMARY CRITERIA

### 6. Summary Slide Completeness and Fidelity
Summary slides should accurately review what was taught — nothing more, nothing less. Check that:
- The summary references all subtopics and key LOs covered in the preceding slides.
- No subtopic or significant concept from the content slides is omitted.
- The summary does NOT introduce new terminology, claims, or implications not present in the research notes (e.g., "professional system diagnostics" or "mechanical integrity" when the source doesn't use those phrases).
- The summary's scope matches the slides it follows — proportional coverage of all subtopics.

Failure pattern: The summary skips a subtopic entirely, or introduces an embellished claim like "accurately spot system issues" when the source only says the gauge "indicates possible problems."

---

## CONTENT FIDELITY CRITERIA

### 7. Missing Data from Research Notes
All important information from the research notes should appear in the slides. Cross-reference the research notes against the slide content to find:
- Key facts, measurements, specifications, or procedural steps that were dropped.
- Definitions or explanations present in the research notes but absent from the slides.
- Important examples or illustrative details that should have been included.

Note: Minor omissions of truly redundant or tangential details are acceptable and should NOT be flagged.

### 8. Invented Data in Slides
Slides should not contain substantive information absent from the research notes. Check for:
- Facts, statistics, or measurements not found in the research notes.
- Examples, analogies, or metaphors that were invented rather than drawn from the source.
- Claims or implications that go beyond what the research notes state (e.g., inferring a gauge "accurately spots issues" when the source only says it "indicates possible problems").

Note: Minor rephrasing for conversational tone or clarity is acceptable — this criterion targets substantive additions of new information only.

---

## PRESENTATION CRITERIA

### 9. Slide Title Quality
Titles should be slide titles short, concise, and logically consistent. Check that:
- Titles are concise (typically 2-5 words).
- No title is a full sentence or question.
- The slide content is understandable without reading the title (the title is a label, not a dependency).
- Titles should follow a logical naming theme based on the topic, subtopic, and learning objectives. For example, if a sequence of slides covers individual components, the titles should clearly name each component — consistency is good when it reflects the content structure.
- Do NOT flag titles as repetitive just because they follow a similar pattern (e.g., "The High-Pressure Gauge", "The Low-Pressure Gauge"). Consistent patterns are correct when the content logically calls for it.
- Only flag titles that are genuinely unclear, misleading, overly long, or that fail to describe their slide's content.
- Content should not rely on the title to make sense — the slide must be understandable even if the title is hidden.
- Titles should not assume they will be read aloud by the narrator.

### 10. Tone and Language:
- Sound like a real technician, not an AI. Use plain language and active voice.
- Use action-oriented verbs: "check," "spot," "find" over "identify," "assess," "determine."
- No buzzwords: "optimal," "leverage," "utilize," "facilitate."
- Explain trade terms when first used.
- Present tense, as if the learner is doing the task right now.
- If the research notes & slide chunks already use plain language, keep it as is. Do not rewrite for tone if the original is already clear and direct. Your job is to restructure and present — not to rewrite for tone if it's not needed.
- See below paraphrasing rules:

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


---

## IMPORTANT RULES

- Evaluate strictly against the criteria above, not general best practices or personal preferences.
- Be specific in your feedback — cite exact slide titles, quote passages, or describe exact gaps.
- For each criterion, make a clear pass/fail determination. If a criterion is fully satisfied, do NOT list it in failed_items.
- A criterion should fail if ANY of its checks are violated, even if other checks within it pass.
- Inline image links: Markdown image links in the slide content (in either `[alt](url)` or `![alt](url)` form) are intentional and handled by a dedicated upstream pipeline step. Do NOT flag their presence, format, placement, or alt text as a defect under any criterion. Do NOT include failed_items asking the reviser to remove, rewrite, reformat, or reposition image links. Treat them as invisible to your evaluation.

---

Output your evaluation in this exact format:

<analysis>
Work through each criterion in order. For each, state whether it passes or fails and explain your reasoning. Cite specific slide titles, content, or research note blocks to support your assessment.
</analysis>

<failed_items>
For each criterion that failed, provide:

Item: [Criterion number and name, e.g., "1. Transition Slide Scope"]
Feedback: [Specific, actionable feedback explaining what is wrong and what needs to change. Reference exact slide titles and content.]

If ALL criteria pass, leave this section empty.
</failed_items>
"""

review_slide_chunks_followup_prompt = """The reviser has made changes to the slide chunks. Here are the diffs from the last revision:

<text_diffs>
{text_diffs}
</text_diffs>

Here are the updated slide chunks after revision:
<slide_chunks>
{slide_chunks}
</slide_chunks>

Re-evaluate the slide chunks against all the criteria. Check whether previous issues have been addressed and identify any remaining or new issues.

<analysis>
[Your analysis here]
</analysis>

<failed_items>
[Failed items, or empty if all pass]
</failed_items>
"""


@traceable(metadata={
    "agent_name": "slide_chunks",
    "step_name": "Slide Chunks Reviewer",
    "function_name": "review_slide_chunks",
    "user_id": st.session_state.get("role", "anonymous"),
    "user_email": st.session_state.get("user_email", "anonymous")
})
def review_slide_chunks(
    course_name, target_audience, topic, research_notes_context,
    slide_chunks, llm="gemini_3_flash",
    message_history=None, text_diffs=None
):
    """
    Review slide chunks for quality issues.

    :param course_name: Course name.
    :param target_audience: Target audience.
    :param topic: Topic name.
    :param research_notes_context: Formatted research notes for cross-reference.
    :param slide_chunks: The slide chunks text to review.
    :param llm: LLM to use for the reviewer.
    :param message_history: Previous message history for continuity.
    :param text_diffs: Diffs from the last revision (for subsequent iterations).
    :return: (analysis, failed_items, updated_message_history)
    """
    reviewer = Chain(llm=llm, tags=["analysis", "failed_items"])

    if message_history:
        reviewer.add_messages(message_history)
        reviewer.add_message(
            role="user",
            content=review_slide_chunks_followup_prompt.format(
                text_diffs=text_diffs,
                slide_chunks=slide_chunks,
            )
        )
    else:
        reviewer.add_message(
            role="user",
            content=review_slide_chunks_prompt.format(
                course_name=course_name,
                target_audience=target_audience,
                topic=topic,
                research_notes_context=research_notes_context,
                slide_chunks=slide_chunks,
            )
        )

    response = reviewer.run()
    updated_history = reviewer.messages_list

    return response["analysis"], response["failed_items"], updated_history


# ─────────────────────────────────────────────────────────────────────────
# REVISER
# ─────────────────────────────────────────────────────────────────────────
sc_reviser_prompt = """You are an expert educational content reviser. Your task is to revise slide chunks based on review feedback.

Course Name: {course_name}
Target Audience: {target_audience}

Topic: {topic}

The research notes that the slide chunks should be based on:
<research_notes>
{research_notes_context}
</research_notes>

Current Slide Chunks:
<slide_chunks>
{slide_chunks}
</slide_chunks>

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
- For SLIDE TITLE NAMING issues: Use `str_replace` to fix individual titles. Titles should be clear, concise labels that logically reflect the content. Consistent naming patterns are fine when the content calls for it.
- For MISSING DATA FROM RESEARCH NOTES: Use `str_replace` to add the missing information into the appropriate slides. Draw the content directly from the research notes provided above.
- For NEW INVENTED DATA: Use `str_replace` to remove or replace invented content with information actually present in the research notes.
- For TRANSITION SLIDE REDUNDANCY: Use `str_replace` to rework the transition slide so it introduces or teases the topic without duplicating specific details from the following content slides.
- Use `str_replace` for targeted, precise edits. Use `full_overwrite` only when major restructuring is truly required.
- Inline image links: Any markdown image links in the slide content — in either `[alt](url)` or `![alt](url)` form — must be preserved. Do NOT delete, rewrite, drop, merge, reposition, or modify the URL or alt text of any existing image link during your revisions, even if your other edits touch the surrounding sentence. Keep each link in the exact position (same surrounding sentence) and exact format (same `[` vs `!` prefix) it currently has. When using `str_replace`, if the `old_text` you are replacing contains an image link, the `new_text` must contain the same image link(s) verbatim in the same relative position. If you use `full_overwrite`, every image link from the current slide chunks must appear in the new text verbatim. Image-link placement is handled by a dedicated upstream pipeline step — treat every `[...](...)` / `![...](...)` as load-bearing content.
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
    "agent_name": "slide_chunks",
    "step_name": "Slide Chunks Reviser",
    "function_name": "run_slide_chunks_reviser_agent",
    "user_id": st.session_state.get("role", "anonymous"),
    "user_email": st.session_state.get("user_email", "anonymous")
})
def run_slide_chunks_reviser_agent(
    slide_chunks, review_analysis, failed_items,
    course_name, target_audience, topic, research_notes_context,
    llm="gemini_3_flash",
    message_history=None, text_diffs=None
):
    """
    Run the reviser agent to fix issues identified by the reviewer.

    :param slide_chunks: Current slide chunks text.
    :param review_analysis: Analysis from the reviewer.
    :param failed_items: Failed criteria feedback from the reviewer.
    :param course_name: Course name.
    :param target_audience: Target audience.
    :param topic: Topic name.
    :param research_notes_context: Formatted research notes for reference.
    :param llm: LLM identifier (unused — hardcoded to Gemini for agent).
    :param message_history: Previous messages for continuity.
    :param text_diffs: Diffs from previous revision for context.
    :return: (revised_slide_chunks_string, updated_message_history)
    """
    from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

    print(f"\n🔄 STARTING SLIDE CHUNKS REVISER AGENT for topic '{topic}'")
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

    tools = [sc_str_replace, sc_full_overwrite, sc_stop]

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
        state_schema=SlideChunksState,
        middleware=[handle_tool_errors],
    )

    if message_history:
        follow_up_content = f"""The reviewer has re-evaluated the slide chunks after your previous revisions. Here's what you changed:

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
        messages = [HumanMessage(content=sc_reviser_prompt.format(
            slide_chunks=slide_chunks,
            review_analysis=review_analysis,
            failed_items=failed_items,
            course_name=course_name,
            target_audience=target_audience,
            topic=topic,
            research_notes_context=research_notes_context,
        ))]

    state = {
        "messages": messages,
        "slide_chunks": {"text": slide_chunks},
    }

    final_state = graph.invoke(
        state,
        {"recursion_limit": 50}
    )

    print("=" * 80)
    print("✅ SLIDE CHUNKS REVISER AGENT COMPLETED")
    print(f"📊 Final slide chunks length: {len(final_state['slide_chunks']['text'])} chars")
    print("=" * 80)

    return final_state["slide_chunks"]["text"], final_state["messages"]


# ─────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────
@traceable(metadata={
    "agent_name": "slide_chunks",
    "step_name": "Slide Chunks Review-Revise",
    "function_name": "review_revise_slide_chunks",
    "user_id": st.session_state.get("role", "anonymous"),
    "user_email": st.session_state.get("user_email", "anonymous")
})
def review_revise_slide_chunks(
    course_name, target_audience, topic, slide_chunks_text,
    subtopics_data, llm="gemini_3_flash", max_iterations=3,
    reviewer_llm="gemini_3_flash"
):
    """
    Run a review-revise quality loop on generated slide chunks.

    :param course_name: Course name.
    :param target_audience: Target audience.
    :param topic: Topic name.
    :param slide_chunks_text: The initial slide chunks text to review and revise.
    :param subtopics_data: List of dicts with keys: subtopic, learning_objectives, research_notes.
    :param llm: LLM for the reviser agent.
    :param max_iterations: Max review-revise cycles (default 3).
    :param reviewer_llm: LLM for the reviewer.
    :return: Final revised slide chunks text.
    """
    # Build the research notes context for the reviewer
    research_notes_context = _build_research_notes_context(subtopics_data)

    if not research_notes_context.strip():
        print(f"⚠️ No research notes available for topic '{topic}'. Skipping review-revise loop.")
        return slide_chunks_text

    # Review-Revise loop
    reviewer_history = None
    reviser_history = None
    last_text_diffs = None

    for iteration in range(1, max_iterations + 1):
        print(f"\n🔄 Slide Chunks Review-Revise iteration {iteration}/{max_iterations} for topic '{topic}'")

        # Run reviewer
        analysis, failed_items, reviewer_history = review_slide_chunks(
            course_name=course_name,
            target_audience=target_audience,
            topic=topic,
            research_notes_context=research_notes_context,
            slide_chunks=slide_chunks_text,
            llm=reviewer_llm,
            message_history=reviewer_history,
            text_diffs=last_text_diffs,
        )

        print(f"📋 Review result (iteration {iteration}): failed_items length = {len(failed_items)} chars")

        # Check if all criteria pass
        if is_failed_items_empty(failed_items):
            print(f"✅ All criteria passed at iteration {iteration} for topic '{topic}'")
            break

        # Save text before revision for diff computation
        text_before = slide_chunks_text

        # Run reviser agent
        slide_chunks_text, reviser_history = run_slide_chunks_reviser_agent(
            slide_chunks=slide_chunks_text,
            review_analysis=analysis,
            failed_items=failed_items,
            course_name=course_name,
            target_audience=target_audience,
            topic=topic,
            research_notes_context=research_notes_context,
            message_history=reviser_history,
            text_diffs=last_text_diffs,
        )

        # Compute diff for next iteration
        last_text_diffs = compute_text_diff(text_before, slide_chunks_text)
        print(f"📊 Revision {iteration} complete. Diff length: {len(last_text_diffs)} chars")

    return slide_chunks_text
