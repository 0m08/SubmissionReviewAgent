# Paraphraser Agent Implementation Plan

**Version:** 1.1
**Date:** 2025-11-19
**Architecture:** Multi-Agent Orchestrator with LangGraph

---

## Key Implementation Decisions

1. **LLM Model:** GPT-4o-mini (default, user-configurable in UI)
2. **Quality Threshold:** Binary pass/fail - any failure triggers refinement
3. **Quick Mode:** Not implemented in MVP
4. **Test Content:** Will be created as needed during implementation

---

## 1. Architecture Overview

### 1.1 Multi-Agent Design Philosophy
The paraphraser uses a **multi-agent orchestrator pattern** where:
- An **orchestrator** coordinates execution
- Each specialized agent is a **tool** the orchestrator can call
- Tools are composable and can be added/removed without changing core logic
- Maximum flexibility for future enhancements

### 1.2 Agent Roles

#### **Agent 1: Paraphraser Agent**
**Purpose:** Transform input text according to spec requirements
**Input:** Original text + trade context
**Output:** Paraphrased text
**Key Behavior:**
- Apply persona (senior technician)
- Use trade-specific terminology
- Maintain conversational tone
- Preserve all factual information

#### **Agent 2: Quality Reviewer Agent**
**Purpose:** Validate output against quality criteria
**Input:** Original text + paraphrased text
**Output:** Quality assessment (pass/fail + issues found)
**Validation Checks:**
- Accuracy: No information added/removed
- Clarity: Improved readability
- Tone: Sounds like real technician
- Technical correctness: Trade terminology appropriate
- Safety: All safety info preserved

#### **Agent 3: Refiner Agent**
**Purpose:** Fix specific issues identified by reviewer
**Input:** Paraphrased text + quality issues
**Output:** Refined text
**Key Behavior:**
- Addresses specific failures from reviewer
- Does NOT re-paraphrase from scratch
- Makes surgical edits only
- Maintains improvements from paraphraser

#### **Orchestrator Agent (LangGraph ReAct)**
**Purpose:** Coordinate the multi-agent workflow
**Tools Available:** paraphrase_text, review_quality, refine_text
**Logic:**
1. Receives user request
2. Calls paraphraser tool
3. Calls reviewer tool
4. If issues found, calls refiner tool
5. Re-reviews if needed
6. Returns final output

---

## 2. LangGraph Architecture

### 2.1 State Schema

```python
from typing import TypedDict, Annotated, List, Dict, Optional
from langgraph.graph import MessagesState

class ParaphraserState(MessagesState):
    """State that flows through the orchestrator graph"""

    # Input parameters
    original_text: str                    # The text to paraphrase
    trade: str                            # e.g., "HVAC"
    specialization: Optional[str]         # e.g., "Residential HVAC"
    preserve_formatting: bool             # Whether to keep structure
    target_length: Optional[str]          # "similar", "concise", "expanded"

    # Working data
    paraphrased_text: Optional[str]       # Current paraphrased version
    quality_report: Optional[Dict]        # Reviewer's assessment
    refinement_count: int                 # Number of refinement iterations

    # Output
    final_text: Optional[str]             # Final approved output
    status: str                           # "in_progress", "completed", "failed"

    # Metadata
    iteration_history: List[Dict]         # Track all iterations for debugging
```

### 2.2 Graph Structure

```
Entry Point (orchestrator)
    ↓
orchestrator_node
    ↓
    └→ calls tools based on LLM decision:
        ├→ paraphrase_text (tool)
        ├→ review_quality (tool)
        └→ refine_text (tool)
    ↓
    └→ loops until quality passes or max_iterations reached
    ↓
END (returns final state)
```

### 2.3 Why ReAct Pattern?
- LLM decides which tool to call and when
- Natural language reasoning between tool calls
- Easy to add new tools (future agents) without changing graph
- Built-in error handling and retry logic

---

## 3. Agent Implementation Details

### 3.1 Paraphraser Agent Tool

**File:** `/agents/paraphraser/paraphrase_agent.py`

```python
from langchain_core.tools import tool
from typing import Optional

@tool("paraphrase_text", parse_docstring=True)
def paraphrase_text(
    text: str,
    trade: str,
    specialization: Optional[str] = None,
    preserve_formatting: bool = True,
    target_length: Optional[str] = "similar"
) -> str:
    """
    Paraphrase text to sound like an experienced tradesperson.

    Args:
        text: The text to paraphrase
        trade: The trade (e.g., "HVAC", "Electrical")
        specialization: Optional sub-specialization
        preserve_formatting: Whether to maintain structure
        target_length: "similar", "concise", or "expanded"

    Returns:
        Paraphrased text
    """
    # Implementation will use Chain class with paraphraser prompt
    from modules.chain import Chain

    # Build prompt from template
    prompt = PARAPHRASER_PROMPT_TEMPLATE.format(
        trade=trade,
        specialization=specialization or "General",
        input_text=text,
        preserve_formatting="yes" if preserve_formatting else "no",
        target_length=target_length or "similar"
    )

    # Execute with Chain
    chain = Chain(llm='openai:gpt-4o-mini', tags=["paraphrased_text"])
    chain.add_message(role="user", content=prompt)
    response = chain.run()

    return response["paraphrased_text"]
```

**Prompt Template:**
```python
PARAPHRASER_PROMPT_TEMPLATE = """
You are a friendly senior {trade} technician with 10-15 years of experience. Your job is to paraphrase technical content so it sounds clear, conversational, and practical—like you're explaining it to another technician.

<trade_context>
Trade: {trade}
Specialization: {specialization}
</trade_context>

<paraphrasing_rules>
1. Use plain language, not corporate jargon
2. Sound like a real technician, not an AI
3. Keep all factual information accurate (never add or remove facts)
4. Use active voice and short sentences
5. Explain trade terminology when first used
6. Maintain safety information completely
7. Be conversational but professional
</paraphrasing_rules>

<input_text>
{input_text}
</input_text>

<formatting>
Preserve formatting: {preserve_formatting}
Target length: {target_length}
</formatting>

Paraphrase the text above. Return ONLY the paraphrased text inside <paraphrased_text> tags.
"""
```

---

### 3.2 Quality Reviewer Agent Tool

**File:** `/agents/paraphraser/review_agent.py`

```python
@tool("review_quality", parse_docstring=True)
def review_quality(
    original_text: str,
    paraphrased_text: str,
    trade: str
) -> Dict[str, any]:
    """
    Review paraphrased text for quality against spec criteria.

    Args:
        original_text: The original input text
        paraphrased_text: The paraphrased version to review
        trade: The trade context

    Returns:
        Dictionary with:
        - passed: bool
        - issues: List[str] (specific problems found)
        - score: int (0-100)
        - feedback: str (detailed feedback)
    """
    from modules.chain import Chain

    prompt = REVIEWER_PROMPT_TEMPLATE.format(
        original_text=original_text,
        paraphrased_text=paraphrased_text,
        trade=trade
    )

    chain = Chain(
        llm='openai:gpt-4o-mini',
        tags=["passed", "issues", "score", "feedback"]
    )
    chain.add_message(role="user", content=prompt)
    response = chain.run()

    return {
        "passed": response["passed"].lower() == "true",
        "issues": response["issues"],
        "score": int(response["score"]),
        "feedback": response["feedback"]
    }
```

**Prompt Template:**
```python
REVIEWER_PROMPT_TEMPLATE = """
You are a quality reviewer for trade training content. Review the paraphrased text against these criteria:

<quality_criteria>
1. ACCURACY: All facts from original are preserved (no additions/deletions)
2. CLARITY: Improved readability, plain language used
3. TONE: Sounds like real {trade} technician, conversational but professional
4. TECHNICAL: Trade terminology appropriate and correctly used
5. SAFETY: All safety information fully preserved
6. AUTHENTICITY: No corporate buzzwords or AI-sounding language
</quality_criteria>

<original_text>
{original_text}
</original_text>

<paraphrased_text>
{paraphrased_text}
</paraphrased_text>

Review the paraphrased text. Return your assessment in these XML tags:
- <passed>true or false</passed>
- <issues>Comma-separated list of specific issues, or "none"</issues>
- <score>Integer 0-100</score>
- <feedback>Detailed explanation of score and issues</feedback>
"""
```

---

### 3.3 Refiner Agent Tool

**File:** `/agents/paraphraser/refine_agent.py`

```python
@tool("refine_text", parse_docstring=True)
def refine_text(
    paraphrased_text: str,
    issues: List[str],
    original_text: str,
    trade: str
) -> str:
    """
    Refine paraphrased text to fix specific quality issues.

    Args:
        paraphrased_text: Current paraphrased version
        issues: List of specific issues to fix
        original_text: Original text for reference
        trade: Trade context

    Returns:
        Refined text with issues addressed
    """
    from modules.chain import Chain

    issues_str = "\n".join(f"- {issue}" for issue in issues)

    prompt = REFINER_PROMPT_TEMPLATE.format(
        paraphrased_text=paraphrased_text,
        issues=issues_str,
        original_text=original_text,
        trade=trade
    )

    chain = Chain(llm='openai:gpt-4o-mini', tags=["refined_text"])
    chain.add_message(role="user", content=prompt)
    response = chain.run()

    return response["refined_text"]
```

**Prompt Template:**
```python
REFINER_PROMPT_TEMPLATE = """
You are refining paraphrased {trade} technical content. Fix ONLY the specific issues listed below. Do NOT re-paraphrase from scratch.

<issues_to_fix>
{issues}
</issues_to_fix>

<current_paraphrased_text>
{paraphrased_text}
</current_paraphrased_text>

<original_text_reference>
{original_text}
</original_text_reference>

Make surgical edits to fix the issues while preserving the improvements already made. Return ONLY the refined text inside <refined_text> tags.
"""
```

---

## 4. Orchestrator Implementation

**File:** `/agents/paraphraser/orchestrator.py`

```python
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from typing import Dict, Optional
import os

def create_paraphraser_orchestrator(
    max_iterations: int = 3,
    llm_model: str = "openai:gpt-4o-mini"
):
    """
    Create the multi-agent paraphraser orchestrator.

    Args:
        max_iterations: Maximum refinement iterations
        llm_model: LLM to use for orchestrator (default: gpt-4o-mini)

    Returns:
        Compiled LangGraph agent
    """
    from langchain_chat_models import init_chat_model
    from .paraphrase_agent import paraphrase_text
    from .review_agent import review_quality
    from .refine_agent import refine_text

    # Initialize LLM
    llm = init_chat_model(llm_model)

    # Define tools
    tools = [paraphrase_text, review_quality, refine_text]

    # Create ReAct agent with tools
    graph = create_agent(
        model=llm,
        tools=tools,
        state_schema=ParaphraserState,
    )

    return graph


def run_paraphraser(
    text: str,
    trade: str = "HVAC",
    specialization: Optional[str] = None,
    preserve_formatting: bool = True,
    target_length: Optional[str] = "similar",
    max_iterations: int = 3,
    llm_model: str = "openai:gpt-4o-mini"
) -> Dict:
    """
    Main entry point for paraphrasing text.

    Args:
        text: Text to paraphrase
        trade: Trade context
        specialization: Optional sub-specialization
        preserve_formatting: Whether to preserve structure
        target_length: "similar", "concise", or "expanded"
        max_iterations: Max refinement loops
        llm_model: LLM model to use

    Returns:
        Dictionary with final_text, status, and metadata
    """
    # Create orchestrator
    orchestrator = create_paraphraser_orchestrator(
        max_iterations=max_iterations,
        llm_model=llm_model
    )

    # Build orchestrator instructions
    instructions = f"""
You are orchestrating a text paraphrasing workflow. Follow these steps:

1. Call 'paraphrase_text' with the provided parameters
2. Call 'review_quality' to check the paraphrased text
3. If review passes, you're done
4. If review fails, call 'refine_text' with the issues
5. Review again (max {max_iterations} refinements)
6. Return the final approved text

Parameters:
- text: {text[:200]}...
- trade: {trade}
- specialization: {specialization}
- preserve_formatting: {preserve_formatting}
- target_length: {target_length}

Start by paraphrasing the text.
"""

    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content=instructions)],
        "original_text": text,
        "trade": trade,
        "specialization": specialization,
        "preserve_formatting": preserve_formatting,
        "target_length": target_length,
        "refinement_count": 0,
        "iteration_history": [],
    }

    # Run orchestrator
    final_state = orchestrator.invoke(
        initial_state,
        {"recursion_limit": 50}
    )

    # Extract final text from messages
    # The last tool call result should contain the approved text
    messages = final_state["messages"]
    final_text = extract_final_text_from_messages(messages)

    return {
        "final_text": final_text,
        "status": "completed",
        "messages": messages,
        "metadata": {
            "refinement_count": final_state.get("refinement_count", 0),
            "trade": trade,
            "specialization": specialization,
        }
    }


def extract_final_text_from_messages(messages):
    """Extract the final paraphrased text from message history"""
    # Look for the last tool result that contains text
    for msg in reversed(messages):
        if hasattr(msg, 'content') and isinstance(msg.content, str):
            # Parse out refined_text or paraphrased_text
            if '<refined_text>' in msg.content:
                return msg.content.split('<refined_text>')[1].split('</refined_text>')[0].strip()
            elif '<paraphrased_text>' in msg.content:
                return msg.content.split('<paraphrased_text>')[1].split('</paraphrased_text>')[0].strip()
    return "Error: Could not extract final text"
```

---

## 5. File Structure

```
agents/paraphraser/
├── __init__.py                    # Package initialization, exports main function
├── PARAPHRASER_AGENT_SPEC.md     # Specification (already exists)
├── IMPLEMENTATION_PLAN.md        # This document
├── orchestrator.py               # Main orchestrator + run_paraphraser()
├── paraphrase_agent.py           # Paraphraser tool
├── review_agent.py               # Quality reviewer tool
├── refine_agent.py               # Refiner tool
├── prompts.py                    # All prompt templates
└── state.py                      # State schema definition

Root directory:
paraphraser.py                    # Streamlit UI page
```

### 5.1 Package Init

**File:** `/agents/paraphraser/__init__.py`

```python
"""
Paraphraser Agent - Multi-Agent Orchestrator

Transforms monotonous technical text into clear, conversational language
that sounds like an experienced tradesperson.
"""

from .orchestrator import run_paraphraser, create_paraphraser_orchestrator
from .state import ParaphraserState

__all__ = [
    'run_paraphraser',
    'create_paraphraser_orchestrator',
    'ParaphraserState',
]
```

---

## 6. Streamlit UI Design

**File:** `/paraphraser.py`

### 6.1 UI Layout

```
┌─────────────────────────────────────────┐
│  Paraphraser Agent                      │
│  Transform technical text into clear,   │
│  conversational language               │
├─────────────────────────────────────────┤
│                                         │
│  [Trade Selection Dropdown]             │
│  ○ HVAC (default)                      │
│  ○ Electrical                          │
│  ○ Plumbing                            │
│                                         │
│  [Specialization] (optional)            │
│  e.g., "Residential HVAC"              │
│                                         │
│  [Text Input - Large Text Area]         │
│  Enter or paste text to paraphrase...  │
│                                         │
│  ☑ Preserve formatting                 │
│  [Target Length] ▼ Similar             │
│                                         │
│  [Paraphrase Text] Button              │
│                                         │
├─────────────────────────────────────────┤
│  Results (shown after processing):      │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │ Original Text                     │ │
│  │ [Shows input]                     │ │
│  └───────────────────────────────────┘ │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │ Paraphrased Text                  │ │
│  │ [Shows output]                    │ │
│  │ [Copy to Clipboard] Button        │ │
│  └───────────────────────────────────┘ │
│                                         │
│  Quality Score: 95/100 ⭐              │
│  Refinement Iterations: 1              │
│                                         │
│  📊 [Show Process Details] (expander)  │
│                                         │
└─────────────────────────────────────────┘
```

### 6.2 UI Implementation

```python
import streamlit as st
from agents.paraphraser import run_paraphraser

st.set_page_config(
    page_title="Paraphraser Agent",
    page_icon="✍️",
    layout="wide"
)

st.title("✍️ Paraphraser Agent")
st.markdown("""
Transform monotonous technical text into clear, conversational language that sounds
like an experienced tradesperson.
""")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")

    trade = st.selectbox(
        "Trade",
        ["HVAC", "Electrical", "Plumbing"],
        index=0
    )

    specialization = st.text_input(
        "Specialization (optional)",
        placeholder="e.g., Residential HVAC",
        help="Narrow down the trade context"
    )

    preserve_formatting = st.checkbox(
        "Preserve formatting",
        value=True,
        help="Maintain bullet points, lists, and paragraph structure"
    )

    target_length = st.selectbox(
        "Target Length",
        ["similar", "concise", "expanded"],
        index=0,
        help="How long should the output be relative to input?"
    )

    max_iterations = st.slider(
        "Max Refinement Iterations",
        min_value=1,
        max_value=5,
        value=3,
        help="Maximum times to refine if quality checks fail"
    )

    llm_model = st.selectbox(
        "LLM Model",
        ["openai:gpt-4o-mini", "openai:gpt-4o", "anthropic:claude-3-5-sonnet-20241022"],
        index=0,
        help="GPT-4o-mini is faster and cheaper, GPT-4o for highest quality"
    )

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Text")
    input_text = st.text_area(
        "Text to paraphrase",
        height=400,
        placeholder="Paste or type technical content here...",
        label_visibility="collapsed"
    )

with col2:
    st.subheader("Paraphrased Text")
    output_placeholder = st.empty()

# Paraphrase button
if st.button("✨ Paraphrase Text", type="primary", use_container_width=True):
    if not input_text.strip():
        st.error("Please enter some text to paraphrase")
    else:
        with st.spinner("Paraphrasing... This may take 10-30 seconds"):
            try:
                result = run_paraphraser(
                    text=input_text,
                    trade=trade,
                    specialization=specialization if specialization else None,
                    preserve_formatting=preserve_formatting,
                    target_length=target_length,
                    max_iterations=max_iterations,
                    llm_model=llm_model
                )

                # Store in session state
                st.session_state.paraphraser_result = result
                st.session_state.paraphraser_input = input_text

                st.success("✅ Paraphrasing complete!")
                st.rerun()

            except Exception as e:
                st.error(f"Error during paraphrasing: {str(e)}")
                st.exception(e)

# Display results if available
if "paraphraser_result" in st.session_state:
    result = st.session_state.paraphraser_result

    with col2:
        st.text_area(
            "Paraphrased output",
            value=result["final_text"],
            height=400,
            label_visibility="collapsed"
        )

        if st.button("📋 Copy to Clipboard"):
            st.write(result["final_text"])  # Streamlit auto-copies
            st.success("Copied!")

    # Metadata display
    st.divider()

    meta_col1, meta_col2, meta_col3 = st.columns(3)

    with meta_col1:
        st.metric("Trade", result["metadata"]["trade"])

    with meta_col2:
        st.metric("Refinement Iterations", result["metadata"]["refinement_count"])

    with meta_col3:
        st.metric("Status", result["status"].upper())

    # Process details expander
    with st.expander("📊 Show Process Details"):
        st.json(result["metadata"])

        st.subheader("Message History")
        for i, msg in enumerate(result.get("messages", [])):
            with st.container():
                st.write(f"**Message {i+1}:** {type(msg).__name__}")
                st.code(str(msg.content)[:500] + "..." if len(str(msg.content)) > 500 else str(msg.content))

# Clear results button
if "paraphraser_result" in st.session_state:
    if st.button("🔄 Clear Results"):
        del st.session_state.paraphraser_result
        del st.session_state.paraphraser_input
        st.rerun()
```

---

## 7. Integration with streamlit_app.py

**File:** `/streamlit_app.py`

Add these lines:

```python
# Around line 285 (with other page definitions)
paraphraser_page = st.Page(
    "paraphraser.py",
    title="Paraphraser",
    icon=":material/edit:",
)

# Around line 383 (page_name_to_object dictionary)
page_name_to_object = {
    # ... existing pages ...
    "paraphraser_page": paraphraser_page,
}

# Around line 415 (tool_pages list - or create new category)
tool_pages = [
    # ... existing tools ...
    "paraphraser_page",
]
```

**Alternative:** Create a new category for "Text Tools" or "Writing Assistants"

---

## 8. Implementation Steps

### Phase 1: Core Agent Implementation (Day 1)
1. ✅ Create file structure in `/agents/paraphraser/`
2. ✅ Implement `state.py` with ParaphraserState schema
3. ✅ Implement `prompts.py` with all prompt templates
4. ✅ Implement `paraphrase_agent.py` tool
5. ✅ Implement `review_agent.py` tool
6. ✅ Implement `refine_agent.py` tool
7. ✅ Test each tool independently

### Phase 2: Orchestrator (Day 1-2)
8. ✅ Implement `orchestrator.py`
9. ✅ Test orchestrator with simple inputs
10. ✅ Debug and refine tool calling logic
11. ✅ Add error handling and logging

### Phase 3: Streamlit UI (Day 2)
12. ✅ Create `/paraphraser.py` UI page
13. ✅ Implement basic input/output interface
14. ✅ Add metadata display and process details
15. ✅ Test UI workflow end-to-end

### Phase 4: Integration (Day 2)
16. ✅ Add page to `streamlit_app.py`
17. ✅ Configure role-based access (if needed)
18. ✅ Test navigation from main app

### Phase 5: Testing & Refinement (Day 3)
19. ✅ Test with various HVAC content
20. ✅ Verify quality criteria enforcement
21. ✅ Test edge cases (very long text, already good text, poor text)
22. ✅ Performance optimization
23. ✅ Add LangSmith tracing

---

## 9. Testing Strategy

### 9.1 Unit Tests
- Test each agent tool independently
- Verify prompt formatting
- Check output parsing
- Validate state transitions

### 9.2 Integration Tests
- Full orchestrator workflow
- Multiple refinement iterations
- Different trade contexts
- Edge cases (empty input, very long text)

### 9.3 Quality Tests
Test cases from spec examples:
1. Formal to conversational
2. Vague to specific
3. Unexplained slang to defined terminology
4. Complex technical to clear
5. Preserving technical values
6. Multi-step instructions

### 9.4 Performance Tests
- Measure latency (target: <30 seconds for 500 words)
- Token usage tracking
- Cost per paraphrase
- Concurrency handling

---

## 10. Future Enhancements (Post-MVP)

### 10.1 Additional Agent Tools
- **Style Checker Tool**: Verify consistent style across content
- **Terminology Validator Tool**: Check against approved trade glossary
- **Safety Emphasis Tool**: Highlight safety warnings more prominently
- **Readability Scorer Tool**: Quantitative readability metrics

### 10.2 UI Improvements
- Batch processing (upload CSV/Excel)
- Before/after comparison view with diff highlighting
- Save/load paraphrasing sessions
- Export to various formats (PDF, DOCX)
- History of past paraphrases

### 10.3 Integration Features
- Make callable from other agents via simple function import
- API endpoint for external access
- Webhook support for automated workflows
- Integration with course content generation pipeline

### 10.4 Advanced Features
- Multi-trade mode (auto-detect trade from content)
- Custom style presets (more/less formal, different experience levels)
- A/B testing with multiple LLM outputs
- Human feedback loop for continuous improvement

---

## 11. Dependencies & Requirements

### 11.1 Python Packages (add to requirements.txt)
```
langgraph>=0.5.4
langchain-core>=0.3.0
langchain-openai>=0.2.0
streamlit>=1.30.0
```

### 11.2 Environment Variables
```
OPENAI_API_KEY=...
LANGCHAIN_API_KEY=...  # For LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=paraphraser-agent
```

### 11.3 LLM Model Access
- OpenAI API (GPT-4, GPT-4o)
- Anthropic API (Claude) - optional
- Gemini API - optional

---

## 12. Cost Estimates

### 12.1 Per Paraphrase (500 words)
Using GPT-4o-mini (default):
- Paraphraser: ~1000 input + 600 output tokens = $0.00051
- Reviewer: ~1200 input + 300 output tokens = $0.00036
- Refiner (if needed): ~1200 input + 600 output tokens = $0.00054
- Orchestrator overhead: ~500 tokens = $0.0001

**Total per paraphrase:** ~$0.001 (without refinement) to $0.002 (with 1 refinement)

Using GPT-4o (optional upgrade):
- Total per paraphrase: ~$0.06 (without refinement) to $0.10 (with 1 refinement)

**GPT-4o-mini is approximately 50x cheaper than GPT-4o**

### 12.2 Cost Optimization
- GPT-4o-mini is default (already optimized)
- Cache prompts where possible
- Limit max iterations to 3
- Users can upgrade to GPT-4o for highest quality when needed

---

## 13. Success Criteria

### 13.1 Technical Success
- ✅ All three agent tools working independently
- ✅ Orchestrator successfully coordinates workflow
- ✅ Quality review catches >90% of spec violations
- ✅ UI loads and processes text without errors
- ✅ Average processing time <30 seconds for 500 words

### 13.2 Quality Success
- ✅ Output passes all quality criteria from spec
- ✅ No information hallucination (verified by reviewer)
- ✅ Improved readability (Flesch score increase)
- ✅ Authentic technician voice (SME validation)
- ✅ Safety information 100% preserved

### 13.3 User Experience Success
- ✅ Intuitive UI (no training needed)
- ✅ Clear feedback during processing
- ✅ Helpful error messages
- ✅ Easy to adjust settings
- ✅ Results easy to copy/export

---

## 14. Risks & Mitigations

### 14.1 Risk: Infinite Refinement Loop
**Mitigation:** Hard limit on iterations (max 3), orchestrator instructions include termination logic

### 14.2 Risk: Quality Reviewer Too Strict
**Mitigation:** Tune reviewer prompt, allow score threshold (e.g., >85 instead of 100)

### 14.3 Risk: High Token Cost
**Mitigation:** Use cheaper models for review, implement caching, add cost warnings in UI

### 14.4 Risk: Slow Processing
**Mitigation:** Parallel tool calls where possible, use faster models, add timeout handling

### 14.5 Risk: Inconsistent Output
**Mitigation:** Lower temperature, detailed prompts, quality review step

---

## 15. Documentation Requirements

### 15.1 Code Documentation
- Docstrings for all functions (Google style)
- Type hints throughout
- Inline comments for complex logic
- README in `/agents/paraphraser/`

### 15.2 User Documentation
- Help text in UI tooltips
- Example inputs in placeholders
- FAQ section (expander in UI)
- Video tutorial (future)

### 15.3 Developer Documentation
- Architecture diagram
- API reference (for programmatic use)
- Contribution guidelines
- Testing guide

---

## Next Steps

1. **Review this plan** - Approve architecture and approach
2. **Begin implementation** - Start with Phase 1 (core agents)
3. **Iterative development** - Build, test, refine each component
4. **User testing** - Get feedback from HVAC SMEs
5. **Production deployment** - Launch in streamlit app

**Estimated Timeline:** 3-4 days for MVP, 1 week for production-ready version

---

**Questions for Discussion:**
1. Should we use a single LLM for all agents or mix models (cheaper for reviewer)?
2. What quality score threshold should trigger refinement (we can make it configurable)?
3. Should we add a "skip review" option for faster processing?
4. Any specific HVAC content we should use for initial testing?
