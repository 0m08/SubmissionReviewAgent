# Quick Agent Reference - Paraphraser Agent Setup

## What You Need to Know

### 1. Core Architecture
- **3-Layer Pattern**: Prompt Template → Agent Function → Orchestration Function
- **Chain Class**: Wraps LLM calls, handles message building, extracts tagged output
- **Sheet-Based State**: All data flows through Google Sheets (not in-memory)

### 2. Key Components

| Component | File | Purpose |
|-----------|------|---------|
| Prompt Template | `paraphrase_text.py` | Defines LLM instructions with placeholders |
| Core Agent Function | `paraphrase_text.py` | Calls Chain, returns typed output |
| Orchestration Function | `run_paraphrase.py` | Manages Sheet I/O and iteration |
| Delete Function | `run_paraphrase.py` | Cleans up generated columns |
| Pipeline Entry | `<pipeline>.py` | Registers step with dependencies |

### 3. Essential Imports

```python
from modules.chain import Chain  # LLM wrapper
from langsmith import traceable  # Monitoring/tracing
from services.sheets_service import get_sheet_data_and_df, save_to_sheet  # Sheet I/O
```

### 4. Execution Signatures

#### Agent Function (Pure LLM)
```python
@traceable(metadata={...})
def paraphrase_text(course_name, target_audience, original_text, ..., llm='gemini_2_flash'):
    agent = Chain(llm=llm, tags=['paraphrased_text', 'explanation'])
    agent.add_message(role='user', content=prompt.format(...))
    response = agent.run()
    return response['paraphrased_text'], response['explanation']
```

#### Orchestration Function (Sheet I/O)
```python
@traceable(metadata={...})
def run_paraphrase_text(sheet, source_worksheet_name, source_column_name, ...):
    sheet_obj, df = get_sheet_data_and_df(sheet, source_worksheet_name)
    # Process rows, call agent function
    save_to_sheet(sheet_obj, df)
```

### 5. Prompt Structure (Template)

```
You are an expert [ROLE].

Context:
<course_name>{course_name}</course_name>
<target_audience>{target_audience}</target_audience>
<input_text>{input_text}</input_text>

Task:
[Specific instructions]

Output format:
<result_tag>
[Your result here]
</result_tag>

<explanation>
[Why you did it this way]
</explanation>
```

### 6. Pipeline Registration

```python
{
    "name": "Paraphrase Slide Content",
    "func": run_paraphrase_text,
    "depends_on": ["Previous step"],  # Dependency tracking
    "args": {
        "sheet": "sheet",  # Maps to session_state["sheet"]
        "source_worksheet_name": "Slide Chunks",  # Direct value
        "course_name": "course_name",  # Maps to session_state
        "llm": llm_model,  # Variable from module scope
    },
    "delete_func": delete_paraphrase_text,  # Cleanup function
    "delete_args": {...}
}
```

### 7. Error Handling

```python
# Idempotency check
if 'paraphrased_text' in df.columns and df['paraphrased_text'].notna().any():
    print("Already completed. Skipping.")
    return

# Skip empty rows
if pd.isna(original_text) or str(original_text).strip() == '':
    continue

# Column existence check
if source_column_name not in df.columns:
    print(f"Column not found")
    return
```

### 8. Supported LLMs

```
'groq' → Groq Mixtral
'gemini_2_flash' → Google Gemini 2 Flash (DEFAULT)
'gpt-4o' → OpenAI GPT-4
'claude-3' → Anthropic Claude
'pplx_deep_research' → Perplexity (for research)
```

### 9. Required Decorators & Metadata

```python
@traceable(metadata={
    "agent_name": "paraphraser",           # Domain name
    "step_name": "Text Paraphraser",       # Step display name
    "function_name": "paraphrase_text",    # Actual function name
    "user_id": st.session_state.get("role", "anonymous")  # Who ran it
})
```

### 10. Common Pitfalls

| Issue | Solution |
|-------|----------|
| Tags not extracting | Make sure tags are in prompt AND Chain tags param |
| Sheet not saving | Call `save_to_sheet(worksheet, df)` after modification |
| Already processed | Add check: `if df[column].notna().any(): return` |
| Dependencies missing | Add all upstream steps to `depends_on` list |
| LLM errors | Add `@try_n_times(3)` decorator for retry |
| Session state empty | Ensure step runs inside Streamlit or set manually |

### 11. File Checklist for New Agent

- [ ] `/agents/paraphraser/__init__.py` (empty)
- [ ] `/agents/paraphraser/paraphrase_text.py` (core logic)
- [ ] `/agents/paraphraser/run_paraphrase.py` (orchestration + delete)
- [ ] Updated `course_outline.py` or pipeline file with pipeline_sections entry
- [ ] Updated `about_agents.md` with agent documentation
- [ ] Tested with sample data

### 12. Testing Locally

```python
# Set up session state
import streamlit as st
st.session_state["role"] = "developer"

# Test core agent
from agents.paraphraser.paraphrase_text import paraphrase_text

result = paraphrase_text(
    course_name="HVAC Fundamentals",
    target_audience="Beginners",
    original_text="Sample text to paraphrase",
    llm="gemini_2_flash"
)
print(result)
```

### 13. Monitoring & Debugging

```python
# LangSmith tracing automatically logs:
- LLM prompts and responses
- Token usage
- Execution time
- User/step information
- Error stack traces

# View at: https://smith.langchain.com/
```

### 14. Production Deployment

```bash
# CLI execution
python run_agent_cli.py \
    --sheet_link "https://docs.google.com/spreadsheets/d/..." \
    --drive_folder_id "folder_id" \
    --agent_name "paraphraser"

# SDK execution (Lightning AI)
python launch_agents_via_sdk.py \
    --sheet_link "..." \
    --drive_folder_id "..." \
    --agent_name "paraphraser"
```

### 15. Key Files Reference

**Must Read:**
- `/modules/chain.py` - Chain class implementation
- `/agents/course_outline/video_based_outline/video_search_queries.py` - Simple agent example
- `/agents/research_notes/generate_notes.py` - Complex agent example

**Services:**
- `/services/sheets_service.py` - Google Sheets operations
- `/services/llm_service.py` - LLM configuration and selection
- `/services/helper_functions.py` - Utility functions

**Execution:**
- `/course_outline.py` - Example pipeline with pipeline_sections
- `/run_agent_cli.py` - CLI entry point
- `/launch_agents_via_sdk.py` - SDK/Cloud entry point

---

## Minimal Working Example

```python
# agents/paraphraser/paraphrase_text.py
from modules.chain import Chain
from langsmith import traceable
import streamlit as st

PROMPT = """Paraphrase: {text}
<result>{paraphrased}</result>"""

@traceable(metadata={"agent_name": "paraphraser", "step_name": "Paraphraser"})
def paraphrase_text(text, llm='gemini_2_flash'):
    agent = Chain(llm=llm, tags=['result'])
    agent.add_message(role='user', content=PROMPT.format(text=text))
    return agent.run()['result']

# agents/paraphraser/run_paraphrase.py
from agents.paraphraser.paraphrase_text import paraphrase_text
from services.sheets_service import get_sheet_data_and_df, save_to_sheet
from langsmith import traceable

@traceable(metadata={"agent_name": "paraphraser", "step_name": "Paraphraser"})
def run_paraphrase_text(sheet, worksheet_name, column_name, llm='gemini_2_flash'):
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    df['paraphrased'] = df[column_name].apply(
        lambda x: paraphrase_text(x, llm) if pd.notna(x) else ''
    )
    save_to_sheet(ws, df)

# In course_outline.py or pipeline
pipeline_sections = [{
    "steps": [{
        "name": "Paraphrase Text",
        "func": run_paraphrase_text,
        "args": {
            "sheet": "sheet",
            "worksheet_name": "Slide Chunks",
            "column_name": "Content",
        },
        "delete_func": lambda sheet, **kw: None,
    }]
}]
```

---

This reference contains everything you need to create and deploy agents following the established patterns!
