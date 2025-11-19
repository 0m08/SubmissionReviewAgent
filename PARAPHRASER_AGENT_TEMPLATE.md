# Paraphraser Agent Specification Template

Based on the existing agent patterns in this codebase, here's the exact structure you should follow:

---

## FILE STRUCTURE

```
/agents/
├── paraphraser/                          [NEW DOMAIN]
│   ├── __init__.py
│   ├── paraphrase_text.py               [CORE AGENT FUNCTION]
│   └── run_paraphrase.py                [ORCHESTRATION + DELETE FUNCTIONS]
```

---

## 1. CREATE: `/agents/paraphraser/__init__.py`

```python
# Empty init file for package structure
```

---

## 2. CREATE: `/agents/paraphraser/paraphrase_text.py`

**Purpose**: Contains the core agent logic (prompt + LLM call)

```python
from modules.chain import Chain
import streamlit as st
from langsmith import traceable

# =========================
# PROMPT TEMPLATE
# =========================
paraphrase_text_prompt = """You are an expert text paraphraser. Your task is to rephrase content while maintaining:
1. The original meaning and intent
2. Technical accuracy
3. All key information
4. Appropriate tone for the target audience
5. Improved clarity and readability

Course Context:
<course_name>
{course_name}
</course_name>

<target_audience>
{target_audience}
</target_audience>

<context>
{context}
</context>

Text to Paraphrase:
<original_text>
{original_text}
</original_text>

Additional Instructions:
{additional_instructions}

Please provide your paraphrased version in the following format:

<paraphrased_text>
[Your paraphrased version here - maintain professional, educational tone]
</paraphrased_text>

<explanation>
[Brief explanation of key changes made and why]
</explanation>
"""

# =========================
# CORE AGENT FUNCTION
# =========================
@traceable(metadata={
    "agent_name": "paraphraser",
    "step_name": "Text Paraphraser",
    "function_name": "paraphrase_text",
    "user_id": st.session_state.get("role", "anonymous")
})
def paraphrase_text(
    course_name,
    target_audience,
    original_text,
    context="",
    additional_instructions="Focus on clarity and readability.",
    llm='gemini_2_flash'
):
    """
    Paraphrases text while maintaining meaning and improving clarity.
    
    :param course_name (str): Name of the course for context
    :param target_audience (str): Target audience for the course
    :param original_text (str): Text to be paraphrased
    :param context (str): Additional context about the content
    :param additional_instructions (str): Special instructions for paraphrasing
    :param llm (str): Language model to use
    :returns: tuple: (paraphrased_text, explanation)
    """
    
    # Create agent with tags for structured output extraction
    paraphrase_agent = Chain(
        llm=llm,
        tags=['paraphrased_text', 'explanation']
    )
    
    # Add the user message with formatted prompt
    paraphrase_agent.add_message(
        role='user',
        content=paraphrase_text_prompt.format(
            course_name=course_name,
            target_audience=target_audience,
            original_text=original_text,
            context=context,
            additional_instructions=additional_instructions
        )
    )
    
    # Run the agent and extract tagged sections
    response = paraphrase_agent.run()
    
    paraphrased_text = response.get('paraphrased_text', '')
    explanation = response.get('explanation', '')
    
    return paraphrased_text, explanation
```

---

## 3. CREATE: `/agents/paraphraser/run_paraphrase.py`

**Purpose**: Orchestration function for Sheet I/O and delete function

```python
from modules.chain import Chain
from agents.paraphraser.paraphrase_text import paraphrase_text
from services.sheets_service import (
    get_sheet_data_and_df,
    save_to_sheet,
    clear_worksheet,
)
import pandas as pd
import streamlit as st
from langsmith import traceable

# =========================
# ORCHESTRATION FUNCTION
# =========================
@traceable(metadata={
    "agent_name": "paraphraser",
    "step_name": "Text Paraphraser",
    "function_name": "run_paraphrase_text",
    "user_id": st.session_state.get("role", "anonymous")
})
def run_paraphrase_text(
    sheet,
    source_worksheet_name,
    target_worksheet_name,
    source_column_name,
    course_name,
    target_audience,
    context="",
    additional_instructions="",
    llm='gemini_2_flash'
):
    """
    Orchestration function that:
    1. Reads text from source worksheet
    2. Calls paraphrase_text() for each row
    3. Writes results to target worksheet
    
    :param sheet: Google Sheets object
    :param source_worksheet_name: Name of sheet with original text
    :param target_worksheet_name: Name of sheet to write paraphrased text
    :param source_column_name: Name of column containing text to paraphrase
    :param course_name: Course name for context
    :param target_audience: Target audience for context
    :param context: Additional context
    :param additional_instructions: Special paraphrasing instructions
    :param llm: LLM model to use
    :return: None
    """
    
    # Read source data
    source_sheet, source_df = get_sheet_data_and_df(
        sheet, 
        source_worksheet_name
    )
    
    # Check if column exists
    if source_column_name not in source_df.columns:
        print(f"Column '{source_column_name}' not found in {source_worksheet_name}")
        return
    
    # Check if paraphrased column already exists
    if 'paraphrased_text' in source_df.columns:
        # Check if already processed
        if source_df['paraphrased_text'].notna().any():
            print("Paraphrased text already exists. Skipping.")
            return
    
    # Add columns if they don't exist
    if 'paraphrased_text' not in source_df.columns:
        source_df['paraphrased_text'] = ''
    if 'paraphrase_explanation' not in source_df.columns:
        source_df['paraphrase_explanation'] = ''
    
    # Process each row
    for idx, row in source_df.iterrows():
        original_text = row[source_column_name]
        
        # Skip if empty
        if pd.isna(original_text) or str(original_text).strip() == '':
            continue
        
        # Skip if already paraphrased
        if pd.notna(row['paraphrased_text']) and str(row['paraphrased_text']).strip() != '':
            continue
        
        print(f"Paraphrasing row {idx + 1}...")
        
        # Call agent function
        paraphrased_text, explanation = paraphrase_text(
            course_name=course_name,
            target_audience=target_audience,
            original_text=str(original_text),
            context=context,
            additional_instructions=additional_instructions,
            llm=llm
        )
        
        # Store results
        source_df.at[idx, 'paraphrased_text'] = paraphrased_text
        source_df.at[idx, 'paraphrase_explanation'] = explanation
    
    # Save results back to sheet
    save_to_sheet(worksheet=source_sheet, df=source_df)
    print(f"Completed paraphrasing. Results saved to {source_worksheet_name}")


# =========================
# DELETE FUNCTION
# =========================
@traceable(metadata={
    "agent_name": "paraphraser",
    "step_name": "Text Paraphraser",
    "function_name": "delete_paraphrase_text",
    "user_id": st.session_state.get("role", "anonymous")
})
def delete_paraphrase_text(
    sheet,
    worksheet_name="Slide Chunks"
):
    """
    Deletes paraphrased text columns from worksheet.
    
    :param sheet: Google Sheets object
    :param worksheet_name: Name of worksheet to clean
    :return: None
    """
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    
    # Remove paraphrase columns if they exist
    columns_to_remove = ['paraphrased_text', 'paraphrase_explanation']
    for col in columns_to_remove:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Clear and save
    clear_worksheet(ws)
    save_to_sheet(ws, df)
    print(f"Removed paraphrased text columns from {worksheet_name}")
```

---

## 4. UPDATE: Add to Pipeline

Add to the appropriate pipeline file (e.g., `slide_chunks.py` or new `paraphraser.py`):

```python
from agents.paraphraser.run_paraphrase import run_paraphrase_text, delete_paraphrase_text

# In pipeline_sections:
{
    "section_name": "Text Paraphrasing",
    "steps": [
        {
            "name": "Paraphrase Slide Content",
            "func": run_paraphrase_text,
            "depends_on": ["Previous step name"],  # Adjust based on your workflow
            "args": {
                "sheet": "sheet",
                "source_worksheet_name": "Slide Chunks",
                "target_worksheet_name": "Slide Chunks",
                "source_column_name": "Slide Content",
                "course_name": "course_name",
                "target_audience": "target_audience",
                "context": "course_background",
                "llm": llm_model,
            },
            "estimated_time": "~ 2-5 minutes (depends on content volume)",
            "description": "Paraphrases slide content for improved clarity while maintaining educational accuracy.",
            "delete_func": delete_paraphrase_text,
            "delete_args": {
                "sheet": "sheet",
                "worksheet_name": "Slide Chunks",
            }
        },
    ]
}
```

---

## 5. UPDATE: Documentation

Add to `/about_agents.md` under appropriate section:

```markdown
### Text Paraphraser
**Description**: Rephases course content for improved clarity and readability while maintaining technical accuracy and educational intent.

**How to Use**:
1. Ensure slide content is in the `Slide Chunks` sheet
2. Run the paraphrasing agent
3. Review paraphrased text in the sheet
4. Compare with original using the explanation column

**Tools Used**: 
- LLM (Language Model)
```

---

## KEY DESIGN DECISIONS

### 1. **Prompt Structure**
- Includes course context (name, audience, background)
- Uses XML tags for structured output (`<paraphrased_text>`, `<explanation>`)
- Specifies constraints (maintain meaning, tone, accuracy)

### 2. **Two-Level Architecture**
- **Core agent** (`paraphrase_text`): Pure LLM logic, no side effects
- **Orchestration** (`run_paraphrase_text`): Handles Sheet I/O, row iteration, error handling

### 3. **Idempotency**
- Checks if already processed before re-running
- Safe to run multiple times without duplication

### 4. **Tracing & Monitoring**
- Uses `@traceable` decorator for LangSmith integration
- Tracks which user ran which step
- Enables debugging and cost analysis

### 5. **Flexible LLM Selection**
- Defaults to `gemini_2_flash` (fast, good quality)
- Can use any supported model (groq, claude, etc.)
- LLM passed as parameter for easy switching

---

## CUSTOMIZATION OPTIONS

### For Multi-Stage Paraphrasing:
```python
# Create separate functions
def generate_initial_paraphrase(...):
    # Generate first version
    
def refine_paraphrase(...):
    # Refine based on criteria
    
# Then orchestrate:
initial = generate_initial_paraphrase(...)
refined = refine_paraphrase(initial, ...)
```

### For Different Paraphrasing Styles:
```python
# Parameterize the prompt
def paraphrase_text(
    original_text,
    style='professional',  # or 'casual', 'technical', etc.
    ...
):
    style_instructions = {
        'professional': 'Use formal, academic language',
        'casual': 'Use conversational, approachable language',
        'technical': 'Emphasize technical precision...',
    }
    
    prompt = prompt_template.format(
        style_guide=style_instructions[style],
        ...
    )
```

### For Batch Processing:
```python
# Use concurrent processing
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_paraphrase_text_parallel(...):
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(
                paraphrase_text,
                row[source_column_name],
                ...
            )
            for _, row in source_df.iterrows()
        ]
        
        for future in as_completed(futures):
            paraphrased, explanation = future.result()
```

---

## TESTING YOUR AGENT

```python
# Quick test before adding to pipeline
if __name__ == "__main__":
    import streamlit as st
    
    # Initialize session state
    st.session_state["role"] = "developer"
    
    # Test the core function
    result = paraphrase_text(
        course_name="HVAC Fundamentals",
        target_audience="Beginners",
        original_text="The refrigerant cycle involves four main components: compressor, condenser, expansion valve, and evaporator.",
        additional_instructions="Make it more accessible for non-technical readers.",
        llm="gemini_2_flash"
    )
    
    print("Paraphrased:", result[0])
    print("Explanation:", result[1])
```

