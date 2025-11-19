# Agent Structure Analysis - Content Generation Workflow

## Overview
This codebase implements a modular, pipeline-based agent system for automated course content generation. Each agent is a Python function that processes inputs, calls LLMs, and outputs structured results.

---

## 1. WHERE AGENT SPECIFICATIONS/CONFIGS ARE STORED

### Primary Locations:
1. **Pipeline Definition**: `/home/user/Content-Generation-Workflow/course_outline.py` (and similar pipeline files)
   - Contains `pipeline_sections` variable with complete agent workflow specifications
   - Defines steps, dependencies, arguments, and handlers

2. **Agent Implementation Files**: `/home/user/Content-Generation-Workflow/agents/`
   - Organized by domain (course_outline, research_notes, graphics_definition, etc.)
   - Each agent is a standalone Python module/function

3. **Documentation**: `/home/user/Content-Generation-Workflow/about_agents.md`
   - Human-readable specification of all agents
   - Describes purpose, how to use, and tools for each agent

### Configuration Structure:
Agents are NOT configured via separate config files (JSON/YAML) but via:
- **Pipeline sections** dictionary in Python modules
- **Function parameters** passed during execution
- **Sheet-based state management** (Google Sheets)

---

## 2. HOW AGENTS ARE DEFINED (CODE STRUCTURE)

### Agent Definition Pattern (3 Levels):

#### Level 1: Core Agent Function
Located in: `agents/<domain>/<subdomain>/<agent_name>.py`

Example: `/agents/course_outline/video_based_outline/video_search_queries.py`

```python
from modules.chain import Chain
from langsmith import traceable

# 1. DEFINE PROMPT TEMPLATE (with variable placeholders)
generate_video_search_queries_prompt = """You are tasked with generating a list of search queries...
<course_name>{course_name}</course_name>
<target_audience>{target_audience}</target_audience>
<course_outline>{course_outline}</course_outline>
...
"""

# 2. CREATE FUNCTION WITH @traceable DECORATOR
@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Video Search Query Generator",
    "function_name": "generate_video_search_queries",
    "user_id": st.session_state.get("role", "anonymous")
})
def generate_video_search_queries(course_name, target_audience, course_outline, llm):
    """
    Function to generate search queries
    
    :param course_name (str): Course name
    :param target_audience (str): Target audience
    :param course_outline (str): Course outline
    :param llm (str): Language model to use
    :returns: search_queries (list): List of search queries
    """
    # 3. INSTANTIATE CHAIN WITH LLM AND TAGS
    generate_query_agent = Chain(
        llm=llm, 
        tags=['search_queries']  # Tags for output extraction
    )
    
    # 4. ADD FORMATTED MESSAGE
    generate_query_agent.add_message(
        role="user",
        content=generate_video_search_queries_prompt.format(
            course_name=course_name,
            target_audience=target_audience,
            course_outline=course_outline
        )
    )
    
    # 5. RUN AND EXTRACT TAGGED OUTPUT
    response = generate_query_agent.run()
    search_queries = extract_csv_lines(response['search_queries'])
    
    return search_queries
```

#### Level 2: Workflow/Orchestration Function
Coordinates multiple agent calls and manages side effects (Google Sheets I/O)

```python
@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Video Search Query Generator",
    "function_name": "run_construct_video_search_queries",
    "user_id": st.session_state.get("role", "anonymous")
})
def run_construct_video_search_queries(
    sheet, 
    course_name, 
    target_audience, 
    worksheet_name='Base Outline', 
    llm='groq'
):
    """
    Orchestration function that:
    1. Reads from Google Sheets
    2. Calls agent function(s)
    3. Writes results back to Google Sheets
    """
    # Get data from sheet
    rough_outline_sheet, rough_outline_df = get_sheet_data_and_df(
        sheet, worksheet_name
    )
    
    # Call agent function
    video_search_queries = generate_video_search_queries(
        course_name=course_name,
        target_audience=target_audience,
        course_outline=course_outline,
        llm=llm
    )
    
    # Save to sheet
    rough_outline_df = add_list_as_new_column(
        rough_outline_df, 
        video_search_queries, 
        'video_search_queries'
    )
    save_to_sheet(worksheet=rough_outline_sheet, df=rough_outline_df)
```

#### Level 3: Pipeline Configuration
In: `course_outline.py`, `research_notes.py`, etc.

```python
pipeline_sections = [
    {
        "section_name": "Section 1: Videos Research",
        "steps": [
            {
                "name": "Video Search Query Generator",
                "func": run_construct_video_search_queries,  # The orchestration function
                "depends_on": [],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Base Outline",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": llm_model,
                },
                "estimated_time": "~ 1 minute",
                "description": "Generates search queries to identify relevant videos.",
                "delete_func": delete_video_search_queries,
                "delete_args": {...}
            },
            # ... more steps
        ]
    }
]
```

---

## 3. EXISTING PATTERNS FOR AGENT BEHAVIOR, PERSONA, AND PROMPTS

### Pattern 1: Structured Analysis Pattern
Used for agents that need to reason through steps before generating output.

**File**: `/agents/graphics_definition/define_graphics/generate_graphics_definition.py`

```python
# Pattern includes two XML sections:
<evaluation_breakdown>
[Analysis: Key instructional message, Core concept, Logical scene segmentation, etc.]
</evaluation_breakdown>

<graphics_definition>
[Actual output with structured scene definitions]
</graphics_definition>
```

### Pattern 2: Multi-Stage Processing
For complex tasks that benefit from generation + refinement

**File**: `/agents/course_outline/web_research_based_outline/web_search_queries.py`

```python
# Two-stage approach:
1. generate_search_query() - Creates initial queries
2. refine_search_queries() - Refines them for quality

# Both use Chain class with tags:
generate_query_agent = Chain(llm=llm, tags=['answer'])
```

### Pattern 3: Context-Aware Research
Agents provide full course context to ensure outputs align with course structure

**File**: `/agents/research_notes/generate_notes.py`

Includes:
- Full course outline
- Target audience
- Learning objectives
- Relevant documents/transcripts

### Pattern 4: Tagged Output Extraction
All agents use XML tags for structured output extraction

```python
# In prompt:
<search_queries>
[First search query]
[Second search query]
</search_queries>

# In code:
generate_query_agent = Chain(llm=llm, tags=['search_queries'])
response = generate_query_agent.run()
search_queries = extract_csv_lines(response['search_queries'])
```

### Persona Patterns:
1. **Expert Educational Content Developer** - Used in research notes generation
2. **Graphics Definition Agent** - For visual content specification
3. **Domain Expert** - For deep research and topic analysis

---

## 4. HOW PARAMETERS ARE PASSED TO AGENTS

### Parameter Passing Hierarchy:

#### Level 1: Environment & Session State
```python
# Via Streamlit session state
st.session_state["course_name"] = "HVAC Fundamentals"
st.session_state["target_audience"] = "Beginners"

# Accessed in agents:
@traceable(metadata={
    "user_id": st.session_state.get("role", "anonymous")
})
```

#### Level 2: Function Arguments
Direct parameters passed to orchestration functions:
```python
run_construct_video_search_queries(
    sheet=sheet_object,
    course_name="HVAC Fundamentals",
    target_audience="Beginners",
    worksheet_name="Base Outline",
    llm="groq"
)
```

#### Level 3: Pipeline Arguments Mapping
In pipeline_sections, args are mapped:
```python
"args": {
    "sheet": "sheet",              # Maps to session_state["sheet"]
    "worksheet_name": "Base Outline",  # Direct value
    "course_name": "course_name",  # Maps to session_state["course_name"]
    "target_audience": "target_audience",  # Maps to session_state["target_audience"]
    "llm": llm_model,              # Variable from module scope
}
```

#### Level 4: Prompt Template Parameters
```python
prompt.format(
    course_name=course_name,
    target_audience=target_audience,
    course_outline=course_outline,
    generated_queries='\n'.join(search_queries)
)
```

---

## 5. THE CHAIN CLASS: CORE ABSTRACTION

**File**: `/modules/chain.py`

The `Chain` class is the core abstraction for all agents:

```python
class Chain:
    def __init__(self, llm='groq', tags=None, use_xml_checker=False, 
                 use_output_parser=True, max_tokens=None):
        self.llm = llm
        self.tags = tags  # For extracting XML tags from output
        self.messages_list = []
        
    def add_message(self, role, content):
        """Add a message to conversation"""
        self.messages_list.append((role, content))
        
    def run(self, **inputs):
        """Execute the chain and return extracted output"""
        chain = self._build_chain()
        response = chain.invoke(inputs)
        
        # If tags specified, extracts those sections from response
        if self.tags:
            response = self.extract_text_in_tags(response)
        
        return response
```

### Key Features:
1. **Message-based conversation** - Supports multi-turn interactions
2. **LLM flexibility** - Configurable LLM (groq, gemini, gpt, etc.)
3. **Automatic tag extraction** - Extracts content within XML tags
4. **Self-correction** - Can retry and fix malformed output
5. **Token logging** - Tracks usage for cost monitoring

---

## 6. DEPENDENCY MANAGEMENT

### Sheet-Based State:
Agents don't maintain state internally; they use Google Sheets:

```python
# Read current state
sheet_obj, df = get_sheet_data_and_df(sheet, worksheet_name)

# Check if step already done
if 'search_queries' in df.columns and df['search_queries'][0] != '':
    print("Already completed. Skipping.")
    return

# Write results
save_to_sheet(worksheet=sheet_obj, df=df)
```

### Dependency Resolution:
Pipeline executor uses `depends_on` field:

```python
{
    "name": "Retrieve Video Transcripts",
    "depends_on": ["Retrieve HVAC School Videos"],
    "func": run_get_transcripts,
    ...
}
```

Pipeline executor ensures prerequisites complete before running.

---

## 7. SUPPORTED LLMS

**File**: `/services/llm_service.py`

```python
# Available LLMs:
- groq (Groq Mixtral)
- gemini_2_flash (Google Gemini 2 Flash)
- gpt-4o (OpenAI)
- claude-3 (Anthropic)
- pplx_deep_research (Perplexity)
- gemini_with_grounding (Gemini with search)
```

Selected via pipeline configuration:
```python
llm_model = st.session_state.get("llm_model", "gemini_2_flash")
```

---

## 8. TRACING & MONITORING

All agents use `@traceable` decorator from LangSmith:

```python
@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Video Search Query Generator",
    "function_name": "generate_video_search_queries",
    "user_id": st.session_state.get("role", "anonymous")
})
def generate_video_search_queries(...):
```

Benefits:
- Automatic LLM call logging
- Performance monitoring
- Debugging and replay
- Token usage tracking

---

## 9. ERROR HANDLING PATTERNS

### Retries:
```python
@try_n_times(3)  # From utils.decorator_helpers
def run(self, **inputs):
    chain = self._build_chain()
    response = chain.invoke(inputs)
```

### Output Validation:
```python
# XML Checking
if self.use_xml_checker:
    response = xml_check_and_fix(response, llm=self.llm)

# Tag Extraction with self-correction
extract_text_in_tags(text)  # Retries with LLM correction if fails
```

---

## 10. AGENT EXECUTION FLOW

### CLI Execution (for external triggers):
```
run_agent_cli.py
├── Load pipeline_sections from module
├── Initialize state from Google Sheets
└── run_all_automated_steps_for_cli()
    ├── For each section:
    │   └── For each step:
    │       ├── Check if dependencies complete
    │       ├── Run step function
    │       ├── Log completion to sheet
    │       └── Handle errors
```

### Streamlit Execution (UI-based):
```
streamlit_app.py or course_outline.py
├── Display agent_ui()
├── User clicks "Run Step"
└── Execute step function
    ├── Update session state
    ├── Run orchestration function
    └── Display results in UI
```

---

## SUMMARY: AGENT CREATION CHECKLIST

To create a new agent like the paraphraser:

1. **Create agent file**: `/agents/<domain>/<agent_name>.py`
2. **Define prompt template** with XML tags for structured output
3. **Create agent function** using Chain class with tags
4. **Create orchestration function** that handles Sheet I/O
5. **Add @traceable decorator** with agent_name and step_name
6. **Create delete function** to clean up generated data
7. **Add to pipeline_sections** in the main pipeline file
8. **Document in about_agents.md**

---

## KEY FILES REFERENCE

- **Core Abstractions**: 
  - `/modules/chain.py` - Chain class
  - `/modules/proposer_agents.py` - Agent class definition

- **Services**:
  - `/services/llm_service.py` - LLM integration
  - `/services/sheets_service.py` - Google Sheets I/O
  - `/services/helper_functions.py` - Utility functions

- **Example Agents**:
  - `/agents/course_outline/video_based_outline/video_search_queries.py` - Simple agent
  - `/agents/research_notes/generate_notes.py` - Complex agent
  - `/agents/graphics_definition/define_graphics/generate_graphics_definition.py` - Structured analysis agent

- **Pipeline Definitions**:
  - `/course_outline.py` - Main course outline pipeline
  - `/research_notes.py` - Research notes pipeline
  - `/slide_chunks.py` - Slide chunks pipeline

