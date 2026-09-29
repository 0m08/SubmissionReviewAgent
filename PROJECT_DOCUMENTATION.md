# Content Generation Workflow - Complete Project Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Workflows & Pages](#workflows--pages)
5. [Core Components](#core-components)
6. [Setup & Installation](#setup--installation)
7. [Usage Guide](#usage-guide)
8. [Development Guide](#development-guide)
9. [API & Services](#api--services)
10. [Agent System](#agent-system)

---

## Project Overview

**Content Generation Workflow** is an AI-powered course generation platform built with Streamlit. The application streamlines the process of creating educational content through a series of automated and semi-automated agentic workflows. Each workflow handles a specific phase of course development, from initial outline creation to final assessment generation.

### Key Features
- **Multi-Stage Course Generation**: From outline to assessment questions
- **AI-Powered Content Creation**: Leverages multiple LLM models (Gemini, GPT)
- **Google Workspace Integration**: Works seamlessly with Google Drive and Sheets
- **Role-Based Access Control**: Different permissions for Admins, Editors, Content Heads, Instructional Designers, and Visual Designers
- **Manual Review Checkpoints**: Human-in-the-loop validation at critical stages
- **Background Processing**: Run agents asynchronously on Lightning.ai
- **Checklist-Based Quality Control**: Automated quality and compliance scoring

### Tech Stack
- **Frontend**: Streamlit
- **Backend**: Python 3.x
- **AI/ML**: LangChain, Google Gemini, OpenAI GPT
- **Storage**: Google Drive, Google Sheets
- **Authentication**: OAuth 2.0
- **Cloud Deployment**: Lightning.ai, Streamlit Cloud
- **Monitoring**: LangTrace, LangSmith

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Streamlit App                            │
│                      (streamlit_app.py)                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                │   Authentication        │
                │   (OAuth 2.0)          │
                └────────────┬────────────┘
                             │
        ┌────────────────────┴────────────────────┐
        │                                         │
┌───────▼────────┐                    ┌──────────▼──────────┐
│  Agent Pages   │                    │  Tool Pages         │
└───────┬────────┘                    └──────────┬──────────┘
        │                                        │
        │                                        │
┌───────▼─────────────────────────────────────┬─▼──────────┐
│  Agentic Workflows (Pipeline Sections)      │  Tools     │
├──────────────────────────────────────────────┴────────────┤
│  • Template Sheet Setup                                   │
│  • Course Outline Generation                              │
│  • Research Notes Generation                              │
│  • Slide Chunks Generation                                │
│  • Graphics Definition                                    │
│  • Assessment Generation                                  │
│  • Image Search & Graphics Search                         │
│  • Quality Compliance Scoring                             │
│  • Video Search Tool                                      │
└───────────────────────────────┬───────────────────────────┘
                                │
                   ┌────────────┴────────────┐
                   │                         │
          ┌────────▼────────┐     ┌─────────▼─────────┐
          │   Services      │     │   LLM Models      │
          ├─────────────────┤     ├───────────────────┤
          │ • Drive         │     │ • Gemini 2 Flash  │
          │ • Sheets        │     │ • Gemini 2.5 Flash│
          │ • Web Search    │     │ • GPT-5 Mini      │
          │ • YouTube       │     │ • Gemini w/Ground │
          │ • Embeddings    │     └───────────────────┘
          │ • LLM           │
          │ • Chunking      │
          └─────────────────┘
```

### Workflow Pattern

Each agentic workflow follows a consistent pattern:

1. **Data Loading**: Load course information from Google Sheets
2. **Pipeline Sections**: Workflows are divided into logical sections
3. **Steps**: Each section contains sequential or parallel steps
4. **Dependencies**: Steps can depend on previous steps
5. **Manual Checkpoints**: Human review points where needed
6. **AI Processing**: LLM-powered generation and revision
7. **Quality Control**: Checklist-based review and scoring
8. **Output**: Results written back to Google Sheets

---

## Project Structure

```
Content-Generation-Workflow/
│
├── streamlit_app.py                 # Main entry point
├── agent_ui_template.py             # Reusable UI template for agents
├── about_agents.py/.md              # Agent documentation
├── launch_agents_via_sdk.py         # Background agent launcher
├── run_agent_cli.py                 # CLI for batch jobs
│
├── agents/                          # Agent implementations
│   ├── course_outline/              # Course outline generation
│   │   ├── video_based_outline/
│   │   ├── web_research_based_outline/
│   │   ├── deep_research/
│   │   ├── outline_consolidation/
│   │   ├── enhance_outline/
│   │   └── course_outline_checklist/
│   │
│   ├── research_notes/              # Research notes generation
│   ├── slide_chunks/                # Slide content generation
│   ├── graphics_definition/         # Graphics definition creation
│   ├── generate_assessments/        # Assessment question generation
│   ├── graphics_search/             # Graphics search tools
│   ├── vector_store_image_search/   # Image search with vectorstore
│   ├── quality_compliance_scoring/  # Quality scoring
│   └── template_sheet_setup/        # Initial sheet setup
│
├── services/                        # Core services
│   ├── drive_service.py             # Google Drive integration
│   ├── sheets_service.py            # Google Sheets integration
│   ├── llm_service.py               # LLM model interface
│   ├── web_search.py                # Web search functionality
│   ├── youtube_video_loader.py      # YouTube integration
│   ├── chunking_service.py          # Text chunking utilities
│   ├── embedding_service.py         # Embedding generation
│   ├── helper_functions.py          # Utility functions
│   └── smart_progress_bar.py        # Progress tracking
│
├── utils/                           # Utility modules
│   ├── role_utils.py                # Role-based access control
│   ├── decorator_helpers.py         # Decorator utilities
│   └── file_helpers.py              # File utilities
│
├── config/                          # Configuration
│   └── user_roles.py                # User role definitions
│
├── modules/                         # Additional modules
│   ├── chain.py                     # LangChain utilities
│   └── proposer_agents.py           # Proposer agent implementations
│
├── assets/                          # Static assets
│   └── google-login-button.png
│
├── .streamlit/                      # Streamlit configuration
├── requirements.txt                 # Python dependencies
├── packages.txt                     # System packages
├── Dockerfile                       # Docker configuration
├── .env.example                     # Environment variables template
└── README.md                        # Setup documentation
```

---

## Workflows & Pages

### 1. Template Sheet Setup
**File**: `template_sheet_agent.py`

**Purpose**: Initialize a Google Sheet with the required structure for course generation.

**Key Steps**:
- Create template worksheets
- Set up standard columns
- Configure initial formatting

---

### 2. Course Outline Generation
**File**: `course_outline.py`

**Purpose**: Generate a comprehensive course outline through multiple research approaches.

#### Workflow Sections

##### Section 1: Videos Research
1. **Video Search Query Generator**: Generate YouTube search queries
2. **Manual Review**: Review and refine queries
3. **Retrieve HVAC School Videos**: Fetch videos from YouTube
4. **Retrieve Video Transcripts**: Get full transcripts
5. **Check Video Relevance**: AI classification of videos
6. **Chunk Videos**: Segment transcripts into chunks
7. **Identify Relevant Chunks**: Find pertinent segments
8. **Manual Review**: Mark relevant videos
9. **Generate Video Based Outlines**: Create outlines from videos
10. **Manual Review**: Consolidation comments
11. **Consolidate Video Based Outlines**: Merge into cohesive outline

##### Section 2: Web Research
1. **Web Search Queries Generator**: Generate web search queries
2. **Obtain Web Article Links**: Search and screen articles
3. **Fetch Web Article Content**: Retrieve article content
4. **Extract Relevant Information**: Extract pertinent info
5. **Generate Research Summaries**: Summarize findings
6. **Manual Review**: Review summaries
7. **Generate Web Research Based Outline**: Create outline

##### Section 3: Deep Research
1. **Deep Research**: Perform in-depth research with grounding
2. **Generate Deep Research Based Outline**: Create research-based outline

##### Section 4: Outline Consolidation
1. **Generate Consolidated Outline**: Merge all outlines
2. **Review and Revise Outline**: Iterative review process

##### Section 5: Enhance Outline (Optional)
1. **Create Topic Outline Sheet**: Manual topic breakdown
2. **Topic Deep Research**: Research each topic
3. **Generate Learning Objectives**: Create LOs
4. **Categorize Learning Objectives**: Classify LOs
5. **Label Learning Objectives**: Tag LOs
6. **Review and Revise Topic Outline**: Refine enhanced outline
7. **Map Topic Outline to Enhanced Outline**: Align structures

##### Section 6: Final Steps
1. **Create Final Outline Sheet**: Flatten outline structure
2. **Checklist Based Review and Revise**: Quality control
3. **Get References for Learning Objectives**: Load references
4. **Retrieve References**: Context retrieval
5. **Retrieve HVAC Videos for LOs**: Find relevant videos

**LLM Models Used**:
- Gemini 2 Flash
- Gemini 2.5 Flash
- Gemini with Grounding

---

### 3. Research Notes Generation
**File**: `research_notes.py`

**Purpose**: Generate detailed research notes for each learning objective.

#### Workflow Sections

##### Section 1: Research Notes Generation
1. **Generate Context from Provided References**: Extract context
2. **Researcher**: Generate initial research notes

##### Section 2: Research Notes Revision
1. **Manually Review the Research Notes**: Human review

##### Section 3: Checklist Based Review-Revise
1. **Checklist Based Review and Revise Agents**: Quality control

**LLM Models Used**:
- Gemini 2.5 Flash
- GPT-5 Mini Thinking

---

### 4. Slide Chunks Generation
**File**: `slide_chunks.py`

**Purpose**: Convert research notes into structured slide content.

#### Workflow Sections

##### Section 1: Generate and Parse Slide Chunks
1. **Generate Slide Chunks from Research Notes**: Create slide content
2. **Slide Chunks Parsing**: Parse and structure slides

##### Section 2: Checklist Review
1. **Slide Chunks Checklist Review and Revise**: Quality control

**LLM Models Used**:
- Gemini 2 Flash
- GPT-5 Mini Thinking

---

### 5. Graphics Definition
**File**: `graphics_definition.py`

**Purpose**: Generate detailed graphics definitions for each slide.

#### Workflow Sections

##### Section 1: Graphics Definition Generation
1. **Generate Graphics Definition**: Scene-by-scene graphics (Semi-automated with human review)

##### Section 2: Graphics Definition Checklist
1. **Checklist Evaluation**: Quality control

##### Section 3: Short Graphics Definition
1. **Short Graphics Definition**: Condensed versions

**LLM Models Used**:
- Gemini 2 Flash

---

### 6. Assessment Generation
**File**: `assessment.py`

**Purpose**: Generate assessment questions for the course.

#### Workflow Sections

##### Section 1: Assessment
1. **Generate Assessment Questions**: Create questions
2. **Review and Revise Assessment Questions**: Refine questions
3. **Update Assessment Checklist**: Quality control
4. **Update Review Agent Checklist**: Final review

**LLM Models Used**:
- Gemini 2.5 Flash

---

### 7. Tools

#### Image Search
**File**: `vector_store_image_search.py`

Search for relevant images using vector similarity.

#### Graphics Search
**File**: `graphics_search.py`

Search for graphics in Google Drive.

#### Quality Compliance Scoring
**File**: `quality_compliance_scoring.py`

Evaluate content quality and compliance.

#### Video Search Tool
**File**: `run_video_search_tool.py`

Search for relevant educational videos.

---

## Core Components

### 1. Agent UI Template
**File**: `agent_ui_template.py`

Provides a reusable UI framework for all agents with:
- Data loading interface
- Step-by-step pipeline execution
- Progress tracking
- Manual review checkpoints
- Automated step execution
- Background job launching
- Error handling
- Session state management

### 2. Authentication System

**OAuth 2.0 Flow**:
1. User clicks "Login with Google"
2. Redirected to Google OAuth consent screen
3. Authorization code exchanged for credentials
4. User info retrieved and role assigned
5. Session established with Drive and Sheets access

**Role-Based Access**:
- **Admin**: Full access to all features
- **Editor**: Access to all workflows
- **Content Head**: Course outline, research notes, quality scoring
- **Instructional Designer**: Research notes, slide chunks, graphics, assessments
- **Visual Designer**: Graphics definition, image search

### 3. Services Layer

#### Drive Service (`services/drive_service.py`)
- OAuth authentication
- File operations
- Folder management
- Service account sharing

#### Sheets Service (`services/sheets_service.py`)
- Read/write operations
- Worksheet management
- Data formatting
- Batch operations

#### LLM Service (`services/llm_service.py`)
- Model abstraction
- Multi-provider support (Gemini, OpenAI)
- Token tracking
- Error handling
- Retry logic

#### Web Search Service (`services/web_search.py`)
- Google search integration
- Result screening
- Content extraction

#### YouTube Service (`services/youtube_video_loader.py`)
- Video search
- Transcript retrieval
- Metadata extraction

### 4. Data Flow

```
Google Sheets (Input)
    ↓
Load Course Info
    ↓
Execute Pipeline Steps
    ↓
AI Processing (LLM)
    ↓
Manual Review (Optional)
    ↓
Write Results (Google Sheets)
```

### 5. Step Execution Pattern

Each step in a workflow follows this pattern:

```python
{
    "name": "Step Name",
    "func": execution_function,
    "depends_on": ["Previous Step"],
    "args": {
        "sheet": "sheet",
        "worksheet_name": "Sheet Name",
        "llm": "model_name"
    },
    "estimated_time": "~ X minutes",
    "description": "What this step does",
    "delete_func": cleanup_function,
    "delete_args": {...}
}
```

**Step Types**:
- **Automated**: Runs without human intervention
- **Manual**: Requires human input/review
- **Semi-automated**: AI generation with human review

---

## Setup & Installation

### Prerequisites
- Python 3.9+
- Google Cloud Project with OAuth 2.0 credentials
- Google Drive API enabled
- Google Sheets API enabled
- API keys for LLM providers (Google AI, OpenAI)

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/SkillCatApp/Content-Generation-Workflow.git
cd Content-Generation-Workflow
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Install system packages** (Linux)
```bash
sudo apt update
sudo xargs -a packages.txt apt install -y
```

4. **Install Playwright**
```bash
playwright install
```

5. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your credentials
```

Required environment variables:
```
OAUTH_CLIENT_ID=your_oauth_client_id
OAUTH_CLIENT_SECRET=your_oauth_client_secret
OAUTH_REDIRECT_URI=http://localhost:8501
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
GDRIVE_SA_B64=your_service_account_base64
LANGTRACE_API_KEY=your_langtrace_key (optional)
LANGCHAIN_API_KEY=your_langsmith_key (optional)
```

6. **Run the application**
```bash
streamlit run streamlit_app.py
```

### Lightning.ai Deployment

See the detailed guide in `README.md` for deploying on Lightning.ai.

---

## Usage Guide

### Starting a New Course

1. **Login**: Authenticate with your Google account
2. **Navigate to Template Sheet Setup**: Create initial sheet structure
3. **Load Course Data**:
   - Enter Google Drive folder ID
   - Enter Google Sheet link
   - Click "Load Data"
4. **Follow the Workflows in Order**:
   - Course Outline
   - Research Notes
   - Slide Chunks
   - Graphics Definition
   - Assessment

### Running an Agent

**UI Method**:
1. Navigate to the agent page
2. Load your course data
3. Click through each step or use "Run All Automated Steps"
4. Review manual checkpoints when prompted

**Background Method** (Admin only):
1. Click "Run the Agent in Background"
2. Monitor progress in terminal or Lightning.ai dashboard
3. Check "Agent logs" sheet for status

**CLI Method**:
```bash
python run_agent_cli.py \
  --sheet_link "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID" \
  --drive_folder_id "YOUR_FOLDER_ID" \
  --agent_name "course_outline"
```

### Manual Review Steps

When a manual step is reached:
1. Read the instructions displayed in the UI
2. Open the specified Google Sheet tab
3. Review AI-generated content
4. Make edits/provide feedback in designated columns
5. Return to the UI and click "Confirm" to proceed

### Deleting/Resetting Steps

Each step has a delete function that clears its outputs. To reset:
1. Find the step in the UI
2. Click the delete/reset button (if visible)
3. Re-run the step

---

## Development Guide

### Adding a New Agent

1. **Create the agent file**
```python
# agents/my_agent/my_agent.py
from agent_ui_template import agent_ui

def run_my_step(sheet, worksheet_name, llm="gemini_2_flash"):
    # Implementation
    pass

def delete_my_step(sheet, worksheet_name):
    # Cleanup
    pass

pipeline_sections = [
    {
        "section_name": "Section 1: My Section",
        "steps": [
            {
                "name": "My Step",
                "func": run_my_step,
                "depends_on": [],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "My Sheet",
                    "llm": "gemini_2_flash"
                },
                "estimated_time": "~ 2 minutes",
                "description": "Description of what this does",
                "delete_func": delete_my_step,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "My Sheet"
                }
            }
        ]
    }
]

agent_ui(step_name="My Agent", pipeline_sections=pipeline_sections)
```

2. **Create the page file**
```python
# my_agent_page.py
from agent_ui_template import agent_ui
from agents.my_agent.my_agent import pipeline_sections

agent_ui(step_name="My Agent", pipeline_sections=pipeline_sections)
```

3. **Register in `streamlit_app.py`**
```python
my_agent_page = st.Page(
    "my_agent_page.py",
    title="My Agent",
    icon=":material/icon:",
)

# Add to appropriate category in page_name_to_object
page_name_to_object["my_agent_page"] = my_agent_page

# Add to agent_pages or tool_pages list
agent_pages.append("my_agent_page")
```

4. **Update role configuration** (`config/user_roles.py`)
```python
ROLE_ACCESS = {
    "Admin": ["my_agent_page", ...],
    # Add to other roles as needed
}
```

### Best Practices

1. **Use the agent_ui_template**: Ensures consistency across agents
2. **Implement delete functions**: Allow users to reset steps
3. **Provide clear descriptions**: Help users understand each step
4. **Handle errors gracefully**: Use try-except blocks
5. **Track progress**: Use SmartProgressBar for long operations
6. **Log to sheets**: Update "Agent logs" sheet with status
7. **Use session state**: Store data between reruns
8. **Test manually**: Verify both automated and manual workflows

### LLM Integration

**Using the LLM Service**:
```python
from services.llm_service import get_llm

llm = get_llm("gemini_2_flash")
response = llm.invoke("Your prompt here")
```

**Available Models**:
- `gemini_2_flash`: Google Gemini 2 Flash
- `gemini_2_5_flash`: Google Gemini 2.5 Flash
- `gpt5_mini_thinking`: OpenAI GPT-5 Mini with reasoning
- `gemini_with_grounding`: Gemini with Google Search grounding

### Sheet Operations

**Reading data**:
```python
from services.sheets_service import get_sheet_data_and_df

worksheet, df = get_sheet_data_and_df(sheet, "Worksheet Name")
```

**Writing data**:
```python
from services.sheets_service import save_to_sheet

save_to_sheet(sheet, "Worksheet Name", data_list, start_row, start_col)
```

**Creating worksheets**:
```python
from services.sheets_service import create_or_read_worksheet

worksheet = create_or_read_worksheet(sheet, "New Sheet", rows=1000, cols=20)
```

---

## API & Services

### Services Overview

| Service | File | Purpose |
|---------|------|---------|
| Drive Service | `drive_service.py` | Google Drive integration |
| Sheets Service | `sheets_service.py` | Google Sheets CRUD operations |
| LLM Service | `llm_service.py` | Language model interface |
| Web Search | `web_search.py` | Web search and content extraction |
| YouTube | `youtube_video_loader.py` | YouTube video retrieval |
| Chunking | `chunking_service.py` | Text segmentation |
| Embedding | `embedding_service.py` | Vector embeddings |
| Web Loaders | `web_page_loaders.py` | Web content loading |

### Key Functions

#### Drive Service
```python
login_with_oauth2()  # OAuth authentication
share_sheet_with_service_account()  # Share with SA
get_service_account_email()  # Get SA email
```

#### Sheets Service
```python
get_sheet_data_and_df()  # Read sheet as DataFrame
save_to_sheet()  # Write data to sheet
create_or_read_worksheet()  # Create/get worksheet
format_worksheet()  # Apply formatting
```

#### LLM Service
```python
get_llm(model_name)  # Get LLM instance
```

---

## Agent System

### Agent Architecture

Each agent is structured as a pipeline of steps:

```
Pipeline
├── Section 1
│   ├── Step 1 (Automated)
│   ├── Step 2 (Manual)
│   └── Step 3 (Automated)
├── Section 2
│   └── ...
└── Section N
```

### Step Dependencies

Steps can depend on previous steps:
```python
{
    "name": "Step 2",
    "depends_on": ["Step 1"],
    # Only runs after Step 1 completes
}
```

### Manual Steps

Manual steps pause execution for human review:
```python
{
    "name": "Manual Review",
    "is_manual_step": True,
    "instructions": [
        "1. Open the sheet",
        "2. Review column X",
        "3. Provide feedback in column Y"
    ],
    "video_link": "https://..."
}
```

### Pre-Execution Functions

Steps can have setup functions that run before execution:
```python
{
    "name": "My Step",
    "pre_exec_func": setup_function,
    "pre_exec_args": {"sheet": "sheet"},
    "pre_exec_always_run": True
}
```

### Hiding Steps

Steps can be hidden based on conditions:
```python
{
    "name": "Optional Step",
    "hide_if_final_outline": True,
    # Hidden when outline is finalized
}
```

---

## Monitoring & Logging

### LangTrace Integration
- Tracks LLM calls
- Monitors token usage
- Logs to token_usage_log.csv

### LangSmith Integration
- Visualizes agent traces
- Debugs chain executions
- Monitors performance

### Agent Logs Sheet
- Records step completion
- Timestamps
- Status tracking
- Error logging

---

## Troubleshooting

### Common Issues

**Authentication Errors**:
- Verify OAuth credentials in .env
- Check redirect URI matches OAuth config
- Ensure APIs are enabled in Google Cloud Console

**Sheet Access Errors**:
- Verify sheet is shared with service account
- Check sheet link format
- Ensure worksheet names exist

**LLM Errors**:
- Check API key validity
- Verify model name spelling
- Check rate limits
- Review token usage

**Import Errors**:
- Run `pip install -r requirements.txt`
- Check Python version (3.9+)
- Verify playwright installation

---

## Contributing

### Development Workflow
1. Create a feature branch
2. Implement changes
3. Test thoroughly
4. Update documentation
5. Submit pull request

### Code Style
- Follow PEP 8
- Use type hints where possible
- Add docstrings to functions
- Keep functions focused and small

---

## License

[Specify your license here]

---

## Support

For issues or questions:
- GitHub Issues: https://github.com/SkillCatApp/Content-Generation-Workflow/issues
- Documentation: See README.md and about_agents.md

---

**Last Updated**: [Current Date]
**Version**: 1.0
