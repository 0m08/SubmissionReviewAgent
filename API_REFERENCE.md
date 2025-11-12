# API Reference - Content Generation Workflow

This document provides a comprehensive reference for all services, functions, and utilities in the Content Generation Workflow system.

---

## Table of Contents

1. [Services](#services)
   - [Drive Service](#drive-service)
   - [Sheets Service](#sheets-service)
   - [LLM Service](#llm-service)
   - [Web Search Service](#web-search-service)
   - [YouTube Service](#youtube-service)
   - [Chunking Service](#chunking-service)
   - [Embedding Service](#embedding-service)
2. [Utilities](#utilities)
   - [Role Utils](#role-utils)
   - [Decorator Helpers](#decorator-helpers)
   - [File Helpers](#file-helpers)
3. [Agent UI Template](#agent-ui-template)
4. [Helper Functions](#helper-functions)

---

## Services

### Drive Service

**File**: `services/drive_service.py`

#### Functions

##### `login_with_oauth2(client_id, client_secret, credentials_file="credentials.json")`

Authenticates using OAuth 2.0 for Google Drive access.

**Parameters**:
- `client_id` (str): OAuth client ID
- `client_secret` (str): OAuth client secret
- `credentials_file` (str): Path to store credentials

**Returns**: `GoogleAuth` object

**Example**:
```python
from services.drive_service import login_with_oauth2

gauth = login_with_oauth2(
    client_id="your_client_id",
    client_secret="your_client_secret"
)
```

---

##### `get_google_oauth_authorization_url(client_id, client_secret, redirect_uri)`

Generates the OAuth authorization URL.

**Parameters**:
- `client_id` (str): OAuth client ID
- `client_secret` (str): OAuth client secret
- `redirect_uri` (str): Redirect URI after authorization

**Returns**: Tuple of (auth_url: str, state: str)

---

##### `exchange_code_for_credentials(client_id, client_secret, redirect_uri, code)`

Exchanges authorization code for credentials.

**Parameters**:
- `client_id` (str): OAuth client ID
- `client_secret` (str): OAuth client secret
- `redirect_uri` (str): Redirect URI
- `code` (str): Authorization code

**Returns**: Credentials object

---

##### `init_clients_from_credentials(creds)`

Initializes Drive and Sheets clients from credentials.

**Parameters**:
- `creds`: Credentials object

**Returns**: Tuple of (gauth, drive, gc)

---

##### `share_sheet_with_service_account(sheet, service_account_email, user_credentials)`

Shares a Google Sheet with a service account.

**Parameters**:
- `sheet`: gspread Sheet object
- `service_account_email` (str): Service account email
- `user_credentials`: User's OAuth credentials

**Returns**: None

---

##### `get_service_account_email()`

Retrieves the service account email from environment variables.

**Returns**: str (service account email)

---

### Sheets Service

**File**: `services/sheets_service.py`

#### Functions

##### `get_sheet_data_and_df(sheet, worksheet_name)`

Reads data from a worksheet and returns it as a DataFrame.

**Parameters**:
- `sheet`: gspread Spreadsheet object
- `worksheet_name` (str): Name of the worksheet

**Returns**: Tuple of (worksheet, DataFrame)

**Example**:
```python
from services.sheets_service import get_sheet_data_and_df

worksheet, df = get_sheet_data_and_df(sheet, "Course info")
course_name = df.loc[0, "Course Name"]
```

---

##### `save_to_sheet(sheet, worksheet_name, data, start_row=1, start_col=1, create_if_not_exists=False)`

Writes data to a worksheet.

**Parameters**:
- `sheet`: gspread Spreadsheet object
- `worksheet_name` (str): Worksheet name
- `data` (list): 2D list of data
- `start_row` (int): Starting row (1-indexed)
- `start_col` (int): Starting column (1-indexed)
- `create_if_not_exists` (bool): Create worksheet if missing

**Returns**: None

**Example**:
```python
from services.sheets_service import save_to_sheet

data = [
    ["Name", "Age"],
    ["Alice", 30],
    ["Bob", 25]
]
save_to_sheet(sheet, "Users", data, start_row=1, start_col=1)
```

---

##### `create_or_read_worksheet(sheet, worksheet_name, rows=1000, cols=26)`

Creates a worksheet if it doesn't exist, or returns existing one.

**Parameters**:
- `sheet`: gspread Spreadsheet object
- `worksheet_name` (str): Worksheet name
- `rows` (int): Number of rows
- `cols` (int): Number of columns

**Returns**: gspread Worksheet object

---

##### `format_worksheet(worksheet, header_format=None, freeze_rows=1)`

Applies formatting to a worksheet.

**Parameters**:
- `worksheet`: gspread Worksheet object
- `header_format` (dict): Format for header row
- `freeze_rows` (int): Number of rows to freeze

**Returns**: None

**Example**:
```python
from services.sheets_service import format_worksheet

header_format = {
    "backgroundColor": {"red": 0.0, "green": 0.5, "blue": 0.0},
    "textFormat": {"bold": True, "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}}
}
format_worksheet(worksheet, header_format, freeze_rows=1)
```

---

##### `batch_update_cells(worksheet, updates)`

Batch update multiple cells efficiently.

**Parameters**:
- `worksheet`: gspread Worksheet object
- `updates` (list): List of dicts with keys 'row', 'col', 'value'

**Returns**: None

**Example**:
```python
updates = [
    {"row": 1, "col": 1, "value": "Header 1"},
    {"row": 1, "col": 2, "value": "Header 2"},
    {"row": 2, "col": 1, "value": "Data 1"}
]
batch_update_cells(worksheet, updates)
```

---

##### `get_worksheet_names(sheet)`

Gets list of all worksheet names in a spreadsheet.

**Parameters**:
- `sheet`: gspread Spreadsheet object

**Returns**: List of worksheet names (str)

---

### LLM Service

**File**: `services/llm_service.py`

#### Functions

##### `get_llm(model_name="gemini_2_flash", temperature=0.2, **kwargs)`

Returns a configured LLM instance.

**Parameters**:
- `model_name` (str): Name of the model
- `temperature` (float): Generation temperature (0.0-1.0)
- `**kwargs`: Additional model-specific parameters

**Returns**: LLM instance (ChatGoogleGenerativeAI or ChatOpenAI)

**Supported Models**:
- `gemini_2_flash`: Google Gemini 2.0 Flash
- `gemini_2_5_flash`: Google Gemini 2.5 Flash
- `gemini_with_grounding`: Gemini with Google Search grounding
- `gpt4_1`: OpenAI GPT-4
- `gpt5_mini_thinking`: OpenAI GPT-5 Mini with reasoning

**Example**:
```python
from services.llm_service import get_llm

llm = get_llm("gemini_2_flash", temperature=0.3)
response = llm.invoke("What is HVAC?")
print(response.content)
```

---

##### `track_token_usage(model_name, input_tokens, output_tokens, cost=0.0)`

Logs token usage to CSV file.

**Parameters**:
- `model_name` (str): Name of the model
- `input_tokens` (int): Number of input tokens
- `output_tokens` (int): Number of output tokens
- `cost` (float): Estimated cost

**Returns**: None

---

### Web Search Service

**File**: `services/web_search.py`

#### Functions

##### `search_web(query, num_results=10)`

Performs a Google search.

**Parameters**:
- `query` (str): Search query
- `num_results` (int): Number of results to return

**Returns**: List of dicts with keys 'title', 'link', 'snippet'

**Example**:
```python
from services.web_search import search_web

results = search_web("HVAC maintenance best practices", num_results=5)
for result in results:
    print(f"{result['title']}: {result['link']}")
```

---

##### `screen_search_results(results, relevance_threshold=0.7)`

Filters search results based on relevance.

**Parameters**:
- `results` (list): List of search results
- `relevance_threshold` (float): Minimum relevance score

**Returns**: List of filtered results

---

##### `extract_article_content(url)`

Extracts main content from a web page.

**Parameters**:
- `url` (str): URL of the article

**Returns**: Dict with keys 'title', 'content', 'author', 'date'

**Example**:
```python
from services.web_search import extract_article_content

content = extract_article_content("https://example.com/article")
print(content['title'])
print(content['content'][:500])
```

---

### YouTube Service

**File**: `services/youtube_video_loader.py`

#### Functions

##### `search_youtube(query, max_results=10)`

Searches for YouTube videos.

**Parameters**:
- `query` (str): Search query
- `max_results` (int): Maximum number of results

**Returns**: List of dicts with video metadata

**Example**:
```python
from services.youtube_video_loader import search_youtube

videos = search_youtube("HVAC troubleshooting", max_results=5)
for video in videos:
    print(f"{video['title']}: {video['url']}")
```

---

##### `get_video_transcript(video_id, languages=['en'])`

Retrieves transcript for a YouTube video.

**Parameters**:
- `video_id` (str): YouTube video ID
- `languages` (list): Preferred languages

**Returns**: str (full transcript)

**Example**:
```python
from services.youtube_video_loader import get_video_transcript

transcript = get_video_transcript("dQw4w9WgXcQ")
print(transcript[:500])
```

---

##### `get_video_metadata(video_id)`

Gets metadata for a YouTube video.

**Parameters**:
- `video_id` (str): YouTube video ID

**Returns**: Dict with keys 'title', 'description', 'duration', 'views', etc.

---

### Chunking Service

**File**: `services/chunking_service.py`

#### Functions

##### `chunk_text(text, chunk_size=1000, chunk_overlap=200, strategy="recursive")`

Chunks text into smaller segments.

**Parameters**:
- `text` (str): Text to chunk
- `chunk_size` (int): Target chunk size in characters
- `chunk_overlap` (int): Overlap between chunks
- `strategy` (str): Chunking strategy ("recursive", "fixed", "semantic")

**Returns**: List of text chunks (str)

**Example**:
```python
from services.chunking_service import chunk_text

text = "Long text content..."
chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk[:100]}...")
```

---

##### `chunk_by_sentences(text, sentences_per_chunk=5)`

Chunks text by sentences.

**Parameters**:
- `text` (str): Text to chunk
- `sentences_per_chunk` (int): Sentences per chunk

**Returns**: List of text chunks (str)

---

##### `semantic_chunking(text, similarity_threshold=0.75)`

Chunks text based on semantic similarity.

**Parameters**:
- `text` (str): Text to chunk
- `similarity_threshold` (float): Minimum similarity to group sentences

**Returns**: List of text chunks (str)

---

### Embedding Service

**File**: `services/embedding_service.py`

#### Functions

##### `get_embeddings(texts, model="text-embedding-004")`

Generates embeddings for texts.

**Parameters**:
- `texts` (list): List of texts to embed
- `model` (str): Embedding model name

**Returns**: List of embedding vectors

**Example**:
```python
from services.embedding_service import get_embeddings

texts = ["HVAC basics", "Air conditioning repair"]
embeddings = get_embeddings(texts)
print(f"Embedding dimension: {len(embeddings[0])}")
```

---

## Utilities

### Role Utils

**File**: `utils/role_utils.py`

#### Functions

##### `get_user_info(email)`

Gets user information and permissions.

**Parameters**:
- `email` (str): User's email address

**Returns**: Dict with keys 'is_authorized', 'role', 'pages'

**Example**:
```python
from utils.role_utils import get_user_info

user_info = get_user_info("user@example.com")
if user_info["is_authorized"]:
    print(f"User role: {user_info['role']}")
    print(f"Accessible pages: {user_info['pages']}")
```

---

##### `get_user_pages(role)`

Gets list of accessible pages for a role.

**Parameters**:
- `role` (str): User role

**Returns**: List of page names (str)

---

##### `check_access(email, page_name)`

Checks if user has access to a page.

**Parameters**:
- `email` (str): User's email
- `page_name` (str): Page to check

**Returns**: bool

---

### Decorator Helpers

**File**: `utils/decorator_helpers.py`

#### Decorators

##### `@retry(max_attempts=3, delay=1, backoff=2)`

Retries a function on failure.

**Parameters**:
- `max_attempts` (int): Maximum retry attempts
- `delay` (float): Initial delay in seconds
- `backoff` (float): Backoff multiplier

**Example**:
```python
from utils.decorator_helpers import retry

@retry(max_attempts=3, delay=2)
def fetch_data():
    # Code that might fail
    pass
```

---

##### `@log_execution(logger=None)`

Logs function execution.

**Parameters**:
- `logger`: Logger instance (optional)

**Example**:
```python
from utils.decorator_helpers import log_execution

@log_execution()
def process_data(data):
    # Process data
    pass
```

---

##### `@measure_time`

Measures function execution time.

**Example**:
```python
from utils.decorator_helpers import measure_time

@measure_time
def slow_function():
    # Long-running code
    pass
```

---

### File Helpers

**File**: `utils/file_helpers.py`

#### Functions

##### `read_file(file_path, encoding='utf-8')`

Reads a file and returns content.

**Parameters**:
- `file_path` (str): Path to file
- `encoding` (str): File encoding

**Returns**: str (file content)

---

##### `write_file(file_path, content, encoding='utf-8')`

Writes content to a file.

**Parameters**:
- `file_path` (str): Path to file
- `content` (str): Content to write
- `encoding` (str): File encoding

**Returns**: None

---

##### `ensure_dir(directory)`

Creates directory if it doesn't exist.

**Parameters**:
- `directory` (str): Directory path

**Returns**: None

---

## Agent UI Template

**File**: `agent_ui_template.py`

### Main Function

##### `agent_ui(step_name, pipeline_sections, outline_finalized=False)`

Creates the UI for an agent workflow.

**Parameters**:
- `step_name` (str): Name of the agent (e.g., "Course Outline")
- `pipeline_sections` (list): List of pipeline section dicts
- `outline_finalized` (bool): Whether outline is finalized

**Pipeline Section Structure**:
```python
{
    "section_name": "Section 1: My Section",
    "steps": [
        {
            "name": "Step Name",
            "func": step_function,
            "depends_on": ["Previous Step"],
            "args": {
                "sheet": "sheet",
                "worksheet_name": "Sheet Name",
                "llm": "gemini_2_flash"
            },
            "estimated_time": "~ 5 minutes",
            "description": "What this step does",
            "delete_func": delete_function,
            "delete_args": {...},
            "is_manual_step": False,
            "instructions": [],
            "video_link": "",
            "pre_exec_func": None,
            "pre_exec_args": {},
            "pre_exec_always_run": False,
            "hide_if_final_outline": False
        }
    ]
}
```

**Example**:
```python
from agent_ui_template import agent_ui

pipeline_sections = [
    {
        "section_name": "Section 1: Generation",
        "steps": [
            {
                "name": "Generate Content",
                "func": generate_content,
                "depends_on": [],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Output",
                    "llm": "gemini_2_flash"
                },
                "estimated_time": "~ 2 minutes",
                "description": "Generates content using AI",
                "delete_func": delete_content,
                "delete_args": {"sheet": "sheet", "worksheet_name": "Output"}
            }
        ]
    }
]

agent_ui(step_name="My Agent", pipeline_sections=pipeline_sections)
```

---

## Helper Functions

**File**: `services/helper_functions.py`

### Key Functions

##### `get_short_name(course_name)`

Generates a short name for a course.

**Parameters**:
- `course_name` (str): Full course name

**Returns**: str (short name)

---

##### `create_final_outline_sheet(sheet)`

Creates the Final Outline sheet by flattening LOs.

**Parameters**:
- `sheet`: gspread Spreadsheet object

**Returns**: None

---

##### `delete_final_outline(sheet, worksheet_name)`

Deletes the Final Outline sheet.

**Parameters**:
- `sheet`: gspread Spreadsheet object
- `worksheet_name` (str): Sheet name

**Returns**: None

---

##### `log_step_completion(sheet, agent_name, step_name, status="completed")`

Logs step completion to Agent logs sheet.

**Parameters**:
- `sheet`: gspread Spreadsheet object
- `agent_name` (str): Name of agent
- `step_name` (str): Name of step
- `status` (str): Status ("completed", "failed", "running")

**Returns**: None

---

##### `get_completed_steps(sheet, agent_name)`

Gets list of completed steps for an agent.

**Parameters**:
- `sheet`: gspread Spreadsheet object
- `agent_name` (str): Name of agent

**Returns**: List of step names (str)

---

## Example: Complete Workflow

Here's a complete example of creating a custom agent:

```python
# my_custom_agent.py
from agent_ui_template import agent_ui
from services.sheets_service import get_sheet_data_and_df, save_to_sheet, create_or_read_worksheet
from services.llm_service import get_llm

def generate_summaries(sheet, worksheet_name, llm="gemini_2_flash"):
    """Generate summaries for content."""
    # Read data
    worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)

    # Get LLM
    model = get_llm(llm)

    # Process each row
    summaries = []
    for idx, row in df.iterrows():
        content = row['content']
        prompt = f"Summarize this content:\n\n{content}"
        summary = model.invoke(prompt).content
        summaries.append([summary])

    # Write back
    save_to_sheet(sheet, worksheet_name, summaries, start_row=2, start_col=3)

def delete_summaries(sheet, worksheet_name):
    """Clear summaries."""
    worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)
    empty_data = [[""]] * len(df)
    save_to_sheet(sheet, worksheet_name, empty_data, start_row=2, start_col=3)

# Define pipeline
pipeline_sections = [
    {
        "section_name": "Section 1: Summary Generation",
        "steps": [
            {
                "name": "Generate Summaries",
                "func": generate_summaries,
                "depends_on": [],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Content",
                    "llm": "gemini_2_flash"
                },
                "estimated_time": "~ 5 minutes",
                "description": "Generates AI summaries for all content",
                "delete_func": delete_summaries,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Content"
                }
            }
        ]
    }
]

# Create UI
agent_ui(step_name="Summary Agent", pipeline_sections=pipeline_sections)
```

---

## Error Handling

### Common Patterns

**Handling API Errors**:
```python
from utils.decorator_helpers import retry

@retry(max_attempts=3, delay=2)
def call_api():
    try:
        # API call
        pass
    except Exception as e:
        print(f"Error: {e}")
        raise
```

**Handling Sheet Errors**:
```python
try:
    worksheet, df = get_sheet_data_and_df(sheet, "Sheet Name")
except gspread.exceptions.WorksheetNotFound:
    # Create worksheet
    worksheet = create_or_read_worksheet(sheet, "Sheet Name")
```

**Handling LLM Errors**:
```python
try:
    llm = get_llm("gemini_2_flash")
    response = llm.invoke(prompt)
except Exception as e:
    # Fallback to different model
    llm = get_llm("gpt4_1")
    response = llm.invoke(prompt)
```

---

## Performance Optimization

### Batch Operations

**Batch Sheet Updates**:
```python
# Instead of multiple updates
for i in range(100):
    worksheet.update_cell(i+1, 1, value)

# Use batch update
updates = [{"row": i+1, "col": 1, "value": value} for i in range(100)]
batch_update_cells(worksheet, updates)
```

**Batch LLM Calls**:
```python
# Instead of sequential calls
for text in texts:
    response = llm.invoke(text)

# Use batch processing
responses = llm.batch(texts)
```

---

## Testing

### Unit Testing Example

```python
import unittest
from services.sheets_service import get_sheet_data_and_df

class TestSheetsService(unittest.TestCase):
    def setUp(self):
        # Setup test sheet
        pass

    def test_get_sheet_data(self):
        worksheet, df = get_sheet_data_and_df(self.sheet, "Test")
        self.assertIsNotNone(df)
        self.assertEqual(len(df), expected_rows)

    def tearDown(self):
        # Cleanup
        pass
```

---

This API reference covers the core functions and services. For more details, refer to the source code and inline documentation.
