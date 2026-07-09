from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
try:
    from langchain_community.chat_models import ChatPerplexity
except ImportError:
    try:
        from langchain_perplexity import ChatPerplexity
    except ImportError:
        from langchain_core.language_models.chat_models import SimpleChatModel
        from langchain_core.messages import BaseMessage
        from typing import List as _List, Optional as _Optional, Any as _Any

        class ChatPerplexity(SimpleChatModel):  # type: ignore
            """
            Fallback when no Perplexity integration is installed.
            """

            model: str = "sonar-deep-research"

            def __init__(self, *args, **kwargs):
                super().__init__()

            @property
            def _llm_type(self) -> str:
                return "perplexity-unavailable"

            def _call(self, messages: "_List[BaseMessage]", stop: "_Optional[_List[str]]" = None, run_manager: "_Optional[_Any]" = None, **kwargs: "_Any") -> str:
                raise ImportError(
                    "ChatPerplexity is unavailable."
                )
from langchain_core.runnables import ConfigurableField
# Output Parsers
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser
import time
import csv
from datetime import datetime
import os
import threading
from google import genai
import json
import streamlit as st
import requests
from google.genai import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.decorator_helpers import try_n_times
from langsmith import traceable

_TOKEN_LOG_LOCK = threading.Lock()


def _to_int(value) -> int:
    try:
        if value is None:
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _meta_get(meta, *keys):
    if meta is None:
        return None
    for key in keys:
        if isinstance(meta, dict):
            if key in meta and meta[key] is not None:
                return meta[key]
        else:
            value = getattr(meta, key, None)
            if value is not None:
                return value
    return None


def extract_token_usage(response_or_result) -> dict:
    """
    Extract billable token counts from either direct SDK responses or LangChain messages.

    output_tokens intentionally includes Gemini thought/reasoning tokens so UI pricing
    matches billable output for thinking-enabled models.
    """
    meta = getattr(response_or_result, "usage_metadata", None)
    if meta is None and isinstance(response_or_result, dict):
        meta = response_or_result.get("usage_metadata")

    input_tokens = _to_int(
        _meta_get(meta, "prompt_token_count", "input_tokens", "prompt_tokens", "input_token_count")
    )
    candidate_output_tokens = _to_int(
        _meta_get(meta, "candidates_token_count", "output_tokens", "completion_tokens", "output_token_count")
    )
    thought_tokens = _to_int(
        _meta_get(meta, "thoughts_token_count", "reasoning_tokens", "thought_token_count")
    )

    if thought_tokens == 0:
        output_details = _meta_get(meta, "output_token_details", "outputTokenDetails")
        thought_tokens = _to_int(
            _meta_get(output_details, "reasoning", "thoughts", "thinking", "reasoning_tokens")
        )

    total_tokens = _to_int(
        _meta_get(meta, "total_token_count", "total_tokens", "totalTokenCount")
    )

    # Avoid double counting when SDK-normalized output_tokens already includes reasoning/thought tokens.
    output_tokens = candidate_output_tokens
    if thought_tokens > 0:
        if total_tokens > 0:
            expected_without_thoughts = input_tokens + candidate_output_tokens
            expected_with_thoughts = expected_without_thoughts + thought_tokens

            if total_tokens == expected_with_thoughts:
                output_tokens = candidate_output_tokens + thought_tokens
            elif total_tokens == expected_without_thoughts:
                output_tokens = candidate_output_tokens
            else:
                inferred_output = max(total_tokens - input_tokens, 0)
                output_tokens = max(candidate_output_tokens, inferred_output)
        else:
            output_tokens = candidate_output_tokens + thought_tokens
    elif total_tokens > 0 and candidate_output_tokens == 0:
        output_tokens = max(total_tokens - input_tokens, 0)

    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "candidate_output_tokens": candidate_output_tokens,
        "thought_tokens": thought_tokens,
        "total_tokens": total_tokens,
    }

def log_token_usage(llm, input_tokens, output_tokens, log_file="token_usage_log.csv", agent_name=None, step_name=None):
    """Appends token usage data to a CSV file."""
    # Try to get agent_name and step_name from parameters first, then fall back to session_state
    try:
        if agent_name is None:
            agent_name = st.session_state.get("agent_name", "")
        if step_name is None:
            step_name = st.session_state.get("current_step", "")
    except (RuntimeError, AttributeError):
        # Handle case where Streamlit session_state is not available
        agent_name = agent_name or ""
        step_name = step_name or ""
    if not agent_name:
        agent_name = os.environ.get("CURRENT_AGENT_NAME", agent_name or "")
    if not step_name:
        step_name = os.environ.get("CURRENT_STEP_NAME", step_name or "")

    # Check if the log file already exists to decide if we need a header row.
    with _TOKEN_LOG_LOCK:
        file_exists = os.path.isfile(log_file)
        with open(log_file, mode="a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["agent_name", "step_name", "timestamp", "llm", "input_tokens", "output_tokens"])
            writer.writerow([agent_name, step_name, datetime.now().isoformat(), llm, input_tokens, output_tokens])


@traceable
@try_n_times(n=5, wait=2, backoff="exponential")
def google_search_with_grounding(prompt, model="gemini-3-flash-preview"):
    """
    Makes an API call to Gemini with search grounding, then attempts to resolve each returned URL in parallel.
    :param prompt: str
    :param model: str
    :return: (response, list_of_uris)
    """
    # Capture session state values before the LLM call
    try:
        current_agent_name = st.session_state.get("agent_name", "")
        current_step_name = st.session_state.get("current_step", "")
    except (RuntimeError, AttributeError):
        current_agent_name = ""
        current_step_name = ""

    client = genai.Client()

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(
                google_search=types.GoogleSearchRetrieval
            )]
        )
    )

    # Log token usage ── pick counts safely, fall back to 0
    token_usage = extract_token_usage(response)
    input_tokens = token_usage["input_tokens"]
    output_tokens = token_usage["output_tokens"]

    try:
        log_token_usage(
            llm=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            log_file="token_usage_log.csv",
            agent_name=current_agent_name,
            step_name=current_step_name,
        )
    except Exception as e:
        print(f"Token usage logging failed: {e}")

    # --- Function to fetch the final (redirected) URL for a single link --- #
    def fetch_final_url(url):
        try:
            r = requests.head(url, allow_redirects=True, timeout=10)
            return r.url
        except requests.exceptions.Timeout:
            print(f"Timeout occurred for URL: {url}")
        except requests.exceptions.RequestException as e:
            print(f"Request error for URL {url}: {e}")
        return None

    def get_uris(response_obj):
        urls = []
        for candidate in getattr(response_obj, "candidates", []) or []:
            grounding_meta = getattr(candidate, "grounding_metadata", None)
            if grounding_meta is None:
                continue

            for chunk in getattr(grounding_meta, "grounding_chunks", []) or []:
                uri = getattr(getattr(chunk, "web", None), "uri", None)
                if uri:
                    urls.append(uri)

        # (optional) keep only first occurrence of each URL
        urls = list(dict.fromkeys(urls))

        # Run requests in parallel
        valid_uris = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Dictionary of future -> original_url
            future_to_url = {executor.submit(fetch_final_url, url): url for url in urls}

            # Collect results as they complete
            for future in as_completed(future_to_url):
                original_url = future_to_url[future]
                try:
                    final_url = future.result()
                    # Only add if we got a valid response
                    if final_url is not None:
                        valid_uris.append(final_url)
                except Exception as exc:
                    # Catch any unexpected exceptions from future
                    print(f"URL {original_url} generated an exception: {exc}")

        return valid_uris

    sources = get_uris(response)
    return response.text, sources


# Function to generate structured output using direct provider APIs
def generate_structured_output(prompt, structured_output, model="gemini-3-flash-preview", temperature=0.7, max_tokens=8192, thinking_level=None):
    """
    Generate structured output using Gemini API directly.

    Args:
        prompt: The input prompt to the model
        structured_output: A Pydantic BaseModel class for structured output
        model: The model to use
        temperature: Temperature for generation
        max_tokens: Maximum tokens for generation
        thinking_level: Thinking level for Gemini 3 models ("minimal", "low", "medium", "high")

    Returns:
        The structured output as a Pydantic model instance or list of instances
    """
    try:
        # Capture session state values before the LLM call
        try:
            current_agent_name = st.session_state.get("agent_name", "")
            current_step_name = st.session_state.get("current_step", "")
        except (RuntimeError, AttributeError):
            current_agent_name = ""
            current_step_name = ""

        client = genai.Client()

        print(type(prompt))
        print(prompt)

        config = {
            'response_mime_type': 'application/json',
            'response_schema': structured_output,
            'temperature': temperature,
            'max_output_tokens': max_tokens,
        }
        
        # Add thinking_level for Gemini 3 models
        if thinking_level and "gemini-3" in model:
            config['thinking_level'] = thinking_level
        
        response = client.models.generate_content(
            model=model,
            contents=str(prompt),
            config=config,
        )

        # Log token usage ── pick counts safely, fall back to 0
        token_usage = extract_token_usage(response)
        input_tokens = token_usage["input_tokens"]
        output_tokens = token_usage["output_tokens"]

        try:
            log_token_usage(
                llm=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                log_file="token_usage_log.csv",
                agent_name=current_agent_name,
                step_name=current_step_name,
            )
        except Exception as e:
            print(f"Token usage logging failed: {e}")

        # Get the JSON response text
        json_response = response.candidates[0].content.parts[0].text
        
        # Parse the JSON into the appropriate Pydantic model
        try:
            # If structured_output is a list type (like list[Topic])
            if hasattr(structured_output, "__origin__") and structured_output.__origin__ is list:
                # Get the item type from the list
                item_type = structured_output.__args__[0]
                json_data = json.loads(json_response)
                # Return a list of parsed model instances
                return [item_type.model_validate(item) for item in json_data]
            else:
                # For single object types
                json_data = json.loads(json_response)
                return structured_output.model_validate(json_data)
        except Exception as e:
            print(f"Failed to parse JSON response into Pydantic model: {e}")
            # If parsing fails, return the raw JSON response
            return json_response
        
    except Exception as e:
        print(f"Error generating structured output: {e}")
        raise e


def call_llm_with_retry(func, *args, max_retries=3, initial_wait=2, **kwargs):
    """
    Retry an LLM call with exponential backoff.
    Works with google.genai client methods and other direct API calls.
    """
    import time
    last_exc = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if attempt < max_retries - 1:
                wait_time = initial_wait * (2 ** attempt)
                print(f"  ⚠️ LLM call failed: {e}. Retrying in {wait_time}s... ({attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"  ❌ LLM call failed after {max_retries} attempts: {e}")
    raise last_exc


# Function to make llm calls
def llm_with_retry(arg, max_retries = 15, structured_output = None, llm_name = None):
    """
    Call an LLM with optional structured output and retry logic.
    Args:
        arg: The input to the LLM.
        max_retries (int): The maximum number of retries in case of failure.
        structured_output (Optional): A Pydantic BaseModel for structured output.
        llm_name (Optional[str]): The name of the LLM to use.
    """
    # Capture session state values at the start of the function
    try:
        current_agent_name = st.session_state.get("agent_name", "")
        current_step_name = st.session_state.get("current_step", "")
    except (RuntimeError, AttributeError):
        current_agent_name = ""
        current_step_name = ""

    # List of models that support direct API structured output
    direct_api_models = ["gemini_2_flash", "gemini_2_flash_thinking", "gemini_flash", "gemini_2_5_flash", "gemini_2_5_flash_lite", "gemini_3_pro", "gemini_3_flash", "gemini_3_flash_thinking"]
    
    # Use direct API for Gemini models with structured output
    if structured_output and (llm_name in direct_api_models or (llm_name is None and structured_output)):
        retries = 0
        while retries < max_retries:
            try:
                # Map LLM name to actual model name for the API
                model_mapping = {
                    "gemini_2_5_flash": "gemini-2.5-flash",
                    "gemini_2_5_flash_lite": "gemini-2.5-flash-lite",
                    "gemini_2_flash": "gemini-2.0-flash",
                    "gemini_2_flash_thinking": "gemini-2.0-flash-thinking",
                    "gemini_flash": "gemini-1.5-flash-latest",
                    "gemini_3_pro": "gemini-3-pro-preview",
                    "gemini_3_flash": "gemini-3-flash-preview",
                    "gemini_3_flash_thinking": "gemini-3-flash-preview",
                    None: "gemini-3-flash-preview"  # Default if no name provided
                }
                
                # Map thinking levels for Gemini 3 models
                thinking_level_mapping = {
                    "gemini_3_flash_thinking": "high",
                    "gemini_3_flash": None,  # Default thinking level
                }
                
                model = model_mapping.get(llm_name, "gemini-3-flash-preview")
                thinking_level = thinking_level_mapping.get(llm_name)
                
                return generate_structured_output(
                    prompt=arg,
                    structured_output=structured_output,
                    model=model,
                    thinking_level=thinking_level
                )
            except KeyboardInterrupt:
                print('Keyboard interrupt')
                raise Exception("Keyboard interrupt")
            except Exception as e:
                print(e)
                wait_time = retries * 5 + 1
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
        else:
            print("Max retries exceeded.")
            return None
    
    # Use LangChain for non-Gemini models or when structured output is not needed
    llm = ChatGoogleGenerativeAI(
        model = "gemini-3-flash-preview",
        temperature = 0.7,
        max_tokens = 8192
        ).configurable_alternatives(
            ConfigurableField(id="llm"),
            default_key = "gemini_2_flash_default",
            gpt4o_mini = ChatOpenAI(model_name = "gpt-4o-mini", temperature = 0.7, max_tokens = 4096),
            gpt4o = ChatOpenAI(model_name = "gpt-4o", temperature = 0.7, max_tokens = 4096),
            gpt4_1 = ChatOpenAI(model_name = "gpt-4.1", temperature = 0.7),
            haiku = ChatAnthropic(model_name = "claude-3-haiku-20240307", temperature = 0.7, max_tokens = 4096),
            sonnet = ChatAnthropic(model_name = "claude-3-5-sonnet-20240620", temperature = 0.7, max_tokens = 4096),
            haiku_3_5 = ChatAnthropic(model_name = "claude-3-5-haiku-20241022", temperature = 0.7, max_tokens = 4096),
            sonnet_4_thinking = ChatAnthropic(model_name="claude-sonnet-4-20250514", temperature=0.7, max_tokens=64000, model_kwargs={"thinking": {"type": "enabled", "budget_tokens": 30000}}),
            #sonnet_4_thinking = ChatAnthropic(model_name="claude-sonnet-4-20250514", temperature=0.7, max_tokens=64000),
            gemini_flash = ChatGoogleGenerativeAI(model = "gemini-1.5-flash-latest", temperature = 0.7, max_tokens = 8192),
            llama_3_1_70b = ChatGroq(model_name = 'llama-3.1-70b-versatile', temperature = 0.7, max_tokens = 4096),
            groq = ChatGroq(model_name = 'llama3-70b-8192', temperature = 0.7, max_tokens = 4096),
            gemini_2_flash = ChatGoogleGenerativeAI(model = "gemini-2.0-flash", temperature = 0.7, max_tokens = 640000),
            o1 = ChatOpenAI(model = 'o1', temperature=1),
            o3_mini = ChatOpenAI(model = 'o3-mini', temperature=1),
            gemini_2_flash_thinking = ChatGoogleGenerativeAI(model = "gemini-2.0-flash-thinking-exp", temperature = 0.7, max_tokens = 640000),
            pplx_deep_research = ChatPerplexity(model = "sonar-deep-research", temperature = 0, pplx_api_key = os.environ.get("PPLX_API_KEY")),
            gemini_2_5_flash = ChatGoogleGenerativeAI(model = "gemini-2.5-flash", temperature = 0.7, max_tokens = 640000),
            gemini_2_5_flash_lite = ChatGoogleGenerativeAI(model = "gemini-2.5-flash-lite", temperature = 0.7, max_tokens = 640000),
            gemini_2_5_pro = ChatGoogleGenerativeAI(model = "gemini-2.5-pro", temperature = 0.7, max_tokens = 640000),
            gemini_3_pro = ChatGoogleGenerativeAI(model = "gemini-3-pro-preview", temperature = 1.0, max_tokens = 640000),
            gemini_3_flash = ChatGoogleGenerativeAI(model = "gemini-3-flash-preview", temperature = 0.7, max_tokens = 640000),
            gemini_3_flash_thinking = ChatGoogleGenerativeAI(model = "gemini-3-flash-preview", temperature = 0.7, max_tokens = 640000, model_kwargs={"thinking_level": "high"}),
            gpt5_thinking = ChatOpenAI(model_name = "gpt-5", max_tokens = 127000, reasoning_effort="high", temperature=1),
            gpt5_mini_thinking = ChatOpenAI(model_name = "gpt-5-mini", max_tokens = 127000, temperature=1), # or "minimal", "low", "medium", "high"
            # gpt5 = ChatOpenAI(model_name = "gpt-5", temperature = 0.7, max_tokens = 8192),
            # gemini_2_flash_open_router = ChatOpenAI(model = 'google/gemini-2.0-flash-exp:free', temperature = 0.7, max_completion_tokens = 8192, base_url = 'https://openrouter.ai/api/v1', api_key = os.environ.get('OPENROUTER_API_KEY'))
            )

    # Optionally add structured output for LangChain method
    if structured_output:
        llm = llm.with_structured_output(structured_output)

    configurable_llm_keys = {
        "gpt4o_mini",
        "gpt4o",
        "gpt4_1",
        "haiku",
        "sonnet",
        "haiku_3_5",
        "sonnet_4_thinking",
        "gemini_flash",
        "llama_3_1_70b",
        "groq",
        "gemini_2_flash",
        "o1",
        "o3_mini",
        "gemini_2_flash_thinking",
        "pplx_deep_research",
        "gemini_2_5_flash",
        "gemini_2_5_flash_lite",
        "gemini_2_5_pro",
        "gemini_3_pro",
        "gemini_3_flash",
        "gemini_3_flash_thinking",
        "gpt5_thinking",
        "gpt5_mini_thinking",
    }

    llm_key = llm_name or "gemini_2_flash_default"
    if llm_name and llm_name in configurable_llm_keys:
        configured_llm = llm.with_config(configurable={"llm": llm_name})
    else:
        configured_llm = llm

    retries = 0                  # Initialize the retry counter
    while retries < max_retries:
        try:
            if retries > 3:             # In such a case, try using gemini flash
                result = llm.with_config(
                    configurable={"llm": 'gemini_2_flash'}
                    ).invoke(arg)
                used_llm_key = "gemini_2_flash"
            else:
                result = configured_llm.invoke(arg)
                used_llm_key = llm_key

            # Log structured token usage — robust + cleaner
            token_usage = extract_token_usage(result)
            input_tokens = token_usage["input_tokens"]
            output_tokens = token_usage["output_tokens"]

            try:
                log_token_usage(
                    llm=used_llm_key,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    log_file="token_usage_log.csv",
                    agent_name=current_agent_name,
                    step_name=current_step_name,
                )
            except Exception as e:
                print(f"LLM usage could not be logged. Error: {e}")

            return result  # Return the successful API response
        except KeyboardInterrupt:
            print('Keyboard interrupt')
            raise Exception("Keyboard interrupt")
        except Exception as e:
            print(e)
            print(retries*5 + 1)
            time.sleep(retries*5 + 1)  # Sleep for the required wait time before retrying
            retries += 1
    else:
        print("Max retries exceeded.")
        return e


# Output parser to get string output
output_parser = StrOutputParser()


# Intialize csv list parser that comes out of the box from langchain as an output parser
csv_list_parser = CommaSeparatedListOutputParser()


# Define a custom output parser
def extract_csv_lines(text: str):
    """
    Function to extract csv lines from text
    :param text: text to parse
    :return: list of lines
    """
    # Try with simple line split, since csv splitter doesn't handle commas in between search query
    try:
        lines = text.strip().split('\n')
        return [line.strip(',').strip() for line in lines]
    except KeyboardInterrupt:
        print('User stopped action')
    except:
        return csv_list_parser.parse(text)
