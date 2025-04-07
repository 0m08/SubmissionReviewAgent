from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatPerplexity
from langchain_core.runnables import ConfigurableField
# Output Parsers
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser
import time
import csv
from datetime import datetime
import os


def log_token_usage(llm, input_tokens, output_tokens, log_file="content/token_usage_log.csv"):
    """Appends token usage data to a CSV file."""
    # Check if the log file already exists to decide if we need a header row.
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["timestamp", "llm", "input_tokens", "output_tokens"])
        writer.writerow([datetime.now().isoformat(), llm, input_tokens, output_tokens])


# Function to make llm calls
def llm_with_retry(arg, max_retries = 15, structured_output = None, llm_name = None):
    """
    Call an LLM with optional structured output and retry logic.
    Args:
        arg: The input to the LLM.
        max_retries (int): The maximum number of retries in case of failure.
        structured_output (Optional[BaseModel]): A Pydantic BaseModel for structured output.
    """
    llm = ChatGoogleGenerativeAI(
        model = "gemini-2.0-flash",
        temperature = 0.7,
        max_tokens = 8192
        ).configurable_alternatives(
            ConfigurableField(id="llm"),
            default_key = "gemini_2_flash_default",
            gpt4o_mini = ChatOpenAI(model_name = "gpt-4o-mini", temperature = 0.7, max_tokens = 4096),
            gpt4o = ChatOpenAI(model_name = "gpt-4o", temperature = 0.7, max_tokens = 4096),
            haiku = ChatAnthropic(model_name = "claude-3-haiku-20240307", temperature = 0.7, max_tokens = 4096),
            sonnet = ChatAnthropic(model_name = "claude-3-5-sonnet-20240620", temperature = 0.7, max_tokens = 4096),
            haiku_3_5 = ChatAnthropic(model_name = "claude-3-5-haiku-20241022", temperature = 0.7, max_tokens = 4096),
            gemini_flash = ChatGoogleGenerativeAI(model = "gemini-1.5-flash-latest", temperature = 0.7, max_tokens = 8192),
            llama_3_1_70b = ChatGroq(model_name = 'llama-3.1-70b-versatile', temperature = 0.7, max_tokens = 4096),
            groq = ChatGroq(model_name = 'llama3-70b-8192', temperature = 0.7, max_tokens = 4096),
            gemini_2_flash = ChatGoogleGenerativeAI(model = "gemini-2.0-flash", temperature = 0.7, max_tokens = 8192),
            o1 = ChatOpenAI(model = 'o1'),
            o3_mini = ChatOpenAI(model = 'o3-mini'),
            gemini_2_flash_thinking = ChatGoogleGenerativeAI(model = "gemini-2.0-flash-thinking-exp", temperature = 0.7, max_tokens = 8192),
            pplx_deep_research = ChatPerplexity(model = "sonar-deep-research", temperature = 0, pplx_api_key = os.environ.get("PPLX_API_KEY")),
            # gemini_2_flash_open_router = ChatOpenAI(model = 'google/gemini-2.0-flash-exp:free', temperature = 0.7, max_completion_tokens = 8192, base_url = 'https://openrouter.ai/api/v1', api_key = os.environ.get('OPENROUTER_API_KEY'))
            )

    # Optionally add structured output
    if structured_output:
        llm = llm.with_structured_output(structured_output)

    retries = 0                  # Initialize the retry counter
    while retries < max_retries:
        try:
            if retries > 3:             # In such a case, try using gemini flash
                result = llm.with_config(
                    configurable={"llm": 'gemini_2_flash'}
                    ).invoke(arg)
                llm_name = 'gemini_2_flash'
            else:
                result = llm.invoke(arg)

            try:
                # Log structured token usage.
                log_token_usage(
                    llm = llm_name,
                    input_tokens = result.usage_metadata['input_tokens'],
                    output_tokens = result.usage_metadata['output_tokens'],
                    log_file = "token_usage_log.csv"
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
