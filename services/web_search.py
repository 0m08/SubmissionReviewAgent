from utils.decorator_helpers import try_n_times
from langchain_core.prompts import ChatPromptTemplate
from services.llm_service import llm_with_retry, output_parser, csv_list_parser
import re
from duckduckgo_search import DDGS
from exa_py import Exa
import os
import streamlit as st
from langsmith import traceable

@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Obtain Web Article Links",
    "function_name": "ddgs_search",
    "user_id": st.session_state.get("role", "anonymous")
})
@try_n_times(2, wait = 5, backoff = 'linear')
def ddgs_search(search_query, max_results=30, backend = 'api'):
    """
    Search the web using DuckDuckGo
    Args:
        search_query (str): The query to search on DuckDuckGo
        max_results (int): The maximum number of results to return
    Returns:
        search_results (list): The response from DuckDuckGo. Contains a list of dictionaries with keys: title, href, body
    """
    ddgs = DDGS()
    search_results = ddgs.text(
        keywords = search_query,          # keywords: keywords for query.
        region = "wt-wt",                 # region: wt-wt, us-en, uk-en, ru-ru, etc. Defaults to "wt-wt".
        safesearch = "moderate",          # safesearch: on, moderate, off. Defaults to "moderate".
        timelimit = None,                 # timelimit: d, w, m, y. Defaults to None.
        backend = backend,                # backend: api, html, lite. Defaults to api. api - collect data from https://duckduckgo.com, html - collect data from https://html.duckduckgo.com, lite - collect data from https://lite.duckduckgo.com.
        max_results = max_results,        # max_results: max number of results. If None, returns results only from the first response. Defaults to None.
        )
    return search_results

@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Obtain Web Article Links",
    "function_name": "exa_search",
    "user_id": st.session_state.get("role", "anonymous")
})
@try_n_times(2, wait = 5, backoff = 'linear')
def exa_search(search_query):
    """
    Search the web with Exa.
    Returns empty list if no results found (instead of raising exception).
    Retries on network/API errors, but returns empty list for "no results" case.
    """
    exa = Exa(api_key = os.environ.get("EXA_API_KEY"))

    # Using search_and_contents for improved results
    response = exa.search_and_contents(
        search_query, 
        type="auto", 
        summary=True
    )

    # Check if response is valid and contains results
    # Return empty list (don't raise exception) - this allows workflow to continue
    if not response or not hasattr(response, 'results') or not response.results:
        print(f"Exa AI returned no valid results for query: {search_query}")
        return []

    # Process results correctly
    search_results = [
        {
            "title": getattr(result, "title", "No title"),
            "href": getattr(result, "url", ""),
            "body": getattr(result, "summary", "No found")
        }
        for result in response.results
    ]

    return search_results

@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Obtain Web Article Links",
    "function_name": "web_search_screening",
    "user_id": st.session_state.get("role", "anonymous")
})
@try_n_times(3)
def web_search_screening(course_name, course_outline, search_query, llm = 'gemini_3_flash'):
    """
    Screen the web search results for relevant articles
    Args:
        search_query (str): The current search query used
    Returns:
        screened_articles (list): List of urls of relevant articles
    """
    print("---SEARCH WEB AND SCREENING SEARCH RESULTS---")

    def convert_search_results_to_md_table(data: list) -> str:
        """
        Convert a list of dictionaries to a markdown table.
        Args:
            data (list): A list of dictionaries, where each dictionary represents a row in the table.
        Returns:
            markdown (str): The markdown table as a string.
        """
        # Return empty table if data is empty
        if not data:
            return "| Index |\n| --- |\n| No data |\n"

        # Escape any special markdown characters
        def escape_md(text):
            return str(text).replace('|', '\\|')

        # Extract columns (keys) from the first dict
        columns = list(data[0].keys())

        # Initialize the markdown table with headers (Index + the dict keys)
        markdown = "| Index | " + " | ".join(escape_md(col) for col in columns) + " |\n"
        markdown += "| --- | " + " | ".join("---" for _ in columns) + " |\n"

        # Add each dictionary as a row in the markdown table
        for idx, item in enumerate(data, start=1):
            row = [escape_md(item.get(col, "")) for col in columns]
            markdown += f"| {idx} | " + " | ".join(row) + " |\n"

        return markdown

    try:
        search_results = ddgs_search(search_query = search_query)
    except Exception as e:
        print(f"DDGS failed with {e}")
        print("Falling back to Exa search")
        search_results = exa_search(search_query = search_query)

    search_results_md = convert_search_results_to_md_table(search_results)

    ## --------------------- PROMPT --------------------- ##

    web_search_screening_prompt = """You are an AI assistant tasked with analyzing search results to identify relevant articles for a specific course. Your goal is to mark articles that are worth checking in more detail based on the course name and search query provided.

The course name is:
<course_name>
{course_name}
</course_name>

The course outline is:
<course_outline>
{course_outline}
</course_outline>

The search query is:
<search_query>
{search_query}
</search_query>

Here are the search results:
<search_results>
{search_results}
</search_results>

Your task is to review each article in the search results and determine if it's worth checking in more detail for the given course and search query. Consider the following criteria:

1. Relevance to the course topic
2. Potential usefulness of the information for the course
3. Credibility and reliability of the source

For each article, provide your reasoning for why it should or should not be marked for further investigation. Be concise but clear in your explanations.

After analyzing all articles, provide a final list of the indices of articles you've marked as worth checking in more detail. Present this list in ascending order.

Format your response as follows:
<output>
<analysis>
1. [Summary of article 1 followed by reasoning, followed by the verdict - "Mark for further investigation" or "Don't mark for further investigation"]
2. [Summary of article 2 followed by reasoning, followed by the verdict - "Mark for further investigation" or "Don't mark for further investigation"]
...
</analysis>

<marked_articles>
[List of indices of marked articles in ascending order as a CSV. Don't put square brackets around the list]
</marked_articles>
</output>

Remember, the goal is to eliminate articles that are definitely not going to have any useful information for the given course and search query. When in doubt, it's better to include an article for further investigation.
"""

    web_search_screening_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("human", web_search_screening_prompt),
        ]
    )

    # Function to extract a list of texts witin the specified xml tags
    def extract_marked_articles(text: str):
        print(text)

        texts = [] # Store the text within each tags as a list
        for tag in ['marked_articles']:
            # regex pattern
            pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
            match_1 = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match_1:
                texts.append(match_1.group(1))
            else:
                raise Exception
                print(f"Unable to extract the text from the {tag} tags")
        return texts[0]

    # To get urls from index
    def get_articles_from_index(index: list):
        return [search_results[int(i) - 1]['href'] for i in index]

    # Define the chain
    web_search_screening_chain = web_search_screening_prompt_template | llm_with_retry | output_parser | extract_marked_articles | csv_list_parser | get_articles_from_index

    # Run the chain
    marked_articles = web_search_screening_chain.with_config(configurable={"llm": llm}).invoke(
        {"course_name": course_name,
        "course_outline": course_outline,
        "search_query": search_query,
        "search_results": search_results_md}
        )

    return marked_articles

