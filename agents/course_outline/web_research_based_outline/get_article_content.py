# from tqdm import tqdm
from services.sheets_service import (
    get_sheet_data_and_df,
    save_to_sheet,
    clear_worksheet,
)
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from services.web_page_loaders import get_docs_from_url
from services.smart_progress_bar import SmartProgressBar
# from services.helper_functions import create_and_populate_columns, escape_single_braces
import requests
import html2text
from bs4 import BeautifulSoup
from tqdm import tqdm
import streamlit as st
from langchain_core.documents import Document
from langchain_community.document_loaders import AsyncHtmlLoader, PyPDFLoader, AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import PyPDFLoader
from services.helper_functions import create_and_populate_columns, escape_single_braces
from services.sheets_service import get_sheet_data_and_df
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.web_page_loaders import clean_mark_article_stdout
from langsmith import traceable


@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Fetch Web Article Content",
    "function_name": "extract_markdown_from_webpage",
    "user_id": st.session_state.get("role", "anonymous")
})
def extract_markdown_from_webpage(url, timeout=10):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }

    try:
        # Fetch the webpage content with a timeout
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()  # Ensure the request was successful
    except requests.exceptions.Timeout:
        return "The request timed out after {} seconds.".format(timeout)
    except requests.exceptions.HTTPError as err:
        return "HTTP error occurred: {}".format(err)
    except requests.exceptions.RequestException as err:
        return "Error during requests to {}: {}".format(url, err)

    # Parse the HTML to remove unwanted content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Remove elements with the class "col-md-3"
    for div in soup.find_all('div', class_='col-md-3'):
        div.decompose()

    # Convert the modified HTML to Markdown
    converter = html2text.HTML2Text()
    converter.ignore_links = False  # Set to True if you want to ignore converting links
    converter.ignore_images = False  # Set to False if you want to include image Markdown
    converter.ignore_tables = False  # Set to True if you want to simplify output by ignoring tables
    markdown = converter.handle(str(soup))

    return markdown

#@try_n_times(2)
@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Fetch Web Article Content",
    "function_name": "get_doc_from_url",
    "user_id": st.session_state.get("role", "anonymous")
})
def get_doc_from_url(url: str, query: str):
    """
    Get the page content of a given URL
    :param url: The URL to get the page content from
    :param query: The search query associated with the content
    :return: The page content as a string
    """
 

    print("---GET PAGE CONTENT FROM URL---")
    
    @traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Fetch Web Article Content",
    "function_name": "fetch_with_jina_ai",
    "user_id": st.session_state.get("role", "anonymous")
})
    def fetch_with_jina_ai(url, query):
        print(f"--- Fetching content from URL using Jina AI: {url} ---")
        try:
            response = requests.get(f"https://r.jina.ai/{url}", timeout=15)
            if response.status_code != 200:
                print(f"Jina AI API request failed for {url}. Status Code: {response.status_code}")
                return None

            extracted_text = response.text.strip()
            if not extracted_text:
                print(f"Jina AI extraction returned empty content for {url}. Skipping.")
                return None

            return Document(
                page_content=extracted_text,
                metadata={'source': url, 'query': query, 'method': 'JinaAI'}
            )
        except requests.exceptions.RequestException as e:
            print(f"Jina AI extraction failed for {url}. Error: {e}")
            return None

    # If URL is a PDF
    if url.endswith('.pdf'):
        try:
            loader = PyPDFLoader(url)
            pages = loader.load()
            doc = pages[0]
            doc.page_content = '\n\n'.join([page.page_content for page in pages])
            doc.metadata['query'] = query
        except Exception as e:
            print(f"PyPDFLoader failed for {url}. Trying Jina AI. Error: {e}")
            return fetch_with_jina_ai(url, query)

    # If URL is a web page
    else:
        try:
            markdown = extract_markdown_from_webpage(url)
            doc = Document(
                page_content=markdown,
                metadata={'source': url, 'query': query}
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Error extracting markdown from {url}. Trying AsyncChromiumLoader. Error: {e}")
            try:
                loader = AsyncChromiumLoader([url])
                docs = loader.load()
                html2text = Html2TextTransformer(ignore_links=False, ignore_images=False)
                doc = html2text.transform_documents(docs)[0]
                doc.metadata['query'] = query
            except Exception as e:
                print(f"AsyncChromiumLoader failed for {url}. Trying Jina AI. Error: {e}")
                return fetch_with_jina_ai(url, query)

    return doc


@traceable(metadata={
    "agent_name": "course_outline",
    "step_name": "Fetch Web Article Content",
    "function_name": "fetch_and_process_article",
    "user_id": st.session_state.get("role", "anonymous")
})
def fetch_and_process_article(index, row):
    """
    Fetches and processes article content for a given row.
    """
    # if index in {21, 233, 151, 93}:  # Skip specific indices
    #     return index, None

    try:
        url = row.get('article')
        search_query = row.get('query')
        article_content = None
        try:
            article_content = clean_mark_article_stdout(url)
        except TimeoutError as e:
            print(f"Timeout error for row {index}: {e}")
            article_content = None
        except Exception as e:
            print(f"Error fetching content from get_text_clean_mark for {index}: {e}")

        if not article_content or article_content.startswith('---\nlink: null'):
            try:
                doc = get_doc_from_url(url=url, query=search_query)
                article_content = doc.page_content if doc else None
            except Exception as e:
                print(f"Error fetching content from get_doc_from_url for {index}: {e}")
                article_content = None

        if article_content:
            article_content = escape_single_braces(article_content)

        # Ensure article_content is not longer than 8,00,000 characters ~ 200,000 tokens
        if article_content and len(article_content) > 800000:
            article_content = article_content[:800000]

        return index, article_content
    except Exception as e:
        print(f"Unexpected error processing index {index}: {e}")
        return index, None
    
    
def run_fetch_article_content(sheet, worksheet_name):
    """
    Fetch article content for all rows in parallel using ThreadPoolExecutor with a progress bar.
    """
    # Read sheet and df
    preliminary_research_sheet, preliminary_research_df = get_sheet_data_and_df(sheet=sheet, sheet_name=worksheet_name)

    # Ensure 'article_content_0' column exists
    if 'article_content_0' not in preliminary_research_df.columns:
        preliminary_research_df['article_content_0'] = ''

    # Check if processing is already complete
    if preliminary_research_df.iloc[-1]['article_content_0']:
        print('Article Content already populated')
        return preliminary_research_df

    futures = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        for index, row in preliminary_research_df.iterrows():
            # Skip if already populated
            if row['article_content_0'] != '':
                continue
            # Submit the task
            futures.append(executor.submit(fetch_and_process_article, index, row))

        # Collect the results as they complete
        total_tasks = len(futures)
        save_interval = 5  # how often to save (in number of completed tasks)

        # Initialize the progress tracker
        progress = SmartProgressBar(total_tasks = total_tasks, description = "Percent complete", save_interval = save_interval)

        # completed_count = 0
        # total_tasks = len(futures)
        # progress_bar = st.progress(0, text="Percent complete: 0%")

        for future in tqdm(as_completed(futures), total=total_tasks):
            index, article_content = future.result()
            if article_content:
                preliminary_research_df = create_and_populate_columns(
                    df=preliminary_research_df,
                    text=article_content,
                    specific_index=index,
                    col_base_name='article_content',
                    chunk_size=49000
                )

            # Update progress
            progress.update()

            # Check if we should save
            if progress.should_save():
                print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
                save_to_sheet(worksheet = preliminary_research_sheet, df = preliminary_research_df)

    # Final save to sheet after all tasks
    print('All rows processed. Saving final DataFrame to sheet.')
    save_to_sheet(worksheet = preliminary_research_sheet, df = preliminary_research_df)

    # Hide the columns
    # hide_columns_by_name(worksheet = preliminary_research_sheet, column_names = column_names, df = preliminary_research_df)

    #         completed_count += 1
    #         if completed_count % 10 == 0:
    #             preliminary_research_df = preliminary_research_df.astype(str)
    #             preliminary_research_sheet.update(
    #                 [preliminary_research_df.columns.values.tolist()] + preliminary_research_df.values.tolist()
    #             )

    #         fraction_complete = completed_count / total_tasks
    #         progress_bar.progress(fraction_complete, text=f"Percent complete: {int(fraction_complete * 100)}%")
    #         # time.sleep(0.1)  # Avoid excessive updates

    # preliminary_research_df = preliminary_research_df.astype(str)
    # preliminary_research_sheet.update(
    #     [preliminary_research_df.columns.values.tolist()] + preliminary_research_df.values.tolist()
    # )
    
    return preliminary_research_df



# def run_get_article_content(sheet, worksheet_name):
#     """
#     This functions gets the article content for all the links in the Preliminary Research sheet

#     :param sheet: The sheet object.
#     :param worksheet_name: The worksheet name.
#     :return: None
#     """

#     # Read the sheets and df
#     preliminary_research_sheet, preliminary_research_df = get_sheet_data_and_df(sheet, worksheet_name)

#     # Ensure 'article_content_0' column exists
#     if 'article_content_0' not in preliminary_research_df.columns:
#         preliminary_research_df['article_content_0'] = ''

#     # Check if processing is already complete
#     if preliminary_research_df.iloc[-1]['article_content_0']:
#         print('Article Content already populated')
#         return

#     # Wrap the core function in a try except block
#     def get_article_content(url, query):
#         try:
#             docs = get_docs_from_url(url = url, query = query)
#             article_content = escape_single_braces(
#                 ''.join([doc.page_content for doc in docs])
#             )
#         except Exception as e:
#             print(f"Error extracting web article content: {e}")
#             article_content = ''
#         return article_content

#     # Prepare for parallel processing
#     futures_map = {}
#     with ThreadPoolExecutor(max_workers=5) as executor:
#         # Submit tasks for each row
#         for index, row in preliminary_research_df.iterrows():
            
#             # Skip if already populated
#             if row['article_content_0'] != '':
#                 continue
            
#             # Submit the task
#             future = executor.submit(
#                 get_article_content,
#                 row['article'],
#                 row['query']
#             )
#             # Map the Future to the index
#             futures_map[future] = index

#         # Collect the results as they complete
#         total_tasks = len(futures_map)
#         save_interval = 5  # how often to save (in number of completed tasks)

#         # Initialize the progress tracker
#         progress = SmartProgressBar(total_tasks = total_tasks, description = "Percent complete", save_interval = save_interval)

#         for future in tqdm(as_completed(futures_map), total=total_tasks):
#             index = futures_map[future]
#             article_content = future.result()
            
#             if article_content:
#                 preliminary_research_df = create_and_populate_columns(
#                     df = preliminary_research_df,
#                     text = article_content,
#                     specific_index = index,
#                     col_base_name = 'article_content',
#                     chunk_size = 49000
#                 )

#             # Update progress
#             progress.update()

#             # Check if we should save
#             if progress.should_save():
#                 print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
#                 save_to_sheet(worksheet = preliminary_research_sheet, df = preliminary_research_df)

#     # Final save to sheet after all tasks
#     print('All rows processed. Saving final DataFrame to sheet.')
#     save_to_sheet(worksheet = preliminary_research_sheet, df = preliminary_research_df)

#     column_names = [column_name for column_name in preliminary_research_df.columns if "article_content" in column_name]

#     # Hide the columns
#     # hide_columns_by_name(worksheet = preliminary_research_sheet, column_names = column_names, df = preliminary_research_df)

#     return


def delete_article_content(sheet, worksheet_name="Preliminary Research"):
    """Remove article_content columns from the Preliminary Research sheet."""
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    cols = [c for c in df.columns if c.startswith("article_content")]
    if cols:
        df = df.drop(columns=cols)
        clear_worksheet(ws)
        save_to_sheet(ws, df)






