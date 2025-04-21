from modules.chain import Chain
from tqdm import tqdm
from services.sheets_service import get_sheet_data_and_df, create_or_read_worksheet, save_to_sheet, format_worksheet, clear_worksheet, delete_worksheet, get_worksheet_names
import pandas as pd
from services.helper_functions import get_outline_with_los
from services.smart_progress_bar import SmartProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from google import genai
from google.genai import types


subtopic_research_prompt = """Research the following topic within a course - {course_name} and target audience - {target_audience}.

Topic - {subtopic}.
"""

def google_search_with_grounding(course_name, target_audience, subtopic_query):
    """
    Makes an API call to Gemini with search grounding, then attempts to resolve each returned URL in parallel.
    :param course_name: str
    :param target_audience: str
    :param subtopic_query: str
    :return: (response, list_of_uris)
    """
    client = genai.Client()
    
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=subtopic_research_prompt.format(
            course_name=course_name,
            target_audience=target_audience,
            subtopic=subtopic_query
        ),
        config=types.GenerateContentConfig(
            tools=[types.Tool(
                google_search=types.GoogleSearchRetrieval
            )]
        )
    )

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

def deep_research_subtopic(course_name, target_audience, subtopic, llm="gemini_with_grounding"):
    """
    Performs deep research on a specific subtopic for a course.
    
    :param course_name (str): The name of the course
    :param target_audience (str): The target audience for the course
    :param subtopic (str): The subtopic to research
    :param llm (str): The language model to use ('pplx_deep_research' or 'gemini_with_grounding')
    :return tuple: A tuple containing (research_content, sources)
    """
    if llm == "gemini_with_grounding":
        research_content, sources = google_search_with_grounding(course_name, target_audience, subtopic)
    else:
        research_agent = Chain(llm=llm, use_output_parser=False)
        
        research_agent.add_message(
            role='user', content=subtopic_research_prompt.format(
                course_name=course_name,
                target_audience=target_audience,
                subtopic=subtopic
            )
        )
        
        response = research_agent.run()
        print(f"Completed research for subtopic: {subtopic}")
        
        # Extract content and citations from the response
        research_content = response.content if hasattr(response, 'content') else str(response)
        sources = response.additional_kwargs.get("citations", []) if hasattr(response, 'additional_kwargs') else []
    
    return research_content, "\n".join(sources)


def create_deep_research_sheet(sheet, worksheet_name):
    """
    Creates the Deep Research sheet and populates it with subtopic_queries from the Rough Outline sheet.
    
    :param sheet (object): Google Sheets object
    :param worksheet_name (str): Name of the worksheet to create or read
    :return: tuple (worksheet, dataframe) - The Deep Research worksheet and its dataframe
    """
    # Create or clear the Deep Research worksheet
    deep_research_sheet, deep_research_df = create_or_read_worksheet(sheet, worksheet_name)

    # Check if df already populated
    if 'subtopic_query' in deep_research_df.columns and 'research' in deep_research_df.columns and 'source' in deep_research_df.columns:
        # Check if rows are populated
        if deep_research_df['subtopic_query'].notna().all():
            print("Deep Research sheet already populated. Skipping creation.")
            return deep_research_sheet, deep_research_df

    # Read the Rough Outline sheet to get topics and subtopics
    rough_outline_sheet, rough_outline_df = get_sheet_data_and_df(sheet, "Rough Outline")    
    
    # Extract subtopics from the Rough Outline
    subtopics_list = []
    
    # Combine Topic and Subtopic for each row
    for _, row in rough_outline_df.iterrows():
        topic = row['Topic']
        subtopic = row['Subtopic']
        if pd.notna(topic) and topic.strip() and topic not in subtopics_list:
            subtopics_list.append(topic)
        if pd.notna(subtopic) and subtopic.strip() and subtopic not in subtopics_list:
            subtopics_list.append(subtopic)
    
    # Create DataFrame with subtopic_query, research, and source columns
    deep_research_df = pd.DataFrame({
        'subtopic_query': subtopics_list, 
        'research': [''] * len(subtopics_list),
        'source': [''] * len(subtopics_list)
    })
    
    # Save to sheet
    save_to_sheet(worksheet=deep_research_sheet, df=deep_research_df)
    format_worksheet(deep_research_sheet)
    
    print(f"Created Deep Research sheet with {len(subtopics_list)} subtopics.")
    return deep_research_sheet, deep_research_df


def run_deep_research(sheet, worksheet_name, course_name, target_audience, llm = "pplx_deep_research"):
    """
    Runs deep research on subtopics in the Deep Research sheet and populates the research and source columns.
    
    :param sheet (object): Google Sheets object
    :param worksheet_name (str): Name of the worksheet to create or read
    :param course_name (str): Name of the course
    :param target_audience (str): Target audience for the course
    :param llm (str): Language model to use (default: 'pplx_deep_research')
    :return: None
    """
    # Get the Deep Research sheet
    deep_research_sheet, deep_research_df = create_deep_research_sheet(sheet = sheet, worksheet_name = worksheet_name)
    
    # If research column is already populated for all rows, skip processing
    if 'research' in deep_research_df.columns and deep_research_df['research'].notna().all() and deep_research_df['research'].str.strip().all():
        print("Research column is already populated for all subtopics. Skipping processing.")
        return
    
    # Ensure research and source columns exist
    if 'research' not in deep_research_df.columns:
        deep_research_df['research'] = ''
    if 'source' not in deep_research_df.columns:
        deep_research_df['source'] = ''
    
    # Prepare for parallel processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers = 5) as executor:
        # Submit tasks for each subtopic
        for index, row in deep_research_df.iterrows():
            subtopic = row['subtopic_query']
            
            # Skip if research is already populated for this subtopic
            if pd.notna(row['research']) and row['research'].strip():
                print(f"Skipping {index}: Research already populated for '{subtopic}'.")
                continue
                
            if not subtopic.strip():
                print(f"Skipping {index}: Empty subtopic.")
                continue
                
            # Submit the task
            future = executor.submit(
                deep_research_subtopic,
                course_name,
                target_audience,
                subtopic,
                llm
            )
            
            # Map the Future to the index
            futures_map[future] = index
        
        # Collect the results as they complete
        total_tasks = len(futures_map)
        save_interval = 1  # how often to save (in number of completed tasks)
        
        # Initialize the progress tracker
        progress = SmartProgressBar(total_tasks = total_tasks, description = "Percent complete", save_interval = save_interval)
        
        # Process results as they complete
        for future in tqdm(as_completed(futures_map), total = total_tasks):
            index = futures_map[future]  # retrieve the index
            research_content, sources = future.result()
            
            # Update the research and source columns
            deep_research_df.loc[index, 'research'] = research_content
            deep_research_df.loc[index, 'source'] = sources
            
            # Update progress
            progress.update()
            
            # Check if we should save
            if progress.should_save():
                print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
                save_to_sheet(worksheet = deep_research_sheet, df = deep_research_df)
    
    # Final save to sheet after all tasks
    print('All subtopics processed. Saving final DataFrame to sheet.')
    save_to_sheet(worksheet = deep_research_sheet, df = deep_research_df)
    
    return
