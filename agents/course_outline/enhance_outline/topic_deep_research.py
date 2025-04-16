from modules.chain import Chain
from tqdm import tqdm
from services.sheets_service import get_sheet_data_and_df, save_to_sheet
import pandas as pd
from services.smart_progress_bar import SmartProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from google import genai
from google.genai import types


# Prompt for researching topics
topic_research_prompt = """Research the following topic within a course - {course_name} and target audience - {target_audience}.

Topic - {topic}.
"""


def google_search_with_grounding(course_name, target_audience, topic_query):
    """
    Makes an API call to Gemini with search grounding, then attempts to resolve each returned URL in parallel.
    :param course_name: str
    :param target_audience: str
    :param topic_query: str
    :return: (response, list_of_uris)
    """
    client = genai.Client()
    
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=topic_research_prompt.format(
            course_name=course_name,
            target_audience=target_audience,
            topic=topic_query
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
        # Gather all URLs first
        urls = []
        for candidate in response_obj.candidates:
            for grounding_chunk in candidate.grounding_metadata.grounding_chunks:
                urls.append(grounding_chunk.web.uri)

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


def deep_research_topic(course_name, target_audience, topic, llm="gemini_with_grounding"):
    """
    Performs deep research on a specific topic for a course.
    
    :param course_name (str): The name of the course
    :param target_audience (str): The target audience for the course
    :param topic (str): The topic to research
    :param llm (str): The language model to use ('pplx_deep_research' or 'gemini_with_grounding')
    :return tuple: A tuple containing (research_content, sources)
    """
    if llm == "gemini_with_grounding":
        research_content, sources = google_search_with_grounding(course_name, target_audience, topic)
    else:
        research_agent = Chain(llm=llm, use_output_parser=False)
        
        research_agent.add_message(
            role='user', content=topic_research_prompt.format(
                course_name=course_name,
                target_audience=target_audience,
                topic=topic
            )
        )
        
        response = research_agent.run()
        print(f"Completed research for topic: {topic}")
        
        # Extract content and citations from the response
        research_content = response.content if hasattr(response, 'content') else str(response)
        sources = response.additional_kwargs.get("citations", []) if hasattr(response, 'additional_kwargs') else []
    
    return research_content, "\n".join(sources)


def run_topic_deep_research(sheet, worksheet_name, course_name, target_audience, llm="pplx_deep_research"):
    """
    Runs deep research on topics in the Topic Deep Research sheet and populates the research and source columns.
    
    :param sheet (object): Google Sheets object
    :param worksheet_name (str): Name of the worksheet to read and update (default: "Topic Deep Research")
    :param course_name (str): Name of the course
    :param target_audience (str): Target audience for the course
    :param llm (str): Language model to use (default: 'pplx_deep_research')
    :return: None
    """
    # Get the Topic Deep Research sheet
    deep_research_sheet, deep_research_df = get_sheet_data_and_df(sheet, worksheet_name)
    
    # Check if the sheet exists and has the required columns
    if deep_research_df.empty:
        raise Exception(f"The '{worksheet_name}' sheet is empty. Please ensure it is properly created.")
    
    if 'topic_query' not in deep_research_df.columns:
        raise Exception(f"The '{worksheet_name}' sheet is missing the required 'topic_query' column.")
    
    # If research column is already populated for all rows, skip processing
    if 'research' in deep_research_df.columns and deep_research_df['research'].notna().all() and deep_research_df['research'].str.strip().all():
        print("Research column is already populated for all topics. Skipping processing.")
        return
    
    # Ensure research and sources columns exist
    if 'research' not in deep_research_df.columns:
        deep_research_df['research'] = ''
    if 'source' not in deep_research_df.columns:
        deep_research_df['source'] = ''
    
    # Prepare for parallel processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit tasks for each topic
        for index, row in deep_research_df.iterrows():
            topic = row['topic_query']
            
            # Skip if research is already populated for this topic
            if pd.notna(row['research']) and row['research'].strip():
                print(f"Skipping {index}: Research already populated for '{topic}'.")
                continue
                
            if not topic.strip():
                print(f"Skipping {index}: Empty topic.")
                continue
                
            # Submit the task
            future = executor.submit(
                deep_research_topic,
                course_name,
                target_audience,
                topic,
                llm
            )
            
            # Map the Future to the index
            futures_map[future] = index
        
        # Collect the results as they complete
        total_tasks = len(futures_map)
        save_interval = 1  # how often to save (in number of completed tasks)
        
        # Initialize the progress tracker
        progress = SmartProgressBar(total_tasks=total_tasks, description="Percent complete", save_interval=save_interval)
        
        # Process results as they complete
        for future in tqdm(as_completed(futures_map), total=total_tasks):
            index = futures_map[future]  # retrieve the index
            research_content, sources = future.result()
            
            # Update the research and sources columns
            deep_research_df.loc[index, 'research'] = research_content
            deep_research_df.loc[index, 'source'] = sources
            
            # Update progress
            progress.update()
            
            # Check if we should save
            if progress.should_save():
                print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
                save_to_sheet(worksheet=deep_research_sheet, df=deep_research_df)
    
    # Final save to sheet after all tasks
    print('All topics processed. Saving final DataFrame to sheet.')
    save_to_sheet(worksheet=deep_research_sheet, df=deep_research_df)
    
    return
