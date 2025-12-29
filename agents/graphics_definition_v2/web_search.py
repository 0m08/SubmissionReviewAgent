from langsmith import traceable
import streamlit as st
from services.sheets_service import get_sheet_data_and_df, save_to_sheet, format_worksheet, clear_worksheet
from services.smart_progress_bar import SmartProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from agents.vector_store_image_search.web_image_search_tool import web_image_search_tool
from agents.graphics_workflow_v2.config.settings import (
    WEB_SEARCH_K,
)


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Web Search",
        "function_name": "execute_web_search_for_query",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def execute_web_search_for_query(query, k=WEB_SEARCH_K):
    """
    Execute web search for a single query and return reference data with URLs.
    
    :param query: Search query string
    :param k: Number of results to retrieve
    :return: List of reference data dictionaries with url
    """
    print(f"🌐 Executing web search for query: \"{query}\"")
    
    # Execute web search
    try:
        web_results_raw = web_image_search_tool(
            query=query,
            k=k
        )
        print(f"Raw results from web search: {len(web_results_raw)} items")
    except Exception as e:
        print(f"❌ Web search error: {str(e)}")
        return []
    
    # Convert web search results to reference data format
    reference_data_list = []
    for result in web_results_raw:
        metadata = result.get("metadata", {})
        
        # Get URL from metadata (source_url field)
        url = metadata.get("source_url", "")
        
        # Get title from metadata (name field)
        title = metadata.get("name", "Untitled")
        
        if url and url.strip():
            reference_data = {
                "url": url,
                "title": title,
            }
            reference_data_list.append(reference_data)
    
    print(f"✅ Found {len(reference_data_list)} results with URLs")
    
    return reference_data_list


def parse_search_queries_column(search_queries_text):
    """
    Parse the search_queries column to extract segments and their queries.
    
    Format:
    ---SEGMENT_1---
    query1
    query2
    query3
    
    ---SEGMENT_2---
    query1
    query2
    query3
    
    :param search_queries_text: The search_queries column content
    :return: List of tuples (segment_number, list_of_queries)
    """
    if not search_queries_text or search_queries_text.strip() == "" or search_queries_text == "nan":
        return []
    
    segments = []
    # Split by segment markers
    segment_pattern = r'---SEGMENT_(\d+)---'
    parts = re.split(segment_pattern, search_queries_text)
    
    # parts[0] is text before first segment (usually empty)
    # Then alternating: segment_num, segment_content, segment_num, segment_content, ...
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            segment_num = int(parts[i])
            segment_content = parts[i + 1].strip()
            
            # Extract queries (each line is a query)
            queries = [q.strip() for q in segment_content.split('\n') if q.strip()]
            
            if queries:
                segments.append((segment_num, queries))
    
    return segments


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Web Search",
        "function_name": "process_web_search_row",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def process_web_search_row(index, row, k=WEB_SEARCH_K):
    """
    Process a single row: parse search queries, execute web searches for each segment, deduplicate, format output.
    
    :param index: Row index
    :param row: Pandas Series with row data
    :param k: Number of results per query
    :return: Tuple of (index, web_results_text)
    """
    try:
        search_queries_text = str(row.get("search_queries", "")).strip()
        
        # Skip if search_queries is empty
        if not search_queries_text or search_queries_text == "nan":
            return index, ""
        
        # Parse segments and queries
        segments = parse_search_queries_column(search_queries_text)
        
        if not segments:
            return index, ""
        
        # Process each segment
        all_segment_results = []
        
        for segment_num, queries in segments:
            print(f"\n{'─'*45}")
            print(f"📦 Processing SEGMENT_{segment_num} with {len(queries)} queries")
            print(f"{'─'*45}")
            
            # Execute all queries for this segment
            all_results_for_segment = []
            seen_urls = set()  # Deduplicate within segment by URL
            
            for query_idx, query in enumerate(queries, 1):
                if not query.strip():
                    continue
                
                print(f"🔍 Query {query_idx}: \"{query}\"")
                
                # Execute search
                results = execute_web_search_for_query(
                    query=query,
                    k=k
                )
                
                print(f"✅ Query {query_idx} returned {len(results)} results")
                
                # Deduplicate by URL within this segment
                for ref in results:
                    url = ref.get("url", "").strip()
                    title = ref.get("title", "Untitled")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_results_for_segment.append({"url": url, "title": title})
            
            # Format segment results with title and URL
            segment_items = [f"Title: {ref['title']} | URL: {ref['url']}" for ref in all_results_for_segment]
            
            print(f"✅ SEGMENT_{segment_num}: Found {len(segment_items)} unique results")
            
            # Format segment results
            if segment_items:
                segment_output = [f"---SEGMENT_{segment_num}---"] + segment_items
                all_segment_results.append('\n'.join(segment_output))
        
        # Join all segments with double newline
        web_results_text = '\n\n'.join(all_segment_results)
        
        return index, web_results_text
        
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        return index, ""


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Web Search",
        "function_name": "run_web_search_for_all_rows",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def run_web_search_for_all_rows(sheet, k=WEB_SEARCH_K, max_workers=5):
    """
    Execute web search for all rows in the Slide Chunks sheet.
    
    :param sheet: The gspread sheet object.
    :param k: Number of results per query (default from settings).
    :param max_workers: Number of parallel workers (default 5).
    :return: None
    """
    worksheet_name = "Slide Chunks"
    
    # Load the worksheet and DataFrame
    worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)
    
    # Ensure web_results column exists
    if "web_results" not in df.columns:
        df["web_results"] = ""
    
    # Prepare for parallel processing - only process rows that need processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks first
        for index, row in df.iterrows():
            search_queries = str(row.get("search_queries", "")).strip()
            web_results = str(row.get("web_results", "")).strip()
            
            # Skip if search_queries is empty
            if not search_queries or search_queries == "nan":
                continue
            
            # Skip if web_results is already filled
            if web_results and web_results != "nan":
                continue
            
            # Submit task for processing
            future = executor.submit(process_web_search_row, index, row, k)
            futures_map[future] = index
        
        # If no rows to process, return early
        if not futures_map:
            print("All rows already processed or no valid search queries found.")
            return
        
        # Initialize progress tracker
        total_tasks = len(futures_map)
        progress = SmartProgressBar(
            total_tasks=total_tasks,
            description="Executing web search",
            save_interval=5
        )
        
        # Collect results as they complete
        for future in as_completed(futures_map):
            index = futures_map[future]
            try:
                row_index, web_results_text = future.result()
                
                # Update dataframe
                df.at[row_index, "web_results"] = web_results_text
                
                # Update progress
                progress.update()
                
                # Save every 5 rows
                if progress.should_save():
                    print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
                    save_to_sheet(worksheet, df)
                    format_worksheet(worksheet)
            except Exception as e:
                print(f"Error getting result for row {index}: {e}")
                progress.update()
    
    # Final save to sheet
    print('All web searches completed. Saving final DataFrame to sheet.')
    save_to_sheet(worksheet, df)
    format_worksheet(worksheet)
    print("✅ Web search complete and saved to sheet.")


def delete_web_results(sheet):
    """
    Remove the 'web_results' column from the specified worksheet.
    
    :param sheet: The gspread sheet object.
    :return: None
    """
    worksheet_name = "Slide Chunks"
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    if "web_results" in df.columns:
        df = df.drop(columns=["web_results"])
        clear_worksheet(ws)
        save_to_sheet(ws, df)
        print(f"🗑️ Deleted 'web_results' column from '{worksheet_name}' worksheet")
    else:
        print(f"ℹ️ 'web_results' column does not exist in '{worksheet_name}' worksheet")

