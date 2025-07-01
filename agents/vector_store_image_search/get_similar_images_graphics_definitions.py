import streamlit as st
import re
from typing import List, Any, Optional
import pandas as pd
from services.sheets_service import get_sheet_data_and_df, save_to_sheet
from services.llm_service import llm_with_retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from agents.vector_store_image_search.langgraph_agent_with_tools import run_graphics_search_graph
import json
import base64
import os
from services.drive_service import login_with_service_account
from pydrive2.drive import GoogleDrive
from dotenv import load_dotenv

sheet = st.session_state.get("sheet")
drive = st.session_state.get("drive")

load_dotenv()
key_bytes = base64.b64decode(os.environ["GDRIVE_SA_B64"])
sa_json = key_bytes.decode()
sa_dict = json.loads(sa_json)
gauth = login_with_service_account(json_str=sa_json)
gauth.ServiceAuth()
drive = GoogleDrive(gauth)


def generate_clean_query_from_scene(
    gtype: str,
    purpose: str,
    visuals: str,
    llm: str = "gemini_2_flash"
) -> Optional[str]:
    """
    Formats a prompt from scene fields and uses your llm_with_retry() function.
    Ensures the returned query starts with the Graphics Type.
    """
    prompt = f"""You are helping generate clean and effective image search queries from technical scene descriptions.

Here is a scene description:

Graphics Type: {gtype}

Purpose of the Scene: {purpose}

Visual Elements:
{visuals}

Instructions:
- Write a short, natural-language query that describes the image this scene is meant to depict.
- The query MUST begin with the Graphics Type (e.g., "Illustration of...", "Infographic showing...").
- Do NOT include references to scene numbers or IDs like "Scene 2_4".
- The query should be specific and visually descriptive, based on the Visual Elements and Purpose.
- Keep it to one or two concise sentences.
- Make it suitable for use as a search prompt for an illustration or infographic.

Output only the query, with no explanation or additional text."""

    try:
        response = llm_with_retry(prompt, llm_name=llm)
        return response.content.strip() if hasattr(response, "content") else str(response).strip()
    except Exception as e:
        print(f"LLM failed: {e}")
        return None


def generate_queries_from_definition_text(definition_text: str, llm: str):
    scene_blocks = re.findall(r"<scene>(.*?)</scene>", definition_text, re.DOTALL)
    clean_queries = []

    for scene in scene_blocks:
        purpose_match = re.search(r"Purpose of the Scene:\s*(.*)", scene)
        gtype_match = re.search(r"Graphics Type:\s*(.*)", scene)
        visuals_match = re.search(
            r"Visual Elements:\s*(.*?)(?:\n\n|\n[A-Z]|Arrangement:|Presentation and Transitions:)",
            scene,
            re.DOTALL,
        )

        gtype = gtype_match.group(1).strip() if gtype_match else ""
        purpose = purpose_match.group(1).strip() if purpose_match else ""
        visuals = visuals_match.group(1).strip().replace("\n", " ") if visuals_match else ""

        if any([gtype, purpose, visuals]):
            query = generate_clean_query_from_scene(gtype, purpose, visuals, llm=llm)
            clean_queries.append(query)

    return clean_queries


def run_generate_queries_from_definition(sheet, sheet_name, llm: str) -> list:
    
    """
    Run the generation of queries from graphics definitions in the Slide Chunks sheet.
    :param sheet: The Google Sheets spreadsheet object.
    :param worksheet_name: The name of the worksheet to process.
    :param llm: The LLM to use for generating queries.
    :return: A list of generated queries for each row in the Slide Chunks sheet.
    """
    # Load the sheet and DataFrame
    slide_chunks_sheet, slide_chunks_df = get_sheet_data_and_df(sheet, sheet_name)
    
    # Check if queries column already exists
    queries_column = "generated_graphics_queries"
    if queries_column in slide_chunks_df.columns and slide_chunks_df[queries_column].notna().any():
        print("Queries already exist. Skipping generation.")
        return slide_chunks_df[queries_column].tolist()

    # Prepare input data
    graphics_definitions = slide_chunks_df["checklist_revised_graphics_definition"].fillna("").tolist()
    all_clean_queries = [None] * len(graphics_definitions)

    # Processing function
    def process_definition(index, definition_text):
        if not definition_text.strip():
            return index, []
        clean_queries = generate_queries_from_definition_text(definition_text, llm=llm)
        return index, clean_queries

    # Parallel execution
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_definition, idx, text) for idx, text in enumerate(graphics_definitions)]

        for future in as_completed(futures):
            idx, result = future.result()
            all_clean_queries[idx] = result
            print(f"✅ Processed row {idx + 1} with {len(result)} queries.")

    # Add queries to the DataFrame
    slide_chunks_df[queries_column] = [json.dumps(qs) for qs in all_clean_queries]


    # Save back to sheet
    save_to_sheet(slide_chunks_sheet, slide_chunks_df)

    return all_clean_queries


def load_queries_from_sheet(sheet, sheet_name, queries_column="generated_graphics_queries"):
    _, slide_chunks_df = get_sheet_data_and_df(sheet, sheet_name)
    
    query_data = []
    for idx, cell in enumerate(slide_chunks_df[queries_column].fillna("")):
        try:
            queries = json.loads(cell) if cell.strip() else []
            if not isinstance(queries, list):
                queries = [queries]
        except json.JSONDecodeError:
            print(f"Row {idx + 1}: Invalid JSON in query cell. Skipping.")
            queries = []
        query_data.append(queries)

    return query_data



def search_images_for_query_list(queries: List[str], llm: str = "gemini_2_flash", max_workers: int = 5) -> List[str]:
    """
    Search for images based on a list of queries using parallel processing.
    
    :param queries: List of search queries.
    :param drive: The Google Drive service object for image search.
    :param llm: The LLM to use for generating queries.
    :param max_workers: Maximum number of threads to use for parallel processing.
    """
    # drive = st.session_state["drive"] 
    
    def search(query):
        try:
            results = run_graphics_search_graph(
                query=query,
                drive=drive,
                k=1,
                llm=llm
            )
            if results:
                metadata = results[0].get("metadata", {})
                # Prefer drive_url > source_url > fallback
                url = metadata.get("drive_url") or metadata.get("source_url") or "No URL found"
                return url
            else:
                return "No result"

        except Exception as e:
            return f"Error: {str(e)}"

    image_links = [None] * len(queries)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(search, query): idx
            for idx, query in enumerate(queries)
        }

        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            image_links[idx] = future.result()

    return image_links


def run_search_images_for_query_list(sheet, sheet_name, llm="gemini_2_flash", k=4):
    """
    Run image search for each query in the Slide Chunks sheet and save results.
    :param spreadsheet: The Google Sheets spreadsheet object.
    :param worksheet_name: The name of the worksheet to process.
    :param drive: The Google Drive service object for image search.
    :param llm: The LLM to use for generating queries.
    :param k: The number of top results to return for each query.
    :return: A list of image URLs for each row in the Slide Chunks sheet.
    
    """
        
    all_image_links_per_row = []
    
    slide_chunks_sheet, slide_chunks_df = get_sheet_data_and_df(sheet, sheet_name)

    all_clean_queries = load_queries_from_sheet(sheet, sheet_name)

    for row_index, query_list in enumerate(all_clean_queries):
        if not isinstance(query_list, list):
            print(f"❌ Row {row_index+1} is not a list! Got: {type(query_list)} - {query_list}")
            continue

        print(f"⚡ Processing row {row_index+1} with {len(query_list)} queries in parallel...")
        image_links = search_images_for_query_list(query_list, llm=llm)
        all_image_links_per_row.append("; ".join(image_links))

    slide_chunks_df["image_urls"] = all_image_links_per_row
    save_to_sheet(slide_chunks_sheet, slide_chunks_df)

    return all_image_links_per_row
