import streamlit as st
import re
from typing import List, Optional, Any
from services.sheets_service import get_sheet_data_and_df, save_to_sheet, clear_worksheet
from concurrent.futures import ThreadPoolExecutor, as_completed
from agents.vector_store_image_search.langgraph_agent_with_tools import run_graphics_search_graph
import json
import base64
import os
from tqdm import tqdm 
import time
import random
from services.drive_service import login_with_service_account
from pydrive2.drive import GoogleDrive
from dotenv import load_dotenv
from modules.chain import Chain

from services.smart_progress_bar import SmartProgressBar



load_dotenv()
key_bytes = base64.b64decode(os.environ["GDRIVE_SA_B64"])
sa_json = key_bytes.decode()
sa_dict = json.loads(sa_json)
gauth = login_with_service_account(json_str=sa_json)
gauth.ServiceAuth()
drive = GoogleDrive(gauth)

sheet = st.session_state.get("sheet")

def parse_flat_graphics_definitions(definition_text: str) -> list:
    """
    Parses a structured graphics definition text where each scene contains:
    - Sentence:
    - Purpose of the Scene:
    - Graphics Type:
    - Followed by lines of visual descriptions (until next 'Sentence:' or end)
    """
    pattern = re.compile(
        r"Sentence:\s*(?P<sentence>.*?)\n"
        r"Purpose of the Scene:\s*(?P<purpose>.*?)\n"
        r"Graphics Type:\s*(?P<graphics_type>.*?)\n"
        r"(?P<visuals>(.*?))(?=\nSentence:|\Z)",  # Stop at next Sentence or EOF
        re.DOTALL
    )

    scenes = []
    for match in pattern.finditer(definition_text.strip()):
        sentence = match.group("sentence").strip()
        purpose = match.group("purpose").strip()
        graphics_type = match.group("graphics_type").strip()
        visuals = match.group("visuals").strip()

        scenes.append({
            "sentence": sentence,
            "purpose": purpose,
            "graphics_type": graphics_type,
            "visuals": visuals
        })

    return scenes


generate_clean_queries_prompt_template = """  
You will be given a detailed graphics definition for a visual scene. Your task is to extract literal, concise image search queries for the sentences provided in the graphics definition. 
A graphics definition is solely defined as all the sentences after the "Graphics Type" line, which includes the visuals and how the images should be used in the scene.
From the provided definition, generate a list of search queries that will help retrieve images matching the visual elements described in the scene. 
In the first sentence after the "Graphics Type" line, it should be the first search query, and subsequent sentences should be used to generate additional queries.
Your queries should be specific to the visual elements mentioned in the definition and should not include any non-visual keywords or phrases.Emphasis should be placed on the visual elements, as they are to be the main focus of the search queries, not the supposrting text or context. 
Before generating queries, check the sentence first, ensure it contains the main subject of the scene, and that it is not just a general statement or introduction.
For example, if the sentences are, 

"The electrical circuit diagram fades in first.
The "Safety" icon appears, and an arrow connects it to the circuit diagram as "safety" is spoken.
The "Efficiency" icon appears, and an arrow connects it to the circuit diagram as "efficiency" is spoken.
The "Troubleshooting" icon appears, and an arrow connects it to the circuit diagram as "effective troubleshooting" is spoken."

You would generate the following queries:
1. "Electrical circuit diagram"
2. "Safety icon"
3. "Efficiency icon"
4. "Troubleshooting icon"

It will be incorrect to generate queries like, "arrow connecting efficiency icon to circuit diagram", or, "arrow connecting troubleshooting icon to circuit diagram", or, "The electrical circuit diagram fades in first" or "The 'Safety' icon appears, or an arrow connects it to the circuit diagram as 'safety' is spoken." 
It is important to focus on the visual elements that can be searched for, such as icons, diagrams, or specific objects mentioned in the definition.
Also, the sentence, purpose, and graphics type can be used to add more information to the queries, but they should not be included in the queries themselves.
Focus only on the visual elements described in the definition below the graphics type line. 
When a query is generated, is should be a sensible sentence, not just words obtained fron the definition and brought together. It should be a complete sentence that describes the visual elements in a way that can be used for image search.
If a definition does not contain any visual elements, do not generate any queries for it. 


Acceptable Queries Should:
- Be short, specific, and visually descriptive.  
- Use clear, descriptive terms that directly relate to the visual elements described. 
- Include all visual elements mentioned in the definition.

 

Format Rules:
- One query per line.
- No numbering or grouping.
- Keep it clean and search-friendly (this will be used directly for image search).  


<examples>
{examples}
</examples>

Graphics Definition:
Sentence: {sentence}
Purpose of the Scene: {purpose}
Graphics Type: {gtype}
Visual Elements: {visuals}


Now output the list of search queries needed to get all relevant visual elements for this scene. Use the format shown above.

Output your response in the following format:
<clean_query_generation>
[Place your image search queries here]
</clean_query_generation>


"""


generate_clean_queries_examples = """
Example Definition:
Sentence: "HVAC systems rely heavily on electricity."
Purpose of the Scene: To visually represent the electrical nature of HVAC systems by highlighting electrical components within a typical unit.
Graphics Type: Illustration
The HVAC unit fades in at the start of the sentence.
As the narration emphasizes "rely heavily on electricity," all electrical components fade in simultaneously to emphasize their importance.
The electricity symbol animates, showing the flow of power into the unit.

Sentence: "Understanding electrical principles is not just helpful, it's essential for HVAC technicians to ensure safety, efficiency, and effective troubleshooting."
Purpose of the Scene: To illustrate the application of electrical knowledge by an HVAC technician while emphasizing safety, efficiency, and effective troubleshooting.
Graphics Type: Illustration
The technician and the HVAC unit fade in together.
As the narration mentions "ensure safety," the hard hat icon appears.
As the narration mentions "efficiency," the energy-saving light bulb icon appears.
As the narration mentions "effective troubleshooting," the wrench and voltmeter icon appears.
The multimeter display changes to indicate a reading, showing the technician actively troubleshooting.
Reusing Previous Graphics:
The HVAC unit from Scene 1 can be partially reused to maintain consistency.

Example Output:
HVAC unit
HVAC electrical components
the electricity symbol
technician illustration
hard hat icon illustration
energy-saving light bulb icon illustration
wrench icon illustration
voltmeter icon illustration
multimeter with display
"""


def generate_clean_query_from_scene(
    sentence: str,
    gtype: str,
    purpose: str,
    visuals: str,
    llm: str = "gemini_2_flash"
) -> Optional[str]:

    """
    Generates search queries for image retrieval from a graphics scene definition.
    """
    non_visual_keywords = ["learning objectives", "by the end", "introduction"]
    if any(kw in visuals.lower() or kw in purpose.lower() for kw in non_visual_keywords):
        return None

    agent = Chain(llm=llm, tags=['clean_query_generation'])

    formatted_prompt = generate_clean_queries_prompt_template.format(
        examples=generate_clean_queries_examples,
        sentence = sentence,
        gtype=gtype,
        purpose=purpose,
        visuals=visuals
    )

    agent.add_message(role='user', content=formatted_prompt)

    try:
        response = agent.run()
        if isinstance(response, dict) and "clean_query_generation" in response:
            return response["clean_query_generation"].strip()
        else:
            print("Unexpected response format:", response)
            return None

    except Exception as e:
        print(f"LLM failed: {e}")
        return None



def generate_queries_from_definition_text(definition_text: str, llm: str) -> str:
    scenes = parse_flat_graphics_definitions(definition_text)
    all_queries = []

    if not scenes:
        print("No valid scenes found in:")
        print(definition_text)
        return ""

    for i, scene in enumerate(scenes):
        if "sentence" not in scene:
            print(f"Scene {i} missing 'sentence': {scene}")
            continue

        try:
            sentence = scene.get("sentence", "").strip()
            gtype = scene.get("graphics_type", "").strip()
            purpose = scene.get("purpose", "").strip()
            visuals = scene.get("visuals", "").strip()

            if not (sentence and purpose and gtype):
                print(f"Incomplete scene at index {i}, skipping.")
                continue

            query_block = generate_clean_query_from_scene(sentence, gtype, purpose, visuals, llm=llm)
            if query_block:
                individual_queries = [q.strip() for q in query_block.splitlines() if q.strip()]
                all_queries.extend(individual_queries)

        except Exception as e:
            print(f"Error processing scene {i}: {e}")
            print("Scene content:", scene)

    return "\n".join(all_queries) if all_queries else ""



def run_generate_queries_from_definition(sheet, sheet_name, llm: str) -> list:
    slide_chunks_sheet, slide_chunks_df = get_sheet_data_and_df(sheet, sheet_name)

    queries_column = "generated_graphics_queries"
    if queries_column in slide_chunks_df.columns and slide_chunks_df[queries_column].notna().any():
        print("Queries already exist. Skipping generation.")
        return slide_chunks_df[queries_column].tolist()

    graphics_definitions = slide_chunks_df["short_graphics_definitions"].fillna("").tolist()
    all_clean_queries = [None] * len(graphics_definitions)

    def process_definition(index, definition_text):
        if not definition_text.strip():
            return index, ""
        clean_queries = generate_queries_from_definition_text(definition_text, llm=llm)
        return index, clean_queries

    futures_map = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        for idx, definition_text in enumerate(graphics_definitions):
            future = executor.submit(process_definition, idx, definition_text)
            futures_map[future] = idx

        total_tasks = len(futures_map)
        save_interval = 5
        progress = SmartProgressBar(total_tasks=total_tasks, description="Percent complete", save_interval=save_interval)

        for future in tqdm(as_completed(futures_map), total=total_tasks):
            try:
                idx, result = future.result()
                all_clean_queries[idx] = result

                # Update DataFrame and sheet
                slide_chunks_df.at[idx, queries_column] = result
                progress.update()

                if progress.should_save():
                    print(f"Saving partial progress after {progress.completed_count} rows.")
                    save_to_sheet(slide_chunks_sheet, slide_chunks_df)

                print(f"Processed and saved row {idx + 1} with {len(result.splitlines())} queries.")
            except Exception as e:
                print(f"Error processing row {futures_map[future]}: {e}")

    # Final save after all tasks
    print("All rows processed. Saving final DataFrame to sheet.")
    save_to_sheet(slide_chunks_sheet, slide_chunks_df)

    return all_clean_queries

# def run_generate_queries_from_definition(sheet, sheet_name, llm: str) -> list:
#     slide_chunks_sheet, slide_chunks_df = get_sheet_data_and_df(sheet, sheet_name)

#     queries_column = "generated_graphics_queries"
#     if queries_column in slide_chunks_df.columns and slide_chunks_df[queries_column].notna().any():
#         print("Queries already exist. Skipping generation.")
#         return slide_chunks_df[queries_column].tolist()

#     graphics_definitions = slide_chunks_df["short_graphics_definitions"].fillna("").tolist()
#     all_clean_queries = [None] * len(graphics_definitions)

#     def process_definition(index, definition_text):
#         if not definition_text.strip():
#             return index, ""
#         clean_queries = generate_queries_from_definition_text(definition_text, llm=llm)
#         return index, clean_queries

#     with st.spinner("⚙️ Generating queries from definitions..."):
#         futures_map = {}
#         with ThreadPoolExecutor(max_workers=5) as executor:
#             for idx, definition_text in enumerate(graphics_definitions):
#                 future = executor.submit(process_definition, idx, definition_text)
#                 futures_map[future] = idx

#             for future in as_completed(futures_map):
#                 try:
#                     idx, result = future.result()
#                     all_clean_queries[idx] = result

#                     # Save result to DataFrame and immediately to sheet
#                     slide_chunks_df.at[idx, queries_column] = result
#                     save_to_sheet(slide_chunks_sheet, slide_chunks_df)

#                     print(f"Processed and saved row {idx + 1} with {len(result.splitlines())} queries.")
#                 except Exception as e:
#                     print(f"Error processing row {futures_map[future]}: {e}")

#     return all_clean_queries





def load_queries_from_sheet(sheet, sheet_name, queries_column="generated_graphics_queries"):
    _, slide_chunks_df = get_sheet_data_and_df(sheet, sheet_name)
    query_data = []

    for cell in slide_chunks_df[queries_column].fillna(""):
        cell_str = cell.strip()
        query_data.append(cell_str if cell_str else "")

    return query_data




# def search_images_for_query_list(queries: List[str], query_image: Any, llm: str = "gemini_2_flash") -> List[str]:
#     """
#     Search for images based on a list of queries sequentially.
    
#     :param queries: List of search queries.
#     :param query_image: An optional query image to enhance the search.
#     :param llm: The LLM to use for generating or refining queries.
#     :return: List of image URLs (one for each query).
#     """
#     image_links = []

#     for query in queries:
#         try:
#             results = run_graphics_search_graph(
#                 query=query,
#                 query_image=query_image,
#                 drive=drive,  # Make sure `drive` is defined in scope
#                 k=5,
#                 llm=llm
#             )
#             if results:
#                 metadata = results[0].get("metadata", {})
#                 # Prefer drive_url > source_url > fallback
#                 url = metadata.get("drive_url") or metadata.get("source_url") or "No URL found"
#                 image_links.append(url)
#             else:
#                 image_links.append("No result")
#         except Exception as e:
#             image_links.append(f"Error: {str(e)}")

#     return image_links

def search_images_for_query_list(
    queries: List[str],
    query_image: Any,
    definition: str | None = None,
    llm: str = "gemini_2_flash",
) -> List[str]:
    """
    Search for images based on a list of queries, injecting contextual graphics definitions
    directly into the query string.
    """
    image_links = []

    for i, query in enumerate(queries):
        # # Inject the graphics context inline into the query
        # query_with_context = f"{query.strip()}\n\nGraphics Context: {graphics_context}" if graphics_context else query.strip()

        # print(f"\n[DEBUG] → Query {i + 1}/{len(queries)}: {query_with_context[:100]}...")

        try:
            results = run_graphics_search_graph(
                query=query,
                query_image=query_image,
                drive=drive,
                k=5,
                definition=definition,
                llm=llm
            )

            if results:
                metadata = results[0].get("metadata", {})
                url = metadata.get("source_url") or metadata.get("drive_url") or "No URL found"
                print(f"[DEBUG] Got result URL: {url}")
                image_links.append(url)
            else:
                print("[DEBUG] No results returned.")
                image_links.append("No result")

        except Exception as e:
            print(f"[ERROR] Exception while processing query '{query}': {e}")
            image_links.append(f"Error: {str(e)}")

    return image_links


def run_search_images_for_query_list(sheet, sheet_name, llm="gemini_2_flash", k=4):
    """
    Run image search for each newline-separated query in a sheet row.
    Each individual query is processed and saved immediately.
    Previously processed queries are skipped.
    """

    slide_chunks_sheet, slide_chunks_df = get_sheet_data_and_df(sheet, sheet_name)
    all_query_blocks = load_queries_from_sheet(sheet, sheet_name)

    pair_column = "image_urls"
    if pair_column not in slide_chunks_df.columns:
        slide_chunks_df[pair_column] = ""

    # Flatten all queries with row context
    task_list = []
    for row_index, query_block in enumerate(all_query_blocks):
        if not isinstance(query_block, str) or not query_block.strip():
            continue

        existing_links_str = slide_chunks_df.at[row_index, pair_column]
        existing_results = {}
        if isinstance(existing_links_str, str) and existing_links_str.strip():
            for line in existing_links_str.strip().split("\n"):
                if " - " in line:
                    q, link = line.split(" - ", 1)
                    existing_results[q.strip()] = link.strip()

        query_list = [q.strip() for q in query_block.strip().split("\n") if q.strip()]
        if not query_list:
            continue

        definition = slide_chunks_df.at[row_index, "short_graphics_definitions"] if "short_graphics_definitions" in slide_chunks_df.columns else ""

        for query in query_list:
            if query in existing_results and existing_results[query].startswith("http"):
                continue
            task_list.append((row_index, query, query_block, definition))

    total_tasks = len(task_list)
    if total_tasks == 0:
        print("All queries already processed. Nothing to do.")
        return slide_chunks_df[pair_column].tolist()

    print(f"Starting image search for {total_tasks} queries...")

    # Progress bar
    save_interval = 10
    progress = SmartProgressBar(total_tasks=total_tasks, description="Percent complete", save_interval=save_interval)

    # Use 3 threads for better reliability
    futures_map = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        for row_index, query, query_block, definition in task_list:
            # Introduce a small staggered delay inside the function to reduce API pressure
            def task_fn(query=query, definition=definition):
                time.sleep(random.uniform(0.8, 1.6))
                return search_images_for_query_list(
                    [query],
                    query_image=None,
                    definition=definition,
                    llm=llm
                )

            future = executor.submit(task_fn)
            futures_map[future] = (row_index, query, query_block)

        for future in tqdm(as_completed(futures_map), total=total_tasks):
            try:
                row_index, query, query_block = futures_map[future]
                image_links = future.result()
                link = image_links[0] if image_links else "No result"

                # Update results while preserving query order
                existing_links_str = slide_chunks_df.at[row_index, pair_column]
                updated_results = {}
                if existing_links_str:
                    for line in existing_links_str.strip().split("\n"):
                        if " - " in line:
                            q, l = line.split(" - ", 1)
                            updated_results[q.strip()] = l.strip()

                updated_results[query] = link if isinstance(link, str) and link.startswith("http") else "No result"

                # Preserve original order
                query_list = [q.strip() for q in query_block.strip().split("\n") if q.strip()]
                updated_lines = [f"{q} - {updated_results.get(q, 'No result')}" for q in query_list]
                slide_chunks_df.at[row_index, pair_column] = "\n".join(updated_lines)

                # Save to sheet
                save_to_sheet(slide_chunks_sheet, slide_chunks_df)

                print(f"Processed query: '{query}' in row {row_index + 1}")

                progress.update()
                if progress.should_save():
                    print(f"Saving intermediate results after {progress.completed_count} queries.")
                    save_to_sheet(slide_chunks_sheet, slide_chunks_df)

            except Exception as e:
                row_index, query, _ = futures_map[future]
                print(f"Error processing query '{query}' (row {row_index + 1}): {e}")

    print("All queries processed. Saving final results.")
    save_to_sheet(slide_chunks_sheet, slide_chunks_df)

    return slide_chunks_df[pair_column].tolist()



def delete_generated_graphics_queries(sheet, worksheet_name="Slide Chunks"):
    """Remove graphics definition related columns from the worksheet."""
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    cols = [
        "generated_graphics_queries",
    ]
    cols = [c for c in cols if c in df.columns]
    if cols:
        df = df.drop(columns=cols)
        clear_worksheet(ws)
        save_to_sheet(ws, df)


def delete_retrieved_image_urls(sheet, worksheet_name="Slide Chunks"):
    """Remove graphics definition related columns from the worksheet."""
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    cols = [
        "image_urls",
    ]
    cols = [c for c in cols if c in df.columns]
    if cols:
        df = df.drop(columns=cols)
        clear_worksheet(ws)
        save_to_sheet(ws, df)


