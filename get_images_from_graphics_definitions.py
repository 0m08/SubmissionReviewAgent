from agent_ui_template import agent_ui

from agents.vector_store_image_search.get_similar_images_graphics_definitions import run_generate_queries_from_definition
from agents.vector_store_image_search.get_similar_images_graphics_definitions import run_search_images_for_query_list, delete_generated_graphics_queries, delete_retrieved_image_urls


pipeline_sections = [
    {
        "section_name": "Section 1: Generate Queries from Graphics Definitions",
        "steps": [
            
            {
                "name": "Generate Queries from Graphics Definitions",
                "func": run_generate_queries_from_definition,
                "depends_on": [],
                "args": {
                    "sheet": "sheet",
                    "sheet_name": "Slide Chunks",
                    "llm": "gemini_2_flash"
                },

                "delete_func": delete_generated_graphics_queries,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Slide Chunks",
                },

                "estimated_time": "10-20 minutes",
                "description": "This function generates queries from the graphics definitions in the `Slide Chunks` sheet and saves them in the same sheet.",
            },
            
            {
                "name": "Search Images for Generated Queries",
                "func": run_search_images_for_query_list,
                "depends_on": ['Generate Queries from Graphics Definitions'],
                "args": {
                    "sheet": "sheet",
                    "sheet_name": "Slide Chunks",
                    "k": 5,
                    "llm":"gemini_2_flash"
                },

                "delete_func": delete_retrieved_image_urls,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Slide Chunks",
                },
                

                "estimated_time": "< 1 minute",
                "description": "This function searches for images based on the generated queries in the `Slide Chunks` sheet and saves the results in the same sheet.",
            },
        ]
    },

        
        ]
agent_ui(step_name = "Image Search with Graphics Definitions", pipeline_sections = pipeline_sections)
