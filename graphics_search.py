from agent_ui_template import agent_ui

from agents.graphics_search.image_search_in_drive import download_images_with_empty_description


# --- 1) Define pipeline as sections, each with its own steps ---
pipeline_sections = [
    {
        "section_name": "Section 1: Image Search in Drive",
        "steps": [
            {
                "name": "Image Search",
                "func": download_images_with_empty_description,
                "depends_on": [],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Folders Searched",
                    "folder_id": "1JqKyF18i7ug300rARo4w9HLiYrI4gGX6",
                    "drive": "drive",
                    "download_path": "/tmp/graphics",
                },
                "estimated_time": "~ 10 - 20 minutes",
                "description": "Downloads images from Drive with empty descriptions.",
            },
        ],
    },
]


agent_ui(step_name = "Graphics Search", pipeline_sections = pipeline_sections)
