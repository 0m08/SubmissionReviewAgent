from agent_ui_template import agent_ui

from agents.slide_chunks.research_notes_parsing import run_research_notes_parsing
from agents.slide_chunks.generate_los import run_learning_objectives_agent
from agents.slide_chunks.review_and_revise_checklist import run_checklist_review_and_revise
from agents.slide_chunks.ai_detection_review_revise import run_ai_detection_review_revise
from agents.slide_chunks.winston_ai_plagiarism_detection import run_plagiarism_detection
from agents.slide_chunks.winston_ai_detection_with_readability import run_ai_detection_with_readability

pipeline_sections = [
    {
        "section_name": "Section 1: Research Notes Parsing",
        "steps": [
            
            {
                "name": "Research Notes Parsing",
                "func": run_research_notes_parsing,
                "depends_on": [],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Research Notes",
                },
            
                "estimated_time": "10-20 minutes",
                "description": "This function parses the research notes from the `Research Notes` sheet and extracts slide chunks in the `Slide Chunks` sheet.",
            },
            
            {
                "name": "Generate Learning Objectives",
                "func": run_learning_objectives_agent,
                "depends_on": ['Research Notes Parsing'],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Slide Chunks",
                    "course_name": "course_name", 
                    "target_audience": "target_audience",
                    "llm":"gemini_2_flash"
                },
            
                "estimated_time": "< 1 minute",
                "description": "This function generates learning objectives based on the slide chunks extracted from the research notes.",
            },
            
            {
                "name": "Review and Revise Checklist",
                "func": run_checklist_review_and_revise,
                "depends_on": ['Generate Learning Objectives'],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Slide Chunks",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": "gemini_2_flash"
                },
            
                "estimated_time": "40-60 minutes",
                "description": "This function reviews and revises the slide chunks based on the review checklist to ensure quality and coherence.",
            },
            
            {
                "name": "AI Detection based Review and Revise Agents",
                "func": run_ai_detection_review_revise,
                "depends_on": ['Review and Revise Checklist'],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Slide Chunks",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": "gemini_2_flash"
                },
            
                "estimated_time": "30-50 minutes",
                "description": "This function reviews and revises the slide chunks based on AI detection to ensure that the content looks Human-generated and not AI-generated.",
            },
            
            {
                "name": "Winston.ai Plagiarism Check",
                "func": run_plagiarism_detection,
                "depends_on": ['AI Detection based Review and Revise Agents'],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Slide Chunks"
                },
            
                "estimated_time": "1-2 minutes",
                "description": "This function checks for plagiarism in the slide chunks using Winston.ai.",
            },
            
            {
                "name": "Winston.ai AI Detection Check with Readability Score based Reviser Agent",
                "func": run_ai_detection_with_readability,
                "depends_on": ['Winston.ai Plagiarism Check'],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Slide Chunks",
                    "course_name": "course_name",
                    "target_audience": "target_audience"
                },
            
                "estimated_time": "5-10 minutes",
                "description": "This function checks for AI detection and readability in the slide chunks using Winston.ai.",
            },
        
        
            
            
        ],
    },
]

agent_ui(step_name = "Slide Chunks", pipeline_sections = pipeline_sections)