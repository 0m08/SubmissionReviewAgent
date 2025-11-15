"""
Search Agent (Level 3)

ReAct agent that finds and validates visual references through iterative search refinement.
"""

from typing import Dict, Any, List
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

from agents.graphics_workflow_v2.state.schemas import SearchAgentState
from agents.graphics_workflow_v2.config.settings import (
    SEARCH_AGENT_MODEL,
    SEARCH_AGENT_RECURSION_LIMIT,
    MAX_SEARCH_ITERATIONS,
    MIN_REFERENCES_PER_SEGMENT,
)
from agents.graphics_workflow_v2.tools.search_agent.search_tools import (
    generate_search_queries,
    execute_image_search,
    check_reference_reuse,
    evaluate_search_results,
    finalize_search,
)
from agents.graphics_workflow_v2.prompts.search_agent_prompts import (
    SEARCH_AGENT_SYSTEM_PROMPT,
    get_search_agent_prompt,
)


def create_search_agent():
    """
    Creates the Search Agent ReAct agent.

    Returns:
        Compiled LangGraph agent
    """
    # Initialize LLM
    llm = init_chat_model(SEARCH_AGENT_MODEL)

    # Create system prompt
    system_prompt = SEARCH_AGENT_SYSTEM_PROMPT.format(
        max_search_iterations=MAX_SEARCH_ITERATIONS,
        min_references_needed=MIN_REFERENCES_PER_SEGMENT,
    )

    # Create agent with tools
    agent = create_react_agent(
        model=llm,
        tools=[
            generate_search_queries,
            execute_image_search,
            check_reference_reuse,
            evaluate_search_results,
            finalize_search,
        ],
        state_schema=SearchAgentState,
        prompt=system_prompt,
    )

    return agent


def run_search_agent(
    graphics_definition: str,
    visual_elements: List[str],
    available_references: Dict[str, Any] = None,
    drive: Any = None,
    filters: Dict[str, Any] = None,
    root_folder_id: str = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run the Search Agent to find references for a graphics definition.

    Args:
        graphics_definition: The graphics definition text
        visual_elements: List of visual elements that need references
        available_references: References from previous segments (for reuse)
        drive: Google Drive instance for search
        filters: Optional filters for search
        root_folder_id: Root folder ID for vector store
        **kwargs: Additional configuration

    Returns:
        Final state with selected_references
    """
    # Create agent
    agent = create_search_agent()

    # Prepare initial state
    initial_state = {
        "messages": [
            {
                "role": "user",
                "content": get_search_agent_prompt(
                    graphics_definition=graphics_definition,
                    visual_elements=visual_elements,
                    available_references=available_references or {},
                    max_iterations=MAX_SEARCH_ITERATIONS,
                    min_references=MIN_REFERENCES_PER_SEGMENT,
                ),
            }
        ],
        "graphics_definition": graphics_definition,
        "visual_elements": visual_elements,
        "available_references": available_references or {},
        "search_queries": [],
        "search_results": [],
        "selected_references": [],
        "search_iteration": 0,
        "max_search_iterations": MAX_SEARCH_ITERATIONS,
        "min_references_needed": kwargs.get("min_references", len(visual_elements)),
        # Pass through search parameters
        "drive": drive,
        "filters": filters,
        "root_folder_id": root_folder_id,
        "search_k": kwargs.get("search_k", 10),
    }

    # Run agent
    final_state = agent.invoke(
        initial_state,
        config={"recursion_limit": SEARCH_AGENT_RECURSION_LIMIT}
    )

    return final_state
