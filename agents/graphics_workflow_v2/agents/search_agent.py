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
    MIN_REFERENCES_PER_SEGMENT,
    SEARCH_K,
)
from agents.graphics_workflow_v2.tools.search_agent.search_tools import (
    generate_search_queries,
    execute_image_search,
    execute_web_image_search,
    check_reference_reuse,
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
    # Note: GPT-5 models only support temperature=1
    llm = init_chat_model(SEARCH_AGENT_MODEL, temperature=1)

    # Create system prompt (no formatting needed - no placeholders)
    system_prompt = SEARCH_AGENT_SYSTEM_PROMPT

    # Create agent with tools
    agent = create_react_agent(
        model=llm,
        tools=[
            generate_search_queries,
            execute_image_search,
            execute_web_image_search,
            check_reference_reuse,
        ],
        state_schema=SearchAgentState,
        prompt=system_prompt,
    )

    return agent


def run_search_agent(
    vo_text: str,
    slide_chunk: str,
    available_references: Dict[str, Any] = None,
    drive: Any = None,
    filters: Dict[str, Any] = None,
    root_folder_id: str = None,
    course_context: Dict[str, Any] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run the Search Agent to find references for a voiceover sentence.

    Args:
        vo_text: The voiceover sentence that needs visualization
        slide_chunk: Full slide content (for context)
        available_references: References from previous segments (for reuse)
        drive: Google Drive instance for search
        filters: Optional filters for search
        root_folder_id: Root folder ID for vector store
        course_context: Optional course context
        **kwargs: Additional configuration

    Returns:
        Final state with selected_references
    """
    print(f"\n{'🔷'*40}")
    print(f"🔍 SEARCH AGENT - Starting Reference Search")
    print(f"{'🔷'*40}")
    print(f"🎙️  VO sentence: \"{vo_text}\"")
    print(f"📄 Slide context: {len(slide_chunk)} chars")
    print(f"♻️  Available references for reuse: {len(available_references or {})}")
    if available_references:
        for ref_id, ref in available_references.items():
            print(f"      └─ [{ref_id}] {ref.get('title', 'Untitled')}")
    print(f"{'🔷'*40}\n")
    
    # Create agent
    agent = create_search_agent()
    print(f"   ✅ Search Agent created")

    # Prepare initial state
    initial_state = {
        "messages": [
            {
                "role": "user",
                "content": get_search_agent_prompt(
                    vo_text=vo_text,
                    slide_chunk=slide_chunk,
                    available_references=available_references or {},
                    min_references=MIN_REFERENCES_PER_SEGMENT,
                ),
            }
        ],
        "vo_text": vo_text,
        "slide_chunk": slide_chunk,
        "available_references": available_references or {},
        "search_queries": [],
        "executed_drive_queries": set(),  # Track executed Drive queries to prevent duplicates
        "executed_web_queries": set(),  # Track executed web queries to prevent duplicates
        "search_results": [],
        "selected_references": [],
        "min_references_needed": kwargs.get("min_references", MIN_REFERENCES_PER_SEGMENT),
        # Pass through search parameters
        "drive": drive,
        "filters": filters,
        "root_folder_id": root_folder_id,
        "search_k": kwargs.get("search_k", SEARCH_K),
        "course_context": course_context,
    }

    # Run agent
    recursion_limit = kwargs.get("recursion_limit", SEARCH_AGENT_RECURSION_LIMIT)
    print(f"   🔎 Running Search Agent...")
    print(f"      ⚙️  Recursion limit: {recursion_limit}")
    print(f"      ⚙️  Min references needed: {initial_state.get('min_references_needed')}\n")
    
    final_state = agent.invoke(
        initial_state,
        config={"recursion_limit": recursion_limit}
    )
    
    # Calculate results
    queries = final_state.get('search_queries', [])
    results = final_state.get('search_results', [])
    selected = final_state.get('selected_references', [])
    
    print(f"\n{'🔵'*40}")
    print(f"✅ SEARCH AGENT COMPLETE")
    print(f"{'🔵'*40}")
    print(f"📊 Search Statistics:")
    print(f"   📝 Queries generated: {len(queries)}")
    if queries:
        for i, q in enumerate(queries, 1):
            print(f"      {i}. {q}")
    print(f"   🔍 Raw search results: {len(results)}")
    print(f"   ✅ Selected references: {len(selected)}")
    
    if selected:
        print(f"\n   📎 Final Selected References:")
        for i, ref in enumerate(selected, 1):
            reused = " (♻️ reused)" if ref.get('reused_from_segment') is not None else ""
            print(f"      {i}. [{ref.get('reference_id', 'N/A')}] {ref.get('title', 'Untitled')} | score: {ref.get('relevance_score', 0):.2f}{reused}")
    
    print(f"{'🔵'*40}\n")

    return final_state
