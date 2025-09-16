
from __future__ import annotations

from typing import Dict, Any, List, TypedDict, Optional
from PIL import Image
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from agents.vector_store_image_search.web_image_search_tool import web_image_search_tool
from agents.vector_store_image_search.graphics_retriever import graphics_retriever
from agents.vector_store_image_search.graphics_retriever_agent import graphics_retriever_agent, graphics_retriever_agent_prompt, image_relevance_prompt, parse_selected_indexes, prepare_images_for_llm
from langgraph.checkpoint.memory import MemorySaver
from modules.chain import Chain


# --- Define state ---
class SearchState(TypedDict, total=False):
    query: str
    drive: Any
    k: int
    llm: str
    query_image: Any
    filters: Dict[str, Any]
    definition: Optional[str]
    results: List[Dict[str, Any]]
    images: List[Dict[str, Any]]
    verdict: str
    turn: int
    used_vector_search: bool
    used_web_search: bool
    should_terminate: bool
    root_folder_id: str

# --- Tool functions ---
@tool
def run_vector_tool(query: str, drive: Any, k: int, query_image: Any = None,
                    filters: Dict[str, Any] | None = None,
                    root_folder_id: str = '1QS6PmCESfgFWNNEpRJDUB0E-t8iMAatH') -> List[Dict[str, Any]]:
    """Search for images using vector search and return top k results."""
    return graphics_retriever(query=query, query_image=query_image, drive=drive, k=k,
                               filters=filters, root_folder_id=root_folder_id)

@tool
def run_web_tool(query: Optional[str] = None, query_image: Optional[Image.Image] = None, k: int = 5) -> List[Dict[str, Any]]:
    """Search for images using web search. Text query is required. Image is optional for refinement."""
    return web_image_search_tool(query=query, query_image=query_image, k=k)


# --- Custom Tool Node ---
def custom_tool_node(state: SearchState) -> SearchState:
    """Node to route and execute tools based on state conditions"""
    # Determine which tool to use
    if not state.get("used_vector_search", False):
        tool_name = "run_vector_tool"
        tool_input = {
            "query": state["query"],
            "query_image": state.get("query_image"),
            "drive": state["drive"],
            "k": state["k"],
            "filters": state.get("filters"),
            "root_folder_id": state.get("root_folder_id"),
        }
        state["used_vector_search"] = True
    elif not state.get("used_web_search", False) and state.get("query"):
        tool_name = "run_web_tool"
        tool_input = {
            "query": state["query"],
            "query_image": state.get("query_image"),
            "k": state["k"]
        }
        state["used_web_search"] = True
    else:
        return state

    # Execute the selected tool
    tools = {"run_vector_tool": run_vector_tool, "run_web_tool": run_web_tool}
    state["results"] = tools[tool_name].invoke(tool_input)
    return state


# --- Agent Node ---
def agent_node(state: SearchState) -> SearchState:
    print("[AGENT] Running agent logic...")

    query = (state.get("query") or "").strip()
    query_image = state.get("query_image")

    if not query and query_image is not None:
        query = "Image-based search input from user"
        state["query"] = query

    # First-time vector search using agent
    if not state.get("used_vector_search", False):
        print("[AGENT] Running first vector search...")
        results = graphics_retriever_agent(
            query=query,
            query_image=query_image,
            drive=state["drive"],
            llm=state.get("llm", "gemini_2_flash"),
            k=state.get("k", 5),
            filters=state.get("filters"),
            max_turns=1,
            verbose=True,
            definition=state.get("definition"),
            root_folder_id=state.get("root_folder_id"),
        )
        state["images"] = results
        state["used_vector_search"] = True

        if results:
            state["verdict"] = "TERMINATE"
            return state
        else:
            # Let the tool node attempt vector search
            state["verdict"] = "CONTINUE"
            return state

    results = state.get("results", [])

    # If no results and web search not yet used, fallback
    if not results and not state.get("used_web_search", False):
        print("[AGENT] No results after requery → using web search...")
        try:
            web_results = web_image_search_tool(
                query=query,
                query_image=query_image,
                k=state.get("k", 1)
            )
            state["images"] = [{
                "image": res["image"],
                "metadata": res.get("metadata", {})
            } for res in web_results] if web_results else []
        except Exception as e:
            print(f"[ERROR] Web search failed: {e}")
            state["images"] = []

        state["used_web_search"] = True
        state["verdict"] = "TERMINATE"
        return state

    elif not results:
        print("[AGENT] No results found. Terminating.")
        state["images"] = []
        state["verdict"] = "TERMINATE"
        return state

    # Evaluate results with LLM
    print("[AGENT] Evaluating results with LLM...")
    chain = Chain(
        llm=state.get("llm", "gemini_2_flash"),
        tags=["observations", "verdict", "selected_indexes", "action", "query"],
        use_xml_checker=True
    )
    chain.add_message(role="system", content=graphics_retriever_agent_prompt)

    formatted_prompt = image_relevance_prompt.format(
        query=state["query"],
        definition=state.get("definition", ""),
    )
    content_parts = prepare_images_for_llm(results, formatted_prompt=formatted_prompt)
    flat_content = "\n".join(map(str, content_parts)) if isinstance(content_parts, list) else str(content_parts)
    chain.add_message(role="user", content=flat_content)

    llm_raw = chain.run()
    content = llm_raw.get("content") if isinstance(llm_raw, dict) else llm_raw
    llm_response = chain.extract_text_in_tags(content)

    verdict = llm_response.get("verdict", "").strip().upper()
    state["verdict"] = verdict

    if verdict == "TERMINATE":
        selected_indexes = parse_selected_indexes(llm_response.get("selected_indexes", ""))
        state["images"] = [
            {
                "image": results[i]["image"],
                "metadata": results[i].get("metadata", {})
            }
            for i in selected_indexes if 0 <= i < len(results)
        ]
    else:
        # Requery path
        updated_query = llm_response.get("query", "").strip()
        if not updated_query or updated_query.lower() == "none":
            updated_query = "Image-based query"

        print(f"[AGENT] Requerying with: {updated_query}")
        state["query"] = updated_query
        state["results"] = []
        state["verdict"] = "CONTINUE"
        state["used_vector_search"] = False  # Allow vector search tool to run again

    return state



# --- Decision Node ---
def decision_node(state: SearchState) -> SearchState:
    verdict = state.get("verdict", "").upper()
    turn = state.get("turn", 1)
    state["turn"] = turn + 1

    if turn >= state.get("max_turns", 5):
        state["should_terminate"] = True
    elif verdict == "TERMINATE" and state.get("images"):
        state["should_terminate"] = True
    else:
        state["should_terminate"] = False

    return state

# --- Conditional Routing ---
def should_continue(state: SearchState) -> str:
    if state.get("should_terminate", False):
        return "end"
    if state.get("verdict", "").upper() in {"WEB_SEARCH", "VECTOR_SEARCH", "CONTINUE"}:
        return "tools"
    return "end"

# --- Build Graph ---
def build_graph():
    graph = StateGraph(SearchState)

    # Add nodes
    graph.add_node("agent", RunnableLambda(agent_node))
    graph.add_node("decision", RunnableLambda(decision_node))
    graph.add_node("tools", RunnableLambda(custom_tool_node))  # Use custom tool node

    graph.set_entry_point("agent")
    graph.add_edge("agent", "decision")
    graph.add_conditional_edges("decision", should_continue, {
        "tools": "tools",
        "end": END
    })
    graph.add_edge("tools", "agent")

    return graph.compile()



# --- Run the memory-enabled graph ---
def run_graphics_search_graph(
    query: str,
    drive: Any,
    query_image: Any,
    k: int = 5,
    *,
    llm: str = "gemini_2_flash",
    max_turns: int = 3,
    filters: Dict[str, Any] | None = None,
    definition: str | None = None,
    root_folder_id: str | None = None,
) -> List[Dict[str, Any]]:
    memory = MemorySaver()
    app = build_graph().with_config(checkpointer=memory)

    initial_state = {
        "query": query,
        "query_image": query_image,
        "drive": drive,
        "k": k,
        "llm": llm,
        "filters": filters,
        "definition": definition,
        "max_turns": max_turns,
        "turn": 1,
        "root_folder_id": root_folder_id,
    }

    final_state = app.invoke(initial_state)
    return final_state.get("images", [])

