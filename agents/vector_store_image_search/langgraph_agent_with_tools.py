
from __future__ import annotations

from typing import Dict, Any, List, TypedDict
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
    filters: Dict[str, Any]
    results: List[Dict[str, Any]]
    images: List[Dict[str, Any]]
    verdict: str
    turn: int
    used_vector_search: bool
    used_web_search: bool
    should_terminate: bool

# --- Tool functions ---
@tool
def run_vector_tool(query: str, drive: Any, k: int) -> List[Dict[str, Any]]:
    """Search for images using vector search and return top k results."""
    return graphics_retriever(query=query, drive=drive, k=k)

@tool
def run_web_tool(query: str, k: int) -> List[Dict[str, Any]]:
    """Search for images using web search and return top k results."""
    return web_image_search_tool(query=query, k=k)

# --- Custom Tool Node ---
def custom_tool_node(state: SearchState) -> SearchState:
    """Node to route and execute tools based on state conditions"""
    # Determine which tool to use
    if not state.get("used_vector_search", False):
        tool_name = "run_vector_tool"
        tool_input = {
            "query": state["query"],
            "drive": state["drive"],
            "k": state["k"]
        }
        state["used_vector_search"] = True
    elif not state.get("used_web_search", False):
        tool_name = "run_web_tool"
        tool_input = {"query": state["query"], "k": state["k"]}
        state["used_web_search"] = True
    else:
        return state  

    # Execute the selected tool
    tools = {"run_vector_tool": run_vector_tool, "run_web_tool": run_web_tool}
    state["results"] = tools[tool_name].invoke(tool_input)
    return state



# --- Agent Node ---
def agent_node(state: SearchState) -> SearchState:
    # Placeholder logic - replace with your agent reasoning logic
    print("[AGENT] Running agent logic...")
    if not state.get("used_vector_search", False):
        results = graphics_retriever_agent(
            query=state["query"],
            drive=state["drive"],
            llm=state.get("llm", "gemini_2_flash"),
            k=state.get("k", 5),
            filters=state.get("filters"),
            max_turns=1,
            verbose=True
        )
        state["images"] = results
        state["used_vector_search"] = True
        if results:
            state["should_terminate"] = True
        else:
            state["verdict"] = "WEB_SEARCH"
        return state
    else:
        results = state.get("results", [])
        if not results:
            state["verdict"] = "TERMINATE"
            return state
        
        chain = Chain(
            llm=state.get("llm", "gemini_2_flash"),
            tags=["observations", "verdict", "selected_indexes", "action", "query"],
            use_xml_checker=True
        )
        chain.add_message(role="system", content=graphics_retriever_agent_prompt)
        formatted_prompt = image_relevance_prompt.format(query=state["query"])
        content_parts = prepare_images_for_llm(results, formatted_prompt=formatted_prompt)
        if isinstance(content_parts, list):
            flat_content = "\n".join(str(part) for part in content_parts)
        else:
            flat_content = str(content_parts)

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
            state["should_terminate"] = True
        else:
            state["query"] = llm_response.get("query", "").strip()

        return state

# --- Decision Node ---
def decision_node(state: SearchState) -> SearchState:
    if state.get("should_terminate", False):
        return state
    verdict = state.get("verdict", "").upper()
    state["turn"] = state.get("turn", 1) + 1
    if state["turn"] > state.get("max_turns", 5):
        state["should_terminate"] = True
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
    k: int = 5,
    *,
    llm: str = "gemini_2_flash",
    max_turns: int = 3,
    filters: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    memory = MemorySaver()

    # Rebuild and compile the graph with memory
    app = build_graph()
    # Initial state
    initial_state = {
        "query": query,
        "drive": drive,
        "k": k,
        "llm": llm,
        "filters": filters,
        "max_turns": max_turns,
        "turn": 1,
    }

    final_state = app.invoke(initial_state)
    return final_state.get("images", [])

