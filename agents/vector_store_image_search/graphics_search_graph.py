from __future__ import annotations

from typing import Dict, Any, List, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from modules.chain import Chain
from agents.vector_store_image_search.graphics_retriever import graphics_retriever
from agents.vector_store_image_search.graphics_retriever_agent import (
    graphics_retriever_agent_prompt,
    image_relevance_prompt,
)



# --- Define state type schema ---
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


# --- Helper to parse index strings ---
def parse_selected_indexes(index_string: str) -> List[int]:
    try:
        return [int(i.strip()) for i in index_string.split(",") if i.strip().isdigit()]
    except Exception:
        return []


# --- Nodes ---
def start_node(state: SearchState) -> SearchState:
    return state


def retriever_node(state: SearchState) -> SearchState:
    results = graphics_retriever(
        query=state["query"],
        drive=state["drive"],
        k=state.get("k", 5),
        filters=state.get("filters"),
    )
    state["results"] = results
    return state


def agent_node(state: SearchState) -> SearchState:
    query = state["query"]
    results = state.get("results", [])
    llm = state.get("llm", "gemini_2_flash")

    chain = Chain(
        llm=llm,
        tags=["observations", "verdict", "selected_indexes", "action", "query"],
        use_xml_checker=True,
    )
    chain.add_message(role="system", content=graphics_retriever_agent_prompt)

    formatted_prompt = image_relevance_prompt.format(query=query)
    flat_prompt = formatted_prompt + "\n\n"
    for i, item in enumerate(results):
        meta = item.get("metadata", {})
        flat_prompt += f"Image {i}: {meta.get('image_title', 'Untitled')} — {meta.get('description', '')}\n"

    chain.add_message(role="user", content=flat_prompt)
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
                "metadata": results[i].get("metadata", {}),
            }
            for i in selected_indexes
            if 0 <= i < len(results)
        ]
    else:
        state["query"] = llm_response.get("query", "").strip()

    return state


def decision_node(state):
    # Example: terminate after 3 turns, or if a flag is set
    if state.get("turn", 1) >= state.get("max_turns", 3) or state.get("should_terminate", False):
        return {"should_terminate": True}
    return {"turn": state.get("turn", 1) + 1}

def should_continue(state):
    if state.get("should_terminate", False):
        return "end"
    return "continue"

graph = StateGraph(SearchState)
# ... add nodes ...
graph.add_conditional_edges(
    "decision",
    should_continue,
    {
        "continue": "retriever",
        "end": END,
    },
)


# --- Build LangGraph ---


def build_graph():
    graph = StateGraph(SearchState)
    
    graph.add_node("start", RunnableLambda(start_node))
    graph.add_node("retriever", RunnableLambda(retriever_node))
    graph.add_node("agent", RunnableLambda(agent_node))
    graph.add_node("decision", RunnableLambda(decision_node))  # Updated node
    
    graph.add_edge("start", "retriever")
    graph.add_edge("retriever", "agent")
    graph.add_edge("agent", "decision")
    
    # Fixed conditional edges
    graph.add_conditional_edges(
        "decision",
        should_continue,  # Uses the new function
        {
            "continue": "retriever",
            "end": END,
        },
    )
    
    graph.set_entry_point("start")
    return graph.compile()



# --- Run the graph ---
def run_graphics_search_graph(
    query: str,
    drive: Any,
    k: int = 5,
    *,
    llm: str = "gemini_2_flash",
    max_turns: int = 3,
    filters: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    app = build_graph()

    final_state = app.invoke({
        "query": query,
        "drive": drive,
        "k": k,
        "llm": llm,
        "filters": filters,
        "max_turns": max_turns,
        "turn": 1,
    })

    return final_state.get("images", [])


def visualize_graph():
    app = build_graph()
    app.draw("graphics_search_graph", format="png")  # Creates graphics_search_graph.png
    print("✅ Graph drawn to graphics_search_graph.png")
