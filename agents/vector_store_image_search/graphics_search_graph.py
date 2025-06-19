from __future__ import annotations

from typing import Dict, Any, List

from langgraph.graph import Graph, END

from modules.chain import Chain
from agents.vector_store_image_search.graphics_retriever import graphics_retriever
from agents.vector_store_image_search.graphics_retriever_agent import (
    graphics_retriever_agent_prompt,
    image_relevance_prompt,
)


def parse_selected_indexes(index_string: str) -> List[int]:
    """Safely parse a comma-separated list of indexes."""
    try:
        return [int(i.strip()) for i in index_string.split(",") if i.strip().isdigit()]
    except Exception:
        return []


# --- Graph Nodes ---

def query_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Entry node – simply return the state."""
    return state


def retriever_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve images using the current query."""
    results = graphics_retriever(
        query=state["query"],
        drive=state["drive"],
        k=state.get("k", 5),
        filters=state.get("filters"),
    )
    state["results"] = results
    return state


def agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate retrieved images and optionally refine the query."""
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


def decision_node(state: Dict[str, Any]) -> str:
    """Decide whether to end or continue searching."""
    if state.get("verdict") == "TERMINATE" or state.get("turn", 1) >= state.get("max_turns", 3):
        return "end"
    state["turn"] = state.get("turn", 1) + 1
    return "continue"


# --- Graph Runner ---

def build_graph() -> Graph:
    """Construct the LangGraph pipeline."""
    graph = Graph()
    graph.add_node("query", query_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("agent", agent_node)
    graph.add_node("decision", decision_node)

    graph.add_edge("query", "retriever")
    graph.add_edge("retriever", "agent")
    graph.add_edge("agent", "decision")
    graph.add_conditional_edges(
        "decision",
        {
            "continue": "retriever",
            "end": END,
        },
    )
    graph.set_entry_point("query")

    return graph.compile()


def run_graphics_search_graph(
    query: str,
    drive: Any,
    k: int = 5,
    *,
    llm: str = "gemini_2_flash",
    max_turns: int = 3,
    filters: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Execute the graphics search graph and return selected images."""
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