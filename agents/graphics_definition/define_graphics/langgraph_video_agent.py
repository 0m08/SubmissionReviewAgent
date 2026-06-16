# from __future__ import annotations

# import re
# import urllib.parse
# from typing import Any, Optional, TypedDict

# from langchain_core.runnables import RunnableLambda
# from langgraph.graph import END, StateGraph
# from langsmith import traceable
# from concurrent.futures import ThreadPoolExecutor, as_completed

# from services.video_clip_tools import (
#     adjust_video_clip_window,
#     compare_video_clips,
#     evaluate_video_clip,
# )
# from services.sheets_service import get_sheet_data_and_df, save_to_sheet, format_worksheet, clear_worksheet
# from services.smart_progress_bar import SmartProgressBar

# import streamlit as st

# __all__ = ["run_video_clip_identification_agent", "build_video_clip_graph"]

# graph_recursion_limit = 400


# class ClipCandidate(TypedDict, total=False):
#     clip_url: str
#     video_id: str
#     start_seconds: int
#     end_seconds: int
#     source: str
#     raw_url: str


# class CandidateEvaluation(TypedDict, total=False):
#     candidate: ClipCandidate
#     status: str
#     visual_description: str
#     relevance_mapping: str
#     recommendation: Optional[str]
#     evaluation_breakdown: str
#     raw_response: str
#     adjustments_applied: int


# class SentenceRecord(TypedDict, total=False):
#     text: str
#     index: int
#     candidates: list[ClipCandidate]
#     candidate_cursor: int
#     evaluations: list[CandidateEvaluation]
#     accepted_candidates: list[CandidateEvaluation]
#     comparison_pool: list[CandidateEvaluation]
#     comparison_result: Optional[dict[str, Any]]
#     status: str
#     final_block: str
#     final_selection: Optional[CandidateEvaluation]


# class VideoClipAgentState(TypedDict, total=False):
#     course_name: str
#     target_audience: str
#     topic_name: str
#     subtopic_name: str
#     slide_title: str
#     slide_chunk: str
#     llm: str
#     sentences: list[SentenceRecord]
#     candidate_pool: list[ClipCandidate]
#     current_sentence_index: Optional[int]
#     max_adjustments_per_candidate: int
#     used_clip_urls: list[str]
#     next_action: str
#     final_outputs: list[str]
#     final_video_clip: str
#     complete: bool


# def split_into_sentences(text):
#     """
#     Split a block of text into sentences using punctuation and newline boundaries.

#     :param text: The input text to split.
#     :return: A list of sentence strings.
#     """
#     if not isinstance(text, str):
#         return []
#     text = text.strip()
#     if not text:
#         return []
#     parts = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+|\n+", text) if segment.strip()]
#     if not parts:
#         return [text]
#     return parts


# def normalize_embed_url(video_id, start, end):
#     """
#     Build a normalized YouTube embed URL with start/end timestamps.

#     :param video_id: The YouTube video identifier.
#     :param start: Start time in seconds.
#     :param end: End time in seconds.
#     :return: Normalized embed URL string.
#     """
#     base = f"https://www.youtube.com/embed/{video_id}?start={start}"
#     if end:
#         base += f"&end={end}"
#     return base


# def extract_grouped_sentence_urls(raw_text):
#     """
#     Parse grouped text in the form "Sentence N:" with subsequent URL lines.

#     :param raw_text: Multiline text potentially containing grouped sentence URL sections.
#     :return: Mapping of 1-based sentence index to list of URLs, or None if not present.
#     """
#     if not isinstance(raw_text, str) or not raw_text.strip():
#         return None

#     header_re = re.compile(r"^\s*Sentence\s+(\d+)\s*:?\s*$", re.IGNORECASE)
#     url_re = re.compile(r"^\s*(https?://\S+)\s*$", re.IGNORECASE)

#     mapping: dict[int, list[str]] = {}
#     current_idx: Optional[int] = None

#     for line in raw_text.splitlines():
#         header_match = header_re.match(line)
#         if header_match:
#             current_idx = int(header_match.group(1))
#             mapping.setdefault(current_idx, [])
#             continue

#         url_match = url_re.match(line)
#         if url_match and current_idx is not None:
#             url = url_match.group(1).strip()
#             if url and url not in mapping[current_idx]:
#                 mapping[current_idx].append(url)

#     return mapping if any(mapping.values()) else None


# def parse_candidate_urls(raw_candidates, source_label="slide_chunk_videos"):
#     """
#     Convert raw candidate URL inputs into normalized `ClipCandidate` entries.

#     :param raw_candidates: String, list, or other representing clip URLs.
#     :param source_label: Source identifier for provenance.
#     :return: List of parsed `ClipCandidate` dicts.
#     """
#     if raw_candidates is None:
#         return []

#     if isinstance(raw_candidates, str):
#         tokens = re.split(r"[\n\r;,]+", raw_candidates)
#     elif isinstance(raw_candidates, list):
#         tokens = raw_candidates
#     else:
#         tokens = [str(raw_candidates)]

#     candidates: list[ClipCandidate] = []
#     seen = set()

#     for token in tokens:
#         url = str(token).strip()
#         if not url:
#             continue
#         parsed = urllib.parse.urlparse(url)
#         video_id = parsed.path.rstrip("/").split("/")[-1] or ""
#         if not video_id:
#             continue

#         qs = urllib.parse.parse_qs(parsed.query)
#         try:
#             start = int(qs.get("start", [0])[0])
#         except (ValueError, TypeError):
#             start = 0

#         end_value = qs.get("end", [None])[0]
#         try:
#             end = int(end_value) if end_value is not None else None
#         except (ValueError, TypeError):
#             end = None

#         normalized_url = normalize_embed_url(video_id, start, end)
#         if normalized_url in seen:
#             continue
#         seen.add(normalized_url)

#         candidates.append(
#             ClipCandidate(
#                 clip_url=normalized_url,
#                 video_id=video_id,
#                 start_seconds=start,
#                 end_seconds=end,
#                 source=source_label,
#                 raw_url=url,
#             )
#         )

#     return candidates


# def ensure_sentence_records_from_grouped(sentences, grouped):
#     """
#     Build per-sentence records using a grouped mapping of URLs.

#     :param sentences: List of sentence strings.
#     :param grouped: Mapping of 1-based sentence index to URL list.
#     :return: List of initialized `SentenceRecord` items.
#     """
#     sentence_records: list[SentenceRecord] = []
#     for idx, sentence_text in enumerate(sentences):
#         urls_for_sentence = grouped.get(idx + 1, [])
#         candidates = parse_candidate_urls(urls_for_sentence, source_label="slide_chunk_videos")
#         sentence_records.append(
#             SentenceRecord(
#                 text=sentence_text,
#                 index=idx,
#                 candidates=candidates,
#                 candidate_cursor=0,
#                 evaluations=[],
#                 accepted_candidates=[],
#                 comparison_pool=[],
#                 comparison_result=None,
#                 status="pending",
#                 final_block="",
#                 final_selection=None,
#             )
#         )
#     return sentence_records


# def filter_available_candidates(sentence, used_urls):
#     """
#     Filter out accepted candidates whose clip URLs are already used elsewhere.

#     :param sentence: The sentence record containing accepted candidates.
#     :param used_urls: URLs already used in prior selections.
#     :return: List of available `CandidateEvaluation` entries.
#     """

#     used = set(used_urls or [])
#     return [
#         evaluation
#         for evaluation in sentence.get("accepted_candidates", [])
#         if evaluation.get("candidate", {}).get("clip_url") not in used
#     ]


# def lookup_candidate_by_url(sentence, clip_url):
#     """
#     Find an accepted candidate evaluation by its normalized clip URL.

#     :param sentence: The sentence record with accepted candidates.
#     :param clip_url: Normalized URL to search for.
#     :return: Matching `CandidateEvaluation` or None.
#     """

#     for evaluation in sentence.get("accepted_candidates", []):
#         candidate = evaluation.get("candidate", {})
#         if normalize_embed_url(candidate["video_id"], candidate["start_seconds"], candidate["end_seconds"]) == clip_url:
#             return evaluation
#     return None


# def evaluate_single_candidate(state, sentence, candidate):
#     """
#     Evaluate a single candidate clip against a sentence, applying adjustments if recommended.

#     :param state: Current agent state.
#     :param sentence: The sentence record being processed.
#     :param candidate: The clip candidate to evaluate.
#     :return: A `CandidateEvaluation` with status and metadata.
#     """
    
#     llm_name = state.get("llm", "gemini_2_5_flash")
#     max_adjustments = state.get("max_adjustments_per_candidate", 2)
#     attempts = 0

#     # Initialize the evaluation
#     evaluation: CandidateEvaluation = CandidateEvaluation(
#         candidate=candidate.copy(),
#         status="error",
#         visual_description="",
#         relevance_mapping="",
#         recommendation=None,
#         evaluation_breakdown="",
#         raw_response="",
#         adjustments_applied=0,
#     )

#     while attempts <= max_adjustments:
#         is_final_attempt = (attempts == max_adjustments)
#         try:
#             response = evaluate_video_clip.invoke(
#                 {
#                     "course_name": state["course_name"],
#                     "target_audience": state["target_audience"],
#                     "topic_name": state["topic_name"],
#                     "subtopic_name": state["subtopic_name"],
#                     "slide_title": state["slide_title"],
#                     "full_slide_text": state["slide_chunk"],
#                     "sentence_text": sentence["text"],
#                     "clip_url": candidate["clip_url"],
#                     "start_seconds": candidate["start_seconds"],
#                     "end_seconds": candidate["end_seconds"],
#                     "llm": llm_name,
#                     "is_final_adjustment_attempt": is_final_attempt,
#                 }
#             )
#         except Exception as exc: 
#             evaluation.update(
#                 status="error",
#                 reason=f"Evaluation failed: {exc}",
#             )
#             return evaluation

#         # Print evaluation response
#         print(f"\n{'='*80}")
#         print(f"EVALUATION ATTEMPT {attempts + 1} (Final: {is_final_attempt})")
#         print(f"Sentence: {sentence['text'][:100]}...")
#         print(f"Status: {response.get('status', 'N/A')}")
#         print(f"\n--- FULL LLM RESPONSE ---")
#         print(response.get('raw_response', 'N/A'))
#         print(f"--- END LLM RESPONSE ---\n")
#         print(f"Parsed Fields:")
#         print(f"  Visual Description: {response.get('visual_description', 'N/A')[:200]}...")
#         print(f"  Relevance Mapping: {response.get('relevance_mapping', 'N/A')[:200]}...")
#         if response.get('recommendation'):
#             print(f"  Recommendation: {response.get('recommendation')}")
#         if response.get('selected_window_start') is not None:
#             print(f"  Selected Window: {response.get('selected_window_start')}s - {response.get('selected_window_end')}s")
#         print(f"{'='*80}\n")

#         # Update the evaluation with the response
#         status_value = (response.get("status") or "").lower()
#         effective_start = response.get("start_seconds")
#         effective_end = response.get("end_seconds")

#         # Update the candidate with the adjusted start and end times
#         if effective_start is not None:
#             candidate["start_seconds"] = int(effective_start)
#         if effective_end is not None:
#             candidate["end_seconds"] = int(effective_end)
#         candidate["clip_url"] = normalize_embed_url(
#             candidate["video_id"],
#             candidate["start_seconds"],
#             candidate["end_seconds"],
#         )

#         # Update the evaluation with the adjusted start and end times
#         evaluation = CandidateEvaluation(
#             candidate=candidate.copy(),
#             status=status_value,
#             visual_description=response.get("visual_description", ""),
#             relevance_mapping=response.get("relevance_mapping", ""),
#             recommendation=response.get("recommendation"),
#             evaluation_breakdown=response.get("evaluation_breakdown", ""),
#             raw_response=response.get("raw_response", ""),
#             adjustments_applied=attempts,
#         )

#         # If the evaluation status is not "adjust", return the evaluation
#         if evaluation["status"] != "adjust":
#             return evaluation

#         # If the evaluation status is "adjust", and there is no recommendation, or we have reached the maximum number of adjustments, return the evaluation
#         recommendation = evaluation.get("recommendation")
#         if not recommendation or attempts >= max_adjustments:
#             return evaluation

#         # Adjust the video clip window
#         try:
#             adjusted = adjust_video_clip_window.invoke(
#                 {
#                     "clip_url": candidate["clip_url"],
#                     "original_start": candidate["start_seconds"],
#                     "original_end": candidate["end_seconds"],
#                     "recommendation": recommendation,
#                 }
#             )
#         except Exception as exc:  
#             evaluation.update(
#                 status="error",
#                 reason=f"Adjustment failed: {exc}",
#             )
#             return evaluation

#         # Log the adjustment that was applied
#         print("[TOOL adjust_video_clip_window] Applied adjustment:")
#         print(
#             f"  original_start={candidate['start_seconds']} "
#             f"original_end={candidate['end_seconds']}"
#         )
#         print(
#             f"  recommendation={recommendation} -> "
#             f"new_start={adjusted.get('start_seconds')} "
#             f"new_end={adjusted.get('end_seconds')}"
#         )

#         candidate["start_seconds"] = int(adjusted["start_seconds"])
#         candidate["end_seconds"] = int(adjusted["end_seconds"])
#         candidate["clip_url"] = normalize_embed_url(candidate["video_id"], candidate["start_seconds"], candidate["end_seconds"])
#         attempts += 1

#     return evaluation


# def select_sentence_node(state):
#     """
#     Choose the next sentence to process or mark the run as complete.

#     :param state: Current agent state.
#     :return: Updated state with `current_sentence_index` set or `complete=True`.
#     """

#     # If the run is complete, return the state
#     if state.get("complete"):
#         return state

#     for sentence in state["sentences"]:
#         if sentence.get("status") != "complete":
#             state["current_sentence_index"] = sentence["index"]
#             sentence["status"] = "evaluating"

#             print("\n" + "=" * 100)
#             print(f"New sentence #{sentence['index'] + 1}/{len(state['sentences'])}")
#             print(f"Sentence text: {sentence['text']}")
#             print("=" * 100)

#             return state

#     state["current_sentence_index"] = None
#     state["complete"] = True
#     return state


# def route_after_selection(state):
#     """
#     Router after selecting a sentence.

#     :param state: Current agent state.
#     :return: "complete" if done, otherwise "continue".
#     """
#     return "complete" if state.get("complete") else "continue"


# def evaluate_candidate_node(state):
#     """
#     Evaluate the next candidate for the current sentence.

#     :param state: Current agent state.
#     :return: Updated state with evaluation appended and cursor advanced.
#     """

#     # Get the current sentence index
#     sentence_idx = state.get("current_sentence_index")
#     if sentence_idx is None:
#         return state

#     # Get the current sentence - in parallel mode, there's only one sentence per state
#     # Find by index field, not list position
#     sentence = None
#     for s in state["sentences"]:
#         if s["index"] == sentence_idx:
#             sentence = s
#             break
    
#     if sentence is None:
#         return state

#     # Get the candidates for the current sentence
#     candidates = sentence.get("candidates", [])
#     cursor = sentence.get("candidate_cursor", 0)

#     # Get the limit for the number of candidates
#     limit = len(candidates)

#     # If the cursor is greater than or equal to the limit, return the state
#     if cursor >= limit:
#         return state

#     # Get the current candidate
#     candidate = candidates[cursor]

#     print("\n" + "-" * 80)
#     print(
#         f"[VIDEO_AGENT] Evaluating candidate #{cursor + 1}/{len(candidates)} "
#         f"for sentence #{sentence['index'] + 1}"
#     )
#     print(f"[VIDEO_AGENT] URL: {candidate.get('clip_url')}")
#     print(
#         f"[VIDEO_AGENT] Window: start={candidate.get('start_seconds')} "
#         f"end={candidate.get('end_seconds')}"
#     )
#     print("-" * 80)

#     evaluation = evaluate_single_candidate(state, sentence, candidate)

#     sentence.setdefault("evaluations", []).append(evaluation)
#     if evaluation.get("status") == "accept":
#         sentence.setdefault("accepted_candidates", []).append(evaluation)

#     sentence["candidate_cursor"] = cursor + 1
#     return state


# def sentence_decision_node(state):
#     """
#     Decide whether to evaluate more, compare, or finalize the current sentence.

#     :param state: Current agent state.
#     :return: Updated state with `next_action` set.
#     """

#     # If the run is complete, return the state
#     if state.get("complete"):
#         state["next_action"] = "complete"
#         return state

#     # Get the current sentence index
#     sentence_idx = state.get("current_sentence_index")
#     if sentence_idx is None:
#         state["next_action"] = "complete"
#         return state

#     # Get the current sentence - find by index field, not list position
#     sentence = None
#     for s in state["sentences"]:
#         if s["index"] == sentence_idx:
#             sentence = s
#             break
    
#     if sentence is None:
#         state["next_action"] = "complete"
#         return state
    
#     candidates = sentence.get("candidates", [])
#     cursor = sentence.get("candidate_cursor", 0)

#     # Get the limit for the number of candidates
#     limit = len(candidates)

#     # If the cursor is less than the limit, return the state
#     if cursor < limit:
#         state["next_action"] = "evaluate_more"
#         return state

#     # Get the available candidates
#     available = filter_available_candidates(sentence, state.get("used_clip_urls", []))

#     # If there are more than one available candidate and no comparison result, set the comparison pool and return the state
#     if len(available) > 1 and not sentence.get("comparison_result"):
#         sentence["comparison_pool"] = available
#         state["next_action"] = "compare"
#         return state

#     # If there are accepted candidates, set the next action to finalize
#     if sentence.get("accepted_candidates"):
#         state["next_action"] = "finalize"
#     else:
#         # If there are no accepted candidates, set the next action to finalize
#         state["next_action"] = "finalize"
#     return state


# def route_from_decision(state):
#     """
#     Router for sentence decision outcomes.

#     :param state: Current agent state.
#     :return: One of "evaluate_more", "compare", "finalize", or "complete".
#     """
#     return state.get("next_action", "complete")


# def compare_candidates_node(state):
#     """
#     Compare accepted candidates for the current sentence to select the best clip.

#     :param state: Current agent state.
#     :return: Updated state with `comparison_result` set and next action to finalize.
#     """

#     # Get the current sentence index
#     sentence_idx = state.get("current_sentence_index")
#     if sentence_idx is None:
#         return state

#     # Get the current sentence - find by index field, not list position
#     sentence = None
#     for s in state["sentences"]:
#         if s["index"] == sentence_idx:
#             sentence = s
#             break
    
#     if sentence is None:
#         state["next_action"] = "finalize"
#         return state
    
#     pool = sentence.get("comparison_pool", [])

#     # If there are less than two candidates in the comparison pool, set the next action to finalize and return the state
#     if len(pool) < 2:
#         state["next_action"] = "finalize"
#         return state

#     # Build the candidates payload
#     candidates_payload: list[dict[str, Any]] = []
#     for evaluation in pool:
#         candidate = evaluation.get("candidate", {})

#         # Normalize the candidate URL
#         clip_url = normalize_embed_url(candidate["video_id"], candidate["start_seconds"], candidate["end_seconds"])
#         # Build the candidate payload
#         candidates_payload.append(
#             {
#                 "clip_url": normalize_embed_url(candidate["video_id"], candidate["start_seconds"], candidate["end_seconds"]),
#                 "visual_description": evaluation.get("visual_description", ""),
#                 "relevance_mapping": evaluation.get("relevance_mapping", ""),
#                 "start_seconds": candidate.get("start_seconds"),
#                 "end_seconds": candidate.get("end_seconds"),
#             }
#         )

#     # Compare the candidates
#     try:
#         comparison = compare_video_clips.invoke(
#             {
#                 "course_name": state["course_name"],
#                 "target_audience": state["target_audience"],
#                 "topic_name": state["topic_name"],
#                 "subtopic_name": state["subtopic_name"],
#                 "slide_title": state["slide_title"],
#                 "full_slide_text": state["slide_chunk"],
#                 "sentence_text": sentence["text"],
#                 "candidate_clips": candidates_payload,
#                 "llm": state.get("llm", "gemini_2_flash"),
#             }
#         )
#         sentence["comparison_result"] = comparison
        
#         # Print comparison response
#         print(f"\n{'='*80}")
#         print(f"COMPARISON RESULT")
#         print(f"Sentence: {sentence['text'][:100]}...")
#         print(f"Candidates Compared: {len(pool)}")
#         print(f"\n--- FULL LLM RESPONSE ---")
#         print(comparison.get('raw_response', 'N/A'))
#         print(f"--- END LLM RESPONSE ---\n")
#         print(f"Parsed Fields:")
#         print(f"  Selected Clip: {comparison.get('clip_url', 'N/A')}")
#         print(f"  Visual Description: {comparison.get('visual_description', 'N/A')[:200]}...")
#         print(f"  Relevance Mapping: {comparison.get('relevance_mapping', 'N/A')[:200]}...")
#         print(f"{'='*80}\n")
        
#     except Exception as exc:  # pragma: no cover
#         sentence["comparison_result"] = {
#             "clip_url": "",
#             "relevance_mapping": f"Comparison failed: {exc}",
#             "visual_description": "",
#             "observations": "",
#             "comparison": "",
#             "decision": "",
#             "raw_response": "",
#         }

#     state["next_action"] = "finalize"
#     return state


# def finalize_sentence_node(state):
#     """
#     Finalize the current sentence by selecting a clip and composing the output block.

#     :param state: Current agent state.
#     :return: Updated state with final block, used URLs, and sentence status.
#     """

#     # Get the current sentence index
#     sentence_idx = state.get("current_sentence_index")
#     if sentence_idx is None:
#         state["next_action"] = "complete"
#         state["complete"] = True
#         return state

#     # Get the current sentence - find by index field, not list position
#     sentence = None
#     for s in state["sentences"]:
#         if s["index"] == sentence_idx:
#             sentence = s
#             break
    
#     if sentence is None:
#         state["next_action"] = "complete"
#         state["complete"] = True
#         return state
    
#     selection: Optional[CandidateEvaluation] = None
#     comparison = sentence.get("comparison_result")
    
#     # Get the used URLs
#     used_urls = state.get("used_clip_urls", [])

#     # If there is a comparison result and a clip URL, look up the candidate by URL
#     if comparison and comparison.get("clip_url"):
#         selection = lookup_candidate_by_url(sentence, comparison["clip_url"])

#     # If there is no selection, and there are accepted candidates, set the selection to the first accepted candidate
#     if selection is None:
#         if sentence.get("accepted_candidates"):
#             selection = sentence["accepted_candidates"][0]

#     # Build the block lines
#     block_lines = [f'When VO "{sentence["text"]}"']
#     if selection:
#         candidate = selection.get("candidate", {})
#         clip_url = normalize_embed_url(candidate["video_id"], candidate["start_seconds"], candidate["end_seconds"])
#         block_lines.append(f"Video URL with timestamps: {clip_url}")
#         block_lines.append(f"Visual description: {selection.get('visual_description', '').strip()}")
#         block_lines.append(f"Relevance mapping: {selection.get('relevance_mapping', '').strip()}")
        
#         # Add the clip URL to the used URLs if it is not already in the list
#         if clip_url not in used_urls:
#             used_urls.append(clip_url)

#         # Set the final selection to the selection
#         sentence["final_selection"] = selection
#     else:
#         block_lines.append("Status: No relevant clip found")

#         # Set the final selection to None
#         sentence["final_selection"] = None

#     # Join the block lines and strip whitespace
#     block_text = "\n".join(block_lines).strip()

#     # Set the final block to the block text
#     sentence["final_block"] = block_text
#     sentence["status"] = "complete"
#     state["final_outputs"] = state.get("final_outputs", []) + [block_text]
#     state["final_video_clip"] = "\n\n".join(state["final_outputs"]).strip()
#     state["current_sentence_index"] = None
#     state["next_action"] = "continue"
#     return state


# def build_video_clip_graph():
#     """
#     Construct the LangGraph state machine for video clip identification.

#     :return: A configured `StateGraph` instance.
#     """

#     # Build the graph
#     graph = StateGraph(VideoClipAgentState)
#     # Add the nodes to the graph
#     graph.add_node("select_sentence", RunnableLambda(select_sentence_node))
#     graph.add_node("sentence_decision", RunnableLambda(sentence_decision_node))
#     # Add the evaluate candidate node to the graph
#     graph.add_node("evaluate_candidate", RunnableLambda(evaluate_candidate_node))
#     # Add the compare candidates node to the graph
#     graph.add_node("compare_candidates", RunnableLambda(compare_candidates_node))
#     # Add the finalize sentence node to the graph
#     graph.add_node("finalize_sentence", RunnableLambda(finalize_sentence_node))
#     # Set the entry point to the select sentence node

#     # Set the entry point to the select sentence node
#     graph.set_entry_point("select_sentence")
#     # Add the conditional edges to the graph

#     # Add the conditional edges to the graph
#     graph.add_conditional_edges(
#         "select_sentence",
#         route_after_selection,
#         {
#             # If the run is complete, return the state
#             "complete": END,
#             # If the run is not complete, continue to the sentence decision node
#             "continue": "sentence_decision",
#         },
#     )

#     # Add the conditional edges to the graph
#     graph.add_conditional_edges(
#         "sentence_decision",
#         route_from_decision,
#         {
#             # If the next action is evaluate more, evaluate the candidate
#             "evaluate_more": "evaluate_candidate",
#             # If the next action is compare, compare the candidates
#             "compare": "compare_candidates",
#             # If the next action is finalize, finalize the sentence
#             "finalize": "finalize_sentence",
#             # If the run is complete, return the state
#             "complete": END,
#         },
#     )

#     # Add the edges to the graph
#     graph.add_edge("evaluate_candidate", "sentence_decision")
#     # Add the edges to the graph
#     graph.add_edge("compare_candidates", "finalize_sentence")
#     # Add the edges to the graph
#     graph.add_edge("finalize_sentence", "select_sentence")

#     # Return the graph
#     return graph


# def prepare_initial_state(course_name, target_audience, topic_name, subtopic_name, slide_title, slide_chunk, candidate_urls, llm, max_adjustments_per_candidate):
#     """
#     Prepare the initial agent state from slide metadata and candidate URLs.

#     :param course_name: The name of the course.
#     :param target_audience: Description of the target audience.
#     :param topic_name: The topic associated with the slide.
#     :param subtopic_name: The subtopic associated with the slide.
#     :param slide_title: The slide's title or heading.
#     :param slide_chunk: The slide body text.
#     :param candidate_urls: Grouped text or list for candidate clip URLs.
#     :param llm: Model identifier for downstream tool calls.
#     :param max_adjustments_per_candidate: Max timestamp adjustment iterations.
#     :return: Initialized `VideoClipAgentState`.
#     """

#     sentence_texts = split_into_sentences(slide_chunk)

#     # Ensure at least one sentence text for empty content
#     if not sentence_texts:
#         sentence_texts = [slide_chunk.strip()] if slide_chunk.strip() else ["No slide content provided."]

#     # Detect grouped per-sentence links (only grouped 'Sentence N:' text is supported)
#     grouped_mapping: dict[int, list[str]] = {}
#     if isinstance(candidate_urls, str):
#         parsed = extract_grouped_sentence_urls(candidate_urls)
#         if parsed:
#             grouped_mapping = parsed

#     # Build sentence records 
#     sentences = ensure_sentence_records_from_grouped(sentence_texts, grouped_mapping)
#     combined_pool: list[ClipCandidate] = []  

#     return VideoClipAgentState(
#         course_name=course_name,
#         target_audience=target_audience,
#         topic_name=topic_name,
#         subtopic_name=subtopic_name,
#         slide_title=slide_title,
#         slide_chunk=slide_chunk,
#         llm=llm,
#         sentences=sentences,
#         candidate_pool=combined_pool,
#         current_sentence_index=None,
#         max_adjustments_per_candidate=max_adjustments_per_candidate,
#         used_clip_urls=[],
#         next_action="evaluate_more",
#         final_outputs=[],
#         final_video_clip="",
#         complete=False,
#     )


# @traceable(
#     metadata={
#         "agent_name": "graphics_definition",
#         "step_name": "LangGraph Video Clip Identification",
#         "function_name": "process_row_for_video_clip",
#         "user_email": st.session_state.get("user_email", "anonymous")
#     }
# )
# def process_row_for_video_clip( slide_chunks_df, idx, course_name, target_audience, llm, max_adjustments_per_candidate=2, sentence_max_workers=5):
#     """
#     Process a single "Slide Chunks" row to identify relevant video clips.

#     :param slide_chunks_df: DataFrame containing slide chunk data.
#     :param idx: Row index to process.
#     :param course_name: The name of the course.
#     :param target_audience: The target audience.
#     :param llm: The language model identifier to use.
#     :param max_adjustments_per_candidate: Max timestamp refinements allowed.
#     :param sentence_max_workers: Number of parallel workers for sentence processing.
#     :return: Dict with row index and the final composed `final_video_clip` value (and error if any).
#     """

#     # Get the row data
#     row = slide_chunks_df.loc[idx].copy()
#     slide_type = str(row.get("Slide Type", "")).strip().lower()
#     slide_chunk = str(row.get("Slide Chunk", "")).strip()
#     slide_title = str(row.get("Slide Chunk Title", "")).strip() or str(row.get("Topic", "")).strip() or "Slide"
#     topic_name = str(row.get("Topic", "")).strip()
#     subtopic_name = str(row.get("Subtopic", "")).strip()

#     # If the slide type is video, transition, or summary, return the row index and value
#     if slide_type in {"video", "transition", "summary"}:
#         return {"idx": idx, "value": "-"}

#     # If the slide chunk is empty, return the row index and value
#     if not slide_chunk:
#         return {"idx": idx, "value": "Error: Missing slide chunk content"}

#     # Get the grouped text
#     grouped_text = str(row.get("slide_chunk_videos", ""))
#     try:
#         # Run the video clip identification agent
#         final_state = run_video_clip_identification_agent(
#             course_name=course_name,
#             target_audience=target_audience,
#             topic_name=topic_name,
#             subtopic_name=subtopic_name,
#             slide_title=slide_title or topic_name or "Slide",
#             slide_chunk=slide_chunk,
#             candidate_urls=grouped_text,
#             llm=llm,
#             max_adjustments_per_candidate=max_adjustments_per_candidate,
#             sentence_max_workers=sentence_max_workers,
#         )

#         # Get the final clip value
#         final_clip_value = final_state.get("final_video_clip", "").strip()
#         # If the final clip value is empty, set it to "Status: No relevant clip found"
#         if not final_clip_value:
#             final_clip_value = "Status: No relevant clip found"
#         # Return the row index and value
#         return {"idx": idx, "value": final_clip_value}
#     # If there is an exception, return the row index and value with the error
#     except Exception as exc:  
#         return {"idx": idx, "value": f"Error: {exc}", "error": str(exc)}


# def process_single_sentence(sentence_record, course_name, target_audience, topic_name, subtopic_name, slide_title, slide_chunk, llm, max_adjustments_per_candidate):
#     """
#     Process a single sentence independently using the graph.
    
#     :param sentence_record: A single SentenceRecord with candidates.
#     :param course_name: The name of the course.
#     :param target_audience: The target audience for the course.
#     :param topic_name: Topic of the slide.
#     :param subtopic_name: Subtopic of the slide.
#     :param slide_title: The title of the slide.
#     :param slide_chunk: Full slide text for context.
#     :param llm: Model name to use for all tool calls.
#     :param max_adjustments_per_candidate: Number of timestamp refinements allowed per clip.
#     :return: Completed sentence record with final_block and final_selection.
#     """
    
#     # Create a per-sentence state with only this sentence
#     per_sentence_state = VideoClipAgentState(
#         course_name=course_name,
#         target_audience=target_audience,
#         topic_name=topic_name,
#         subtopic_name=subtopic_name,
#         slide_title=slide_title,
#         slide_chunk=slide_chunk,
#         llm=llm,
#         sentences=[sentence_record],  # Only one sentence
#         candidate_pool=[],
#         current_sentence_index=None,
#         max_adjustments_per_candidate=max_adjustments_per_candidate,
#         used_clip_urls=[],  # Each sentence starts with empty used URLs
#         next_action="evaluate_more",
#         final_outputs=[],
#         final_video_clip="",
#         complete=False,
#     )
    
#     # Build and run the graph for this sentence
#     graph = build_video_clip_graph().compile()
#     final_state = graph.invoke(per_sentence_state, config={"recursion_limit": graph_recursion_limit})
    
#     # Return the completed sentence record
#     return final_state["sentences"][0]


# @traceable(
#     metadata={
#         "agent_name": "graphics_definition",
#         "step_name": "LangGraph Video Clip Identification",
#         "function_name": "run_video_clip_identification_agent",
#         "user_email": st.session_state.get("user_email", "anonymous")
#     }
# )
# def run_video_clip_identification_agent(course_name, target_audience, topic_name, subtopic_name, slide_title, slide_chunk, candidate_urls, llm = "gemini_2_5_flash", max_adjustments_per_candidate=2, sentence_max_workers=5):
#     """
#     Run the LangGraph-based Video Clip Identification agent for a single slide.

#     :param course_name: The name of the course.
#     :param target_audience: The target audience for the course.
#     :param topic_name: Topic of the slide.
#     :param subtopic_name: Subtopic of the slide.
#     :param slide_title: The title of the slide.
#     :param slide_chunk: Slide text containing multiple sentences.
#     :param candidate_urls: Grouped text or list of timestamped embed URLs.
#     :param llm: Model name to use for all tool calls.
#     :param max_adjustments_per_candidate: Number of timestamp refinements allowed per clip.
#     :param sentence_max_workers: Number of parallel workers for sentence processing. Default: 5.
#     :return: Final state with `final_video_clip`, evaluations, and bookkeeping data.
#     """

#     # Prepare the initial state to get sentence records
#     initial_state = prepare_initial_state(
#         course_name=course_name,
#         target_audience=target_audience,
#         topic_name=topic_name,
#         subtopic_name=subtopic_name,
#         slide_title=slide_title,
#         slide_chunk=slide_chunk,
#         candidate_urls=candidate_urls,
#         llm=llm,
#         max_adjustments_per_candidate=max_adjustments_per_candidate,
#     )

#     sentence_records = initial_state["sentences"]
    
#     print(f"\n🚀 Processing {len(sentence_records)} sentences IN PARALLEL with {sentence_max_workers} workers")
    
#     completed_sentences = []
    
#     with ThreadPoolExecutor(max_workers=sentence_max_workers) as executor:
#         future_map = {
#             executor.submit(
#                 process_single_sentence,
#                 sentence_record,
#                 course_name,
#                 target_audience,
#                 topic_name,
#                 subtopic_name,
#                 slide_title,
#                 slide_chunk,
#                 llm,
#                 max_adjustments_per_candidate,
#             ): sentence_record
#             for sentence_record in sentence_records
#         }
        
#         for future in as_completed(future_map):
#             original_sentence_record = future_map[future]
#             sentence_idx = original_sentence_record["index"]
#             try:
#                 completed_sentence = future.result()
#                 completed_sentences.append(completed_sentence)
#                 print(f"✅ Completed sentence {sentence_idx + 1}/{len(sentence_records)}")
#             except Exception as exc:
#                 print(f"❌ Error processing sentence {sentence_idx + 1}: {exc}")
#                 # Create a fallback sentence record
#                 fallback = original_sentence_record.copy()
#                 fallback["status"] = "error"
#                 fallback["final_block"] = f'When VO "{fallback["text"]}"\nStatus: Error - {exc}'
#                 fallback["final_selection"] = None
#                 completed_sentences.append(fallback)
    
#     # Sort by original sentence index to maintain order
#     completed_sentences.sort(key=lambda s: s["index"])
    
#     # Build final output
#     final_outputs = [s["final_block"] for s in completed_sentences]
#     final_video_clip = "\n\n".join(final_outputs).strip()
    
#     # Construct final state
#     final_state = initial_state.copy()
#     final_state["sentences"] = completed_sentences
#     final_state["final_outputs"] = final_outputs
#     final_state["final_video_clip"] = final_video_clip
#     final_state["complete"] = True
    
#     print(f"✅ Parallel sentence processing complete. Total sentences: {len(completed_sentences)}")
    
#     return final_state


# @traceable(
#     metadata={
#         "agent_name": "graphics_definition",
#         "step_name": "LangGraph Video Clip Identification",
#         "function_name": "run_video_clip_identidication_agent_for_all_rows",
#         "user_email": st.session_state.get("user_email", "anonymous")
#     }
# )
# def run_video_clip_identidication_agent_for_all_rows(sheet, course_name, target_audience, slide_chunks_sheet="Slide Chunks", llm='gemini_2_5_flash', max_adjustments_per_candidate=2, skip_existing=True, save_interval=5, parallel_max_workers=5, sentence_max_workers=5):
#     """
#     Run the video clip identification agent across all eligible rows in a sheet.

#     :param sheet: Google Sheet client object.
#     :param course_name: The name of the course.
#     :param target_audience: The target audience for the course.
#     :param slide_chunks_sheet: Worksheet name that stores slide chunks.
#     :param llm: Model name to use for tool calls.
#     :param max_adjustments_per_candidate: Number of timestamp refinements allowed.
#     :param skip_existing: Skip rows with non-empty `final_video_clip`.
#     :param save_interval: Interval (in rows) to persist progress.
#     :param parallel_max_workers: Number of worker threads for parallel row processing (row-level parallelism).
#     :param sentence_max_workers: Number of worker threads for parallel sentence processing within a row. Default: 5.
#     :return: Dict with processed row indices, errors, and total row count.
#     """

#     # Get the slide chunks worksheet and dataframe
#     slide_chunks_ws, slide_chunks_df = get_sheet_data_and_df(sheet, slide_chunks_sheet)
#     # If the final video clip column is not in the dataframe, add it
#     if "final_video_clip" not in slide_chunks_df.columns:
#         slide_chunks_df["final_video_clip"] = ""

#     # Get the rows to process
#     rows_to_process: list[int] = []
#     # Iterate over the rows
#     for idx, row in slide_chunks_df.iterrows():
#         existing_value = str(row.get("final_video_clip", "")).strip()
#         # If the skip existing flag is set and the existing value is not empty, skip the row
#         if skip_existing and existing_value:
#             # Add the row index to the rows to process
#             continue
#         rows_to_process.append(idx)

#     # If there are no rows to process, return the processed rows, errors, and total rows
#     if not rows_to_process:
#         print("All rows already contain final_video_clip values. Nothing to process.")
#         return {"processed_rows": [], "errors": [], "total_rows": 0}

#     # Initialize the progress bar
#     progress = SmartProgressBar(
#         total_tasks=len(rows_to_process),
#         description="LangGraph Video Clip Agent",
#         save_interval=save_interval,
#     )

#     # Initialize the processed rows and errors
#     processed_rows: list[int] = []
#     errors: list[dict[str, Any]] = []

#     # Parallel branch
#     if parallel_max_workers and parallel_max_workers > 1:
#         with ThreadPoolExecutor(max_workers=parallel_max_workers) as executor:
#             future_map = {
#                 executor.submit(
#                     process_row_for_video_clip,
#                     slide_chunks_df,
#                     idx,
#                     course_name=course_name,
#                     target_audience=target_audience,
#                     llm=llm,
#                     max_adjustments_per_candidate=max_adjustments_per_candidate,
#                     sentence_max_workers=sentence_max_workers,
#                 ): idx
#                 for idx in rows_to_process
#             }
#             for future in as_completed(future_map):
#                 # Get the result
#                 result = future.result()
#                 idx = result["idx"]
#                 value = result["value"]
#                 slide_chunks_df.at[idx, "final_video_clip"] = value

#                 if "error" in result:
#                     errors.append({"row_index": idx, "error": result["error"]})

#                 processed_rows.append(idx)
#                 progress.update()

#                 if progress.should_save():
#                     save_to_sheet(slide_chunks_ws, slide_chunks_df)
#                     format_worksheet(slide_chunks_ws)

#         save_to_sheet(slide_chunks_ws, slide_chunks_df)
#         format_worksheet(slide_chunks_ws)

#         return {
#             "processed_rows": processed_rows,
#             "errors": errors,
#             "total_rows": len(rows_to_process),
#         }


#     # Iterate over the rows to process
#     for idx in rows_to_process:
#         # Process the row
#         result = process_row_for_video_clip(
#             slide_chunks_df,
#             idx,
#             course_name=course_name,
#             target_audience=target_audience,
#             llm=llm,
#             max_adjustments_per_candidate=max_adjustments_per_candidate,
#             sentence_max_workers=sentence_max_workers,
#         )
#         # Set the final video clip value
#         slide_chunks_df.at[idx, "final_video_clip"] = result["value"]
#         # If there is an error, add the row index and error to the errors list
#         if "error" in result:
#             errors.append({"row_index": idx, "error": result["error"]})

#         # Add the row index to the processed rows list
#         processed_rows.append(idx)
#         # Update the progress bar
#         progress.update()

#         # If the progress bar should save, save the sheet and format the worksheet
#         if progress.should_save():
#             save_to_sheet(slide_chunks_ws, slide_chunks_df)
#             format_worksheet(slide_chunks_ws)

#     # Save the sheet and format the worksheet
#     save_to_sheet(slide_chunks_ws, slide_chunks_df)
#     format_worksheet(slide_chunks_ws)

#     # Return the processed rows, errors, and total rows
#     return {
#         "processed_rows": processed_rows,
#         "errors": errors,
#         "total_rows": len(rows_to_process),
#     }


# def delete_video_clip_from_sheet(sheet, worksheet_name="Slide Chunks"):
#     """
#     Remove the `final_video_clip` column from a worksheet.

#     :param sheet: Google Sheet client object.
#     :param worksheet_name: Worksheet name to modify.
#     :return: None.
#     """

#     # Get the slide chunks worksheet and dataframe
#     slide_chunks_ws, slide_chunks_df = get_sheet_data_and_df(sheet, worksheet_name)
#     # If the final video clip column is in the dataframe, drop it
#     if "final_video_clip" in slide_chunks_df.columns:
#         slide_chunks_df = slide_chunks_df.drop(columns=["final_video_clip"])
#         clear_worksheet(slide_chunks_ws)
#         save_to_sheet(slide_chunks_ws, slide_chunks_df)
#         format_worksheet(slide_chunks_ws)