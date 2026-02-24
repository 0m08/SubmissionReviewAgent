# from __future__ import annotations

# import re
# from typing import Any, Dict, List, Optional
# from typing_extensions import TypedDict

# from langchain_core.tools import tool

# from modules.chain import Chain
# from google import genai
# from google.genai import types


# evaluate_clip_prompt = """You are a senior HVAC instructional designer and Video Content Quality Specialist. Your job is to watch and assess whether the provided youtube video segment is the right visual to display during the narration of the given sentence in an e-learning slide, based solely on what appears on screen in that video segment.

# A) Input:

# Here are the inputs for your evaluation: 

# <input>

# <course_info>

# Course name that the slide sentence belongs to: {course_name}
# Target audience: {target_audience}
# Topic of the slide: {topic_name}
# Subtopic of the slide: {subtopic_name}

# </course_info>

# <slide_info>

# Slide title: {slide_title}

# Full slide content: "{full_slide_text}"

# Sentence that the video clip must support: "{sentence_text}"

# </slide_info>

# <youtube_video_clip_info>

# URL of the Video clip that you will be watching: {clip_url}
# Start timestamp in seconds of this YouTube video clip: {start_seconds}
# End timestamp in seconds of this YouTube video clip: {end_seconds}

# </youtube_video_clip_info>

# </input>

# B) Instructions and Guidelines: You must base your judgment only on what you SEE in the clip – ignore any audio or narration. Follow the rules below:

# 1. Context interpretation
#    - The full slide content is provided so you can understand the context and correctly interpret what the sentence you are evaluating - and its pronouns, if any — refers to when selecting visuals.

# 2. Relevance to the sentence content
#    - The visuals must directly illustrate the idea expressed in the specific sentence you are evaluating.
#    - Because this visual will be shown on screen while the sentence is being narrated in the e-learning slide, it must help the learner clearly understand what the sentence narration is referring to.
#    - Check whether the objects, components, or actions visible on screen match the meaning of the sentence, not just the general topic of the slide.
#    - Reject any clip that is unrelated, loosely related, or does not meaningfully support the sentence being narrated.
#    - Remember, you are evaluating the video clip only for the specific sentence provided, not the entire slide. The goal is to find a visual that supports that one sentence at the moment it is narrated.

# 3. Visual quality and clarity
#    - Accept steady, well-lit, and clearly framed shots where all important visual details are easy for a learner to see and understand.
#    - Reject clips that are blurry, shaky, obstructed, poorly lit, or dominated by irrelevant subjects (such as people talking to camera, interviews, static slides, or low-resolution screen recordings).
#    - The visual content should be clear enough for a learner to immediately recognize what is happening or being shown for it to be accepted.

# 4. Timing and timestamp selection
#    - If you find the video clip to be acceptable for the sentence, determine whether the entire timestamp window is relevant or whether only a portion of the clip provides the strongest visual support.
#    - Identify the exact start and end seconds inside the clip that contain the best instructional moment for the evaluated sentence.
#    - You do not need to use the entire timestamp range. If only a smaller part of the provided clip is relevant, selecting that shorter portion is also allowed.
#    - Check how the clip begins: does the relevant action start immediately, or does the clip contain unnecessary lead-in footage?
#    - Check how the clip ends: does the relevant action finish cleanly, or does the clip include extra footage after the useful moment?
#    - If trimming improves relevance or clarity, recommend a shorter window using the standard timestamp adjustment tags.
#    - Ideally, the selected video clip should be between 5 and 15 seconds long.

# 5. When to recommend a timestamp adjustment
#    - In some cases, the visuals within the current timestamp window may be relevant to the evaluated sentence, but the window itself might not include the full instructional moment. In such situations, you may recommend adjusting the start and/or end timestamps.
#    - You cannot see any footage outside the provided timestamp range, but you may still infer that the relevant action began earlier or continues beyond the current window if the visible edges of the clip strongly suggest it. In such cases, you may recommend extending the start time, the end time, or both, using the standardized timestamp adjustment tags.
#    - Any recommendation to extend the timestamps is an informed guess based solely on what you can see inside the clip. Your guess must come from what is happening at the very beginning or end of the clip - such as when the clip cuts in while an action is already underway, or cuts out before the action is finished.
#    - Any extension must stay within the allowed 30-second limit at each end of the clip. This means the start time can be moved earlier by up to 30 seconds, and the end time can be moved later by up to 30 seconds, but never beyond those limits.
#    - All timestamp adjustments must use one of these 3 standardized formats:
#      extend_start_to_<seconds>
#      or
#      extend_end_to_<seconds>
#      or
#      extend_start_to_<seconds>_and_extend_end_to_<seconds>
#      eg. extend_start_to_10_and_extend_end_to_20, extend_start_to_58, extend_end_to_129
#    - When using the standardized timestamp adjustment formats, follow these rules:
#      a) For extend_start_to_<seconds>, the new start value must be earlier (numerically lower) than the current start timestamp of the provided clip.
#      b) For extend_end_to_<seconds>, the new end value must be later (numerically higher) than the current end timestamp of the provided clip.
#      c) For extend_start_to_<seconds>_and_extend_end_to_<seconds>, the new start must be earlier than the clip’s current start timestamp, and the new end must be later than the clip’s current end timestamp.
#         (All still constrained by the 30-second maximum extension per side.)

#     {adjustment_context}

# C) Output requirements: After watching the video clip, you must produce a response only in the required XML format.
# Your output must strictly follow the structure below:

# 1. <evaluation_breakdown> section
# This is your scratchpad and reasoning area. Here you must analyze, think and explain your thought process step-by-step using the following required subsections:
#    - Sentence Understanding: Summarize what the evaluated sentence means and what visual idea the learner needs to understand. If the meaning of the sentence depends on context from the full slide—such as when a pronoun refers to a component mentioned earlier—briefly clarify that reference as part of your understanding.
#    - Visual Observations: Describe exactly what is seen in the clip (camera angles, equipment, actions, objects, tools, on-screen overlays, etc).
#    - Relevance Assessment: Evaluate whether the visuals meaningfully support the sentence being evaluated. Determine whether the entire clip is relevant or only a specific portion. If only part of the clip is relevant, identify approximately where that portion begins and ends within the current timestamp window. If the relevant clip appears to start before the clip begins or continue beyond the clip’s end, note these observations so they can inform a potential timestamp adjustment.
#    - Visual Quality and Clarity Assessment: If the clip seems relevant to the sentence, assess whether the visuals are clear enough for a learner to easily see and understand what is happening. Comment on lighting, stability, sharpness, framing, and how well important details (components, actions, labels, etc.) are visible. Note any issues — such as blurriness, poor lighting, shaky camera movement, distracting elements, etc — that would make the clip hard to use in an e-learning slide.
#    - Timing and Timestamp Assessment: If the clip seems relevant to the sentence, analyze whether the current timestamp window is well chosen for the evaluated sentence. Explain whether the relevant action starts too late, too early, or at an appropriate time within the clip. Comment on any unnecessary lead-in or extra footage after the useful moment. If you believe that trimming the window or extending it slightly before or after would improve alignment with the sentence, describe your reasoning here and indicate which direction (earlier start, later end, or both) would be helpful.
#    - Decision Rationale: Provide a clear explanation that leads directly to your final status choice (accept, reject, or adjust). Connect your decision to your relevance, quality, and timing assessments. If you are recommending a timestamp adjustment, briefly explain why adjusting the start and/or end time (rather than accepting or rejecting the clip outright) is the most appropriate outcome.

# 2. <evaluation> section
# This section contains your final judgment based on your analysis. It must include the following elements:

# a) <status>:
# Must be one of the following - 

# accept — When the clip is fully appropriate for the evaluated sentence. It is relevant, clear, and requires no timestamp extensions.
# reject — When the clip does not support the sentence. This may be due to irrelevant content, incorrect components, unclear visuals, poor framing or lighting, distracting footage, or any issue that makes the clip unsuitable for use.
# adjust — When the clip appears relevant and usable, but the current timestamp window does not fully capture the best instructional moment. A refinement—either earlier, later, or both—is needed to include the complete relevant action. IMPORTANT: Only use "adjust" status if you are NOT on the final adjustment attempt. If this is your last allowed adjustment attempt, you MUST choose either "accept" or "reject" instead of "adjust".

# b) <selected_window> (required when status = accept)
# When your status is "accept", you must provide the exact start and end timestamps that should be used as the final clip window.
# If the entire clip window is relevant, return the original start and end timestamps.
# If only a portion is relevant, return the more precise start and end timestamps within the provided window.
# Always use this format: start=<seconds> end=<seconds>
# (Don't include this field in your response when status is reject or adjust)

# c) <visual_description> (required when status = accept)
# Provide a clear, concise description of what appears on screen during the timestamp window of the video clip that you selected as acceptable for the sentence.
# (Don't include this field in your response when status is reject or adjust)

# d) <relevance_mapping> (required when status = accept)
# Explain clearly how the visuals in the selected timestamp window support the sentence. Describe which parts of the imagery correspond to the key idea or action in the sentence, and why this clip is an appropriate visual match.
# (Don't include this field in your response when status is reject or adjust)

# e) <recommendations> (required when status = adjust)
# Include this tag only if your status is "adjust". Inside this tag, provide exactly one of the following standardized timestamp adjustment formats:
# extend_start_to_<seconds>
# extend_end_to_<seconds>
# extend_start_to_<seconds>_and_extend_end_to_<seconds>
# (Don't include this field in your response when status is accept or reject)

# So based on the above guidelines regarding the output section, you should always strictly provide your output in the following format:

# <response>

# <evaluation_breakdown>
# (Your detailed reasoning goes here, following all required subsections)
# </evaluation_breakdown>

# <status>
# accept|reject|adjust
# </status>

# <selected_window> 
# (When status is accept)
# start=<seconds> end=<seconds>
# </selected_window>

# <visual_description>
# (When status is accept)
# </visual_description>

# <relevance_mapping>
# (When status is accept)
# </relevance_mapping>

# <recommendations>
# (When status is adjust)
# extend_start_to_<seconds>
# Or
# extend_end_to_<seconds>
# Or
# extend_start_to_<seconds>_and_extend_end_to_<seconds>
# </recommendations>

# </response>
# """


# compare_clips_prompt = """You are a senior HVAC instructional designer and Video Content Quality Specialist. Your job is to compare multiple candidate YouTube video clips — each previously marked as acceptable for supporting the same slide sentence — and select the single strongest match. You must carefully watch every candidate clip exactly as it is given and determine which one provides the clearest, most accurate, and most instructionally useful visual support for the sentence being narrated in the e-learning slide. Your evaluation must be based only on what you see on screen. Ignore the audio of the video clip. The goal is to identify which candidate provides the best visual evidence for helping the learner understand the precise meaning of the sentence as it is being narrated.

# A) Input:

# Here are the inputs for your evaluation: 

# <input>

# <course_info>

# Course name that the slide sentence belongs to: {course_name}
# Target audience: {target_audience}
# Topic of the slide: {topic_name}
# Subtopic of the slide: {subtopic_name}

# </course_info>

# <slide_info>

# Slide title: {slide_title}

# Full slide content: "{full_slide_text}"

# Sentence that the video clip must support: "{sentence_text}"

# </slide_info>

# <accepted_clips>

# The following are the candidate clips that were previously marked as acceptable for supporting this slide sentence:

# {candidate_block}

# </accepted_clips>

# </input>

# B) Instructions and Guidelines: You must base your judgment only on what you SEE in the clip – ignore any audio or narration. Follow the rules below:

# 1. Base your judgment only on what is visible
#    - You must judge each clip entirely on what you see in the video.
#    - Ignore audio, narration, or assumptions about what might be happening outside the visible footage.
#    - You may not assume anything that is not visually confirmed.

# 2. Focus on supporting the exact sentence
#    - Each candidate clip must be judged only on how well it visually supports the specific sentence being narrated.
#    - Use the full slide content to correctly interpret pronouns or references inside the sentence.
#    - Avoid being influenced by the broader topic or subtopic. The winner must support the sentence, not the slide in general.

# 3. Compare clips across four key dimensions

# a) Visual relevance
#    - Which clip most directly shows the component, action, object, or idea described in the sentence?
#    - If multiple clips seem visually relevant, choose the one that presents the idea most clearly and most closely matches the sentence’s intended meaning.

# b) Visual clarity and quality
#    - Prefer clips that are sharp, well-framed, and properly lit.
#    - Reject shaky, blurry, obstructed, or visually confusing footage.

# c) Instructional strength
#    - Which clip helps a learner understand the sentence immediately and with minimal cognitive effort?
#    - Strong clips highlight the correct component or action without distractions.
#    - Prefer clips that show the concept up close, from a useful angle, or in a way that isolates the important detail.

# d) Technical accuracy
# - The visuals in the clip must correctly reflect what the sentence is describing, without misrepresenting the idea.
# - Choose the clip that most accurately depicts the concept, object, process, detail, situation, etc. referred to in the sentence.

# C) Output requirements: After watching all candidate clips, you must produce your response only in the required XML format. Your output must strictly follow the structure below:

# 1. <evaluation_breakdown> section
# This is your step-by-step reasoning area. Here you must analyze, think, and explain your thought process using the following required subsections:

# - Sentence Understanding: Summarize what the sentence means and what visual idea the learner needs to understand. If the meaning depends on context from the full slide—such as pronouns or earlier references — briefly clarify these to establish the visual target.
# - Clip-by-Clip Observations: Describe exactly what is seen in each candidate clip. For every clip, note key visual elements such as components, objects, processes visible, camera framing and angles, any relevant actions shown, visual clarity or issues, overall instructional usefulness, etc. Each candidate clip must be addressed separately and consistently.
# - Comparative Analysis: Compare the clips directly against each other using the four key dimensions from the guidelines - Visual relevance, Visual clarity and quality, Instructional strength and Technical accuracy. Explain how the clips differ and why one is stronger than the others. Your analysis must show that you reviewed every clip and compared them thoroughly.
# - Decision Rationale: Provide a clear explanation that leads directly to your final selection. Justify why the chosen clip is the strongest match for the sentence and why it is superior to the other candidates. Your reasoning must reference your comparative findings above.

# 2. <comparison> section
# This section contains your final, structured selection output. It must follow exactly the XML structure shown below:

# <comparison>

# <clip_url>
# (The full timestamped URL of the clip selected from the accepted clip input) 
# </clip_url>

# <visual_description>
# (Provide a clear, concise description of what appears on screen of the chosen video clip)
# </visual_description>

# <relevance_mapping>
# (Explain clearly how the visuals in the selected video clip support the sentence. Describe which parts of the imagery correspond to the key idea or action in the sentence, and why this clip is an appropriate visual match)
# </relevance_mapping>

# </comparison>

# So based on the above guidelines regarding the output section, you should always strictly provide your output in the following format:

# <response>

# <evaluation_breakdown>
# (Your detailed reasoning goes here, following all required subsections)
# </evaluation_breakdown>

# <comparison>

# <clip_url>
# (The selected clip URL)
# </clip_url>

# <visual_description>
# (The visual description of the selected clip)
# </visual_description>

# <relevance_mapping>
# (The relevance mapping of the selected clip)
# </relevance_mapping>

# </comparison>

# </response>
# """


# class ClipEvaluationOutput(TypedDict, total=False):
#     status: str
#     visual_description: str
#     relevance_mapping: str
#     recommendation: Optional[str]
#     evaluation_breakdown: str
#     raw_response: str
#     clip_url: str
#     start_seconds: int
#     end_seconds: Optional[int]
#     selected_window_start: Optional[int]
#     selected_window_end: Optional[int]


# class ClipComparisonOutput(TypedDict, total=False):
#     clip_url: str
#     visual_description: str
#     relevance_mapping: str
#     observations: str
#     comparison: str
#     decision: str
#     raw_response: str


# class ComparisonCandidate(TypedDict, total=False):
#     clip_url: str
#     visual_description: str
#     relevance_mapping: str
#     start_seconds: Optional[int]
#     end_seconds: Optional[int]


# def parse_evaluation_xml(xml_text):
#     """
#     Parse the evaluate_video_clip tool's output XML using regex extraction 

#     :param xml_text: The XML response text from the LLM.
#     :return: A dictionary containing parsed evaluation fields.
#     """
    
#     # Parse the XML text
#     text = (xml_text or "").strip()

#     # Parse the status
#     status_match = re.search(r"<status>\s*(.*?)\s*</status>", text, flags=re.IGNORECASE | re.DOTALL)
#     status = (status_match.group(1).strip() if status_match else "").lower() or "error"
    
#     # Parse the visual description
#     visual_match = re.search(r"<visual_description>\s*(.*?)\s*</visual_description>", text, flags=re.IGNORECASE | re.DOTALL)
#     visual_description = visual_match.group(1).strip() if visual_match else ""
    
#     # Parse the relevance mapping
#     relevance_match = re.search(r"<relevance_mapping>\s*(.*?)\s*</relevance_mapping>", text, flags=re.IGNORECASE | re.DOTALL)
#     relevance_mapping = relevance_match.group(1).strip() if relevance_match else ""
    
#     # Parse the evaluation breakdown
#     eval_match = re.search(r"<evaluation_breakdown>\s*(.*?)\s*</evaluation_breakdown>", text, flags=re.IGNORECASE | re.DOTALL)
#     evaluation_breakdown = eval_match.group(1).strip() if eval_match else ""

#     # Parse the selected window
#     selected_start: Optional[int] = None
#     selected_end: Optional[int] = None
#     sw_match = re.search(r"<selected_window>\s*(.*?)\s*</selected_window>", text, flags=re.IGNORECASE | re.DOTALL)
#     sw_raw = sw_match.group(1).strip() if sw_match else ""
#     if sw_raw:
#         times_match = re.search(
#             r"start\s*=\s*(\d+)\s*end\s*=\s*(\d+)",
#             sw_raw,
#             flags=re.IGNORECASE,
#         )
#         if times_match:
#             selected_start = int(times_match.group(1))
#             selected_end = int(times_match.group(2))

#     # Parse recommendation
#     rec_match = re.search(r"<recommendations>\s*(.*?)\s*</recommendations>", text, flags=re.IGNORECASE | re.DOTALL)
#     recommendation = (rec_match.group(1).strip() if rec_match else "") or None

#     # Return the parsed evaluation output
#     return ClipEvaluationOutput(
#         status=status,
#         visual_description=visual_description,
#         relevance_mapping=relevance_mapping,
#         recommendation=recommendation,
#         evaluation_breakdown=evaluation_breakdown,
#         raw_response=xml_text,
#         selected_window_start=selected_start,
#         selected_window_end=selected_end,
#     )


# def parse_comparison_xml(xml_text):
#     """
#     Parse the compare_video_clips tool's output XML using regex extraction.

#     :param xml_text: The XML response text from the LLM.
#     :return: A dictionary containing parsed comparison fields.
#     """

#     # Parse the XML text
#     text = (xml_text or "").strip()

#     # Parse the clip URL
#     clip_match = re.search(r"<clip_url>\s*(.*?)\s*</clip_url>", text, flags=re.IGNORECASE | re.DOTALL)
#     clip_url = clip_match.group(1).strip() if clip_match else ""
    
#     # Parse the visual description
#     visual_match = re.search(r"<visual_description>\s*(.*?)\s*</visual_description>", text, flags=re.IGNORECASE | re.DOTALL)
#     visual_description = visual_match.group(1).strip() if visual_match else ""
    
#     # Parse the relevance mapping
#     relevance_match = re.search(r"<relevance_mapping>\s*(.*?)\s*</relevance_mapping>", text, flags=re.IGNORECASE | re.DOTALL)
#     relevance_mapping = relevance_match.group(1).strip() if relevance_match else ""
    
#     # Parse the observations
#     obs_match = re.search(r"<observations>\s*(.*?)\s*</observations>", text, flags=re.IGNORECASE | re.DOTALL)
#     observations = obs_match.group(1).strip() if obs_match else ""
    
#     # Parse the comparison
#     comp_match = re.search(r"<comparison>\s*(.*?)\s*</comparison>", text, flags=re.IGNORECASE | re.DOTALL)
#     comparison_text = comp_match.group(1).strip() if comp_match else ""
    
#     # Parse the decision
#     dec_match = re.search(r"<decision>\s*(.*?)\s*</decision>", text, flags=re.IGNORECASE | re.DOTALL)
#     decision_text = dec_match.group(1).strip() if dec_match else ""

#     # Return the parsed comparison output
#     return ClipComparisonOutput(
#         clip_url=clip_url,
#         visual_description=visual_description,
#         relevance_mapping=relevance_mapping,
#         observations=observations,
#         comparison=comparison_text,
#         decision=decision_text,
#         raw_response=xml_text,
#     )


# def invoke_gemini(parts, llm, temperature=0.7):
#     """
#     Invoke the Gemini API with the given parts and configuration.

#     :param parts: List of Gemini Part objects (text, video, etc.).
#     :param llm: Model identifier to use.
#     :param temperature: Temperature setting for generation.
#     :return: Text response from the model.
#     """
    
#     client = genai.Client()
#     model = "gemini-2.5-flash"
#     response = client.models.generate_content(
#         model=model,
#         contents=types.Content(parts=parts),
#         config=types.GenerateContentConfig(
#             temperature=temperature,
#         ),
#     )
#     if hasattr(response, "text") and response.text:
#         return response.text
#     if getattr(response, "candidates", None):
#         first_candidate = response.candidates[0]
#         if getattr(first_candidate, "content", None) and first_candidate.content.parts:
#             part = first_candidate.content.parts[0]
#             if hasattr(part, "text"):
#                 return part.text
#     return str(response)


# @tool("evaluate_video_clip", parse_docstring=True)
# def evaluate_video_clip(course_name, target_audience, topic_name, subtopic_name, slide_title, full_slide_text, sentence_text, clip_url, start_seconds, end_seconds, llm="gemini_2_5_flash", is_final_adjustment_attempt=False):
#     """
#     Watch a timestamped clip and assess whether it visually supports the slide sentence.

#     :param course_name: Name of the course.
#     :param target_audience: Target audience for the course.
#     :param topic_name: Topic associated with the slide.
#     :param subtopic_name: Subtopic associated with the slide.
#     :param slide_title: Title of the slide.
#     :param full_slide_text: The full slide content from which the sentence is extracted.
#     :param sentence_text: The narration sentence to support with visuals.
#     :param clip_url: YouTube embed URL of the clip.
#     :param start_seconds: Start time in seconds for the evaluated window.
#     :param end_seconds: End time in seconds for the evaluated window.
#     :param llm: Gemini model identifier to use for multimodal video evaluation.
#     :param is_final_adjustment_attempt: If True, model knows it cannot use "adjust" and must choose accept/reject.
#     :return: Parsed evaluation output including status, visual_description, relevance_mapping, optional recommendation, evaluation_breakdown, raw_response, clip_url, start_seconds, end_seconds, selected_window_start, selected_window_end.
#     """

#     adjustment_context = ""
#     if is_final_adjustment_attempt:
#         adjustment_context = """<adjustment_status>
# IMPORTANT: No more adjustments allowed: This particular clip has already been evaluated and given recommendations for timestamp adjustments previously. You must now strictly make a definitive decision: either "accept" the clip or "reject" it entirely. Do not suggest "adjust" status anymore - choose only "accept" or "reject".
# </adjustment_status>"""

#     prompt_text = evaluate_clip_prompt.format(
#         course_name=course_name,
#         target_audience=target_audience,
#         topic_name=topic_name,
#         subtopic_name=subtopic_name,
#         slide_title=slide_title,
#         full_slide_text=full_slide_text,
#         sentence_text=sentence_text,
#         clip_url=clip_url,
#         start_seconds=start_seconds,
#         end_seconds=end_seconds,
#         adjustment_context=adjustment_context,
#     )
#     parts = [
#         build_video_part(clip_url, start_seconds, end_seconds),
#         types.Part(text=prompt_text),
#     ]
#     xml_text = invoke_gemini(parts, llm, temperature=0.7)

#     parsed = parse_evaluation_xml(xml_text)
#     parsed["clip_url"] = clip_url
#     selected_start = parsed.pop("selected_window_start", None)
#     selected_end = parsed.pop("selected_window_end", None)
#     parsed["start_seconds"] = selected_start if selected_start is not None else start_seconds
#     parsed["end_seconds"] = selected_end if selected_end is not None else end_seconds
#     return parsed


# def parse_recommendation_text(recommendation):
#     """
#     Parse timestamp adjustment recommendation text.

#     :param recommendation: Recommendation text such as 'extend_start_to_<seconds>' or 'extend_end_to_<seconds>'.
#     :return: Dictionary with 'start' and 'end' keys containing target seconds or None.
#     """
#     recommendation = (recommendation or "").strip()
#     if not recommendation:
#         return {"start": None, "end": None}

#     matches = re.findall(r"(extend_(?:start|end)_to)_([0-9]+)", recommendation)
#     targets: Dict[str, Optional[int]] = {"start": None, "end": None}
#     for token, value in matches:
#         seconds = int(value)
#         if "extend_start" in token:
#             targets["start"] = seconds
#         elif "extend_end" in token:
#             targets["end"] = seconds
#     return targets


# @tool("adjust_video_clip_window", parse_docstring=True)
# def adjust_video_clip_window(clip_url, original_start, original_end, recommendation):
#     """
#     Adjusts the video clip window timestamps based on the evaluator's recommendation.

#     :param clip_url: URL of the video clip.
#     :param original_start: Original start time (seconds).
#     :param original_end: Original end time (seconds).
#     :param recommendation: Recommendation text such as 'extend_start_to_<seconds>', 'extend_end_to_<seconds>', or 'extend_start_to_<seconds>_and_extend_end_to<seconds>'.
#     :return: Dict with adjusted window: 'clip_url', 'start_seconds', 'end_seconds'.
#     """
    
#     targets = parse_recommendation_text(recommendation)

#     new_start = original_start if targets["start"] is None else targets["start"]
#     new_end = original_end if targets["end"] is None else targets["end"]

#     if new_end is not None and new_end <= new_start:
#         new_end = new_start + 1

#     return {
#         "clip_url": clip_url,
#         "start_seconds": int(new_start),
#         "end_seconds": int(new_end) if new_end is not None else int(new_start + 1),
#     }


# def build_candidate_video_parts(candidate_clips):
#     """
#     Build Gemini video parts for all candidate clips.

#     :param candidate_clips: List of candidate clip dictionaries.
#     :return: List of Gemini Part objects containing video data and metadata.
#     """
    
#     parts: List[types.Part] = []
#     for idx, candidate in enumerate(candidate_clips, start=1):
#         clip_url = candidate.get("clip_url")
#         if not clip_url:
#             continue
#         description = (
#             f"Candidate {idx}:\n"
#             f"URL: {clip_url}\n"
#         )
#         parts.append(types.Part(text=description))
#         parts.append(
#             build_video_part(
#                 clip_url,
#                 candidate.get('start_seconds'),
#                 candidate.get('end_seconds'),
#             )
#         )
#     return parts


# @tool("compare_video_clips", parse_docstring=True)
# def compare_video_clips(course_name, target_audience, topic_name, subtopic_name, slide_title, full_slide_text, sentence_text, candidate_clips, llm="gemini_2_5_flash"):
#     """
#     Compare acceptable clips for a sentence and select the strongest visual match.

#     :param course_name: Name of the course.
#     :param target_audience: Target audience for the course.
#     :param topic_name: Topic associated with the slide.
#     :param subtopic_name: Subtopic associated with the slide.
#     :param slide_title: Title of the slide containing the sentence.
#     :param full_slide_text: The full slide content from which the sentence is extracted.
#     :param sentence_text: The sentence that each candidate should support.
#     :param candidate_clips: List of candidate clip dictionaries.
#     :param llm: Gemini model identifier to use for multimodal video comparison.
#     :return: Parsed comparison output including clip_url, visual_description, relevance_mapping, observations, comparison, decision, raw_response.
#     """

#     formatted_blocks = []
#     for idx, candidate in enumerate(candidate_clips, start=1):
#         block = (
#             f"Candidate {idx}:\n"
#             f"URL: {candidate.get('clip_url')}\n"
#         )
#         formatted_blocks.append(block.strip())
#     candidate_block = "\n\n".join(formatted_blocks)
    
#     prompt = compare_clips_prompt.format(
#         course_name=course_name,
#         target_audience=target_audience,
#         topic_name=topic_name,
#         subtopic_name=subtopic_name,
#         slide_title=slide_title,
#         full_slide_text=full_slide_text,
#         sentence_text=sentence_text,
#         candidate_block=candidate_block,
#     )

#     candidate_parts = build_candidate_video_parts(candidate_clips)
#     xml_text = invoke_gemini(candidate_parts + [types.Part(text=prompt)], llm, temperature=0.7)

#     return parse_comparison_xml(xml_text)