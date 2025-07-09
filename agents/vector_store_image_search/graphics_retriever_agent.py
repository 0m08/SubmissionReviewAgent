import base64
from io import BytesIO
from PIL.Image import Image as PILImageType
from typing import List, Dict, Any
from modules.chain import Chain, xml_check_and_fix
from services.llm_service import llm_with_retry
from agents.vector_store_image_search.graphics_retriever import graphics_retriever

graphics_retriever_agent_prompt = """
You are an expert in visual content evaluation, tasked with selecting the most visually relevant image(s) to match a given query.

Your goal is to identify images that best match the intent of the search query based on visual content, not just metadata or similarity scores.

For each search turn, you will receive:
- The search query used to retrieve results.
- A list of image search results.

Your responsibilities are as follows:

---

1. VISUAL INSPECTION

- Carefully examine the actual images.
- Pay close attention to distinguishing features when terms may be visually similar but semantically different. Do not select images that contradict the core terms in the query.
- Use metadata (title, description) only to support what you *see*.
- Ignore high similarity scores if the visuals don't match the intent.

---

2. IF RELEVANT IMAGES ARE FOUND

If you found any clearly relevant and visually matching images, mark them and set verdict = TERMINATE.
Do not suggest refinements if a usable set of results has already been found.

---

3. IF RESULTS ARE POOR OR IRRELEVANT

- Determine why the query underperformed:
  - Is the query vague, overly broad, or ambiguous?
  - Are domain terms present but not contextualized?
  - Is the visual concept too generic to be meaningful?

- If the query is vague but still thematically relevant (e.g., "gear thing", "air moving", "box with lines", "empty room"), you must CONTINUE and refine the query.

- When refining:
  - Disambiguate vague terms (e.g., "ventilation" → "airflow through HVAC duct")
  - Add context (e.g., "gear diagram icon", "wall-mounted disconnect switch", "return air grille airflow schematic")
  - Clarify function, setting, or visual type (e.g., "training diagram", "schematic", "photo of used fuses")
  - Preserve the **intent and scope** of the original query.


- If multiple vague queries fail, fallback to a **context-respecting structured query**:
  - Use visual scene interpretation for general queries.
  - Use common object phrasing (e.g., "a thing to hit nails" → "manual hammer tool in use")
  - Avoid switching to unrelated technical domains (e.g., do NOT interpret "empty room" as needing "mechanical diagram" unless clearly implied)

---

4. IF AFTER 3 REFINED ATTEMPTS NO MATCH IS FOUND

- Set verdict to "TERMINATE" and return "NONE" only if:
  - The query is fundamentally unrelated to visual domains (e.g., abstract emotion, pure text queries)
  - All refinements have failed to yield even loosely related results
  - The subject cannot be visually represented in any technical, general, or instructional form

---

RESPONSE FORMAT:

Respond using these exact tags:

<observations>
Summarize what you saw in the images and their relevance.
</observations>

<verdict>
["TERMINATE" if you're confident in your image selection(s) or none are good at all, otherwise "CONTINUE"]
</verdict>

<selected_indexes>
[If TERMINATE: return the indexes of selected images from the list above, e.g., "0", "1". Use this to identify relevant images. Leave blank if CONTINUE.]
</selected_indexes>

<action>
[If CONTINUE: explain what is wrong with the current query and how you'll refine it.
 If TERMINATE: briefly state that the query was sufficient, or that no relevant images were found after exhaustive attempts.]
</action>

<query>
[If CONTINUE: provide your improved, more visual search query.
 If TERMINATE: repeat the original query to confirm no refinement was needed or indicate failure after 3 attempts.]
</query>

---

Additional guidance for query refinement and termination:

- Only suggest query refinements if the original query is relevant but vague or ambiguous. Refinements should focus on clarifying intent, adding context, or disambiguating terms related to the domain.
- Never shift to a different domain unless explicitly supported by the query itself.
- Avoid using engineering, schematic, HVAC, or symbolic defaults unless the query includes related terms.
- Use the <action> tag to clearly explain your reasoning.
- Always prefer refining over terminating unless you're confident the query is completely outside of all visual domains.
- Always preserve the user's apparent context (scene, object, use-case) when requerying vague inputs.
"""



image_relevance_prompt = ("""
You are an expert in evaluating images for relevance to a search query.The user query may be textual, visual, or both.

You will be shown:
- The original search query: {query}
- A set of top-k candidate images

Your job is to:
1. Visually inspect each image.
2. Comment on whether the image matches the query intent.
3. Choose the best matching images by their index (starting from 0).
4. If none fit well, suggest a better query — but **preserve the context and avoid switching domains**.


Respond in this exact XML format:

<observations>
<Your reasoning for each image.">
</observations>

<verdict>
TERMINATE or CONTINUE
</verdict>

<selected_indexes>
e.g. 0, 2
</selected_indexes>

<action>
If CONTINUE: explain what is wrong with the current query and how to refine it. Leave empty if TERMINATE.
</action>

<query>
Refined query if verdict is CONTINUE. Leave empty if verdict is TERMINATE.
</query>
""")


def pil_to_base64_data_uri(image: PILImageType) -> str:
    
    """
    Converts a PIL image to a data URI.
    :param image: PIL image.
    :return: Data URI.
    """
    buffered = BytesIO()
    image.convert("RGB").save(buffered, format="JPEG")
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def prepare_images_for_llm(
    images: List[Dict],
    formatted_prompt: str
) -> List[Dict[str, Any]]:
    """
    Prepares prompt content for vision LLM. 
    Only sends images and the core prompt
    
    :param images: List of image dictionaries with PIL images under 'image' key.
    :param formatted_prompt: Main prompt string to be shown before the images.
    :return: List of prompt parts with 'text' and 'image_url' elements.
    """
    parts = []

    # Add the prompt instruction as text
    parts.append({
        "type": "text",
        "text": formatted_prompt.strip()
    })

    # Add each image as a base64 data URI
    for item in images:
        img = item.get("image")
        if isinstance(img, PILImageType):
            parts.append({
                "type": "image_url",
                "image_url": pil_to_base64_data_uri(img)
            })

    return parts



def parse_selected_indexes(index_string):
    """
    Safely extract index list from string like '0, 2' or '1'
    :param index_string: String to parse.
    :return: List of indexes.
    """
    try:
        return [int(i.strip()) for i in index_string.split(",") if i.strip().isdigit()]
    except:
        return []



def graphics_retriever_agent(
    query,
    drive,
    llm,
    k,
    query_image: Any,
    max_turns: int = 3,
    filters: dict = None,
    verbose: bool = True
):
    """
    LLM-driven image selection agent using actual image objects and a multi-turn refinement process.
    :param query: Initial search query.
    :param drive: Google Drive instance for image retrieval.
    :param llm: Language model for reasoning and evaluation.
    :param k: Number of top results to retrieve per turn.
    :param max_turns: Maximum number of refinement turns.
    :param filters: Optional filters for image metadata (e.g., mime_type, image_title).
    :return: List of selected images with metadata, or empty if none found.
    """

    # Initialize custom chain for vision reasoning
    chain = Chain(
        llm=llm,
        tags=["observations", "verdict", "selected_indexes", "action", "query"],
        use_xml_checker=True
    )
    original_query = query

    chain.add_message(role="system", content=graphics_retriever_agent_prompt)

    for turn in range(max_turns):
        if verbose:
            print(f"\nTurn {turn + 1}: Query = '{query}'")

        # Step 1: Retrieve top-k image metadata + PIL objects
        results = graphics_retriever(query, query_image, drive=drive, k=k, filters=filters)

        if not results:
            print("No results returned.")
            return []

        # Step 2: Format prompt from template
        formatted_prompt = image_relevance_prompt.format(query=original_query)



        if verbose:
            print(f"Sending {len(results)} images to LLM for evaluation...")

         # Step 4: Prepare message list and call LLM directly
        content_parts = prepare_images_for_llm(results, formatted_prompt=formatted_prompt)
        chain.add_message(role="user", content=content_parts)

        # Call the LLM with the raw conversation
        llm_raw = llm_with_retry(chain.messages_list, llm_name=llm)

        # Normalize raw output to a plain string
        if hasattr(llm_raw, "content"):
            raw_text = llm_raw.content
        elif isinstance(llm_raw, dict):
            raw_text = llm_raw.get("content") or llm_raw.get("text", "")
        else:
            raw_text = llm_raw

        # Validate XML if enabled (mirrors Chain.run behaviour)
        if chain.use_xml_checker:
            content = xml_check_and_fix(raw_text, llm=llm)
        else:
            content = raw_text

        # Save the AI response back to the conversation history
        chain.add_message(role="ai", content=content)

        # Now extract XML tags from string content
        try:
            llm_response = chain.extract_text_in_tags(content)
        except Exception as e:
            print("Failed to extract tags from LLM response.")
            print("Raw content:\n", content)
            raise e


        if verbose:
            print("\nLLM Response:")
            for tag in ["observations", "verdict", "selected_indexes", "action", "query"]:
                print(f"{tag.capitalize()}: {llm_response.get(tag, '')}")

        # Step 5: Handle output from LLM
        verdict = llm_response.get("verdict", "").strip().upper()
        if verdict == "TERMINATE":
            selected_indexes = parse_selected_indexes(llm_response.get("selected_indexes", ""))
            # Filter only the image and its metadata
            final_images_with_metadata = []
            for i in selected_indexes:
                if 0 <= i < len(results):
                    metadata = results[i].get("metadata", {})
                    final_images_with_metadata.append({
                        "image": results[i]["image"],
                        "metadata": metadata,
                        "url": metadata.get("url", "")  # Optional: promote URL to top level
                    })

            return final_images_with_metadata if final_images_with_metadata else []



        # Step 6: Refine query if needed
        query = llm_response.get("query", "").strip()
        if not query:
            print("LLM returned CONTINUE but gave no new query.")
            return []

    print("Max turns reached. Returning no selection.")
    return []