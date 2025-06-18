import base64
from io import BytesIO
from PIL.Image import Image as PILImageType
from typing import List, Dict, Any
from modules.chain import Chain
from agents.vector_store_image_search.graphics_retriever import graphics_retriever

graphics_retriever_agent_prompt = """
You are an expert in visual content evaluation, tasked with selecting the most visually relevant image(s) to match a given query. You have access to a vector database that returns both image metadata and the actual image content.

Your goal is to select up to k image(s) requested by the user, that best match the query based on visual alignment — not just text. If no good images are found, you may refine the query and re-search up to 3 times. If nothing fits after all attempts, return "NONE".

For each search turn, you will receive:
- The search query used to retrieve results.
- A list of image search results.

Your responsibilities are as follows:

---

1. VISUAL INSPECTION

- Carefully examine the actual images.
- Use visual reasoning to understand what is shown (e.g., objects, diagrams, environments, actions).
- Use metadata (title, description) only to support what you *see*.
- Ignore high similarity scores if the visuals don't match the intent.

---

2. IF RELEVANT IMAGES ARE FOUND

- If k strong visual matches are found, return those.
- Set the verdict to "TERMINATE".
- Output the selected image(s) as rendered images (optional), and return their indexes.

---

3. IF RESULTS ARE POOR OR IRRELEVANT

- Determine why the query underperformed.
- Suggest how to reword or refine it for better visual alignment.
- This may involve:
  - Disambiguating vague terms (e.g., "ventilation" → "airflow through HVAC duct")
  - Adding context (e.g., setting, action, device type)
  - Substituting synonyms or clarifying the intent (e.g., “technician fixing ductwork”)

- Set the verdict to "CONTINUE" and provide the refined query.

---

4. IF AFTER 3 TURNS NO MATCH IS FOUND

- Set verdict to "TERMINATE" and return "NONE".

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
 If TERMINATE: briefly state that the query was sufficient, or that no relevant images were found.]
</action>

<query>
[If CONTINUE: provide your improved, more visual search query.
 If TERMINATE: repeat the original query to confirm no refinement was needed.]
</query>

Be sure to wrap each section inside the correct tags exactly as shown above. Do not include Markdown or extra text.
"""


image_relevance_prompt = ("""
You are an expert in evaluating images for relevance to a search query.

You will be shown:
- A search query
- A set of top-k candidate images

Your job is to:
1. Visually inspect each image.
2. Comment on whether the image matches the query intent.
3. Choose the best matching images by their index (starting from 0).
4. If none fit well, suggest a better query.

Respond in this exact XML format:

<observations>
<Your reasoning for each image, e.g., "Image 0 shows HVAC equipment, relevant to 'air conditioner'">
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



def pil_to_base64(img):
    """
    Converts a PIL image to a base64-encoded data URI.
    :param img: PIL image.
    :return: Base64-encoded data URI.
    """
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


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
    Prepares prompt content for visualization.
    :param images: List of image objects.
    :param formatted_prompt: Prompt template.
    :return: List of prompt parts.
    """
    parts = []

    # Add the prompt text
    parts.append({
        "type": "text",
        "text": formatted_prompt.strip()
    })

    # Add images
    for i, item in enumerate(images):
        img = item.get("image")
        if isinstance(img, PILImageType):
            parts.append({
                "type": "image_url",
                "image_url": pil_to_base64_data_uri(img)
            })
            parts.append({
                "type": "text",
                "text": f"Image {i+1} — Similarity: {item.get('similarity', 0.0):.4f}"
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
    chain.add_message(role="system", content=graphics_retriever_agent_prompt)

    for turn in range(max_turns):
        if verbose:
            print(f"\nTurn {turn + 1}: Query = '{query}'")

        # Step 1: Retrieve top-k image metadata + PIL objects
        results = graphics_retriever(query, drive=drive, k=k, filters=filters)

        if not results:
            print("No results returned.")
            return []

        # Step 2: Format prompt from template
        formatted_prompt = image_relevance_prompt.format(query=query)

        # (base64 + "image_url" part type)
        content_parts = prepare_images_for_llm(results, formatted_prompt=formatted_prompt)

        if verbose:
            print(f"Sending {len(results)} images to LLM for evaluation...")

        # Step 4: Add to chain and call LLM
        # Flatten all parts into a single string prompt
        flat_prompt = formatted_prompt + "\n\n"
        for i, item in enumerate(results):
            img_meta = item.get("metadata", {})
            flat_prompt += f"Image {i}: {img_meta.get('image_title', 'Untitled')} — {img_meta.get('description', '')}\n"

        chain.add_message(role="user", content=flat_prompt)

        llm_raw = chain.run()

        # If it's a dict, unwrap `.content`
        if isinstance(llm_raw, dict):
            content = llm_raw.get("content") or llm_raw.get("text", "")
        else:
            content = llm_raw  # plain string

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
            final_images_with_metadata = [
                {
                    "image": results[i]["image"],
                    "metadata": results[i].get("metadata", {})
                }
                for i in selected_indexes if 0 <= i < len(results)
            ]

            return final_images_with_metadata if final_images_with_metadata else []



        # Step 6: Refine query if needed
        query = llm_response.get("query", "").strip()
        if not query:
            print("LLM returned CONTINUE but gave no new query.")
            return []

    print("Max turns reached. Returning no selection.")
    return []