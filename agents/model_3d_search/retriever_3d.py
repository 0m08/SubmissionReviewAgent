from typing import List, Dict, Any, Optional, Union
from PIL import Image

from agents.model_3d_search.vectorstore_3d import (
    load_3d_chroma_db,
    embed_text_gemini,
    embed_image_gemini,
    CENTRAL_3D_MODELS_FOLDER_ID,
)


def search_vectorstore_3d(
    search_type: str = "text",
    query_text: Optional[str] = None,
    query_image: Optional[Union[Image.Image, bytes]] = None,
    category_filter: Optional[Union[str, List[str]]] = None,
    extension_filter: Optional[Union[str, List[str]]] = None,
    top_n: int = 10,
    drive=None,
    parent_folder_id: str = CENTRAL_3D_MODELS_FOLDER_ID,
) -> List[Dict[str, Any]]:
    """
    Direct vector similarity search against single `3d_models_unified` ChromaDB collection.

    :param search_type: 'text' (Search by Model Name) or 'image' (Search by Preview Image).
    :param query_text: Text string query.
    :param query_image: Uploaded reference image or file bytes.
    :param category_filter: Category metadata filter (str or List[str]).
    :param extension_filter: Extension metadata filter (str or List[str]).
    :param top_n: Top N candidate assets to return ordered directly by vector similarity.
    """
    dbs = load_3d_chroma_db(drive=drive, parent_folder_id=parent_folder_id)
    unified_chroma = dbs["unified"]

    # Construct Chroma metadata filter dict (supporting multi-select lists)
    filter_conditions = []

    if category_filter:
        cats = [c for c in ([category_filter] if isinstance(category_filter, str) else category_filter) if c and c != "All"]
        if len(cats) == 1:
            filter_conditions.append({"category": cats[0]})
        elif len(cats) > 1:
            filter_conditions.append({"category": {"$in": cats}})

    if extension_filter:
        exts = [e for e in ([extension_filter] if isinstance(extension_filter, str) else extension_filter) if e and e != "All"]
        if len(exts) == 1:
            filter_conditions.append({"extension": exts[0]})
        elif len(exts) > 1:
            filter_conditions.append({"extension": {"$in": exts}})

    chroma_filter = None
    if len(filter_conditions) == 1:
        chroma_filter = filter_conditions[0]
    elif len(filter_conditions) > 1:
        chroma_filter = {"$and": filter_conditions}

    # 1. Generate Query Vector
    if search_type == "text" and query_text:
        query_vector = embed_text_gemini(query_text)
    elif search_type == "image" and query_image is not None:
        query_vector = embed_image_gemini(query_image)
    else:
        return []

    # 2. Query Collection Directly
    total_docs = unified_chroma._collection.count()
    if total_docs == 0:
        return []

    query_kwargs = {
        "query_embeddings": [query_vector],
        "n_results": min(top_n, total_docs, 50),
        "include": ["metadatas", "distances"],
    }
    if chroma_filter:
        query_kwargs["where"] = chroma_filter

    try:
        raw_res = unified_chroma._collection.query(**query_kwargs)
    except Exception as e:
        print(f"⚠️ Vectorstore search exception: {e}")
        return []

    results = []
    if raw_res and "metadatas" in raw_res and raw_res["metadatas"]:
        metadatas = raw_res["metadatas"][0]
        distances = raw_res.get("distances", [[]])[0]

        for meta, dist in zip(metadatas, distances):
            item = dict(meta)
            # Distance -> similarity conversion
            item["relevance_score"] = round(float(1.0 - dist) if dist <= 1.0 else float(1.0 / (1.0 + dist)), 4)
            results.append(item)

    return results
