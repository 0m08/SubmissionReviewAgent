"""
Supabase pgvector service for SkillCat course vectorstore.
Provides vector similarity search and full-text search against the
courses table, returning LangChain Document objects for compatibility
with the existing retrieval pipeline.
"""

import os
import logging
import requests
from typing import Any, Dict, List
from langchain_classic.schema import Document
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

SUPABASE_URL = "https://mdiulditosmnaqhimcdh.supabase.co"

# Fields returned by the RPC functions that should go into Document.metadata
# Array fields from Postgres (text[]) need to be joined into strings for
# compatibility with the existing Chroma-based pipeline.
_ARRAY_FIELDS = {
    "course_topics",
    "course_learning_objectives",
    "course_keywords",
    "course_processes_covered",
    "course_equipment_covered",
    "course_practice_scenarios",
    "course_competencies_level1",
    "course_competencies_level2",
}

_METADATA_FIELDS = [
    "course_id", "course_name", "course_description", "course_link",
    "course_category", "course_topics", "course_learning_objectives",
    "course_duration_hours", "course_prerequisites", "course_keywords",
    "course_processes_covered", "course_equipment_covered",
    "course_practice_scenarios", "course_competencies_level1",
    "course_competencies_level2", "nps_score", "course_image", "course_status",
]


def _get_headers() -> Dict[str, str]:
    api_key = os.getenv("SUPABASE_SECRET_API_KEY")
    if not api_key:
        raise ValueError("SUPABASE_SECRET_API_KEY environment variable is not set")
    return {
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _row_to_document(row: Dict[str, Any]) -> Document:
    """Convert a Supabase RPC result row to a LangChain Document."""
    metadata = {}
    for field in _METADATA_FIELDS:
        value = row.get(field)
        if value is None:
            continue
        # Join array fields into comma-separated strings
        if field in _ARRAY_FIELDS and isinstance(value, list):
            metadata[field] = ", ".join(str(v) for v in value)
        else:
            metadata[field] = value

    return Document(
        page_content=row.get("page_content", ""),
        metadata=metadata,
    )


def vector_search(query_embedding: List[float], k: int = 20) -> List[Document]:
    """
    Perform cosine similarity search via the match_courses RPC function.
    Returns LangChain Document objects.
    """
    url = f"{SUPABASE_URL}/rest/v1/rpc/match_courses"
    headers = _get_headers()

    resp = requests.post(url, headers=headers, json={
        "query_embedding": query_embedding,
        "match_count": k,
    })
    resp.raise_for_status()
    rows = resp.json()

    return [_row_to_document(row) for row in rows]


def fulltext_search(query: str, k: int = 20) -> List[Document]:
    """
    Perform Postgres full-text search via the fulltext_search_courses RPC.
    Returns LangChain Document objects.
    """
    url = f"{SUPABASE_URL}/rest/v1/rpc/fulltext_search_courses"
    headers = _get_headers()

    resp = requests.post(url, headers=headers, json={
        "search_query": query,
        "match_count": k,
    })
    resp.raise_for_status()
    rows = resp.json()

    return [_row_to_document(row) for row in rows]
