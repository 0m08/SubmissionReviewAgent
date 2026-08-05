"""
Supabase + Gemini Embedding 2 helpers for External Reference assets.

Talks ONLY to public.external_ref_assets and match_external_ref_assets.
Never calls curriculum tables or match_courses.
"""

import hashlib
import os
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Same Content_Courses project as curriculum mapping (hardcoded URL; key from env).
SUPABASE_URL = "https://mdiulditosmnaqhimcdh.supabase.co"
TABLE_NAME = "external_ref_assets"
MATCH_RPC = "match_external_ref_assets"

GEMINI_MODEL_NAME = "gemini-embedding-2"
OUTPUT_DIMENSIONALITY = 3072


def _get_supabase_headers():
    """
    Build HTTP headers for Supabase REST requests using the secret API key.

    :return: Dict with apikey, Authorization, Content-Type, and Prefer headers
    """
    api_key = (os.getenv("SUPABASE_SECRET_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("SUPABASE_SECRET_API_KEY is not set in .env")
    return {
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=representation",
    }


def _get_gemini_client():
    """
    Create a Gemini client using GOOGLE_API_KEY from the environment.

    :return: Configured genai.Client instance
    """
    api_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set in .env")
    return genai.Client(api_key=api_key)


def make_asset_id(sheet_id, source_link, asset_url, start_time=None, end_time=None):
    """
    Build a stable asset id for upserts from sheet, source, URL, and optional clip times.

    :param sheet_id: Course spreadsheet id
    :param source_link: External reference source link
    :param asset_url: Asset URL (Drive or YouTube)
    :param start_time: Optional clip start time in seconds
    :param end_time: Optional clip end time in seconds
    :return: 32-character hex SHA-256 digest
    """
    raw = f"{sheet_id}|{source_link.strip()}|{asset_url.strip()}|{start_time}|{end_time}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def embed_image_bytes(image_bytes, mime_type="image/jpeg", text_prefix=""):
    """
    Embed one image with Gemini Embedding 2 (document / retrieval side).

    :param image_bytes: Raw image bytes
    :param mime_type: MIME type for the image (default image/jpeg)
    :param text_prefix: Optional text prepended before the image part
    :return: Embedding vector as a list of floats
    """
    if not image_bytes:
        raise ValueError("image_bytes is empty")
    client = _get_gemini_client()
    contents = []
    if text_prefix and text_prefix.strip():
        contents.append(text_prefix.strip())
    contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type or "image/jpeg"))
    result = client.models.embed_content(
        model=GEMINI_MODEL_NAME,
        contents=contents,
        config=types.EmbedContentConfig(
            output_dimensionality=OUTPUT_DIMENSIONALITY,
            task_type="RETRIEVAL_DOCUMENT",
        ),
    )
    values = result.embeddings[0].values
    return list(values)


def embed_video_bytes(video_bytes, mime_type="video/mp4", text_prefix=""):
    """
    Embed one video clip with Gemini Embedding 2 (document / retrieval side).

    :param video_bytes: Raw video bytes
    :param mime_type: MIME type for the video (default video/mp4)
    :param text_prefix: Optional text prepended before the video part
    :return: Embedding vector as a list of floats
    """
    if not video_bytes:
        raise ValueError("video_bytes is empty")
    client = _get_gemini_client()
    contents = []
    if text_prefix and text_prefix.strip():
        contents.append(text_prefix.strip())
    contents.append(types.Part.from_bytes(data=video_bytes, mime_type=mime_type or "video/mp4"))
    result = client.models.embed_content(
        model=GEMINI_MODEL_NAME,
        contents=contents,
        config=types.EmbedContentConfig(
            output_dimensionality=OUTPUT_DIMENSIONALITY,
            task_type="RETRIEVAL_DOCUMENT",
        ),
    )
    values = result.embeddings[0].values
    return list(values)


def embed_query_text(query):
    """
    Embed a natural-language search query (query / retrieval side).

    :param query: Search query text
    :return: Embedding vector as a list of floats
    """
    text = (query or "").strip()
    if not text:
        raise ValueError("query is empty")
    client = _get_gemini_client()
    result = client.models.embed_content(
        model=GEMINI_MODEL_NAME,
        contents=[text],
        config=types.EmbedContentConfig(
            output_dimensionality=OUTPUT_DIMENSIONALITY,
            task_type="RETRIEVAL_QUERY",
        ),
    )
    return list(result.embeddings[0].values)


def upsert_asset_row(row):
    """
    Upsert one row into external_ref_assets only.

    :param row: Dict including asset_id, embedding, and metadata fields
    :return: None
    """
    if not row.get("asset_id"):
        raise ValueError("asset_id is required")
    if not row.get("sheet_id"):
        raise ValueError("sheet_id is required")
    payload = dict(row)
    if "indexed_at" not in payload:
        payload["indexed_at"] = datetime.now(timezone.utc).isoformat()
    url = f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}"
    resp = requests.post(url, headers=_get_supabase_headers(), json=payload, timeout=120)
    if not resp.ok:
        raise RuntimeError(f"Supabase upsert failed ({resp.status_code}): {resp.text[:500]}")


def list_source_links_for_sheet(sheet_id):
    """
    List distinct source_link values already indexed for this sheet_id.

    :param sheet_id: Course spreadsheet id
    :return: List of unique source_link strings (empty when sheet_id is missing)
    """
    if not sheet_id:
        return []
    url = f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}"
    params = {
        "select": "source_link",
        "sheet_id": f"eq.{sheet_id}",
    }
    headers = _get_supabase_headers()
    headers["Prefer"] = "return=representation"
    resp = requests.get(url, headers=headers, params=params, timeout=60)
    if not resp.ok:
        raise RuntimeError(f"Supabase list source_links failed ({resp.status_code}): {resp.text[:500]}")
    rows = resp.json() or []
    seen = set()
    out = []
    for row in rows:
        link = (row.get("source_link") or "").strip()
        if link and link not in seen:
            seen.add(link)
            out.append(link)
    return out


def delete_assets_for_source(sheet_id, source_link):
    """
    Delete indexed rows for one External References link on one sheet (external_ref_assets only).

    :param sheet_id: Course spreadsheet id
    :param source_link: External reference source link to delete
    :return: Number of deleted rows (0 when sheet_id or source_link is missing)
    """
    if not sheet_id or not source_link:
        return 0
    url = f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}"
    params = {
        "sheet_id": f"eq.{sheet_id}",
        "source_link": f"eq.{source_link}",
    }
    headers = _get_supabase_headers()
    headers["Prefer"] = "return=representation"
    resp = requests.delete(url, headers=headers, params=params, timeout=60)
    if not resp.ok:
        raise RuntimeError(f"Supabase delete failed ({resp.status_code}): {resp.text[:500]}")
    deleted = resp.json() if resp.content else []
    return len(deleted) if isinstance(deleted, list) else 0


def count_assets_for_source(sheet_id, source_link):
    """
    Count indexed rows for a sheet_id and source_link pair.

    :param sheet_id: Course spreadsheet id
    :param source_link: External reference source link
    :return: Row count (0 when sheet_id or source_link is missing)
    """
    if not sheet_id or not source_link:
        return 0
    url = f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}"
    params = {
        "select": "asset_id",
        "sheet_id": f"eq.{sheet_id}",
        "source_link": f"eq.{source_link}",
    }
    headers = _get_supabase_headers()
    headers["Prefer"] = "count=exact"
    headers["Range-Unit"] = "items"
    headers["Range"] = "0-0"
    resp = requests.get(url, headers=headers, params=params, timeout=60)
    if not resp.ok:
        raise RuntimeError(f"Supabase count failed ({resp.status_code}): {resp.text[:500]}")
    content_range = resp.headers.get("Content-Range") or ""
    # e.g. "*/12" or "0-0/12"
    if "/" in content_range:
        total = content_range.split("/")[-1]
        try:
            return int(total)
        except ValueError:
            pass
    data = resp.json() if resp.content else []
    return len(data) if isinstance(data, list) else 0


def search_external_ref_assets(query, sheet_id, k=20, asset_type=None):
    """
    Text search against external_ref_assets only, filtered by sheet_id.

    :param query: Natural-language search query
    :param sheet_id: Course spreadsheet id (filter before ranking via RPC WHERE)
    :param k: Max results to return
    :param asset_type: Optional filter: 'image' or 'video_segment'
    :return: List of matching asset row dicts from the match RPC
    """
    embedding = embed_query_text(query)
    url = f"{SUPABASE_URL}/rest/v1/rpc/{MATCH_RPC}"
    payload = {
        "query_embedding": embedding,
        "match_count": int(k),
        "filter_sheet_id": sheet_id,
        "filter_asset_type": asset_type,
    }
    resp = requests.post(url, headers=_get_supabase_headers(), json=payload, timeout=120)
    if not resp.ok:
        raise RuntimeError(f"Supabase match RPC failed ({resp.status_code}): {resp.text[:500]}")
    rows = resp.json() or []
    return rows if isinstance(rows, list) else []


def delete_assets_for_sheet(sheet_id):
    """
    Delete all external_ref_assets rows for one spreadsheet id (refuses if sheet_id is missing; curriculum/Drive untouched).

    :param sheet_id: Course spreadsheet id
    :return: Number of deleted rows
    """
    sid = (sheet_id or "").strip()
    if not sid:
        raise ValueError(
            "delete_assets_for_sheet refused: sheet_id is required "
            "(refusing unfiltered delete of external_ref_assets)"
        )
    if TABLE_NAME != "external_ref_assets":
        raise RuntimeError(
            f"Safety abort: unexpected TABLE_NAME '{TABLE_NAME}' "
            "(expected 'external_ref_assets')"
        )

    url = f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}"
    params = {
        "sheet_id": f"eq.{sid}",
    }
    headers = _get_supabase_headers()
    headers["Prefer"] = "return=representation"
    resp = requests.delete(url, headers=headers, params=params, timeout=120)
    if not resp.ok:
        raise RuntimeError(
            f"Supabase delete-by-sheet failed ({resp.status_code}): {resp.text[:500]}"
        )
    deleted = resp.json() if resp.content else []
    return len(deleted) if isinstance(deleted, list) else 0
