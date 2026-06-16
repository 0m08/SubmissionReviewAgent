# SkillCat Vectorstore Migration: ChromaDB to Supabase pgvector

## Table of Contents

1. [Overview](#overview)
2. [Why We Migrated](#why-we-migrated)
3. [Architecture](#architecture)
4. [Prerequisites](#prerequisites)
5. [Step-by-Step: Supabase Database Setup](#step-by-step-supabase-database-setup)
6. [Step-by-Step: Edge Function for Auto-Embedding](#step-by-step-edge-function-for-auto-embedding)
7. [Step-by-Step: Python Code Changes](#step-by-step-python-code-changes)
8. [Step-by-Step: Backfill Existing Data](#step-by-step-backfill-existing-data)
9. [How the Retrieval Pipeline Works End-to-End](#how-the-retrieval-pipeline-works-end-to-end)
10. [Key Design Decisions and Gotchas](#key-design-decisions-and-gotchas)
11. [Files Summary](#files-summary)
12. [How to Revert to ChromaDB](#how-to-revert-to-chromadb)
13. [How to Replicate for Another Table](#how-to-replicate-for-another-table)
14. [Troubleshooting](#troubleshooting)

---

## Overview

This document describes the migration of the SkillCat course vectorstore from a local ChromaDB instance (stored as snapshots on Google Drive) to a live Supabase pgvector-backed vectorstore with automatic embedding.

**What changed:** SkillCat course retrieval now queries the live `courses` table in Supabase instead of a static ChromaDB snapshot.

**What didn't change:** NexTech and Video vectorstores remain on ChromaDB. The retrieval pipeline (EnsembleRetriever + CohereRerank) is unchanged — only the underlying data source for SkillCat was swapped.

**Supabase project:** `https://mdiulditosmnaqhimcdh.supabase.co`

**Embedding model:** Cohere `embed-english-v3.0` (1024 dimensions)

---

## Why We Migrated

### Problems with ChromaDB Approach

1. **Stale snapshots:** The ChromaDB vectorstore was built from a Google Sheet ("1 Dec" tab), persisted locally, then uploaded to Google Drive. Any new courses added to the sheet required a manual rebuild and re-upload via `create_skillcat_vectorstore.py`.
2. **BM25 pickle staleness:** The BM25 keyword retriever was serialized as a pickle file on Google Drive. It went out of sync whenever the Chroma vectorstore was updated without also regenerating the pickle.
3. **Slow cold starts:** On first use, the system had to download the entire Chroma folder (~SQLite DB) + BM25 pickle from Google Drive to `/tmp/`, adding significant latency.
4. **Duplicate data:** Course data lived in the Google Sheet, the Supabase `courses` table, AND the Chroma vectorstore — three copies to keep in sync.

### Benefits of Supabase pgvector

1. **Always up-to-date:** The `courses` table is the single source of truth. When a course is added or updated, it's immediately searchable.
2. **Auto-embedding:** A Supabase Edge Function automatically generates embeddings via Cohere whenever a row is inserted or updated — no manual rebuild step.
3. **No file management:** No more downloading/uploading Chroma folders or pickle files from Google Drive.
4. **Built-in full-text search:** Postgres `tsvector` replaces the BM25 pickle, and it's always in sync with the data.

---

## Architecture

### Before (ChromaDB)

```
Google Sheet ("1 Dec" tab)
    │
    ▼ (manual script: create_skillcat_vectorstore.py)
ChromaDB (local SQLite) ──► Upload to Google Drive
    │
    ▼ (on query: download from Drive to /tmp/)
Chroma Retriever (k=20) ──┐
                           ├── EnsembleRetriever (50/50) ──► CohereRerank ──► Results
BM25 Pickle (from Drive) ─┘
```

### After (Supabase pgvector)

```
Supabase `courses` table (single source of truth)
    │
    ├── ON INSERT/UPDATE (synchronous, ~0ms):
    │   └── BEFORE trigger: courses_update_search_columns()
    │       ├── Computes `page_content` = course_name + description + topics + objectives
    │       └── Computes `fts` = to_tsvector('english', page_content)
    │
    ├── ON INSERT/UPDATE (asynchronous, ~1-3s):
    │   └── Database Webhook ──► Edge Function `embed-course`
    │       ├── Checks: embedding IS NULL AND page_content IS NOT EMPTY
    │       ├── Calls Cohere embed-english-v3.0 (input_type: "search_document")
    │       └── UPDATEs the row's `embedding` column (VECTOR(1024))
    │
    ▼ (on query: direct RPC calls to Supabase via REST API)
SupabaseVectorRetriever (pgvector cosine, k=20) ──┐
                                                    ├── EnsembleRetriever (50/50) ──► CohereRerank ──► Results
SupabaseFTSRetriever (Postgres FTS, k=20) ─────────┘
```

---

## Prerequisites

Before starting the migration, you need:

1. **A Supabase project** with the target table already existing (in our case, the `courses` table)
2. **Supabase credentials:**
   - `SUPABASE_SECRET_API_KEY` (service role key) in your `.env` file
   - The project URL (e.g., `https://mdiulditosmnaqhimcdh.supabase.co`)
3. **A Cohere API key** for the embedding model (`embed-english-v3.0`, 1024 dimensions)
4. **Access to the Supabase Dashboard** for:
   - SQL Editor (to run database setup queries)
   - Edge Functions (to deploy the auto-embedding function)
   - Database Webhooks (to trigger the Edge Function)

### Existing `courses` Table Schema (before migration)

This is the table we're adding vector search to. Your table will differ — adapt the column names accordingly.

```sql
CREATE TABLE public.courses (
  id                              BIGSERIAL PRIMARY KEY,
  skillmap_id                     BIGINT REFERENCES skillmaps(skillmap_id),
  course_id                       INTEGER UNIQUE,
  course_week_count               INTEGER,
  course_name                     TEXT NOT NULL,
  course_category                 TEXT,
  course_learning_plan            TEXT,
  course_status                   TEXT DEFAULT 'Not Started',
  course_prerequisites            TEXT,
  course_image                    TEXT,
  course_screenshots              TEXT[],
  course_description              TEXT,
  course_link                     TEXT,
  course_competencies_level1      TEXT[],
  course_competencies_level2      TEXT[],
  course_topics                   TEXT[],        -- Array of topic strings
  course_topics_count             INTEGER,
  course_learning_objectives      TEXT[],        -- Array of objective strings
  course_processes_covered        TEXT[],
  course_equipment_covered        TEXT[],
  course_keywords                 TEXT[],
  course_practice_scenarios       TEXT[],
  course_contributor_ids          INTEGER[],
  short_video_tiktok              TEXT,
  short_video_youtube             TEXT,
  short_video_instagram           TEXT,
  created_at                      TIMESTAMPTZ DEFAULT NOW(),
  updated_at                      TIMESTAMPTZ DEFAULT NOW(),
  course_duration_hours           REAL,
  course_release_date             TEXT,
  course_proposed_date            TEXT,
  nps_score                       REAL,
  course_category_group           TEXT,
  data_completeness_acknowledged  BOOLEAN DEFAULT FALSE,
  sort_order                      INTEGER DEFAULT 0
);
```

Key detail: Several columns are `TEXT[]` (Postgres arrays). These need special handling — `array_to_string()` in SQL and list-to-string conversion in Python.

---

## Step-by-Step: Supabase Database Setup

Run each of these SQL statements **in order** in the **Supabase Dashboard > SQL Editor**.

### 1. Enable the pgvector extension

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

This adds the `VECTOR` data type and distance operators (`<=>` for cosine, `<->` for L2, `<#>` for inner product) to your Postgres instance.

### 2. Add the embedding column

```sql
ALTER TABLE courses ADD COLUMN IF NOT EXISTS embedding VECTOR(1024);
```

The dimension (1024) must match your embedding model. Cohere `embed-english-v3.0` produces 1024-dimensional vectors. If you use a different model (e.g., OpenAI `text-embedding-3-small` at 1536 dimensions), change this number.

### 3. Add `page_content` and `fts` columns

```sql
ALTER TABLE courses ADD COLUMN IF NOT EXISTS page_content TEXT;
ALTER TABLE courses ADD COLUMN IF NOT EXISTS fts TSVECTOR;
```

- `page_content`: Concatenated text used for embedding and display. This mirrors what ChromaDB stored as the document's `page_content`.
- `fts`: Postgres full-text search vector, automatically derived from `page_content`. Replaces the BM25 pickle file.

**Why not `GENERATED ALWAYS AS`?** We initially tried:
```sql
ALTER TABLE courses ADD COLUMN page_content TEXT
  GENERATED ALWAYS AS (
    COALESCE(course_name, '') || ' ' || COALESCE(course_description, '') || ...
  ) STORED;
```
This fails with `ERROR: 42P17: generation expression is not immutable` because `array_to_string()` and `to_tsvector()` are not marked `IMMUTABLE` in Postgres (they depend on locale/dictionary settings). A `BEFORE` trigger is the correct alternative.

### 4. Create the trigger function and trigger

This trigger fires BEFORE every INSERT and UPDATE, automatically computing `page_content` and `fts` from the row's source fields.

```sql
CREATE OR REPLACE FUNCTION courses_update_search_columns()
RETURNS TRIGGER AS $$
BEGIN
  NEW.page_content := COALESCE(NEW.course_name, '') || ' ' ||
    COALESCE(NEW.course_description, '') || ' ' ||
    COALESCE(array_to_string(NEW.course_topics, ', '), '') || ' ' ||
    COALESCE(array_to_string(NEW.course_learning_objectives, ', '), '');

  NEW.fts := to_tsvector('english', NEW.page_content);

  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER courses_search_columns_trigger
  BEFORE INSERT OR UPDATE ON courses
  FOR EACH ROW
  EXECUTE FUNCTION courses_update_search_columns();
```

**What this does:**
- `page_content` = course_name + course_description + course_topics (joined) + course_learning_objectives (joined). This is the same concatenation used by `create_skillcat_vectorstore.py` in the old ChromaDB approach.
- `fts` = Postgres full-text search vector of `page_content`, using the `'english'` dictionary for stemming/stop words.
- `BEFORE INSERT OR UPDATE` means the values are set on the same write — no separate UPDATE needed.

**Customization:** To add more fields to the search text, add them to the concatenation in the trigger function. For example, to include keywords:
```sql
NEW.page_content := ... || ' ' || COALESCE(array_to_string(NEW.course_keywords, ', '), '');
```

### 5. Backfill `page_content` and `fts` for existing rows

The trigger only fires on new INSERT/UPDATE operations. To populate the columns for rows that already exist:

```sql
UPDATE courses SET page_content = NULL;
```

This is a no-op on data but fires the BEFORE UPDATE trigger for every row, which computes and sets both `page_content` and `fts`.

**Verify it worked:**

```sql
SELECT id, course_name, LEFT(page_content, 100) AS content_preview
FROM courses
LIMIT 5;
```

You should see `content_preview` populated with concatenated text from each course's name, description, topics, and objectives.

### 6. Create the full-text search (GIN) index

```sql
CREATE INDEX IF NOT EXISTS idx_courses_fts ON courses USING GIN (fts);
```

A GIN index makes `@@` (full-text match) queries fast. Without this, every FTS query would do a full table scan.

### 7. Create the vector similarity search RPC function

This is called by the Python code via `POST /rest/v1/rpc/match_courses`.

```sql
CREATE OR REPLACE FUNCTION match_courses(
  query_embedding VECTOR(1024),
  match_count INT DEFAULT 20
)
RETURNS TABLE (
  id BIGINT,
  course_id INTEGER,
  page_content TEXT,
  course_name TEXT,
  course_description TEXT,
  course_link TEXT,
  course_category TEXT,
  course_topics TEXT[],
  course_learning_objectives TEXT[],
  course_duration_hours REAL,
  course_prerequisites TEXT,
  course_keywords TEXT[],
  course_processes_covered TEXT[],
  course_equipment_covered TEXT[],
  course_practice_scenarios TEXT[],
  course_competencies_level1 TEXT[],
  course_competencies_level2 TEXT[],
  nps_score REAL,
  course_image TEXT,
  course_status TEXT,
  similarity FLOAT
)
LANGUAGE sql STABLE AS $$
  SELECT
    c.id, c.course_id, c.page_content,
    c.course_name, c.course_description, c.course_link,
    c.course_category, c.course_topics, c.course_learning_objectives,
    c.course_duration_hours, c.course_prerequisites, c.course_keywords,
    c.course_processes_covered, c.course_equipment_covered,
    c.course_practice_scenarios, c.course_competencies_level1,
    c.course_competencies_level2, c.nps_score,
    c.course_image, c.course_status,
    1 - (c.embedding <=> query_embedding) AS similarity
  FROM courses c
  WHERE c.embedding IS NOT NULL
  ORDER BY c.embedding <=> query_embedding
  LIMIT match_count;
$$;
```

**How it works:**
- `<=>` is pgvector's cosine distance operator (0 = identical, 2 = opposite).
- `1 - (c.embedding <=> query_embedding)` converts distance to similarity (1 = identical, -1 = opposite).
- `WHERE c.embedding IS NOT NULL` skips rows that haven't been embedded yet (e.g., freshly inserted rows where the Edge Function hasn't run yet).
- `LANGUAGE sql STABLE` tells Postgres this function has no side effects, enabling query optimizations.

### 8. Create the full-text search RPC function

This is called by the Python code via `POST /rest/v1/rpc/fulltext_search_courses`.

```sql
CREATE OR REPLACE FUNCTION fulltext_search_courses(
  search_query TEXT,
  match_count INT DEFAULT 20
)
RETURNS TABLE (
  id BIGINT,
  course_id INTEGER,
  page_content TEXT,
  course_name TEXT,
  course_description TEXT,
  course_link TEXT,
  course_category TEXT,
  course_topics TEXT[],
  course_learning_objectives TEXT[],
  course_duration_hours REAL,
  course_prerequisites TEXT,
  course_keywords TEXT[],
  course_processes_covered TEXT[],
  course_equipment_covered TEXT[],
  course_practice_scenarios TEXT[],
  course_competencies_level1 TEXT[],
  course_competencies_level2 TEXT[],
  nps_score REAL,
  course_image TEXT,
  course_status TEXT,
  rank FLOAT
)
LANGUAGE sql STABLE AS $$
  SELECT
    c.id, c.course_id, c.page_content,
    c.course_name, c.course_description, c.course_link,
    c.course_category, c.course_topics, c.course_learning_objectives,
    c.course_duration_hours, c.course_prerequisites, c.course_keywords,
    c.course_processes_covered, c.course_equipment_covered,
    c.course_practice_scenarios, c.course_competencies_level1,
    c.course_competencies_level2, c.nps_score,
    c.course_image, c.course_status,
    ts_rank(c.fts, websearch_to_tsquery('english', search_query)) AS rank
  FROM courses c
  WHERE c.fts @@ websearch_to_tsquery('english', search_query)
  ORDER BY rank DESC
  LIMIT match_count;
$$;
```

**How it works:**
- `websearch_to_tsquery('english', search_query)` parses the query using Google-style syntax (supports `"quoted phrases"`, `OR`, `-exclusion`).
- `@@` is the full-text match operator.
- `ts_rank()` scores each matching document by relevance.
- This replaces BM25 — it's not identical ranking, but serves the same purpose of keyword-based recall in the hybrid search pipeline.

### 9. Verify everything

```sql
-- Check new columns exist
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'courses'
  AND column_name IN ('embedding', 'page_content', 'fts')
ORDER BY column_name;
-- Expected: 3 rows (embedding: USER-DEFINED, fts: tsvector, page_content: text)

-- Check RPC functions exist
SELECT routine_name
FROM information_schema.routines
WHERE routine_name IN ('match_courses', 'fulltext_search_courses');
-- Expected: 2 rows

-- Check trigger exists
SELECT trigger_name, event_manipulation, action_timing
FROM information_schema.triggers
WHERE trigger_name = 'courses_search_columns_trigger';
-- Expected: 2 rows (INSERT and UPDATE, both BEFORE)

-- Test full-text search
SELECT course_name, rank
FROM fulltext_search_courses('HVAC', 5);
-- Expected: courses with HVAC-related names, ranked by relevance
```

**Note:** Vector search (`match_courses`) can't be tested yet — embeddings don't exist until the Edge Function runs (next step).

---

## Step-by-Step: Edge Function for Auto-Embedding

The Edge Function watches for INSERT/UPDATE events on the `courses` table and automatically generates an embedding for any row where `embedding IS NULL`.

### 1. Write the Edge Function

**File location in the repo:** `supabase/functions/embed-course/index.ts`

```typescript
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const COHERE_API_KEY = Deno.env.get("COHERE_API_KEY")!;
const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;

serve(async (req) => {
  try {
    const payload = await req.json();
    const record = payload.record;

    if (!record) {
      return new Response(JSON.stringify({ error: "No record in payload" }), {
        status: 400,
      });
    }

    // Skip if embedding already exists (avoid infinite loop from our own UPDATE)
    if (record.embedding) {
      return new Response(JSON.stringify({ skipped: true, reason: "embedding already exists" }), {
        status: 200,
      });
    }

    const pageContent = record.page_content;
    if (!pageContent || !pageContent.trim()) {
      return new Response(JSON.stringify({ skipped: true, reason: "no page_content" }), {
        status: 200,
      });
    }

    // Call Cohere embed API
    const cohereResp = await fetch("https://api.cohere.ai/v1/embed", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${COHERE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        texts: [pageContent],
        model: "embed-english-v3.0",
        input_type: "search_document",
        truncate: "END",
      }),
    });

    if (!cohereResp.ok) {
      const errText = await cohereResp.text();
      console.error("Cohere API error:", errText);
      return new Response(JSON.stringify({ error: "Cohere API failed", details: errText }), {
        status: 500,
      });
    }

    const cohereData = await cohereResp.json();
    const embedding = cohereData.embeddings[0];

    // Write embedding back to the row
    const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);
    const { error } = await supabase
      .from("courses")
      .update({ embedding: JSON.stringify(embedding) })
      .eq("id", record.id);

    if (error) {
      console.error("Supabase update error:", error.message);
      return new Response(JSON.stringify({ error: error.message }), {
        status: 500,
      });
    }

    console.log(`Embedded course id=${record.id} doc_id=skillcat_${record.course_id}`);
    return new Response(
      JSON.stringify({ success: true, course_id: record.course_id }),
      { status: 200 }
    );
  } catch (err) {
    console.error("Unexpected error:", err);
    return new Response(JSON.stringify({ error: String(err) }), {
      status: 500,
    });
  }
});
```

**Critical details:**

- **Infinite loop prevention:** The function checks `if (record.embedding)` and skips if it already exists. Without this, the function's own `UPDATE` (writing the embedding) would re-trigger the webhook, creating an infinite loop.
- **Asymmetric embedding:** `input_type: "search_document"` is used for documents. At query time, the Python code uses `input_type: "search_query"` via `embedding_model.embed_query()`. This asymmetry is how Cohere's model is designed to work — it produces different vectors for documents vs queries to optimize retrieval.
- **`truncate: "END"`:** If `page_content` exceeds the model's max input tokens, Cohere truncates from the end rather than erroring.

### 2. Deploy the Edge Function

**Option A: Via Supabase CLI**

```bash
# Install CLI (Windows - use Scoop, NOT npm)
scoop bucket add supabase https://github.com/supabase/scoop-bucket.git
scoop install supabase

# From the project root
supabase login
supabase link --project-ref mdiulditosmnaqhimcdh

# Set the Cohere secret
supabase secrets set COHERE_API_KEY=your-cohere-api-key-here

# Deploy
supabase functions deploy embed-course
```

**Important:** `npm install -g supabase` does NOT work. The Supabase CLI explicitly blocks global npm installation. Use Scoop on Windows, Homebrew on macOS, or download the binary directly.

**Option B: Via Supabase Dashboard (no CLI needed)**

1. Go to **Supabase Dashboard > Edge Functions**
2. Click **Create a new function**
3. Name it `embed-course`
4. Paste the full contents of `index.ts` from above
5. Under **Edge Function Secrets**, add `COHERE_API_KEY` with your Cohere API key
6. Click **Deploy**

`SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` are automatically available to all Edge Functions — you do NOT need to set them as secrets.

### 3. Create the Database Webhook

This connects the `courses` table to the Edge Function.

1. Go to **Supabase Dashboard > Database > Webhooks**
2. Click **Create a new webhook**
3. Configure:
   - **Name:** `embed-course-webhook`
   - **Table:** `courses`
   - **Events:** check both **INSERT** and **UPDATE**
   - **Type:** Supabase Edge Function
   - **Edge Function:** select `embed-course`
   - **HTTP Headers:** add `Authorization: Bearer <your-service-role-key>`
4. Click **Create webhook**

### 4. Test the Edge Function

Insert a test row and check that the embedding appears:

```sql
-- Insert a test course
INSERT INTO courses (course_name, course_description, course_topics)
VALUES ('Test HVAC Course', 'Introduction to heating and cooling systems', ARRAY['HVAC Basics', 'Thermodynamics']);

-- Wait 2-3 seconds, then check
SELECT id, course_name, LEFT(page_content, 80) AS content, embedding IS NOT NULL AS has_embedding
FROM courses
WHERE course_name = 'Test HVAC Course';
```

If `has_embedding` is `true`, the Edge Function is working. If not, check the Edge Function logs in **Dashboard > Edge Functions > embed-course > Logs**.

```sql
-- Clean up test row
DELETE FROM courses WHERE course_name = 'Test HVAC Course';
```

---

## Step-by-Step: Python Code Changes

### New File: `agents/curriculum_mapping_tool/supabase_vectorstore_service.py`

This is the Python interface to the Supabase vectorstore. It calls the RPC functions we created and converts results to LangChain `Document` objects.

```python
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

# Array fields from Postgres (text[]) that need to be joined into strings
# for compatibility with the existing Chroma-based pipeline, which stored
# all metadata as flat strings.
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

# All metadata fields to extract from RPC results into Document.metadata.
# These must match the column names returned by the RPC functions.
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
```

**Key design decisions:**

- **REST API, not Python SDK:** The existing `supabase_service.py` uses raw `requests` rather than the `supabase` Python package. We follow the same pattern for consistency.
- **Array-to-string conversion:** Postgres returns `text[]` columns as JSON arrays. The old Chroma pipeline stored metadata as flat strings. `_row_to_document()` joins arrays with `", "` to maintain compatibility with downstream code that expects strings (e.g., the LLM prompt that displays course topics).
- **`_METADATA_FIELDS` list:** This controls which columns end up in `Document.metadata`. It must include `course_link` because `retrieve_course_docs()` in `curriculum_retriever.py` filters documents by `course_link` presence.

### Modified File: `agents/curriculum_mapping_tool/curriculum_retriever.py`

Four changes were made to this file:

#### Change 1: Import the Supabase service functions

```python
from agents.curriculum_mapping_tool.supabase_vectorstore_service import (
    vector_search as supabase_vector_search,
    fulltext_search as supabase_fulltext_search,
)
```

Also import `BaseRetriever`:

```python
from langchain_classic.schema import BaseRetriever, Document
```

#### Change 2: Add `"backend": "supabase"` to `SKILLCAT_CFG`

```python
SKILLCAT_CFG = {
    "name": "skillcat",
    "backend": "supabase",   # <-- This one line switches SkillCat to Supabase
    "central_folder_id": "14Xeg7riEhPOL_zaVQ2eo7bVRa9lTr-W1",
    "chroma_folder_name": "chroma_skillcat_db",
    "collection_name": "skillcat_courses",
    "bm25_pickle_name": "bm25_skillcat_db.pkl",
}
```

The old Chroma keys are kept so reverting is a one-line change.

#### Change 3: Add two LangChain-compatible retriever classes

```python
class SupabaseVectorRetriever(BaseRetriever):
    """LangChain BaseRetriever wrapping Supabase pgvector search."""

    k: int = 20

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        embedding_model = get_embedding_model()
        query_embedding = embedding_model.embed_query(query)
        return supabase_vector_search(query_embedding, k=self.k)


class SupabaseFTSRetriever(BaseRetriever):
    """LangChain BaseRetriever wrapping Supabase full-text search."""

    k: int = 20

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        return supabase_fulltext_search(query, k=self.k)
```

**Critical: Must extend `BaseRetriever`, not plain `object`.** LangChain's `EnsembleRetriever` validates that all retrievers are instances of `Runnable` (the base class for all LangChain runnables). `BaseRetriever` inherits from `Runnable`. Plain Python classes will fail with:

```
pydantic_core._pydantic_core.ValidationError: Input should be an instance of Runnable
```

The abstract method to implement is `_get_relevant_documents(self, query: str, **kwargs)` — note the underscore prefix and the `**kwargs`.

**Embedding at query time:** `SupabaseVectorRetriever` calls `get_embedding_model().embed_query(query)`. The `embed_query` method in LangChain's `CohereEmbeddings` automatically uses `input_type: "search_query"`, while the Edge Function uses `input_type: "search_document"`. This asymmetry is by design — Cohere's model optimizes for retrieval by encoding queries and documents differently.

#### Change 4: Branch in the two load functions

In `load_vector_db_retriever()`:

```python
def load_vector_db_retriever(cfg, drive, k=20):
    cache_key = cfg["name"]
    if cache_key in _VECTOR_RETRIEVER_CACHE:
        return _VECTOR_RETRIEVER_CACHE[cache_key]

    # NEW: Supabase path
    if cfg.get("backend") == "supabase":
        retriever = SupabaseVectorRetriever(k=k)
        _VECTOR_RETRIEVER_CACHE[cache_key] = (None, retriever)
        return None, retriever

    # Existing Chroma path (unchanged)
    embedding_model = get_embedding_model()
    chroma_db = load_chroma_db(cfg, embedding_model, drive)
    retriever = chroma_db.as_retriever(search_kwargs={"k": k})
    _VECTOR_RETRIEVER_CACHE[cache_key] = (chroma_db, retriever)
    return chroma_db, retriever
```

In `load_bm25_retriever_with_pydrive()`:

```python
def load_bm25_retriever_with_pydrive(cfg, drive, k=20):
    cache_key = cfg["name"]
    if cache_key in _BM25_CACHE:
        return _BM25_CACHE[cache_key]

    # NEW: Supabase path
    if cfg.get("backend") == "supabase":
        retriever = SupabaseFTSRetriever(k=k)
        _BM25_CACHE[cache_key] = retriever
        return retriever

    # Existing BM25/pickle path (unchanged)
    central_folder_id = cfg["central_folder_id"]
    # ... rest of existing code ...
```

**No changes needed downstream.** The following functions work unchanged because they operate on the abstract retriever interface:

- `get_ensemble_retriever()` — Creates `EnsembleRetriever(retrievers=[bm25, vec], weights=[0.5, 0.5])`
- `get_compression_retriever()` — Wraps ensemble with `CohereRerank(model="rerank-v3.5", top_n=15)`
- `retrieve_course_docs()` — Calls `retriever.invoke(query)`, filters for `course_link`, returns top-k
- `map_one_row()` — Orchestrates retrieval across all three sources (SkillCat, NexTech, Video)
- `run_curriculum_mapping()` — Parallel row mapping with ThreadPoolExecutor

---

## Step-by-Step: Backfill Existing Data

After the Edge Function and Database Webhook are active, existing rows need embeddings.

### 1. Trigger the webhook for all existing rows

```sql
UPDATE courses SET course_name = course_name;
```

This is a no-op on data (sets each name to itself) but fires the UPDATE webhook for every row. The Edge Function sees `embedding IS NULL` and calls Cohere to embed each one.

### 2. Monitor progress

Run this query repeatedly over the next 1-2 minutes:

```sql
SELECT
  COUNT(*) AS total,
  COUNT(embedding) AS embedded,
  COUNT(*) - COUNT(embedding) AS pending
FROM courses;
```

When `pending` reaches 0, all courses are embedded.

### 3. Create the vector similarity index

Now that embeddings exist, create the HNSW index for fast similarity search:

```sql
CREATE INDEX IF NOT EXISTS idx_courses_embedding
  ON courses USING hnsw (embedding vector_cosine_ops);
```

**Why HNSW over IVFFlat?** IVFFlat requires a training step that builds cluster centroids from existing data. If you try to create an IVFFlat index on a table with no embeddings, it can fail or produce a useless index. HNSW (Hierarchical Navigable Small World) builds incrementally and works with any number of rows.

### 4. Verify vector search works

```sql
SELECT course_name, similarity
FROM match_courses(
  (SELECT embedding FROM courses WHERE course_name ILIKE '%hvac%' LIMIT 1),
  5
);
```

This takes one course's embedding and finds the 5 most similar courses. You should see related HVAC courses.

---

## How the Retrieval Pipeline Works End-to-End

When a user runs curriculum mapping, the following happens for each row:

```
1. Query: "HVAC Basics - Electricity" (category + course name)
         │
         ▼
2. SupabaseVectorRetriever.invoke(query)
   ├── Embeds query via Cohere embed_query() [input_type: "search_query"]
   ├── POST /rest/v1/rpc/match_courses {query_embedding, match_count: 20}
   └── Returns 20 Documents (sorted by cosine similarity)
         │
3. SupabaseFTSRetriever.invoke(query)
   ├── POST /rest/v1/rpc/fulltext_search_courses {search_query, match_count: 20}
   └── Returns 20 Documents (sorted by ts_rank)
         │
         ▼
4. EnsembleRetriever (weights: [0.5 FTS, 0.5 Vector])
   └── Merges and re-scores results from both retrievers
         │
         ▼
5. CohereRerank (model: "rerank-v3.5", top_n: 15)
   └── Reranks the merged results using Cohere's reranker
         │
         ▼
6. retrieve_course_docs() filters for course_link, returns top k=5
         │
         ▼
7. select_best_resources_unified() — LLM picks the best match
   from SkillCat (Supabase), NexTech (Chroma), and Video (Chroma)
```

---

## Key Design Decisions and Gotchas

### 1. `GENERATED ALWAYS AS` Doesn't Work for This

Postgres requires generated column expressions to be `IMMUTABLE`. `array_to_string()` and `to_tsvector()` are only `STABLE` (their output depends on locale/dictionary settings). The workaround is a `BEFORE INSERT OR UPDATE` trigger.

### 2. Infinite Loop Prevention in the Edge Function

The Edge Function writes an `embedding` to the row via `UPDATE`. This UPDATE triggers the webhook again. Without the `if (record.embedding)` guard, you get an infinite loop. The guard makes the function idempotent — it only embeds rows with `embedding IS NULL`.

### 3. Array Fields Need Conversion

The Supabase `courses` table uses `text[]` arrays for topics, objectives, keywords, etc. The old ChromaDB metadata stored these as flat strings. The Python `_row_to_document()` function joins arrays with `", "` to maintain compatibility. If the downstream LLM prompt or display code is updated to handle arrays natively, this conversion can be removed.

### 4. Asymmetric Embedding (search_document vs search_query)

Cohere `embed-english-v3.0` uses different `input_type` values for documents vs queries:
- **Documents** (Edge Function): `input_type: "search_document"` — optimized for being found
- **Queries** (Python retriever): `input_type: "search_query"` — optimized for finding

Using the same `input_type` for both will degrade retrieval quality. LangChain's `CohereEmbeddings.embed_query()` automatically sets `input_type: "search_query"`.

### 5. Postgres FTS vs BM25

Postgres full-text search uses a different ranking algorithm than BM25 (it uses `ts_rank` which is based on term frequency and inverse document frequency, but with different normalization). Results will be slightly different. For this use case (hybrid retrieval where the Cohere reranker is the final arbiter), the difference is negligible.

### 6. Embedding Latency Window

After an INSERT/UPDATE, there's a ~1-3 second window where the course has `page_content` and `fts` (set synchronously by the trigger) but no `embedding` (set asynchronously by the Edge Function). During this window, the course will appear in full-text search but not vector search. This is acceptable for a course catalog that isn't updated in real-time.

### 7. The `drive` Parameter is Still Required

The load functions still accept a `drive` parameter. In the Supabase path, it's not used, but the function signature remains unchanged for compatibility with NexTech and Video which still use it.

---

## Files Summary

### Created

| File | Purpose |
|------|---------|
| `agents/curriculum_mapping_tool/supabase_vectorstore_service.py` | Python service: Supabase vector search + FTS via REST API, returns LangChain Documents |
| `supabase/functions/embed-course/index.ts` | Edge Function: auto-embeds courses via Cohere on insert/update |
| `agents/curriculum_mapping_tool/SUPABASE_VECTORSTORE_MIGRATION.md` | This documentation |

### Modified

| File | Changes |
|------|---------|
| `agents/curriculum_mapping_tool/curriculum_retriever.py` | Added `BaseRetriever` import, added `"backend": "supabase"` to `SKILLCAT_CFG`, added `SupabaseVectorRetriever` and `SupabaseFTSRetriever` classes, added Supabase branching in `load_vector_db_retriever()` and `load_bm25_retriever_with_pydrive()` |

### Not Modified (remain on ChromaDB)

| File | Reason |
|------|--------|
| `agents/curriculum_mapping_tool/create_skillcat_vectorstore.py` | Kept as-is for reference/fallback; no longer called for SkillCat |
| `agents/curriculum_mapping_tool/create_nextech_vectorstore.py` | NexTech stays on Chroma |
| `agents/course_outline/video_search_tool/create_video_vectorstore.py` | Video stays on Chroma |
| `agents/course_outline/video_search_tool/video_retriever.py` | Video retrieval unchanged |
| `services/embedding_service.py` | Shared by both paths, unchanged |

### SQL Objects Created in Supabase

| Object | Type | Purpose |
|--------|------|---------|
| `vector` | Extension | Enables `VECTOR` data type and distance operators |
| `courses.embedding` | Column (`VECTOR(1024)`) | Stores Cohere embeddings |
| `courses.page_content` | Column (`TEXT`) | Concatenated search text |
| `courses.fts` | Column (`TSVECTOR`) | Full-text search vector |
| `courses_update_search_columns()` | Function | Trigger function: computes page_content + fts |
| `courses_search_columns_trigger` | Trigger (`BEFORE INSERT OR UPDATE`) | Calls the above function |
| `match_courses()` | Function (RPC) | Vector similarity search |
| `fulltext_search_courses()` | Function (RPC) | Full-text keyword search |
| `idx_courses_embedding` | Index (HNSW) | Fast approximate nearest neighbor search |
| `idx_courses_fts` | Index (GIN) | Fast full-text search |
| `embed-course` | Edge Function | Auto-embeds courses via Cohere API |
| `embed-course-webhook` | Database Webhook | Triggers Edge Function on INSERT/UPDATE |

### Environment Variables

| Variable | Where Used | Purpose |
|----------|-----------|---------|
| `SUPABASE_SECRET_API_KEY` | Python (`.env`) | Service role key for REST API calls |
| `COHERE_API_KEY` | Edge Function (Supabase secrets) | Cohere API key for embedding |

---

## How to Revert to ChromaDB

Remove or comment out one line in `curriculum_retriever.py`:

```python
SKILLCAT_CFG = {
    "name": "skillcat",
    # "backend": "supabase",  # Comment out to revert to ChromaDB
    "central_folder_id": "14Xeg7riEhPOL_zaVQ2eo7bVRa9lTr-W1",
    "chroma_folder_name": "chroma_skillcat_db",
    "collection_name": "skillcat_courses",
    "bm25_pickle_name": "bm25_skillcat_db.pkl",
}
```

The Chroma code paths are fully intact and will resume downloading from Google Drive as before. The Supabase columns, functions, Edge Function, and webhook can remain in place without affecting anything — they just won't be queried.

---

## How to Replicate for Another Table

If you want to add the same vector search capability to another table (e.g., NexTech courses in a `nextech_courses` table), follow this checklist:

### Database Setup

1. `CREATE EXTENSION IF NOT EXISTS vector;` (only needed once per database)
2. Add columns: `embedding VECTOR(<dim>)`, `page_content TEXT`, `fts TSVECTOR`
3. Create trigger function that concatenates your searchable fields into `page_content` and computes `fts`
4. Create `BEFORE INSERT OR UPDATE` trigger
5. Backfill: `UPDATE <table> SET page_content = NULL;`
6. Create GIN index on `fts`
7. Create RPC functions: `match_<table>(query_embedding, match_count)` and `fulltext_search_<table>(search_query, match_count)` — adapt column names to your table
8. Create HNSW index on `embedding` (after backfill)

### Edge Function

1. Copy `supabase/functions/embed-course/index.ts`
2. Change `.from("courses")` to your table name
3. Deploy as a new Edge Function
4. Create a Database Webhook pointing to it

### Python

1. In `supabase_vectorstore_service.py`, add new functions (e.g., `vector_search_nextech()`) that call your new RPC functions. Update `_METADATA_FIELDS` and `_ARRAY_FIELDS` to match your table's columns.
2. In `curriculum_retriever.py`, add `"backend": "supabase"` to the relevant `CFG` dict. The existing branching logic will automatically route to the Supabase path.

---

## Troubleshooting

### Embeddings not appearing after insert

1. Check the Database Webhook is active: **Dashboard > Database > Webhooks** — verify it shows as enabled
2. Check Edge Function logs: **Dashboard > Edge Functions > embed-course > Logs** — look for errors
3. Verify `COHERE_API_KEY` is set: **Dashboard > Edge Functions > Secrets**
4. Check `page_content` is populated (the trigger must have run):
   ```sql
   SELECT id, LEFT(page_content, 50) FROM courses WHERE embedding IS NULL LIMIT 5;
   ```
5. Check that the webhook header has the correct `Authorization: Bearer <service-role-key>`

### Vector search returning no results

1. Verify embeddings exist: `SELECT COUNT(*) FROM courses WHERE embedding IS NOT NULL;`
2. Test the RPC function directly:
   ```sql
   SELECT course_name, similarity
   FROM match_courses((SELECT embedding FROM courses LIMIT 1), 5);
   ```
3. Check the HNSW index exists:
   ```sql
   SELECT indexname FROM pg_indexes WHERE tablename = 'courses' AND indexname = 'idx_courses_embedding';
   ```

### Full-text search returning no results

1. Verify FTS column is populated: `SELECT id, fts FROM courses WHERE fts IS NOT NULL LIMIT 5;`
2. Test directly: `SELECT course_name FROM fulltext_search_courses('HVAC', 5);`
3. Try simpler terms — `websearch_to_tsquery` uses stemming, so "heating" matches "heat" but very short/unusual terms may not match

### Python `EnsembleRetriever` validation error

```
pydantic_core._pydantic_core.ValidationError: Input should be an instance of Runnable
```

The custom retriever classes MUST extend `BaseRetriever` (from `langchain_classic.schema`), not plain Python classes. `BaseRetriever` inherits from `Runnable`, which is what `EnsembleRetriever` validates. The abstract method to implement is `_get_relevant_documents(self, query: str, **kwargs)`.

### Cohere API rate limiting

If backfilling many rows at once, the Edge Function may hit Cohere's rate limit. Check Edge Function logs for 429 errors. Solutions:
- Wait and re-trigger: `UPDATE courses SET embedding = NULL WHERE embedding IS NULL;`
- Batch embed manually in Python using `CohereEmbeddings.embed_documents()` and update rows directly via the Supabase REST API
