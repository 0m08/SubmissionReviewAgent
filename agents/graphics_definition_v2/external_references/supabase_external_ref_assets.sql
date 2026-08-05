-- =============================================================================
-- External Reference Assets vectorstore (Content_Courses project)
-- =============================================================================
-- SAFE / CREATE-ONLY for this project:
--   - Creates NEW table: public.external_ref_assets
--   - Creates NEW indexes on that table only
--   - Creates NEW RPC: public.match_external_ref_assets
--   - Does NOT alter, drop, truncate, or update any existing tables
--   - Does NOT modify match_courses or any curriculum objects
--
-- How to run:
--   Supabase Dashboard → Content_Courses → SQL Editor → New query → paste → Run
-- =============================================================================

-- pgvector is usually already enabled for curriculum; IF NOT EXISTS is a no-op if present.
CREATE EXTENSION IF NOT EXISTS vector;

-- New table only. Gemini Embedding 2 output dimensionality = 3072.
CREATE TABLE IF NOT EXISTS public.external_ref_assets (
  asset_id          TEXT PRIMARY KEY,
  sheet_id          TEXT NOT NULL,
  course_name       TEXT,
  source_link       TEXT NOT NULL,
  asset_url         TEXT NOT NULL,
  asset_type        TEXT NOT NULL CHECK (asset_type IN ('image', 'video_segment')),
  source_kind       TEXT NOT NULL CHECK (source_kind IN ('pdf_extract', 'drive_video', 'youtube_video')),
  mime_type         TEXT,
  title             TEXT,
  drive_folder_id   TEXT,
  video_id          TEXT,
  segment_index     INTEGER,
  start_time        DOUBLE PRECISION,
  end_time          DOUBLE PRECISION,
  indexed_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  embedding         VECTOR(3072)
);

-- Indexes for sync + filtered retrieval (this table only).
CREATE INDEX IF NOT EXISTS idx_external_ref_assets_sheet_id
  ON public.external_ref_assets (sheet_id);

CREATE INDEX IF NOT EXISTS idx_external_ref_assets_sheet_source
  ON public.external_ref_assets (sheet_id, source_link);

-- NOTE: Do NOT create HNSW/IVFFlat on VECTOR(3072).
-- pgvector HNSW/IVFFlat reject > 2000 dimensions. Gemini Embedding 2 uses 3072.
-- Sheet-scoped searches stay fast via sheet_id indexes above; vector scan is fine
-- for this pool size. (If we later reduce output_dimensionality to <=2000, we can add HNSW.)

-- Block anon/publishable access; service_role bypasses RLS.
ALTER TABLE public.external_ref_assets ENABLE ROW LEVEL SECURITY;

-- New RPC only (name must stay distinct from match_courses).
CREATE OR REPLACE FUNCTION public.match_external_ref_assets(
  query_embedding VECTOR(3072),
  match_count INT DEFAULT 20,
  filter_sheet_id TEXT DEFAULT NULL,
  filter_asset_type TEXT DEFAULT NULL
)
RETURNS TABLE (
  asset_id TEXT,
  sheet_id TEXT,
  course_name TEXT,
  source_link TEXT,
  asset_url TEXT,
  asset_type TEXT,
  source_kind TEXT,
  mime_type TEXT,
  title TEXT,
  drive_folder_id TEXT,
  video_id TEXT,
  segment_index INTEGER,
  start_time DOUBLE PRECISION,
  end_time DOUBLE PRECISION,
  indexed_at TIMESTAMPTZ,
  similarity FLOAT
)
LANGUAGE sql
STABLE
AS $$
  SELECT
    a.asset_id,
    a.sheet_id,
    a.course_name,
    a.source_link,
    a.asset_url,
    a.asset_type,
    a.source_kind,
    a.mime_type,
    a.title,
    a.drive_folder_id,
    a.video_id,
    a.segment_index,
    a.start_time,
    a.end_time,
    a.indexed_at,
    (1 - (a.embedding <=> query_embedding))::FLOAT AS similarity
  FROM public.external_ref_assets a
  WHERE a.embedding IS NOT NULL
    AND (filter_sheet_id IS NULL OR a.sheet_id = filter_sheet_id)
    AND (filter_asset_type IS NULL OR a.asset_type = filter_asset_type)
  ORDER BY a.embedding <=> query_embedding
  LIMIT match_count;
$$;
