# Vector Search & RAG Pipeline Architecture Report

This document catalogs every vector search and RAG (Retrieval Augmented Generation) pipeline in the course outline generation system.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Pipeline 1: Research Notes RAG](#pipeline-1-research-notes-rag)
3. [Pipeline 2: Video Search RAG (Text-Based)](#pipeline-2-video-search-rag-text-based)
4. [Pipeline 3: Video Search RAG (Multimodal / Vertex AI)](#pipeline-3-video-search-rag-multimodal--vertex-ai)
5. [Pipeline 4: Graphics / Image Search RAG](#pipeline-4-graphics--image-search-rag)
6. [Pipeline 5: Curriculum Mapping - SkillCat (Supabase pgvector)](#pipeline-5-curriculum-mapping---skillcat-supabase-pgvector)
7. [Pipeline 6: Curriculum Mapping - NexTech (ChromaDB)](#pipeline-6-curriculum-mapping---nextech-chromadb)
8. [Shared Services](#shared-services)
9. [Storage & Persistence](#storage--persistence)
10. [Summary Comparison Table](#summary-comparison-table)

---

## Architecture Overview

The system contains **6 distinct RAG pipelines** that serve different purposes across course outline generation, curriculum mapping, and content retrieval. They share common infrastructure (embedding service, chunking service, Google Drive storage) but differ in vector databases, retrieval strategies, and data sources.

```
                        +---------------------+
                        |   Shared Services    |
                        |  - Cohere Embeddings |
                        |  - Chunking Service  |
                        |  - Google Drive      |
                        +----------+----------+
                                   |
          +------------------------+------------------------+
          |            |           |           |            |
  Research Notes   Video Search  Graphics   Curriculum   Curriculum
   (ChromaDB)     (ChromaDB +   (ChromaDB)  SkillCat    NexTech
                   Vertex AI)               (Supabase)  (ChromaDB)
```

---

## Pipeline 1: Research Notes RAG

### Purpose
Retrieves relevant reference material (web pages, YouTube transcripts, video chunks) to generate research notes for a course topic.

### Key Files
| File | Role |
|------|------|
| `agents/research_notes/vector_store.py` | ChromaDB initialization and management |
| `agents/research_notes/retriever.py` | Vector + BM25 ensemble retriever setup |
| `agents/research_notes/retriever_agent.py` | Agentic query refinement loop |
| `agents/research_notes/load_references.py` | Document loading and chunking |
| `agents/research_notes/generate_notes.py` | Uses retrieval for note generation |

### Architecture

```
Input Sources (Web pages, YouTube transcripts, Video chunks)
        |
        v
  load_references.py
  - Loads documents from URLs, YouTube, Google Sheets
  - Converts to LangChain Document objects
        |
        v
  chunking_service.py
  - Semantic chunker (Chonkie SDPMChunker, model: minishlab/potion-base-8M)
  - OR Markdown chunker (splits on headers)
  - OR General chunker (auto-detects format)
  - Token limit: ~2000 per chunk
        |
        v
  vector_store.py
  - Embeds chunks with Cohere embed-english-v3.0
  - Stores in ChromaDB (persistent, SQLite-backed)
  - Also builds BM25 index (pickled)
        |
        v
  retriever.py
  +---> ChromaDB similarity search (k=20)
  +---> BM25 retriever (k=20)
  |         |
  v         v
  EnsembleRetriever (weights: 0.5 vector / 0.5 BM25)
        |
        v
  CohereRerank v3.5 (top_n=15)
        |
        v
  Retrieved Documents --> generate_notes.py --> LLM generates research notes
```

### Configuration
- **Vector DB**: ChromaDB (persistent)
- **Embedding Model**: Cohere `embed-english-v3.0`
- **Retrieval**: Ensemble (Vector + BM25) + Cohere Rerank
- **Local Storage**: `/tmp/temp_chroma_folder/{short_course_name}_chroma_research_db/`
- **Drive Storage**: Course root folder > `Vectorstore files` > `chroma_research_db/`
- **BM25 Pickle**: Course root folder > `Pickle files` > `bm25_research_db.pkl`

---

## Pipeline 2: Video Search RAG (Text-Based)

### Purpose
Searches HVAC video transcripts to find relevant video segments for course topics.

### Key Files
| File | Role |
|------|------|
| `agents/course_outline/video_search_tool/video_retriever.py` | Main retrieval with ensemble + reranking |
| `agents/course_outline/video_search_tool/create_video_vectorstore.py` | Builds vectorstore from transcript data |
| `agents/course_outline/video_search_tool/update_video_vectorstore.py` | Incremental vectorstore updates |
| `agents/course_outline/video_search_tool/video_retriever_agent.py` | Agentic query refinement |

### Architecture

```
Google Sheet: "HVAC School Video Chunks"
        |
        v
  create_video_vectorstore.py
  - Combines: video_title + chapter_title + text_0
  - Extracts metadata: video_id, start_time, end_time,
    segment_index, channel, published, transcript
        |
        v
  Embedded with Cohere embed-english-v3.0
  Stored in ChromaDB collection: "video_embeddings"
        |
        v
  video_retriever.py
  +---> ChromaDB similarity search (k=20)
  +---> BM25 retriever (k=20)
  |         |
  v         v
  EnsembleRetriever (0.5 / 0.5 weights)
        |
        v
  CohereRerank v3.5 (top_n=15)
        |
        v
  Retrieved Video Segments (with timestamps, video IDs)
```

### Configuration
- **Vector DB**: ChromaDB (persistent)
- **Collection**: `video_embeddings`
- **Embedding Model**: Cohere `embed-english-v3.0`
- **Retrieval**: Ensemble (Vector + BM25) + Cohere Rerank
- **Local Storage**: `/tmp/temp_chroma_folder/chroma_video_db/`
- **Drive Storage**: Drive folder `1kovlkUd3pN5IGDB16LC2H8grvmQOhXHy` > `Vectorstore files` > `chroma_video_db`

---

## Pipeline 3: Video Search RAG (Multimodal / Vertex AI)

### Purpose
Multimodal video search that accepts text, image, or video queries to find relevant HVAC video segments. Searches over HVAC channel videos using visual and textual embeddings.

### Key Files
| File | Role |
|------|------|
| `video_search_hvac_channels.py` | Multimodal embedding and search logic |
| `agents/course_outline/video_search_tool/video_retriever.py` | `load_new_video_embeddings_chroma_db()` function |

### Architecture

```
Query (Text / Image / Video)
        |
        v
  Vertex AI MultiModalEmbeddingModel
  Model: multimodalembedding@001
  Embedding Dimension: 1408
        |
        +---> Text query  --> get_text_embeddings()
        +---> Image query --> get_image_embeddings() (PIL or base64 JPEG)
        +---> Video query --> get_video_embeddings() (first 30s, MP4/WebM)
        |
        v
  ChromaDB collection.query()
  - fetch_k = k * 5 (overfetch)
  - Cosine similarity
        |
        v
  Post-Processing:
  - Deduplicate by (video_id, start_time, end_time)
  - Diversity filter: max_per_video = ceil(k / 4)
        |
        v
  Top-k Video Segments
```

### Configuration
- **Vector DB**: ChromaDB
- **Embedding Model**: Google Vertex AI `multimodalembedding@001` (1408 dimensions)
- **Input Types**: Text, Image (PIL/base64), Video (MP4/WebM)
- **Local Storage**: `/tmp/HVAC Video Embeddings/`
- **Drive Folder**: `15H9thXq02JX3ldADSj1oD78mbV-fXfvu` (VIDEO_EMBEDDINGS_FOLDER_ID)
- **Auth**: `VERTEX_AI_SA_B64` environment variable (service account)

---

## Pipeline 4: Graphics / Image Search RAG

### Purpose
Finds relevant HVAC graphics and images for course content. Supports both text-to-image and image-to-image search. Falls back to web image search when vector results are insufficient.

### Key Files
| File | Role |
|------|------|
| `agents/vector_store_image_search/graphics_retriever.py` | Main search across text + image embedding collections |
| `agents/vector_store_image_search/create_vectorstore.py` | Builds dual vectorstores from image data |
| `agents/vector_store_image_search/graphics_retriever_agent.py` | Agentic query refinement |
| `agents/vector_store_image_search/langgraph_agent_with_tools.py` | LangGraph agent routing vector vs web search |
| `agents/vector_store_image_search/web_image_search_tool.py` | Web fallback for poor vector results |

### Architecture

```
Image Corpus (Google Drive folders)
        |
        v
  create_vectorstore.py
  - Extracts metadata from filenames and Google Sheets
  - Computes perceptual hashes (imagehash.phash) + MD5
  - Creates TWO collections:
        |
        +---> Collection: "text_embeddings"
        |     Embedding: Cohere embed-english-v3.0
        |     Content: image titles, descriptions, metadata text
        |
        +---> Collection: "image_embeddings"
              Embedding: Cohere embed-v4.0 (image embedding)
              Content: actual image data (CLIP-style vectors)

---

  Query (Text or Image)
        |
        +---> Text Query:
        |     +---> text_embeddings collection (Cohere text embed)
        |     +---> image_embeddings collection (text-to-image CLIP)
        |     Combine results from both
        |
        +---> Image Query:
              +---> image_embeddings collection (Cohere image embed v4.0)
        |
        v
  similarity_search_with_score() / raw .query()  (k=50)
        |
        v
  Post-Processing:
  - Combine results from both collections
  - Sort by similarity score
  - Deduplicate by: phash, content MD5, image_id, drive_url, filename
  - Download and verify images from Drive
  - Apply metadata filters
        |
        v
  Top-k Image Results

  If results insufficient:
        |
        v
  LangGraph Agent (langgraph_agent_with_tools.py)
  - Routes to: run_vector_tool OR run_web_tool
  - State machine with conditional routing
  - web_image_search_tool.py as fallback
```

### Configuration
- **Vector DB**: ChromaDB (persistent, dual collections)
- **Text Embedding**: Cohere `embed-english-v3.0`
- **Image Embedding**: Cohere `embed-v4.0`
- **Local Storage**: `/tmp/temp_chroma_folder/chroma_graphics_db_/`
- **Drive Storage**: Course root > `Vectorstore files` > graphics DB
- **Deduplication**: Perceptual hash, MD5, image_id, drive_url

---

## Pipeline 5: Curriculum Mapping - SkillCat (Supabase pgvector)

### Purpose
Searches the SkillCat course catalog to find existing courses that match curriculum topics. This is the **Supabase-based pipeline** using PostgreSQL + pgvector for server-side vector search.

### Key Files
| File | Role |
|------|------|
| `agents/curriculum_mapping_tool/supabase_vectorstore_service.py` | Supabase REST API vector search + full-text search |
| `agents/curriculum_mapping_tool/supabase_service.py` | Supabase REST client for CRUD operations |
| `agents/curriculum_mapping_tool/supabase_export.py` | Exports curriculum results back to Supabase |
| `agents/curriculum_mapping_tool/curriculum_retriever.py` | Orchestrates retrieval across all sources |
| `agents/curriculum_mapping_tool/create_skillcat_vectorstore.py` | Legacy ChromaDB vectorstore builder (migrated to Supabase) |
| `supabase/functions/embed-course/index.ts` | Edge Function: auto-embeds courses on insert/update |

### Architecture

#### Indexing Pipeline (Write Path)

```
Google Sheet (Tab: "1 Dec") or Direct Insert
        |
        v
  courses table (Supabase PostgreSQL)
  - INSERT/UPDATE triggers Edge Function
        |
        v
  Edge Function: embed-course (index.ts)
  - Runtime: Deno (Supabase Edge Functions)
  - Extracts page_content from record
  - Calls Cohere API:
      model: embed-english-v3.0
      input_type: search_document
      truncate: END
  - Stores embedding back to "embedding" column
  - Guards against infinite loops (skips if embedding exists)
        |
        v
  courses.embedding column (pgvector type)
  - Indexed for cosine similarity search
```

#### Search Pipeline (Read Path)

```
User Query (course topic / skill)
        |
        v
  Cohere embed-english-v3.0
  input_type: search_query
        |
        v
  Two parallel search paths:
        |
        +---> RPC: match_courses (Vector Search)
        |     POST /rest/v1/rpc/match_courses
        |     Params: query_embedding, match_count=20
        |     Engine: pgvector cosine similarity
        |
        +---> RPC: fulltext_search_courses (Full-Text Search)
              POST /rest/v1/rpc/fulltext_search_courses
              Params: search_query, match_count=20
              Engine: PostgreSQL tsvector/tsquery
        |
        v
  Results converted to LangChain Documents
  - page_content: combined course text
  - metadata: course_id, course_name, course_link,
    course_description, course_category, course_topics,
    course_learning_objectives, course_duration_hours,
    course_keywords, course_competencies, nps_score, etc.
        |
        v
  SupabaseVectorRetriever / SupabaseFTSRetriever
  (Custom LangChain BaseRetriever subclasses)
        |
        v
  curriculum_retriever.py orchestration
```

#### Custom Retriever Classes

```python
class SupabaseVectorRetriever(BaseRetriever):
    k: int = 20
    def _get_relevant_documents(self, query):
        query_embedding = get_embedding_model().embed_query(query)
        return supabase_vector_search(query_embedding, k=self.k)

class SupabaseFTSRetriever(BaseRetriever):
    k: int = 20
    def _get_relevant_documents(self, query):
        return supabase_fulltext_search(query, k=self.k)
```

### Supabase Database Schema

**Table: `courses`**
| Column | Type | Purpose |
|--------|------|---------|
| id | int | Primary key |
| page_content | text | Combined text for embedding |
| embedding | vector | pgvector column (1024 dims) |
| course_id | text | External course identifier |
| course_name | text | Course title |
| course_description | text | Course description |
| course_link | text | URL to course |
| course_category | text | Category classification |
| course_topics | text[] | Array of topics |
| course_learning_objectives | text[] | Array of learning objectives |
| course_duration_hours | float | Duration |
| course_keywords | text[] | Array of keywords |
| course_competencies_level1 | text[] | Level 1 competencies |
| course_competencies_level2 | text[] | Level 2 competencies |
| nps_score | float | Net promoter score |
| course_status | text | Active/inactive status |

**Table: `custom_curriculums`** - Stores generated curriculum structures

**Table: `curriculum_courses`** - Links courses to curriculums with concept mappings

### Configuration
- **Vector DB**: Supabase PostgreSQL + pgvector
- **Supabase URL**: `https://mdiulditosmnaqhimcdh.supabase.co`
- **Auth**: `SUPABASE_SECRET_API_KEY` environment variable
- **Embedding Model**: Cohere `embed-english-v3.0` (1024 dimensions)
- **Search Methods**: pgvector cosine similarity + PostgreSQL full-text search
- **Edge Function**: `embed-course` (auto-embeds on insert/update)

---

## Pipeline 6: Curriculum Mapping - NexTech (ChromaDB)

### Purpose
Searches the NexTech course catalog for curriculum mapping, using a local ChromaDB vectorstore synced via Google Drive.

### Key Files
| File | Role |
|------|------|
| `agents/curriculum_mapping_tool/create_nextech_vectorstore.py` | Builds ChromaDB from Google Sheets |
| `agents/curriculum_mapping_tool/curriculum_retriever.py` | Retrieval orchestration |

### Architecture

```
Google Sheet (All worksheets with "Course Name" + "Course Link" columns)
        |
        v
  create_nextech_vectorstore.py
  - Combines: course_name + duration + prerequisites + competencies
  - Metadata: course_name, duration, prerequisites, etc.
        |
        v
  Embedded with Cohere embed-english-v3.0
  Stored in ChromaDB collection: "nextech_courses"
        |
        v
  curriculum_retriever.py
  +---> ChromaDB similarity search
  +---> BM25 retriever
  |         |
  v         v
  EnsembleRetriever
        |
        v
  CohereRerank v3.5
        |
        v
  Retrieved NexTech Courses
```

### Configuration
- **Vector DB**: ChromaDB (persistent)
- **Collection**: `nextech_courses`
- **Embedding Model**: Cohere `embed-english-v3.0`
- **Retrieval**: Ensemble (Vector + BM25) + Cohere Rerank
- **Drive Storage**: `Vectorstore files - NexTech` > `chroma_nextech_db/`

---

## Shared Services

### Embedding Service
**File**: `services/embedding_service.py`

```python
def get_embedding_model():
    return CohereEmbeddings(model="embed-english-v3.0")
```

Used by all pipelines except the Vertex AI multimodal pipeline (which uses its own model).

### Chunking Service
**File**: `services/chunking_service.py`

Three strategies available:

| Strategy | Library | Model/Method | Token Limit | Use Case |
|----------|---------|-------------|-------------|----------|
| Semantic | Chonkie (SDPMChunker) | `minishlab/potion-base-8M`, threshold 0.5 | ~2000 | General documents |
| Markdown | RecursiveCharacterTextSplitter | Splits on `#`, `##`, `###` headers | ~2000 | Markdown content |
| General | Auto-detect | Chooses semantic or markdown based on content | ~2000 | Unknown format |

### Reranking
All ensemble-based pipelines use **Cohere Rerank v3.5** with `top_n=15` as the final compression step.

### Agentic Query Refinement
Shared pattern across Research Notes, Video Search, and Graphics Search:

```
Initial Query --> Retrieve top-k --> LLM evaluates relevance
    |                                        |
    |              +-- RELEVANT: TERMINATE, return results
    |              |
    +-- NOT RELEVANT: Refine query, retry (max 3 turns)
                   |
                   +-- Exhausted: Return best results or NONE
```

---

## Storage & Persistence

### Google Drive Folder Structure
```
Root Folder (per course)
+-- Vectorstore files/
|   +-- chroma_research_db/        (Research Notes)
|   +-- chroma_video_db/           (Video Search)
|   +-- chroma_graphics_db/        (Graphics Search)
+-- Vectorstore files - SkillCat/
|   +-- chroma_skillcat_db/        (SkillCat - legacy, migrated to Supabase)
+-- Vectorstore files - NexTech/
|   +-- chroma_nextech_db/         (NexTech courses)
+-- Pickle files/
    +-- bm25_research_db.pkl       (BM25 index for research notes)
    +-- bm25_retriever.pkl         (BM25 index for video search)
```

### Local Cache
All ChromaDB databases are cached locally under `/tmp/temp_chroma_folder/` to avoid repeated downloads from Google Drive. Lock files (`_download_lock`) prevent race conditions during concurrent access.

### Supabase (Cloud)
SkillCat courses are stored server-side in Supabase PostgreSQL with pgvector, eliminating the need for local ChromaDB + Google Drive sync for that dataset.

---

## Summary Comparison Table

| Pipeline | Vector DB | Embedding Model | Dimensions | Retrieval Strategy | Data Source |
|----------|-----------|-----------------|------------|-------------------|-------------|
| Research Notes | ChromaDB | Cohere embed-v3.0 | 1024 | Ensemble (Vector+BM25) + Rerank | Web, YouTube, Video Chunks |
| Video Search (Text) | ChromaDB | Cohere embed-v3.0 | 1024 | Ensemble (Vector+BM25) + Rerank | HVAC Video Transcripts |
| Video Search (Multimodal) | ChromaDB | Vertex AI multimodal | 1408 | Cosine + dedup + diversity | HVAC Channel Videos |
| Graphics (Text) | ChromaDB | Cohere embed-v3.0 | 1024 | Similarity search (k=50) | Image metadata/titles |
| Graphics (Image) | ChromaDB | Cohere embed-v4.0 | - | Raw vector query (k=50) | Actual image data |
| SkillCat Courses | Supabase pgvector | Cohere embed-v3.0 | 1024 | pgvector cosine + PostgreSQL FTS | SkillCat course catalog |
| NexTech Courses | ChromaDB | Cohere embed-v3.0 | 1024 | Ensemble (Vector+BM25) + Rerank | NexTech course catalog |

### Environment Variables Required
| Variable | Used By |
|----------|---------|
| `COHERE_API_KEY` | All pipelines (embeddings + reranking) |
| `VERTEX_AI_SA_B64` | Multimodal video search |
| `GDRIVE_SA_B64` | Google Drive storage for all ChromaDB pipelines |
| `SUPABASE_SECRET_API_KEY` | SkillCat curriculum mapping (Supabase) |

### Observability
All major retrieval functions are decorated with `@traceable` for LangSmith tracing, enabling end-to-end visibility into retrieval latency, relevance, and pipeline behavior.
