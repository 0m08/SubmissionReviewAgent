import time
import random
import re
from tqdm import tqdm
from services.smart_progress_bar import SmartProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.sheets_service import get_sheet_data_and_df, save_to_sheet, clear_worksheet
from agents.course_outline.video_search_tool.video_retriever import load_new_video_embeddings_chroma_db
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from modules.chain import Chain
from langsmith import traceable
import streamlit as st


search_query_generation_prompt = """You are a search query optimization agent tasked with transforming slide content sentences into highly effective search queries for retrieving relevant educational youtube videos related to the HVAC field. Your goal is to process multiple sentences from a single slide chunk and convert each sentence into a precise, context-aware query that maximizes the likelihood of finding the most relevant video segments from a specialized HVAC video database.

These are the inputs:

<course_information>
Course Name: {course_name}
Target Audience: {target_audience}
</course_information>

<content_context>
Topic: {topic_name}
Subtopic: {subtopic_name}
</content_context>

<slide_chunk_sentences>
{slide_sentences}
</slide_chunk_sentences>

Note: The sentences provided above are from a single slide and are presented in order. They are related to each other and together convey a cohesive instructional concept within the given topic and subtopic.

Your task is to analyze each sentence within its course and content context, then generate an optimized search query for each sentence that will retrieve the most relevant HVAC instructional videos from a vector database using semantic similarity search.

Follow these guidelines to create effective search queries:

1. Understand Sentence Relationships:
   - The sentences are from the same slide and are instructionally related.
   - Consider how sentences build on each other when resolving references.
   - If a pronoun in a later sentence refers to a concept introduced in an earlier sentence, use that context to resolve it.

2. Resolve Vague References and Pronouns:
   - If a sentence contains pronouns (e.g., "it", "this", "these") or vague references, use the topic, subtopic, and surrounding sentences to resolve them into specific technical terms.
   - For example: If sentence 1 introduces "reversing valve" and sentence 2 says "it regulates flow direction", convert sentence 2's query to reference "reversing valve" explicitly.
   - Always make implicit technical terms explicit by incorporating topic/subtopic information.

3. Optimize for Vector Similarity Search:
   - Each query will be used for semantic similarity matching against video transcripts and descriptions.
   - Include key concepts, component names, actions, relationships, etc that would appear in relevant video content.
   - Each query should be comprehensive enough to capture the full meaning but focused enough to retrieve specific content.

4. Maintain Instructional Focus:
   - Preserve the learning intent from each sentence (e.g., explanation, demonstration, identification, troubleshooting).
   - If a sentence implies a specific skill or knowledge area, ensure that's reflected in the query.
   - Frame queries to match how an instructor would teach the concept in a video.

5. Query Length and Specificity:
   - Aim for queries that are typically 5-15 words long - substantial enough to provide context but not overly verbose.
   - Include enough detail to differentiate each query from related concepts.

6. Consider Related Concepts:
   - Think about what related terms, processes, or components might be discussed alongside the main concept in an HVAC instructional video.
   - Include these contextual terms if they help narrow down to the most relevant videos.

7. Avoid Over-Specification:
   - Don't add information that's not implied by the sentence or the immediate topic/subtopic context.
   - Don't force the inclusion of the course name unless it's genuinely relevant to finding better videos.
   - Keep each query focused on the core concept from its corresponding sentence.

8. Ensure Query Diversity:
   - While maintaining consistency in terminology, ensure each query is distinct and targets its specific sentence's concept.
   - Avoid creating identical or nearly identical queries for different sentences unless the sentences themselves are repetitive.


Process each sentence systematically and provide your analysis and optimized search query for each one.

Present your output in the following structured format:

<output>

<overall_analysis>

Briefly analyze the slide chunk as a whole:
- What is the main instructional concept being taught across all sentences?
- What HVAC component, system, process, etc is the primary focus?
- Are there any cross-sentence references or dependencies that need to be resolved?
- What kind of search query to assign for each sentence?
</overall_analysis>

<sentence_queries>
Based on your above overall analysis, for each sentence, provide:

<sentence_1>

<original_sentence>
[The original sentence text]
</original_sentence>

<search_query>
[Your optimized search query for this sentence]
</search_query>

</sentence_1>

<sentence_2>

<original_sentence>
[The original sentence text]
</original_sentence>

<search_query>
[Your optimized search query for this sentence]
</search_query>

</sentence_2>

[Continue for all sentences...]

</sentence_queries>

</output>

Examples to guide your work:

<good_example>
Topic: "Refrigeration Components"
Subtopic: "Reversing Valve"
Sentences:
1. "The reversing valve is a key component in heat pump systems."
2. "It controls refrigerant flow direction during heating and cooling modes."
3. "Listen for clicking sounds that indicate it's switching between modes."

Good Output:
Sentence 1 Query: "Reversing valve key component in heat pump systems function and purpose"
Sentence 2 Query: "How reversing valve controls refrigerant flow direction in heating and cooling modes"
Sentence 3 Query: "Reversing valve clicking sounds when switching between heating and cooling modes"

Why Good: Each query is distinct, uses consistent terminology ("reversing valve"), resolves the implicit "it" in sentences 2 and 3, and maintains instructional focus.
</good_example>

<poor_example>
Topic: "Compressor Maintenance"
Subtopic: "Scroll Compressor"
Sentences:
1. "Scroll compressors use two spiral-shaped elements."
2. "One element remains stationary while the other orbits."
3. "This creates compression pockets for refrigerant."

Poor Queries:
Sentence 1: "spiral elements"
Sentence 2: "it orbits"
Sentence 3: "compression pockets for refrigerant"

Why Poor: Too vague, doesn't resolve pronouns ("it", "this"), missing technical terms ("scroll compressor"), lacks context, not instructionally focused.

Better Queries:
Sentence 1: "Scroll compressor two spiral-shaped scrolling elements design and configuration"
Sentence 2: "Scroll compressor stationary and orbiting scroll element operation and movement"
Sentence 3: "How scroll compressor creates compression pockets for refrigerant using orbiting scroll"
</poor_example>
"""


@traceable(metadata={
    "agent_name": "graphics_definition",
    "step_name": "Search Query Generation Agent",
    "function_name": "search_query_generation_agent",
    "user_email": st.session_state.get("user_email", "anonymous")
})
def search_query_generation_agent(course_name, target_audience, topic_name, subtopic_name, slide_sentences, llm="gemini_2_5_flash"):
    """
    Generate search queries for a given slide chunk.

    :param course_name: The name of the course.
    :param target_audience: The target audience of the course.
    :param topic_name: The name of the topic.
    :param subtopic_name: The name of the subtopic.
    :param slide_sentences: The sentences from the slide chunk (formatted as "Sentence 1: text...\nSentence 2: text...").
    :param llm: The LLM to use for query generation.
    :return: The raw LLM response with optimized queries.
    """

    search_query_chain = Chain(llm = llm)
    
    search_query_chain.add_message(
        role = "user",
        content = search_query_generation_prompt.format(
            course_name = course_name,
            target_audience = target_audience,
            topic_name = topic_name,
            subtopic_name = subtopic_name,
            slide_sentences = slide_sentences
        )
    )
    response = search_query_chain.run()
    return response


def parse_optimized_queries(llm_response):
    """
    Parse the LLM response to extract optimized search queries for each sentence.
    
    :param llm_response: The raw LLM response containing XML-formatted queries.
    :return: Dictionary mapping sentence index (1-based) to optimized query string.
    """
    
    queries_dict = {}
    
    # Extract all sentence blocks using regex
    sentence_pattern = r'<sentence_(\d+)>.*?<search_query>\s*(.*?)\s*</search_query>.*?</sentence_\1>'
    matches = re.findall(sentence_pattern, llm_response, re.DOTALL)
    
    for sentence_num, query in matches:
        sentence_idx = int(sentence_num)
        query_text = query.strip()
        if query_text:
            queries_dict[sentence_idx] = query_text
    
    return queries_dict


def format_sentences_for_prompt(sentences):
    """
    Format a list of sentences into the required format for the prompt.
    
    :param sentences: List of sentence strings.
    :return: Formatted string with "Sentence 1: ...\nSentence 2: ..." format.
    """

    formatted_lines = []
    for idx, sentence in enumerate(sentences, 1):
        formatted_lines.append(f"Sentence {idx}: {sentence}")
    return "\n\n".join(formatted_lines)


def split_into_sentences(text):
    """
    Split text into sentences using regex pattern. Handles periods, exclamation marks, and question marks.
    
    :param text: Input text to split.
    :return: List of sentences (stripped).
    """
    
    # Split on sentence-ending punctuation followed by space or end of string
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text.strip())
    
    # Filter out empty sentences and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def retrieve_relevant_docs_for_slide_chunks(course_name, target_audience, drive, topic_name, subtopic_name, slide_chunk, k=10, use_reranking=True):
    """
    Video Embedding based retrieval.
    
    :param course_name: The name of the course.
    :param target_audience: The target audience of the course.
    :param drive: Authenticated GoogleDrive instance (PyDrive2).
    :param topic_name: The name of the topic.
    :param subtopic_name: The name of the subtopic.
    :param slide_chunk: The slide chunk content.
    :param k: Number of top results to return.
    :param use_reranking: Whether to use Cohere reranking (default: True).
    :return: List of retrieved documents.
    """
    
    # Use the video embeddings vectorstore
    video_embeddings_folder_id = '1SSvxx1EJ3zgMPfvD8Zy5DaEgmI2prys7'
    
    # Load video embeddings retriever from Google Drive
    video_embeddings_chroma = load_new_video_embeddings_chroma_db(drive, video_embeddings_folder_id)
    video_embeddings_retriever = video_embeddings_chroma.as_retriever(
        search_kwargs={"k": k * 2 if use_reranking else k}  # Get more if reranking
    )
    
    # Optionally use Cohere reranking
    if use_reranking:
        compressor = CohereRerank(
            model="rerank-v3.5",
            top_n=k,
        )
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=video_embeddings_retriever
        )
    else:
        retriever = video_embeddings_retriever
    
    query = slide_chunk
    
    try:
        docs = retriever.invoke(query)
    except Exception as e:
        print(f"Video embeddings retriever failed: {e}")
        docs = []
    
    return docs[:k] if len(docs) > k else docs


@traceable(metadata={
    "agent_name": "graphics_definition",
    "step_name": "Task Function for Slide Chunks",
    "function_name": "task_fn_slide_chunks",
    "user_email": st.session_state.get("user_email", "anonymous")
})
def task_fn_slide_chunks(task, course_name, target_audience, drive, k=10, use_reranking=True, llm="gemini_2_5_flash"):
    """
    Worker function: run dual retrieval (direct sentence + LLM-optimized query) and build grouped YouTube embed URLs per sentence.
    
    Returns:
      - grouped_text: string to store in slide_chunk_videos with format:
            Sentence 1:
            <url>
            <url>
            
            Sentence 2:
            <url>
            ...
    """
    print(f"\nStarting task for Slide Chunk: {task['slide_chunk'][:100]}...")
    time.sleep(random.uniform(0.6, 1.4))  
    
    # Split slide chunk into sentences
    sentences = split_into_sentences(task['slide_chunk'])
    print(f"Split into {len(sentences)} sentences")
    
    # Generate optimized queries for all sentences using LLM
    print("Generating optimized search queries via LLM...")
    formatted_sentences = format_sentences_for_prompt(sentences)
    
    try:
        llm_response = search_query_generation_agent(
            course_name=course_name,
            target_audience=target_audience,
            topic_name=task['topic'],
            subtopic_name=task['subtopic'],
            slide_sentences=formatted_sentences,
            llm=llm
        )
        
        print("\n" + "="*80)
        print("LLM RESPONSE:")
        print("="*80)
        print(llm_response)
        print("="*80 + "\n")
        
        optimized_queries = parse_optimized_queries(llm_response)
        print(f"Generated {len(optimized_queries)} optimized queries")
    
    except Exception as e:
        print(f"Failed to generate optimized queries: {e}")
        optimized_queries = {}
    
    sentence_to_urls = {}
    new_context_chunks = []
    seen = set()  
    
    # Process each sentence with dual retrieval
    for sentence_idx, sentence in enumerate(sentences, 1):
        if not sentence.strip():
            continue
            
        print(f"\n  Processing sentence {sentence_idx}/{len(sentences)}: {sentence[:60]}...")
        
        per_sentence_k = 5  # 5 from each method = up to 10 total per sentence
        
        # METHOD 1: Direct sentence retrieval
        print(f"Method 1: Direct sentence query...")
        retrieved_docs_direct = retrieve_relevant_docs_for_slide_chunks(
            course_name=course_name,
            target_audience=target_audience,
            drive=drive,
            topic_name=task['topic'],
            subtopic_name=task['subtopic'],
            slide_chunk=sentence,  # Use raw sentence as query
            k=per_sentence_k,
            use_reranking=use_reranking
        )
        print(f"Method 1 retrieved: {len(retrieved_docs_direct)} docs")
        
        # METHOD 2: LLM-optimized query retrieval (new method)
        retrieved_docs_optimized = []
        if sentence_idx in optimized_queries:
            optimized_query = optimized_queries[sentence_idx]
            print(f"Method 2: Optimized query: '{optimized_query[:60]}...'")
            retrieved_docs_optimized = retrieve_relevant_docs_for_slide_chunks(
                course_name=course_name,
                target_audience=target_audience,
                drive=drive,
                topic_name=task['topic'],
                subtopic_name=task['subtopic'],
                slide_chunk=optimized_query,  # Use LLM-optimized query
                k=per_sentence_k,
                use_reranking=use_reranking
            )
            print(f"Method 2 retrieved: {len(retrieved_docs_optimized)} docs")
        else:
            print(f"Method 2 skipped: No optimized query available")
        
        # Combine results from both methods
        all_docs = retrieved_docs_direct + retrieved_docs_optimized
        
        # Process combined docs with deduplication
        sentence_urls = []
        for doc in all_docs:
            metadata = getattr(doc, "metadata", {})
            
            vid_id = metadata.get("video_id")
            if not vid_id:
                continue
            
            # Get timestamps directly from metadata 
            start_sec = metadata.get("start_time", 0)
            end_sec = metadata.get("end_time", None)
            segment_key = f"{vid_id}_{start_sec}"
            
            # Deduplicate: skip if same video_id + start_time already seen across the whole slide
            if segment_key in seen:
                continue
            seen.add(segment_key)
            
            # Use timestamps directly from metadata 
            abs_start = int(start_sec) if start_sec is not None else 0
            abs_end = int(end_sec) if end_sec is not None else None
            
            if abs_end is not None:
                url = f"https://www.youtube.com/embed/{vid_id}?start={abs_start}&end={abs_end}"
            else:
                url = f"https://www.youtube.com/embed/{vid_id}?start={abs_start}"
            
            sentence_urls.append(url)
            
            transcript = metadata.get("text_0", "").strip()
            if transcript:
                chunk = f"============= Doc id: AUTO =============\n{transcript}\n\n"
                new_context_chunks.append((vid_id, chunk))
        
        print(f" Total unique URLs for sentence {sentence_idx}: {len(sentence_urls)}")
        
        if sentence_urls:
            sentence_to_urls[sentence_idx] = sentence_urls
    
    # Build grouped text block (always include headers, even if no URLs)
    lines = []
    for idx in range(1, len(sentences) + 1):
        urls = sentence_to_urls.get(idx, [])
        lines.append(f"Sentence {idx}:")
        lines.extend(urls)
        lines.append("")  # blank line between sentences
    grouped_text = "\n".join(lines).strip()

    total_urls = sum(len(v) for v in sentence_to_urls.values())
    print(f" Finished Slide Chunk → {total_urls} URLs across {len(sentence_to_urls)} sentences with results\n")
    
    return grouped_text, new_context_chunks


@traceable(metadata={
    "agent_name": "graphics_definition",
    "step_name": "Run Video Search for Slide Chunks",
    "function_name": "run_video_search_for_slide_chunks",
    "user_email": st.session_state.get("user_email", "anonymous")
})
def run_video_search_for_slide_chunks(course_name, target_audience, sheet, worksheet_name, drive, k=10, use_reranking=True, verbose=True, llm="gemini_2_5_flash"):
    """
    Search for relevant videos using slide chunk content with dual retrieval (direct + LLM-optimized queries).
    
    :param course_name: The name of the course.
    :param target_audience: The target audience of the course.
    :param sheet: The Google Sheets object.
    :param worksheet_name: The name of the worksheet containing slide chunks.
    :param drive: Authenticated GoogleDrive instance (PyDrive2).
    :param k: Number of top results to retrieve per slide chunk.
    :param use_reranking: Whether to use Cohere reranking (default: True).
    :param verbose: Whether to print progress messages.
    :param llm: The LLM to use for query generation 
    :return: List of slide_chunk_videos column values after processing.
    """
    
    slide_chunks_worksheet, slide_chunks_df = get_sheet_data_and_df(
        sheet=sheet,
        sheet_name=worksheet_name
    )
    
    if "Slide Chunk" not in slide_chunks_df.columns:
        raise ValueError("Sheet must contain a column named 'Slide Chunk'")
    
    # Add slide_chunk_videos column if it doesn't exist
    if "slide_chunk_videos" not in slide_chunks_df.columns:
        slide_chunks_df["slide_chunk_videos"] = ""
    
    task_list = []
    for row_index, row in slide_chunks_df.iterrows():
        slide_content = row.get("Slide Chunk")
        if not isinstance(slide_content, str) or not slide_content.strip():
            continue
        
        # Skip rows that already have slide chunk videos
        if row.get("slide_chunk_videos") and str(row["slide_chunk_videos"]).strip():
            continue
            
        # Skip certain slide types that don't need video clips
        slide_type = row.get("Slide Type", "").strip().lower()
        if slide_type in ["video", "transition", "summary"]:
            continue
            
        task_list.append({
            "row_index": row_index,
            "slide_chunk": slide_content.strip(),
            "topic": row.get("Topic", ""),
            "subtopic": row.get("Subtopic", ""),
        })
    
    total_tasks = len(task_list)
    print(f"Starting slide chunk video retrieval for {total_tasks} slides...")
    
    if total_tasks == 0:
        print("No slide chunks to process for video search.")
        return slide_chunks_df["slide_chunk_videos"].tolist()
    
    max_workers = 10
    futures_map = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for task in task_list:
            future = executor.submit(
                task_fn_slide_chunks,
                task,
                course_name,
                target_audience,
                drive,
                k,
                use_reranking,
                llm
            )
            futures_map[future] = task
        
        progress = SmartProgressBar(total_tasks=total_tasks, description="Percent complete", save_interval=5)
        
        for future in tqdm(as_completed(futures_map), total=total_tasks):
            task = futures_map[future]
            row_index = task["row_index"]
            slide_chunk = task["slide_chunk"]
            
            try:
                grouped_text, new_context_chunks = future.result()
                
                # Update slide_chunk_videos column with grouped text per sentence
                slide_chunks_df.at[row_index, "slide_chunk_videos"] = grouped_text
                
                progress.update()
                if progress.should_save():
                    save_to_sheet(slide_chunks_worksheet, slide_chunks_df)
            
            except Exception as e:
                print(f"Error processing slide chunk '{slide_chunk[:50]}...' (row {row_index + 1}): {e}")
    
    print("All slide chunks processed. Saving final results.")
    save_to_sheet(slide_chunks_worksheet, slide_chunks_df)
    return slide_chunks_df["slide_chunk_videos"].tolist()


def delete_slide_chunk_videos(sheet, worksheet_name="Slide Chunks"):
    """Remove the slide_chunk_videos column from the worksheet."""
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    if "slide_chunk_videos" in df.columns:
        df = df.drop(columns=["slide_chunk_videos"])
        clear_worksheet(ws)
        save_to_sheet(ws, df)
        print(f"✅ Removed slide_chunk_videos column from {worksheet_name}")
    else:
        print(f"✅ slide_chunk_videos column not found in {worksheet_name}")