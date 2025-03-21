from langchain_core.documents import Document
import json
from tqdm import tqdm
from services.web_page_loaders import get_docs_from_url, extract_image_links_from_markdown
from services.youtube_video_loader import get_video_id_from_url
from services.youtube_video_loader import get_yt_chapters_chunks_as_docs
from services.chunking_service import general_chunker
# import concurrent.futures
from services.sheets_service import get_sheet_data_and_df
import streamlit as st
from services.smart_progress_bar import SmartProgressBar


## Video Based Outline References
def get_video_chunk_doc_list(videos_research_df, video_chunks_df):
    """
    This function gets the video chunk docs list.

    :param video_research_sheet_name: The name of the sheet containing the video research data.
    :param video_chunk_sheet_name: The name of the sheet containing the video chunk data.
    :return: A list of `Document` objects representing the video chunk data.
    """

    # Initialize list to store the video doc chunks
    video_chunk_doc_list = []

    # Get the chunks for all relevant marked videos from video chunks df
    for ind, row in videos_research_df[videos_research_df['Manual Review'] == 'Yes'].iterrows():
        #print(row['video_url'])

        # Filter to get rows from video chunks df
        temp_df = video_chunks_df[video_chunks_df['video_id'] == row['video_id']]

        # Get text col count
        text_col_count = len([col for col in temp_df.columns if 'text_' in col])

        # Loop through each row and get page content and metadata and construct Document object
        for i, r in temp_df.iterrows():
            page_content = ''.join(
                    [r[f'text_{i}'] for i in range(text_col_count)]
                )
            metadata = json.loads(r['metadata'])
            video_chunk_doc_list.append(Document(page_content=page_content, metadata=metadata))

    return video_chunk_doc_list


## Client References
def get_client_reference_doc_list(videos_research_df, video_chunks_df, client_reference_df):
    """
    This function gets the client reference docs list.

    :param videos_research_df: The DataFrame containing video research data.
    :param video_chunks_df: The DataFrame containing video chunk data.
    :param client_reference_df: The DataFrame containing client reference data.
    :return: A list of `Document` objects representing the client reference data.
    """

    # Initialize list to store doc chunks
    client_reference_doc_chunk_list = []

    # Loop through the client reference df
    for ind, row in tqdm(client_reference_df.iterrows(), total = client_reference_df.shape[0]):

        # Determine reference type
        reference_type = row['Reference Type']
        print(reference_type)

        # Section to process web articles
        if reference_type == 'Web Article':
            # Load and chunk this
            docs = get_docs_from_url(url = row['Source Link'], query = 'Client reference')

        # Section to process youtube videos
        elif reference_type == 'Youtube Video':
            video_id = get_video_id_from_url(row['Source Link'])
            # Check if video already chunked
            if video_id in video_chunks_df['video_id'].values:
                print(f"Video {video_id} already chunked")

                # Check if video added to video docs
                if video_id in videos_research_df[videos_research_df['Manual Review'] == 'Yes']['video_id'].values:
                    print(f"Video {video_id} already added to video docs")
                    continue
                # If video not marked relevant, it would have not been added. Add it.
                else:
                    print(f"Video {video_id} not added to video docs")
                    # Add the chunks
                    temp_df = video_chunks_df[video_chunks_df['video_id'] == video_id]

                    # Get text col count
                    text_col_count = len([col for col in temp_df.columns if 'text_' in col])

                    docs = [
                        Document(
                            page_content = ''.join([r[f'text_{i}'] for i in range(text_col_count)]),
                            metadata = json.loads(r['metadata'])
                        )
                        for i, r in temp_df.iterrows()
                    ]
            # If not already chunked
            else:
                # Load and chunk this
                docs = get_yt_chapters_chunks_as_docs(video_id = video_id, video_title = row['Reference Title'], llm = 'gemini_2_flash')

        # Section to process everything else such as audio podcasts, doc, etc.
        else:
            # Get content from the sheet. Functionality to process these reference type needs to be added in future.
            reference_content = row['Reference Content']
            # Chunk the text
            chunked_list = general_chunker(text = reference_content)
            # chunked_list = semantic_chunker(text = reference_content, chunk_size = 2000, chubker = sdpm_chunker)
            # Get text list as Document objects list
            docs = [
                Document(
                    page_content = chunk["text"],
                    metadata = {
                        "source": row["Source Link"],
                        "query": "Client reference"
                    }
                )
                for chunk in chunked_list
            ]

        #print(len(docs))
        client_reference_doc_chunk_list.extend(docs)


    return client_reference_doc_chunk_list


## Web Research References
def get_web_research_doc_list(videos_research_df, client_reference_df, preliminary_research_df):
    """
    This function gets the web research docs list.

    :param videos_research_df: The DataFrame containing video research data.
    :param client_reference_df: The DataFrame containing client reference data.
    :param preliminary_research_df: The DataFrame containing preliminary research data.
    :return: A list of `Document` objects representing the web research data.
    """

    # To apply the filter, get as string to avoid error
    preliminary_research_df = preliminary_research_df.astype(str)

    # Filter to get rows that contain the string <objective> to get web urls
    # web_urls = preliminary_research_df[preliminary_research_df['learning_objectives'].str.contains('<objective>')]['article'].to_list()
    filtered_df = preliminary_research_df[preliminary_research_df['learning_objectives'].str.contains('<objective>')]

    web_research_doc_chunk_list = []

    # Get video_transcript column count
    article_content_col_count = len([col for col in filtered_df.columns if 'article_content_' in col])

    total_tasks = filtered_df.shape[0]

    # Initialize the progress tracker
    progress = SmartProgressBar(total_tasks = total_tasks, description = "Percent complete")

    for index, row in tqdm(filtered_df.iterrows(), total = total_tasks):

        article_content = ''.join(
            [row[f'article_content_{i}'] for i in range(article_content_col_count)]
        )

        # Chunk the doc
        chunked_list = general_chunker(article_content)
        chunked_docs = []
        for chunk in chunked_list:
            chunked_docs.append(
                Document(
                    page_content = chunk["text"],
                    # Update metadata with chunk info
                    metadata = {
                        "source": row["article"],
                        "query": row["query"],
                        "level": chunk["level"] if "level" in chunk else "",
                        "title": chunk["title"] if "title" in chunk else "",
                        "images": '\n'.join(extract_image_links_from_markdown(chunk["text"]))
                    }
                )
            )
        
        web_research_doc_chunk_list.extend(chunked_docs)

        # Update progress
        progress.update()
    
    return web_research_doc_chunk_list

        # return chunked_docs

    # # Initialize the list to store the doc chunks
    # web_research_doc_chunk_list = []

    # def process_web_urls(url):
    #     # Check if url not already present in other two lists
    #     # Video Research list

    #     # Client Reference list
    #     if url in client_reference_df['Source Link'].values:
    #         print(f"{url} already present in client references")
    #         return []

    #     # Video Chunks list
    #     if url in videos_research_df[videos_research_df['Manual Review'] == 'Yes']['video_url'].to_list():
    #         print(f"{url} already present in video chunks")
    #         return []

    #     # Load and chunk this
    #     try:
    #         docs = get_docs_from_url(url = url, query = 'Web research')
    #     except:
    #         return []

    #     return docs

    # # Run in parallel
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     futures = {
    #         executor.submit(process_web_urls, url): ind
    #         for ind, url in enumerate(web_urls)
    #     }

    #     for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
    #         docs = future.result()
    #         if docs:
    #             web_research_doc_chunk_list.extend(docs)

    # return web_research_doc_chunk_list


## Deep Research References
def get_deep_research_doc_list(deep_research_df, videos_research_df, client_reference_df, preliminary_research_df):
    """
    This function gets the deep research docs list from the "Deep Research" sheet.
    Each cell in the "source" column may contain multiple links separated by new lines.

    :param deep_research_df: The DataFrame containing deep research data.
    :param videos_research_df: The DataFrame containing video research data.
    :param client_reference_df: The DataFrame containing client reference data.
    :param preliminary_research_df: The DataFrame containing preliminary research data.
    :return: A list of `Document` objects representing the deep research data.
    """
    # Initialize list to store doc chunks
    deep_research_doc_chunk_list = []

    # Initialize the progress tracker
    total_tasks = deep_research_df.shape[0]
    progress = SmartProgressBar(total_tasks = total_tasks, description = "Percent complete")

    filtered_df = preliminary_research_df[preliminary_research_df['learning_objectives'].str.contains('<objective>')]
    article_content_col_count = len([col for col in filtered_df.columns if 'article_content_' in col])
    
    # Track all processed sources across all rows to avoid duplicates
    processed_sources = set()
    
    # Loop through the deep research df
    for ind, row in tqdm(deep_research_df.iterrows(), total = total_tasks):
        # Get sources from the source column
        sources = row['source'].split('\n') if isinstance(row['source'], str) else []

        # Get sources unique list
        sources = list(set(sources))

        # Get the subtopic query
        subtopic_query = row['subtopic_query'] if 'subtopic_query' in row and isinstance(row['subtopic_query'], str) else 'Deep research'

        # Process each source link
        for source in sources:
            source = source.strip()
            if not source:
                continue
            
            # Skip if this source has already been processed in any row
            if source in processed_sources:
                print(f"{source} already processed in a previous row")
                continue
            
            # Add to processed sources
            processed_sources.add(source)
            
            # Check if source already exists in other sources
            
            # Check in client reference list
            if source in client_reference_df['Source Link'].values:
                print(f"{source} already present in client references")
                continue
            
            # Check in video research list
            video_id = get_video_id_from_url(source) if ('youtube.com' in source or 'youtu.be' in source) else None
            if video_id and video_id in videos_research_df[videos_research_df['Manual Review'] == 'Yes']['video_id'].to_list():
                print(f"{source} already present in video research")
                continue
            
            # Check in web research list (preliminary research)
            if 'article' in preliminary_research_df.columns and source in preliminary_research_df['article'].values:
                # Check learning objective col to check if already covered
                if source in filtered_df['article'].values:
                    print(f"{source} already present in web research")
                    continue
                else:
                    # Get source row from preliminary_research_df, not from filtered_df
                    source_row = preliminary_research_df[preliminary_research_df['article'] == source]
                    if not source_row.empty:
                        # Get article content
                        article_content = ''.join(
                            [source_row[f'article_content_{i}'].values[0] for i in range(article_content_col_count) 
                             if f'article_content_{i}' in source_row.columns and i < len(source_row.columns)]
                        )

                        # Chunk the doc
                        chunked_list = general_chunker(article_content)
                        chunked_docs = []
                        for chunk in chunked_list:
                            chunked_docs.append(
                                Document(
                                    page_content = chunk["text"],
                                    # Update metadata with chunk info
                                    metadata = {
                                        "source": source,
                                        "query": subtopic_query,
                                        "level": chunk["level"] if "level" in chunk else "",
                                        "title": chunk["title"] if "title" in chunk else "",
                                        "images": '\n'.join(extract_image_links_from_markdown(chunk["text"]))
                                    }
                                )
                            )
                        
                        deep_research_doc_chunk_list.extend(chunked_docs)
                        continue
            
            # Check if it's a YouTube video
            if video_id:
                # Load and chunk this
                try:
                    docs = get_yt_chapters_chunks_as_docs(video_id = video_id, video_title = f"Deep Research: {video_id}", llm = 'gemini_2_flash')
                    # Update metadata with the subtopic query
                    for doc in docs:
                        doc.metadata["query"] = subtopic_query
                    deep_research_doc_chunk_list.extend(docs)
                except Exception as e:
                    print(f"Error processing YouTube video {source}: {e}")
                    continue
            # Check if it's a web article
            else:
                # Load and chunk this
                try:
                    docs = get_docs_from_url(url = source, query = subtopic_query)
                    deep_research_doc_chunk_list.extend(docs)
                except Exception as e:
                    print(f"Error processing URL {source}: {e}")
                    continue
            
        # Update progress
        progress.update()
    
    return deep_research_doc_chunk_list


### Get All Chunks as Docs
def get_all_chunks_as_docs(sheet, video_research_sheet_name = 'Videos Research', video_chunk_sheet_name = 'Video Chunks', 
                          client_reference_sheet_name = 'Client References', web_research_sheet_name = 'Preliminary Research',
                          deep_research_sheet_name = 'Deep Research'):
    """
    This function gets all chunks as docs.

    :param sheet: The Google Sheets object.
    :param video_research_sheet_name: The name of the sheet containing the video research data.
    :param video_chunk_sheet_name: The name of the sheet containing the video chunk data.
    :param client_reference_sheet_name: The name of the sheet containing client reference data.
    :param web_research_sheet_name: The name of the sheet containing web research data.
    :param deep_research_sheet_name: The name of the sheet containing deep research data.
    :return: A list of `Document` objects representing all chunks.
    """
    # Load the sheets and df
    print("Loading the sheets...")
    videos_research_sheet, videos_research_df = get_sheet_data_and_df(sheet = sheet, sheet_name = video_research_sheet_name)
    video_chunks_sheet, video_chunks_df = get_sheet_data_and_df(sheet = sheet, sheet_name = video_chunk_sheet_name)
    client_reference_sheet, client_reference_df = get_sheet_data_and_df(sheet = sheet, sheet_name = client_reference_sheet_name)
    preliminary_research_sheet, preliminary_research_df = get_sheet_data_and_df(sheet = sheet, sheet_name = web_research_sheet_name)
    deep_research_sheet, deep_research_df = get_sheet_data_and_df(sheet = sheet, sheet_name = deep_research_sheet_name)
    
    with st.spinner(text = "Getting video chunks...", show_time = True):
        print("Getting video chunks...")
        video_chunk_doc_list = get_video_chunk_doc_list(
            videos_research_df = videos_research_df,
            video_chunks_df = video_chunks_df
        )
    
    with st.spinner(text = "Getting client references chunks...", show_time = True):
        print("Getting client references chunks...")
        client_reference_doc_chunk_list = get_client_reference_doc_list(
            videos_research_df = videos_research_df,
            video_chunks_df = video_chunks_df,
            client_reference_df = client_reference_df
        )

    with st.spinner(text = "Getting web research chunks...", show_time = True):
        print("Getting web research chunks...")
        web_research_doc_chunk_list = get_web_research_doc_list(
            videos_research_df = videos_research_df,
            client_reference_df = client_reference_df,
            preliminary_research_df = preliminary_research_df
        )
    
    with st.spinner(text = "Getting deep research chunks...", show_time = True):
        print("Getting deep research chunks...")
        deep_research_doc_chunk_list = get_deep_research_doc_list(
            deep_research_df = deep_research_df,
            videos_research_df = videos_research_df,
            client_reference_df = client_reference_df,
            preliminary_research_df = preliminary_research_df
        )

    print("Total chunks:", len(video_chunk_doc_list) + len(client_reference_doc_chunk_list) + 
          len(web_research_doc_chunk_list) + len(deep_research_doc_chunk_list))

    return video_chunk_doc_list + client_reference_doc_chunk_list + web_research_doc_chunk_list + deep_research_doc_chunk_list
