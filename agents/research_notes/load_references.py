from langchain_core.documents import Document
import json
from tqdm import tqdm
from services.web_page_loaders import get_docs_from_url, extract_image_links_from_markdown, get_webpage_title_fallback
from services.youtube_video_loader import get_video_id_from_url, get_yt_chapters_chunks_as_docs
from services.chunking_service import general_chunker
from services.helper_functions import create_and_populate_columns
from services.sheets_service import (
    get_sheet_data_and_df,
    create_or_read_worksheet,
    save_to_sheet,
    format_worksheet,
    delete_worksheet,
)
import streamlit as st
from services.smart_progress_bar import SmartProgressBar
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from langchain_core.documents import Document
from services.sheets_service import safe_get_sheet_data_and_df
from langsmith import traceable


@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Load References",
    "function_name": "list_video_research_references",
    "user_id": st.session_state.get("role", "anonymous")
})
## Video Based Outline References
def list_video_research_references(references_df, videos_research_df, video_chunks_df):
    """
    This function gets the video chunk docs list.

    :param references_df: The DataFrame containing the references data.
    :param video_research_sheet_name: The name of the sheet containing the video research data.
    :param video_chunk_sheet_name: The name of the sheet containing the video chunk data.
    :return: A list of `Document` objects representing the video chunk data.
    """
    # Check if the references_df already has the video research references
    if "source_origin" in references_df.columns and "Video Research" in references_df["source_origin"].values:
        print("Video research references already present in the references DataFrame.")
        return references_df

    # Skip if video_chunks_df is empty or does not have video_id column
    if 'video_id' not in video_chunks_df.columns or video_chunks_df.empty:
        print("No video chunks found, skipping video research references.")
        return references_df

    # Get the chunks for all relevant marked videos from video chunks df
    for index, row in videos_research_df[videos_research_df['Manual Review'].str.contains('Yes', na=False)].iterrows():

        source, source_origin, title, reference_type = row['video_url'], 'Video Research', row['title'], 'Youtube Video'

        # Check if reference already present in the df
        if source in references_df['source'].values:
            print(f"Reference {source} already present, skip adding to df.")
            continue

        chunks_dict = {"chunks": []}

        # Filter to get rows from video chunks df
        temp_df = video_chunks_df[video_chunks_df['video_id'] == row['video_id']]

        # Get text col count
        text_col_count = len([col for col in temp_df.columns if 'text_' in col])

        # Loop through each row and get page content and metadata
        for i, r in temp_df.iterrows():
            page_content = ''.join(
                    [r[f'text_{i}'] for i in range(text_col_count)]
                )
            metadata = json.loads(r['metadata'])
            chunks_dict['chunks'].append(
                {"page_content": page_content, "metadata": metadata}
            )

        # Get the video doc chunks
        references_df = pd.concat([references_df, pd.DataFrame({
            'source': [source],
            'source_origin': [source_origin],
            'title': [title],
            'reference_type': [reference_type],
        })], ignore_index=True)

        # Add the chunks to the references df
        references_df = create_and_populate_columns(
            df = references_df,
            text = json.dumps(chunks_dict),
            specific_index =  references_df.shape[0] - 1, # last row
            col_base_name = 'chunks',
            chunk_size = 49000,
        )

    return references_df


@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Load References",
    "function_name": "list_client_references",
    "user_id": st.session_state.get("role", "anonymous")
})
## Client References
def list_client_references(references_df, videos_research_df, video_chunks_df, preliminary_research_df, client_reference_df):
    """
    This function gets the client reference docs list.

    :param references_df: The DataFrame containing the references data.
    :param videos_research_df: The DataFrame containing video research data.
    :param video_chunks_df: The DataFrame containing video chunk data.
    :param preliminary_research_df: The DataFrame containing preliminary research data.
    :param client_reference_df: The DataFrame containing client reference data.
    :return: A list of `Document` objects representing the client reference data.
    """

    # Check if the references_df already has the client reference references
    if "source_origin" in references_df.columns and "Client Reference" in references_df["source_origin"].values:
        print("Client reference already present in the references DataFrame.")
        return references_df

    # Loop through the client reference df
    for ind, row in tqdm(client_reference_df.iterrows(), total = client_reference_df.shape[0]):

        chunks_dict = {"chunks": []}

        # Determine reference type
        source, source_origin, title, reference_type = row['Source Link'], 'Client Reference', row['Reference Title'], row['Reference Type']
        print(reference_type)

        # Check if reference already present in the df
        if source in references_df['source'].values:
            print(f"Reference {source} already present, skip adding to df.")
            continue

        # Check if reference content is already added in the sheet
        if row['Reference Content'] != "":
            # Get content from the sheet.
            reference_content = row['Reference Content']
            # Chunk the text
            chunked_list = general_chunker(text = reference_content)
            # Get text list as Document objects list
            docs = [
                {
                    "page_content": chunk["text"],
                    "metadata": {
                        "source": row["Source Link"],
                        "query": "Client reference"
                    }
                }
                for chunk in chunked_list
            ]

            chunks_dict['chunks'] = docs

            # Add to references df
            references_df = pd.concat([references_df, pd.DataFrame({
                'source': [source],
                'source_origin': [source_origin],
                'title': [title],
                'reference_type': [reference_type],
            })], ignore_index=True)

            # Add the chunks to the references df
            references_df = create_and_populate_columns(
                df = references_df,
                text = json.dumps(chunks_dict),
                specific_index = references_df.shape[0] - 1,  # last row
                col_base_name = 'chunks',
                chunk_size = 49000,
            )

        # Section to process web articles
        elif reference_type == 'Web Article':
            references_df = populate_reference_df_from_preliminary_research_df(
                source = source,
                source_origin = source_origin,
                title = title,
                reference_type = reference_type,
                query = 'Client reference',
                preliminary_research_df = preliminary_research_df,
                references_df = references_df
            )

        # Section to process youtube videos
        elif reference_type == 'Youtube Video':
            references_df = populate_reference_df_from_video_chunks_df(
                source = source,
                source_origin = source_origin,
                title = title,
                reference_type = reference_type,
                videos_research_df = videos_research_df,
                video_chunks_df = video_chunks_df,
                references_df = references_df
            )
    
    return references_df

@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Load References",
    "function_name": "populate_reference_df_from_video_chunks_df",
    "user_id": st.session_state.get("role", "anonymous")
})
def populate_reference_df_from_video_chunks_df(source, source_origin, title, reference_type, videos_research_df, video_chunks_df, references_df):
    """
    This function populates the references DataFrame with video chunks.

    :param source: The source URL of the video.
    :param source_origin: The origin of the source (e.g., "Client Reference").
    :param title: The title of the video.
    :param reference_type: The type of reference (e.g., "Youtube Video").
    :param videos_research_df: The DataFrame containing video research data.
    :param video_chunks_df: The DataFrame containing video chunk data.
    :param references_df: The DataFrame containing the references data.
    :return: The updated references DataFrame.
    """
    # Skip if video_chunks_df is empty or does not have video_id column
    if 'video_id' not in video_chunks_df.columns or video_chunks_df.empty:
        print("No video chunks found, skipping video chunk reference for this source.")
        return references_df

    video_id = get_video_id_from_url(source)

    # Check if video already chunked
    if video_id in video_chunks_df['video_id'].values:

        # Check if video added to video docs
        if source in references_df['source'].values:
            print(f"Video {video_id} already added to reference df")
            return references_df

        # If video not marked relevant, it would have not been added. Add it.
        else:
            # Add the chunks
            temp_df = video_chunks_df[video_chunks_df['video_id'] == video_id]

            # Get text col count
            text_col_count = len([col for col in temp_df.columns if 'text_' in col])

            docs = [
                {
                    "page_content": ''.join([str(r[f'text_{i}']) if pd.notna(r[f'text_{i}']) else '' for i in range(text_col_count)]),
                    "metadata": json.loads(r['metadata'])
                }
                for i, r in temp_df.iterrows()
            ]

            chunks_dict = {"chunks": docs}

            # Add to references df
            references_df = pd.concat([references_df, pd.DataFrame({
                'source': [source],
                'source_origin': [source_origin],
                'title': [title],
                'reference_type': [reference_type],
            })], ignore_index=True)

            # Add the chunks to the references df
            references_df = create_and_populate_columns(
                df = references_df,
                text = json.dumps(chunks_dict),
                specific_index = references_df.shape[0] - 1,  # last row
                col_base_name = 'chunks',
                chunk_size = 49000,
            )

            return references_df

    # If not already chunked
    else:
        # Load and chunk this
        # docs = get_yt_chapters_chunks_as_docs(video_id = video_id, video_title = row['Reference Title'], llm = 'gemini_2_flash')
        # Add to references df
        references_df = pd.concat([references_df, pd.DataFrame({
            'source': [source],
            'source_origin': [source_origin],
            'title': [title],
            'reference_type': [reference_type],
        })], ignore_index=True)
    
    return references_df

@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Load References",
    "function_name": "populate_reference_df_from_preliminary_research_df",
    "user_id": st.session_state.get("role", "anonymous")
})
def populate_reference_df_from_preliminary_research_df(source, source_origin, title, reference_type, query, preliminary_research_df, references_df):
    """
    This function populates the references DataFrame with data from the preliminary research DataFrame.

    :param source: The source URL of the article.
    :param source_origin: The origin of the source (e.g., "Web Research").
    :param title: The title of the article.
    :param reference_type: The type of reference (e.g., "Web Article").
    :param query: The query associated with the article.
    :param preliminary_research_df: The DataFrame containing preliminary research data.
    :param references_df: The DataFrame containing the references data.
    :return: The updated references DataFrame.
    """
    # Check if source already exists in references_df
    if source in references_df['source'].values:
        print(f"{source} already present in references df")
        return references_df

    # Otherwise add the reference to reference df
    references_df = pd.concat([references_df, pd.DataFrame({
        'source': [source],
        'source_origin': [source_origin],
        'title': [title],
        'reference_type': [reference_type],
    })], ignore_index=True)

    # Check in web research list (preliminary research)
    if 'article' in preliminary_research_df.columns and source in preliminary_research_df['article'].values:

        # Get source row from preliminary_research_df, not from filtered_df
        source_row = preliminary_research_df[preliminary_research_df['article'] == source]
        
        if not source_row.empty:
            article_content_col_count = len([col for col in preliminary_research_df.columns if 'article_content_' in col])
            # Get article content
            article_content = ''.join(
                [str(source_row[f'article_content_{i}'].values[0]) if pd.notna(source_row[f'article_content_{i}'].values[0]) else '' for i in range(article_content_col_count) 
                    if f'article_content_{i}' in source_row.columns and i < len(source_row.columns)]
            )

            # Chunk the doc
            chunked_list = general_chunker(article_content)
            chunked_docs = []
            for chunk in chunked_list:
                chunked_docs.append(
                    {
                        "page_content": chunk["text"],
                        "metadata": {
                            "source": source,
                            "query": query,
                            "level": chunk["level"] if "level" in chunk else "",
                            "title": chunk["title"] if "title" in chunk else "",
                            "images": '\n'.join(extract_image_links_from_markdown(chunk["text"]))
                        }
                    }
                )
            
            chunks_dict = {"chunks": chunked_docs}
            
            # Add the chunks to the references df
            references_df = create_and_populate_columns(
                df = references_df,
                text = json.dumps(chunks_dict),
                specific_index = references_df.shape[0] - 1,  # last row
                col_base_name = 'chunks',
                chunk_size = 49000,
            )
    
    return references_df

@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Load References",
    "function_name": "list_web_research_references",
    "user_id": st.session_state.get("role", "anonymous")
})
## Web Research References
def list_web_research_references(references_df, preliminary_research_df):
    """
    This function gets the web research references from the preliminary research DataFrame.

    :param references_df: The DataFrame containing the references data.
    :param preliminary_research_df: The DataFrame containing preliminary research data.
    :return: The updated references DataFrame.
    """
    # Check if the references_df already has the web research references
    if "source_origin" in references_df.columns and "Web Research" in references_df["source_origin"].values:
        print("Web research references already present in the references DataFrame.")
        return references_df

    # To apply the filter, get as string to avoid error
    preliminary_research_df = preliminary_research_df.astype(str)

    # Filter to get rows that contain the string <objective> to get web urls
    filtered_df = preliminary_research_df[preliminary_research_df['learning_objectives'].str.contains('<objective>')]

    for index, row in tqdm(filtered_df.iterrows(), total = filtered_df.shape[0]):

        source, source_origin, title, reference_type = row['article'], 'Web Research', row['query'], 'Web Article'

        references_df = populate_reference_df_from_preliminary_research_df(
            source = source,
            source_origin = source_origin,
            title = title,
            reference_type = reference_type,
            query = title,
            preliminary_research_df = preliminary_research_df,
            references_df = references_df,
        )

    return references_df        

        # return chunked_docs

    # # Initialize the list to store the doc chunks
    # web_research_doc_chunk_list = []

    # # def process_web_urls(url):
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


@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Load References",
    "function_name": "list_deep_research_references",
    "user_id": st.session_state.get("role", "anonymous")
})
## Deep Research References
def list_deep_research_references(references_df, deep_research_df, videos_research_df, video_chunks_df, preliminary_research_df):
    """
    This function gets the deep research references from the deep research DataFrame.

    :param references_df: The DataFrame containing the references data.
    :param deep_research_df: The DataFrame containing deep research data.
    :param videos_research_df: The DataFrame containing video research data.
    :param video_chunks_df: The DataFrame containing video chunk data.
    :param preliminary_research_df: The DataFrame containing preliminary research data.
    :return: The updated references DataFrame.
    """
    # Check if the references_df already has the deep research references
    if "source_origin" in references_df.columns and "Deep Research" in references_df["source_origin"].values:
        print("Deep research references already present in the references DataFrame.")
        return references_df

    # Loop through the deep research df
    for ind, row in tqdm(deep_research_df.iterrows(), total = deep_research_df.shape[0]):

        source_origin, title = 'Deep Research', row['subtopic_query']

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
            if source in references_df['source'].values:
                print(f"{source} already present in references df")
                continue
            
            reference_type = "Youtube Video" if ('youtube.com' in source or 'youtu.be' in source) else "Web Article"

            if reference_type == "Youtube Video":
                references_df = populate_reference_df_from_video_chunks_df(
                    source = source,
                    source_origin = source_origin,
                    title = title,
                    reference_type = reference_type,
                    videos_research_df = videos_research_df,
                    video_chunks_df = video_chunks_df,
                    references_df = references_df
                )
            else:
                references_df = populate_reference_df_from_preliminary_research_df(
                    source = source,
                    source_origin = source_origin,
                    title = title,
                    reference_type = reference_type,
                    query = subtopic_query,
                    preliminary_research_df = preliminary_research_df,
                    references_df = references_df
                )

    return references_df


@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Load References",
    "function_name": "list_topic_deep_research_references",
    "user_id": st.session_state.get("role", "anonymous")
})
## Topic Deep Research References
def list_topic_deep_research_references(references_df, topic_deep_research_df, videos_research_df, video_chunks_df, preliminary_research_df):
    """
    This function gets the topic deep research references from the topic deep research DataFrame.

    :param references_df: The DataFrame containing references data.
    :param topic_deep_research_df: The DataFrame containing topic deep research data.
    :param videos_research_df: The DataFrame containing video research data.
    :param video_chunks_df: The DataFrame containing video chunk data.
    :param preliminary_research_df: The DataFrame containing preliminary research data.
    :return: The updated references DataFrame.
    """
    # Check if the references_df already has the topic deep research references
    if "source_origin" in references_df.columns and "Topic Deep Research" in references_df["source_origin"].values:
        print("Topic deep research references already present in the references DataFrame.")
        return references_df
    
    # Loop through the topic deep research df
    for ind, row in tqdm(topic_deep_research_df.iterrows(), total = topic_deep_research_df.shape[0]):

        source_origin, title = 'Topic Deep Research', row['topic_query']

        # Get sources from the source column
        sources = row['source'].split('\n') if isinstance(row['source'], str) else []

        # Get sources unique list
        sources = list(set(sources))

        # Get the topic query
        topic_query = row['topic_query'] if 'topic_query' in row and isinstance(row['topic_query'], str) else 'Topic Deep Research'

        # Process each source link
        for source in sources:
            source = source.strip()
            if not source:
                continue
            
            # Skip if this source has already been processed in any row
            if source in references_df['source'].values:
                print(f"{source} already present in references df")
                continue
            
            reference_type = "Youtube Video" if ('youtube.com' in source or 'youtu.be' in source) else "Web Article"

            if reference_type == "Youtube Video":
                references_df = populate_reference_df_from_video_chunks_df(
                    source = source,
                    source_origin = source_origin,
                    title = title,
                    reference_type = reference_type,
                    videos_research_df = videos_research_df,
                    video_chunks_df = video_chunks_df,
                    references_df = references_df
                )
            else:
                references_df = populate_reference_df_from_preliminary_research_df(
                    source = source,
                    source_origin = source_origin,
                    title = title,
                    reference_type = reference_type,
                    query = topic_query,
                    preliminary_research_df = preliminary_research_df,
                    references_df = references_df
                )

    return references_df


@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Load References",
    "function_name": "list_topic_outline_references",
    "user_id": st.session_state.get("role", "anonymous")
})
## Topic Outline References
def list_topic_outline_references(references_df, topic_outline_df, videos_research_df, video_chunks_df, preliminary_research_df):
    """
    This function gets the topic outline references from the topic outline DataFrame.

    :param references_df: The DataFrame containing references data.
    :param topic_outline_df: The DataFrame containing topic outline data.
    :param videos_research_df: The DataFrame containing video research data.
    :param video_chunks_df: The DataFrame containing video chunk data.
    :param preliminary_research_df: The DataFrame containing preliminary research data.
    :return: The updated references DataFrame.
    """
    # Check if the references_df already has the topic outline references
    if "source_origin" in references_df.columns and "Topic Outline" in references_df["source_origin"].values:
        print("Topic outline references already present in the references DataFrame.")
        return references_df

    # Columns to process
    columns_to_process = ['References to be used as is', 'References for content']
    columns_to_process = [col for col in columns_to_process if col in topic_outline_df.columns]
    
    # Loop through the topic outline df
    for ind, row in tqdm(topic_outline_df.iterrows(), total=topic_outline_df.shape[0]):
        # Get the topic and learning objective for metadata
        topic = row['Topic'] if 'Topic' in row and isinstance(row['Topic'], str) else 'Topic Outline'
        learning_objective = row['Learning Objective'] if 'Learning Objective' in row and isinstance(row['Learning Objective'], str) else ''
        
        # Query for metadata
        query = f"{topic}: {learning_objective}"
        
        sources = []
        # Get sources from the reference column
        for ref_column in columns_to_process:
            if ref_column not in row or not isinstance(row[ref_column], str) or not row[ref_column].strip():
                continue
            
            # Get sources            
            sources.extend(row[ref_column].split('\n'))
        
        # Get sources unique list
        sources = list(set([source.strip() for source in sources if source.strip()]))
        
        # Process each source link
        for source in sources:
            # Skip if this source has already been processed in any row
            if source in references_df['source'].values:
                print(f"{source} already present in references df")
                continue
            
            reference_type = "Youtube Video" if ('youtube.com' in source or 'youtu.be' in source) else "Web Article"

            if reference_type == "Youtube Video":
                references_df = populate_reference_df_from_video_chunks_df(
                    source = source,
                    source_origin = "Topic Outline",
                    title = topic,
                    reference_type = reference_type,
                    videos_research_df = videos_research_df,
                    video_chunks_df = video_chunks_df,
                    references_df = references_df
                )
            else:
                references_df = populate_reference_df_from_preliminary_research_df(
                    source = source,
                    source_origin = "Topic Outline",
                    title = topic,
                    reference_type = reference_type,
                    query = query,
                    preliminary_research_df = preliminary_research_df,
                    references_df = references_df
                )

    return references_df

@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Load References",
    "function_name": "list_references",
    "user_id": st.session_state.get("role", "anonymous")
})
# Enlist all the references in a sheet along with source type
def list_references(sheet, videos_research_df, video_chunks_df, client_reference_df, preliminary_research_df, deep_research_df, topic_deep_research_df, topic_outline_df):
    """
    This function lists all the references in a sheet along with their source type.

    :param sheet: The name of the sheet containing the data.
    :param videos_research_df: The DataFrame containing video research data.
    :param video_chunks_df: The DataFrame containing video chunks data.
    :param client_reference_df: The DataFrame containing client reference data.
    :param preliminary_research_df: The DataFrame containing preliminary research data.
    :param deep_research_df: The DataFrame containing deep research data.
    :param topic_deep_research_df: The DataFrame containing topic deep research data.
    :param topic_outline_df: The DataFrame containing topic outline data.
    :return: A list of dictionaries representing the references and their source types.
    """

    # Create / read sheet
    references_sheet, references_df = create_or_read_worksheet(sheet = sheet, worksheet_name = "All References", rows = 2000, cols = 30)

    if references_df.empty:
        print("Creating new references DataFrame...")
        references_df = pd.DataFrame(columns = ['source', 'source_origin', 'title', 'reference_type'])

    # Add video research references
    references_df = list_video_research_references(references_df, videos_research_df, video_chunks_df)

    # Add client reference references
    # references_df = list_client_references(references_df, videos_research_df, video_chunks_df, preliminary_research_df, client_reference_df)

    # Add web research references
    references_df = list_web_research_references(references_df, preliminary_research_df)

    # Add deep research references
    references_df = list_deep_research_references(references_df, deep_research_df, videos_research_df, video_chunks_df, preliminary_research_df)

    # Conditionally add topic deep research references
    if not topic_deep_research_df.empty:
        references_df = list_topic_deep_research_references(
            references_df, topic_deep_research_df, videos_research_df, video_chunks_df, preliminary_research_df
        )

    # Conditionally add topic outline references
    if not topic_outline_df.empty:
        references_df = list_topic_outline_references(
            references_df, topic_outline_df, videos_research_df, video_chunks_df, preliminary_research_df
        )
    
    # Drop any references row with empty source
    references_df = references_df.fillna('')
    references_df = references_df[references_df['source'].str.strip() != '']

    # Load external references from Course info sheet
    try:
        _, course_info_df_ext = get_sheet_data_and_df(sheet, 'Course info')
        if 'External References' in course_info_df_ext.columns:
            external_refs_raw = course_info_df_ext.loc[0, 'External References']
            if pd.notna(external_refs_raw) and str(external_refs_raw).strip():
                external_urls = [u.strip() for u in str(external_refs_raw).split('\n') if u.strip()]
                for ext_url in external_urls:
                    if ext_url in references_df['source'].values:
                        continue
                    ext_ref_type = "Youtube Video" if ('youtube.com' in ext_url or 'youtu.be' in ext_url) else "Web Article"
                    new_row = {
                        'source': ext_url,
                        'source_origin': 'External References',
                        'title': '',
                        'reference_type': ext_ref_type,
                        'chunks_0': ''
                    }
                    references_df = pd.concat([references_df, pd.DataFrame([new_row])], ignore_index=True)
    except Exception as e:
        print(f"Error processing External References from Course info: {e}")

    try:
        _, course_info_df = get_sheet_data_and_df(sheet, 'Course info')
        outline_stage = course_info_df.loc[0, 'Outline Stage'] if 'Outline Stage' in course_info_df.columns else None
        if outline_stage == 'Initial':
            _, base_outline_df = get_sheet_data_and_df(sheet, 'Base Outline')
            # Only process if all required columns exist
            required_cols = ['References', 'Reference type', 'Reference usage']
            if all(col in base_outline_df.columns for col in required_cols):
                for idx, row in tqdm(base_outline_df.iterrows(), total=base_outline_df.shape[0], desc='Base Outline References'):
                    ref = str(row['References']).strip()
                    ref_type = str(row['Reference type']).strip()
                    ref_usage = str(row['Reference usage']).strip()
                    # Skip if any required value is empty
                    if not ref or not ref_type or not ref_usage:
                        continue
                    # Only process YouTube Video and Web Article
                    if ref_type not in ['Youtube Video', 'Web Article']:
                        continue
                    # Skip if already present
                    if ref in references_df['source'].values:
                        continue
                    # Add row to references_df
                    new_row = {
                        'source': ref,
                        'source_origin': 'References',
                        'title': '',
                        'reference_type': ref_type,
                        'chunks_0': ''
                    }
                    references_df = pd.concat([references_df, pd.DataFrame([new_row])], ignore_index=True)
    except Exception as e:
        print(f"Error processing Base Outline references: {e}")

    # Fill all chunk columns' nan with empty string before saving
    chunk_cols = [col for col in references_df.columns if col.startswith('chunks_')]
    references_df[chunk_cols] = references_df[chunk_cols].fillna('')

    # Final save to sheet after all tasks
    print('All topics processed. Saving final DataFrame to sheet.')
    save_to_sheet(worksheet=references_sheet, df=references_df)
    format_worksheet(worksheet=references_sheet)
    return references_sheet, references_df

@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Load References",
    "function_name": "load_references",
    "user_id": st.session_state.get("role", "anonymous")
})
def load_references(sheet, video_research_sheet_name = 'Videos Research', video_chunk_sheet_name = 'Video Chunks', 
                          client_reference_sheet_name = 'Client References', web_research_sheet_name = 'Preliminary Research',
                          deep_research_sheet_name = 'Deep Research', topic_outline_sheet_name = 'Topic Outline',
                          topic_deep_research_sheet_name = 'Topic Deep Research', llm = "gemini_2_flash"):
    """
    """
    # Load the sheets and df
    print("Loading the sheets...")
    videos_research_sheet, videos_research_df = safe_get_sheet_data_and_df(sheet = sheet, sheet_name = video_research_sheet_name)
    video_chunks_sheet, video_chunks_df = safe_get_sheet_data_and_df(sheet = sheet, sheet_name = video_chunk_sheet_name)
    # client_reference_sheet, client_reference_df = get_sheet_data_and_df(sheet = sheet, sheet_name = client_reference_sheet_name)
    preliminary_research_sheet, preliminary_research_df = safe_get_sheet_data_and_df(sheet = sheet, sheet_name = web_research_sheet_name)
    deep_research_sheet, deep_research_df = safe_get_sheet_data_and_df(sheet = sheet, sheet_name = deep_research_sheet_name)
    topic_outline_sheet, topic_outline_df = safe_get_sheet_data_and_df(sheet = sheet, sheet_name = topic_outline_sheet_name)
    topic_deep_research_sheet, topic_deep_research_df = safe_get_sheet_data_and_df(sheet = sheet, sheet_name = topic_deep_research_sheet_name)

    # If video chunks df is empty, create an empty DataFrame
    if video_chunks_df.empty:
        video_chunks_df = pd.DataFrame(columns = ['video_id'])

    # Defensive fallbacks so downstream list_*_references functions can filter safely
    # when a research section was skipped and its sheet doesn't exist.
    if videos_research_df.empty:
        videos_research_df = pd.DataFrame(columns = ['video_id', 'video_url', 'title', 'Manual Review'])
    if preliminary_research_df.empty:
        preliminary_research_df = pd.DataFrame(columns = ['article', 'query', 'learning_objectives'])
    if deep_research_df.empty:
        deep_research_df = pd.DataFrame(columns = ['subtopic_query', 'source'])

    # Enlist the sources from all sheets in a single sheet
    references_sheet, references_df = list_references(sheet, videos_research_df, video_chunks_df, None, preliminary_research_df, deep_research_df, topic_deep_research_df if not topic_deep_research_df.empty else pd.DataFrame(), topic_outline_df if not topic_outline_df.empty else pd.DataFrame())

    print("Loading references to chunks...")
    # Prepare for parallel processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit tasks for each topic
        for index, row in references_df.iterrows():

            # Skip if chunks present
            if row["chunks_0"] != "":
                continue

            if row.get("source_origin", "") in ("References", "External References"):
                if row["reference_type"] == "Youtube Video":
                    future = executor.submit(
                        get_yt_chapters_chunks_as_docs,
                        video_id = get_video_id_from_url(row["source"]),
                        video_title = None,
                        timestamped_transcript = None,
                        llm = llm
                    )
                elif row["reference_type"] == "Web Article":
                    future = executor.submit(
                        get_docs_from_url,
                        url = row["source"],
                        query = ""
                    )
                else:
                    continue  # skip Google Drive Video or unknown types
                futures_map[future] = index
                continue

            # Submit the task based on reference type (existing logic)
            if row["reference_type"] == "Youtube Video":
                future = executor.submit(
                    get_yt_chapters_chunks_as_docs,
                    video_id = get_video_id_from_url(row["source"]),
                    video_title = row["title"],
                    timestamped_transcript = None,
                    llm = llm
                )
            else:
                future = executor.submit(
                    get_docs_from_url,
                    url = row["source"],
                    query = row["title"]
                )

            # Map the Future to the index
            futures_map[future] = index

        # Collect the results as they complete
        total_tasks = len(futures_map)
        save_interval = 5  # how often to save (in number of completed tasks)
        
        # A simple progress object
        pending = set(futures_map.keys())

        # Initialize the progress tracker
        progress = SmartProgressBar(total_tasks=total_tasks, description="Percent complete", save_interval=save_interval)

        # Create a TQDM progress bar
        with tqdm(total=total_tasks, desc="Percent complete") as pbar:
            # Process in small batches until done or stuck
            while pending:
                # Wait for at least one future to complete, up to 60 seconds
                done, pending = wait(
                    pending,
                    timeout=60,
                    return_when=FIRST_COMPLETED
                )
                
                if not done:
                    # If no tasks completed within 60s, you can either break,
                    # or continue waiting, or handle differently.
                    print("No tasks completed within 60 seconds. Breaking out...")
                    for fut in pending:
                        if not fut.done():
                            fut.cancel()
                    print("Cancelled futures")
                    save_to_sheet(worksheet=references_sheet, df=references_df)
                    # Now shut down without waiting
                    executor.shutdown(wait=False, cancel_futures=True)
                    print("Canceled pending tasks and shut down executor.")
                    return
                    # break

                # Process the futures that have finished
                for future in done:
                    index = futures_map[future]
                    try:
                        docs = future.result()  # if it raised an error, you can catch below

                        # Turn docs into JSON
                        chunks_dict = {"chunks": []}
                        chunks_dict['chunks'] = [
                            {"page_content": doc.page_content, "metadata": doc.metadata}
                            for doc in docs
                        ]

                        # Update the references DataFrame
                        references_df = create_and_populate_columns(
                            df=references_df,
                            text=json.dumps(chunks_dict),
                            specific_index=index,
                            col_base_name='chunks',
                            chunk_size=49000
                        )
                        # Update the Title for Base Outline / External references
                        if references_df.at[index, 'source_origin'] in ('References', 'External References') and len(docs) > 0:
                            # Try to get a title from the first doc's metadata
                            title = docs[0].metadata.get('title', '') if hasattr(docs[0], 'metadata') else ''
                            if not title:
                                # fallback: for YouTube, use video_title; for web, use query or page title
                                if references_df.at[index, 'reference_type'] == 'Youtube Video':
                                    title = docs[0].metadata.get('video_title', '')
                                elif references_df.at[index, 'reference_type'] == 'Web Article':
                                    # Fallback: scrape the web page title
                                    title = get_webpage_title_fallback(references_df.at[index, 'source'])
                            references_df.at[index, 'title'] = title
                    except Exception as e:
                        print(f"Error processing index {index}: {e}")
                        references_df.at[index, 'chunks_0'] = "Error"
                        references_df.at[index, 'chunks_1'] = str(e)

                    # Update progress
                    pbar.update(1)
                    progress.update()

                    # Check if we should save partial progress
                    if progress.should_save():
                        print(f"Saving partial progress to sheet after {progress.completed_count} tasks completed.")
                        save_to_sheet(worksheet=references_sheet, df=references_df)

    # Final save to sheet after all tasks
    print('All topics processed. Saving final DataFrame to sheet.')
    save_to_sheet(worksheet=references_sheet, df=references_df)
    
    return


### Get All Chunks as Docs
def get_all_chunks_as_docs(sheet, worksheet_name = "All References"):
    """
    This function gets all chunks as docs from the all references sheet.

    :param sheet: The Google Sheets object.
    :param worksheet_name: The name of the sheet containing all references data.
    :return: A list of Document objects representing the chunks.
    """
    # Load the references sheet
    references_sheet, references_df = get_sheet_data_and_df(sheet = sheet, sheet_name = worksheet_name)

    reference_docs_chunk_list = []

    chunks_col_count = len([col for col in references_df.columns if 'chunks_' in col])

    for index, row in tqdm(references_df.iterrows(), total = references_df.shape[0]):

        # Skip if chunks is Error
        if row['chunks_0'] == "Error":
            continue

        source, source_origin, title, reference_type = row['source'], row['source_origin'], row['title'], row['reference_type'] 
        # Get the chunks from the row
        chunks = "".join([row[f'chunks_{i}'] for i in range(chunks_col_count)])

        if chunks == "":
            continue

        # Convert to JSON and get the chunks
        chunks_dict = json.loads(chunks)

        for chunk in chunks_dict['chunks']:
            page_content = chunk["page_content"]
            metadata = chunk["metadata"]
            if "source" not in metadata:
                metadata["source"] = source
            if "source_origin" not in metadata:
                metadata["source_origin"] = source_origin
            if "title" not in metadata:
                metadata["title"] = title
            if "reference_type" not in metadata:
                metadata["reference_type"] = reference_type

            reference_docs_chunk_list.append(
                Document(
                    page_content = page_content,
                    metadata = metadata
                )
            )

    return reference_docs_chunk_list

    # Load the sheets and df
    # print("Loading the sheets...")
    # videos_research_sheet, videos_research_df = get_sheet_data_and_df(sheet = sheet, sheet_name = video_research_sheet_name)
    # video_chunks_sheet, video_chunks_df = get_sheet_data_and_df(sheet = sheet, sheet_name = video_chunk_sheet_name)
    # client_reference_sheet, client_reference_df = get_sheet_data_and_df(sheet = sheet, sheet_name = client_reference_sheet_name)
    # preliminary_research_sheet, preliminary_research_df = get_sheet_data_and_df(sheet = sheet, sheet_name = web_research_sheet_name)
    # deep_research_sheet, deep_research_df = get_sheet_data_and_df(sheet = sheet, sheet_name = deep_research_sheet_name)
    # topic_outline_sheet, topic_outline_df = get_sheet_data_and_df(sheet = sheet, sheet_name = topic_outline_sheet_name)
    # topic_deep_research_sheet, topic_deep_research_df = get_sheet_data_and_df(sheet = sheet, sheet_name = topic_deep_research_sheet_name)
    
    # with st.spinner(text = "Getting video chunks...", show_time = True):
    #     print("Getting video chunks...")
    #     video_chunk_doc_list = get_video_chunk_doc_list(
    #         videos_research_df = videos_research_df,
    #         video_chunks_df = video_chunks_df
    #     )
    
    # with st.spinner(text = "Getting client references chunks...", show_time = True):
    #     print("Getting client references chunks...")
    #     client_reference_doc_chunk_list = get_client_reference_doc_list(
    #         videos_research_df = videos_research_df,
    #         video_chunks_df = video_chunks_df,
    #         client_reference_df = client_reference_df
    #     )

    # with st.spinner(text = "Getting web research chunks...", show_time = True):
    #     print("Getting web research chunks...")
    #     web_research_doc_chunk_list = get_web_research_doc_list(
    #         videos_research_df = videos_research_df,
    #         client_reference_df = client_reference_df,
    #         preliminary_research_df = preliminary_research_df
    #     )
    
    # with st.spinner(text = "Getting deep research chunks...", show_time = True):
    #     print("Getting deep research chunks...")
    #     deep_research_doc_chunk_list = get_deep_research_doc_list(
    #         deep_research_df = deep_research_df,
    #         videos_research_df = videos_research_df,
    #         client_reference_df = client_reference_df,
    #         preliminary_research_df = preliminary_research_df
    #     )
    
    # with st.spinner(text = "Getting topic deep research chunks...", show_time = True):
    #     print("Getting topic deep research chunks...")
    #     topic_deep_research_doc_chunk_list = get_topic_deep_research_doc_list(
    #         topic_deep_research_df = topic_deep_research_df,
    #         videos_research_df = videos_research_df,
    #         client_reference_df = client_reference_df,
    #         preliminary_research_df = preliminary_research_df
    #     )
    
    # with st.spinner(text = "Getting topic outline references chunks...", show_time = True):
    #     print("Getting topic outline references chunks...")
    #     topic_outline_doc_chunk_list = get_topic_outline_reference_doc_list(
    #         topic_outline_df = topic_outline_df,
    #         videos_research_df = videos_research_df,
    #         video_chunks_df = video_chunks_df,
    #         client_reference_df = client_reference_df,
    #         preliminary_research_df = preliminary_research_df,
    #         deep_research_df = deep_research_df
    #     )

    # print("Total chunks:", len(video_chunk_doc_list) + len(client_reference_doc_chunk_list) + 
    #       len(web_research_doc_chunk_list) + len(deep_research_doc_chunk_list) + 
    #       len(topic_deep_research_doc_chunk_list) + len(topic_outline_doc_chunk_list))

    # return video_chunk_doc_list + client_reference_doc_chunk_list + web_research_doc_chunk_list + deep_research_doc_chunk_list + topic_deep_research_doc_chunk_list + topic_outline_doc_chunk_list


def delete_all_references(sheet, worksheet_name="All References"):
    """Delete the All References worksheet."""
    delete_worksheet(sheet, worksheet_name)

