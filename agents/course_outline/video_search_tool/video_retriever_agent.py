from modules.chain import Chain
from services.llm_service import csv_list_parser
from agents.course_outline.video_search_tool.youtube_search_tool import search_youtube_videos
from agents.course_outline.video_search_tool.video_retriever import get_compression_retriever

retriver_agent_system_prompt = """You are a retriever agent with access to a knowledge base. Your task is to retrieve the best results for a given query.

Here are the details of this task you should know:

<course_details>
We are creating a course with following details -
Course Name: {course_name}
Target audience: {target_audience}
Course outline:
{course_outline}
</course_details>


<current_focus>
Within this course, the current focus is to retrieve relevant videos for learning objectives for the following topic, subtopic and this specific learning objective:
Topic: {topic_name}
Subtopic: {subtopic_name}
Learning objective: {learning_objective}
</current_focus>

Your task is to retrieve the best results for the given learning objective while keeping the topic, subtopic and the overall course context in mind. 


Thus, formulate a search query that can effectively retrieve the relevant information.

Best practices for formulating search query:
- The search query you create in most cases can be same / very similar to the learning objective.
- The search query will be executed in a vectorstore (similarity search). Thus, you can employ techniques such as query expansion, query decomposition, HyDE, etc. incase the simple query is not yeilding good results.
- The topic and subtopic can be used to provide additional context to the learning objective, if the learning objective if looked in isolation is vague / too generic.For example, if the learning objective is "Recognize its placement within the condenser", the "it" is vague. But if we know the topic is "Refrigeration Components" and the subtopic is "Reversing Valve", we can infer that "it" refers to the "Reversing Valve". 
- Thus, the search query can be modified to "Recognize the placement of the Reversing Valve within the condenser".
- Evolve and adapt your search queries incase they don't yeild relevant results or incase you need some more surrounding information to cover the LO properly.
- If initial queries don't yield perfect matches, try broader queries or related concepts to ensure you find at least one relevant video.

Present yout output in the following format:
<observations>
[Place your observations of the step taken within these tags. It is ok for this section to be quite long and comprehensive.
Start by analysing each and every doc and then formulate overall observations.]
</observations>
<verdict>
[Based on your observations so far, decide whether to continue with retrieval or to stop. Verdict should be one of these - TERMINATE or CONTINUE]
</verdict>
<selected_video_ids>
[Put the ids of all the docs that are highly relevant for the current LO we are focussing on. Each doc id should be in a new line. New doc ids should be appended to the same list.]
</selected_video_ids>
<action>
[In case the verdict is to CONTINUE, think and ideate how / what query to create next to get relevant retrievals for the current LO. If verdict is to terminate, leave this blank.]
</action>
<query>
[If continue, then place the query within these tags, else leave this as blank.]
</query>
<search_on>
["VECTOR STORE" or "YOUTUBE SEARCH". Only choose youtube search if multiple queries against vector store did not retrieve relevant docs. Leave blank if no query.]
</search_on>

Remember, your goal is to retrieve the best information possible for the given learning objective. Be mindful of the docs you select. Make sure you follow the output format.
"""


def get_docs_as_string(docs, all_docs):
    """
    This function wraps the docs in a readable manner while removing any duplicate docs.

    :param docs: The docs to be wrapped.
    :param all_docs: The list of all docs.
    :return: The docs as a string and the updated list of all docs.
    """
    docs_as_string = ''
    for doc in docs:
        if doc in all_docs:
            # Still add to doc string but don't add actual content
            doc_id = all_docs.index(doc)
            docs_as_string += f'============= Doc id: {doc_id} =============\n'
            docs_as_string += 'Page content not added since this doc is repeating and has already appeared earlier.\n\n'
        else:
            docs_as_string += f'============= Doc id: {len(all_docs)} =============\n'
            docs_as_string += doc.page_content + '\n\n'
            all_docs.append(doc)

    # Replace {} with {{}}
    docs_as_string = docs_as_string.replace('{', '{{')
    docs_as_string = docs_as_string.replace('}', '}}')

    return docs_as_string, all_docs


def retrieve_relevant_docs(
    course_name,
    target_audience,
    course_outline,
    drive,
    topic_name,
    subtopic_name,
    learning_objective,
    max_turns=3,
    llm='gemini_2_flash'
):
    
    """
    This function retrieves relevant docs for a given learning objective using a retriever agent.
    :param course_name: The name of the course.
    :param target_audience: The target audience of the course.
    :param course_outline: The outline of the course.
    : param drive: Authenticated GoogleDrive instance (PyDrive2).
    : param topic_name: The name of the topic.
    : param subtopic_name: The name of the subtopic.
    : param learning_objective: The learning objective.
    : param max_turns: The maximum number of turns for the agent.
    : param llm: The LLM to be used.
    : return: List of selected video ids and all docs retrieved during the process.
    """

    retriver_agent = Chain(llm=llm, tags=['verdict', 'selected_video_ids', 'action', 'query', 'search_on'])
    central_folder_id = '1kovlkUd3pN5IGDB16LC2H8grvmQOhXHy'

    compression_retriever = get_compression_retriever(
        central_folder_id, drive,
        retriever_1_weight=0.5,
        retriever_2_weight=0.5
    )

    retriver_agent.add_message(
        role='system',
        content=retriver_agent_system_prompt.format(
            course_name=course_name,
            target_audience=target_audience,
            course_outline=course_outline,
            topic_name=topic_name,
            subtopic_name=subtopic_name,
            learning_objective=learning_objective
        )
    )
    all_docs = []

    # List to store selected doc ids
    selected_video_ids = []

    # For first user message, query is same as LO
    query = learning_objective

    for turn in range(max_turns):
        # --- Use compression retriever first ---
        print(f"[DEBUG] Turn {turn+1}: Using compression retriever for '{query}'")
        try:
            docs = compression_retriever.invoke(query)
        except Exception as e:
            print(f"[ERROR] Compression retriever failed: {e}")
            docs = []

        
        docs_as_string, all_docs = get_docs_as_string(docs, all_docs)

        retriver_agent.add_message(
            role='user',
            content=f'Query: {query}\n\nRetrieved_docs:\n{docs_as_string}'
        )

        response = retriver_agent.run()

        # Collect agent-selected IDs
        new_ids = csv_list_parser.invoke(response['selected_video_ids'])
        for doc_id in new_ids:
            if doc_id not in selected_video_ids:
                selected_video_ids.append(doc_id)

        if selected_video_ids:
            print(f"[DEBUG] Found {len(selected_video_ids)} relevant videos at turn {turn+1}. Terminating early.")
            return selected_video_ids, all_docs

        # --- Otherwise, refine and check if YouTube search is requested ---
        query = response.get('query', query)
        search_on = response.get('search_on', '')

        if "youtube search" in search_on.lower():
            print(f"[DEBUG] Turn {turn+1}: Using YouTube search retriever for query '{query}'")
            try:
                yt_search_retriever = search_youtube_videos(query, max_results=5)
                docs = yt_search_retriever.invoke(query)
            except Exception as e:
                print(f"[ERROR] YouTube retriever failed: {e}")
                docs = []

            # NOTE: Not passing through get_docs_as_string here
            return docs, all_docs

    print(f"[DEBUG] Returning {len(all_docs)} total docs, {len(selected_video_ids)} selected IDs")
    return selected_video_ids, all_docs