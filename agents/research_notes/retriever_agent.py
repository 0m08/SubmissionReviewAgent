from modules.chain import Chain
from services.llm_service import csv_list_parser
from agents.research_notes.retriever import get_compression_retriever
from services.sheets_service import get_sheet_data_and_df
from tqdm import tqdm
from services.helper_functions import create_and_populate_columns
from services.helper_functions import get_outline_with_los


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
Within this course, the current focus is to create research notes for the following topic, subtopic and this specific learning objective:
Topic: {topic_name}
Subtopic: {subtopic_name}
Learning objective: {learning_objective}
</current_focus>

Your task is to retrieve the best results for the given learning objective while keeping the topic, subtopic and the overall course context in mind.

Thus, formulate a search query that can effectively retrieve the relevant information.

Best practices for formulating search query:
- The search query you create in most cases can be same / very similar to the learning objective.
- The search query will be executed in a vectorstore (similarity search). Thus, you can employ techniques such as query expansion, query decomposition, HyDE, etc. incase the simple query is not yeilding good results.
- Modify search query to include additional context incase the learning objective if looked in isolation is vague / too generic.
- Evolve and adapt your search queries incase they don't yeild relevant results or incase you need some more surrounding information to cover the LO properly.

Output format:
<observations>
[Place your observations of the step taken within these tags. It is ok for this section to be quite long and comprehensive.
Start by analysing each and every doc and then formulate overall observations.]
</observations>
<verdict>
[Based on your observations so far, decide whether to continue with retrieval or to stop. Verdict should be one of these - TERMINATE or CONTINUE]
</verdict>
<selected_doc_ids>
[Put the ids of all the docs that are highly relevant for the current LO we are focussing on. Each doc id should be in a new line. New doc ids should be appended to the same list.]
</selected_doc_ids>
<action>
[In case the verdict is to CONTINUE, think and ideate how / what query to create next to get relevant retrievals for the current LO. If verdict is to terminate, leave this blank.]
</action>
<query>
[If continue, then place the query within these tags, else leave this as blank.]
</query>

Remember, your goal is to retrieve the best information possible for the given learning objective. Be mindful of the docs you select.
"""


# Helper Function to wrap docs in a readable manner while removing any duplicate docs
def get_docs_as_string(docs, all_docs):
    """
    This function wraps the docs in a readable manner while removing any duplicate docs.

    :param: docs: The docs to be wrapped.
    :param: all_docs: The list of all docs.
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


def retrieve_relevant_docs(compression_retriever, course_name, target_audience, course_outline, topic_name, subtopic_name, learning_objective, max_turns = 5, llm = 'groq'):
    """
    This function retrieves the relevant documents for the given learning objective.

    :param: compression_retriever: The compression retriever.
    :param: course_name: The name of the course.
    :param: target_audience: The target audience of the course.
    :param: course_outline: The outline of the course.
    :param: topic_name: The name of the topic.
    :param: subtopic_name: The name of the subtopic.
    :param: learning_objective: The learning objective.
    :param: max_turns: The maximum number of turns to retrieve the relevant documents.
    :param: llm: The language model to use.
    :return: The selected doc ids and all docs.
    """

    # Construct the chain
    retriver_agent = Chain(llm = llm, tags = ['verdict', 'selected_doc_ids', 'action', 'query'])

    # Add system message
    retriver_agent.add_message(
        role = 'system',
        content = retriver_agent_system_prompt.format(
            course_name = course_name,
            target_audience = target_audience,
            course_outline = course_outline,
            topic_name = topic_name,
            subtopic_name = subtopic_name,
            learning_objective = learning_objective
        )
    )

    # Run the retriver first time with LO as query
    # docs = retriever_with_filter(query = learning_objective)
    docs = compression_retriever.invoke(learning_objective)

    # List of all docs captured
    all_docs = []

    # List to store selected doc ids
    selected_doc_ids = []

    # For first user message, query is same as LO
    query = learning_objective

    for i in range(max_turns):

        # Get the docs as string and update all docs list
        docs_as_string, all_docs = get_docs_as_string(docs, all_docs)

        # Add user message - query + retrieved docs
        retriver_agent.add_message(
            role = 'user',
            content = f'Query: {query}\n\nRetrived_docs:\n{docs_as_string}'
        )

        # Run the agent
        response = retriver_agent.run()

        # Get selected doc ids
        selected_doc_ids.extend(
            csv_list_parser.invoke(response['selected_doc_ids'])
        )

        # Check the verdict
        verdict = response['verdict']
        # Stop the loop if terminate
        if 'TERMINATE' in verdict:
            print('Terminating the loop')
            break
        # Else continue the loop
        else:
            # Extract the next query and get the docs
            query = response['query']

    return selected_doc_ids, all_docs


def run_retriever_agent_for_all_rows(root_folder_id, drive, sheet, worksheet_name, course_name, target_audience, llm):
    """
    This function runs the retriever agent for all df rows.

    :param root_folder_id: The ID of the root folder containing 'Pickle files' and 'Vectorstore files'.
    :param drive: Authenticated GoogleDrive instance (PyDrive2).
    :param sheet: The sheet object.
    :param worksheet_name: The worksheet name.
    :param course_name: The course name.
    :param target_audience: The target audience.
    :param llm: The language model to use.
    :return: None
    """
    # Load the retriever
    compression_retriever = get_compression_retriever(
        course_name = course_name,
        root_folder_id = root_folder_id,
        drive = drive
    )

    # Read the sheet and df
    course_outline_with_lo_sheet, course_outline_with_lo_df = get_sheet_data_and_df(sheet = sheet, sheet_name = worksheet_name)
    # Get the course outline
    course_outline = get_outline_with_los(df = course_outline_with_lo_df, include_learning_objectives = False)

    # Create columns in df if not already present
    if 'context_0' not in course_outline_with_lo_df.columns:
        course_outline_with_lo_df['context_0'] = ''
    if 'Research Notes' not in course_outline_with_lo_df.columns:
        course_outline_with_lo_df['Research Notes'] = ''
    
    # Check if this step is already done by checking last row of Context column
    if course_outline_with_lo_df.iloc[-1]['context_0'] != '':
        print('Context already populated')
        return
    
    # Otherwise run the for loop
    for index, row in tqdm(course_outline_with_lo_df.iterrows(), total=course_outline_with_lo_df.shape[0]):
        
        # Check if row already populated
        if row['context_0'] != '':
            print(f'Skipping row {index}. Already populated')
            continue
        
        # Get the LOs for this row / subtopic
        learning_objectives = row['Learning Objectives'].split('\n')

        context = ""
        context_docs = []
        for lo in learning_objectives:
            # Run the retriever agent
            selected_doc_ids, all_docs = retrieve_relevant_docs(
                compression_retriever = compression_retriever,
                course_name = course_name,
                target_audience = target_audience,
                course_outline = course_outline,
                topic_name = row['Topic'],
                subtopic_name = row['Subtopic'],
                learning_objective = lo,
                llm = llm
            )

            for id in selected_doc_ids:
                # Check if id can be converted to int
                if id.isdigit():
                    context_doc = all_docs[int(id)]
                    # Check if doc already added
                    if context_doc in context_docs:
                        print(f'Skipping doc id {id} as it is already added')
                        continue
                    # Else add the doc to the context
                    else:
                        context += f'============= Doc id: {len(context_docs)} =============\n'
                        context += context_doc.page_content + '\n\n'
                        context_docs.append(context_doc)
                else:
                    print(f'Skipping doc id {id} as it is not an integer')
                    continue
        
        # Update the df row with context
        course_outline_with_lo_df = create_and_populate_columns(
            df = course_outline_with_lo_df,
            text = context,
            specific_index = index,
            col_base_name = 'context',
            chunk_size = 49000
        )

        # Save to sheet every N rows
        N = 5
        if index % N == 0:
            print(f'Saving to sheet at row {index}')
            course_outline_with_lo_df = course_outline_with_lo_df.astype(str)
            course_outline_with_lo_sheet.update([course_outline_with_lo_df.columns.values.tolist()] + course_outline_with_lo_df.values.tolist())
        
    # Save to sheet at the end of the loop
    print('Saving to sheet at the end of the loop')
    course_outline_with_lo_df = course_outline_with_lo_df.astype(str)
    course_outline_with_lo_sheet.update([course_outline_with_lo_df.columns.values.tolist()] + course_outline_with_lo_df.values.tolist())

    return


def manual_input_review_context(sheet, worksheet_name):
    """
    This function checks if the user has properly reviewed the context and marked the review as Done or not.

    :param sheet: The sheet object.
    :param worksheet_name: The worksheet name.
    """

    course_outline_with_lo_sheet, course_outline_with_lo_df = get_sheet_data_and_df(sheet = sheet, sheet_name = worksheet_name)

    if "Context Review" not in course_outline_with_lo_df.columns:
        raise Exception(f"Manually add the following column `Context Review` inside this sheet - {course_outline_with_lo_sheet.url}.\nReview the context_n columns and enter `Done` in the first row of `Context Review` column.")
    
    elif course_outline_with_lo_df["Context Review"].values[0] != "Done":
        raise ValueError("Review the context_n columns and enter `Done` in the first row of `Context Review` column.")
    else:
        print("Context is reviewed")
        return True
