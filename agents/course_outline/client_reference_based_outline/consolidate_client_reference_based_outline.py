from modules.chain import Chain
import pandas as pd
from services.sheets_service import get_sheet_data_and_df, save_to_sheet
from modules.proposer_agents import get_proposer_and_aggregator_agents
from tqdm import tqdm
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.smart_progress_bar import SmartProgressBar
from services.helper_functions import get_outline_with_los


consolidate_reference_outlines_prompt = """You are tasked with creating a final course outline based on partial, potentially overlapping outlines generated from reference content. Your goal is to produce a coherent, well-structured course outline that effectively covers all necessary topics.

First, review the following course information:

<course_info>
Course Name: <course_name>{course_name}</course_name>

Target Audience: <target_audience>{target_audience}</target_audience>

Tentative Outline:
<tentative_outline>
{tentative_outline}
</tentative_outline>

Required Topic Count:
<required_topic_count>
{required_topic_count}
</required_topic_count>
</course_info>

Now, examine the partial outlines generated from reference content:

<partial_outlines>
{partial_outlines}
</partial_outlines>

Your task is to analyze these partial outlines and create a final, comprehensive course outline. Follow these steps:

1. Analyze the partial outlines:
   - Identify common themes and topics across the outlines
   - Note any contradictions or inconsistencies
   - Recognize the overall structure and flow of the course
   - Compare the partial outlines to the tentative outline provided in the course info
   - Pay attention to any optional user comments or suggestions for the partial outlines

2. Identify the best outline structure:
   - Evaluate which outline(s) cover the concepts most effectively
   - Consider how well each outline aligns with the course name and target audience

3. Combine the outlines:
   - Use the best outline structure as a foundation
   - Merge overlapping content, avoiding duplication
   - Resolve any contradictions by choosing the most logical or frequently mentioned information
   - Incorporate unique concepts from other outlines in a calculated manner
   - Organize the content in a coherent and logical order
   - Ensure all key topics from the partial outlines are included

4. Enforce the required topic count strictly:
   - The course outline must include a specific number of main topics as defined in the Required Topic Count. Follow the rules below carefully to stay within that limit
     a. If a single number is provided (e.g., “8”), generate exactly that number of main topics. You may allow ±1 variation if absolutely unavoidable to maintain logical structure or topic coherence
     b. If a range is provided (e.g., “5–8”), the number of main topics must strictly fall within that range. Do not exceed the upper limit or fall below the lower limit. Strictly do not add or remove topics beyond this range, even if you believe it might improve coverage or flow

5. Create the final course outline:
   - Use the course information to guide the overall structure
   - Incorporate the combined content from the partial outlines
   - Ensure the outline aligns with the course name, target audience, and tentative outline provided in the course info
   - Be detailed with the outline. Use sentences if necessary to explain exactly what is to be covered such that there is no guesswork down the line on what material to include
   - Only include points present in the partial outlines
   - DO NOT add additional points from yourself even if they are present in the tentative outline and not present in partial outlines

Before presenting the final outline, show your thought process inside <outline_synthesis> tags. In this section:
- List and summarize the main topics from each partial outline
- Compare these topics with the tentative outline
- Identify any gaps or inconsistencies
- Create a map of how topics from different outlines relate to each other
- Consider the following questions:
  - How does the best outline compare to the tentative outline?
  - What additions or modifications are necessary to create a comprehensive course outline?
  - How have you ensured that the final outline maintains a logical flow and avoids hindering elements?
  - What unique concepts from other outlines have you incorporated, and why?
  - How many main topics are expected based on the Required Topic Count, and how have you ensured the final outline stays within that range?

It's OK for this section to be quite long.

Present your final course outline using the following format:

<final_outline>
I. Main Topic 1
   A. Subtopic 1
      1. Specific point (Brief explanation if necessary)
      2. Specific point (Brief explanation if necessary)
   B. Subtopic 2
      1. Specific point (Brief explanation if necessary)
      2. Specific point (Brief explanation if necessary)

II. Main Topic 2
   A. Subtopic 1
      1. Specific point (Brief explanation if necessary)
      2. Specific point (Brief explanation if necessary)
   B. Subtopic 2
      1. Specific point (Brief explanation if necessary)
      2. Specific point (Brief explanation if necessary)

[Continue with additional main topics as needed, but remember to stay strictly within the specified required topic count.]
</final_outline>

Remember to maintain a logical flow, ensure comprehensive coverage of the course content, and align the final outline with the provided course information. Your analysis and final outline should demonstrate thorough reasoning and careful consideration of all provided materials.
"""


def run_propose_consolidated_reference_outlines(sheet, worksheet_name, course_name, target_audience, llm=None):
    """
    Propose a consolidated reference outline
    :param sheet: The sheet object.
    :param worksheet_name: The worksheet name.
    :param course_name: The course name.
    :param target_audience: The target audience.
    :param llm: The language model to use.
    :return: None
    """
    
    _, client_reference_df = get_sheet_data_and_df(sheet, 'Client References')
    outline_consolidation_sheet, outline_consolidation_df = get_sheet_data_and_df(sheet, worksheet_name)
    rough_outline_sheet, rough_outline_df = get_sheet_data_and_df(sheet = sheet, sheet_name = "Rough Outline")

    if 'Client reference based outline' in outline_consolidation_df['Source'].values:
        print("Client reference based outline already present in source column")
        return

    # Get the course outline
    course_outline = get_outline_with_los(
        df = rough_outline_df,
        include_learning_objectives = False
    )

    partial_outlines = ''
    i = 0
    for ind, row in client_reference_df.iterrows():
        partial_outlines += f"""<outline id={i} user_comments='{row['consolidation_comments']}'>
{row['outline']}
</outline>

"""
        i += 1

    partial_outlines = partial_outlines.strip()
    print(partial_outlines)

    use_agent_llm = True if llm is None else False
    proposer_responses = {}

    _, agents_df = get_sheet_data_and_df(sheet, 'Agents')
    proposer_agents, _ = get_proposer_and_aggregator_agents(agents_df)

    # Fetch topic count from Course Info
    _, course_info_df = get_sheet_data_and_df(sheet, "Course info")
    raw_count = course_info_df.loc[0, "Required Topic Count"]
    required_topic_count = str(raw_count).strip() if pd.notna(raw_count) and str(raw_count).strip() else "unspecified"

    # Function to process each agent in parallel
    def process_agent(proposer_agent):
        print(proposer_agent.name)
        agent_llm = proposer_agent.llm if use_agent_llm else llm
        print(f"LLM: {agent_llm}")
        print('.' * 50)

        consolidate_reference_outlines_agent = Chain(llm=agent_llm, tags=['final_outline'])

        consolidate_reference_outlines_agent.add_message(
            role="system",
            content=proposer_agent.description
        )

        consolidate_reference_outlines_agent.add_message(
            role="user",
            content=consolidate_reference_outlines_prompt.format(
                course_name=course_name,
                target_audience=target_audience,
                tentative_outline=course_outline,
                partial_outlines=partial_outlines,
                required_topic_count=required_topic_count  
            )
        )

        response = consolidate_reference_outlines_agent.run()
        return proposer_agent.name, response['final_outline']

    # Parallel execution of proposer agents
    with ThreadPoolExecutor(max_workers = 5) as executor:
        futures_map = {
            executor.submit(process_agent, proposer_agent): proposer_agent
            for proposer_agent in proposer_agents
        }

        # Collect the results as they complete
        total_tasks = len(futures_map)
        save_interval = 5  # how often to save (in number of completed tasks)

        # Initialize the progress tracker
        progress = SmartProgressBar(total_tasks = total_tasks, description = "Percent complete", save_interval = save_interval)

        # Now, pass only the futures (the keys) to as_completed:
        for future in tqdm(as_completed(futures_map), total=total_tasks):
            name, response = future.result()
            proposer_responses[name] = response

            # Update progress
            progress.update()

    # Construct reference outlines dictionary
    reference_outlines_dict = {'Source': 'Client reference based outline'}
    for key, value in proposer_responses.items():
        reference_outlines_dict[key] = value

    outline_consolidation_df = pd.concat([outline_consolidation_df, pd.DataFrame(reference_outlines_dict, index=[0])], ignore_index=True)
    
    save_to_sheet(worksheet = outline_consolidation_sheet, df = outline_consolidation_df)
    print('Saved to sheet')

    return

