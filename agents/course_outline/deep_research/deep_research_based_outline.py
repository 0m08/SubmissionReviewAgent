from modules.proposer_agents import get_proposer_and_aggregator_agents
from modules.chain import Chain
from services.helper_functions import get_outline_with_los
from services.sheets_service import get_sheet_data_and_df, save_to_sheet
import pandas as pd
from services.smart_progress_bar import SmartProgressBar
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


generate_deep_research_outline_prompt = """You are tasked with creating a comprehensive course outline for an HVAC technician training program. This is a crucial task as the outline will serve as the foundation for the entire course, ensuring that all necessary topics are covered in a logical and learner-friendly sequence.

Here's the information you'll be working with:

1. Course Name:
<course_name>
{course_name}
</course_name>

2. Course Background Information:
<course_background>
{course_background}
</course_background>

3. User Guidelines:
<user_guidelines>
{user_guidelines}
</user_guidelines>

4. Rough Outline (concepts to cover):
<rough_outline>
{rough_outline}
</rough_outline>

5. Preliminary Research:
<preliminary_research>
{preliminary_research}
</preliminary_research>

Your task is to analyze and synthesize this information to create a well-structured course outline. Follow these steps:

1. Carefully review all the provided information.
2. Identify the key topics that need to be covered based on the rough outline and preliminary research.
3. Organize these topics into a logical sequence, considering the learning progression for entry-level HVAC technicians.
4. Break down main topics into subtopics, ensuring each section is self-contained and can be studied independently if needed.
5. Ensure the outline adheres to the user guidelines and aligns with the course background information.
6. Be detailed with the outline. Use sentences if necessary to explain exactly what is to be covered such that there is no guesswork down the line on what material to include

Before creating the final outline, use the <scratchpad> tags to plan your approach. Consider the following:
- How can you structure the outline to make it easy for learners to follow?
- What is the most logical sequence for the topics?
- How can you ensure each topic is self-contained yet connected to the overall course flow?
- Are there any topics from the rough outline or preliminary research that need to be expanded or condensed?

After your planning, create the course outline. Use the following format for your outline:

<course_outline>
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

[Continue with additional main topics as needed]
</course_outline>

Ensure that your outline is comprehensive, well-structured, and tailored to the needs of entry-level HVAC technicians. The topics should progress logically and build upon each other where appropriate.
"""


def process_agent_outline(proposer_agent, course_name, course_background, concepts_to_include, research_summary, course_objective_guidelines, llm=None):
    """
    Process a single agent's outline generation task
    
    :param proposer_agent: The agent to use for generating the outline
    :param course_name: Name of the course
    :param course_background: Background information for the course
    :param concepts_to_include: Concepts to include in the outline
    :param research_summary: Research summary from Deep Research
    :param course_objective_guidelines: Guidelines for course objectives
    :param llm: Language model to use (if None, use agent's llm)
    :return: Tuple of (agent_name, outline)
    """
    print(proposer_agent.name)
    if llm is None:
        llm = proposer_agent.llm  # If no llm is provided, use the llm of the agent
    print(f"LLM: {llm}")
    print('.'*50)

    generate_outline_agent = Chain(llm=llm, tags=["course_outline"])
            
    generate_outline_agent.add_message(
        role="system",
        content=proposer_agent.description
    )

    generate_outline_agent.add_messages(
        [
            ("user", generate_deep_research_outline_prompt.format(
                course_name=course_name,
                course_background=course_background,
                user_guidelines=course_objective_guidelines,
                rough_outline=concepts_to_include,
                preliminary_research=research_summary
            ))
        ]
    )

    response = generate_outline_agent.run()
    outline = response['course_outline']
    print('='*50)
    
    return proposer_agent.name, outline


def propose_web_research_outline_with_agents(sheet, course_name, course_background, llm=None):
    """
    Generate the course outline based on deep research
    :param sheet: The Google Sheets object
    :param course_name: Name of the course
    :param course_background: Background information for the course
    :param llm: llm model to use
    :return: proposer_responses: dictionary of agent responses
    """
    use_agent_llm = True if llm is None else False
    proposer_responses = {}
    
    _, rough_outline_df = get_sheet_data_and_df(sheet, 'Rough Outline')
    _, deep_research_df = get_sheet_data_and_df(sheet, 'Deep Research')  # Get the Deep Research DataFrame
    _, agents_df = get_sheet_data_and_df(sheet, 'Agents')  # Get only the DataFrame

    proposer_agents, _ = get_proposer_and_aggregator_agents(agents_df)  # Unpack proposer_agents correctly


    concepts_to_include = get_outline_with_los(df=rough_outline_df, include_learning_objectives=False, include_prefix=False)
    
    # Use the research column from Deep Research sheet
    research_summary = '\n\n---\n\n'.join(deep_research_df['research'].to_list())
    course_objective_guidelines = '\n'.join(rough_outline_df['Course Objective Guidelines']).strip()

    # Prepare for parallel processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit tasks for each agent
        for proposer_agent in proposer_agents:
            # Set the LLM to use
            agent_llm = proposer_agent.llm if use_agent_llm else llm
            
            # Submit the task
            future = executor.submit(
                process_agent_outline,
                proposer_agent,
                course_name,
                course_background,
                concepts_to_include,
                research_summary,
                course_objective_guidelines,
                agent_llm
            )
            
            # Map the Future to the agent name
            futures_map[future] = proposer_agent.name
        
        # Collect the results as they complete
        total_tasks = len(futures_map)
        
        # Initialize the progress tracker
        progress = SmartProgressBar(total_tasks=total_tasks, description="Percent complete")
        
        # Process results as they complete
        for future in tqdm(as_completed(futures_map), total=total_tasks):
            agent_name, outline = future.result()
            
            # Store the result
            proposer_responses[agent_name] = outline
            
            # Update progress
            progress.update()

    return proposer_responses


def run_generate_deep_research_outline(sheet, worksheet_name, course_name, course_background, llm='gemini_2_flash'):
    """
    Consolidate all outlines and review/revise course outlines based on deep research.

    :param sheet: The Google Sheets object.
    :param worksheet_name: The name of the worksheet containing outlines.
    :param course_name: Name of the course.
    :param course_background: Background information for the course.
    :param llm: The language model to use (default: 'gemini_2_flash').
    :return: Updated consolidated outlines and outline review DataFrame.
    """

    outline_consolidation_sheet, outline_consolidation_df = get_sheet_data_and_df(sheet, worksheet_name)

    # Check if 'Deep research based outline' is already present
    if 'Deep research based outline' in outline_consolidation_df['Source'].values:
        print("Deep research based outline already present in source column")
        return

    proposer_responses = propose_web_research_outline_with_agents(
        sheet=sheet,
        course_name=course_name,
        course_background=course_background,
        llm=llm
    )
    
    deep_research_outlines_dict = {}
    deep_research_outlines_dict['Source'] = 'Deep research based outline'
    for key, value in proposer_responses.items():        
        deep_research_outlines_dict[key] = value
    
    outline_consolidation_df = pd.concat([
        outline_consolidation_df, 
        pd.DataFrame(deep_research_outlines_dict, index=[3])
    ], ignore_index=True)
    
    # Save to sheet
    print('Saved to sheet')
    save_to_sheet(worksheet=outline_consolidation_sheet, df=outline_consolidation_df)
    
    return
