import streamlit as st
import pandas as pd
from modules.chain import Chain
from services.sheets_service import get_sheet_data_and_df, save_to_sheet
import difflib
from services.helper_functions import get_outline_with_los
from services.smart_progress_bar import SmartProgressBar


review_course_outline_prompt = """You are an experienced instructional designer tasked with reviewing and improving a course outline. Your goal is to provide a comprehensive analysis of the outline, identifying any issues and offering suggestions for improvement.

First, review the following information about the course:

Course Name:
<course_name>
{course_name}
</course_name>

Target Audience:
<target_audience>
{target_audience}
</target_audience>

User Comments (made during the course outline generation phase):
<user_comments>
{user_comments}
</user_comments>

Tentative Outline (used to generate the course outline):
<tentative_outline>
{tentative_outline}
</tentative_outline>

Now, carefully study the full course outline:

<course_outline>
{course_outline}
</course_outline>

Here's the freshly added user feedback on the course outline:
<user_feedback>
{user_feedback}
</user_feedback>

Your task is to analyze this course outline thoroughly. Follow these steps:

1. Summarize your understanding of the course requirements based on the provided information.

2. For each major section of the course outline:
   a. Assess its relevance to the course objectives and target audience.
   b. Evaluate the logical flow and progression of topics.
   c. Check for any gaps in content or redundant information.
   d. Consider the depth and breadth of coverage for each topic.
   e. Provide a brief assessment of its strengths and weaknesses.
   f. If necessary, offer specific suggestions for improvement.

3. After analyzing all sections, provide an overall assessment of the course outline, including its structure, comprehensiveness, and alignment with the course objectives.

4. List key recommendations for improving the overall course outline.

As you review, consider these questions:
- Does the outline align with the course name and target audience?
- Are the sections and subsections well-organized and coherent?
- Is there a good balance between theoretical concepts and practical applications?
- Does the outline address all key points mentioned in the tentative outline?
- Have the user comments been adequately addressed in the outline?

Refrain from giving generic suggestions such as:
- Add more practical examples and basic troubleshooting scenarios.
- Include more visual aids and simple demonstrations to help explain complex concepts
- Add hands-on activities or demonstrations where appropriate
- Include knowledge check points throughout the course
- Consider adding a glossary of basic terms
-Develop clear learning objectives for each main section

All of the above tasks are to be done while keeping the user feedback on the course outline in your mind.

Present your final review in the following format:

<course_outline_review>
<course_requirements>
[Present your understanding of the course objectives and requirements]
</course_requirements>

<overall_assessment>
[Provide a summary of your overall assessment of the course outline]
</overall_assessment>

<section_analysis>
<section_name>[Name of the section]</section_name>
<section_summary>[Summary of what's currently included in this section</section_summary>
<assessment>
[Your assessment of the section. It should include analysis of all the points mentioned i.e. relevance, logical flow, gaps / redundant info, coverage depth & breadth, strengths & weaknesses. It is okay for this section to be quite long.]
</assessment>
<suggestions>[Your suggestions for improvement, if any (optional)]</suggestions>
</section_analysis>

[Repeat the section_analysis for each major section]

<final_recommendations>
[Provide a list of key recommendations for improving the overall course outline. Be clear and verbose with the list to avoid any confusion / guesswork]
</final_recommendations>
<verdict>
["APPROVED" or "REJECTED".]
</verdict>
</course_outline_review>

Remember to be constructive in your feedback. Your goal is to help improve the course outline to better serve the target audience and meet the course objectives.

NOTE: DO NOT stop the analysis midway / skip through sections.
NOTE: You should output the entire analysis without worrying about output token limits.
NOTE: DO NOT ask questions such as "Due to the comprehensive nature of the review, I'll continue in subsequent responses. Would you like me to proceed with the next sections?".
NOTE: Instead continue to output without worrying about the output token limits.
"""


def compare_text_versions(text1: str, text2: str):
    """
    Compare two versions of text and display them side by side in Streamlit
    with color-coded highlights for changes.

    :param text1: The first text version.
    :param text2: The second text version.
    """
    # Split text into lines
    lines1 = text1.splitlines()
    lines2 = text2.splitlines()

    # Use difflib to get differences
    diff = list(difflib.ndiff(lines1, lines2))

    # Separate differences for each version
    version1_lines = []
    version2_lines = []

    for line in diff:
        if line.startswith("- "):  # Removed from version 1
            version1_lines.append(f'<span style="background-color: #f7b6b6;">{line[2:]}</span>')  # Red
            version2_lines.append("")  # Keep space for alignment
        elif line.startswith("+ "):  # Added in version 2
            version1_lines.append("")  # Keep space for alignment
            version2_lines.append(f'<span style="background-color: #b6f7b6;">{line[2:]}</span>')  # Green
        elif line.startswith("? "):  # Changed (used to indicate modifications in words)
            pass  # Ignore helper lines
        else:  # Unchanged
            version1_lines.append(line[2:])
            version2_lines.append(line[2:])

    # Streamlit UI with two columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Version 1", unsafe_allow_html=True)
        st.markdown("<br>".join(version1_lines), unsafe_allow_html=True)

    with col2:
        st.markdown("### Version 2", unsafe_allow_html=True)
        st.markdown("<br>".join(version2_lines), unsafe_allow_html=True)


def run_review_and_revise_outline(sheet, course_name, target_audience, llm='gemini_2_flash'):
    """
    Runs the review and revision process for course outlines using AI.

    :param sheet: The Google Sheets object.
    :param course_name: Name of the course.
    :param target_audience: Target audience for the course.
    :param llm: The language model to use (default: 'gemini_2_flash').
    :return: None
    """

    st.write(f"Note: Populate the Verdict and the Manual Feedback columns before running this cell.\n")
    st.write("Note: The automation will only read the last (populated) row in the Outline Review sheet. Thus make sure to put your inputs in the last (populated) row.\n")
    st.write("Note: You can run this cell many number of times until you are satifised with the results.\n")
    st.write("Note: Scroll down the output of this cell till the very end to see the outline difference i.e the change between the last two outlines.")

    outline_review_sheet, outline_review_df = get_sheet_data_and_df(sheet, 'Outline Review')

    # Initiate the agent
    review_revise_agent = Chain(llm=llm)

    # Check if already approved
    last_verdict = outline_review_df.iloc[-1]['Verdict'].strip()
    if 'approved' in last_verdict.lower():
        print("Course Outline Approved")
        return

    # Get the last row's manual feedback
    last_manual_feedback = outline_review_df.iloc[-1]['Manual Feedback'].strip()

    # Raise an exception if the user has not provided manual feedback
    if not last_manual_feedback:
        raise Exception("Error: 'Manual Feedback' column is empty in the last row. Please add your feedback before continuing.")

    # # Pause execution until the user confirms
    # if st.button("Press to continue"):
    #     st.write("-" * 100)

    # Initialize the progress tracker
    progress = SmartProgressBar(total_tasks = 2, description = "Percent complete")
    
    revise_outline_prompt = """Output the revised course outline within <revised_outline> tags."""


    _, rough_outline_df = get_sheet_data_and_df(sheet, 'Rough Outline')
    
    course_outline = get_outline_with_los(df = rough_outline_df, include_learning_objectives = False, include_prefix = False)
    # Collect User Comments
    course_objective_guidelines = '\n'.join(rough_outline_df['Course Objective Guidelines']).strip()
    
    # print(course_objective_guidelines)
    
    _, client_reference_df = get_sheet_data_and_df(sheet, 'Client References')
    client_reference_comments = '\n'.join(comment for comment in client_reference_df['consolidation_comments'].to_list() if comment)
    client_comments = '\n'.join(comment for comment in client_reference_df['Client comments'].to_list() if comment)

    _, videos_research_df = get_sheet_data_and_df(sheet, 'Videos Research')
    videos_research_comments = '\n'.join(comment for comment in videos_research_df['consolidation_comments'].to_list() if comment)

    all_user_comments = f"""Comments made by user on course objective guidelines:
{course_objective_guidelines}

---

Comments made by client while sharing references:
{client_comments}

---

Comments made by user on client references:
{client_reference_comments}

---

Comments made by user on videos research:
{videos_research_comments}
"""

    print(all_user_comments)
    
    # Create the list of messages
    messages = []
    for ind, row in outline_review_df.iterrows():
        # Get the values
        outline = row['Outline']
        ai_suggestions = row['AI Suggestions']
        manual_feedback = row['Manual Feedback']

        # Add the initial prompt message
        if ind == 0:
            messages.append(
                (
                    "user",
                    review_course_outline_prompt.format(
                        course_name=course_name,
                        target_audience=target_audience,
                        user_comments=all_user_comments,
                        tentative_outline=course_outline,
                        course_outline=outline,
                        user_feedback=manual_feedback,
                    )
                )
            )
        else:  # AI revised outline for later rows
            messages.append(("ai", f"<revised_outline>\n{outline}\n</revised_outline>"))
            messages.append(
                (
                    "user",
                    f"Review the above outline in a similar fashion as done previously. Here is the latest user feedback on the above outline:\n{manual_feedback}.\nRemember to output within the <course_outline_review> tags in the same format."
                )
            )

        # Stop at the last row
        if ind + 1 == outline_review_df.shape[0]:
            break

        messages.append(("ai", f"<course_outline_review>\n{ai_suggestions}\n</course_outline_review>"))
        messages.append(("user", revise_outline_prompt))

    # Add messages to the agent
    review_revise_agent.add_messages(messages)

    # Run the agent to review
    review_revise_agent.tags = ['course_outline_review']
    reviewer_response = review_revise_agent.run()

    # Update progress
    progress.update()

    # Add AI review suggestions to df
    outline_review_df.loc[outline_review_df.shape[0] - 1, 'AI Suggestions'] = reviewer_response['course_outline_review']

    # Add revise message user prompt
    review_revise_agent.add_message(role="user", content=revise_outline_prompt)

    # Run the agent to revise
    review_revise_agent.tags = ['revised_outline']
    reviser_response = review_revise_agent.run()

    # Update progress
    progress.update()

    # Add revision as a new row
    outline_review_df = pd.concat([outline_review_df, pd.DataFrame([{
        'Turn': int(outline_review_df.iloc[-1]['Turn']) + 1,
        'Outline': reviser_response['revised_outline'],
        'Verdict': '',
        'AI Suggestions': '',
        'Manual Feedback': ''
    }])])

    # Save to sheet
    save_to_sheet(worksheet = outline_review_sheet, df = outline_review_df)

    # Print the diff
    st.write("### Comparing the Two Most Recent Versions: ")
    compare_text_versions(
        outline_review_df.iloc[-2]['Outline'],
        outline_review_df.iloc[-1]['Outline']
    )

    raise Exception("You got this error since the outline is not Approved. If the outline looks good to you, enter Approved in the `Verdict` column last row. If the outline doesn't look good, you can enter Rejected in the `Verdict` column and enter your Feedback in the `Manual Feedback` column and run the agent again to generate a new outline.")
    # return outline_review_df

