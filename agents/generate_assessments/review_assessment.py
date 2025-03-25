from modules.chain import Chain
import pandas as pd
from services.sheets_service import get_sheet_data_and_df
# from agents.generate_assessments.generate_assessment_questions import run_generate_assessment_question



review_assessment_prompt = """As an expert Review Agent, provide a review of the following assessment question for the course titled {course_name}, tailored to the {target_audience}. Below is the slide content on which the assessment question is based on:

<topic_slides>
{topic_slides}
</topic_slides>

Here is the assessment question you need to review:

<assessment_question>
{assessment_question}
</assessment_question>

Review the assessment question based on the following checklist criteria:

<checklist_criteria>
{checklist_criteria}
</checklist_criteria>

Conduct your review of the assessment question based on the checklist criteria. Give your evaluation of the assessment question against each of the checklist criterion. Based on your evaluation, assign a verdict of "PASS" or "FAIL". If your verdict is "PASS", give  feedback stating "No feedback". If your verdict is "FAIL", give reasons for this negative verdict and how the assessment question can be fixed. Ensure that you do not rewrite the assessment question; instead, focus on providing constructive feedback.

Make sure to reply in the following output format:
<output>

<question_text>
[Enter the entire question content here]
</question_text>

<evaluations>

<evaluation>
<item_name>
[The exact content of the checklist item being evaluated]
</item_name>
<analysis>
[Your evaluation of the assessment question against this criterion]
</analysis>
<verdict>
["PASS" or "FAIL" based on evaluation]
</verdict>
<feedback_summary>
[Give summary of the feedback. If the verdict is 'PASS', it should be 'No feedback'. If the verdict is 'FAIL', give reasons for this negative verdict]
</feedback_summary>
<improvement_suggestions>
[Give suggestions for improvement only if the verdict is 'FAIL'. If verdict is 'PASS', this should strictly be left empty]
</improvement_suggestions>
</evaluation>

[Repeat the evaluation block for each checklist item in the same format.]

<evaluations>

</output>
"""

# Function to review assessment question
def review_assessment(course_name, target_audience, assessment_question, slides, checklist_criteria, llm = 'gemini_2_flash'):
  """
  This function reviews an assessment question based on a checklist criterion.
  :param course_name: The name of the course.
  :param target_audience: The target audience for the course.
  :param assessment_question: The assessment question to review.
  :param slides: The slides for the topic.
  :param checklist_criteria: The checklist criteria to evaluate the assessment question.
  :param llm: The language model to use.
  :return: The reviewer response.
  """
  review_assessment_agent = Chain(llm = llm, tags = ['evaluation'])

  review_assessment_agent.add_message(
      role = "user",
      content = review_assessment_prompt.format(
          course_name = course_name,
          target_audience = target_audience,
          topic_slides = slides,
          assessment_question = assessment_question,
          checklist_criteria = '\n'.join(checklist_criteria)
      )
  )

  reviewer_response = review_assessment_agent.run()

  return reviewer_response

# def run_review_assessment(sheet, worksheet_name):
#     """
#     This function reviews an assessment question based on a checklist criterion.
#     :param sheet: The Google Sheet object.
#     :param worksheet_name: The name of the worksheet.
#     :return: The reviewer response.
#     """
#     _, slide_chunks_df = get_sheet_data_and_df(sheet, worksheet_name)
#     unique_topics = pd.unique(slide_chunks_df['Topic'])
#     questions_by_topic = run_generate_assessment_question(sheet, 'Slide Chunks', 'gemini_2_flash')
#     topic_slides_data = slide_chunks_df[slide_chunks_df['Topic'] == unique_topics[0]]
    
#     assessment_question = assessment_question = questions_by_topic[unique_topics[0]][0]
    
#     slides = "\n---\n".join(
#     "Topic Name: " + topic_slides_data['Slide Title'] + "\n" + "Slide Content: " + topic_slides_data['Slide Content']
# )

    
#     reviewer_response = review_assessment(
#         assessment_question = assessment_question,
#         slides = slides,
#         checklist_criteria = topic_slides_data['Review Criteria'].to_list(),
#         llm = 'gemini_2_flash')
    
#     question_feedback = []

#     # Loop through the evaluations in the response to check if any feedback is present
#     for evaluation in reviewer_response['evaluation']:
#         # Extract items from evaluation text as dict
#         evaluation_dict = extract_text_in_tags(
#             tags = ['item_name', 'analysis', 'verdict', 'feedback_summary', 'improvement_suggestions'],
#             text = evaluation
#         )

#         # Check the verdict
#         if 'pass' not in evaluation_dict['verdict'].lower():
#             # Append to list
#             question_feedback.append(f"Feedback summary: {evaluation_dict['feedback_summary']}\nImprovement suggestions: {evaluation_dict['improvement_suggestions']}")
    
#     return reviewer_response, question_feedback
