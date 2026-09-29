import os
import unittest
from PIL import Image
from agents.submission_reviewer.schemas import (
    ChecklistResultItem,
    SubmissionReviewOutput,
)
from agents.submission_reviewer.drive_image_helper import extract_all_drive_ids
from agents.submission_reviewer.reviewer import review_single_submission
from agents.submission_reviewer.image_identifier import identify_media
from agents.submission_reviewer.sheet_processor import format_agent_checklist_text




class TestSubmissionReviewer(unittest.TestCase):

    def test_extract_drive_ids(self):
        sample_text = (
            "https://drive.google.com/file/d/1ABC_file_id123/view?usp=sharing "
            "https://drive.google.com/drive/folders/2DEF_folder_id456"
        )
        extracted = extract_all_drive_ids(sample_text)
        self.assertEqual(len(extracted), 2)
        self.assertEqual(extracted[0][0], "1ABC_file_id123")
        self.assertEqual(extracted[1][0], "2DEF_folder_id456")

    def test_missing_images_fallback(self):
        result = review_single_submission(
            activity_name="System Identification",
            activity_instructions="Identify low pressure port.",
            reviewer_checklist="1. Low pressure port visible.",
            user_comment="Here is my photo.",
            images=[]  # Empty images
        )
        self.assertEqual(result.agent_grade, "Fail (Unsure)")
        self.assertTrue("couldn't open" in result.agent_comment.lower() or "missing" in result.agent_comment.lower())

    def test_format_checklist_text(self):
        output = SubmissionReviewOutput(
            checklist_evaluations=[
                ChecklistResultItem(
                    item_id="1",
                    instruction_or_condition="Pressure gauge visible",
                    followed=True,
                    is_fail_if_condition=False,
                    comment="Gauge clearly shows 45 PSI."
                ),
                ChecklistResultItem(
                    item_id="2",
                    instruction_or_condition="Safety cap installed",
                    followed=False,
                    is_fail_if_condition=True,
                    comment="Safety cap is missing from valve."
                ),
            ],
            agent_comment="You showed the pressure gauge fine, but forgot the safety cap.",
            agent_grade="Fail"
        )
        formatted = format_agent_checklist_text(output)
        self.assertIn("[PASSED] Pressure gauge visible", formatted)
        self.assertIn("[FAIL-IF TRIGGERED] Safety cap installed", formatted)


    def test_synthetic_image_evaluation(self):
        # Create a simple red test image
        img = Image.new("RGB", (100, 100), color="red")
        result = review_single_submission(
            activity_name="Red Patch Activity",
            activity_instructions="Upload a red image patch.",
            reviewer_checklist="1. Image must contain red color.\n2. Fail if image is green.",
            user_comment="Attached my test patch.",
            images=[img]
        )
        self.assertIn(result.agent_grade, ["Pass", "Fail", "Pass (Unsure)", "Fail (Unsure)"])
        self.assertIsNotNone(result.agent_comment)
        self.assertTrue(len(result.agent_comment) > 5)

    def test_identify_media_no_images(self):
        res = identify_media("Do you identify this? and if yes give information about it.", images=[])
        self.assertIn("No images provided", res)

    def test_activity_edge_cases_parameter(self):
        img = Image.new("RGB", (100, 100), color="blue")
        result = review_single_submission(
            activity_name="Portable Unit Check",
            activity_instructions="Upload a photo of your split system unit.",
            activity_edge_cases="Photo of a portable unit is acceptable",
            user_comment="Here is my portable unit.",
            images=[img]
        )
        self.assertIn(result.agent_grade, ["Pass", "Fail", "Pass (Unsure)", "Fail (Unsure)"])
        self.assertIsNotNone(result.agent_comment)


if __name__ == "__main__":
    unittest.main()


