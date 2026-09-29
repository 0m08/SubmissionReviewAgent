import csv
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from agents.submission_reviewer.moodle_review_runner import (
    extract_google_doc_id,
    find_student_folder,
    load_student_submission_assets,
    evaluate_and_update_worksheet,
)
from agents.submission_reviewer.schemas import SubmissionReviewOutput, ChecklistResultItem


class TestMoodleReviewRunner(unittest.TestCase):

    def test_extract_google_doc_id(self):
        url1 = "https://docs.google.com/document/d/1whICRKo83X7djnfZOwTe9QoRxGmfrsKV_KnBHv-1Na8/edit?usp=drive_link"
        url2 = "https://drive.google.com/file/d/1_XTnq0HmCfLFNjJy_tUzj9a6-S5uEbaNGS0SupXDaeY/view"
        raw_id = "1whICRKo83X7djnfZOwTe9QoRxGmfrsKV_KnBHv-1Na8"

        self.assertEqual(extract_google_doc_id(url1), "1whICRKo83X7djnfZOwTe9QoRxGmfrsKV_KnBHv-1Na8")
        self.assertEqual(extract_google_doc_id(url2), "1_XTnq0HmCfLFNjJy_tUzj9a6-S5uEbaNGS0SupXDaeY")
        self.assertEqual(extract_google_doc_id(raw_id), "1whICRKo83X7djnfZOwTe9QoRxGmfrsKV_KnBHv-1Na8")

    def test_find_student_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            submissions_dir = os.path.join(tmpdir, "Submissions")
            os.makedirs(submissions_dir, exist_ok=True)

            student_folder = os.path.join(submissions_dir, "Om Aryan_Participant_2788085")
            os.makedirs(student_folder, exist_ok=True)

            # Test exact match
            found = find_student_folder(submissions_dir, "Om Aryan", "Participant 2788085")
            self.assertEqual(found, student_folder)

            # Test missing student
            missing = find_student_folder(submissions_dir, "Unknown Student", "Participant 9999999")
            self.assertIsNone(missing)

    def test_load_student_submission_assets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            text_file = os.path.join(tmpdir, "online_text.txt")
            with open(text_file, "w", encoding="utf-8") as tf:
                tf.write("Student comment here")

            # Create dummy image
            from PIL import Image
            img_file = os.path.join(tmpdir, "photo.jpg")
            img = Image.new("RGB", (50, 50), color="green")
            img.save(img_file)

            images, videos, online_text = load_student_submission_assets(tmpdir)
            self.assertEqual(len(images), 1)
            self.assertEqual(len(videos), 0)
            self.assertEqual(online_text, "Student comment here")

    @patch("agents.submission_reviewer.moodle_review_runner.review_single_submission")
    def test_evaluate_and_update_worksheet_strict_columns(self, mock_review):
        mock_review.return_value = SubmissionReviewOutput(
            checklist_evaluations=[
                ChecklistResultItem(
                    item_id="1",
                    instruction_or_condition="Check photo",
                    followed=True,
                    is_fail_if_condition=False,
                    comment="Looks great."
                )
            ],
            agent_comment="Awesome work on identifying the unit!",
            agent_grade="Pass"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "grading_worksheet.csv")
            submissions_dir = os.path.join(tmpdir, "Submissions")
            os.makedirs(submissions_dir, exist_ok=True)

            student_folder = os.path.join(submissions_dir, "Jane Doe_Participant_1001")
            os.makedirs(student_folder, exist_ok=True)

            # Create folder for Tiffany Parker (Reopened, should still be skipped!)
            reopened_folder = os.path.join(submissions_dir, "Tiffany Parker_Participant_1003")
            os.makedirs(reopened_folder, exist_ok=True)
            with open(os.path.join(reopened_folder, "test.txt"), "w") as f:
                f.write("old submission file")

            # Create standard Moodle 11-column CSV
            fieldnames = [
                "Identifier", "Full name", "Email address", "Status", "Grade", "Scale",
                "Grade can be changed", "Last modified (submission)", "Online text",
                "Last modified (grade)", "Feedback comments"
            ]
            rows = [
                {
                    "Identifier": "Participant 1001",
                    "Full name": "Jane Doe",
                    "Email address": "jane@example.com",
                    "Status": "Submitted for grading",
                    "Grade": "",
                    "Scale": "Fail   Pass",
                    "Grade can be changed": "Yes",
                    "Last modified (submission)": "Monday, 1 Jan",
                    "Online text": "Here is my work",
                    "Last modified (grade)": "-",
                    "Feedback comments": ""
                },
                {
                    "Identifier": "Participant 1002",
                    "Full name": "Bob Smith",
                    "Email address": "bob@example.com",
                    "Status": "No submission",
                    "Grade": "",
                    "Scale": "Fail   Pass",
                    "Grade can be changed": "Yes",
                    "Last modified (submission)": "-",
                    "Online text": "",
                    "Last modified (grade)": "-",
                    "Feedback comments": ""
                },
                {
                    "Identifier": "Participant 1003",
                    "Full name": "Tiffany Parker",
                    "Email address": "tiffany@example.com",
                    "Status": "Reopened",
                    "Grade": "",
                    "Scale": "Fail   Pass",
                    "Grade can be changed": "Yes",
                    "Last modified (submission)": "Tuesday, 2 Jan",
                    "Online text": "Old text",
                    "Last modified (grade)": "-",
                    "Feedback comments": "Old comment"
                }
            ]

            with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            # Run evaluation
            summary = evaluate_and_update_worksheet(
                assignment_folder=tmpdir,
                instructions_text="Test instructions",
                guardrails_text="Test guardrails",
            )

            self.assertEqual(summary["evaluated_count"], 1)

            # Re-read CSV and verify strict writeback
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                updated_fields = reader.fieldnames
                updated_rows = list(reader)

            # Assert exact fields preserved (NO checklist columns added)
            self.assertEqual(updated_fields, fieldnames)

            # Assert Jane Doe updated
            jane = updated_rows[0]
            self.assertEqual(jane["Grade"], "Pass")
            self.assertEqual(jane["Feedback comments"], "Awesome work on identifying the unit!")
            self.assertEqual(jane["Status"], "Submitted for grading")
            self.assertEqual(jane["Scale"], "Fail   Pass")
            self.assertEqual(jane["Online text"], "Here is my work")

            # Assert Bob Smith untouched (No submission)
            bob = updated_rows[1]
            self.assertEqual(bob["Grade"], "")
            self.assertEqual(bob["Feedback comments"], "")
            self.assertEqual(bob["Status"], "No submission")

            # Assert Tiffany Parker untouched (Reopened)
            tiffany = updated_rows[2]
            self.assertEqual(tiffany["Grade"], "")
            self.assertEqual(tiffany["Feedback comments"], "Old comment")
            self.assertEqual(tiffany["Status"], "Reopened")


if __name__ == "__main__":
    unittest.main()
