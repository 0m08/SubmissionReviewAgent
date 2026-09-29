import email
import os
import tempfile
import pytest
from email.message import EmailMessage

from agents.submission_reviewer.Ingestion.ingestion import (
    MoodleConfig,
    parse_email_for_otp,
    map_worksheet_to_submissions,
)


def test_parse_email_for_otp_plain_text():
    msg = EmailMessage()
    msg["Subject"] = "Your Verification Code"
    msg["From"] = "noreply@guroo.app"
    msg["To"] = "student@skillcatlabs.com"
    msg.set_content("Hello,\nYour verification code is 849201. Please use this within 10 minutes.")

    code, dt = parse_email_for_otp(msg)
    assert code == "849201"


def test_parse_email_for_otp_html():
    msg = EmailMessage()
    msg["Subject"] = "Moodle Security Verification"
    msg["From"] = "noreply@guroo.app"
    msg["To"] = "student@skillcatlabs.com"
    msg.set_content("Fallback text")
    msg.add_alternative(
        "<html><body><p>Your one-time password is: <b>592104</b></p></body></html>",
        subtype="html",
    )

    code, dt = parse_email_for_otp(msg)
    assert code == "592104"


def test_moodle_config_validation():
    cfg = MoodleConfig(moodle_user=None, moodle_pass=None)
    with pytest.raises(ValueError, match="Missing required Moodle credential"):
        cfg.validate()

    valid_cfg = MoodleConfig(moodle_user="test_user", moodle_pass="test_pass")
    valid_cfg.validate()  # Should not raise
    assert valid_cfg.upload_to_drive is True


def test_map_worksheet_to_submissions():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = os.path.join(tmpdir, "grading_worksheet.csv")
        extract_dir = os.path.join(tmpdir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)

        # Create dummy extracted files
        student1_file = os.path.join(extract_dir, "Alice Smith_101_assignsubmission_file_photo.jpg")
        student2_file = os.path.join(extract_dir, "Bob Jones_102_assignsubmission_file_doc.pdf")
        with open(student1_file, "w") as f:
            f.write("dummy")
        with open(student2_file, "w") as f:
            f.write("dummy")

        # Create dummy CSV
        csv_content = (
            "Identifier,Full name,Email address,Status,Grade\n"
            "Participant 101,Alice Smith,alice@example.com,Submitted for grading,\n"
            "Participant 102,Bob Jones,bob@example.com,Submitted for grading,\n"
            "Participant 103,Charlie Brown,charlie@example.com,No submission,\n"
        )
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write(csv_content)

        records = map_worksheet_to_submissions(csv_file, extract_dir)
        assert len(records) == 3
        
        alice = next(r for r in records if r.full_name == "Alice Smith")
        assert len(alice.submission_files) == 1
        assert "photo.jpg" in alice.submission_files[0]

        charlie = next(r for r in records if r.full_name == "Charlie Brown")
        assert len(charlie.submission_files) == 0
