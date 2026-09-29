import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from agents.submission_reviewer.Upload.upload import (
    build_upload_worksheet_url,
    upload_grading_worksheet_to_moodle,
)


class TestMoodleUpload(unittest.IsolatedAsyncioTestCase):

    def test_build_upload_worksheet_url(self):
        target = "https://planning.guroo.app/mod/assign/view.php?id=11922&action=grading"
        upload_url = build_upload_worksheet_url(target)

        self.assertIn("id=11922", upload_url)
        self.assertIn("plugin=offline", upload_url)
        self.assertIn("pluginsubtype=assignfeedback", upload_url)
        self.assertIn("action=viewpluginpage", upload_url)
        self.assertIn("pluginaction=uploadgrades", upload_url)

        # Dynamic ID test 1: Different URL
        url_other = "https://planning.guroo.app/mod/assign/view.php?id=45678&action=grading"
        self.assertIn("id=45678", build_upload_worksheet_url(url_other))

        # Dynamic ID test 2: Assignment folder name
        self.assertIn("id=99887", build_upload_worksheet_url("Assignment_99887_2026-09-25"))

        # Dynamic ID test 3: Raw numeric ID string
        self.assertIn("id=12345", build_upload_worksheet_url("12345"))

    async def test_upload_grading_worksheet_mocked_flow(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(b"Identifier,Grade,Feedback comments\nParticipant 1,Pass,Good work\n")
            csv_path = f.name

        try:
            # Mock Playwright Page
            page = AsyncMock()
            page.url = "https://planning.guroo.app/mod/assign/view.php?id=11922&pluginaction=uploadgrades"

            # Mock locators
            choose_btn = AsyncMock()
            choose_btn.first = choose_btn
            choose_btn.wait_for = AsyncMock()
            choose_btn.click = AsyncMock()

            modal = AsyncMock()
            modal.first = modal
            modal.wait_for = AsyncMock()

            upload_tab = AsyncMock()
            upload_tab.first = upload_tab
            upload_tab.is_visible = AsyncMock(return_value=True)
            upload_tab.click = AsyncMock()

            file_input = AsyncMock()
            file_input.first = file_input
            file_input.set_input_files = AsyncMock()

            upload_btn = AsyncMock()
            upload_btn.first = upload_btn
            upload_btn.wait_for = AsyncMock()
            upload_btn.click = AsyncMock()

            encoding_sel = AsyncMock()
            encoding_sel.first = encoding_sel
            encoding_sel.is_visible = AsyncMock(return_value=True)
            encoding_sel.select_option = AsyncMock()

            separator_radio = AsyncMock()
            separator_radio.first = separator_radio
            separator_radio.is_visible = AsyncMock(return_value=True)
            separator_radio.check = AsyncMock()

            overwrite_cb = AsyncMock()
            overwrite_cb.first = overwrite_cb
            overwrite_cb.is_visible = AsyncMock(return_value=True)
            overwrite_cb.check = AsyncMock()

            submit_btn = AsyncMock()
            submit_btn.first = submit_btn
            submit_btn.click = AsyncMock()

            content_box = AsyncMock()
            content_box.first = content_box
            content_box.is_visible = AsyncMock(return_value=True)
            content_box.inner_text = AsyncMock(return_value="Set grade for Om Aryan to Fail\nSet field Feedback comments")

            confirm_btn = AsyncMock()
            confirm_btn.first = confirm_btn
            confirm_btn.wait_for = AsyncMock()
            confirm_btn.click = AsyncMock()

            def locator_side_effect(selector, *args, **kwargs):
                mock_loc = AsyncMock()
                mock_loc.first = mock_loc
                mock_loc.wait_for = AsyncMock()
                mock_loc.click = AsyncMock()
                mock_loc.is_visible = AsyncMock(return_value=True)
                mock_loc.set_input_files = AsyncMock()
                mock_loc.select_option = AsyncMock()
                mock_loc.check = AsyncMock()
                mock_loc.inner_text = AsyncMock(return_value="Set grade for Om Aryan to Fail")

                if "Choose a file" in selector:
                    return choose_btn
                elif "filepicker" in selector or "moodle-dialogue" in selector:
                    return modal
                elif "Upload a file" in selector:
                    return upload_tab
                elif "repo_upload_file" in selector or "input[type='file']" in selector:
                    return file_input
                elif "Upload this file" in selector:
                    return upload_btn
                elif "encoding" in selector:
                    return encoding_sel
                elif "separator" in selector or "comma" in selector:
                    return separator_radio
                elif "overwrite" in selector or "allowoverwriting" in selector:
                    return overwrite_cb
                elif "Confirm" in selector:
                    return confirm_btn
                elif "Upload grading worksheet" in selector or "submitbutton" in selector:
                    return submit_btn
                return mock_loc

            page.locator = MagicMock(side_effect=locator_side_effect)

            success = await upload_grading_worksheet_to_moodle(
                page=page,
                csv_path=csv_path,
                target_url="https://planning.guroo.app/mod/assign/view.php?id=11922&action=grading",
            )

            self.assertTrue(success)
            self.assertTrue(page.goto.called)
            self.assertTrue(choose_btn.click.called)
            self.assertTrue(file_input.set_input_files.called)
            self.assertTrue(upload_btn.click.called)
            self.assertTrue(submit_btn.click.called)
            self.assertTrue(confirm_btn.click.called)

        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)


if __name__ == "__main__":
    unittest.main()
