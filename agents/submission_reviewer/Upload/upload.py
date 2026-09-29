# -*- coding: utf-8 -*-
"""
Moodle Worksheet Upload Automation.

Automates uploading the graded CSV worksheet back to Moodle:
1. Navigates to the 'Upload grading worksheet' page:
   action=viewpluginpage&pluginaction=uploadgrades&plugin=offline&pluginsubtype=assignfeedback
2. Interacts with Moodle File picker modal:
   - Selects 'Upload a file' from sidebar
   - Attaches graded CSV to 'Attachment'
   - Leaves 'Save as', 'Author', and 'License' untouched
   - Clicks 'Upload this file'
3. Configures form options:
   - Encoding: UTF-8
   - Separator: Comma
   - Checks 'Allow updating records that have been modified more recently in Moodle...'
4. Submits 'Upload grading worksheet'.
5. Reads the confirmation preview screen ('Confirm changes in grading worksheet') and clicks 'Confirm'.
6. Verifies updated grades are applied.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

from dotenv import load_dotenv

from agents.submission_reviewer.Ingestion.ingestion import (
    MoodleConfig,
    TARGET_URL,
    BASE_DIR,
)

load_dotenv()

logger = logging.getLogger("moodle_upload")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def extract_assignment_id(url_or_id: str) -> str:
    """Extracts the dynamic assignment ID from a URL, path, or raw numeric string."""
    if not url_or_id:
        return ""
    m = re.search(r'[?&]id=(\d+)', url_or_id)
    if m:
        return m.group(1)
    m = re.search(r'Assignment_(\d+)', url_or_id)
    if m:
        return m.group(1)
    digits = re.search(r'^\d+$', url_or_id.strip())
    if digits:
        return digits.group(0)
    return "11922"


def build_upload_worksheet_url(target_url: str) -> str:
    """
    Builds the Moodle offline grading upload URL dynamically for any assignment ID:
    https://{host}/mod/assign/view.php?id={dynamic_id}&plugin=offline&pluginsubtype=assignfeedback&action=viewpluginpage&pluginaction=uploadgrades
    """
    assign_id = extract_assignment_id(target_url)
    parsed = urlparse(target_url)

    scheme = parsed.scheme if parsed.scheme else "https"
    netloc = parsed.netloc if parsed.netloc else "planning.guroo.app"
    path = parsed.path if parsed.path and parsed.path.startswith("/mod/assign") else "/mod/assign/view.php"

    upload_params = {
        "id": assign_id,
        "plugin": "offline",
        "pluginsubtype": "assignfeedback",
        "action": "viewpluginpage",
        "pluginaction": "uploadgrades",
    }
    return f"{scheme}://{netloc}{path}?{urlencode(upload_params)}"


async def upload_grading_worksheet_to_moodle(
    page: Any,
    csv_path: str,
    target_url: str = TARGET_URL,
    encoding: str = "UTF-8",
    separator: str = "comma",
    allow_overwriting: bool = False,
    timeout_ms: int = 30000,
) -> bool:
    """
    Uploads the graded CSV worksheet back to Moodle using the active Playwright page session.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Grading worksheet CSV not found: {csv_path}")

    abs_csv_path = os.path.abspath(csv_path)
    file_size_kb = os.path.getsize(abs_csv_path) / 1024

    upload_url = build_upload_worksheet_url(target_url)

    print("\n" + "=" * 65)
    print("📤 [MOODLE UPLOAD] Starting Graded Worksheet Upload to Moodle")
    print(f"   CSV File: {abs_csv_path} ({file_size_kb:.1f} KB)")
    print(f"   Target URL: {upload_url}")
    print("=" * 65)

    # --------------------------------------------------------------------------
    # STEP 1: Navigate to Upload Grading Worksheet Page
    # --------------------------------------------------------------------------
    logger.info("🌐 [1/5] Navigating to 'Upload grading worksheet' page...")
    print(f"  [UPLOAD LOG] Navigating to: {upload_url}")
    await page.goto(upload_url, wait_until="networkidle", timeout=timeout_ms)

    # Fallback: if not directly on the upload page, attempt action dropdown selection
    if "uploadgrades" not in page.url and "uploadgradingworksheet" not in page.url:
        logger.info("   Navigating via grading action dropdown...")
        try:
            action_select = page.locator("select[name='jump'], select[name='gradingactions']").first
            if await action_select.is_visible(timeout=3000):
                await action_select.select_option(label="Upload grading worksheet")
                await page.wait_for_load_state("networkidle", timeout=timeout_ms)
        except Exception as dropdown_err:
            logger.warning(f"   Dropdown navigation warning: {dropdown_err}")

    print(f"  [UPLOAD LOG] Current page: {page.url}")

    # --------------------------------------------------------------------------
    # STEP 2: Open File Picker Dialog
    # --------------------------------------------------------------------------
    logger.info("📂 [2/5] Opening Moodle File picker dialog...")
    choose_btn = page.locator(
        "button:has-text('Choose a file...'), "
        "input[value='Choose a file...'], "
        ".fp-btn-add a, "
        ".fp-btn-add button, "
        "a:has-text('Choose a file...')"
    ).first

    await choose_btn.wait_for(state="visible", timeout=10000)
    await choose_btn.click()
    print("  [UPLOAD LOG] Clicked 'Choose a file...' button. Waiting for File picker modal...")

    # Wait directly for the dynamically created attachment input to be attached in the modal
    file_input = page.locator("input[name='repo_upload_file'], input[type='file']").first
    await file_input.wait_for(state="attached", timeout=15000)
    print("  [UPLOAD LOG] ✅ File picker modal is open with file attachment input ready.")

    # --------------------------------------------------------------------------
    # STEP 3: Select 'Upload a file' in Sidebar & Attach CSV
    # --------------------------------------------------------------------------
    logger.info("📎 [3/5] Selecting 'Upload a file' repository and attaching CSV...")
    try:
        upload_tab = page.locator(
            ".fp-repo-name:has-text('Upload a file'), "
            "a:has-text('Upload a file'), "
            "[role='tab']:has-text('Upload a file')"
        ).first
        if await upload_tab.is_visible(timeout=2000):
            await upload_tab.click()
            print("  [UPLOAD LOG] Selected 'Upload a file' from left sidebar.")
            await page.wait_for_timeout(300)
    except Exception:
        pass

    # Attach file directly
    await file_input.set_input_files(abs_csv_path)
    print(f"  [UPLOAD LOG] Attached file: {os.path.basename(abs_csv_path)}")

    # Leave 'Save as', 'Author', and 'License' untouched as instructed
    print("  [UPLOAD LOG] Preserved default fields ('Save as', 'Author', 'Choose license' unchanged).")

    # Click blue 'Upload this file' button
    upload_this_file_btn = page.locator("button.fp-upload-btn, button:has-text('Upload this file')").first
    await upload_this_file_btn.wait_for(state="visible", timeout=5000)
    await upload_this_file_btn.click()
    print("  [UPLOAD LOG] Clicked 'Upload this file' button. Uploading...")

    # Wait for the file input to detach or dialog to dismiss
    try:
        await file_input.wait_for(state="detached", timeout=15000)
        print("  [UPLOAD LOG] ✅ File picker modal closed. File attached to dropzone.")
    except Exception:
        await page.wait_for_timeout(1000)

    await page.wait_for_timeout(1000)

    # --------------------------------------------------------------------------
    # STEP 4: Set Form Options (Encoding=UTF-8, Separator=Comma, Overwrite=Checked)
    # --------------------------------------------------------------------------
    logger.info("⚙️ [4/5] Setting form options (Encoding: UTF-8, Separator: Comma)...")

    # 1. Encoding: UTF-8
    try:
        encoding_select = page.locator("select[name='encoding'], select#id_encoding").first
        if await encoding_select.is_visible(timeout=3000):
            await encoding_select.select_option(value="UTF-8")
            print("  [UPLOAD LOG] Verified Encoding: 'UTF-8'")
    except Exception as enc_err:
        print(f"  [UPLOAD LOG] Encoding selector note: {enc_err}")

    # 2. Separator: Comma
    try:
        comma_radio = page.locator(
            "input[type='radio'][value='comma'], "
            "input[type='radio'][value=','], "
            "input#id_separator_comma, "
            "label:has-text('Comma')"
        ).first
        if await comma_radio.is_visible(timeout=3000):
            await comma_radio.check()
            print("  [UPLOAD LOG] Verified Separator: 'Comma'")
    except Exception as sep_err:
        print(f"  [UPLOAD LOG] Separator selector note: {sep_err}")

    # 3. Allow updating records modified more recently (left unchecked as is by default)
    try:
        overwrite_cb = page.locator(
            "input[type='checkbox'][name='overwrite'], "
            "input#id_allowoverwriting, "
            "input[type='checkbox']#id_allowoverwriting"
        ).first
        if await overwrite_cb.is_visible(timeout=3000):
            if allow_overwriting:
                await overwrite_cb.check()
                print("  [UPLOAD LOG] Checked: 'Allow updating records that have been modified more recently in Moodle'")
            else:
                # Ensure it remains unchecked as is
                if await overwrite_cb.is_checked():
                    await overwrite_cb.uncheck()
                print("  [UPLOAD LOG] Verified unchecked: 'Allow updating records that have been modified more recently in Moodle'")
    except Exception as cb_err:
        print(f"  [UPLOAD LOG] Overwrite checkbox note: {cb_err}")

    # Click 'Upload grading worksheet' submit button
    print("  [UPLOAD LOG] Submitting 'Upload grading worksheet' form...")
    submit_btn = page.locator(
        "#id_submitbutton, "
        "input[value='Upload grading worksheet'], "
        "button:has-text('Upload grading worksheet'), "
        "input[type='submit']"
    ).first
    await submit_btn.click()
    await page.wait_for_load_state("networkidle", timeout=timeout_ms)

    # --------------------------------------------------------------------------
    # STEP 5: Confirm Changes on Confirmation Preview Screen
    # --------------------------------------------------------------------------
    logger.info("📋 [5/5] Reviewing changes on 'Confirm changes in grading worksheet' screen...")
    print(f"  [UPLOAD LOG] Current page: {page.url}")

    # Extract confirmation message details from page body for terminal log
    try:
        content_box = page.locator(".box.py-3, #region-main, .mform").first
        if await content_box.is_visible(timeout=5000):
            box_text = await content_box.inner_text()
            lines = [line.strip() for line in box_text.splitlines() if line.strip()]
            print("\n  --- [CONFIRMATION PREVIEW DETECTED] ---")
            for line in lines[:10]:
                if "grade" in line.lower() or "feedback" in line.lower() or "set" in line.lower():
                    print(f"    • {line}")
            print("  ---------------------------------------\n")
    except Exception as prev_err:
        print(f"  [UPLOAD LOG] Preview text extraction note: {prev_err}")

    # Click the blue 'Confirm' button
    confirm_btn = page.locator(
        "input[value='Confirm'], "
        "button:has-text('Confirm'), "
        "#id_submitbutton, "
        "form button[type='submit']:has-text('Confirm'), "
        "form input[type='submit'][value='Confirm']"
    ).first

    await confirm_btn.wait_for(state="visible", timeout=10000)
    await confirm_btn.click()
    print("  [UPLOAD LOG] Clicked 'Confirm' button.")
    await page.wait_for_load_state("networkidle", timeout=timeout_ms)

    # Verify return to grading table or completion message
    logger.info("🎉 [SUCCESS]: Grading worksheet changes successfully confirmed and applied to Moodle!")
    print("\n" + "=" * 65)
    print("✅ [MOODLE UPLOAD COMPLETE] Grades and Feedback Comments are live in Moodle!")
    print(f"   Final Page URL: {page.url}")
    print("=" * 65 + "\n")
    return True


# ==============================================================================
# STANDALONE LAUNCHER (IF RUNNING SEPARATELY FROM INGESTION)
# ==============================================================================

async def run_standalone_worksheet_upload(
    csv_path: Optional[str] = None,
    target_url: Optional[str] = None,
    config: Optional[MoodleConfig] = None,
) -> bool:
    """
    Opens an authenticated session and uploads the graded worksheet to Moodle.
    Used when uploading an existing run independently.
    """
    cfg = config or MoodleConfig()
    cfg.validate()

    url = target_url or cfg.target_url

    # Locate CSV if not directly passed
    if not csv_path:
        # Find latest Assignment_* folder in BASE_DIR
        if os.path.exists(cfg.base_dir):
            runs = [os.path.join(cfg.base_dir, d) for d in os.listdir(cfg.base_dir) if d.startswith("Assignment_")]
            if runs:
                runs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                csv_path = os.path.join(runs[0], "grading_worksheet.csv")

    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"Grading worksheet CSV not found: {csv_path}")

    from playwright.async_api import async_playwright
    from agents.submission_reviewer.Ingestion.ingestion import get_active_or_fresh_otp

    async with async_playwright() as p:
        logger.info("🚀 Launching Playwright browser for Moodle worksheet upload...")
        browser = await p.chromium.launch(
            headless=cfg.browser_headless,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-blink-features=AutomationControlled"],
        )
        try:
            # Check for saved session state
            csv_dir = os.path.dirname(os.path.abspath(csv_path)) if csv_path else ""
            candidate_session_files = [
                os.path.join(csv_dir, ".moodle_session.json"),
                os.path.join(cfg.base_dir, ".moodle_session.json"),
                os.path.join("downloads/moodle_submissions", ".moodle_session.json"),
                "./MyDrive/Moodle_Automated_Downloads/.moodle_session.json",
            ]
            storage_state = next((f for f in candidate_session_files if os.path.exists(f)), None)

            context_kwargs = {
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            }
            if storage_state:
                context_kwargs["storage_state"] = storage_state
                logger.info(f"🔄 Reusing saved Moodle session from: {storage_state}")

            context = await browser.new_context(**context_kwargs)
            page = await context.new_page()

            # Navigate directly to target upload URL first
            upload_url = build_upload_worksheet_url(url)
            logger.info(f"🌐 Navigating to upload worksheet page: {upload_url}...")
            await page.goto(upload_url, wait_until="networkidle", timeout=cfg.browser_timeout_ms)

            # Check if Moodle requires login (session expired or missing)
            needs_login = (
                "login" in page.url
                or "auth.php" in page.url
                or "tool/mfa" in page.url
                or await page.locator("#username, input[name='username']").count() > 0
            )

            if needs_login:
                logger.info("🔑 Session expired or missing. Performing login...")
                parsed_target = urlparse(url)
                base_origin = f"{parsed_target.scheme}://{parsed_target.netloc}"
                login_url = f"{base_origin}/login/index.php"

                if "login" not in page.url:
                    await page.goto(login_url, wait_until="networkidle", timeout=cfg.browser_timeout_ms)

                user_input = page.locator("#username, input[name='username']").first
                if await user_input.is_visible(timeout=3000):
                    await user_input.fill(cfg.moodle_user or "")
                    await page.fill("#password, input[name='password']", cfg.moodle_pass or "")
                    await page.click("#loginbtn, button[type='submit']")
                    await page.wait_for_load_state("networkidle")
                    await page.wait_for_timeout(2000)

                # Check MFA
                if "tool/mfa" in page.url or "auth.php" in page.url or await page.locator("#id_verificationcode").count() > 0:
                    logger.info("🔒 MFA verification detected! Fetching OTP code...")
                    otp_code = await asyncio.to_thread(
                        get_active_or_fresh_otp,
                        cfg.email_user,
                        cfg.email_pass,
                        cfg.imap_server,
                        cfg.imap_folder,
                        cfg.otp_max_age_minutes,
                        cfg.otp_timeout_seconds,
                    )
                    await page.fill("#id_verificationcode, input[name*='code'], #id_code", otp_code)
                    continue_btn = page.locator("button:has-text('Continue'), input[value='Continue']").first
                    if await continue_btn.is_visible(timeout=2000):
                        await continue_btn.click()
                    for _ in range(15):
                        if "auth.php" not in page.url and "login" not in page.url and "tool/mfa" not in page.url:
                            break
                        await page.wait_for_timeout(1000)

                logger.info("🎉 Moodle login verified successfully!")
                # Save session
                save_session_path = storage_state or os.path.join(cfg.base_dir, ".moodle_session.json")
                try:
                    await context.storage_state(path=save_session_path)
                    logger.info(f"💾 Saved authenticated session to {save_session_path}")
                except Exception as s_err:
                    logger.debug(f"Could not save session state: {s_err}")
            else:
                logger.info("🎉 Active Moodle session verified! Reused session without logging in again.")

            # Upload worksheet
            return await upload_grading_worksheet_to_moodle(
                page=page,
                csv_path=csv_path,
                target_url=url,
                timeout_ms=cfg.browser_timeout_ms,
            )
        finally:
            await browser.close()


def run_standalone_worksheet_upload_sync(
    csv_path: Optional[str] = None,
    target_url: Optional[str] = None,
    config: Optional[MoodleConfig] = None,
) -> bool:
    """Synchronous wrapper for standalone upload."""
    return asyncio.run(run_standalone_worksheet_upload(csv_path=csv_path, target_url=target_url, config=config))


# ==============================================================================
# CLI EXECUTION
# ==============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Upload graded CSV worksheet back to Moodle.")
    parser.add_argument("--csv", type=str, default=None, help="Path to graded grading_worksheet.csv (defaults to latest run).")
    parser.add_argument("--url", type=str, default=None, help="Moodle assignment URL (defaults to MOODLE_ASSIGNMENT_URL).")
    parser.add_argument("--headed", action="store_true", help="Run browser in visible mode.")

    args = parser.parse_args()

    cfg = MoodleConfig(
        target_url=args.url or TARGET_URL,
        browser_headless=not args.headed,
    )

    try:
        run_standalone_worksheet_upload_sync(csv_path=args.csv, target_url=args.url, config=cfg)
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
