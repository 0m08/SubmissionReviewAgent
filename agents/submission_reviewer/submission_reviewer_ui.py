import streamlit as st
import gspread
import traceback
import requests
import io
from typing import Optional
from PIL import Image

from agents.submission_reviewer.sheet_processor import (
    process_activity_sheet,
    parse_activity_instructions_tab,
)
from agents.submission_reviewer.reviewer import review_single_submission
from agents.submission_reviewer.image_identifier import (
    generate_image_description,
    score_description_match,
    identify_media,
)
from agents.submission_reviewer.hvac_eval_processor import (
    run_agent1_pipeline,
    run_agent2_pipeline,
    run_full_evaluation_pipeline,
)

from agents.submission_reviewer.drive_image_helper import (
    fetch_images_from_drive_links,
    extract_all_drive_ids,
)


DEFAULT_SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1RjJsuygOCFU_s5RYDT7iBxdpOdlSXOG4jESCsJhmzs0/edit?usp=sharing"
HVAC_EVAL_SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1sKpdTfK1U8CQsz6y9nhW5camRV1DDL66rv33WfvaGE4/edit?usp=sharing"




from bs4 import BeautifulSoup
from urllib.parse import urljoin


def fetch_media_and_text_from_url(raw_url: str, session_creds=None, max_images: int = 5):
    """
    Fetches images and optional page text from any URL:
    - Google Drive links (files or folders)
    - Direct image URLs (.jpg, .png, .webp, etc.)
    - Webpages/Articles (extracts embedded images and article text context)
    Returns: (images_list, page_text_str)
    """
    images = []
    page_text = ""
    raw_url = raw_url.strip()

    # 1. Google Drive URLs
    if "drive.google.com" in raw_url or "drive.usercontent.google.com" in raw_url or extract_all_drive_ids(raw_url):
        try:
            drive_imgs, _ = fetch_images_from_drive_links(raw_url, session_creds=session_creds, max_images=max_images)
            images.extend(drive_imgs)
        except Exception as d_err:
            print(f"  [URL LOADER] Error fetching Google Drive images: {d_err}")
        return images, page_text

    # 2. Standard Web URLs (direct image or webpage)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"}
    try:
        resp = requests.get(raw_url, headers=headers, timeout=15)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "").lower()

        # Check if direct binary image
        if content_type.startswith("image/") or any(raw_url.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif"]):
            try:
                img = Image.open(io.BytesIO(resp.content))
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                images.append(img)
                return images, page_text
            except Exception:
                pass

        # Try parsing as image binary directly
        try:
            img = Image.open(io.BytesIO(resp.content))
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            images.append(img)
            return images, page_text
        except Exception:
            pass  # Not a binary image, parse HTML webpage

        # 3. HTML Webpage parsing (extract images and text)
        soup = BeautifulSoup(resp.text, "html.parser")

        # Extract clean text context
        for element in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            element.decompose()
        text_content = soup.get_text(separator=" ", strip=True)
        if text_content:
            page_text = text_content[:2000]

        # Extract embedded images
        img_tags = soup.find_all("img")
        for tag in img_tags:
            if len(images) >= max_images:
                break
            src = tag.get("src") or tag.get("data-src") or tag.get("srcset")
            if not src:
                continue
            if " " in src:
                src = src.split()[0]
            abs_url = urljoin(raw_url, src)
            if not abs_url.startswith(("http://", "https://")):
                continue
            if any(skip in abs_url.lower() for skip in ["logo", "icon", "avatar", "pixel", ".svg"]):
                continue

            try:
                img_resp = requests.get(abs_url, headers=headers, timeout=8)
                if img_resp.status_code == 200:
                    sub_img = Image.open(io.BytesIO(img_resp.content))
                    if sub_img.width >= 80 and sub_img.height >= 80:
                        if sub_img.mode not in ("RGB", "L"):
                            sub_img = sub_img.convert("RGB")
                        images.append(sub_img)
            except Exception:
                continue

    except Exception as e:
        print(f"  [URL LOADER] Error loading URL '{raw_url}': {e}")

    return images, page_text


def render_submission_reviewer_ui():
    st.header("🔍 User Submission Reviewer Agent")
    st.markdown(
        "Evaluates user activity submissions (Drive images + User Comments) against "
        "Activity Instructions and Reviewer Checklists using Gemini Multimodal reasoning."
    )

    tab1, tab2, tab3 = st.tabs([
        "📊 Batch Sheet Reviewer",
        "🧪 Single Submission Sandbox",
        "🔎 Identification Sandbox",
    ])

    with tab1:
        st.subheader("Process Google Sheet Submissions")

        col_url, col_ref = st.columns([4, 1])
        with col_url:
            sheet_url = st.text_input(
                "Central Google Sheet URL",
                value=DEFAULT_SPREADSHEET_URL,
                key="reviewer_sheet_url"
            )
        with col_ref:
            st.write("") # Alignment spacing
            st.write("")
            refresh_tabs = st.button("🔄 Refresh Tabs", help="Click to reload worksheets from Google Sheets")

        gc = st.session_state.get("gc") or st.session_state.get("gspread_client")
        if not gc:
            st.warning("⚠️ Google Authentication required. Please log in with Google OAuth first.")
            return

        # Fetch tabs dynamically
        available_tabs = []
        instructions_map = {}
        SYSTEM_TABS = {"activity instructions", "guardrails"}

        if sheet_url:
            try:
                sheet = gc.open_by_url(sheet_url)
                all_worksheets = [ws.title for ws in sheet.worksheets()]
                
                # Try parsing activity instructions tab if present
                try:
                    instructions_map = parse_activity_instructions_tab(sheet)
                except Exception as instr_err:
                    st.info(f"Note: Could not parse instructions tab: {instr_err}")
                
                # Activity tabs are any worksheet titles not in SYSTEM_TABS
                # or any tab names explicitly registered in Activity Instructions
                sheet_activity_tabs = [t for t in all_worksheets if t.strip().lower() not in SYSTEM_TABS]
                instr_tab_names = [t for t in instructions_map.keys() if t not in sheet_activity_tabs and t.strip().lower() not in SYSTEM_TABS]
                
                available_tabs = sheet_activity_tabs + instr_tab_names
                
                if refresh_tabs:
                    st.toast(f"✅ Loaded {len(available_tabs)} activity tab(s) from spreadsheet!")
            except Exception as e:
                st.error(f"Could not load spreadsheet tabs: {e}")

        st.markdown("### 🎯 Target Activity Tab Selection")
        if available_tabs:
            st.caption(f"📋 Found **{len(available_tabs)}** activity tab(s): {', '.join(available_tabs)}")

        run_mode = st.radio(
            "Select Run Scope",
            options=["Specific Tab", "All Activity Tabs"],
            index=0,
            horizontal=True
        )

        selected_tabs = []
        if run_mode == "Specific Tab":
            if available_tabs:
                chosen_tab = st.selectbox(
                    "Select Activity Tab to Process",
                    options=available_tabs,
                    index=0,
                    help="Only rows in this activity tab will be evaluated."
                )
                selected_tabs = [chosen_tab]
            else:
                custom_tab = st.text_input(
                    "Enter Activity Tab Name (e.g. System Identification)",
                    value="System Identification"
                )
                selected_tabs = [custom_tab.strip()] if custom_tab else []
        else:
            selected_tabs = available_tabs

        col_opt1, col_opt2, col_opt3 = st.columns(3)
        with col_opt1:
            overwrite_existing = st.checkbox(
                "Overwrite existing Agent_Grade entries",
                value=True,
                help="If unchecked, rows with existing Agent_Grade values will be skipped."
            )
        with col_opt2:
            mentor_mode_select = st.radio(
                "Mentorship Leeway Mode",
                options=["Standard (Strict)", "Mentor Leeway (Balanced)", "Ultra-Relaxed (High Pass Rate >80%)"],
                index=2,
                help="Ultra-Relaxed passes almost all attempts with visible effort, failing ONLY super critical core faults (e.g. Conduction vs Convection mix-up)."
            )
        with col_opt3:
            model_provider_select = st.radio(
                "Primary AI Model",
                options=["GPT-5.6 Luna (High Reasoning)", "Gemini 3.5 Flash (High Thinking)"],
                index=0,
                help="Switch between GPT-5.6 Luna and Gemini 3.5 Flash with High Thinking budget."
            )

        is_lenient_ui = mentor_mode_select in ("Mentor Leeway (Balanced)", "Ultra-Relaxed (High Pass Rate >80%)")
        is_ultra_ui = (mentor_mode_select == "Ultra-Relaxed (High Pass Rate >80%)")
        selected_model_code = "gemini-3.5-flash-thinking" if "Gemini" in model_provider_select else "gpt-5.6-luna"

        if st.button("🚀 Run Submission Reviewer Agent", type="primary"):
            if not selected_tabs:
                st.error("Please provide or select a valid activity tab to run.")
            else:
                st.info(f"Targeting Tab(s): {', '.join(selected_tabs)} | Mode: '{mentor_mode_select}' | Model: '{model_provider_select}'. Detailed logs in console.")
                progress_placeholder = st.progress(0, text="Starting submission review...")
                try:
                    def progress_cb(current, total, msg):
                        pct = int((current / max(1, total)) * 100)
                        progress_placeholder.progress(pct / 100, text=msg)

                    summary = process_activity_sheet(
                        sheet_url=sheet_url,
                        gc_client=gc,
                        target_tab_names=selected_tabs,
                        overwrite_existing=overwrite_existing,
                        progress_callback=progress_cb,
                        lenient_mode=is_lenient_ui,
                        ultra_lenient_mode=is_ultra_ui,
                        primary_model_choice=selected_model_code,
                    )

                    progress_placeholder.progress(1.0, text="✅ Processing complete!")
                    st.success(f"✅ Submission review for {', '.join(selected_tabs)} successfully finished!")

                    st.markdown("### 📈 Execution Summary")
                    col1, col2 = st.columns(2)
                    col1.metric("Total Evaluated Rows", summary.get("evaluated_rows", 0))
                    col2.metric("Skipped Rows ('No Submission')", summary.get("skipped_rows", 0))
                    
                    st.json(summary.get("tab_summaries", {}))
                except Exception as err:
                    progress_placeholder.empty()
                    st.error(f"❌ Failed to process sheet:\n{traceback.format_exc()}")

    with tab2:
        st.subheader("Sandbox Test Single Submission")
        col_left, col_right = st.columns(2)

        with col_left:
            activity_name_input = st.text_input("Activity Name", value="System Identification")
            instructions_input = st.text_area(
                "Activity Instructions",
                value="Identify the high pressure and low pressure ports. Ensure safety caps are visible.",
                height=120
            )
            edge_cases_input = st.text_area(
                "Activity Edge Cases (Optional)",
                value="",
                placeholder="e.g. Photo of a portable unit is acceptable",
                height=80,
                help="Specific reviewer rules specifying acceptable vs unacceptable submission variations."
            )
            # Reviewer Checklist disabled — reviews are performed solely against Activity Instructions.
            checklist_input = ""
            user_comment_input = st.text_area("User Comment", value="Here are the photos of both ports.", height=80)
            uploaded_files = st.file_uploader(
                "Upload Submission Images or Videos",
                type=["jpg", "png", "jpeg", "webp", "mp4", "mov", "avi", "webm"],
                accept_multiple_files=True
            )

        with col_right:
            st.markdown("### AI Review Preview")
            sandbox_mode_select = st.radio(
                "Mentorship Leeway Mode",
                options=["Standard (Strict)", "Mentor Leeway (Balanced)", "Ultra-Relaxed (High Pass Rate >80%)"],
                index=2,
                key="sandbox_mentor_mode"
            )
            sandbox_model_select = st.radio(
                "Primary AI Model",
                options=["GPT-5.6 Luna (High Reasoning)", "Gemini 3.5 Flash (High Thinking)"],
                index=0,
                key="sandbox_model_choice"
            )
            sb_lenient = sandbox_mode_select in ("Mentor Leeway (Balanced)", "Ultra-Relaxed (High Pass Rate >80%)")
            sb_ultra = (sandbox_mode_select == "Ultra-Relaxed (High Pass Rate >80%)")
            sb_model_code = "gemini-3.5-flash-thinking" if "Gemini" in sandbox_model_select else "gpt-5.6-luna"

            if st.button("🧪 Evaluate Sandbox Submission"):
                images = []
                videos = []
                if uploaded_files:
                    for f in uploaded_files:
                        ext = f.name.split(".")[-1].lower() if "." in f.name else ""
                        if ext in ("mp4", "mov", "avi", "webm") or (f.type and f.type.startswith("video/")):
                            mime = f.type or ("video/mp4" if ext == "mov" else f"video/{ext}")
                            videos.append({
                                "bytes": f.getvalue(),
                                "mime_type": mime,
                                "filename": f.name
                            })
                        else:
                            try:
                                images.append(Image.open(f))
                            except Exception:
                                pass

                with st.spinner("Analyzing submission with AI Reviewer..."):
                    result = review_single_submission(
                        activity_name=activity_name_input,
                        activity_instructions=instructions_input,
                        reviewer_checklist=checklist_input,
                        user_comment=user_comment_input,
                        images=images,
                        videos=videos,
                        activity_edge_cases=edge_cases_input,
                        lenient_mode=sb_lenient,
                        ultra_lenient_mode=sb_ultra,
                        primary_model_choice=sb_model_code,
                    )


                if result.agent_grade == "Pass":
                    st.success(f"**Agent Grade:** {result.agent_grade}")
                elif result.agent_grade == "Pass (Unsure)":
                    st.info(f"**Agent Grade:** {result.agent_grade}")
                elif result.agent_grade == "Fail (Unsure)":
                    st.warning(f"**Agent Grade:** {result.agent_grade}")
                else:
                    st.error(f"**Agent Grade:** {result.agent_grade}")

                st.markdown(f"**Agent Comment:**\n{result.agent_comment}")
                st.markdown("**Checklist Breakdown:**")
                for item in result.checklist_evaluations:
                    status_icon = "✅" if item.followed else "❌"
                    st.markdown(f"- {status_icon} **{item.instruction_or_condition}**: {item.comment}")

    with tab3:
        st.subheader("📊 HVAC Multimodal Evaluation Pipeline")
        st.markdown(
            "Benchmark multimodal AI models on HVAC technical assets using a **Two-Agent Architecture**:\n"
            "- **Agent 1**: Generates concise technical descriptions from HVAC image links & prompts.\n"
            "- **Agent 2**: Evaluates AI descriptions against human ground-truth descriptions using **Gemini 3.7 Flash** to issue deterministic `Yes` / `No` match scores."
        )

        # ---------------------------------------------------------------------
        # 1. Central Sheet Benchmark Runner
        # ---------------------------------------------------------------------
        eval_sheet_url = st.text_input(
            "Central Evaluation Test Sheet URL",
            value=HVAC_EVAL_SPREADSHEET_URL,
            key="hvac_eval_sheet_url",
            help="Google Sheet URL containing Image Link, Prompt, Image Description (Ground Truth), Model output, and Scoring agent Match columns."
        )

        gc = st.session_state.get("gc") or st.session_state.get("gspread_client")
        eval_tabs = []
        target_eval_sheet = None

        if eval_sheet_url and gc:
            try:
                target_eval_sheet = gc.open_by_url(eval_sheet_url)
                eval_tabs = [ws.title for ws in target_eval_sheet.worksheets()]
            except Exception as e:
                st.warning(f"Could not connect to Google Sheet: {e}")

        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        with col_cfg1:
            selected_eval_tab = st.selectbox(
                "Select Worksheet Tab",
                options=eval_tabs if eval_tabs else ["Sheet1"],
                index=0,
                key="hvac_eval_tab_select"
            )
        with col_cfg2:
            agent1_model_ui = st.selectbox(
                "Agent 1 Model (Description Generator)",
                options=["GPT-5.6 Luna Max", "Gemini 3.7 Flash", "Gemini 3.5 Flash", "GPT-4o"],
                index=0,
                key="hvac_agent1_model_select"
            )
        with col_cfg3:
            agent2_model_ui = st.selectbox(
                "Agent 2 Model (Automated QA Judge)",
                options=["Gemini 3.7 Flash (Default)", "Gemini 3.5 Flash"],
                index=0,
                key="hvac_agent2_model_select"
            )

        # Map UI choice to internal model codes
        model_code_map = {
            "GPT-5.6 Luna Max": "gpt-5.6-luna",
            "Gemini 3.7 Flash": "gemini-3.7-flash",
            "Gemini 3.5 Flash": "gemini-3.5-flash",
            "GPT-4o": "gpt-4o",
        }
        agent1_model_code = model_code_map.get(agent1_model_ui, "gpt-5.6-luna")
        agent2_model_code = "gemini-3.7-flash" if "3.7" in agent2_model_ui else "gemini-3.5-flash"
        overwrite_hvac_results = st.checkbox(
            "Overwrite existing output entries",
            value=False,
            help="If unchecked, rows that already have non-empty output values will be skipped automatically."
        )

        st.divider()
        st.markdown("### 🚀 Execute Pipeline Stage")

        btn_col1, btn_col2, btn_col3 = st.columns(3)

        with btn_col1:
            run_agent1_btn = st.button(
                "🤖 Run Agent 1 Only\n(Generate Descriptions)",
                type="primary",
                use_container_width=True,
                key="btn_run_agent1"
            )

        with btn_col2:
            run_agent2_btn = st.button(
                "⚖️ Run Agent 2 Only\n(Score Matches: Yes / No)",
                type="primary",
                use_container_width=True,
                key="btn_run_agent2"
            )

        with btn_col3:
            run_full_btn = st.button(
                "⚡ Run Full Pipeline\n(Agent 1 + Agent 2)",
                type="primary",
                use_container_width=True,
                key="btn_run_full_pipeline"
            )

        # Execution Progress & Output Area
        if run_agent1_btn or run_agent2_btn or run_full_btn:
            if not target_eval_sheet:
                st.error("⚠️ Google Auth or Sheet URL invalid. Please connect first.")
            else:
                progress_bar = st.progress(0, text="Initializing...")
                gc_creds = st.session_state.get("google_credentials")

                def update_progress(current: int, total: int, msg: str):
                    pct = float(current) / float(max(1, total))
                    progress_bar.progress(pct, text=msg)

                try:
                    if run_agent1_btn:
                        st.info(f"Running Agent 1 with **{agent1_model_ui}** on tab **'{selected_eval_tab}'**...")
                        res = run_agent1_pipeline(
                            sheet=target_eval_sheet,
                            worksheet_name=selected_eval_tab,
                            model_choice=agent1_model_code,
                            overwrite_existing=overwrite_hvac_results,
                            session_creds=gc_creds,
                            progress_callback=update_progress,
                        )
                        progress_bar.progress(1.0, text="✅ Agent 1 complete!")
                        st.success(f"✅ Processed {len(res)} row(s)!")

                    elif run_agent2_btn:
                        st.info(f"Running Agent 2 Scoring Judge (**{agent2_model_ui}**) for model **{agent1_model_ui}** on tab **'{selected_eval_tab}'**...")
                        res = run_agent2_pipeline(
                            sheet=target_eval_sheet,
                            worksheet_name=selected_eval_tab,
                            target_model=agent1_model_code,
                            scoring_model=agent2_model_code,
                            overwrite_existing=overwrite_hvac_results,
                            progress_callback=update_progress,
                        )
                        progress_bar.progress(1.0, text="✅ Agent 2 complete!")
                        st.success(f"✅ Scored {len(res)} row(s)!")
                        if res:
                            st.json(res[:5])

                    elif run_full_btn:
                        st.info(f"Running Full Evaluation Pipeline on tab **'{selected_eval_tab}'**...")
                        res = run_full_evaluation_pipeline(
                            sheet=target_eval_sheet,
                            worksheet_name=selected_eval_tab,
                            agent1_model=agent1_model_code,
                            agent2_model=agent2_model_code,
                            overwrite_existing=overwrite_hvac_results,
                            session_creds=gc_creds,
                            progress_callback=update_progress,
                        )
                        progress_bar.progress(1.0, text="✅ Full Evaluation Pipeline finished!")
                        st.success("✅ Agent 1 and Agent 2 pipeline completed!")

                except Exception as eval_err:
                    progress_bar.empty()
                    st.error(f"❌ Evaluation error: {eval_err}\n{traceback.format_exc()}")

        st.divider()

        # ---------------------------------------------------------------------
        # 2. Interactive Single-Image Playground
        # ---------------------------------------------------------------------
        with st.expander("🧪 Playground: Single Photo Identification & QA Match Evaluator", expanded=False):
            col_id_left, col_id_right = st.columns(2)

            with col_id_left:
                uploaded_id_file = st.file_uploader(
                    "Upload HVAC Asset Photo",
                    type=["jpg", "jpeg", "png", "webp"],
                    key="id_file_uploader",
                )
                id_url_input = st.text_input(
                    "Or Image / Drive URL",
                    placeholder="https://...",
                    key="id_url_input",
                )
                id_eval_prompt = st.text_area(
                    "Agent 1 Prompt",
                    value="Provide a detailed technical description of the HVAC equipment, component names, and visible tools in this image.",
                    height=80,
                    key="id_eval_prompt",
                )

            with col_id_right:
                selected_llm = st.selectbox(
                    "Select Agent 1 Model",
                    options=["GPT-5.6 Luna", "Gemini 3.7 Flash", "Gemini 3.5 Flash"],
                    index=0,
                    key="id_llm_select",
                )
                id_human_gt = st.text_area(
                    "Human Ground-Truth Description (For Agent 2 QA)",
                    placeholder="Optional: Enter ground truth description to test Agent 2 match scoring...",
                    height=120,
                    key="id_human_gt",
                )
                run_single_btn = st.button("🔍 Run Playground Analysis", type="secondary", key="btn_run_single", use_container_width=True)

            # Process single media
            preview_images = []
            page_text_context = ""

            if uploaded_id_file is not None:
                try:
                    img = Image.open(uploaded_id_file)
                    if img.mode not in ("RGB", "L"):
                        img = img.convert("RGB")
                    preview_images.append(img)
                except Exception as img_err:
                    st.error(f"Could not read uploaded photo: {img_err}")

            if id_url_input and id_url_input.strip():
                raw_url = id_url_input.strip()
                with st.spinner("Loading image from URL..."):
                    gc_creds = st.session_state.get("google_credentials")
                    url_imgs, page_text_context = fetch_media_and_text_from_url(raw_url, session_creds=gc_creds, max_images=3)
                    if url_imgs:
                        preview_images.extend(url_imgs)

            if run_single_btn:
                if not preview_images and not page_text_context:
                    st.warning("⚠️ Please upload a photo or provide a valid image/Drive URL.")
                else:
                    final_prompt = id_eval_prompt.strip()
                    if page_text_context:
                        final_prompt += f"\n\nExtracted Webpage Content:\n{page_text_context}"

                    model_code = "gemini-3.7-flash" if "3.7" in selected_llm else ("gemini-3.5-flash" if "Gemini" in selected_llm else "gpt-5.6-luna")

                    st.markdown("**🤖 Agent 1 Description Output:**")
                    with st.spinner(f"Generating description with {selected_llm}..."):
                        ai_output = generate_image_description(
                            prompt=final_prompt,
                            images=preview_images,
                            model_choice=model_code,
                        )
                    st.info(ai_output)

                    if id_human_gt and id_human_gt.strip():
                        st.markdown("**⚖️ Agent 2 Scoring Result (Gemini 3.7 Flash):**")
                        with st.spinner("Scoring AI vs. Human Description..."):
                            match_score, rationale = score_description_match(
                                ai_description=ai_output,
                                human_description=id_human_gt,
                                model_choice="gemini-3.7-flash",
                            )
                        if match_score == "Yes":
                            st.success(f"**Match Result:** {match_score} ✅\n\n**Rationale:** {rationale}")
                        else:
                            st.error(f"**Match Result:** {match_score} ❌\n\n**Rationale:** {rationale}")






