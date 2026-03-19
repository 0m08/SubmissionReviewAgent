import streamlit as st
import gspread
from dotenv import load_dotenv
from pydrive2.drive import GoogleDrive
import traceback
import csv
import re
import time
from services.sheets_service import get_sheet_data_and_df, create_or_read_worksheet, format_worksheet, save_to_sheet
from services.smart_progress_bar import SmartProgressBar
from services.drive_service import login_with_oauth2, share_sheet_with_service_account, get_service_account_email
from services.activity_tracking_service import (
    track_step_start,
    track_step_complete,
    track_step_error,
    track_run_all_start,
    track_run_in_background_start,
)
from datetime import datetime
import os
from langtrace_python_sdk import langtrace # Must precede any llm module imports
import tempfile, json, base64
import subprocess
import sys
from services.helper_functions import get_short_name
import re

# Mapping from display names used in the Streamlit UI to the
# agent names expected by the SDK/CLI scripts.
AGENT_CODE_MAP = {
    "Course Outline": "course_outline",
    "Research Notes": "research_notes",
    "Slide Chunks": "slide_chunks",
    "Graphics Definition": "graphics_definition",
    "Graphics Definition V2": "graphics_definition_v2",
    "Assessment": "assessment",
    "Graphics Search": "graphics_search",
}

TOKEN_USAGE_LOG_FILE = "token_usage_log.csv"


def _set_step_context(agent_name: str, step_name: str) -> None:
    os.environ["CURRENT_AGENT_NAME"] = agent_name or ""
    os.environ["CURRENT_STEP_NAME"] = step_name or ""


def _clear_step_context() -> None:
    os.environ["CURRENT_AGENT_NAME"] = ""
    os.environ["CURRENT_STEP_NAME"] = ""


def _is_llm_step(step: dict) -> bool:
    return "llm" in step.get("args", {}) or step.get("is_llm_step", False)


def _get_token_log_offset(log_file: str) -> int:
    try:
        return os.path.getsize(log_file)
    except OSError:
        return 0


def _record_step_token_offset(step_name: str) -> None:
    if "step_token_offsets" not in st.session_state:
        st.session_state["step_token_offsets"] = {}
    st.session_state["step_token_offsets"][step_name] = _get_token_log_offset(TOKEN_USAGE_LOG_FILE)


def _pop_step_token_offset(step_name: str) -> int | None:
    offsets = st.session_state.get("step_token_offsets", {})
    return offsets.pop(step_name, None)


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    if seconds < 1:
        return f"{seconds:.2f}s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    total_seconds = int(round(seconds))
    minutes, secs = divmod(total_seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins}m {secs}s"


def _format_cost(cost: float, currency: str = "$") -> str:
    return f"{currency}{cost:,.4f}"


def _iter_token_log_rows(log_file: str, start_offset: int | None = None):
    if not os.path.exists(log_file):
        return
    try:
        with open(log_file, mode="r", newline="") as csvfile:
            if start_offset:
                file_size = os.path.getsize(log_file)
                if start_offset > file_size:
                    start_offset = 0
                csvfile.seek(start_offset)
                reader = csv.reader(csvfile)
                for row in reader:
                    yield row
                return

            reader = csv.reader(csvfile)
            first_row = next(reader, None)
            if not first_row:
                return
            header = [col.strip().lower() for col in first_row[:6]]
            if header != ["agent_name", "step_name", "timestamp", "llm", "input_tokens", "output_tokens"]:
                yield first_row
            for row in reader:
                yield row
    except Exception:
        return


def _safe_int(value: str) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def _resolve_pricing_for_model(llm_name: str, pricing: dict | None) -> dict | None:
    if not pricing:
        return None

    # Optional per-model pricing support.
    model_pricing = pricing.get("models", {})
    if isinstance(model_pricing, dict):
        if llm_name in model_pricing:
            return model_pricing[llm_name]
        normalized_name = (llm_name or "").strip().lower().replace("-", "_")
        for key, value in model_pricing.items():
            if key.strip().lower().replace("-", "_") == normalized_name:
                return value

    # Backward-compatible flat pricing.
    if "input_per_million" in pricing and "output_per_million" in pricing:
        return pricing

    default_pricing = pricing.get("default")
    if isinstance(default_pricing, dict):
        return default_pricing

    return None


def _sum_tokens_for_step(
    agent_name: str,
    step_name: str,
    start_time: datetime | None,
    end_time: datetime | None,
    log_file: str,
    start_offset: int | None = None,
    pricing: dict | None = None,
) -> tuple[int, int, float | None]:
    total_input = 0
    total_output = 0
    total_cost = 0.0
    has_cost = False
    if not step_name:
        return total_input, total_output, None
    for row in _iter_token_log_rows(log_file, start_offset):
        if not row or len(row) < 6:
            continue
        row_agent, row_step, timestamp, row_llm, input_tokens, output_tokens = row[:6]
        if row_step != step_name:
            continue
        if agent_name and row_agent != agent_name:
            continue
        if timestamp:
            try:
                ts = timestamp.rstrip("Z")
                ts_dt = datetime.fromisoformat(ts)
            except Exception:
                continue
            if start_time and ts_dt < start_time:
                continue
            if end_time and ts_dt > end_time:
                continue
        row_input = _safe_int(input_tokens)
        row_output = _safe_int(output_tokens)
        total_input += row_input
        total_output += row_output

        row_pricing = _resolve_pricing_for_model(row_llm, pricing)
        if row_pricing:
            total_cost += _calculate_step_cost(row_input, row_output, row_pricing)
            has_cost = True
    return total_input, total_output, (total_cost if has_cost else None)


def _calculate_step_cost(input_tokens: int, output_tokens: int, pricing: dict) -> float:
    input_rate = float(pricing.get("input_per_million", 0.0))
    output_rate = float(pricing.get("output_per_million", 0.0))
    return (input_tokens * input_rate / 1_000_000) + (output_tokens * output_rate / 1_000_000)


def _record_step_metrics(step: dict, duration_seconds: float, start_time: datetime, end_time: datetime, llm_pricing: dict | None):
    if "step_metrics" not in st.session_state:
        st.session_state["step_metrics"] = {}
    metrics = {"duration_seconds": duration_seconds}

    if llm_pricing and _is_llm_step(step):
        try:
            agent_name = st.session_state.get("agent_name", "")
        except (RuntimeError, AttributeError):
            agent_name = ""
        start_offset = _pop_step_token_offset(step["name"])
        input_tokens, output_tokens, step_cost = _sum_tokens_for_step(
            agent_name=agent_name,
            step_name=step["name"],
            start_time=start_time,
            end_time=end_time,
            log_file=TOKEN_USAGE_LOG_FILE,
            start_offset=start_offset,
            pricing=llm_pricing,
        )
        metrics["input_tokens"] = input_tokens
        metrics["output_tokens"] = output_tokens
        if step_cost is not None:
            metrics["cost"] = step_cost
        elif llm_pricing and "input_per_million" in llm_pricing and "output_per_million" in llm_pricing:
            metrics["cost"] = _calculate_step_cost(input_tokens, output_tokens, llm_pricing)

    st.session_state["step_metrics"][step["name"]] = metrics


def agent_ui(step_name: str, pipeline_sections: list[dict], outline_finalized: bool = False, llm_pricing: dict | None = None, top_instructions: str | None = None, top_toggles: list[dict] | None = None):
    st.session_state["outline_finalized"] = outline_finalized
    st.title(f"{step_name} Agent")

    if "agent_name" not in st.session_state:
        st.session_state["agent_name"] = ""
    if st.session_state["agent_name"] != step_name:
        st.session_state["agent_name"] = step_name

    if "sheet" in st.session_state:
        load_completed_steps(st.session_state["sheet"], step_name)

    if "google_api_key" not in st.session_state:
        st.session_state["google_api_key"] = ""

    with st.sidebar:
        st.text_input(label="Google API Key", type = "password", key = "google_api_key", value = st.session_state["google_api_key"])
        # st.selectbox(label="LLM Model", options= ["gemini_2_flash", "gpt4_1"], index = None, key = "llm_model")

    if st.session_state["google_api_key"] != "":
        os.environ["GOOGLE_API_KEY"] = st.session_state["google_api_key"]

    if "current_step" not in st.session_state:
        st.session_state["current_step"] = None
    if "step_metrics" not in st.session_state:
        st.session_state["step_metrics"] = {}


    # --- 1) Define pipeline as sections, each with its own steps ---

    # --- 2) Initialize session states for each step ---
    for section in pipeline_sections:
        for step in section["steps"]:
            step_key = f"{step['name']}_done"
            if step_key not in st.session_state:
                st.session_state[step_key] = False

            # Only initialize pre-execution state if the step has a pre_exec_func
            if "pre_exec_func" in step:
                pre_exec_key = f"{step['name']}_pre_executed"
                if pre_exec_key not in st.session_state:
                    st.session_state[pre_exec_key] = False

    # --- 3) Hide "Load Data" inputs once data is loaded ---
    if "sheet" not in st.session_state:
        root_folder_id = st.text_input("Enter course Drive folder ID")
        sheet_link = st.text_input("Enter Google Sheet link")

        # Button to load data
        if st.button("Load Data"):
            load_dotenv()  # Load env variables from .env
            # Check if the Google API key is provided
            if st.session_state["google_api_key"] != "":
                os.environ["GOOGLE_API_KEY"] = st.session_state["google_api_key"]
            try:
                if os.environ.get('LANGTRACE_ON', 'false') == "true":
                    langtrace.init(api_key = os.environ.get('LANGTRACE_API_KEY'))

                # --- OAuth2 for both Google Drive and Google Sheets ---
                oauth_client_id = os.environ.get("OAUTH_CLIENT_ID")
                oauth_client_secret = os.environ.get("OAUTH_CLIENT_SECRET")
                
                if not oauth_client_id or not oauth_client_secret:
                    st.error("OAuth credentials not found. Please set OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET environment variables.")
                    st.stop()
                
                # Use existing authenticated clients from session if available
                if "drive" in st.session_state and "gc" in st.session_state:
                    drive = st.session_state["drive"]
                    gc = st.session_state["gc"]
                else:
                    st.stop()
                sheet = gc.open_by_url(sheet_link)
                course_info_sheet, course_info_df = get_sheet_data_and_df(sheet, 'Course info')

                st.session_state["root_folder_id"] = root_folder_id
                st.session_state["sheet"] = sheet
                st.session_state["sheet_link"] = sheet_link
                st.session_state["course_name"] = course_info_df['Course Name'][0]
                st.session_state["target_audience"] = course_info_df['Target Audience & Industry'][0]
                st.session_state["course_background"] = course_info_df['Course Background'][0]
                st.session_state["course_objective_guidelines"] = course_info_df['Course Objective Guidelines'][0]
                st.session_state["checklist_sheet_link"] = course_info_df['Checklist Link'][0]
                st.session_state["drive"] = drive
                st.session_state["gc"] = gc
                
                # Set the langchain project name for langsmith
                os.environ["LANGCHAIN_PROJECT"] = get_short_name(st.session_state["course_name"]) + " " + sheet.id
                
                # Load previously completed steps from Agent logs
                load_completed_steps(sheet, step_name)

                st.success("Data loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading data: {e}")
                # Display the full stack trace
                st.text(traceback.format_exc())
    else:
        # If data is already loaded, simply confirm it to the user
        # Create two columns: left for info, right for the button
        col1, col2 = st.columns([3, 1])  # Adjust the ratio as needed

        # Full-width green info bar with default text size
        st.markdown(
            f"""
            <div style='background-color: #1b4636; color: #fff; padding: 1.2em 1em; border-radius: 12px; width: 100%; font-weight: 500; margin-bottom: 1.5em;'>
                Data already loaded for course: <b>{st.session_state['course_name']}</b>. Proceed below.
            </div>
            """,
            unsafe_allow_html=True
        )

        # Two columns for the buttons, placed horizontally on the same line
        button_col1, button_col2 = st.columns([1, 1])
        with button_col1:
            run_all_automated = st.button("Run All Automated Steps", type="primary")
        with button_col2:
            run_in_background = st.button("Run the Agent in Background", type="primary")

        agent_code_for_state = AGENT_CODE_MAP.get(step_name, step_name.lower().replace(" ", "_"))
        background_link_key = f"background_job_link::{agent_code_for_state}"
        background_name_key = f"background_job_name::{agent_code_for_state}"
        if st.session_state.get(background_link_key):
            st.markdown(
                f"**Background job:** [{st.session_state[background_link_key]}]({st.session_state[background_link_key]})"
            )
        elif st.session_state.get(background_name_key):
            st.markdown(f"**Background job:** `{st.session_state[background_name_key]}`")

        # Admin exclusive features
        if 'role' in st.session_state: #and st.session_state['role'] == 'Admin':
            # Skip Manual Steps
            st.checkbox(label = "Skip Manual Steps", value = False, key = "skip_manual_step")
            
            # # Reset completed steps option
            # if st.button("Reset All Completed Steps"):
            #     for section in pipeline_sections:
            #         for step in section["steps"]:
            #             step_key = f"{step['name']}_done"
            #             st.session_state[step_key] = False
            #     # Clear the sheet log
            #     clear_agent_logs(st.session_state["sheet"], st.session_state["agent_name"])
            #     # st.rerun()
            
            # Add "Run All Automated Steps" button
            # if st.button("Run All Automated Steps", type="primary"):
            #     st.session_state["automation_in_progress"] = True
            #     run_all_automated_steps(pipeline_sections)

            if run_all_automated:
                st.session_state["automation_in_progress"] = True

                ctx = _get_tracking_context()
                if ctx["gc"]:
                    track_run_all_start(
                        ctx["gc"], ctx["user_email"], ctx["agent_name"],
                        ctx["course_name"], ctx["sheet_link"],
                    )
                run_all_automated_steps(pipeline_sections, llm_pricing)

            if run_in_background:
                ctx = _get_tracking_context()
                if ctx["gc"]:
                    track_run_in_background_start(
                        ctx["gc"], ctx["user_email"], ctx["agent_name"],
                        ctx["course_name"], ctx["sheet_link"],
                    )
                agent_code = AGENT_CODE_MAP.get(step_name, step_name.lower().replace(" ", "_"))
                sheet_link = st.session_state.get("sheet_link")
                folder_id = st.session_state.get("root_folder_id")
                if not sheet_link or not folder_id:
                    st.error("Sheet link or Drive folder ID missing. Please reload data.")
                else:
                    # Ensure environment variables are loaded
                    load_dotenv()
                    
                    # Share the sheet with service account if needed
                    if "oauth_credentials" in st.session_state and "sheet" in st.session_state:
                        try:
                            service_account_email = get_service_account_email()
                            if service_account_email:
                                creds = st.session_state["oauth_credentials"]
                                sheet = st.session_state["sheet"]
                                
                                # Share the sheet with service account
                                share_sheet_with_service_account(sheet, service_account_email, creds)
                        except Exception as e:
                            pass
                    
                    user_email = st.session_state.get("user_email", "") or ""
                    # Collect current toggle values to forward to background job
                    toggle_values = {}
                    if top_toggles:
                        for toggle in top_toggles:
                            tkey = toggle["key"]
                            toggle_values[tkey] = st.session_state.get(tkey, toggle.get("default", False))
                    cmd = [
                        sys.executable,
                        "launch_agents_via_sdk.py",
                        "--sheet_link",
                        sheet_link,
                        "--drive_folder_id",
                        folder_id,
                        "--agent_name",
                        agent_code,
                        "--user_email",
                        user_email,
                    ]
                    if toggle_values:
                        cmd.extend(["--toggles", json.dumps(toggle_values)])
                    with st.spinner("Submitting background job..."):
                        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                        logs = ""
                        job_link = None
                        job_name = None
                        link_re = re.compile(r"^\[JOB_LINK\]\s+(?P<link>\S+)\s*$")
                        name_re = re.compile(r"^\[JOB_NAME\]\s+(?P<name>.+?)\s*$")
                        start = time.time()
                        while True:
                            if process.stdout is None:
                                break
                            line = process.stdout.readline()
                            if not line:
                                if process.poll() is not None:
                                    break
                                if time.time() - start > 10:
                                    break
                                time.sleep(0.1)
                                continue
                            logs += line
                            m = link_re.match(line.strip())
                            if m:
                                job_link = m.group("link")
                                break
                            m2 = name_re.match(line.strip())
                            if m2:
                                job_name = m2.group("name")
                        try:
                            if process.stdout is not None:
                                process.stdout.close()
                        except Exception:
                            pass
                        try:
                            process.wait(timeout=2)
                        except Exception:
                            pass

                    if job_link:
                        st.session_state[background_link_key] = job_link
                        st.session_state.pop(background_name_key, None)
                        st.success("Background job submitted.")
                        st.markdown(f"**Background job:** [{job_link}]({job_link})")
                    elif job_name:
                        st.session_state[background_name_key] = job_name
                        st.session_state.pop(background_link_key, None)
                        st.success("Background job submitted.")
                        st.markdown(f"**Background job:** `{job_name}`")
                        with st.expander("Launcher output (no job link found)", expanded=False):
                            st.code(logs[-5000:] if len(logs) > 5000 else logs)
                    else:
                        st.error("Background job submission did not return a job link.")
                        with st.expander("Launcher output", expanded=True):
                            st.code(logs[-5000:] if len(logs) > 5000 else logs)

    # --- 4) Display pipeline steps in nested sections ---
    if "sheet" in st.session_state:
        step_global_count = 1  # So we can label steps 1,2,3 across sections

        # Optional instructions shown above all sections (e.g. "before you run" setup)
        if top_instructions:
            st.info(top_instructions)

        # Optional toggles shown above all sections
        if top_toggles:
            for toggle in top_toggles:
                toggle_key = toggle["key"]
                if toggle_key not in st.session_state:
                    st.session_state[toggle_key] = toggle.get("default", False)
                st.toggle(toggle["label"], key=toggle_key)

        # Filter sections that have at least one visible step
        visible_sections = [
            section for section in pipeline_sections
            if any(
                not (st.session_state.get("outline_finalized", False) and step.get("hide_if_final_outline", False))
                for step in section["steps"]
            )
        ]

        # Use only visible sections for correct numbering
        for section_idx, section in enumerate(visible_sections, start=1):

            # Filter out steps that should be hidden
            visible_steps = [
                step for step in section["steps"]
                if not (st.session_state.get("outline_finalized", False) and step.get("hide_if_final_outline", False))
            ]

            # Skip section if no visible steps remain
            if not visible_steps:
                continue

            with st.container(border=True):
                st.header(f"Section {section_idx}: {section['section_name'].split(':', 1)[1].strip() if ':' in section['section_name'] else section['section_name']}", divider=True)
                for step in visible_steps:
                    step_key = f"{step['name']}_done"
                    
                    # Check if dependencies are satisfied
                    dependencies_satisfied = all(
                        st.session_state.get(f"{dep}_done", False) or (
                            st.session_state.get("outline_finalized", False) and
                            any(dep == s["name"] and s.get("hide_if_final_outline", False) for sec in pipeline_sections for s in sec["steps"])
                        )
                        for dep in step["depends_on"]
                    )
                    

                    # # Code to hide if step not ready.
                    # if not dependencies_satisfied:
                    #     continue

                    # Manual step formatting
                    is_manual = step.get("is_manual_step", False)
                    manual_icon = "👨‍💻" if is_manual else ""

                    # Expander 
                    with st.expander(label = f"Step {step_global_count}: {step['name']}", expanded = (not st.session_state[step_key] and dependencies_satisfied)):
                        # Subheader 
                        st.subheader(f"{manual_icon}Step {step_global_count}: {step['name']}")
                        step_global_count += 1

                        # Always show estimated time
                        st.write(f"**Estimated Time:** {step.get('estimated_time', 'N/A')}")

                        # Add video link if provided (Streamlit-styled)
                        if 'video_link' in step:
                            st.markdown(f"""<div style='margin: 0.5rem 0;'>
                                <a href='{step['video_link']}' target='_blank' 
                                   style='color: #FF4B4B; font-size: 0.9em; text-decoration: none; 
                                          display: inline-flex; align-items: center; gap: 4px;'>
                                    <span>📺</span>
                                    <span style='border-bottom: 1px solid rgba(255,75,75,0.2);'>Click here - Video Guide</span>
                                </a>
                            </div>""", unsafe_allow_html=True)

                        if not dependencies_satisfied:
                            # If dependencies are not done, show a message & skip
                            missing_steps = [
                                dep for dep in step["depends_on"]
                                if not st.session_state.get(f"{dep}_done", False)
                            ]
                            missing_list = ", ".join(missing_steps)
                            st.warning(f"Waiting on these steps to be done first: {missing_list}")
                            continue

                        if not st.session_state[step_key]:
                            # This step is not done yet

                            # If this step has pre-exec setup, show instructions so user knows what to do before running
                            if "pre_exec_func" in step and "pre_exec_instructions" in step:
                                st.info(step["pre_exec_instructions"])

                            # If we have a description, only show it while the user can act on the step
                            if "description" in step:
                                st.info(step["description"])

                            if "instructions" in step:
                                for instruction in step["instructions"]:
                                    st.write(instruction)
                                button_name = f"Confirm {step['name']}"
                                button_type = "primary"
                            else:
                                button_name = f"Run {step['name']}"
                                button_type = "secondary"

                            # NEW CODE: Run pre-execution function if it exists and hasn't been run yet
                            if "pre_exec_func" in step:
                                pre_exec_key = f"{step['name']}_pre_executed"
                                # Check if we should always run the pre_exec function or only if it hasn't been run yet
                                always_run = step.get("pre_exec_always_run", False)
                                if always_run or not st.session_state.get(pre_exec_key, False):
                                    try:
                                        # Gather pre-execution arguments from session_state
                                        pre_kwargs = {}
                                        if "pre_exec_args" in step:
                                            for arg_name, session_key in step["pre_exec_args"].items():
                                                if isinstance(session_key, str) and session_key in st.session_state:
                                                    pre_kwargs[arg_name] = st.session_state[session_key]
                                                else:
                                                    pre_kwargs[arg_name] = session_key
                                        
                                        # Run the pre-execution function
                                        with st.spinner(f"Loading preview data for {step['name']}..."):
                                            step["pre_exec_func"](**pre_kwargs)
                                        
                                        # Mark pre-execution as done
                                        st.session_state[pre_exec_key] = True
                                    except Exception as e:
                                        st.error(f"Error in pre-execution for {step['name']}: {e}")
                                        st.text(traceback.format_exc())

                            if st.button(button_name, type=button_type, key=f"btn_{step['name']}"):
                                ctx = _get_tracking_context()
                                _start_time = time.perf_counter()
                                try:
                                    # Add current step to session state
                                    st.session_state["current_step"] = step["name"]
                                    _set_step_context(st.session_state.get("agent_name", ""), step["name"])
                                    if llm_pricing is not None and _is_llm_step(step):
                                        _record_step_token_offset(step["name"])

                                    # Gather actual arguments from session_state
                                    kwargs = {}
                                    for arg_name, session_key in step["args"].items():
                                        # If the session_key is a string that matches a valid session_state key, retrieve it
                                        if isinstance(session_key, str) and session_key in st.session_state:
                                            kwargs[arg_name] = st.session_state[session_key]
                                        else:
                                            # or if it's a literal / direct value, pass it through
                                            kwargs[arg_name] = session_key

                                    # Track step start
                                    if ctx["gc"]:
                                        track_step_start(
                                            ctx["gc"], ctx["user_email"], ctx["agent_name"],
                                            step["name"], ctx["course_name"], ctx["sheet_link"],
                                            run_mode="manual",
                                        )

                                    if "instructions" in step:
                                        # If it is a manual input type function
                                        start_perf = time.perf_counter()
                                        start_time = datetime.now()
                                        response = step["func"](**kwargs)
                                        end_time = datetime.now()
                                        duration_seconds = time.perf_counter() - start_perf
                                        if response:
                                            st.session_state[step_key] = True
                                            # Log the completed step
                                            log_completed_step(st.session_state["sheet"], st.session_state["agent_name"], step["name"])

                                            if llm_pricing is not None:
                                                _record_step_metrics(step, duration_seconds, start_time, end_time, llm_pricing)

                                            if ctx["gc"]:
                                                track_step_complete(
                                                    ctx["gc"], ctx["user_email"], ctx["agent_name"],
                                                    step["name"], ctx["course_name"], ctx["sheet_link"],
                                                    duration_seconds=time.perf_counter() - _start_time,
                                                    run_mode="manual",
                                                )

                                            st.success(f"{step['name']} completed!")

                                            # Only continue automation if it was explicitly triggered
                                            if st.session_state.get("automation_in_progress", False):
                                                run_all_automated_steps(pipeline_sections, llm_pricing)
                                                st.rerun()
                                            st.rerun()

                                        else:
                                            st.warning(f"{step['name']} not completed!")
                                    else:
                                        # Run the actual function
                                        start_perf = time.perf_counter()
                                        start_time = datetime.now()
                                        step["func"](**kwargs)
                                        end_time = datetime.now()
                                        duration_seconds = time.perf_counter() - start_perf
                                        st.session_state[step_key] = True
                                        # Log the completed step
                                        log_completed_step(st.session_state["sheet"], st.session_state["agent_name"], step["name"])

                                        if llm_pricing is not None:
                                            _record_step_metrics(step, duration_seconds, start_time, end_time, llm_pricing)

                                        if ctx["gc"]:
                                            track_step_complete(
                                                ctx["gc"], ctx["user_email"], ctx["agent_name"],
                                                step["name"], ctx["course_name"], ctx["sheet_link"],
                                                duration_seconds=time.perf_counter() - _start_time,
                                                run_mode="manual",
                                            )

                                        st.success(f"{step['name']} completed!")
                                        st.rerun()
                                except Exception as e:
                                    if ctx["gc"]:
                                        track_step_error(
                                            ctx["gc"], ctx["user_email"], ctx["agent_name"],
                                            step["name"], ctx["course_name"], ctx["sheet_link"],
                                            error_message=str(e)[:500],
                                            run_mode="manual",
                                        )
                                    st.error(f"Error running {step['name']}: {e}")
                                    st.text(traceback.format_exc())
                                finally:
                                    _clear_step_context()
                        else:
                            # st.write(f"{step['name']}: **Done**")
                            st.write("**Status:** Done")
                            metrics = st.session_state.get("step_metrics", {}).get(step["name"])
                            if metrics:
                                st.write(f"**Time Taken:** {_format_duration(metrics.get('duration_seconds'))}")
                                if metrics.get("cost") is not None:
                                    currency = "$"
                                    if llm_pricing:
                                        currency = llm_pricing.get("currency", "$")
                                    st.write(f"**Total Cost:** {_format_cost(metrics['cost'], currency)}")

                            # Add delete button for completed steps
                            if st.button("Delete Step", type="secondary", key=f"delete_{step['name']}"):
                                try:
                                    # Get all dependent steps
                                    affected_steps = get_dependent_steps(pipeline_sections, step["name"])
                                    
                                    affected_list = ", ".join(affected_steps)
                                    st.warning(f"Deleting this step will also delete these dependent steps: {affected_list}")
                                    # if st.button("Confirm Delete", type="secondary", key=f"confirm_delete_{step['name']}"):
                                    delete_steps(st.session_state["sheet"], st.session_state["agent_name"], 
                                                affected_steps, pipeline_sections)
                                    st.success(f"Deleted step {step['name']} and its dependencies!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error deleting {step['name']}: {e}")
                                    st.text(traceback.format_exc())

                            # If this is the very last step in the entire pipeline, celebrate
                            if (
                                section_idx == len(visible_sections)
                                and step == visible_steps[-1]
                            ):
                                st.balloons()
                                st.toast(
                                    f"You have successfully generated the {step_name}!",
                                    icon=":material/done_all:",
                                )

    # Debug
    # st.write(st.session_state)


def get_dependent_steps(pipeline_sections, step_name):
    """
    Find all steps that depend on the given step (directly or indirectly).
    Returns a list of step names including the starting step.
    Only includes steps that are marked as done in the session state.
    Steps are returned in reverse dependency order (most dependent first).
    """
    all_steps = {}
    # First, build a dictionary of all steps and their dependencies
    for section in pipeline_sections:
        for step in section["steps"]:
            all_steps[step["name"]] = step["depends_on"]
    
    # Initialize with the starting step
    dependent_steps = []
    
    # Function to recursively find dependent steps
    def find_dependents(step_to_check):
        for current_step, dependencies in all_steps.items():
            # Only include steps that are marked as done
            if step_to_check in dependencies and current_step not in dependent_steps and st.session_state.get(f"{current_step}_done", False):
                find_dependents(current_step)
                dependent_steps.append(current_step)
    
    # Start the recursive search
    find_dependents(step_name)
    
    # Add the starting step at the end
    dependent_steps.append(step_name)
    
    return dependent_steps


def delete_steps(sheet, agent_name, step_names, pipeline_sections):
    """
    Delete the specified steps from the session state and the logs.
    Also executes any delete functions associated with the steps.
    
    Note: Only steps that are marked as done will be included in the deletion process.
    This is ensured by the get_dependent_steps function which filters out steps
    that are not marked as done.
    """
    # Get the Agent logs worksheet
    worksheet, df = get_sheet_data_and_df(sheet, "Agent logs")
    
    # Filter logs to remove the specified steps for this agent
    new_df = df[~((df["Agent Name"] == agent_name) & (df["Step Name"].isin(step_names)))]
    
    # Clear the worksheet
    worksheet.clear()
    
    # Add header row
    # worksheet.append_row(["Agent Name", "Step Name", "Timestamp"])
    
    # Add remaining logs back
    if not new_df.empty:
        save_to_sheet(worksheet = worksheet, df = new_df)
    
    # Initialize the progress tracker
    progress = SmartProgressBar(total_tasks = len(step_names), description = "Percent complete")

    # Reset session state for the deleted steps
    for step_name in step_names:
        st.session_state[f"{step_name}_done"] = False
        pre_exec_key = f"{step_name}_pre_executed"
        if pre_exec_key in st.session_state:
            st.session_state[pre_exec_key] = False
        if "step_metrics" in st.session_state:
            st.session_state["step_metrics"].pop(step_name, None)
        if "step_token_offsets" in st.session_state:
            st.session_state["step_token_offsets"].pop(step_name, None)
        
        # Execute the delete function if it exists
        for section in pipeline_sections:
            for step in section["steps"]:
                if step["name"] == step_name and "delete_func" in step:
                    # try:
                    # Gather delete function arguments
                    delete_kwargs = {}
                    if "delete_args" in step:
                        for arg_name, session_key in step["delete_args"].items():
                            if isinstance(session_key, str) and session_key in st.session_state:
                                delete_kwargs[arg_name] = st.session_state[session_key]
                            else:
                                delete_kwargs[arg_name] = session_key
                    
                    # Execute the delete function
                    with st.spinner(text = f"Deleting {step_name}...", show_time = True):
                        step["delete_func"](**delete_kwargs)
                    # except Exception as e:
                    #     st.error(f"Error in delete function for {step_name}: {e}")
                    #     st.text(traceback.format_exc())
        
        # Update progress
        progress.update()



def run_all_automated_steps(pipeline_sections, llm_pricing: dict | None = None):
    """Run all steps in the pipeline that have their dependencies satisfied.
    - If skip_manual_step is checked - run all steps including manual ones.
    - If unchecked - stop execution when a manual step is reached.
    """
    progress_made = True
    
    # Keep iterating as long as we're making progress
    while progress_made:
        progress_made = False
        
        step_global_count = 0
        # Go through all steps in all sections
        for section in pipeline_sections:
            for step in section["steps"]:

                #Skip step if it's hidden due to finalized outline
                if st.session_state.get("outline_finalized", False) and step.get("hide_if_final_outline", False):
                    continue

                step_key = f"{step['name']}_done"
                step_global_count += 1

                # Skip if already done
                if st.session_state.get(step_key, False):
                    continue

                # Check if dependencies are satisfied
                dependencies_satisfied = all(
                    st.session_state.get(f"{dep}_done", False) or (
                        st.session_state.get("outline_finalized", False) and
                        any(dep == s["name"] and s.get("hide_if_final_outline", False) for sec in pipeline_sections for s in sec["steps"])
                    )
                    for dep in step["depends_on"]
                )                    
            
                if not dependencies_satisfied:
                    continue

                # Check if this is a manual step
                is_manual = "instructions" in step

                # Stop at manual steps unless "skip_manual_step" is checked
                if is_manual and not st.session_state.get("skip_manual_step", False):
                    st.info(f" Paused at manual step: **{step['name']}**. Please complete it manually to continue.")
                    return  # Exit early, waiting for manual confirmation

                if dependencies_satisfied:
                    ctx = _get_tracking_context()
                    _start_time = time.perf_counter()
                    try:
                        with st.spinner(text = f"Running: Step {step_global_count}. {step['name']}...", show_time = True):
                            # Add current step to session state
                            st.session_state["current_step"] = step["name"]
                            _set_step_context(st.session_state.get("agent_name", ""), step["name"])
                            if llm_pricing is not None and _is_llm_step(step):
                                _record_step_token_offset(step["name"])

                            # Gather actual arguments from session_state
                            kwargs = {}
                            for arg_name, session_key in step["args"].items():
                                # If the session_key is a string that matches a valid session_state key, retrieve it
                                if isinstance(session_key, str) and session_key in st.session_state:
                                    kwargs[arg_name] = st.session_state[session_key]
                                else:
                                    # or if it's a literal / direct value, pass it through
                                    kwargs[arg_name] = session_key

                            # Track step start
                            if ctx["gc"]:
                                track_step_start(
                                    ctx["gc"], ctx["user_email"], ctx["agent_name"],
                                    step["name"], ctx["course_name"], ctx["sheet_link"],
                                    run_mode="automated",
                                )

                            # Run the function
                            start_perf = time.perf_counter()
                            start_time = datetime.now()
                            step["func"](**kwargs)
                            end_time = datetime.now()
                            duration_seconds = time.perf_counter() - start_perf
                            st.session_state[step_key] = True
                            # Log the completed step
                            log_completed_step(st.session_state["sheet"], st.session_state["agent_name"], step["name"])

                            if llm_pricing is not None:
                                _record_step_metrics(step, duration_seconds, start_time, end_time, llm_pricing)

                            if ctx["gc"]:
                                track_step_complete(
                                    ctx["gc"], ctx["user_email"], ctx["agent_name"],
                                    step["name"], ctx["course_name"], ctx["sheet_link"],
                                    duration_seconds=time.perf_counter() - _start_time,
                                    run_mode="automated",
                                )

                            progress_made = True
                            st.success(f"Auto-run: Step {step_global_count}. {step['name']} completed!")

                    except Exception as e:
                        if ctx["gc"]:
                            track_step_error(
                                ctx["gc"], ctx["user_email"], ctx["agent_name"],
                                step["name"], ctx["course_name"], ctx["sheet_link"],
                                error_message=str(e)[:500],
                                run_mode="automated",
                            )
                        st.error(f"Error auto-running Step {step_global_count}. {step['name']}: {e}")
                        st.text(traceback.format_exc())
                        return
                    finally:
                        _clear_step_context()
    st.session_state["automation_in_progress"] = False


def log_completed_step(sheet, agent_name, step_name):
    """Log a completed step to the Agent logs worksheet."""
    try:
        worksheet, df = get_sheet_data_and_df(sheet, "Agent logs")
        
        # Create a timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create a new log entry
        log_data = [agent_name, step_name, timestamp]
        
        # Add the log entry to the worksheet
        worksheet.append_row(log_data)
    except Exception as e:
        st.error(f"Error logging completed step: {e}")


def load_completed_steps(sheet, agent_name):
    """Load previously completed steps from the Agent logs worksheet."""
    try:
        worksheet, df = create_or_read_worksheet(sheet, "Agent logs")
        
        # If the dataframe is empty (only has headers), return
        if df.empty:
            # Add header row
            worksheet.append_row(["Agent Name", "Step Name", "Timestamp"])
            format_worksheet(worksheet = worksheet)
            return
        
        # Filter logs for the current agent
        agent_logs = df[df["Agent Name"] == agent_name]
        
        # Mark each logged step as completed in the session state
        for _, row in agent_logs.iterrows():
            step_key = f"{row['Step Name']}_done"
            st.session_state[step_key] = True
                
    except Exception as e:
        st.error(f"Error loading completed steps: {e}")


# def clear_agent_logs(sheet, agent_name):
#     """Clear logs for a specific agent from the Agent logs worksheet."""
#     try:
#         # Get the Agent logs worksheet
#         worksheet, df = get_sheet_data_and_df(sheet, "Agent logs")
        
#         # Filter out logs for the current agent
#         new_df = df[df["Agent Name"] != agent_name]
        
#         # Clear the worksheet
#         worksheet.clear()
        
#         # Add header row
#         worksheet.append_row(["Agent Name", "Step Name", "Timestamp"])
        
#         # Add remaining logs back
#         if not new_df.empty:
#             for _, row in new_df.iterrows():
#                 worksheet.append_row(row.tolist())
        
#         st.success("All steps have been reset!")

#     except Exception as e:
#         st.error(f"Error clearing agent logs: {e}")
