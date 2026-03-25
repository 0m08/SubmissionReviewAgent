import streamlit as st
import os
import sys
import json
from dotenv import load_dotenv

# Add directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))            # .../automated
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))  # workspace root
sys.path.insert(0, current_dir)   # Add current directory for automated_voiceover_reviewer
sys.path.insert(0, parent_dir)    # Add workspace root for agents/services packages

# Load local modules
from automated_voiceover_reviewer import run_automation
from llm_call_tracker import tracker as llm_tracker

load_dotenv()

st.set_page_config(page_title="Voiceover Reviewer Automation", layout="wide")

st.title("🎙️ Voiceover Reviewer Automation")
st.caption(
    "Batch process Technical Accuracy and Copyright transformations for entire Google Sheets. "
    "Transformed image links are written back into the **same source tab** under the `final_graphics` column."
)

# Check if user is authenticated (using session state from main app or local login)
if "gc" not in st.session_state or "drive" not in st.session_state:
    # Attempt local login if running standalone for development
    try:
        from services.drive_service import login_with_service_account
        import gspread

        service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        if service_account_json:
            with st.spinner("Authenticating with service account..."):
                gauth = login_with_service_account(json_str=service_account_json)
                from pydrive2.drive import GoogleDrive
                st.session_state["drive"] = GoogleDrive(gauth)
                st.session_state["gc"] = gspread.service_account_from_dict(json.loads(service_account_json))
                st.success("✅ Authenticated via Service Account")
        else:
            st.error("❌ Authentication required")
            st.info("Please login from the main application or configure GOOGLE_SERVICE_ACCOUNT_JSON")
            st.stop()
    except Exception as e:
        st.error(f"❌ Authentication failed: {e}")
        st.stop()

# --- UI COMPONENTS ---
st.subheader("📊 Execution Settings")

col1, col2 = st.columns(2)

with col1:
    sheet_url = st.text_input(
        "Google Sheets URL",
        placeholder="https://docs.google.com/spreadsheets/d/...",
        value=st.session_state.get("last_sheet_url", "")
    )
    source_tab = st.text_input(
        "Source Tab Name",
        "Slide Chunksb",
        help="Results will be written back into this same tab under the `final_graphics` column."
    )

with col2:
    drive_folder = st.text_input("Output Drive Folder Name", "Voiceover Reviewer Automation")
    image_size   = st.selectbox("Image Quality", ["1K", "2K", "4K"], index=0)
    skip_filled  = st.checkbox("Skip already-filled rows", value=True, help="Only process rows where the 'final_graphics' column is empty.")

st.success(
    "⚡ **Fully Automatic Parallelism** — The pipeline scans every row in the sheet, "
    "counts the total number of valid subsegments, and launches **exactly one dedicated worker "
    "per subsegment**. If the sheet contains 50 subsegments across all rows, 50 workers run "
    "simultaneously. No manual worker configuration is required or available.",
    icon="🤖"
)

st.info(
    "💡 **Output location:** The `final_graphics` column will be added (or updated) directly "
    "in the source tab you specify above. No separate output tab is created.",
    icon="📌"
)

st.divider()

# --- RUN LOGIC ---

if st.button("🚀 Start Automation Pipeline", type="primary", width='stretch'):
    if not sheet_url:
        st.error("❌ Please provide a Google Sheets URL")
    else:
        st.session_state["last_sheet_url"] = sheet_url
        st.session_state["pipeline_running"] = True

        try:
            status_placeholder = st.empty()
            with st.spinner("Executing Pipeline… Check terminal for detailed logs."):
                status_placeholder.info(
                    "⏳ Phase 1: Scanning rows and pre-creating Drive folders…\n"
                    "Phase 2 will launch one worker per subsegment automatically.\n"
                    "Phase 3 will write any remaining rows back to the sheet."
                )

                # Progress bar (updated via callback from run_automation)
                progress_bar  = st.progress(0.0, text="⏳ Waiting for Phase 1 to complete…")
                progress_text = st.empty()

                def update_progress(completed: int, total: int):
                    pct = completed / total if total else 1.0
                    progress_bar.progress(
                        pct,
                        text=f"⚡ Phase 2 in progress — {completed} / {total} subsegments complete"
                    )
                    progress_text.markdown(
                        f"`{completed}/{total}` subsegments processed · "
                        f"{'%.0f' % (pct * 100)}% complete"
                    )

                run_automation(
                    sheet_url          = sheet_url,
                    source_tab         = source_tab,
                    output_tab         = "",   # unused — kept for signature compatibility
                    output_folder_name = drive_folder,
                    gc                 = st.session_state["gc"],
                    drive              = st.session_state["drive"],
                    progress_callback  = update_progress,
                    skip_filled_rows   = skip_filled,
                )

                progress_bar.progress(1.0, text="✅ All subsegments processed — results written to sheet")
                progress_text.empty()

            st.session_state["pipeline_running"] = False
            st.success(
                f"✅ Automation Complete! Results written to the `final_graphics` column "
                f"in the **'{source_tab}'** tab."
            )
            st.balloons()

            st.markdown(f"[🔗 Open Google Sheet]({sheet_url})")

        except Exception as e:
            st.session_state["pipeline_running"] = False
            st.error(f"❌ Execution failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

st.divider()

# =============================================================================
# LIVE STATS PANEL
# =============================================================================

st.subheader("📈 Pipeline Stats")

snap = llm_tracker.snapshot()
has_data = snap["total_calls"] > 0

is_running = st.session_state.get("pipeline_running", False)
if is_running:
    st.info("🔄 Pipeline is running — stats update every 5 seconds. Refresh the page or wait for completion.")

llm_tracker.render_stats_panel()

col_r1, col_r2 = st.columns([1, 4])
with col_r1:
    if st.button("🔄 Refresh Stats"):
        st.rerun()

with col_r2:
    if has_data:
        if st.button("🗑️ Clear Stats"):
            llm_tracker.reset()
            st.rerun()

st.divider()
st.info("""
**Workflow Summary:**

1. **Auto-Parallel Execution (3 Phases)**:
   - **Phase 1 [Serial]**: Scans every row, counts every valid subsegment, pre-creates all
     Drive folders. Determines N = total subsegments.
   - **Phase 2 [Parallel]**: Launches exactly N worker threads — one per subsegment — all
     running simultaneously regardless of which row they belong to.
   - **Phase 3 [Serial]**: Aggregates each worker's result, rebuilds the text block per row,
     and writes the `final_graphics` column back to the sheet.
   - **Zero manual configuration** — worker count always equals subsegment count.

2. **Source sheet is the output sheet**:
   - A `final_graphics` column is created (or updated) automatically in the source tab.
   - Each row's reconstructed text keeps the original formatting — only the
     `Graphics to use:` link is swapped with the transformed Drive image link.
   - Subsegments with invalid links are kept verbatim; `(snapshot)` tags are removed.

3. **Drive Storage (for audit)**:
   - `Row_N_SlideTitle/` folders per slide row.
   - `Subsegment_N/` subfolders per image.
   - Progressive review images: `Reviewer_1_Round_1.png`, `Reviewer_2_Round_1.png`, `FINAL_Image.png`.

4. **Intelligent Review Pipeline**:
   - Accuracy Review → Copyright Audit → Smart Rollback → Dynamic Stopping.

5. **Full Transparency**: Every review round is saved to Drive for audit and analysis.
""")
