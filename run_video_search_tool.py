import os
import json
import base64
import gspread
import streamlit as st
from dotenv import load_dotenv
from pydrive2.drive import GoogleDrive
from services.sheets_service import get_sheet_data_and_df
from services.drive_service import login_with_service_account
from agents.course_outline.video_search_tool.update_video_vectorstore import update_video_vectorstore
from agents.course_outline.video_search_tool.video_retriever import video_retriever, video_search_retriever_agent
from agents.course_outline.video_search_tool.create_video_vectorstore import create_video_vectorstore, chroma_db_exists

llm_model = st.session_state.get("llm_model", "gemini_2_flash") or "gemini_2_flash"
# --------------------- Auth --------------------- #
load_dotenv()
key_bytes = base64.b64decode(os.environ["GDRIVE_SA_B64"])
sa_json = key_bytes.decode()
sa_dict = json.loads(sa_json)

gauth = login_with_service_account(json_str=sa_json)
gauth.ServiceAuth()
drive = GoogleDrive(gauth)
gc = gspread.service_account_from_dict(sa_dict)

st.session_state["drive"] = drive
st.session_state["gc"] = gc

# --------------------- App Header --------------------- #
st.markdown("## SkillCat HVAC Video Search Tool")
st.markdown("Use this tool to create, update, and search for HVAC videos based on text queries.")

# --------------------- Roles & Task Selection --------------------- #
if "role" not in st.session_state:
    st.session_state.role = None

# Use effective role (impersonated role if set, otherwise actual role)
role = st.session_state.get("impersonated_role", st.session_state.get("role"))
task_options = []

if role == "Admin":
    task_options = ["Create Vectorstore", "Update Vectorstore", "Search Videos"]
elif role == "Managers":
    task_options = ["Update Vectorstore", "Search Videos"]
elif role == "Visual Designer":
    task_options = ["Search Videos"]
else:
    task_options = ["Search Videos"]
    # st.warning("Your role does not have access to any tasks.")

task = st.selectbox("Choose a task:", task_options) if task_options else None

# --------------------- Google Sheet Input --------------------- #
sheet = None
if task in ["Create Vectorstore", "Update Vectorstore"]:
    st.subheader("Provide Google Sheet")
    sheet_url = st.text_input("Enter your Google Sheet URL:")
    if sheet_url:
        try:
            sheet = gc.open_by_url(sheet_url)
            st.session_state["sheet"] = sheet
            st.success("Sheet loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load sheet: {e}")
    sheet = st.session_state.get("sheet")

central_folder_id = '1cUBmd1H1hBHSLohnAF68VJTEU9XcFXFK'

# --------------------- Task: Create Vectorstore --------------------- #
if task == "Create Vectorstore":
    if not sheet:
        st.info("Load a Google Sheet above to continue.")
    else:
        db_exists = chroma_db_exists(drive, central_folder_id)
        if db_exists:
            st.success("Video Vectorstore already exists in Drive.")
        else:
            if st.button("Create Video Vectorstore"):
                with st.spinner("⏳ Building and uploading video vectorstore..."):
                    create_video_vectorstore(sheet, drive)
                st.success("Video vectorstore built and uploaded successfully!")

# --------------------- Task: Update Vectorstore --------------------- #
elif task == "Update Vectorstore":
    if not sheet:
        st.info("Load a Google Sheet above to continue.")
    else:
        if st.button("Check and Update Video Vectorstore"):
            update_required = False

        # Directly get the specific worksheet
        try:
            sheet_name = "HVAC School Video Chunks"
            worksheet = sheet.worksheet(sheet_name)
            _, df = get_sheet_data_and_df(sheet, sheet_name)

            # Ensure 'Video Description' exists
            if 'Video Description' in df.columns:
                # Check if vectorstore columns exist
                if 'vectorized' not in df.columns or 'embedding_ts' not in df.columns:
                    update_required = True
                else:
                    # Check if there are any rows that need vectorization
                    mask = (df['vectorized'] != 'TRUE') & df['Video Description'].str.strip().astype(bool)
                    if mask.any():
                        update_required = True
        except Exception as e:
            st.error(f"Error accessing worksheet '{sheet_name}': {e}")
            update_required = False

        # Update vectorstore if needed
        if update_required:
            with st.spinner("Updating Video Vectorstore... This may take a few minutes."):
                update_video_vectorstore(sheet, drive)
            st.success("Video Vectorstore updated.")
        else:
            st.info("No updates needed. Video Vectorstore is up to date.")

# --------------------- Task: Search Videos --------------------- #
elif task == "Search Videos":
    st.markdown("#### Search HVAC Videos")
    query = st.text_input("Enter search query (e.g., 'refrigeration cycle')", placeholder="Type your query here")
    num_results = st.number_input("Number of videos to retrieve", min_value=1, max_value=20, value=5, step=1)

    filters = {}
    with st.expander("Apply Filters (Optional)"):
        st.caption("Filter videos by channel")
        
        # Channel filter
        channel_filter = st.selectbox(
            "Channel",
            options=["All", "HVAC School", "LOVE2HVAC with Ty Branaman"]
        )
        
        if channel_filter and channel_filter != "All":
            filters["channel"] = channel_filter

    use_agent = st.toggle("Agent Mode", value=False)

    if st.button("Run Video Search"):
        if not query:
            st.warning("Please enter a search query.")
        else:
            with st.spinner("Searching videos..."):
                if use_agent:
                    results = video_search_retriever_agent(
                        query=query,
                        drive=drive,
                        llm=llm_model,
                        k=num_results,
                        filters=filters

                    )
                else:
                    results = video_retriever(
                        query=query,
                        drive=drive,
                        k=num_results,
                        filters=filters
                        )

            # Display Results
            if results:
                st.subheader(f"Top {len(results)} Videos")

                for idx, video_data in enumerate(results):
                    video_title = video_data.get("video_title", f"Video {idx+1}")
                    video_id    = video_data.get("video_id")
                    start_sec   = int(video_data.get("start_time") or 0)
                    end_sec     = video_data.get("end_time")
                    end_sec     = int(end_sec) if end_sec is not None else None

                    # Build YouTube embed URL
                    base = f"https://www.youtube.com/embed/{video_id}"
                    params = [f"start={start_sec}"]
                    if end_sec is not None:
                        params.append(f"end={end_sec}")
                    params += ["controls=0", "modestbranding=1", "rel=0", "fs=0", "disablekb=1", "playsinline=1"]
                    iframe_url = base + "?" + "&".join(params)

                    # Render iframe directly
                    st.markdown(
                        f"""
                        <iframe src="{iframe_url}" width="640" height="360" frameborder="0" 
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                        allowfullscreen></iframe>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Clickable title (to the same embed URL)
                    st.markdown(f"[{video_title}]({iframe_url})", unsafe_allow_html=True)
                    st.caption(f"Channel: {video_data.get('channel','Unknown')}")

                    # Collapsible transcript
                    with st.expander("Show Transcript"):
                        st.write(video_data.get("transcript","Transcript not available."))

                    st.markdown("---")


            else:
                st.warning("No videos found.")