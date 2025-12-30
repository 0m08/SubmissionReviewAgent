import streamlit as st
import time
from services.drive_service import (
    login_with_oauth2,  # kept for legacy CLI/local usage
    get_google_oauth_authorization_url,
    exchange_code_for_credentials,
    init_clients_from_credentials,
)
from pydrive2.drive import GoogleDrive
import gspread
import base64
import json
import os
from dotenv import load_dotenv
from streamlit_clickable_images import clickable_images
from utils.role_utils import get_user_info, get_user_pages
from config.logging_config import get_logger, setup_logging
from mcp_ui_app import mcp_ui_page

# from jira import JIRA

# load_dotenv()
# Helper to load image as base64
def load_image_as_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    return "data:image/png;base64," + base64.b64encode(data).decode()

# Initialize logging early so any downstream imports inherit the handlers.
setup_logging()
logger = get_logger(__name__)

# 1) Initialize Session State for user role
if "role" not in st.session_state:
    st.session_state.role = None


# skillcat_logo_image = Image.open("assets/SkillCat-Logo.png")
# skillcat_helmet_image = Image.open("assets/SkillCat-Helmet.png")

# st.logo(
#     image = skillcat_logo_image,
#     size = "medium",
#     # link = "https://www.skillcatapp.com/",
#     icon_image = skillcat_helmet_image
# )

#######################
# 2) Define "pages"
#######################

def login():
    """Direct Google OAuth authentication - no username/password required."""
    st.header("SkillCat AI Agents Ecosystem")
    
    # Handle OAuth callback first (when Google redirects back with ?code=...)
    load_dotenv()
    params = st.query_params
    oauth_client_id = os.getenv("OAUTH_CLIENT_ID")
    oauth_client_secret = os.getenv("OAUTH_CLIENT_SECRET")
    redirect_uri = (
        os.getenv("OAUTH_REDIRECT_URI")
        or os.getenv("OAUTH_REDIRECT_URI_LIGHTNING")
        or os.getenv("OAUTH_REDIRECT_URI_LOCAL")
        or "http://localhost:8501"
    )

    if "code" in params and oauth_client_id and oauth_client_secret:
        code = params.get("code")
        state_received = params.get("state")
        state_expected = st.session_state.get("oauth_state")

        if state_expected and state_received and state_received != state_expected:
            st.error("Authentication state mismatch. Please try logging in again.")
            st.query_params.clear()
            st.stop()

        try:
            creds = exchange_code_for_credentials(
                client_id=oauth_client_id,
                client_secret=oauth_client_secret,
                redirect_uri=redirect_uri,
                code=code,
            )
            gauth, drive, gc = init_clients_from_credentials(
                creds,
                client_id=oauth_client_id,
                client_secret=oauth_client_secret
            )

            about = drive.GetAbout()
            user_email = about.get('user', {}).get('emailAddress', '')

            if user_email:
                user_info = get_user_info(user_email)
                if user_info["is_authorized"]:
                    st.session_state["drive"] = drive
                    st.session_state["gc"] = gc
                    st.session_state["oauth_authenticated"] = True
                    st.session_state["role"] = user_info["role"]
                    st.session_state["user_email"] = user_email
                    st.session_state["user_pages"] = user_info["pages"]
                    st.session_state.pop("oauth_state", None)
                    st.query_params.clear()
                    st.rerun()
                else:
                    st.error(f"Access denied. Email {user_email} is not authorized to access this application.")
            else:
                st.error("Could not retrieve user email. Please try again.")
        except Exception as e:
            st.error(f"Authentication failed: {e}")
            st.stop()
    
    # role_choice = st.text_input("Enter your login name: ")
    # password = st.text_input("Enter the password: ")
    # if st.button("Log in"):
    #     if role_choice in authenticated_roles:
    #         if password == authenticated_roles[role_choice]:
    #             st.success("Logging in...")
    #             st.session_state.role = role_choice
    
    st.info("🔐 Click below to authenticate and access the AI Agents")

    # Load Google login button from assets folder
    image_path = os.path.join(
        os.path.dirname(__file__), "assets", "google-login-button.png"
    )
    img_b64 = load_image_as_base64(image_path)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        clicked = clickable_images(
            [img_b64], 
            titles=["Login with Google"],
            div_style={
                "display": "flex", 
                "justify-content": "center",
                "background-color": "white",
                "border": "2px solid #dadce0",
                "border-radius": "8px",
                "padding": "8px 16px",
                "box-shadow": "0 2px 4px rgba(0,0,0,0.1)"
            },
            img_style={
                "cursor": "pointer", 
                "height": "60px",
                "border-radius": "4px"
            },
        )

        if clicked == 0:  # Image clicked
            if oauth_client_id and oauth_client_secret:
                auth_url, state = get_google_oauth_authorization_url(
                    client_id=oauth_client_id,
                    client_secret=oauth_client_secret,
                    redirect_uri=redirect_uri,
                )
                st.session_state["oauth_state"] = state
                st.markdown(
                    f'<meta http-equiv="refresh" content="0; url={auth_url}">',
                    unsafe_allow_html=True,
                )
                st.stop()
            try:
                load_dotenv()

                # Check if OAuth credentials are available
                oauth_client_id = os.getenv("OAUTH_CLIENT_ID")
                oauth_client_secret = os.getenv("OAUTH_CLIENT_SECRET")

                if oauth_client_id and oauth_client_secret:
                    st.info("🔐 Logging you in...")

                    # OAuth authentication for Drive
                    gauth = login_with_oauth2(
                        client_id=oauth_client_id,
                        client_secret=oauth_client_secret,
                        credentials_file="credentials.json"
                    )
                    drive = GoogleDrive(gauth)

                    # Service account for Sheets
                    # key_bytes = base64.b64decode(os.environ["GDRIVE_SA_B64"])
                    # sa_json = key_bytes.decode()
                    # sa_dict = json.loads(sa_json)
                    # gc = gspread.service_account_from_dict(sa_dict)
                   
                    # OAuth authentication for Sheets (using same credentials as Drive)
                    gc = gspread.authorize(gauth.credentials)

                    # Get user email and role
                    try:
                        # Get user info from Google Drive API
                        about = drive.GetAbout()
                        user_email = about.get('user', {}).get('emailAddress', '')
                        
                        if user_email:
                            # Get user role and page access
                            user_info = get_user_info(user_email)
                            
                            if user_info["is_authorized"]:
                                # Store in session state
                                st.session_state["drive"] = drive
                                st.session_state["gc"] = gc
                                st.session_state["oauth_authenticated"] = True
                                st.session_state["role"] = user_info["role"]
                                st.session_state["user_email"] = user_email
                                st.session_state["user_pages"] = user_info["pages"]
                                
                                st.success(f"✅ Authentication successful! Welcome {user_info['role']} - {user_email}")
                                st.rerun()
                            else:
                                st.error(f"❌ Access denied. Email {user_email} is not authorized to access this application.")
                                st.session_state["oauth_authenticated"] = False
                        else:
                            st.error("❌ Could not retrieve user email. Please try again.")
                            st.session_state["oauth_authenticated"] = False
                    except Exception as e:
                        st.error(f"❌ Error getting user information: {str(e)}")
                        st.session_state["oauth_authenticated"] = False

                else:
                    st.error("❌ OAuth credentials not configured. Please contact administrator.")
                    st.session_state["oauth_authenticated"] = False

            except Exception as e:
                st.error(f"Authentication failed: {e}")
                st.session_state["oauth_authenticated"] = False
    
    # else:
    #     st.error("Failed to authenticate. Wrong passsword. Try again.")
    #     time.sleep(2)
    # else:
    #     st.error("Failed to authenticate. Wrong user name. Try again.")
    #     time.sleep(2)
    # st.rerun()


def logout():
    """Immediately logs the user out by clearing role and authentication."""
    # Delete all the items in Session state
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()


def list_of_agents():
    st.header("SkillCat AI Agents Ecosystem")
    st.write("Here's the list of agentic workflows:")
    st.markdown(
"""Worflow Name | Informational Course | Instructional Course | Practical Course
|- | - | - | - |
:material/toc: Course Outline | :material/check_box: Usable | :material/check_box: Usable | :material/check_box: Usable
:material/quick_reference_all: Research Notes | :material/check_box: Usable | :material/check_box: Usable | :material/check_box: Usable
:material/topic: Slide Chunks | :material/check_box: Usable | :material/check_box: Usable | :material/check_box: Usable
:material/image: Graphics Definition | :material/check_box_outline_blank: Usable | :material/check_box_outline_blank: Usable | :material/check_box_outline_blank: Usable
:material/quiz: Assessment | :material/check_box: Usable | :material/check_box: Usable | :material/check_box: Usable
"""
    )

    st.info(""":material/info: To run any of the above agentic workflows, navigate to the corresponding page from the left side panel""")

    # # Connect to Jira and fetch issue description
    # try:
    #     jira = JIRA(server=os.environ['JIRA_SITE'], 
    #                 basic_auth=(os.environ['JIRA_EMAIL'], os.environ['JIRA_API_TOKEN']))
        
    #     issue = jira.issue('SGP-2789', expand="renderedFields,names,schema")
    #     desc_html = issue.renderedFields.description
        
    #     st.markdown("### Jira Issue Description")
    #     st.markdown(desc_html, unsafe_allow_html=True)
    # except Exception as e:
    #     st.error(f"Error fetching Jira issue: {str(e)}")


# We can either define Page objects inline (pointing to .py files or callables)
# or just define them here. For simplicity, let's define some stubs as Page objects.

# --- Account pages ---
list_of_agents_page = st.Page(list_of_agents, title = "Home", icon = ":material/list:")
logout_page = st.Page(logout, title="Log out", icon=":material/logout:")
about_agents_page = st.Page("about_agents.py", title="About Agents", icon=":material/info:")
role_switch_page = st.Page("role_switch.py", title="Role Switch", icon=":material/swap_horiz:")

# --- Outline pages ---
template_sheet_setup_page = st.Page(
    "template_sheet_agent.py",
    title="Template Sheet Setup",
    icon=":material/content_copy:",
)

course_outline_page = st.Page(
    "course_outline.py",
    title="Course Outline",
    icon=":material/toc:",
    # Optional: default=(role == "Requester") or any logic
)

research_notes_page = st.Page(
    "research_notes.py",
    title="Research Notes",
    icon=":material/quick_reference_all:",
    # Optional: default=(role == "Requester") or any logic
)

slide_chunks_page = st.Page(
    "slide_chunks.py",
    title="Slide Chunks",
    icon=":material/topic:",
    # Optional: default=(role == "Requester") or any logic
)

graphics_definition_page = st.Page(
    "graphics_definition.py",
    title="Graphics Definiton",
    icon=":material/image:",
    # Optional: default=(role == "Requester") or any logic
)

graphics_definition_v2_page = st.Page(
    "graphics_definition_v2.py",
    title="Graphics Definition V2",
    icon=":material/auto_awesome:",
)

assessments_generation_page = st.Page(
    "assessment.py",
    title="Assessment",
    icon=":material/quiz:",
    # Optional: default=(role == "Requester") or any logic
)

workflow_directory_page = st.Page(
    "workflow_directory.py",
    title="Workflow Agents",
    icon=":material/account_tree:",
)


graphics_search_page = st.Page(
    "graphics_search.py",
    title="Graphics Search",
    icon=":material/image:",
    # Optional: default=(role == "Requester") or any logic
)

vectorstore_page = st.Page(
    "vector_store_image_search.py",              # path to the page script
    title="Image Search",          # label shown in the sidebar
    icon=":material/storage:",     # use a material or emoji icon
    # Optional: default=True or role-based logic
)

get_images_page = st.Page(
    "get_images_from_graphics_definitions.py",
    title="Image Search with Graphics Definitions",
    icon=":material/image_search:",
)

quality_compliance_scoring_page = st.Page(
    "quality_compliance_scoring.py",
    title="Quality Compliance Scoring",
    icon=":material/check_circle:",
)

video_search_tool_page = st.Page(
    "run_video_search_tool.py",
    title="Video Search Tool",
    icon=":material/video_library:",
)

graphics_v2_slideshow_page = st.Page(
    "graphics_v2_slideshow.py",
    title="Graphics Definition V2 Output Slideshow",
    icon="🎬",
)

paraphraser_page = st.Page(
    "paraphraser.py",
    title="Paraphraser",
    icon=":material/edit:",
)

image_translation_page = st.Page(
    "image_translation.py",
    title="Image Translation",
    icon=":material/translate:",
)

mcp_ui_page_obj = st.Page(
    mcp_ui_page,
    title="MCP Server Manager",
    icon=":material/settings:",
)

image_editing_tool_page = st.Page(
    "image_editing_tool.py",
    title="Image Editing",
    icon=":material/palette:",
)

curriculum_mapping_tool_page = st.Page(
    "curriculum_mapping_tool.py",
    title="Curriculum Mapping Tool",
    icon=":material/menu_book:",
)
#######################
# 3) Common app layout
#######################

# Title and logo are always visible on every page, since they're in the main script.
# st.title("Course Generation Agent")
# st.logo("images/horizontal_blue.png", icon_image="images/icon_blue.png")

#######################
# 4) Dynamic Navigation
#######################

# We'll build a dictionary of pages for the "logged in" scenario,
# plus one for the "logged out" scenario.

# role_based_page_access_dict = {
#     "Admin": [course_outline_page, research_notes_page, slide_chunks_page, graphics_definition_page, assessments_generation_page, vectorstore_page, get_images_page, quality_compliance_scoring_page, video_search_tool_page],
#     "Editor": [course_outline_page, research_notes_page, slide_chunks_page, graphics_definition_page, assessments_generation_page, vectorstore_page, get_images_page, quality_compliance_scoring_page, video_search_tool_page],
#     "Content Head": [course_outline_page, research_notes_page, quality_compliance_scoring_page, video_search_tool_page],
#     "Instructional Designer": [research_notes_page, slide_chunks_page, graphics_definition_page, assessments_generation_page, vectorstore_page, get_images_page, quality_compliance_scoring_page, video_search_tool_page],
#     "Visual Designer": [graphics_definition_page, vectorstore_page, get_images_page, quality_compliance_scoring_page, video_search_tool_page],
# }

# Create a mapping from page names to actual page objects
page_name_to_object = {
    "course_outline_page": course_outline_page,
    "research_notes_page": research_notes_page,
    "slide_chunks_page": slide_chunks_page,
    "graphics_definition_page": graphics_definition_page,
    "graphics_definition_v2_page": graphics_definition_v2_page,
    "assessments_generation_page": assessments_generation_page,
    "workflow_directory_page": workflow_directory_page,
    "graphics_search_page": graphics_search_page,
    "vectorstore_page": vectorstore_page,
    "get_images_page": get_images_page,
    "quality_compliance_scoring_page": quality_compliance_scoring_page,
    "video_search_tool_page": video_search_tool_page,
    "template_sheet_setup_page": template_sheet_setup_page,
    "graphics_v2_slideshow_page": graphics_v2_slideshow_page,
    "paraphraser_page": paraphraser_page,
    "image_translation_page": image_translation_page,
    "mcp_ui_page": mcp_ui_page_obj,
    "image_editing_tool_page": image_editing_tool_page,
    "curriculum_mapping_tool_page": curriculum_mapping_tool_page,
}

# Define which pages belong to which category
agent_pages = [
    "template_sheet_setup_page",
    # "workflow_directory_page",
    "course_outline_page",
    "research_notes_page",
    "slide_chunks_page",
    "graphics_definition_page",
    "graphics_definition_v2_page",
    "assessments_generation_page",
    "get_images_page"]
tool_pages = [
    # "graphics_search_page",
    "vectorstore_page",
    "quality_compliance_scoring_page",
    "video_search_tool_page",
    "graphics_v2_slideshow_page",
    "paraphraser_page",
    "image_translation_page",
    "mcp_ui_page",
    "image_editing_tool_page"
    "curriculum_mapping_tool_page",
]

if st.session_state.role:
    # The user is logged in with a valid role
    page_dict = {}

    # Determine which role to use for page access (impersonated role or actual role)
    effective_role = st.session_state.get("impersonated_role", st.session_state.get("role"))
    
    # Build account pages - include role switch for admins
    account_pages = [
        list_of_agents_page, 
        # about_agents_page, 
        workflow_directory_page,
    ]
    
    # Add role switch page only for admins
    if st.session_state.get("role") == "Admin":
        account_pages.append(role_switch_page)
    
    account_pages.append(logout_page)
    
    # Get user's accessible pages based on their effective role (impersonated or actual)
    if effective_role == st.session_state.get("role"):
        # Use original user pages if not impersonating
        user_page_names = st.session_state.get("user_pages", [])
    else:
        # Use pages for the impersonated role
        user_page_names = get_user_pages(effective_role)
    
    # Split pages into agents and tools
    user_agent_pages = [page_name_to_object[page_name] for page_name in user_page_names if page_name in page_name_to_object and page_name in agent_pages]
    user_tool_pages = [page_name_to_object[page_name] for page_name in user_page_names if page_name in page_name_to_object and page_name in tool_pages]

    page_dict["Agents Overview"] = account_pages
    page_dict["Agentic Workflows"] = user_agent_pages
    page_dict["Tools"] = user_tool_pages

    # Create the navigation
    # This returns the page that should be run
    current_page = st.navigation(page_dict)



else:
    # User not logged in, or role is None
    # Show only the login page
    current_page = st.navigation([st.Page(login, title="Login", icon=":material/login:")])


# Finally, call run() on whichever page the user selected in the nav.
current_page.run()
