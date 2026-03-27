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
import hashlib
from dotenv import load_dotenv
from streamlit_clickable_images import clickable_images
from utils.role_utils import get_user_info, get_user_pages, role_requires_oauth
from config.logging_config import get_logger, setup_logging
from mcp_ui_app import mcp_ui_page
from services.activity_tracking_service import track_login, track_page_view
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

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

# Sidebar navigation: wrap long page titles instead of ellipsis truncation.
st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] a {
        display: flex !important;
        align-items: baseline !important;
        gap: 0.35rem !important;
    }
    [data-testid="stSidebarNav"] a,
    [data-testid="stSidebarNav"] a * {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        word-break: break-word !important;
    }
    [data-testid="stSidebarNav"] a svg {
        flex-shrink: 0 !important;
        margin-top: 0 !important;
        transform: translateY(0.14rem);
    }
    [data-testid="stSidebarNav"] a > span:first-child,
    [data-testid="stSidebarNav"] a > div:first-child {
        flex-shrink: 0 !important;
        align-self: baseline !important;
        transform: translateY(0.14rem) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 1) Initialize Session State for user role
if "role" not in st.session_state:
    st.session_state.role = None

# Lightweight in-memory auth cache keyed by browser fingerprint.
# This avoids forced re-login when Streamlit reconnects and session_state is reset.
_AUTH_CACHE = {}

# Lightweight in-memory progress cache keyed by browser fingerprint.
# Only JSON-serializable, non-auth keys are stored to keep overhead low.
_PROGRESS_CACHE = {}
_PROGRESS_TTL_SECONDS = 6 * 60 * 60
_PROGRESS_SKIP_KEYS = {
    "drive",
    "gc",
    "role",
    "user_email",
    "user_pages",
    "oauth_authenticated",
    "oauth_state",
    "google_oauth_refresh_token",
    "impersonated_role",
}


def _browser_fingerprint() -> str:
    """Build a stable per-browser key from request metadata."""
    try:
        cookies = getattr(st.context, "cookies", {}) or {}
    except Exception:
        cookies = {}

    xsrf = cookies.get("_streamlit_xsrf", "")
    user_agent = ""
    ip_addr = ""
    try:
        headers = getattr(st.context, "headers", {}) or {}
        user_agent = headers.get("user-agent", "")
        ip_addr = headers.get("x-forwarded-for", "") or headers.get("x-real-ip", "")
    except Exception:
        pass

    raw = f"{xsrf}|{user_agent}|{ip_addr}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_auth_session(refresh_token: str, user_email: str, role: str, user_pages: list):
    """Store minimum auth context needed to rebuild a disconnected session."""
    if not refresh_token:
        return
    key = _browser_fingerprint()
    _AUTH_CACHE[key] = {
        "refresh_token": refresh_token,
        "user_email": user_email,
        "role": role,
        "user_pages": user_pages,
        "updated_at": time.time(),
    }


def _is_json_safe(value) -> bool:
    try:
        json.dumps(value)
        return True
    except Exception:
        return False


def _cache_progress_snapshot():
    """Persist a small, JSON-safe snapshot of non-auth session state."""
    if not st.session_state.get("role"):
        return

    snapshot = {}
    for key, value in st.session_state.items():
        if key in _PROGRESS_SKIP_KEYS or key.startswith("_"):
            continue
        if not _is_json_safe(value):
            continue
        snapshot[key] = value

    key = _browser_fingerprint()
    _PROGRESS_CACHE[key] = {
        "user_email": st.session_state.get("user_email", ""),
        "updated_at": time.time(),
        "snapshot": snapshot,
    }


def _restore_progress_snapshot():
    """Restore non-auth progress keys once after reconnect."""
    if st.session_state.get("_progress_restored"):
        return

    key = _browser_fingerprint()
    cached = _PROGRESS_CACHE.get(key)
    if not cached:
        st.session_state["_progress_restored"] = True
        return

    updated_at = cached.get("updated_at", 0)
    if time.time() - updated_at > _PROGRESS_TTL_SECONDS:
        _PROGRESS_CACHE.pop(key, None)
        st.session_state["_progress_restored"] = True
        return

    snapshot = cached.get("snapshot", {})
    for s_key, s_val in snapshot.items():
        if s_key not in st.session_state:
            st.session_state[s_key] = s_val

    st.session_state["_progress_restored"] = True


def _try_rehydrate_auth_session() -> bool:
    """Try to rebuild login state after reconnect without forcing OAuth UI."""
    if st.session_state.get("role"):
        return True

    key = _browser_fingerprint()
    cached = _AUTH_CACHE.get(key)
    if not cached:
        return False

    refresh_token = cached.get("refresh_token", "")
    if not refresh_token:
        return False

    oauth_client_id = os.getenv("OAUTH_CLIENT_ID", "").strip()
    oauth_client_secret = os.getenv("OAUTH_CLIENT_SECRET", "").strip()
    if not oauth_client_id or not oauth_client_secret:
        return False

    try:
        creds = Credentials(
            None,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=oauth_client_id,
            client_secret=oauth_client_secret,
            scopes=[
                "https://www.googleapis.com/auth/drive",
                "https://www.googleapis.com/auth/spreadsheets",
            ],
        )
        creds.refresh(Request())

        gauth, drive, gc = init_clients_from_credentials(
            creds,
            client_id=oauth_client_id,
            client_secret=oauth_client_secret,
        )

        # Resolve latest role/page mapping from authorized list.
        about = drive.GetAbout()
        user_email = about.get("user", {}).get("emailAddress", "") or cached.get("user_email", "")
        if not user_email:
            return False

        user_info = get_user_info(user_email)
        if not user_info.get("is_authorized"):
            return False

        st.session_state["drive"] = drive
        st.session_state["gc"] = gc
        st.session_state["oauth_authenticated"] = True
        st.session_state["role"] = user_info["role"]
        st.session_state["user_email"] = user_email
        st.session_state["user_pages"] = user_info["pages"]
        st.session_state["google_oauth_refresh_token"] = refresh_token

        _cache_auth_session(
            refresh_token=refresh_token,
            user_email=user_email,
            role=user_info["role"],
            user_pages=user_info["pages"],
        )
        return True
    except Exception as e:
        logger.warning(f"Failed to rehydrate auth session: {e}")
        return False


_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_logo_path = os.path.join(_APP_DIR, "assets", "SkillCat-Logo.png")
_helmet_path = os.path.join(_APP_DIR, "assets", "SkillCat-Helmet.png")
try:
    if os.path.exists(_logo_path) and os.path.exists(_helmet_path):
        st.logo(
            image=_logo_path,
            size="medium",
            icon_image=_helmet_path,
        )
except Exception:
    pass  # logo is non-critical; skip silently if anything goes wrong

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
                    if getattr(creds, "refresh_token", None):
                        st.session_state["google_oauth_refresh_token"] = creds.refresh_token
                        _cache_auth_session(
                            refresh_token=creds.refresh_token,
                            user_email=user_email,
                            role=user_info["role"],
                            user_pages=user_info["pages"],
                        )
                    else:
                        st.session_state.pop("google_oauth_refresh_token", None)

                    # Track login event
                    track_login(gc, user_email)

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
                                try:
                                    _rt = getattr(gauth.credentials, "refresh_token", None)
                                    if _rt:
                                        st.session_state["google_oauth_refresh_token"] = _rt
                                        _cache_auth_session(
                                            refresh_token=_rt,
                                            user_email=user_email,
                                            role=user_info["role"],
                                            user_pages=user_info["pages"],
                                        )
                                    else:
                                        st.session_state.pop("google_oauth_refresh_token", None)
                                except Exception:
                                    st.session_state.pop("google_oauth_refresh_token", None)

                                # Track login event
                                track_login(gc, user_email)

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
    
    # --- Email-only login for roles that don't need Google OAuth ---
    st.divider()
    st.markdown("**External user?** Log in with your email address:")
    email_input = st.text_input("Email address", key="email_login_input")
    if st.button("Log in with email"):
        if email_input:
            user_info = get_user_info(email_input.strip())
            if user_info["is_authorized"] and not role_requires_oauth(user_info["role"]):
                st.session_state["role"] = user_info["role"]
                st.session_state["user_email"] = email_input.strip()
                st.session_state["user_pages"] = user_info["pages"]
                st.session_state["oauth_authenticated"] = False

                # Set up a service-account gspread client for activity tracking
                try:
                    load_dotenv()
                    sa_key_bytes = base64.b64decode(os.environ["GDRIVE_SA_B64"])
                    sa_dict = json.loads(sa_key_bytes.decode())
                    gc_sa = gspread.service_account_from_dict(sa_dict)
                    st.session_state["gc"] = gc_sa
                    track_login(gc_sa, email_input.strip())
                except Exception:
                    pass  # tracking is best-effort

                st.rerun()
            else:
                st.error("This email is not authorized for email-only login. Please use Google sign-in above.")
        else:
            st.warning("Please enter your email address.")


def logout():
    """Immediately logs the user out by clearing role and authentication."""
    # Remove cached auth for this browser fingerprint.
    fp = _browser_fingerprint()
    _AUTH_CACHE.pop(fp, None)
    _PROGRESS_CACHE.pop(fp, None)
    # Delete all the items in Session state
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()


PAGE_DESCRIPTIONS = {
    "template_sheet_setup_page": {
        "icon": ":material/content_copy:",
        "title": "Template Sheet Setup",
        "description": "Set up and validate your template course sheet before running any agentic workflows.",
        "category": "agent",
    },
    "course_outline_page": {
        "icon": ":material/toc:",
        "title": "Course Outline",
        "description": "Generate course outlines using multiple agents for video search, web research, deep research, and outline consolidation.",
        "category": "agent",
    },
    "research_notes_page": {
        "icon": ":material/quick_reference_all:",
        "title": "Research Notes",
        "description": "Run the research notes pipeline including retrieval, generation, review/revision, and checklist validation.",
        "category": "agent",
    },
    "slide_chunks_page": {
        "icon": ":material/topic:",
        "title": "Slide Chunks",
        "description": "Generate slide chunks from research notes, including learning objectives and checklist review.",
        "category": "agent",
    },
    "graphics_definition_v2_page": {
        "icon": ":material/auto_awesome:",
        "title": "Graphics Definition",
        "description": "Create graphics definitions through image/video segmentation, search query generation, and aggregation with review.",
        "category": "agent",
    },
    "assessments_generation_page": {
        "icon": ":material/quiz:",
        "title": "Assessment",
        "description": "Generate and review assessment questions for courses with automated checklist validation.",
        "category": "agent",
    },
    "vectorstore_page": {
        "icon": ":material/storage:",
        "title": "Image Search",
        "description": "Search for relevant images from Google Drive and web sources using vector embeddings.",
        "category": "tool",
    },
    "quality_compliance_scoring_page": {
        "icon": ":material/check_circle:",
        "title": "Quality Compliance Scoring",
        "description": "Analyze reviewed course checklists and generate compliance metrics automatically.",
        "category": "tool",
    },
    "video_search_tool_page": {
        "icon": ":material/video_library:",
        "title": "Video Search Tool",
        "description": "Search HVAC videos by transcript or visual content from YouTube channels.",
        "category": "tool",
    },
    "slideshow_streamlit_page": {
        "icon": ":material/slideshow:",
        "title": "Slideshow",
        "description": "Generate slideshows and videos by combining images, audio narration, and transitions.",
        "category": "tool",
    },
    "paraphraser_page": {
        "icon": ":material/edit:",
        "title": "Paraphraser",
        "description": "Transform technical text into clear, conversational language for a specific trade context.",
        "category": "tool",
    },
    "image_translation_page": {
        "icon": ":material/translate:",
        "title": "Image Translation",
        "description": "Translate text within images inline while preserving the original layout and composition.",
        "category": "tool",
    },
    "mcp_ui_page": {
        "icon": ":material/settings:",
        "title": "MCP Server Manager",
        "description": "Manage MCP servers and interact with agents that use tools from connected servers.",
        "category": "tool",
    },
    "image_editing_tool_page": {
        "icon": ":material/palette:",
        "title": "Image Editing",
        "description": "Upload and edit images with AI while preserving the original composition.",
        "category": "tool",
    },
    "curriculum_mapping_tool_page": {
        "icon": ":material/menu_book:",
        "title": "Curriculum Mapping Tool",
        "description": "Map course curriculum to external standards (SkillCat, Nextech) and export to PDF.",
        "category": "tool",
    },
    "aggregation_agent_page": {
        "icon": ":material/slideshow:",
        "title": "Graphics Definition Review",
        "description": "Review and provide feedback on graphics assignments with AI suggestions and voiceover generation.",
        "category": "tool",
    },
    "workflow_directory_page": {
        "icon": ":material/account_tree:",
        "title": "Workflow Agents",
        "description": "View visual workflow diagrams for each generation pipeline with interactive navigation.",
        "category": "agent",
    },
}


def list_of_agents():
    st.header("SkillCat AI Agents Ecosystem")

    effective_role = st.session_state.get("impersonated_role", st.session_state.get("role"))
    if effective_role == st.session_state.get("role"):
        user_page_names = st.session_state.get("user_pages", [])
    else:
        user_page_names = get_user_pages(effective_role)

    user_agents = [PAGE_DESCRIPTIONS[p] for p in user_page_names if p in PAGE_DESCRIPTIONS and PAGE_DESCRIPTIONS[p]["category"] == "agent"]
    user_tools = [PAGE_DESCRIPTIONS[p] for p in user_page_names if p in PAGE_DESCRIPTIONS and PAGE_DESCRIPTIONS[p]["category"] == "tool"]

    if user_agents:
        st.subheader("Agentic Workflows")
        for page in user_agents:
            st.markdown(f"{page['icon']} **{page['title']}** — {page['description']}")

    if user_tools:
        st.subheader("Tools")
        for page in user_tools:
            st.markdown(f"{page['icon']} **{page['title']}** — {page['description']}")

    st.info(":material/info: Navigate to any page from the left side panel to get started.")

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

# graphics_definition_page = st.Page(
#     "graphics_definition.py",
#     title="Graphics Definiton (Old)",
#     icon=":material/image:",
#     # Optional: default=(role == "Requester") or any logic
# )

graphics_definition_v2_page = st.Page(
    "graphics_definition_v2.py",
    title="Graphics Definition",
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

# get_images_page = st.Page(
#     "get_images_from_graphics_definitions.py",
#     title="Image Search with Graphics Definitions",
#     icon=":material/image_search:",
# )

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

# graphics_definition_v2_slideshow_page = st.Page(
#     "graphics_definition_v2_video_generator.py",
#     title="Graphics V2 Slideshow",
#     icon=":material/slideshow:",
# )

slideshow_streamlit_page = st.Page(
    "slideshow_streamlit.py",
    title="Slideshow",
    icon=":material/slideshow:",
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

aggregation_agent_page = st.Page(
    "graphics_definition_v2_slideshow.py",
    title="View Graphics Definition Agent Outputs and Add Human Feedback",
    icon=":material/slideshow:",
)

# pptx_exporter_page = st.Page(
#     "pptx_exporter.py",
#     title="PPTX Exporter",
#     icon=":material/present_to_all:",
# )

# video_embeddings_page = st.Page(
#     "video_embeddings.py",
#     title="Video Embeddings",
#     icon=":material/menu_book:",
# )

# video_search_hvac_channels_page = st.Page(
#     "video_search_hvac_channels.py",
#     title="Video Search inside 'HVAC School' & 'Love2HVAC with TY' Youtube Channel)",
#     icon=":material/search:",
# )



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

# Try restoring login when Streamlit reconnects and session_state was reset.
_try_rehydrate_auth_session()
_restore_progress_snapshot()

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
    #"graphics_definition_page": graphics_definition_page,
    "graphics_definition_v2_page": graphics_definition_v2_page,
    "assessments_generation_page": assessments_generation_page,
    "workflow_directory_page": workflow_directory_page,
    "graphics_search_page": graphics_search_page,
    "vectorstore_page": vectorstore_page,
    #"get_images_page": get_images_page,
    "quality_compliance_scoring_page": quality_compliance_scoring_page,
    "video_search_tool_page": video_search_tool_page,
    "template_sheet_setup_page": template_sheet_setup_page,
    #"graphics_definition_v2_slideshow_page": graphics_definition_v2_slideshow_page,
    "slideshow_streamlit_page": slideshow_streamlit_page,
    "paraphraser_page": paraphraser_page,
    "image_translation_page": image_translation_page,
    "mcp_ui_page": mcp_ui_page_obj,
    "image_editing_tool_page": image_editing_tool_page,
    "curriculum_mapping_tool_page": curriculum_mapping_tool_page,
    #"video_embeddings_page": video_embeddings_page,
    "aggregation_agent_page": aggregation_agent_page,
    #"pptx_exporter_page": pptx_exporter_page,
}

# Define which pages belong to which category
agent_pages = [
    "template_sheet_setup_page",
    # "workflow_directory_page",
    "course_outline_page",
    "research_notes_page",
    "slide_chunks_page",
    #"graphics_definition_page",
    "graphics_definition_v2_page",
    "assessments_generation_page"]
    #"get_images_page"]
tool_pages = [
    # "graphics_search_page",
    "vectorstore_page",
    "quality_compliance_scoring_page",
    "video_search_tool_page",
    #"video_embeddings_page",
    #"graphics_definition_v2_slideshow_page",
    "slideshow_streamlit_page",
    "paraphraser_page",
    "image_translation_page",
    "mcp_ui_page",
    "image_editing_tool_page",
    "curriculum_mapping_tool_page",
    "aggregation_agent_page",
    #"pptx_exporter_page",
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
    ]

    # Add workflow directory page only if user has access
    if "workflow_directory_page" in (st.session_state.get("user_pages", []) if effective_role == st.session_state.get("role") else get_user_pages(effective_role)):
        account_pages.append(workflow_directory_page)
    
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


# Track page navigation
if st.session_state.role and "gc" in st.session_state and "user_email" in st.session_state:
    track_page_view(
        st.session_state["gc"],
        st.session_state["user_email"],
        current_page.title
    )

# Finally, call run() on whichever page the user selected in the nav.
current_page.run()

# Save a lightweight progress snapshot for reconnect recovery.
_cache_progress_snapshot()
