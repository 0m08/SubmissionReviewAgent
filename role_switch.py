import streamlit as st
from utils.role_utils import get_user_pages
from config.user_roles import role_page_access

# Check if user is admin
if st.session_state.get("role") != "Admin":
    st.error("❌ Access denied. This page is only available to Admin users.")
    st.stop()

st.header("🔀 Role Switching")
st.markdown("**Switch to any role to experience the UI as that user would see it.**")

# Get current impersonated role (defaults to actual role if not set)
current_impersonated_role = st.session_state.get("impersonated_role", st.session_state.get("role"))

# Display current status
col1, col2 = st.columns([2, 1])

with col1:
    if current_impersonated_role == st.session_state.get("role"):
        st.success(f"✅ Currently viewing as: **{current_impersonated_role}** (Your actual role)")
    else:
        st.warning(f"🎭 Currently impersonating: **{current_impersonated_role}**")

with col2:
    if current_impersonated_role != st.session_state.get("role"):
        if st.button("🔄 Return to Admin", type="primary"):
            st.session_state["impersonated_role"] = st.session_state.get("role")
            st.rerun()

st.divider()

# Role selection
st.subheader("Switch Role")

# Available roles (excluding the current actual role to avoid confusion)
available_roles = ["Managers", "Instructional Designer", "Visual Designer"]

# Add current role if it's not Admin 
if st.session_state.get("role") != "Admin":
    available_roles.append(st.session_state.get("role"))

selected_role = st.selectbox(
    "Select a role to switch to:",
    options=available_roles,
    index=available_roles.index(current_impersonated_role) if current_impersonated_role in available_roles else 0,
    help="Choose a role to experience the UI as that user would see it"
)

# Show what pages this role can access
# if selected_role:
#     accessible_pages = get_user_pages(selected_role)
    
#     st.subheader(f"Pages accessible to {selected_role}:")
    
#     if accessible_pages:
#         # Create a display of accessible pages with same icons as sidebar
#         page_display_names = {
#             "course_outline_page": ":material/toc: Course Outline",
#             "research_notes_page": ":material/quick_reference_all: Research Notes", 
#             "slide_chunks_page": ":material/topic: Slide Chunks",
#             "graphics_definition_page": ":material/image: Graphics Definition",
#             "assessments_generation_page": ":material/quiz: Assessment",
#             "graphics_search_page": ":material/image: Graphics Search",
#             "vectorstore_page": ":material/storage: Image Search",
#             "get_images_page": ":material/image_search: Image Search with Graphics Definitions",
#             "quality_compliance_scoring_page": ":material/check_circle: Quality Compliance Scoring",
#             "video_search_tool_page": ":material/video_library: Video Search Tool"
#         }
        
#         cols = st.columns(2)
#         for i, page in enumerate(accessible_pages):
#             with cols[i % 2]:
#                 display_name = page_display_names.get(page, page)
#                 st.write(display_name)
#     else:
#         st.info("No pages accessible to this role.")

# Switch button
if st.button("🔄 Switch to Selected Role", type="primary", disabled=selected_role == current_impersonated_role):
    st.session_state["impersonated_role"] = selected_role
    st.success(f"✅ Switched to {selected_role} role!")
    st.rerun()
