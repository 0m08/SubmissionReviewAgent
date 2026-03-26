# User Role Configuration
# This file contains email-to-role mappings for the AI Agents

# Available Roles:
# - Admin: Full access to all pages and features (Innovation Team + Admin Team)
# - Managers: Access to all pages, but cannot create vectorstore in Image Search and Video Search (tiffany, shalini, ramesh)
# - Instructional Designer: Access to research notes, slide chunks, graphics definition, assessments, image search, quality compliance, video search (remaining Content Team)
# - Visual Designer: Access to graphics-related tools, image search, quality compliance, video search (all Visual Designer Team)
# - HVAC School Users: Access to video search tool only

# Email to Role Mapping

email_to_role_mapping = {
    # Admin - Full access to all agents (Innovation Team + Admin Team)
    "akash@skillcatapp.com": "Admin",
    "ruchir@skillcatapp.com": "Admin",
    "dilip@skillcatapp.com": "Admin",
    "nancy@skillcatapp.com": "Admin",
    "om@skillcatapp.com": "Admin",
    "niket@skillcatapp.com": "Admin",
    
    # Managers - Access to all pages, but cannot create vectorstore in Image Search and Video Search
    "tiffany@skillcatapp.com": "Managers",
    "shalini@skillcatapp.com": "Managers",
    "ramesh@skillcatapp.com": "Managers",
    "erastus@skillcatapp.com": "Managers",
    "eshwar@skillcatapp.com": "Managers",
    "daniel@skillcatapp.com": "Managers",
    
    # Instructional Designer - Access to research notes, slide chunks, graphics definition, assessments, image search, quality compliance, video search
    "abheri@skillcatapp.com": "Instructional Designer",
    "akshat@skillcatapp.com": "Instructional Designer",
    "dheeraj@skillcatapp.com": "Instructional Designer",
    "elvin@skillcatapp.com": "Instructional Designer",
    "gouri@skillcatapp.com": "Instructional Designer",
    "jkariuki@skillcatapp.com": "Instructional Designer",
    "joezer@skillcatapp.com": "Instructional Designer",
    "lucie@skillcatapp.com": "Instructional Designer",
    "moreska@skillcatapp.com": "Instructional Designer",
    "namai@skillcatapp.com": "Instructional Designer",
    "ritika@skillcatapp.com": "Instructional Designer",
    "roopalakshmi@skillcatapp.com": "Instructional Designer",
    "ruthn@skillcatapp.com": "Instructional Designer",
    "sampurna@skillcatapp.com": "Instructional Designer",
    "surabhi@skillcatapp.com": "Instructional Designer",
    "trizah@skillcatapp.com": "Instructional Designer",
    "vinaypal@skillcatapp.com": "Instructional Designer",
    "isaac@skillcatapp.com": "Instructional Designer",
    "nate@skillcatapp.com": "Instructional Designer",
         
    # Visual Designer - Access to graphics-related tools, image search, quality compliance, video search
    "ajay@skillcatapp.com": "Visual Designer",
    "helmo@skillcatapp.com": "Visual Designer",
    "kopil@skillcatapp.com": "Visual Designer",
    "nilesh@skillcatapp.com": "Visual Designer",
    "pratik@skillcatapp.com": "Visual Designer",
    "siddhikumar@skillcatapp.com": "Visual Designer",
    "vaidehi@skillcatapp.com": "Visual Designer",
    "vijay@skillcatapp.com": "Visual Designer",

    # HVAC School Users - Access to video search tool only
    "dilip.d.pandey1710@gmail.com": "HVAC School Users",
}

# Default role for users not found in email_to_role_mapping
# Options: "Admin", "Managers", "Instructional Designer", "Visual Designer", "Basic User", None
default_role = None  # No access for users not in email_to_role_mapping

# Role-based page access configuration
# This defines which pages each role can access
role_page_access = {
    "Admin": [
        "template_sheet_setup_page", "workflow_directory_page", "course_outline_page", "research_notes_page", "slide_chunks_page",
        "graphics_definition_page", "graphics_definition_v2_page", "assessments_generation_page",
        "graphics_search_page", "vectorstore_page", "get_images_page",
        "quality_compliance_scoring_page", "video_search_tool_page", "paraphraser_page", "image_translation_page",
        "mcp_ui_page", "slideshow_streamlit_page", "image_editing_tool_page", "curriculum_mapping_tool_page", "video_embeddings_page", "aggregation_agent_page", "pptx_exporter_page"
    ],
    "Managers": [
        "template_sheet_setup_page", "workflow_directory_page", "course_outline_page", "research_notes_page", "slide_chunks_page",
        "graphics_definition_page", "graphics_definition_v2_page", "assessments_generation_page",
        "graphics_search_page", "vectorstore_page", "get_images_page",
        "quality_compliance_scoring_page", "video_search_tool_page",
        "paraphraser_page", "image_translation_page", "image_editing_tool_page", "pptx_exporter_page", "aggregation_agent_page"
    ],
    "Instructional Designer": [
        "template_sheet_setup_page", "workflow_directory_page", "course_outline_page", "research_notes_page", "slide_chunks_page",
        "graphics_definition_v2_page", "assessments_generation_page", "vectorstore_page", "get_images_page",
        "video_search_tool_page", "quality_compliance_scoring_page",
        "paraphraser_page", "image_translation_page", "image_editing_tool_page", "pptx_exporter_page", "aggregation_agent_page"
    ],
    "Visual Designer": [
        "template_sheet_setup_page", "workflow_directory_page", "graphics_definition_page", "graphics_definition_v2_page", "graphics_search_page", "vectorstore_page", "get_images_page",
        "quality_compliance_scoring_page", "video_search_tool_page",
        "paraphraser_page", "image_translation_page", "image_editing_tool_page", "pptx_exporter_page", "aggregation_agent_page"
    ],
    "Basic User": [
        "vectorstore_page", "video_search_tool_page",
        "paraphraser_page","image_translation_page", "image_editing_tool_page",
        "curriculum_mapping_tool_page"
    ],
    "HVAC School Users": [
        "video_search_tool_page"
    ],
    None: [
        # No pages accessible for users with None role (denied access)
    ]
}

# Roles that do NOT require Google OAuth (Drive/Sheets) permissions.
# Users with these roles can log in with just their email address.
no_oauth_roles = {
    "HVAC School Users",
}

# Get all available roles (excluding Admin and None)
def get_available_roles(exclude_admin=True, exclude_none=True):
    """
    Get list of available roles for role switching.
    
    Args:
        exclude_admin: Whether to exclude Admin role (default: True)
        exclude_none: Whether to exclude None role (default: True)
        
    Returns:
        List of available role names
    """
    roles = list(role_page_access.keys())
    if exclude_admin and "Admin" in roles:
        roles.remove("Admin")
    if exclude_none and None in roles:
        roles.remove(None)
    return roles
