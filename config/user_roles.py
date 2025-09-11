# User Role Configuration
# This file contains email-to-role mappings for the AI Agents

# Available Teams and Roles:
# - Admin Team
# - Innovation Team
# - Content Team
# - Visual Designer Team

# Email to Role Mapping

email_to_role_mapping = {
    # Admin Team - Full access to all agents
    "akash@skillcatapp.com": "Admin Team",
    "ruchir@skillcatapp.com": "Admin Team",
    
    # Innovation Team - Full access to all agents
    "dilip@skillcatapp.com": "Innovation Team",
    "nancy@skillcatapp.com": "Innovation Team",
    "om@skillcatapp.com": "Innovation Team",
   # "niket@skillcatapp.com": "Innovation Team",
    
    # Content Team 
    "abheri@skillcatapp.com": "Content Team",
    "akshat@skillcatapp.com": "Content Team",
    "dheeraj@skillcatapp.com": "Content Team",
    "elvin@skillcatapp.com": "Content Team",
    "erastus@skillcatapp.com": "Content Team",
    "gouri@skillcatapp.com": "Content Team",
    "jkariuki@skillcatapp.com": "Content Team",
    "joezer@skillcatapp.com": "Content Team",
    "lucie@skillcatapp.com": "Content Team",
    "moreska@skillcatapp.com": "Content Team",
    "namai@skillcatapp.com": "Content Team",
    "ritika@skillcatapp.com": "Content Team",
    "roopalakshmi@skillcatapp.com": "Content Team",
    "ruthn@skillcatapp.com": "Content Team",
    "sampurna@skillcatapp.com": "Content Team",
    "shalini@skillcatapp.com": "Content Team",
    "surabhi@skillcatapp.com": "Content Team",
    "tiffany@skillcatapp.com": "Content Team",
    "trizah@skillcatapp.com": "Content Team",
    "niket@skillcatapp.com": "Content Team",
         
    # Visual Designer Team 
    "ajay@skillcatapp.com": "Visual Designer Team",
    "helmo@skillcatapp.com": "Visual Designer Team",
    "kopil@skillcatapp.com": "Visual Designer Team",
    "nilesh@skillcatapp.com": "Visual Designer Team",
    "pratik@skillcatapp.com": "Visual Designer Team",
    "ramesh@skillcatapp.com": "Visual Designer Team",
    "siddhikumar@skillcatapp.com": "Visual Designer Team",
    "vaidehi@skillcatapp.com": "Visual Designer Team",
    "vijay@skillcatapp.com": "Visual Designer Team",
}

# Default role for users not found in EMAIL_TO_ROLE_MAPPING
default_role = None  # No access for users not in email_to_role_mapping

# Role-based page access configuration
# This defines which pages each team can access
role_page_access = {
    "Admin Team": [
        "course_outline_page", "research_notes_page", "slide_chunks_page", 
        "graphics_definition_page", "assessments_generation_page", 
        "graphics_search_page", "vectorstore_page", "get_images_page", 
        "quality_compliance_scoring_page", "video_search_tool_page"
    ],
    "Innovation Team": [
        "course_outline_page", "research_notes_page", "slide_chunks_page", 
        "graphics_definition_page", "assessments_generation_page", 
        "graphics_search_page", "vectorstore_page", "get_images_page", 
        "quality_compliance_scoring_page", "video_search_tool_page"
    ],
    "Content Team": [
        "course_outline_page", "research_notes_page", "slide_chunks_page", "assessments_generation_page",
        "graphics_search_page", "vectorstore_page", "get_images_page",
        "quality_compliance_scoring_page", "video_search_tool_page"
    ],
    "Visual Designer Team": [
        "graphics_definition_page", "graphics_search_page", "vectorstore_page", "get_images_page", 
        "quality_compliance_scoring_page", "video_search_tool_page"
    ],
    None: [
        # No pages accessible for users with None role (denied access)
    ]
}
