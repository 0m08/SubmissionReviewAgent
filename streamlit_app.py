import streamlit as st
import time

# 1) Initialize Session State for user role
if "role" not in st.session_state:
    st.session_state.role = None

authenticated_roles = {
    "Editor": "Editor",
    "Admin": "Admin",
    "Content Head": "ch", 
    "Instructional Designer": "id", 
    "Visual Designer": "vd",
}

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
    """A simple 'Login' page as a function. 
       Called if user is not logged in."""
    st.header("Login Page")
    role_choice = st.text_input("Enter your login name: ")
    password = st.text_input("Enter the password: ")
    if st.button("Log in"):
        if role_choice in authenticated_roles:
            if password == authenticated_roles[role_choice]:
                st.success("Logging in")
                st.session_state.role = role_choice
            else:
                st.error("Failed to authenticate. Wrong passsword. Try again.")
                time.sleep(2)
        else:
            st.error("Failed to authenticate. Wrong user name. Try again.")
            time.sleep(2)
        st.rerun()


def logout():
    """Immediately logs the user out by clearing role."""
    # st.session_state.role = None
    # Delete all the items in Session state
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()


def list_of_agents():
    st.header("Welcome")
    st.write("Here's the list of agents:")
    st.markdown(
"""Agent Name | Status
-|-
:material/toc: Course Outline Agent | :material/check_circle: Done
:material/quick_reference_all: Research Notes Agent | :material/check_circle: Done
:material/topic: Slide Chunks Agent | :material/check_circle: Done
:material/image: Graphics Definition Agent | :material/check_circle: Done
:material/quiz: Assessment Agent | :material/check_circle: Done
"""
    )

    st.info(""":material/info: To run any of the above agents, navigate to the corresponding page from the left side panel""")


# We can either define Page objects inline (pointing to .py files or callables)
# or just define them here. For simplicity, let's define some stubs as Page objects.

# --- Account pages ---
list_of_agents_page = st.Page(list_of_agents, title = "List of agents", icon = ":material/list:")
logout_page = st.Page(logout, title="Log out", icon=":material/logout:")

# --- Outline pages ---
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

assessments_generation_page = st.Page(
    "assessment.py",
    title="Assessment",
    icon=":material/quiz:",
    # Optional: default=(role == "Requester") or any logic
)


graphics_search_page = st.Page(
    "graphics_search.py",
    title="Graphics Search",
    icon=":material/image:",
    # Optional: default=(role == "Requester") or any logic
)

vectorstore_page = st.Page(
    "vector_store_image_search.py",              # path to the page script
    title="Vector Store",          # label shown in the sidebar
    icon=":material/storage:",     # use a material or emoji icon
    # Optional: default=True or role-based logic
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


role_based_page_access_dict = {
    "Admin": [course_outline_page, research_notes_page, slide_chunks_page, graphics_definition_page, assessments_generation_page, vectorstore_page],
    "Editor": [course_outline_page, research_notes_page, slide_chunks_page, graphics_definition_page, assessments_generation_page, vectorstore_page],
    "Content Head": [course_outline_page, research_notes_page],
    "Instructional Designer": [research_notes_page, slide_chunks_page, graphics_definition_page, assessments_generation_page, vectorstore_page],
    "Visual Designer": [graphics_definition_page, vectorstore_page],
}


if st.session_state.role in authenticated_roles:
    # The user is logged in (role != None)
    page_dict = {}

    account_pages = [list_of_agents_page, logout_page]
    user_pages = role_based_page_access_dict[st.session_state.role]

    page_dict["Account"] = account_pages
    page_dict["User Pages"] = user_pages

    # Create the navigation
    # This returns the page that should be run
    current_page = st.navigation(page_dict)



else:
    # User not logged in, or role is None
    # Show only the login page
    current_page = st.navigation([st.Page(login, title="Login", icon=":material/login:")])


# Finally, call run() on whichever page the user selected in the nav.
current_page.run()
