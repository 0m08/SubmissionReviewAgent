import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import os
import json
import sys
import inspect
import re
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse
from dotenv import load_dotenv

# Add workspace root directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # graphics_asset_creation
agents_dir = os.path.dirname(parent_dir)  # agents
workspace_dir = os.path.dirname(agents_dir)  # workspace root
sys.path.insert(0, workspace_dir)

from agents.graphics_asset_creation.automated.automated_voiceover_reviewer import review_and_edit_image
from agents.graphics_asset_creation.reviewers.voiceover_reviewer import reviewer_agent
from agents.graphics_asset_creation.reviewers.copyright_reviewer import copyright_reviewer_agent
from agents.graphics_asset_creation.reviewers.voiceover_focus_agent import (
    run_adaptive_focus_or_illustrator,
    VoiceFocusAnalysis,
    FocusIntervention,
)
from agents.graphics_asset_creation.reviewers.technical_accuracy_validator_agent import (
    run_technical_accuracy_validator_agent,
    TechnicalValidationAnalysis,
)
from agents.graphics_asset_creation.illustrator.illustrator_agent import illustrator_agent

# Load environment variables
load_dotenv()

# Configure page
try:
    st.set_page_config(page_title="Voiceover Reviewer", layout="wide")
except st.errors.StreamlitAPIException:
    pass

# Helper function for formatting (used in both pages)
def _resolve_instructions(review_result):
    """Return a unified instructions object regardless of schema version."""
    if isinstance(review_result, dict):
        raw = review_result.get("recommended_instructions") or review_result.get("transformation_instructions")
    else:
        raw = getattr(review_result, "recommended_instructions", None) or getattr(review_result, "transformation_instructions", None)

    if raw is None:
        return None
    if isinstance(raw, dict):
        return SimpleNamespace(**raw)
    return raw

def _get_instruction_text(instructions, name: str) -> str:
    """Safely fetch instruction text as a trimmed string."""
    if not instructions:
        return ""
    if isinstance(instructions, dict):
        value = instructions.get(name, "")
    else:
        value = getattr(instructions, name, "")
    if value is None:
        return ""
    return str(value).strip()

def format_recommended_instructions(instructions):
    """Format recommended instructions for display."""
    if not instructions:
        return "None"

    formatted = []
    subject_focus = _get_instruction_text(instructions, "subject_focus")
    if subject_focus:
        formatted.append(f"**Subject Focus:** {subject_focus}")
    perspective_and_camera = _get_instruction_text(instructions, "perspective_and_camera")
    if perspective_and_camera:
        formatted.append(f"**Perspective & Camera:** {perspective_and_camera}")
    additional_comments = _get_instruction_text(instructions, "additional_comments")
    if additional_comments:
        formatted.append(f"**Additional Comments:** {additional_comments}")

    return "\n\n".join(formatted) if formatted else "None"


def _extract_google_drive_file_id(url: str) -> str:
    """Extract a Drive file id from common share URL formats."""
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    if "drive.google.com" not in host and "docs.google.com" not in host:
        return ""

    # Format: /file/d/<id>/view
    path_match = re.search(r"/file/d/([^/]+)", parsed.path or "")
    if path_match:
        return path_match.group(1)

    # Format: /open?id=<id> or any ?id=<id>
    query_id = parse_qs(parsed.query or "").get("id", [""])[0]
    if query_id:
        return query_id

    # Format: /uc?export=...&id=<id> already includes id
    return ""


def _normalize_image_url(raw_url: str) -> str:
    """Convert supported Google Drive share links into direct download links."""
    url = (raw_url or "").strip()
    if not url:
        return ""

    file_id = _extract_google_drive_file_id(url)
    if file_id:
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url


def _load_image_from_url(raw_url: str, timeout: int = 20) -> Image.Image:
    """Load an image from a web or Google Drive URL."""
    fetch_url = _normalize_image_url(raw_url)
    response = requests.get(fetch_url, timeout=timeout, allow_redirects=True)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))

from agents.graphics_asset_creation.generator.image_generator import generate_asset
# Sidebar navigation
st.sidebar.title("🎙️ Navigation")
page = st.sidebar.radio(
    "Select Mode:",
    ["Full Workflow", "Review Only", "Voiceover Focus Agent", "Technical Accuracy Validator Agent", "Illustrator Agent", "Image Generator (Test)"],
    index=0,
)

# ============================================================================
# FULL WORKFLOW PAGE
# ============================================================================
if page == "Full Workflow":
    st.title("🎙️ Voiceover Reviewer Agent - Full Workflow")
    st.markdown("Test the reviewer agent with automatic editing - provides reference image and slide context.")
    
    # Main inputs
    st.subheader("📷 Reference Image")
    image_input_method = st.radio("Choose input method:", ["URL", "Upload"], horizontal=True)

    if image_input_method == "URL":
        image_url = st.text_input("Image URL (Google Drive, web link, etc.)", placeholder="https://...")
        uploaded_file = None
    else:
        uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "webp"])
        image_url = None

    st.subheader("📝 Slide Context")
    col1, col2 = st.columns(2)
    with col1:
        slide_title = st.text_input("Slide Title", placeholder="Enter slide title...")
    with col2:
        visual_instruction = st.text_input("Visual Instruction", placeholder="Visual requirements...")

    slide_content = st.text_area(
        "Slide Content",
        placeholder="Main content of the slide...",
        height=100
    )
    
    voiceover = st.text_area(
        "Voiceover",
        placeholder="Voiceover script text...",
        height=100
    )

    # Helper for Loading Image
    def get_initial_image():
        if not image_url and not uploaded_file:
            return None
        try:
            if image_url:
                return _load_image_from_url(image_url, timeout=20)
            else:
                return Image.open(uploaded_file)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return None

    # Placeholder for live updates
    progress_container = st.container()

    def update_ui_live(round_data):
        with progress_container:
            agent_name = round_data.get('agent', 'Reviewer')
            verdict = round_data['review'].verdict
            verdict_emoji = "✅" if verdict == "Yes" else "❌"
            
            with st.expander(f"Round {round_data['round']} - [{agent_name}] {verdict_emoji} Verdict: {verdict}", expanded=True):
                col_img, col_details = st.columns([1, 2])
                
                with col_img:
                    st.image(round_data['image'], caption=f"Round {round_data['round']} Image", width='stretch')
                
                with col_details:
                    st.markdown(f"### 🤖 Agent: {agent_name}")
                    
                    st.markdown("#### 📝 Description")
                    st.write(round_data['review'].description)
                    
                    st.markdown("#### 🔍 Analysis")
                    st.write(round_data['review'].analysis)
                    
                    st.markdown(f"#### {verdict_emoji} Verdict: **{verdict}**")
                    
                    if verdict == "No":
                        st.markdown("#### 🎨 Recommended Instructions")
                        instructions = _resolve_instructions(round_data['review'])
                        instructions_text = format_recommended_instructions(instructions)
                        st.markdown(instructions_text)
                        st.warning(f"Applying {agent_name} editing instructions...")
                    else:
                        st.success(f"Image approved by {agent_name}!")

    # --- WORKFLOW EXECUTION ---
    if any([image_url, uploaded_file]) and any([slide_title, slide_content, voiceover, visual_instruction]):
        
        # Initialize session state for steps
        if 'stage' not in st.session_state:
            st.session_state.stage = 1 # 1: Accuracy, 2: Copyright
        if 'base_image' not in st.session_state:
            st.session_state.base_image = None
        if 'accurate_image' not in st.session_state:
            st.session_state.accurate_image = None
        if 'final_image' not in st.session_state:
            st.session_state.final_image = None

        # Load base image if not present
        if st.session_state.base_image is None:
            st.session_state.base_image = get_initial_image()

        if st.session_state.base_image:
            # Show reference image persistently
            st.markdown("---")
            st.subheader("📌 Reference Image")
            st.image(st.session_state.base_image, caption="Original Reference Image", width=400)
            
            # --- STEP 1: TECHNICAL ACCURACY ---
            st.markdown("---")
            st.header("Step 1: Technical & Educational Accuracy")
            
            if st.session_state.stage == 1:
                st.image(st.session_state.base_image, caption="Base Image", width=400)
                if st.button("🚀 Start Stage 1: Technical Review", type="primary", width='stretch'):
                    with st.spinner("Executing Stage 1: Technical Alignment..."):
                        result, edited_image, history = review_and_edit_image(
                            st.session_state.base_image, 
                            slide_title=slide_title,
                            slide_content=slide_content,
                            voiceover=voiceover,
                            visual_instruction=visual_instruction,
                            image_size="1K",
                            callback=update_ui_live,
                            target_stage="accuracy"
                        )
                        st.session_state.accurate_image = edited_image if edited_image else st.session_state.base_image
                        st.session_state.stage_1_history = history
                        st.session_state.stage = 2
                        st.rerun()
            
            elif st.session_state.stage >= 2:
                col_s1_v, col_s1_r = st.columns([3, 1])
                with col_s1_v:
                    st.success("✅ Stage 1 Complete: Technical Accuracy Achieved.")
                with col_s1_r:
                    if st.button("🔄 Rerun Stage 1", width='stretch'):
                        st.session_state.stage = 1
                        for key in ["accurate_image", "final_image", "stage_1_history", "stage_2_history", "stage_2_complete"]:
                            if key in st.session_state: del st.session_state[key]
                        st.rerun()

                with st.expander("View Stage 1 Review History"):
                    for itm in st.session_state.get('stage_1_history', []):
                        verdict_emoji = "✅" if itm['review'].verdict == "Yes" else "❌"
                        agent_name = itm.get('agent', 'Reviewer')
                        
                        with st.container():
                            st.markdown(f"### Round {itm['round']} - [{agent_name}] {verdict_emoji} Verdict: {itm['review'].verdict}")
                            
                            col_hist_img, col_hist_details = st.columns([1, 2])
                            
                            with col_hist_img:
                                st.image(itm['image'], caption=f"Round {itm['round']} Image", width='stretch')
                            
                            with col_hist_details:
                                st.markdown("**📝 Description:**")
                                st.write(itm['review'].description)
                                
                                st.markdown("**🔍 Analysis:**")
                                st.write(itm['review'].analysis)
                                
                                if itm['review'].verdict == "No":
                                    st.markdown("**🎨 Recommended Instructions:**")
                                    instructions_text = format_recommended_instructions(_resolve_instructions(itm['review']))
                                    st.markdown(instructions_text)
                            
                            st.divider()

            # --- STEP 2: COPYRIGHT TRANSFORMATION ---
            if st.session_state.stage == 2:
                st.markdown("---")
                st.header("Step 2: Copyright & IP Audit")
                st.image(st.session_state.accurate_image, caption="Accurate Image (Input for Stage 2)", width=400)
                
                if st.button("🛡️ Start Stage 2: Copyright Transformation", type="primary", width='stretch'):
                    with st.spinner("Executing Stage 2: Copyright Audit & Refinement..."):
                        result, final_image, history = review_and_edit_image(
                            st.session_state.accurate_image, 
                            slide_title=slide_title,
                            slide_content=slide_content,
                            voiceover=voiceover,
                            visual_instruction=visual_instruction,
                            image_size="1K",
                            callback=update_ui_live,
                            target_stage="full"
                        )
                        st.session_state.final_image = final_image if final_image else st.session_state.accurate_image
                        st.session_state.stage_2_history = history
                        st.session_state.stage = 3
                        st.rerun()
            
            elif st.session_state.stage == 3:
                if st.button("🔄 Rerun Stage 2", width='stretch'):
                    st.session_state.stage = 2
                    for key in ["final_image", "stage_2_history", "stage_2_complete"]:
                        if key in st.session_state: del st.session_state[key]
                    st.rerun()
            
            # --- FINAL RESULTS ---
            if st.session_state.stage == 3:
                st.markdown("---")
                st.header("🎉 Final Validated Graphic")
                st.success("The asset is now both technically accurate and legally safe.")
                
                with st.expander("View Stage 2 Review History"):
                    for itm in st.session_state.get('stage_2_history', []):
                        verdict_emoji = "✅" if itm['review'].verdict == "Yes" else "❌"
                        agent_name = itm.get('agent', 'Reviewer')
                        
                        with st.container():
                            st.markdown(f"### Round {itm['round']} - [{agent_name}] {verdict_emoji} Verdict: {itm['review'].verdict}")
                            
                            col_hist_img, col_hist_details = st.columns([1, 2])
                            
                            with col_hist_img:
                                st.image(itm['image'], caption=f"Round {itm['round']} Image", width='stretch')
                            
                            with col_hist_details:
                                st.markdown("**📝 Description:**")
                                st.write(itm['review'].description)
                                
                                st.markdown("**🔍 Analysis:**")
                                st.write(itm['review'].analysis)
                                
                                if itm['review'].verdict == "No":
                                    st.markdown("**🎨 Recommended Instructions:**")
                                    instructions_text = format_recommended_instructions(_resolve_instructions(itm['review']))
                                    st.markdown(instructions_text)
                            
                            st.divider()
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.image(st.session_state.base_image, caption="Original reference", width='stretch')
                with col_b:
                    st.image(st.session_state.final_image, caption="Final transformed asset", width='stretch')
                
                # Download
                buf = BytesIO()
                st.session_state.final_image.save(buf, format="PNG")
                st.download_button(
                    label="⬇️ Download Final Asset",
                    data=buf.getvalue(),
                    file_name="final_validated_asset.png",
                    mime="image/png",
                    width='stretch'
                )
                
                if st.button("🔄 Reset workflow to start over"):
                    for key in ["stage", "base_image", "accurate_image", "final_image", "stage_1_history", "stage_2_history"]:
                        if key in st.session_state: del st.session_state[key]
                    st.rerun()

    else:
        st.info("Please provide image input and slide context to begin the workflow.")

# ============================================================================
# REVIEW ONLY PAGE
# ============================================================================
elif page == "Review Only":
    st.title("🔍 Image Review Only")
    st.markdown("Get review feedback from Technical Accuracy or Copyright reviewer without automatic editing.")
    
    st.markdown("---")
    
    # Reviewer selection
    reviewer_type = st.radio(
        "Select Reviewer Agent:",
        ["Technical & Educational Accuracy", "Copyright & IP Compliance"],
        horizontal=True
    )
    
    st.markdown("---")
    
    # Image input
    st.subheader("📷 Image Input")
    review_image_method = st.radio("Choose input method:", ["URL", "Upload"], horizontal=True, key="review_image_method")
    
    review_image = None
    if review_image_method == "URL":
        review_image_url = st.text_input("Image URL", placeholder="https://...", key="review_image_url")
        if review_image_url:
            try:
                review_image = _load_image_from_url(review_image_url, timeout=20)
                st.image(review_image, caption="Loaded Image", width=400)
            except Exception as e:
                st.error(f"Error loading image: {e}")
    else:
        review_uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "webp"], key="review_uploaded")
        if review_uploaded:
            review_image = Image.open(review_uploaded)
            st.image(review_image, caption="Uploaded Image", width=400)
    
    # Context inputs
    st.subheader("📝 Slide Context")
    col1, col2 = st.columns(2)
    with col1:
        review_slide_title = st.text_input("Slide Title", placeholder="Enter slide title...", key="review_slide_title")
    with col2:
        review_visual_instruction = st.text_input("Visual Instruction", placeholder="Visual requirements...", key="review_visual_instruction")
    
    review_slide_content = st.text_area(
        "Slide Content",
        placeholder="Main content of the slide...",
        height=100,
        key="review_slide_content"
    )
    
    review_voiceover = st.text_area(
        "Voiceover",
        placeholder="Voiceover script text...",
        height=100,
        key="review_voiceover"
    )
    
    # Reference Images for Copyright Reviewer (Optional)
    reference_images = []
    if reviewer_type == "Copyright & IP Compliance":
        st.markdown("---")
        st.subheader("🖼️ Reference Images (Optional)")
        st.markdown("*Original reference images to compare against the edited image for similarity analysis*")
        
        ref_input_method = st.radio("Choose reference input method:", ["Upload", "URLs"], horizontal=True, key="ref_input_method")
        
        if ref_input_method == "Upload":
            uploaded_ref_files = st.file_uploader(
                "Upload Reference Images:",
                type=["png", "jpg", "jpeg", "webp"],
                accept_multiple_files=True,
                help="Upload one or more original/reference images for comparison",
                key="copyright_reference_uploader"
            )
            
            if uploaded_ref_files:
                for uploaded_file in uploaded_ref_files:
                    try:
                        img = Image.open(uploaded_file)
                        reference_images.append(img)
                    except Exception as e:
                        st.error(f"Failed to load {uploaded_file.name}: {str(e)}")
        
        else:  # URLs
            ref_urls_text = st.text_area(
                "Reference Image URLs (one per line):",
                placeholder="https://example.com/image1.jpg\nhttps://example.com/image2.jpg",
                height=100,
                help="Enter one URL per line for multiple reference images",
                key="copyright_reference_urls"
            )
            
            if ref_urls_text:
                ref_urls = [url.strip() for url in ref_urls_text.split('\n') if url.strip()]
                for idx, url in enumerate(ref_urls):
                    try:
                        img = _load_image_from_url(url, timeout=20)
                        reference_images.append(img)
                    except Exception as e:
                        st.error(f"Failed to load reference image {idx+1} from URL: {str(e)}")
        
        # Display reference images
        if reference_images:
            st.success(f"📌 {len(reference_images)} reference image(s) loaded for similarity comparison")
            cols = st.columns(min(3, len(reference_images)))
            for idx, img in enumerate(reference_images):
                with cols[idx % 3]:
                    st.image(img, caption=f"Reference {idx+1}", width='stretch')
    
    # Review button
    if review_image and any([review_slide_title, review_slide_content, review_voiceover, review_visual_instruction]):
        if st.button("🔍 Get Review", type="primary", width='stretch'):
            with st.spinner(f"Running {reviewer_type} review..."):
                try:
                    # Call appropriate reviewer based on selection
                    if reviewer_type == "Copyright & IP Compliance":
                        # Call copyright reviewer with reference images
                        review_result = copyright_reviewer_agent(
                            image=review_image,
                            slide_title=review_slide_title,
                            voiceover=review_voiceover,
                            reference_images=reference_images if reference_images else None
                        )
                        
                        st.markdown("---")
                        st.header("📋 Copyright Review Results")
                        
                        # Display reference image info if provided
                        if reference_images and hasattr(review_result, 'similarity_scores') and review_result.similarity_scores:
                            st.info(f"📊 Similarity: {review_result.similarity_scores}")
                        
                        # Display review details in two columns
                        col_rev_img, col_rev_details = st.columns([1, 2])
                        
                        with col_rev_img:
                            st.image(review_image, caption="Reviewed Image", width='stretch')
                        
                        with col_rev_details:
                            st.markdown(f"### ⚖️ Copyright & IP Compliance Review")
                            
                            st.markdown("#### 📝 Image Description")
                            with st.expander("View Description", expanded=False):
                                st.write(review_result.description)
                            
                            st.markdown("#### 🔍 Copyright Analysis")
                            with st.expander("View Analysis", expanded=True):
                                st.write(review_result.analysis)
                        
                        # Transformation Instructions - Only show non-empty fields
                        st.markdown("---")
                        st.subheader("🔄 Recommended Transformations")
                        
                        instructions = _resolve_instructions(review_result)
                        has_instructions = False

                        if instructions:
                            col_t1, col_t2 = st.columns(2)

                            with col_t1:
                                subject_focus = _get_instruction_text(instructions, "subject_focus")
                                if subject_focus:
                                    has_instructions = True
                                    with st.expander("🎯 Subject Focus Changes", expanded=True):
                                        st.markdown(subject_focus)

                                perspective_and_camera = _get_instruction_text(instructions, "perspective_and_camera")
                                if perspective_and_camera:
                                    has_instructions = True
                                    with st.expander("📷 Perspective & Camera", expanded=True):
                                        st.markdown(perspective_and_camera)

                                lighting_and_environment = _get_instruction_text(instructions, "lighting_and_environment")
                                if lighting_and_environment:
                                    has_instructions = True
                                    with st.expander("💡 Lighting & Environment", expanded=True):
                                        st.markdown(lighting_and_environment)

                            with col_t2:
                                visual_style = _get_instruction_text(instructions, "visual_style")
                                if visual_style:
                                    has_instructions = True
                                    with st.expander("🎨 Visual Style Transformation", expanded=True):
                                        st.markdown(visual_style)

                                background_setting = _get_instruction_text(instructions, "background_setting")
                                if background_setting:
                                    has_instructions = True
                                    with st.expander("🌄 Background Setting", expanded=True):
                                        st.markdown(background_setting)

                                color_grading = _get_instruction_text(instructions, "color_grading")
                                if color_grading:
                                    has_instructions = True
                                    with st.expander("🎨 Color Grading", expanded=True):
                                        st.markdown(color_grading)

                            additional_comments = _get_instruction_text(instructions, "additional_comments")
                            if additional_comments:
                                has_instructions = True
                                with st.expander("📝 Additional Comments", expanded=True):
                                    st.markdown(additional_comments)

                        if not has_instructions:
                            st.info("✅ No specific transformations recommended - current image is acceptable.")
                        
                        # Export Report
                        st.markdown("---")
                        similarity_section = ""
                        if hasattr(review_result, 'similarity_scores') and review_result.similarity_scores:
                            similarity_section = f"\n## Similarity Scores\n{review_result.similarity_scores}\n"
                        
                        report = f"""# Copyright Review Report

## Slide Context
- **Title:** {review_slide_title if review_slide_title else "N/A"}
- **Voiceover:** {review_voiceover if review_voiceover else "N/A"}
{similarity_section}
## Image Description
{review_result.description}

## Copyright Analysis
{review_result.analysis}

## Transformation Instructions
"""
                        if has_instructions:
                            if instructions.subject_focus and instructions.subject_focus.strip():
                                report += f"\n### Subject Focus\n{instructions.subject_focus}\n"
                            if instructions.visual_style and instructions.visual_style.strip():
                                report += f"\n### Visual Style\n{instructions.visual_style}\n"
                            if instructions.perspective_and_camera and instructions.perspective_and_camera.strip():
                                report += f"\n### Perspective & Camera\n{instructions.perspective_and_camera}\n"
                            if instructions.lighting_and_environment and instructions.lighting_and_environment.strip():
                                report += f"\n### Lighting & Environment\n{instructions.lighting_and_environment}\n"
                            if instructions.background_setting and instructions.background_setting.strip():
                                report += f"\n### Background Setting\n{instructions.background_setting}\n"
                            if instructions.color_grading and instructions.color_grading.strip():
                                report += f"\n### Color Grading\n{instructions.color_grading}\n"
                            if instructions.additional_comments and instructions.additional_comments.strip():
                                report += f"\n### Additional Comments\n{instructions.additional_comments}\n"
                        else:
                            report += "\nNo specific transformations recommended.\n"
                        
                        st.download_button(
                            label="📄 Download Review Report (Markdown)",
                            data=report,
                            file_name="copyright_review_report.md",
                            mime="text/markdown",
                            width='stretch'
                        )
                    
                    else:
                        # Call the technical accuracy reviewer agent
                        review_result = reviewer_agent(
                            reference_image=review_image,
                            slide_title=review_slide_title,
                            slide_content=review_slide_content,
                            voiceover=review_voiceover,
                            visual_instruction=review_visual_instruction
                        )
                        
                        st.markdown("---")
                        st.header("📋 Review Results")
                        
                        # Display verdict with appropriate styling
                        verdict_emoji = "✅" if review_result.verdict == "Yes" else "❌"
                        if review_result.verdict == "Yes":
                            st.success(f"{verdict_emoji} **Verdict: APPROVED**")
                        else:
                            st.error(f"{verdict_emoji} **Verdict: NEEDS IMPROVEMENT**")
                        
                        # Display review details
                        col_rev_img, col_rev_details = st.columns([1, 2])
                        
                        with col_rev_img:
                            st.image(review_image, caption="Reviewed Image", width='stretch')
                        
                        with col_rev_details:
                            st.markdown(f"### 🤖 Reviewer: {reviewer_type}")
                            
                            st.markdown("#### 📝 Description")
                            st.write(review_result.description)
                            
                            st.markdown("#### 🔍 Analysis")
                            st.write(review_result.analysis)
                            
                            if review_result.verdict == "No":
                                st.markdown("#### 🎨 Recommended Instructions")
                                instructions_text = format_recommended_instructions(_resolve_instructions(review_result))
                                st.markdown(instructions_text)
                    
                except Exception as e:
                    st.error(f"Error during review: {e}")
                    import traceback
                    st.error(f"Details: {traceback.format_exc()}")
    else:
        st.info("Please provide an image and slide context to get a review.")

# ============================================================================
# VOICEOVER FOCUS AGENT PAGE
# ============================================================================
elif page == "Voiceover Focus Agent":
    st.title("🎯 Voiceover Focus Agent")
    st.markdown(
        """
        Runs the **adaptive post-pipeline router**:
        it decides between Voiceover Focus (surgical edits) and Illustrator (concept-level visualization)
        before applying edits.

        **Possible interventions:** Zoom · Highlight · Annotate · Remove · No Change
        """
    )
    st.divider()

    # ── Inputs ───────────────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("📷 Final Image")
        vfa_input_method = st.radio(
            "Input method:", ["Upload", "URL"], horizontal=True, key="vfa_input_method"
        )

        vfa_image: Image.Image | None = None

        if vfa_input_method == "URL":
            vfa_url = st.text_input(
                "Image URL",
                placeholder="https://…",
                key="vfa_url",
            )
            if vfa_url:
                try:
                    vfa_image = _load_image_from_url(vfa_url, timeout=20)
                    st.image(vfa_image, caption="Loaded Image", width="stretch")
                except Exception as exc:
                    st.error(f"Failed to load image from URL: {exc}")
        else:
            vfa_uploaded = st.file_uploader(
                "Upload Image",
                type=["png", "jpg", "jpeg", "webp"],
                key="vfa_upload",
            )
            if vfa_uploaded:
                vfa_image = Image.open(vfa_uploaded)
                st.image(vfa_image, caption="Uploaded Image", width="stretch")

    with col_right:
        st.subheader("🎙️ Voiceover Segment")
        vfa_voiceover = st.text_area(
            "Voiceover narration for this slide segment",
            placeholder=(
                "e.g. 'The blue inlet valve on the left side of the pump housing "
                "controls the rate of fluid entering the system …'"
            ),
            height=160,
            key="vfa_voiceover",
        )

        st.subheader("📝 Optional Slide Context")
        vfa_slide_title = st.text_input(
            "Slide title",
            placeholder="Optional: enter slide title",
            key="vfa_slide_title",
        )
        vfa_slide_content = st.text_area(
            "Slide content",
            placeholder="Optional: enter slide content",
            height=90,
            key="vfa_slide_content",
        )
        vfa_visual_instruction = st.text_area(
            "Visual instruction",
            placeholder="Optional: enter intended visual instruction",
            height=90,
            key="vfa_visual_instruction",
        )

        st.subheader("📋 Styling Guide (optional override)")
        vfa_styling_override = st.text_area(
            "Paste a custom styling guide, or leave blank to use the project default.",
            height=100,
            key="vfa_styling_override",
        )

    st.divider()

    # ── Run button ────────────────────────────────────────────────────────────
    can_run = vfa_image is not None and bool((vfa_voiceover or "").strip())
    run_btn = st.button(
        "🎯 Run Adaptive Focus Router",
        type="primary",
        disabled=not can_run,
        width="stretch",
        key="vfa_run_btn",
    )

    if not can_run:
        st.info("Provide both an image and the voiceover text to enable the agent.")

    if run_btn and can_run:
        styling_text = vfa_styling_override.strip() if vfa_styling_override else ""

        # ── Status / progress area ────────────────────────────────────────────
        status_area = st.empty()
        with status_area.container():
            st.info("⏳ Running pre-analysis route decision, then selected agent path …")

        try:
            # Keep compatibility with older loaded module versions in Streamlit runtime.
            requested_kwargs = {
                "image": vfa_image,
                "voiceover": vfa_voiceover.strip(),
                "styling_guide_text": styling_text,
                "slide_title": (vfa_slide_title or "").strip(),
                "slide_content": (vfa_slide_content or "").strip(),
                "visual_instruction": (vfa_visual_instruction or "").strip(),
            }
            supported_params = set(inspect.signature(run_adaptive_focus_or_illustrator).parameters.keys())
            call_kwargs = {k: v for k, v in requested_kwargs.items() if k in supported_params}
            result = run_adaptive_focus_or_illustrator(**call_kwargs)

            status_area.empty()

            analysis: VoiceFocusAnalysis = result.analysis
            references_specific_element = bool((analysis.referenced_element or "").strip())
            intervention_types_raw = getattr(analysis, "intervention_types", []) or []
            intervention_types = [
                (x.value if hasattr(x, "value") else str(x)).strip()
                for x in intervention_types_raw
                if (x.value if hasattr(x, "value") else str(x)).strip()
            ]
            if not intervention_types:
                intervention_types = [analysis.intervention_type.value]
            editor_instructions = [x for x in getattr(analysis, "editor_instructions", []) if isinstance(x, str) and x.strip()]
            intervention_justified = any(t != "none" for t in intervention_types) and (
                bool((analysis.editor_instruction or "").strip()) or bool(editor_instructions)
            )
            free_text_analysis = getattr(analysis, "free_text_analysis", "")
            module_editing_instructions = getattr(analysis, "module_editing_instructions", {}) or {}
            if hasattr(module_editing_instructions, "model_dump"):
                module_editing_instructions = module_editing_instructions.model_dump()
            elif not isinstance(module_editing_instructions, dict):
                module_editing_instructions = {}

            # ── Analysis summary card ─────────────────────────────────────────
            st.subheader("🧠 Agent Reasoning")

            # Colour the decision banner
            if not intervention_justified:
                st.success(
                    f"✅ **No intervention needed** — {analysis.justification}"
                )
            else:
                itype_label = ", ".join([t.upper() for t in intervention_types if t != "none"]) or "NONE"
                st.warning(
                    f"⚙️ **Intervention: {itype_label}** — {analysis.justification}"
                )

            st.markdown("**Free-text clutter/noise analysis (Call 1)**")
            st.write(free_text_analysis or "No free-text analysis available.")

            # Expandable reasoning detail
            with st.expander("📊 Full Analysis Detail", expanded=False):
                c_a, c_b = st.columns(2)
                with c_a:
                    st.markdown("**Voiceover topic**")
                    st.write(analysis.voiceover_topic)

                    st.markdown("**References a specific element?**")
                    st.write("Yes ✅" if references_specific_element else "No ❌")

                    if analysis.referenced_element:
                        st.markdown("**Referenced element**")
                        st.write(analysis.referenced_element)

                with c_b:
                    st.markdown("**Intervention type selected**")
                    st.code(", ".join(intervention_types), language=None)

                if analysis.editor_instruction:
                    st.markdown("**Editor instruction sent**")
                    st.info(analysis.editor_instruction)

                if editor_instructions:
                    st.markdown("**Editor instructions (ordered)**")
                    for idx, instruction in enumerate(editor_instructions, start=1):
                        st.write(f"{idx}. {instruction}")

                st.markdown("**Editing instructions sent to image editing module**")
                st.json(module_editing_instructions)

                if free_text_analysis:
                    st.markdown("**Free-text clutter/noise analysis (Call 1)**")
                    st.write(free_text_analysis)

            # ── Image comparison ──────────────────────────────────────────────
            st.subheader("🖼️ Result")
            col_orig, col_out = st.columns(2)

            with col_orig:
                st.markdown("**Original Image**")
                st.image(vfa_image, width="stretch")

            with col_out:
                if result.was_modified:
                    st.markdown(f"**Modified Image** *(intervention: {', '.join(intervention_types)})*")
                    st.image(result.output_image, width="stretch")
                    st.caption(result.modification_notes)
                else:
                    st.markdown("**Output Image** *(unchanged)*")
                    st.image(result.output_image, width="stretch")
                    st.caption("No modifications were applied.")

            # ── Download ──────────────────────────────────────────────────────
            st.divider()
            dl_buf = BytesIO()
            out_img = result.output_image
            if not isinstance(out_img, Image.Image):
                out_img = vfa_image   # safety fallback
            out_img_rgb = out_img.copy()
            if out_img_rgb.mode != "RGB":
                out_img_rgb = out_img_rgb.convert("RGB")
            out_img_rgb.save(dl_buf, format="PNG")

            filename = (
                "focus_modified_image.png"
                if result.was_modified
                else "focus_unchanged_image.png"
            )

            dl_col, report_col = st.columns(2)
            with dl_col:
                st.download_button(
                    label="⬇️ Download Output Image",
                    data=dl_buf.getvalue(),
                    file_name=filename,
                    mime="image/png",
                    width="stretch",
                )

            # Markdown report
            report_md = f"""# Voiceover Focus Agent Report

## Inputs
**Voiceover:** {vfa_voiceover.strip()}
**Slide Title:** {(vfa_slide_title or '').strip() or 'N/A'}
**Slide Content:** {(vfa_slide_content or '').strip() or 'N/A'}
**Visual Instruction:** {(vfa_visual_instruction or '').strip() or 'N/A'}

## Analysis
| Field | Value |
|---|---|
| Voiceover Topic | {analysis.voiceover_topic} |
| References Specific Element | {references_specific_element} |
| Referenced Element | {analysis.referenced_element or 'N/A'} |
| Intervention Justified | {intervention_justified} |
| Intervention Type(s) | {', '.join(intervention_types)} |

## Free-text Clutter/Noise Analysis (Call 1)
{free_text_analysis or 'N/A'}

## Justification
{analysis.justification}

## Editor Instruction
{analysis.editor_instruction or 'N/A'}

## Editor Instructions (Ordered)
{chr(10).join([f"{i + 1}. {ins}" for i, ins in enumerate(editor_instructions)]) if editor_instructions else 'N/A'}

## Editing Instructions Sent To Image Editing Module
{module_editing_instructions}

## Modification Notes
{result.modification_notes}
"""
            with report_col:
                st.download_button(
                    label="📄 Download Report (Markdown)",
                    data=report_md,
                    file_name="voiceover_focus_report.md",
                    mime="text/markdown",
                    width="stretch",
                )

            # ── Reset ─────────────────────────────────────────────────────────
            if st.button("🔄 Reset", key="vfa_reset"):
                for k in [
                    "vfa_url",
                    "vfa_voiceover",
                    "vfa_styling_override",
                    "vfa_slide_title",
                    "vfa_slide_content",
                    "vfa_visual_instruction",
                ]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()

        except Exception as exc:
            status_area.empty()
            st.error(f"Agent error: {exc}")
            import traceback as _tb
            st.error(_tb.format_exc())

# ============================================================================
# TECHNICAL ACCURACY VALIDATOR AGENT PAGE
# ============================================================================
elif page == "Technical Accuracy Validator Agent":
    st.title("🛠️ Technical Accuracy Validator Agent")
    st.markdown(
        """
        Compares **reference image** against **final image** and classifies each detected
        difference as either **creative liberty** or **technical inaccuracy**.

        If technical inaccuracies are confirmed, the agent applies only targeted technical
        corrections while preserving valid creative liberties.
        """
    )
    st.divider()

    col_ref, col_final = st.columns(2, gap="large")

    def _render_image_input(input_key_prefix: str, title: str):
        st.subheader(title)
        method = st.radio(
            "Input method:", ["Upload", "URL"], horizontal=True, key=f"{input_key_prefix}_method"
        )

        image_obj = None
        if method == "URL":
            url = st.text_input("Image URL", placeholder="https://...", key=f"{input_key_prefix}_url")
            if url:
                try:
                    image_obj = _load_image_from_url(url, timeout=20)
                    st.image(image_obj, caption=f"Loaded {title}", width="stretch")
                except Exception as exc:
                    st.error(f"Failed to load image from URL: {exc}")
        else:
            uploaded = st.file_uploader(
                "Upload Image",
                type=["png", "jpg", "jpeg", "webp"],
                key=f"{input_key_prefix}_upload",
            )
            if uploaded:
                image_obj = Image.open(uploaded)
                st.image(image_obj, caption=f"Uploaded {title}", width="stretch")

        return image_obj

    with col_ref:
        tav_reference_image = _render_image_input("tav_ref", "📷 Reference Image (Input)")

    with col_final:
        tav_final_image = _render_image_input("tav_final", "🖼️ Final Image (Post-Agent 1)")

    st.divider()

    st.subheader("📝 Context (same as Agent 1)")
    tav_voiceover = st.text_area(
        "Voiceover narration for this slide segment",
        placeholder="Enter the voiceover text for this segment...",
        height=160,
        key="tav_voiceover",
    )

    tav_slide_title = st.text_input(
        "Slide title",
        placeholder="Optional: enter slide title",
        key="tav_slide_title",
    )

    tav_slide_content = st.text_area(
        "Slide content",
        placeholder="Optional: enter slide content",
        height=90,
        key="tav_slide_content",
    )

    can_run_tav = (
        tav_reference_image is not None
        and tav_final_image is not None
        and bool((tav_voiceover or "").strip())
    )

    run_tav_btn = st.button(
        "🛠️ Run Technical Accuracy Validator",
        type="primary",
        disabled=not can_run_tav,
        width="stretch",
        key="tav_run_btn",
    )

    if not can_run_tav:
        st.info("Provide reference image, final image, and voiceover text to enable the agent.")

    if run_tav_btn and can_run_tav:
        status_area = st.empty()
        with status_area.container():
            st.info("⏳ Running Agent 2 technical validation and correction...")

        try:
            requested_kwargs = {
                "reference_image": tav_reference_image,
                "final_image": tav_final_image,
                "voiceover": tav_voiceover.strip(),
                "slide_title": (tav_slide_title or "").strip(),
                "slide_content": (tav_slide_content or "").strip(),
            }
            supported_params = set(inspect.signature(run_technical_accuracy_validator_agent).parameters.keys())
            call_kwargs = {k: v for k, v in requested_kwargs.items() if k in supported_params}
            result = run_technical_accuracy_validator_agent(**call_kwargs)

            status_area.empty()

            analysis: TechnicalValidationAnalysis = result.analysis
            verdict = (analysis.verdict or "approved").strip().lower()
            region_verdicts = analysis.region_verdicts or []

            if verdict == "approved":
                st.success(f"✅ Approved — {analysis.justification or 'No technical inaccuracies detected.'}")
            else:
                st.warning(
                    f"⚙️ Correction required — {analysis.justification or 'Technical inaccuracies were found and corrected.'}"
                )

            st.subheader("🧠 Agent Analysis")
            st.markdown("**Call 1 free-text analysis**")
            st.write(analysis.free_text_analysis or "No free-text analysis available.")

            with st.expander("📊 Structured Decision Detail", expanded=False):
                st.markdown("**Reference technical summary**")
                st.write(analysis.reference_summary or "N/A")

                st.markdown("**Final image summary**")
                st.write(analysis.final_summary or "N/A")

                st.markdown("**Region classifications**")
                if region_verdicts:
                    for idx, rv in enumerate(region_verdicts, start=1):
                        st.write(f"{idx}. Region: {rv.region_name or 'N/A'}")
                        st.write(f"   Category: {rv.category}")
                        st.write(f"   Reason: {rv.reason or 'N/A'}")
                        if rv.correction_instruction:
                            st.write(f"   Correction: {rv.correction_instruction}")
                else:
                    st.write("No region-level classifications returned.")

                st.markdown("**Compromised technical details**")
                if analysis.compromised_details:
                    for item in analysis.compromised_details:
                        st.write(f"- {item}")
                else:
                    st.write("None")

                st.markdown("**Ordered corrections**")
                if analysis.ordered_corrections:
                    for i, instruction in enumerate(analysis.ordered_corrections, start=1):
                        st.write(f"{i}. {instruction}")
                else:
                    st.write("None")

                st.markdown("**Editing instructions sent to image editing module**")
                module_editing_instructions = analysis.module_editing_instructions
                if hasattr(module_editing_instructions, "model_dump"):
                    module_editing_instructions = module_editing_instructions.model_dump()
                st.json(module_editing_instructions)

            st.subheader("🖼️ Result")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Reference Image**")
                st.image(tav_reference_image, width="stretch")
                st.markdown("**Input Final Image**")
                st.image(tav_final_image, width="stretch")
            with col_b:
                if result.was_modified:
                    st.markdown("**Output Image (Corrected)**")
                    st.image(result.output_image, width="stretch")
                else:
                    st.markdown("**Output Image (Unchanged)**")
                    st.image(result.output_image, width="stretch")
                st.caption(result.modification_notes)

            out_img = result.output_image if isinstance(result.output_image, Image.Image) else tav_final_image
            out_buf = BytesIO()
            out_rgb = out_img.copy()
            if out_rgb.mode != "RGB":
                out_rgb = out_rgb.convert("RGB")
            out_rgb.save(out_buf, format="PNG")

            out_name = "technical_accuracy_corrected.png" if result.was_modified else "technical_accuracy_unchanged.png"

            report_lines = [
                "# Technical Accuracy Validator Agent Report",
                "",
                "## Inputs",
                f"**Voiceover:** {tav_voiceover.strip()}",
                f"**Slide Title:** {(tav_slide_title or '').strip() or 'N/A'}",
                f"**Slide Content:** {(tav_slide_content or '').strip() or 'N/A'}",
                "",
                "## Verdict",
                f"**Verdict:** {analysis.verdict}",
                f"**Justification:** {analysis.justification or 'N/A'}",
                "",
                "## Call 1 Free-text Analysis",
                analysis.free_text_analysis or "N/A",
                "",
                "## Region Classifications",
            ]

            if region_verdicts:
                for idx, rv in enumerate(region_verdicts, start=1):
                    report_lines.append(
                        f"{idx}. Region: {rv.region_name or 'N/A'} | Category: {rv.category} | Reason: {rv.reason or 'N/A'}"
                    )
                    if rv.correction_instruction:
                        report_lines.append(f"   - Correction: {rv.correction_instruction}")
            else:
                report_lines.append("None")

            report_lines.extend([
                "",
                "## Compromised Details",
                *([f"- {x}" for x in analysis.compromised_details] if analysis.compromised_details else ["None"]),
                "",
                "## Ordered Corrections",
                *([f"{i}. {x}" for i, x in enumerate(analysis.ordered_corrections, start=1)] if analysis.ordered_corrections else ["None"]),
                "",
                "## Modification Notes",
                result.modification_notes,
            ])

            report_md = "\n".join(report_lines)

            dl_img_col, dl_report_col = st.columns(2)
            with dl_img_col:
                st.download_button(
                    label="⬇️ Download Output Image",
                    data=out_buf.getvalue(),
                    file_name=out_name,
                    mime="image/png",
                    width="stretch",
                )
            with dl_report_col:
                st.download_button(
                    label="📄 Download Report (Markdown)",
                    data=report_md,
                    file_name="technical_accuracy_validator_report.md",
                    mime="text/markdown",
                    width="stretch",
                )

            if st.button("🔄 Reset", key="tav_reset"):
                for k in [
                    "tav_ref_url",
                    "tav_final_url",
                    "tav_voiceover",
                    "tav_slide_title",
                    "tav_slide_content",
                ]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()

        except Exception as exc:
            status_area.empty()
            st.error(f"Error during technical validation: {exc}")
            import traceback

            st.error(f"Details: {traceback.format_exc()}")

# ============================================================================
# ILLUSTRATOR AGENT PAGE
# ============================================================================
elif page == "Illustrator Agent":
    st.title("🎨 Illustrator Agent")
    st.markdown(
        """
        Transforms images into illustration-style visuals that make the voiceover concept self-explanatory.
        """
    )
    ca_mode = st.radio("Execution Mode:", ["Single Image", "Batch Sheet Automation"], horizontal=True, key="ca_exec_mode")
    st.divider()

    if ca_mode == "Batch Sheet Automation":
        st.subheader("📊 Batch Execution Settings")
        
        ca_batch_url = st.text_input("Google Sheets URL", placeholder="https://docs.google.com/spreadsheets/d/...")
        ca_batch_tab = st.text_input("Source Tab Name", "Sheet1")
        ca_batch_folder = st.text_input("Output Drive Folder Name", "Illustrator Agent Outputs")
        
        st.info("Ensure headers match exactly: `Slide Chunk Title`, `Slide Chunk`, `Voiceover`, `Output Image`.")
        
        if st.button("🚀 Start Batch Processing", type="primary", width="stretch"):
            if not ca_batch_url:
                st.error("Please provide a valid Google Sheets URL.")
            else:
                from agents.graphics_asset_creation.automated.automated_illustrator import run_illustrator_automation
                from services.drive_service import login_with_service_account
                import gspread
                
                # Setup Auth
                try:
                    if "gc" in st.session_state and "drive" in st.session_state:
                        gc = st.session_state["gc"]
                        drive = st.session_state["drive"]
                        st.info("Using authenticated session from main app...")
                    else:
                        sa_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
                        if not sa_json:
                            st.error("No authenticated session found and GOOGLE_SERVICE_ACCOUNT_JSON not set.")
                            st.stop()
                        gc = gspread.service_account_from_dict(json.loads(sa_json))
                        gauth = login_with_service_account(json_str=sa_json)
                        from pydrive2.drive import GoogleDrive
                        drive = GoogleDrive(gauth)
                        
                    progress_bar = st.progress(0, text="Initializing Data...")
                    
                    def ca_prog(curr, total):
                        pct = curr / total if total > 0 else 1.0
                        progress_bar.progress(pct, text=f"Processing {curr}/{total} rows...")
                        
                    with st.spinner("Processing rows in parallel..."):
                        run_illustrator_automation(
                            sheet_url=ca_batch_url,
                            source_tab=ca_batch_tab,
                            output_folder_name=ca_batch_folder,
                            gc=gc,
                            drive=drive,
                            progress_callback=ca_prog
                        )
                        
                    progress_bar.progress(1.0, text="✅ Processing Complete!")
                    st.success("Batch successfully written back to the sheet!")
                    st.balloons()
                        
                except Exception as e:
                    st.error(f"Execution Failed: {e}")
                    import traceback
                    st.error(traceback.format_exc())
                    
        st.stop()

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("📷 Image Input")
        ca_input_method = st.radio(
            "Input method:", ["Upload", "URL"], horizontal=True, key="ca_input_method"
        )

        ca_image = None
        if ca_input_method == "URL":
            ca_url = st.text_input("Image URL", placeholder="https://...", key="ca_url")
            if ca_url:
                try:
                    ca_image = _load_image_from_url(ca_url, timeout=20)
                    st.image(ca_image, caption="Loaded Image", width="stretch")
                except Exception as exc:
                    st.error(f"Failed to load image: {exc}")
        else:
            ca_uploaded = st.file_uploader(
                "Upload Image", type=["png", "jpg", "jpeg", "webp"], key="ca_upload"
            )
            if ca_uploaded:
                ca_image = Image.open(ca_uploaded)
                st.image(ca_image, caption="Uploaded Image", width="stretch")

    with col_right:
        st.subheader("📝 Context")
        ca_voiceover = st.text_area(
            "Voiceover narration",
            placeholder="Enter the voiceover text...",
            height=160,
            key="ca_voiceover",
        )
        ca_slide_title = st.text_input(
            "Slide title", placeholder="Optional: enter slide title", key="ca_slide_title"
        )
        ca_slide_content = st.text_area(
            "Slide content", placeholder="Optional: enter slide content", height=90, key="ca_slide_content"
        )
        ca_visual_instruction = st.text_area(
            "Visual instruction", placeholder="Optional: enter intended visual instruction", height=90, key="ca_visual_instruction"
        )

    st.divider()

    from agents.graphics_asset_creation.illustrator.illustrator_agent import (
        run_illustrator_pipeline,
    )

    if 'ca_result' not in st.session_state:
        st.session_state.ca_result = None
    if 'ca_edited_image' not in st.session_state:
        st.session_state.ca_edited_image = None

    can_run_ca = ca_image is not None and bool((ca_voiceover or "").strip())
    run_ca_btn = st.button(
        "🎯 Run Illustrator Analysis + Generation",
        type="primary",
        disabled=not can_run_ca,
        width="stretch",
        key="ca_run_btn",
    )

    if not can_run_ca:
        st.info("Provide an image and voiceover text to enable the agent.")

    if run_ca_btn and can_run_ca:
        with st.spinner("⏳ Running Illustrator pipeline (analysis + generation)..."):
            try:
                result, edited_image, ui_log, conv_history = run_illustrator_pipeline(
                    image=ca_image,
                    voiceover=ca_voiceover.strip(),
                    slide_title=(ca_slide_title or "").strip(),
                    slide_content=(ca_slide_content or "").strip(),
                    visual_instruction=(ca_visual_instruction or "").strip(),
                    image_size="1K",
                    aspect_ratio="16:9",
                    quick_mode=True,
                )
                st.session_state.ca_result = result
                st.session_state.ca_edited_image = edited_image
                st.rerun()
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                import traceback as _tb
                st.error(_tb.format_exc())

    if st.session_state.ca_result is not None:
        result = st.session_state.ca_result
        st.success("✅ Analysis Complete")
        
        st.subheader("🧠 Illustrator Analysis")
        st.markdown("**Image Description:**")
        st.write(result.image_description)
        
        st.markdown("**Voiceover Concept Gap:**")
        st.write(result.voiceover_gap_analysis)
        
        st.subheader("🎨 Recommended Instructions")
        st.info(f"**Confidence Score:** {result.confidence_score}/100")
        if result.recommended_instructions:
            col1, col2 = st.columns(2)
            with col1:
                if getattr(result.recommended_instructions, "visual_style", ""):
                    st.markdown("**Visual Style:**")
                    st.write(result.recommended_instructions.visual_style)
                if getattr(result.recommended_instructions, "lighting_and_environment", ""):
                    st.markdown("**Lighting & Environment:**")
                    st.write(result.recommended_instructions.lighting_and_environment)
            with col2:
                if getattr(result.recommended_instructions, "color_grading", ""):
                    st.markdown("**Color Grading:**")
                    st.write(result.recommended_instructions.color_grading)
            
            additional_comments = getattr(result.recommended_instructions, "additional_comments", "")
            if additional_comments:
                st.markdown("**Additional Comments:**")
                st.write(additional_comments)
        
    if st.session_state.ca_edited_image is not None:
        st.success("✅ Image Generation Complete")
        col_orig, col_edit = st.columns(2)
        with col_orig:
            st.markdown("**Original Image**")
            st.image(ca_image, width="stretch")
        with col_edit:
            st.markdown("**Generated Illustration**")
            st.image(st.session_state.ca_edited_image, width="stretch")
            
        buf = BytesIO()
        st.session_state.ca_edited_image.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download Generated Illustration",
            data=buf.getvalue(),
            file_name="compliance_illustrated_asset.png",
            mime="image/png",
            width='stretch'
        )

        st.markdown("---")
        if st.button("🔄 Start Over", width="stretch"):
            for k in ["ca_result", "ca_edited_image"]:
                if k in st.session_state: del st.session_state[k]
            st.rerun()

# ============================================================================
# IMAGE GENERATOR (TEST) PAGE
# ============================================================================
elif page == "Image Generator (Test)":
    import importlib
    import agents.graphics_asset_creation.generator.image_generator as ig_module
    importlib.reload(ig_module)
    from agents.graphics_asset_creation.generator.image_generator import generate_asset, run_analysis_stage, run_generation_stage

    st.title("🚀 Image Generator & Review Loop")
    st.markdown("Generate high-quality HVAC educational images with convergent review loops.")
    
    gen_mode = st.radio("Execution Mode:", ["Single Image", "Batch Sheet Automation"], horizontal=True, key="gen_exec_mode")
    st.divider()

    if gen_mode == "Batch Sheet Automation":
        st.subheader("📊 Batch Generation Settings")
        
        gen_batch_url = st.text_input("Google Sheets URL", placeholder="https://docs.google.com/spreadsheets/d/...")
        gen_batch_tab = st.text_input("Source Tab Name", "Sheet1")
        gen_batch_folder = st.text_input("Output Drive Folder Name", "Generated HVAC Assets")
        
        st.info("Required Headers: `Slide Chunk Title`, `Slide Chunk`, `Voiceover`. Outputs: `Generated Brief`, `Generated Asset Image`.")
        
        if st.button("🚀 Start Batch Generation", type="primary", use_container_width=True):
            if not gen_batch_url:
                st.error("Please provide a valid Google Sheets URL.")
            else:
                from agents.graphics_asset_creation.automated.automated_generation import run_generation_automation
                from services.drive_service import login_with_service_account
                import gspread
                
                try:
                    if "gc" in st.session_state and "drive" in st.session_state:
                        gc = st.session_state["gc"]
                        drive = st.session_state["drive"]
                        st.info("Using authenticated session from main app...")
                    else:
                        import base64
                        sa_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

                        if not sa_json:
                            # Fallback to GDRIVE_SA_B64 if available
                            sa_b64 = os.getenv("GDRIVE_SA_B64")
                            if sa_b64:
                                try:
                                    sa_json = base64.b64decode(sa_b64).decode("utf-8")
                                except Exception as e:
                                    st.error(f"Failed to decode GDRIVE_SA_B64: {e}")
                                    st.stop()
                            else:
                                st.error("Neither GOOGLE_SERVICE_ACCOUNT_JSON nor GDRIVE_SA_B64 environment variables are set. Please check your .env file.")
                                st.stop()

                        gc = gspread.service_account_from_dict(json.loads(sa_json))
                        gauth = login_with_service_account(json_str=sa_json)
                        from pydrive2.drive import GoogleDrive
                        drive = GoogleDrive(gauth)
                    
                    progress_bar = st.progress(0, text="Initializing...")
                    
                    def gen_prog(curr, total):
                        pct = curr / total if total > 0 else 1.0
                        progress_bar.progress(pct, text=f"Generated {curr}/{total} images...")
                        
                    with st.spinner("Batch generating images..."):
                        run_generation_automation(
                            sheet_url=gen_batch_url,
                            source_tab=gen_batch_tab,
                            output_folder_name=gen_batch_folder,
                            gc=gc,
                            drive=drive,
                            progress_callback=gen_prog
                        )
                        
                    progress_bar.progress(1.0, text="✅ Batch Complete!")
                    st.success("Assets generated and written back to sheet!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Batch failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        st.stop()

    # ── Session State Initialization ─────────────────────────────────────────
    if 'gs_stage' not in st.session_state:
        st.session_state.gs_stage = 1
    if 'gs_history' not in st.session_state:
        st.session_state.gs_history = []
    if 'gs_brief' not in st.session_state:
        st.session_state.gs_brief = None
    if 'gs_image' not in st.session_state:
        st.session_state.gs_image = None
    
    # ── Inputs ───────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        gen_slide_title = st.text_input("Slide Title", placeholder="e.g. Refrigerant cycle ...")
    with col2:
        # User requested to only provide title, content and voiceover.
        pass
        
    gen_slide_content = st.text_area("Slide Content", height=100)
    gen_voiceover = st.text_area("Voiceover (Focus Segment)", height=100)
    
    st.divider()

    # ── STAGE 1 — BRIEF GENERATION ───────────────────────────────────────────
    st.header("Step 1: Instruction Generation Brief")
    
    if st.session_state.gs_stage == 1:
        run_s1 = st.button("📝 Run Step 1: Generate Brief", type="primary", use_container_width=True)
        if run_s1:
            if not gen_voiceover.strip():
                st.error("Provide a voiceover segment to begin.")
            else:
                with st.spinner("⏳ Running Stage 1: Analysis & Iterative Brief Review..."):
                    try:
                        brief_history = []
                        brief = run_analysis_stage(
                            slide_title=gen_slide_title,
                            slide_content=gen_slide_content,
                            voiceover_focus=gen_voiceover,
                            revision_history=brief_history
                        )
                        st.session_state.gs_brief = brief
                        st.session_state.gs_history.extend(brief_history)
                        st.session_state.gs_stage = 2
                        st.rerun()
                    except Exception as e:
                        st.error(f"Stage 1 failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())
    else:
        st.success("✅ Step 1 Complete: JSON Brief Finalized.")
        with st.expander("📝 View Final Approved JSON Brief", expanded=True):
            st.code(st.session_state.gs_brief, language="json")
        
        # Show Feedback History for Step 1
        s1_history = [r for r in st.session_state.gs_history if r.get("stage") == "instruction"]
        num_revisions_s1 = sum(1 for r in s1_history if r.get("verdict") == "REVISE")
        
        if num_revisions_s1 > 0:
            with st.expander(f"👁️ View Step 1 Feedback Attempts ({num_revisions_s1} revisions)", expanded=False):
                for round_data in s1_history:
                    if round_data.get("verdict") == "REVISE":
                        st.markdown(f"**Round {round_data['round']} Feedback:**")
                        for issue in round_data.get("new_issues", []):
                            st.write(f"• {issue}")
                        st.divider()
        else:
            st.info("Direct approval: Brief was perfect on the first attempt.")

    # ── STAGE 2 — IMAGE GENERATION ───────────────────────────────────────────
    if st.session_state.gs_stage >= 2:
        st.divider()
        st.header("Step 2: Image Generation")
        
        if st.session_state.gs_stage == 2:
            run_s2 = st.button("🖼️ Run Step 2: Generate Image", type="primary", use_container_width=True)
            if run_s2:
                with st.spinner("⏳ Running Stage 2: Multimodal Image Review & Generation..."):
                    try:
                        image_history = []
                        image, meta = run_generation_stage(
                            approved_json_brief=st.session_state.gs_brief,
                            slide_title=gen_slide_title,
                            slide_content=gen_slide_content,
                            voiceover_focus=gen_voiceover,
                            aspect_ratio="16:9",
                            image_size="1K",
                            revision_history=image_history
                        )
                        st.session_state.gs_image = image
                        st.session_state.gs_history.extend(image_history)
                        st.session_state.gs_stage = 3
                        st.rerun()
                    except Exception as e:
                        st.error(f"Stage 2 failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        
        elif st.session_state.gs_stage == 3:
            st.success("✅ Step 2 Complete: Professional HVAC Asset Generated.")
            st.image(st.session_state.gs_image, caption="Final Approved Asset", use_container_width=True)
            
            # Show Feedback History for Step 2
            s2_history = [r for r in st.session_state.gs_history if r.get("stage") == "image"]
            num_revisions_s2 = sum(1 for r in s2_history if r.get("verdict") == "REVISE")
            
            if num_revisions_s2 > 0:
                with st.expander(f"👁️ View Step 2 Feedback Attempts ({num_revisions_s2} revisions)", expanded=False):
                    for round_data in s2_history:
                        if round_data.get("verdict") == "REVISE":
                            st.markdown(f"**Round {round_data['round']} Feedback:**")
                            for issue in round_data.get("new_issues", []):
                                st.write(f"• {issue}")
                            st.divider()
            else:
                st.info("Direct approval: Image was accurate and clear on the first attempt.")

            # Download
            buf = BytesIO()
            st.session_state.gs_image.save(buf, format="PNG")
            st.download_button(
                label="⬇️ Download Final Asset",
                data=buf.getvalue(),
                file_name="hvac_technical_graphic.png",
                mime="image/png",
                use_container_width=True
            )

    # ── Reset UI ─────────────────────────────────────────────────────────────
    if st.session_state.gs_stage > 1:
        st.divider()
        if st.button("🔄 Reset Workflow", use_container_width=True):
            for k in ["gs_stage", "gs_history", "gs_brief", "gs_image"]:
                if k in st.session_state: del st.session_state[k]
            st.rerun()
