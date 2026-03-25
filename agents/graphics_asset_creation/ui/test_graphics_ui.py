"""
Minimal Streamlit UI for Testing Graphics Creation Functions
"""
import streamlit as st
import sys
from pathlib import Path
import os

# Add parent directory to path to import graphics_creation
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    generate_graphics_definition,
    analyze_reference_image_quality,
    identify_link_type,
    generate_image_with_review_loop,
    process_drive_image,
    process_web_image
)
from PIL import Image
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Graphics Creation Workflow",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 Graphics Creation Workflow")
st.markdown("**End-to-end workflow with Technical Accuracy Validation at every step**")
st.markdown("⚡ **Technical Accuracy Checker** validates outputs before proceeding")
st.markdown("---")

# Initialize session state
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "definition_result" not in st.session_state:
    st.session_state.definition_result = None
if "generated_image" not in st.session_state:
    st.session_state.generated_image = None
if "generation_history" not in st.session_state:
    st.session_state.generation_history = []
if "reference_image" not in st.session_state:
    st.session_state.reference_image = None
# Context information
if "slide_title" not in st.session_state:
    st.session_state.slide_title = "Introduction to Cloud Computing"
if "slide_chunk" not in st.session_state:
    st.session_state.slide_chunk = "Cloud computing provides on-demand access to computing resources like servers, storage, and databases over the internet. It enables scalability and cost efficiency."
if "voiceover" not in st.session_state:
    st.session_state.voiceover = "on-demand access to computing resources"
if "visual_instruction" not in st.session_state:
    st.session_state.visual_instruction = "Show cloud infrastructure with servers and databases connected"
if "visual_style" not in st.session_state:
    st.session_state.visual_style = "Illustration"

# Common inputs section
st.header("📋 Context Information")
st.markdown("*This information will be used throughout all steps*")
col1, col2 = st.columns(2)

with col1:
    slide_title = st.text_input(
        "Slide Title",
        value=st.session_state.slide_title,
        key="input_slide_title",
        help="The title of the slide"
    )
    if slide_title != st.session_state.slide_title:
        st.session_state.slide_title = slide_title
    
    slide_chunk = st.text_area(
        "Slide Content",
        value=st.session_state.slide_chunk,
        key="input_slide_chunk",
        height=120,
        help="The full content of the slide"
    )
    if slide_chunk != st.session_state.slide_chunk:
        st.session_state.slide_chunk = slide_chunk

with col2:
    voiceover = st.text_input(
        "Voiceover Focus",
        value=st.session_state.voiceover,
        key="input_voiceover",
        help="The specific phrase or concept to emphasize"
    )
    if voiceover != st.session_state.voiceover:
        st.session_state.voiceover = voiceover
    
    visual_instruction = st.text_area(
        "Visual Instruction",
        value=st.session_state.visual_instruction,
        key="input_visual_instruction",
        height=120,
        help="Simple instruction for what visual to create"
    )
    if visual_instruction != st.session_state.visual_instruction:
        st.session_state.visual_instruction = visual_instruction
    
    visual_style = st.radio(
        "Visual Style",
        options=["Illustration", "Real World Image"],
        index=0 if st.session_state.visual_style == "Illustration" else 1,
        horizontal=True,
        help="Choose between stylized illustration or photorealistic real-world image"
    )
    if visual_style != st.session_state.visual_style:
        st.session_state.visual_style = visual_style

st.markdown("---")

# Step 1: Reference Image Analysis
st.header("📍 Step 1: Reference Image Analysis")
st.markdown("*Provide a reference image to get suggested modifications*")

reference_image_path = st.text_input(
    "Reference Image Path/URL",
    value="",
    placeholder="https://drive.google.com/file/d/... or https://example.com/image.jpg",
    help="Enter a local file path, Google Drive link, or web URL"
)

if reference_image_path:
    link_type = identify_link_type(reference_image_path)
    if link_type == "drive":
        st.info("🔗 Google Drive link detected - will be converted to direct download")
    elif link_type == "web":
        st.info("🌐 Web URL detected")
    elif link_type == "unknown" and not os.path.exists(reference_image_path):
        st.warning("⚠️ Unknown link type or file not found")

col_a, col_b, col_c = st.columns([1, 1, 2])
with col_a:
    if st.button("🔍 Analyze Reference Image", type="secondary", disabled=not reference_image_path):
        if all([st.session_state.slide_title, st.session_state.slide_chunk, st.session_state.voiceover, reference_image_path]):
            with st.spinner("Analyzing reference image..."):
                try:
                    analysis = analyze_reference_image_quality(
                        reference_image_path=reference_image_path,
                        slide_chunk=st.session_state.slide_chunk,
                        slide_title=st.session_state.slide_title,
                        voiceover=st.session_state.voiceover
                    )
                    
                    if analysis:
                        st.session_state.analysis_result = analysis
                        
                        # Check if analysis has TA warning prefix
                        if analysis.startswith("⚠️ TECHNICAL ACCURACY WARNING"):
                            st.warning("⚠️ Analysis completed but technical accuracy concerns detected")
                        else:
                            st.success("✅ Analysis completed with technical accuracy validated!")
                        
                        # Load reference image for later use
                        try:
                            link_type = identify_link_type(reference_image_path)
                            image_data = None
                            
                            if link_type == "drive":
                                image_data = process_drive_image(reference_image_path)
                            elif link_type == "web":
                                image_data = process_web_image(reference_image_path)
                            else:
                                # Local file
                                with open(reference_image_path, 'rb') as f:
                                    image_data = f.read()
                            
                            if image_data:
                                st.session_state.reference_image = Image.open(BytesIO(image_data))
                        except Exception as e:
                            st.warning(f"⚠️ Image analyzed but couldn't load for generation: {str(e)}")
                    else:
                        st.error("❌ Failed to analyze image.")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Fill in all context fields first.")

with col_b:
    if st.button("🗑️ Clear Analysis"):
        st.session_state.analysis_result = None
        st.session_state.reference_image = None
        st.rerun()

# Display analysis result
if st.session_state.analysis_result:
    st.markdown("### 📊 Analysis Result:")
    
    # Check for TA warning
    if st.session_state.analysis_result.startswith("⚠️ TECHNICAL ACCURACY WARNING"):
        # Split the warning and analysis
        parts = st.session_state.analysis_result.split("\n", 1)
        if len(parts) == 2:
            st.warning(parts[0])  # Show warning
            st.markdown("**Suggested Modifications:**")
            st.info(parts[1])  # Show analysis
        else:
            st.warning(st.session_state.analysis_result)
    else:
        st.success("✅ **Technical Accuracy Validated**")
        st.markdown("**Suggested Modifications:**")
        st.info(st.session_state.analysis_result)

st.markdown("---")

# Step 2: Generate Graphics Definition
st.header("📍 Step 2: Generate Graphics Definition")

if st.button("🚀 Generate Graphics Definition", type="primary"):
    if all([st.session_state.slide_title, st.session_state.slide_chunk, st.session_state.voiceover, st.session_state.visual_instruction]):
        with st.spinner("Generating graphics definition with technical accuracy validation..."):
            try:
                result = generate_graphics_definition(
                    slide_chunk=st.session_state.slide_chunk,
                    slide_title=st.session_state.slide_title,
                    visual_instruction=st.session_state.visual_instruction,
                    voiceover=st.session_state.voiceover,
                    reference_analysis=st.session_state.analysis_result,  # Pass analysis if available
                    visual_style=st.session_state.visual_style
                )
                
                if result:
                    st.session_state.definition_result = result
                    st.success("✅ Graphics definition generated and validated!")
                else:
                    st.error("❌ Failed to generate graphics definition.")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        st.warning("⚠️ Please fill in all context fields.")

# Display definition result
if st.session_state.definition_result:
    st.markdown("### 📝 Generated Graphics Definition:")
    st.code(st.session_state.definition_result, language="xml")
    
    # Show if analysis was used
    if st.session_state.analysis_result:
        st.success("✨ This definition incorporates the suggested modifications from the reference image analysis!")

st.markdown("---")

# Step 3: Generate Image with Review Loop
st.header("📍 Step 3: Generate Image from Definition")
st.markdown("*Generate an image using Nano Banana Pro based on the graphics definition*")

# Configuration options
col_config1, col_config2, col_config3 = st.columns(3)

with col_config1:
    quick_mode = st.checkbox(
        "Quick Mode",
        value=False,
        help="Skip the review loop and generate once"
    )

with col_config2:
    aspect_ratio = st.selectbox(
        "Aspect Ratio",
        options=["Default", "16:9", "9:16", "1:1", "4:3", "3:4"],
        index=0,
        help="Aspect ratio for the generated image"
    )
    aspect_ratio = None if aspect_ratio == "Default" else aspect_ratio

with col_config3:
    image_size = st.selectbox(
        "Image Size",
        options=["1K", "2K", "4K"],
        index=0,
        help="Resolution of the generated image"
    )

# Generate image button
col_gen1, col_gen2 = st.columns([1, 3])
with col_gen1:
    generate_disabled = not st.session_state.definition_result
    
    if st.button("🎨 Generate Image", type="primary", disabled=generate_disabled):
        if st.session_state.definition_result:
            with st.spinner(f"Generating image {'(Quick Mode)' if quick_mode else '(with Review Loop)'}..."):
                try:
                    final_image, ui_history = generate_image_with_review_loop(
                        graphics_definition=st.session_state.definition_result,
                        slide_chunk=st.session_state.slide_chunk,
                        slide_title=st.session_state.slide_title,
                        voiceover=st.session_state.voiceover,
                        reference_image=st.session_state.reference_image,
                        quick_mode=quick_mode,
                        custom_criteria=None,
                        aspect_ratio=aspect_ratio,
                        image_size=image_size
                    )
                    
                    st.session_state.generated_image = final_image
                    st.session_state.generation_history = ui_history
                    st.success("✅ Image generated successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Image generation failed: {str(e)}")
        else:
            st.warning("⚠️ Please generate a graphics definition first (Step 2).")

with col_gen2:
    if not st.session_state.definition_result:
        st.info("💡 Complete Step 2 first to enable image generation")

# Display generated image
if st.session_state.generated_image:
    st.markdown("### 🖼️ Generated Image:")
    st.image(st.session_state.generated_image, width='stretch')
    
    # Download button
    img_buffer = BytesIO()
    st.session_state.generated_image.save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()
    
    st.download_button(
        label="⬇️ Download Image",
        data=img_bytes,
        file_name="generated_graphics.png",
        mime="image/png"
    )
    
    # Show generation history
    if st.session_state.generation_history:
        with st.expander("📊 View Generation History"):
            total_rounds = len([h for h in st.session_state.generation_history if h['type'] == 'generation'])
            st.write(f"**Total Rounds:** {total_rounds}")
            
            for entry in st.session_state.generation_history:
                if entry['type'] == 'generation':
                    st.write(f"**Round {entry['round']} - Generation**")
                    if 'metadata' in entry:
                        st.write(f"- Input tokens: {entry['metadata'].get('input_tokens', 'N/A')}")
                        st.write(f"- Output tokens: {entry['metadata'].get('output_tokens', 'N/A')}")
                
                elif entry['type'] == 'review':
                    status = "✅ APPROVED" if entry['approved'] else "❌ REJECTED"
                    st.write(f"**Round {entry['round']} - Review: {status}**")
                    
                    # Display technical accuracy info if available
                    if 'technical_accuracy' in entry:
                        ta = entry['technical_accuracy']
                        if ta.get('is_accurate', True):
                            st.success(f"✅ Technical Accuracy: {ta.get('accuracy_score', 'N/A')}/100")
                        else:
                            st.warning(f"⚠️ Technical Accuracy: {ta.get('accuracy_score', 'N/A')}/100")
                            if ta.get('missing_components'):
                                st.write(f"  - Missing: {', '.join(ta['missing_components'])}")
                            if ta.get('incorrect_elements'):
                                st.write(f"  - Incorrect: {', '.join(ta['incorrect_elements'])}")
                            if ta.get('label_issues'):
                                st.write(f"  - Label Issues: {', '.join(ta['label_issues'])}")
                    
                    if 'visual_score' in entry:
                        st.write(f"- Visual Quality: {entry['visual_score']}/10")
                    if not entry['approved']:
                        st.write(f"**Feedback:**")
                        st.text_area("", value=entry['feedback'], height=150, key=f"feedback_{entry['round']}", disabled=True)
                    if 'metadata' in entry:
                        st.write(f"- Confidence score: {entry['metadata'].get('confidence_score', 'N/A')}")
                
                st.write("---")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <small>Graphics Creation Workflow | Powered by Streamlit & Gemini</small>
    </div>
    """,
    unsafe_allow_html=True
)
