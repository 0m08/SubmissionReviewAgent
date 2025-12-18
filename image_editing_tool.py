import streamlit as st
import os
from io import BytesIO
from PIL import Image
import requests
from agents.image_editing.image_editing import (
    image_editing_with_review_loop,
    EDITING_OPTIONS
)

def main():
    """
    Main Streamlit application for Image Editing Tool.
    """
    # Page configuration
    try:
        st.set_page_config(
            page_title="Image Editing Tool",
            page_icon="🎨",
            layout="wide"
        )
    except Exception:
        pass
    
    # Header
    st.title("🎨Image Editing Tool")
    st.markdown("""
    Upload a reference image and specify editing instructions for various aspects.
    AI will generate an edited version while preserving the fundamental composition.
    """)
    
    # Initialize session state
    if "edited_image_result" not in st.session_state:
        st.session_state.edited_image_result = None
    if "editing_history" not in st.session_state:
        st.session_state.editing_history = None
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = None
    if "reference_image" not in st.session_state:
        st.session_state.reference_image = None
    if "editing_instructions" not in st.session_state:
        st.session_state.editing_instructions = {}
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Input")
        
        # Input method selector
        input_method = st.radio(
            "Select input method:",
            ["Upload File", "Image URL"],
            horizontal=True
        )
        
        image = None
        image_source = None
        
        if input_method == "Upload File":
            # Image upload
            uploaded_file = st.file_uploader(
                "Upload a reference image",
                type=["png", "jpg", "jpeg", "webp"],
                help="This image will serve as the reference for editing"
            )
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                image_source = uploaded_file.name
        
        else:  # Image URL
            image_url = st.text_input(
                "Enter image URL:",
                placeholder="https://example.com/image.jpg",
                help="Provide a direct URL to an image"
            )
            
            if image_url and image_url.strip():
                try:
                    with st.spinner("Loading image from URL..."):
                        response = requests.get(image_url, timeout=10)
                        response.raise_for_status()
                        image = Image.open(BytesIO(response.content))
                        image_source = image_url.split('/')[-1] or "url_image.png"
                        st.success("✅ Image loaded successfully!")
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Failed to load image from URL: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Error processing image: {str(e)}")
        
        # Display image if available
        if image:
            st.session_state.reference_image = image
            st.session_state.reference_filename = image_source
            st.image(image, caption="Reference Image", use_container_width=True)
            
            st.markdown("---")
            st.subheader("🎯 Editing Instructions")
            st.markdown("*Provide instructions for any aspects you want to edit. Leave others blank to keep them unchanged.*")
            
            # Create editing instruction inputs
            editing_instructions = {}
            
            # Subject Focus
            with st.expander("👤 Subject Focus", expanded=False):
                st.markdown("*Control what the image focuses on - change the main subject, add/remove elements, or adjust emphasis*")
                subject_focus = st.text_area(
                    "Instructions:",
                    placeholder="E.g., 'Make the person in the center more prominent'",
                    height=100,
                    key="subject_focus",
                    help="Specify changes to the subject or focal point of the image"
                )
                if subject_focus and subject_focus.strip():
                    editing_instructions["subject_focus"] = subject_focus
            
            # Visual Style
            with st.expander("🎭 Visual Style", expanded=False):
                st.markdown("*Change the artistic style, mood, or aesthetic of the image*")
                visual_style = st.text_area(
                    "Instructions:",
                    placeholder="E.g.,'Convert to a minimalist illustration style'",
                    height=100,
                    key="visual_style",
                    help="Specify the desired artistic style or aesthetic"
                )
                if visual_style and visual_style.strip():
                    editing_instructions["visual_style"] = visual_style
            
            # Perspective and Camera
            with st.expander("📷 Perspective and Camera", expanded=False):
                st.markdown("*Adjust the viewing angle, camera position, or field of view*")
                perspective = st.text_area(
                    "Instructions:",
                    placeholder="E.g.,'Make it a close-up shot', 'Adjust to a wider angle perspective'",
                    height=100,
                    key="perspective_and_camera",
                    help="Specify changes to camera angle or perspective"
                )
                if perspective and perspective.strip():
                    editing_instructions["perspective_and_camera"] = perspective
            
            # Lighting and Environment
            with st.expander("💡 Lighting and Environment", expanded=False):
                st.markdown("*Modify lighting conditions, time of day, or atmospheric effects*")
                lighting = st.text_area(
                    "Instructions:",
                    placeholder="E.g., 'Change to natural lighting', 'Add dramatic shadows and highlights'",
                    height=100,
                    key="lighting_and_environment",
                    help="Specify lighting or environmental changes"
                )
                if lighting and lighting.strip():
                    editing_instructions["lighting_and_environment"] = lighting
            
            # Background Setting
            with st.expander("🏞️ Background Setting", expanded=False):
                st.markdown("*Change the location, scenery, or context of the background*")
                background = st.text_area(
                    "Instructions:",
                    placeholder="E.g., 'Replace with a blurred technical setting'",
                    height=100,
                    key="background_setting",
                    help="Specify changes to the background or setting"
                )
                if background and background.strip():
                    editing_instructions["background_setting"] = background
            
            # Color Grading
            with st.expander("🌈 Color Grading", expanded=False):
                st.markdown("*Adjust colors, tones, saturation, or apply color treatments*")
                color_grading = st.text_area(
                    "Instructions:",
                    placeholder="E.g., 'Apply warm orange and teal color grading', 'Make it black and white', 'Increase vibrancy and saturation'",
                    height=100,
                    key="color_grading",
                    help="Specify color adjustments or grading"
                )
                if color_grading and color_grading.strip():
                    editing_instructions["color_grading"] = color_grading
            
            # Additional Comments
            with st.expander("📝 Additional Comments", expanded=False):
                st.markdown("*Add any other instructions or requirements that don't fit into the categories above*")
                additional_comments = st.text_area(
                    "Instructions:",
                    placeholder="E.g., 'Ensure the image looks natural', 'Keep facial features unchanged'",
                    height=100,
                    key="additional_comments",
                    help="Specify any other editing instructions or general requirements"
                )
                if additional_comments and additional_comments.strip():
                    editing_instructions["additional_comments"] = additional_comments
            
            st.markdown("---")
            
            # Output Settings
            st.subheader("⚙️ Output Settings")
            
            col_aspect, col_size = st.columns(2)
            
            with col_aspect:
                aspect_ratio_options = ["Same as input", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
                aspect_ratio_selection = st.selectbox(
                    "Aspect Ratio:",
                    options=aspect_ratio_options,
                    index=0,
                    help="Select output aspect ratio. 'Same as input' preserves the original aspect ratio."
                )
                aspect_ratio = None if aspect_ratio_selection == "Same as input" else aspect_ratio_selection
            
            with col_size:
                image_size = st.selectbox(
                    "Image Size:",
                    options=["1K", "2K", "4K"],
                    index=0,
                    help="Select output image resolution"
                )
            
            st.markdown("---")
            
            # Custom Review Criteria
            st.subheader("🔍 Custom Review Criteria")
            custom_criteria = st.text_area(
                "Specify additional criteria for the reviewer to check (optional):",
                placeholder="E.g., 'Ensure the brand logo remains clearly visible'",
                height=100,
                key="custom_criteria",
                help="Add any specific quality checks or requirements you want the reviewer to verify"
            )
            
            st.markdown("---")
            
            # Quick Mode checkbox
            quick_mode = st.checkbox(
                "⚡ Quick Mode",
                value=False,
                help="Enable to get immediate results without automated quality review. Faster but may need manual refinement."
            )
            
            # Generate button
            if st.button("Edit Image", type="primary", use_container_width=True):
                if not editing_instructions:
                    st.warning("⚠️ Please provide at least one editing instruction.")
                else:
                    # Clear previous results
                    st.session_state.edited_image_result = None
                    st.session_state.editing_history = None
                    st.session_state.editing_instructions = editing_instructions
                    
                    # Show selected instructions summary
                    st.info(f"**Selected edits:** {', '.join([EDITING_OPTIONS[k] for k in editing_instructions.keys()])}")
                    
                    # Run editing pipeline with loader
                    spinner_text = "Generating edited image..." if quick_mode else "Generating and reviewing edited image... This may take a moment."
                    with st.spinner(spinner_text):
                        try:
                            final_edited_image, history, conversation_history = image_editing_with_review_loop(
                                reference_image=image,
                                editing_instructions=editing_instructions,
                                quick_mode=quick_mode,
                                custom_criteria=custom_criteria.strip() if custom_criteria else None,
                                aspect_ratio=aspect_ratio,
                                image_size=image_size
                            )
                            
                            # Store results in session state
                            st.session_state.edited_image_result = final_edited_image
                            st.session_state.editing_history = history
                            st.session_state.conversation_history = conversation_history
                            st.session_state.aspect_ratio = aspect_ratio
                            st.session_state.image_size = image_size
                            
                            # Count rounds
                            total_rounds = len([h for h in history if h["type"] == "generation"])
                            approved = any(h.get("approved", False) for h in history if h["type"] == "review")
                            
                            if quick_mode:
                                st.success("✅ Image editing complete! (Quick Mode)")
                            elif approved:
                                st.success(f"✅ Image editing complete! Approved in {total_rounds} round(s).")
                            else:
                                st.warning(f"⚠️ Image editing complete after {total_rounds} rounds. Review did not fully approve, but this is the best result.")
                            
                        except Exception as e:
                            st.error(f"❌ Image editing failed: {str(e)}")
    
    with col2:
        st.header("📥 Output")
        
        # Display results if available
        if st.session_state.edited_image_result:
            # Display edited image
            st.image(
                st.session_state.edited_image_result, 
                caption="Edited Image", 
                use_container_width=True
            )
            
            # Download button for edited image
            buf = BytesIO()
            st.session_state.edited_image_result.save(buf, format="PNG")
            
            # Generate filename from original with _edited suffix
            original_name = st.session_state.get("reference_filename", "image.png")
            name_parts = os.path.splitext(original_name)
            edited_filename = f"{name_parts[0]}_edited{name_parts[1]}"
            
            st.download_button(
                label="📥 Download Image",
                data=buf.getvalue(),
                file_name=edited_filename,
                mime="image/png",
                use_container_width=True
            )
            
            # Side-by-side comparison in expander
            if st.session_state.reference_image:
                with st.expander("🔍 Compare Reference vs Edited", expanded=True):
                    comp_col1, comp_col2 = st.columns(2)
                    with comp_col1:
                        st.image(st.session_state.reference_image, caption="Reference", use_container_width=True)
                    with comp_col2:
                        st.image(st.session_state.edited_image_result, caption="Edited", use_container_width=True)
            
            # Manual Feedback Section
            st.markdown("---")
            st.subheader("✏️ Manual Feedback & Refinement")
            st.markdown("*Not satisfied with the result? Provide specific feedback to improve the edited image.*")
            
            manual_feedback = st.text_area(
                "What would you like to improve?",
                height=120,
                key="manual_feedback_input",
                help="Provide specific feedback on what you'd like to change in the current edited image"
            )
            
            if st.button("🔄 Regenerate with Feedback", type="secondary", use_container_width=True):
                if manual_feedback and manual_feedback.strip():
                    # Use conversation history to continue the chat
                    with st.spinner("Applying your feedback and regenerating..."):
                        try:
                            # Import the necessary function
                            from agents.image_editing.image_editing import generate_edited_image
                            
                            # Get conversation history from session state
                            conversation_history = st.session_state.get("conversation_history", None)
                            
                            if conversation_history is None:
                                st.warning("⚠️ No conversation history found. Starting fresh...")
                                conversation_history = []
                            
                            # Generate with manual feedback as correction using existing chat
                            refined_image, gen_metadata, model_content, user_content = generate_edited_image(
                                reference_image=st.session_state.reference_image,
                                editing_instructions=st.session_state.editing_instructions,
                                correction_feedback=manual_feedback,
                                conversation_history=conversation_history,
                                aspect_ratio=st.session_state.get("aspect_ratio"),
                                image_size=st.session_state.get("image_size", "1K")
                            )
                            
                            # Update conversation history
                            conversation_history.append(user_content)
                            conversation_history.append(model_content)
                            st.session_state.conversation_history = conversation_history
                            
                            # Debug log the updated conversation history
                            from agents.image_editing.image_editing import debug_print_history
                            current_round = len([h for h in st.session_state.editing_history if h["type"] in ["generation", "manual_refinement"]]) if st.session_state.editing_history else 1
                            debug_print_history(conversation_history, f"Manual Feedback {current_round}")
                            
                            # Update the result
                            st.session_state.edited_image_result = refined_image
                            
                            # Add to history
                            if st.session_state.editing_history:
                                current_round = len([h for h in st.session_state.editing_history if h["type"] == "generation"]) + 1
                                st.session_state.editing_history.append({
                                    "round": current_round,
                                    "type": "manual_refinement",
                                    "feedback": manual_feedback,
                                    "refined_image": refined_image,
                                    "metadata": gen_metadata
                                })
                            
                            st.success("✅ Image regenerated with your feedback!")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Regeneration failed: {str(e)}")
                else:
                    st.warning("⚠️ Please provide feedback before regenerating.")
            
if __name__ == "__main__":
    main()
else:
    main()
