import streamlit as st
import os
from io import BytesIO
from PIL import Image
from agents.image_translation.image_translation import image_translation_with_review_loop

def main():
    """
    Main Streamlit application for Image Translation Tool.
    """
    # Page configuration
    try:
        st.set_page_config(
            page_title="Image Translation Tool",
            page_icon="🌐",
            layout="wide"
        )
    except Exception:
        pass
    
    # Header
    st.title("Image Translation Tool")
    st.markdown("""
    Upload an image with text and get it translated inline.
    """)
    
    # Initialize session state
    if "translated_image_result" not in st.session_state:
        st.session_state.translated_image_result = None
    if "translation_history" not in st.session_state:
        st.session_state.translation_history = None
    if "original_image" not in st.session_state:
        st.session_state.original_image = None
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Input")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Upload an image with text",
            type=["png", "jpg", "jpeg", "webp"]
        )
        
        # Display uploaded image
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.session_state.original_image = image
            st.session_state.original_filename = uploaded_file.name
            st.image(image, caption="Original Image", use_container_width=True)
            
            # Translation direction selector
            translation_direction = st.selectbox(
                "🌍 Translation Direction",
                options=["Spanish to English", "English to Spanish"],
                help="Select the direction for text translation"
            )
            
            # User comment
            user_comment = st.text_area(
                "Additional comments or requirements (optional)",
                placeholder="E.g., 'Keep technical terms', 'Preserve brand names', etc.",
                height=100
            )
            
            # Translate button
            if st.button("🚀 Translate Image", type="primary", use_container_width=True):
                # Clear previous results
                st.session_state.translated_image_result = None
                st.session_state.translation_history = None
                
                # Run translation pipeline with loader
                with st.spinner("Translating image... This may take a moment."):
                    try:
                        final_translated_image, history = image_translation_with_review_loop(
                            image=image,
                            translation_direction=translation_direction,
                            user_comment=user_comment if user_comment else None
                        )
                        
                        # Store results in session state
                        st.session_state.translated_image_result = final_translated_image
                        st.session_state.translation_history = history
                        
                        st.success("✅ Translation complete!")
                        
                    except Exception as e:
                        st.error(f"❌ Translation failed: {str(e)}")
    
    with col2:
        st.header("📥 Output")
        
        # Display results if available
        if st.session_state.translated_image_result:
            # Display translated image
            st.image(
                st.session_state.translated_image_result, 
                caption="Translated Image", 
                use_container_width=True
            )
            
            # Download button for translated image
            buf = BytesIO()
            st.session_state.translated_image_result.save(buf, format="PNG")
            
            # Generate filename from original with _translated prefix
            original_name = st.session_state.get("original_filename", "image.png")
            name_parts = os.path.splitext(original_name)
            translated_filename = f"{name_parts[0]}_translated{name_parts[1]}"
            
            st.download_button(
                label="📥 Download",
                data=buf.getvalue(),
                file_name=translated_filename,
                mime="image/png",
                use_container_width=True
            )
            
            # Side-by-side comparison in expander
            if st.session_state.original_image:
                with st.expander("Compare Original vs Translated"):
                    comp_col1, comp_col2 = st.columns(2)
                    with comp_col1:
                        st.image(st.session_state.original_image, caption="Original", use_container_width=True)
                    with comp_col2:
                        st.image(st.session_state.translated_image_result, caption="Translated", use_container_width=True)

if __name__ == "__main__":
    main()
else:
    main()
