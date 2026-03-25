"""
Graphics Asset Creation Tool - Main App Integration

This page provides functionality to create graphics assets from reference images
by analyzing them and automatically generating editing instructions.
"""

import streamlit as st
import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from utils import (
    create_graphics_asset_from_reference,
    display_editing_instructions,
    EDITING_OPTIONS,
    process_sheet_batch
)

st.title("🎨 Graphics Asset Creation Tool")
st.caption("Analyze reference images → Generate editing instructions → Apply edits automatically")

# Check if user is authenticated
if "gc" not in st.session_state or "drive" not in st.session_state:
    st.error("❌ Authentication required")
    st.info("Please login from the main application to use this tool")
    st.stop()

# Mode selection
mode = st.radio("Processing Mode", ["Single Image", "Batch from Google Sheets"], horizontal=True)

st.divider()

# =============================================================================
# SINGLE IMAGE MODE
# =============================================================================

if mode == "Single Image":
    # Input
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📥 Reference Image")
        uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"])
        url = st.text_input("Or paste URL", placeholder="https://...")
        
        ref_input = uploaded if uploaded else (url if url else None)
        
        if ref_input:
            try:
                if isinstance(ref_input, str):
                    st.caption(f"URL: {ref_input[:50]}...")
                else:
                    st.image(ref_input, width='stretch')
            except:
                pass
    
    with col2:
        st.subheader("📝 Context")
        title = st.text_input("Slide Title*", "Cloud Architecture")
        content = st.text_area("Slide Content*", "Cloud computing provides scalable infrastructure...", height=80)
        focus = st.text_input("Voiceover Focus*", "the three-tier cloud service model")
        instruction = st.text_area("Visual Instruction*", "Show IaaS, PaaS, SaaS layers", height=80)
    
    # Options
    with st.expander("⚙️ Options"):
        col1, col2, col3 = st.columns(3)
        quick = col1.checkbox("Quick Mode", False, help="Skip review loop for faster generation")
        ratio = col2.selectbox("Aspect Ratio", [None, "16:9", "4:3", "1:1"])
        size = col3.selectbox("Size", ["1K", "2K", "4K"])
        
        # Multi-stage editing toggle
        st.markdown("---")
        multi_stage = st.checkbox(
            "🔄 Multi-Stage Editing",
            value=True,
            help="""Process edits in 4 sequential stages:
            • Stage 1: Geometry & Perspective (camera angle, layout)
            • Stage 2: Subject Specifics (educational details, faults)
            • Stage 3: Environment & Lighting (background, lighting)
            • Stage 4: Stylization (final style, color treatment)
            
            Each stage is processed separately with its own LLM call for better results."""
        )
        
        custom = st.text_input("Custom criteria (optional)")
    
    # Generate
    if st.button("🚀 Generate", type="primary", width='stretch'):
        if not all([ref_input, title, content, focus, instruction]):
            st.error("❌ Fill all required fields (*)")
        else:
            # Create placeholders for real-time cumulative stage updates
            stage_placeholders = None
            stage_callback = None
            
            if multi_stage:
                progress_container = st.container()
                with progress_container:
                    st.markdown("### 🔄 Live Stage Progress")
                    st.caption("*Watch as each stage completes and builds on the previous one*")
                    stage_cols = st.columns(4)
                    stage_placeholders = {
                        'stage1': {'status': stage_cols[0].empty(), 'image': stage_cols[0].empty()},
                        'stage2': {'status': stage_cols[1].empty(), 'image': stage_cols[1].empty()},
                        'stage3': {'status': stage_cols[2].empty(), 'image': stage_cols[2].empty()},
                        'stage4': {'status': stage_cols[3].empty(), 'image': stage_cols[3].empty()}
                    }
                    
                    stage_labels = {
                        'stage1': '🔲 Stage 1: Geometry',
                        'stage2': '🎯 Stage 2: Details', 
                        'stage3': '💡 Stage 3: Lighting',
                        'stage4': '🎨 Stage 4: Style'
                    }
                    
                    # Initialize stage displays
                    for stage_key, placeholders in stage_placeholders.items():
                        placeholders['status'].markdown(f"**{stage_labels[stage_key]}**\n⏳ Waiting...")
                
                # Define callback for real-time cumulative updates
                def stage_callback(stage_key, stage_name, edited_image):
                    if stage_placeholders and stage_key in stage_placeholders:
                        placeholders = stage_placeholders[stage_key]
                        placeholders['status'].markdown(f"**{stage_labels[stage_key]}**\n✅ Complete!")
                        placeholders['image'].image(edited_image, caption=f"Output → Input for next", width='stretch')
                
                # Store in session state
                st.session_state['_stage_callback'] = stage_callback
            
            with st.spinner("Processing..."):
                result, edits, history = create_graphics_asset_from_reference(
                    reference_image_input=ref_input,
                    slide_title=title,
                    slide_content=content,
                    voiceover_focus=focus,
                    visual_instruction=instruction,
                    quick_mode=quick,
                    custom_review_criteria=custom if custom else None,
                    aspect_ratio=ratio,
                    image_size=size,
                    use_multi_stage=multi_stage
                )
            
            if result:
                st.success("✅ Done!")
                st.session_state.update({'img': result, 'edits': edits, 'hist': history})
            else:
                st.error("❌ Failed - check console logs")
    
    # Results
    if 'img' in st.session_state:
        st.divider()
        display_editing_instructions(st.session_state['edits'])
        
        st.subheader("🎨 Final Result")
        st.image(st.session_state['img'], width='stretch')
        
        # Download
        from io import BytesIO
        buf = BytesIO()
        st.session_state['img'].save(buf, format='PNG')
        st.download_button("⬇️ Download", buf.getvalue(), "result.png", "image/png", width='stretch')
        
        # Stage-by-stage progression (if multi-stage mode was used)
        if st.session_state.get('hist'):
            # Check if stages are present (multi-stage mode)
            has_stages = any('stage' in e for e in st.session_state['hist'])
            
            if has_stages:
                with st.expander("🔄 Stage-by-Stage Progression", expanded=False):
                    st.markdown("*See how the image evolved through each stage:*")
                    
                    # Group history by stage
                    stages_dict = {}
                    for e in st.session_state['hist']:
                        if 'stage' in e:
                            stage_name = e['stage']
                            if stage_name not in stages_dict:
                                stages_dict[stage_name] = []
                            stages_dict[stage_name].append(e)
                    
                    # Display each stage
                    for stage_name, stage_entries in stages_dict.items():
                        st.markdown(f"### {stage_name}")
                        
                        # Find the final approved image for this stage
                        final_image = None
                        for e in reversed(stage_entries):
                            if e['type'] == 'generation' and 'edited_image' in e:
                                final_image = e['edited_image']
                                break
                        
                        if final_image:
                            col1, col2 = st.columns([1, 1])
                            with col1:
                                st.image(final_image, caption=f"Output from {stage_name}", width='stretch')
                            with col2:
                                st.markdown("**Details:**")
                                for e in stage_entries:
                                    if e['type'] == 'review':
                                        status = "✅ Approved" if e.get('approved') else "❌ Rejected"
                                        st.write(f"Round {e['round']}: {status}")
                                        if not e.get('approved'):
                                            st.caption(f"Feedback: {e.get('feedback', '')[:150]}")
                        st.divider()
            
            # Full detailed history
            with st.expander(f"📜 Detailed History ({len(st.session_state['hist'])} steps)", expanded=False):
                for e in st.session_state['hist']:
                    # Show stage if available (multi-stage mode)
                    stage_label = f" - {e['stage']}" if 'stage' in e else ""
                    st.write(f"**Round {e['round']} - {e['type'].title()}{stage_label}**")
                    if e['type'] == 'generation' and 'edited_image' in e:
                        st.image(e['edited_image'], width=300)
                    elif e['type'] == 'review':
                        status = "✅ Approved" if e.get('approved') else "❌ Rejected"
                        st.write(f"{status}: {e.get('feedback', '')[:100]}")
                    st.caption(str(e.get('metadata', {})))
                    st.divider()

# =============================================================================
# BATCH PROCESSING MODE
# =============================================================================

else:
    st.subheader("📊 Batch Processing from Google Sheets")
    
    # Sheet configuration
    sheet_url = st.text_input(
        "Google Sheets URL",
        value="https://docs.google.com/spreadsheets/d/1Umn42IIfyobvHaZ5Srm6PsEzKjvBIcBFBErOGmYytuU/edit?usp=sharing",
        help="URL of the Google Sheet containing slide data"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        source_tab = st.text_input("Source Tab Name", "Slide Chunksb")
    with col2:
        output_tab = st.text_input("Output Tab Name", "Asset Creation Test")
    
    drive_folder = st.text_input(
        "Drive Folder Name",
        "Graphics Asset Creation Output",
        help="Name of the Google Drive folder where images will be stored"
    )
    
    # Options
    with st.expander("⚙️ Processing Options"):
        col1, col2, col3 = st.columns(3)
        quick = col1.checkbox("Quick Mode", False, help="Skip review loop for faster processing")
        size = col2.selectbox("Image Size", ["1K", "2K", "4K"])
        workers = col3.number_input(
            "Parallel Workers",
            min_value=1,
            max_value=5,
            value=2,
            help="Number of parallel workers. Lower = more stable but slower. Recommended: 1-2 to avoid rate limits."
        )
        
        st.markdown("---")
        multi_stage = st.checkbox(
            "🔄 Multi-Stage Editing",
            value=True,
            help="""Process edits in 4 sequential stages:
            • Stage 1: Geometry & Perspective (camera angle, layout)
            • Stage 2: Subject Specifics (educational details, faults)
            • Stage 3: Environment & Lighting (background, lighting)
            • Stage 4: Stylization (final style, color treatment)
            
            Each stage is processed separately with its own LLM call for better transformation results."""
        )
    
    # Info
    st.info("""
    **How it works:**
    1. Reads from the source tab
    2. Parses sub-segments from 'final_graphics_definition' column
    3. Filters for valid external web image links (skips YouTube/Drive)
    4. Processes each sub-segment: analyze → generate edits → apply
    5. Uploads images to specified Drive folder
    6. Saves results to output tab with editing instructions
    
    **Performance Tips:**
    - Use 1-2 workers for most stable processing (default: 2)
    - Higher worker counts (3-5) may hit API rate limits
    - Failed rows automatically retry up to 2 times with exponential backoff
    """)
    
    # Process button
    if st.button("🚀 Start Batch Processing", type="primary", width='stretch'):
        if not sheet_url:
            st.error("❌ Please enter Google Sheets URL")
        else:
            with st.spinner("Processing batch... This may take several minutes..."):
                result_df = process_sheet_batch(
                    sheet_url=sheet_url,
                    gc=st.session_state["gc"],
                    drive=st.session_state["drive"],
                    source_tab=source_tab,
                    output_tab=output_tab,
                    quick_mode=quick,
                    image_size=size,
                    drive_folder_name=drive_folder,
                    max_workers=workers,
                    use_multi_stage=multi_stage
                )
            
            if not result_df.empty:
                st.success(f"✅ Batch processing complete! Processed {len(result_df)} sub-segments.")
                st.session_state['batch_results'] = result_df
            else:
                st.warning("⚠️ No results generated. Check console logs for details.")
    
    # Display results
    if 'batch_results' in st.session_state:
        st.divider()
        st.subheader("📊 Batch Results")
        st.dataframe(st.session_state['batch_results'], width='stretch')
        
        # Download CSV
        csv = st.session_state['batch_results'].to_csv(index=False)
        st.download_button(
            "⬇️ Download Results CSV",
            csv,
            "batch_results.csv",
            "text/csv",
            width='stretch'
        )
