import os
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()

# Verify API key
api_key = os.getenv("GOOGLE_API_KEY")

# Set Page Config
st.set_page_config(
    page_title="HVAC Educational Image Generator",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown(
    """
    <style>
    /* Sleek dark mode / high contrast brand alignment */
    body {
        background-color: #0F111A;
        color: #E2E8F0;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header card */
    .header-card {
        background: linear-gradient(135deg, #242052 0%, #171330 100%);
        padding: 2.5rem;
        border-radius: 16px;
        border-left: 8px solid #F05523;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin-bottom: 2rem;
    }
    .header-title {
        font-family: 'Fira Sans', sans-serif;
        color: #FFFFFF;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        color: #B9C9D1;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }

    /* Container panels */
    .stCard {
        background-color: #171923;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #2D3748;
        box-shadow: 0 4px 6px rgba(0,0,0,0.15);
        margin-bottom: 1.5rem;
    }

    /* Custom button styling */
    div.stButton > button {
        background: linear-gradient(90deg, #F05523 0%, #FF7A45 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        padding: 0.6rem 2rem !important;
        border-radius: 8px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(240, 85, 35, 0.4) !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(240, 85, 35, 0.6) !important;
    }
    div.stButton > button:active {
        transform: translateY(0);
    }
    
    /* Input field enhancement */
    div[data-baseweb="input"], div[data-baseweb="textarea"] {
        background-color: #1A202C !important;
        border-color: #4A5568 !important;
        border-radius: 8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header Card
st.markdown(
    """
    <div class="header-card">
        <h1 class="header-title">❄️ HVAC Educational Image Generator</h1>
        <p class="header-subtitle">Create factually accurate, context-rich instructional visuals conforming strictly to curriculum styling guidelines.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Check API key first
if not api_key:
    st.error("⚠️ `GOOGLE_API_KEY` was not found in environment or `.env` file. Please enter it in the sidebar.")

# Sidebar Settings
st.sidebar.markdown("### ⚙️ Engine Settings")
if not api_key:
    api_key_input = st.sidebar.text_input("Google API Key", type="password")
    if api_key_input:
        os.environ["GOOGLE_API_KEY"] = api_key_input
        api_key = api_key_input

aspect_ratio = st.sidebar.selectbox("Aspect Ratio", ["16:9", "4:3", "1:1", "9:16"], index=0)
image_size = st.sidebar.selectbox("Image Resolution", ["1K", "2K"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    ### 🎨 Annotation Style Rulebook
    * **White Box:** Solid white, soft rounded corners (must be small; if multiple, all must be the same size).
    * **Orange Border:** Solid orange (#F05523) outline.
    * **Black Text:** Plain black text inside the box.
    * **Orange Connector:** Straight orange arrow matching border thickness.
    * **Authentic Colors:** Equipment stays realistic (no orange tools/parts).
    """
)

# Import generator
try:
    from agents.graphics_asset_creation.generator.image_generator_v2 import generate_asset
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "agents", "graphics_asset_creation", "generator"))
    from image_generator_v2 import generate_asset

# Main Layout with Tabs
tab_gen, tab_prompt = st.tabs(["🎨 HVAC Visual Generator", "📸 Image Prompter"])

with tab_gen:
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("📝 Instructional Inputs")
        
        slide_title = st.text_input(
            "Slide Title (Optional)", 
            value="", 
            placeholder="e.g., HVAC Manifold Gauge Set"
        )
        
        slide_content = st.text_area(
            "Slide Content / Bullets (Optional)", 
            value="", 
            placeholder="Enter main text shown on the slide...",
            height=120
        )
        
        voiceover_focus = st.text_area(
            "Voiceover Focus (Optional)", 
            value="", 
            placeholder="Enter the specific sentence or action being highlighted in the audio...",
            height=120
        )
        
        run_parallel = st.checkbox("Generate Multiple Prompts in Parallel", value=False)
        if run_parallel:
            human_strategy = st.text_area(
                "Human Strategies / Central Ideas (Separate with '---' or newlines)", 
                value="A realistic scene showing a home basement where a cool copper refrigerant line is forming condensation droplets in the humid air, with a technician pointing to it.\n---\nAn outdoor condenser unit in a residential backyard on a sunny day.",
                placeholder="Separate distinct prompts with '---' or newlines...",
                height=180
            )
        else:
            human_strategy = st.text_area(
                "Human Strategy / Central Idea", 
                value="A realistic scene showing a home basement where a cool copper refrigerant line is forming condensation droplets in the humid air, with a technician pointing to it.", 
                placeholder="Direct the model to prioritize a specific scene structure, layout, or topic focus...",
                height=100
            )
        
        generate_btn = st.button("🚀 Generate Visual Asset(s)")

    with col2:
        st.subheader("🖼️ Generated Asset Preview")
        
        if generate_btn:
            if not api_key:
                st.error("Error: Please provide a valid `GOOGLE_API_KEY` in the sidebar or environment.")
            elif not slide_title.strip() and not human_strategy.strip():
                st.warning("⚠️ Please provide at least a Slide Title or a Human Strategy / Central Idea to generate the visual.")
            else:
                # Parse prompts
                if run_parallel:
                    raw_prompts = human_strategy.split("---") if "---" in human_strategy else human_strategy.split("\n")
                    strategies = [p.strip() for p in raw_prompts if p.strip()]
                else:
                    strategies = [human_strategy.strip()]

                from concurrent.futures import ThreadPoolExecutor

                # Progress indicator
                progress_placeholder = st.empty()
                progress_placeholder.info(f"🧠 Starting parallel generation of {len(strategies)} asset(s)...")

                def run_single_prompt(idx, strategy):
                    try:
                        res = generate_asset(
                            slide_title=slide_title,
                            slide_content=slide_content,
                            voiceover_focus=voiceover_focus,
                            aspect_ratio=aspect_ratio,
                            image_size=image_size,
                            human_strategy=strategy
                        )
                        return {"index": idx, "strategy": strategy, "result": res, "error": None}
                    except Exception as exc:
                        return {"index": idx, "strategy": strategy, "result": None, "error": str(exc)}

                with ThreadPoolExecutor(max_workers=min(len(strategies), 5)) as executor:
                    futures = [executor.submit(run_single_prompt, i, strat) for i, strat in enumerate(strategies)]
                    results = [f.result() for f in futures]

                progress_placeholder.empty()

                for item in results:
                    idx = item["index"]
                    strat = item["strategy"]
                    err = item["error"]
                    res = item["result"]

                    st.markdown(f"### 🖼️ Asset {idx + 1}")
                    st.markdown(f"**Prompt:** {strat}")

                    if err:
                        st.error(f"❌ Failed to generate asset {idx + 1}: {err}")
                        continue

                    if res.get("error"):
                        st.error(f"❌ Error occurred: {res['error']}")
                        continue

                    image = res.get("image")
                    instructions = res.get("instructions")
                    metadata = res.get("metadata")
                    history = res.get("revision_history", [])

                    if image:
                        st.image(image, caption=f"Generated HVAC Graphic {idx + 1} ({aspect_ratio})", use_container_width=True)
                        
                        import io
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format='JPEG')
                        img_byte_arr = img_byte_arr.getvalue()
                        st.download_button(
                            label=f"💾 Download JPG (Asset {idx + 1})",
                            data=img_byte_arr,
                            file_name=f"hvac_visual_{idx + 1}.jpg",
                            mime="image/jpeg",
                            key=f"download_btn_{idx}"
                        )
                    else:
                        st.warning(f"⚠️ No image was returned by the generator model for asset {idx + 1}.")

                    with st.expander(f"🛠️ View Details for Asset {idx + 1}"):
                        t1, t2, t3 = st.tabs(["📋 Structured JSON Brief", "📜 Self-Correction Loop", "🛠️ Technical Metadata"])
                        
                        with t1:
                            if instructions:
                                try:
                                    import json
                                    parsed_json = json.loads(instructions)
                                    st.json(parsed_json)
                                except Exception:
                                    st.text(instructions)
                            else:
                                st.info("No brief generated.")
                                
                        with t2:
                            if history:
                                st.markdown("### Revision & Critique History")
                                for r_idx, record in enumerate(history):
                                    stage_lbl = "Planning Brief" if record.get("stage") == "instruction" else "Image Generation"
                                    verdict = record.get("verdict", "PASS")
                                    color = "green" if verdict == "PASS" else "red"
                                    st.markdown(f"**Round {record.get('round')} ({stage_lbl})** — Verdict: <span style='color:{color}; font-weight:bold;'>{verdict}</span>", unsafe_allow_html=True)
                                    issues = record.get("new_issues", [])
                                    if issues:
                                        for issue in issues:
                                            st.write(f"- {issue}")
                            else:
                                st.info("Direct approval (No revision loops triggered).")
                                
                        with t3:
                            st.json(metadata)
        else:
            st.info("Enter inputs on the left and click **Generate Visual Asset(s)** to start the pipeline.")

with tab_prompt:
    col_p1, col_p2 = st.columns([1, 1], gap="large")
    
    with col_p1:
        st.subheader("📸 Image Prompter Inputs")
        
        drive_url = st.text_input(
            "Google Drive Image URL",
            value="https://drive.google.com/file/d/1XW_e-yP4Ff93c0s6m-PZ94JbOWhU-V5D/view?usp=sharing",
            key="prompter_drive_url",
            help="Paste a Google Drive view link or any public image URL."
        )
        
        img_type = st.text_input(
            "Style Type / Medium",
            value="Realistic Photograph",
            key="prompter_image_type",
            help="Describe the style (e.g. Photograph, 3D Render, Cutaway)"
        )
        
        generate_prompt_btn = st.button("✨ Reverse Engineer Prompt", key="prompter_generate_btn")
        
    with col_p2:
        st.subheader("🖥️ Preview & Output")
        
        if drive_url:
            from services.image_prompt_generator import download_any_image, generate_image_prompt
            try:
                with st.spinner("Downloading image preview..."):
                    img = download_any_image(drive_url)
                    st.image(img, caption="Source Image Preview", use_container_width=True)
            except Exception as e:
                st.error(f"Could not load image preview: {e}")
                
        if generate_prompt_btn and drive_url:
            if not api_key:
                st.error("Error: Please provide a valid `GOOGLE_API_KEY` in the sidebar or environment.")
            else:
                with st.spinner("Analyzing image and generating prompt..."):
                    try:
                        result_prompt = generate_image_prompt(
                            drive_image_url=drive_url,
                            image_type=img_type
                        )
                        st.success("✅ Prompt Generated Successfully!")
                        st.code(result_prompt, language="text")
                    except Exception as e:
                        st.error(f"Generation failed: {e}")
