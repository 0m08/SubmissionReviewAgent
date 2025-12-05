"""
Paraphraser Agent UI

Simple Streamlit interface for the multi-agent paraphraser.
Transform monotonous technical text into clear, conversational language.
"""

import streamlit as st
from agents.paraphraser import run_paraphraser
from services.helper_functions import compare_text_versions
import json

# Page config
st.set_page_config(
    page_title="Paraphraser Agent",
    page_icon="✍️",
    layout="wide"
)

# Title and description
st.title("✍️ Paraphraser Agent")
st.markdown("""
Transform monotonous technical text into clear, conversational language that sounds
like an experienced tradesperson.
""")

# Settings expander
with st.expander("⚙️ Settings & Configuration", expanded=False):

    st.subheader("📋 Trade Context")
    trade = st.selectbox(
        "Trade",
        ["HVAC", "Electrical", "Plumbing"],
        index=0,
        help="Select the trade context for terminology"
    )

    specialization = st.text_input(
        "Specialization",
        value="",
        placeholder="e.g., Residential HVAC",
        help="Narrow down the trade context (optional)"
    )

    st.divider()

    st.subheader("📐 Output Formatting")
    preserve_formatting = st.checkbox(
        "Preserve formatting",
        value=True,
        help="Maintain bullet points, lists, and paragraph structure"
    )

    target_length = st.selectbox(
        "Target Length",
        ["similar", "concise", "expanded"],
        index=0,
        help="How long should the output be relative to input?"
    )

    st.divider()

    st.subheader("⚡ Processing Settings")
    
    quick_mode = st.checkbox(
        "⚡ Quick Mode",
        value=True,
        help="Skip quality review and refinement. Returns paraphrased text immediately (faster but may be less polished)"
    )
    
    max_iterations = st.slider(
        "Max Refinement Iterations",
        min_value=1,
        max_value=5,
        value=3,
        help="Maximum times to refine if quality checks fail",
        disabled=quick_mode
    )

    llm_model = st.selectbox(
        "LLM Model",
        ["openai:gpt-5-mini", "openai:gpt-5", "anthropic:claude-4-5-sonnet", "google_genai:gemini-2.5-flash", "google_genai:gemini-3-pro"],
        format_func=lambda x: {
            "openai:gpt-5-mini": "OpenAI GPT-5 Mini",
            "openai:gpt-5": "OpenAI GPT-5",
            "anthropic:claude-4-5-sonnet": "Anthropic Claude 4.5 Sonnet",
            "google_genai:gemini-2.5-flash": "Google Gemini 2.5 Flash",
            "google_genai:gemini-3-pro": "Google Gemini 3 Pro"
        }[x],
        index=0,
        help="GPT-5 Mini is faster and cheaper, GPT-5 for highest quality"
    )

    st.divider()

    st.subheader("🎛️ Advanced Options")
    st.caption("Optional: Customize quality criteria, examples, and tone")

    custom_quality_criteria = st.text_area(
        "Custom Quality Checklist (optional)",
        placeholder="Leave empty to use default criteria, or enter custom checklist:\n1. ACCURACY: ...\n2. CLARITY: ...\n3. TONE: ...",
        height=150,
        help="Override the default quality criteria for the reviewer agent",
        disabled=quick_mode
    )

    custom_examples = st.text_area(
        "Custom Examples (optional)",
        placeholder="Leave empty to use defaults, or provide examples:\n\nBAD: \"Technicians must...\"\nGOOD: \"Shut the power off...\"\n\nBAD: \"Ensure optimal...\"\nGOOD: \"Good airflow...\"",
        height=150,
        help="Provide custom before/after examples to guide the paraphraser",
        disabled=quick_mode
    )

    custom_tone_instructions = st.text_area(
        "Custom Tone Instructions (optional)",
        placeholder="Leave empty for default tone, or specify:\ne.g., 'More formal', 'Very casual', 'Safety-focused', etc.",
        height=100,
        help="Add specific tone instructions beyond the default",
        disabled=quick_mode
    )

st.divider()

# Input area
st.subheader("📝 Input Text")
input_text = st.text_area(
    "Text to paraphrase",
    height=300,
    placeholder="Paste or type technical content here...\n\nExample: 'Technicians must de-energize the system to mitigate hazards prior to engaging with components.'",
    label_visibility="collapsed",
    key="input_text_area"
)

# Character count
if input_text:
    st.caption(f"Characters: {len(input_text)} | Words: {len(input_text.split())}")

# Action buttons
col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 1])

with col_btn1:
    paraphrase_button = st.button(
        "✨ Paraphrase Text",
        type="primary",
        use_container_width=True,
        disabled=not input_text or not input_text.strip()
    )

with col_btn2:
    if "paraphraser_result" in st.session_state:
        if st.button("🔄 Clear Results", use_container_width=True):
            del st.session_state.paraphraser_result
            if "paraphraser_input" in st.session_state:
                del st.session_state.paraphraser_input
            if "paraphraser_quick_mode" in st.session_state:
                del st.session_state.paraphraser_quick_mode
            st.rerun()

with col_btn3:
    pass  # Spacer

# Process paraphrasing
if paraphrase_button:
    # Store quick_mode in session state for use in results display
    st.session_state.paraphraser_quick_mode = quick_mode
    if not input_text.strip():
        st.error("❌ Please enter some text to paraphrase")
    else:
        # Create progress tracking containers
        progress_placeholder = st.empty()
        status_container = st.container()

        # Progress messages mapping
        node_messages = {
            "paraphrase": "📝 Transforming text to conversational style...",
            "review": "🔍 Reviewing quality and checking criteria...",
            "refine": "✨ Refining based on quality feedback..."
        }

        # Track state for progress display
        progress_state = {"current_step": "", "refinement_count": 0}

        def update_progress(node_name: str, state: dict):
            """Callback to update UI with current progress"""
            progress_state["current_step"] = node_name
            progress_state["refinement_count"] = state.get("refinement_count", 0)

            # Build progress message
            base_msg = node_messages.get(node_name, f"Processing {node_name}...")

            # Add refinement info if applicable
            if node_name == "refine":
                iteration = state.get("refinement_count", 0)
                base_msg = f"✨ Refining based on quality feedback (iteration {iteration}/{max_iterations})..."
            elif node_name == "review" and progress_state["refinement_count"] > 0:
                base_msg = "🔍 Re-reviewing quality after refinement..."

            progress_placeholder.info(base_msg)

        with st.spinner("🔄 Processing... This may take 10-30 seconds"):
            try:
                # Run paraphraser with progress callback
                result = run_paraphraser(
                    text=input_text,
                    trade=trade,
                    specialization=specialization if specialization else None,
                    preserve_formatting=preserve_formatting,
                    target_length=target_length,
                    max_iterations=max_iterations,
                    llm_model=llm_model,
                    custom_quality_criteria=custom_quality_criteria if custom_quality_criteria.strip() else None,
                    custom_examples=custom_examples if custom_examples.strip() else None,
                    custom_tone_instructions=custom_tone_instructions if custom_tone_instructions.strip() else None,
                    quick_mode=quick_mode,
                    progress_callback=update_progress
                )

                # Store in session state
                st.session_state.paraphraser_result = result
                st.session_state.paraphraser_input = input_text

                progress_placeholder.success("✅ Paraphrasing complete!")
                st.rerun()

            except Exception as e:
                progress_placeholder.empty()
                st.error(f"❌ Error during paraphrasing: {str(e)}")
                with st.expander("🔍 Error Details"):
                    st.exception(e)

# Display output if results available
if "paraphraser_result" in st.session_state:
    result = st.session_state.paraphraser_result

    st.divider()

    # Output text area
    st.subheader("✨ Paraphrased Text")
    output_text = st.text_area(
        "Paraphrased output",
        value=result["final_text"],
        height=300,
        label_visibility="collapsed",
        key="output_text_area"
    )

    # Character count for output
    st.caption(f"Characters: {len(result['final_text'])} | Words: {len(result['final_text'].split())}")

    # Only show process details if not in quick mode
    if not st.session_state.get("paraphraser_quick_mode", False):
        st.divider()
        st.subheader("📊 Process Details")

        # Metrics row
        met_col1, met_col2, met_col3, met_col4 = st.columns(4)

        with met_col1:
            st.metric("Trade", result.get("metadata", {}).get("trade", trade))

        with met_col2:
            st.metric("Refinements", result.get("refinement_count", 0))

        with met_col3:
            status = result.get("status", "unknown").upper()
            st.metric("Status", status)

        with met_col4:
            if result.get("quality_report"):
                score = result["quality_report"].get("score", "N/A")
                st.metric("Quality Score", f"{score}/100" if isinstance(score, int) else score)
            else:
                st.metric("Quality Score", "N/A")

        # Quality report details
        if result.get("quality_report"):
            with st.expander("📋 Quality Review Details"):
                qr = result["quality_report"]

                col_q1, col_q2 = st.columns(2)

                with col_q1:
                    passed = qr.get("passed", False)
                    if passed:
                        st.success("✅ **Passed Quality Review**")
                    else:
                        st.warning("⚠️ **Quality Issues Found**")

                    st.write(f"**Score:** {qr.get('score', 0)}/100")

                with col_q2:
                    issues = qr.get("issues", [])
                    if issues:
                        st.write("**Issues:**")
                        for issue in issues:
                            st.write(f"- {issue}")
                    else:
                        st.write("**No issues found**")

                if qr.get("feedback"):
                    st.write("**Feedback:**")
                    st.info(qr["feedback"])

        # Text diff comparison
        with st.expander("🔀 View Differences"):
            compare_text_versions(
                st.session_state.get("paraphraser_input", ""),
                result["final_text"],
                version1_name="Original Text",
                version2_name="Paraphrased Text"
            )

        # Full message history (for debugging)
        if result.get("messages"):
            with st.expander("🔍 Message History (Debug)"):
                st.write(f"Total messages: {len(result['messages'])}")

                for i, msg in enumerate(result["messages"]):
                    msg_type = type(msg).__name__
                    msg_content = str(msg.content) if hasattr(msg, 'content') else str(msg)

                    with st.container():
                        st.write(f"**Message {i+1}:** `{msg_type}`")

                        # Truncate long messages
                        if len(msg_content) > 1000:
                            st.code(msg_content[:1000] + "...\n[truncated]", language=None)
                        else:
                            st.code(msg_content, language=None)

                    if i < len(result["messages"]) - 1:
                        st.divider()
    else:
        # Quick mode: show simple comparison and status
        st.divider()
        st.info("⚡ **Quick Mode**: Paraphrasing completed without quality review. Results shown above.")
        
        with st.expander("🔀 View Differences"):
            compare_text_versions(
                st.session_state.get("paraphraser_input", ""),
                result["final_text"],
                version1_name="Original Text",
                version2_name="Paraphrased Text"
            )

# Footer
st.divider()
st.caption("💡 Tip: For best results, provide complete sentences or paragraphs rather than fragments.")

