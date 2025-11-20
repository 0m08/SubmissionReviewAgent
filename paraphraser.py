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

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")

    # === TRADE CONTEXT ===
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
        placeholder="e.g., Residential HVAC, Commercial Refrigeration",
        help="Narrow down the trade context for more specific terminology (optional)"
    )

    st.divider()

    # === OUTPUT FORMATTING ===
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

    # === PROCESSING SETTINGS ===
    st.subheader("⚡ Processing Settings")

    max_iterations = st.slider(
        "Max Refinement Iterations",
        min_value=1,
        max_value=5,
        value=3,
        help="Maximum times to refine if quality checks fail"
    )

    llm_model = st.selectbox(
        "LLM Model",
        ["openai:gpt-4o-mini", "openai:gpt-4o", "anthropic:claude-3-5-sonnet-20241022"],
        index=0,
        help="GPT-4o-mini is faster and cheaper, GPT-4o for highest quality"
    )

    st.divider()

    # === ADVANCED OPTIONS ===
    with st.expander("🎛️ Advanced Options", expanded=False):
        st.caption("Customize quality criteria, examples, and tone")

        custom_quality_criteria = st.text_area(
            "Custom Quality Checklist (optional)",
            placeholder="Leave empty to use default criteria, or enter custom checklist:\n1. ACCURACY: ...\n2. CLARITY: ...\n3. TONE: ...",
            height=150,
            help="Override the default quality criteria for the reviewer agent"
        )

        custom_examples = st.text_area(
            "Custom Examples (optional)",
            placeholder="Leave empty to use defaults, or provide examples:\n\nBAD: \"Technicians must...\"\nGOOD: \"Shut the power off...\"\n\nBAD: \"Ensure optimal...\"\nGOOD: \"Good airflow...\"",
            height=150,
            help="Provide custom before/after examples to guide the paraphraser"
        )

        custom_tone_instructions = st.text_area(
            "Custom Tone Instructions (optional)",
            placeholder="Leave empty for default tone, or specify:\ne.g., 'More formal', 'Very casual', 'Safety-focused', etc.",
            height=100,
            help="Add specific tone instructions beyond the default"
        )

    st.divider()

    # Info section
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        **Multi-Agent Process:**
        1. **Paraphraser** transforms the text
        2. **Reviewer** validates quality
        3. **Refiner** fixes issues (if needed)
        4. Repeats until quality passes

        **Quality Criteria:**
        - Accuracy (no information added/removed)
        - Clarity (improved readability)
        - Tone (sounds like real technician)
        - Technical correctness
        - Safety info preserved
        """)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Input Text")
    input_text = st.text_area(
        "Text to paraphrase",
        height=400,
        placeholder="Paste or type technical content here...\n\nExample: 'Technicians must de-energize the system to mitigate hazards prior to engaging with components.'",
        label_visibility="collapsed",
        key="input_text_area"
    )

    # Character count
    if input_text:
        st.caption(f"Characters: {len(input_text)} | Words: {len(input_text.split())}")

with col2:
    st.subheader("✨ Paraphrased Text")

    # Show placeholder or result
    if "paraphraser_result" in st.session_state:
        result = st.session_state.paraphraser_result

        output_text = st.text_area(
            "Paraphrased output",
            value=result["final_text"],
            height=400,
            label_visibility="collapsed",
            key="output_text_area"
        )

        # Character count for output
        st.caption(f"Characters: {len(result['final_text'])} | Words: {len(result['final_text'].split())}")

        # Copy button
        if st.button("📋 Copy to Clipboard", use_container_width=True):
            st.code(result["final_text"], language=None)
            st.success("✅ Text displayed above - you can copy it from there")
    else:
        st.info("👈 Enter text and click 'Paraphrase Text' to get started")
        st.text_area(
            "Output will appear here",
            value="",
            height=400,
            label_visibility="collapsed",
            disabled=True,
            key="output_placeholder"
        )

# Action buttons
st.divider()

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
            st.rerun()

with col_btn3:
    pass  # Spacer

# Process paraphrasing
if paraphrase_button:
    if not input_text.strip():
        st.error("❌ Please enter some text to paraphrase")
    else:
        # Show progress
        with st.spinner("🔄 Paraphrasing... This may take 10-30 seconds"):
            progress_placeholder = st.empty()
            progress_placeholder.info("📝 Step 1/3: Paraphrasing text...")

            try:
                # Run paraphraser
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
                    custom_tone_instructions=custom_tone_instructions if custom_tone_instructions.strip() else None
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

# Display metadata if results available
if "paraphraser_result" in st.session_state:
    result = st.session_state.paraphraser_result

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

# Footer
st.divider()
st.caption("💡 Tip: For best results, provide complete sentences or paragraphs rather than fragments.")
