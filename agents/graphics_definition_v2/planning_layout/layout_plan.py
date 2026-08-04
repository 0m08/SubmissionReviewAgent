from modules.chain import Chain
from langsmith import traceable
import streamlit as st
from services.sheets_service import get_sheet_data_and_df, save_to_sheet, format_worksheet, clear_worksheet
from services.smart_progress_bar import SmartProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import re

load_dotenv()


layout_planning_prompt = """You are a senior instructional visual planning agent specializing in HVAC e-learning content. Your task is to create a scene-wise layout plan for a single input slide row.

This layout plan will be used upstream of graphics retrieval and selection.

You will be given course context and one input slide row. You must:
- identify one or more scenes from the slide narration,
- choose exactly one layout category for each scene from the predefined layout library,
- define what visual slots are required in each scene.

These are the inputs:

<course_information>
Course name: {course_name}
Topic name: {topic_name}
Subtopic name: {subtopic_name}
</course_information>

<slide_information>
Slide Type: {slide_type}
Slide title: {slide_title}
Slide content: {slide_content}
</slide_information>

<layout_library>
1) single_visual_hero
- Description: One dominant visual supports the full scene.
- Typical use: introducing a concept, simple explanation, single core idea.
- Required slots: 1 (primary_visual)

2) two_item_split_comparison
- Description: Two visuals shown side-by-side for direct comparison.
- Typical use: before vs after, good vs bad, option A vs option B.
- Required slots: 2 (left_visual, right_visual)

3) multi_panel_grid
- Description: Three or four visuals shown as equal or near-equal panels.
- Typical use: enumeration, components listing, multiple parallel items.
- Required slots: 3 or 4 (panel_1, panel_2, panel_3, optional panel_4)

4) main_plus_supporting_inset
- Description: One main visual with one smaller supporting visual.
- Typical use: main concept plus detail, zoomed support, contextual supplement.
- Required slots: 2 (main_visual, inset_visual)

</layout_library>

Instructions:

1. Scope and Responsibility
   - Your task is to plan the layout of the slide based on the slide content and the layout library.
   - Do not invent new layout categories outside the given layout library.

2. Scene Identification Rules
   - A scene is a coherent narration span that should be visualized as one layout unit.
   - One slide row may have one scene or multiple scenes.
   - Keep scenes as coarse as possible while preserving clarity.
   - Do not aggressively micro-segment by clause unless visual intent clearly changes.
   - If the narration shifts to a clearly different explanatory intent, create a new scene.

3. Layout Selection Rules
   - For each scene, choose exactly one layout category from the layout library.
   - Choose based on instructional intent, not stylistic novelty.
   - Prefer the simplest layout that can clearly support the scene.
   - If multiple layouts are plausible, choose the one with highest instructional clarity and lowest visual complexity.

4. Layout Validity and Disambiguation
   - Use `two_item_split_comparison` only when two semantically parallel items need to be compared or displayed side by side.
   - Use `multi_panel_grid` for 3-4 parallel items where equal treatment is needed.
   - Use `main_plus_supporting_inset` when one visual should dominate and one secondary detail supports it.

5. Slot Planning Rules
   - For every scene, define all required slots for the chosen layout.
   - Slots must be role-based and unambiguous.
   - Each slot description must state what the learner needs to see, not how to search.

6. Scene Narration Span Rules
   - Each scene must include the exact narration span from the input slide content that the scene covers.
   - Always preserve original wording from the input slide content.
   - Scene spans must follow narration order and collectively cover the slide content.

7. Transition Slide Handling
   - If Slide Type is "Transition", default to one scene with `single_visual_hero` unless the slide content explicitly demands another structure.

8. Strictness Requirements
   - Use only the allowed layout categories and required slot conventions.
   - Do not add new fields outside the output schema.
   - Do not omit required sections.

Output Format:
Always provide your output strictly in this exact format:

<output>

<evaluation_breakdown>
Use this section as your reasoning scratchpad before finalizing the layout plan.

In this section, you should:
- Explain the slide's instructional intent and likely learner needs.
- Identify possible scene boundaries and justify final scene split decisions.
- Evaluate candidate layout categories for each scene and explain why the selected one is best.
- Explain slot requirements for each chosen layout.
- Note any edge cases and how you resolved them.

It is acceptable for this section to be detailed and verbose if needed for correctness.
</evaluation_breakdown>

<layout_plan>

<scene>
<scene_id>SCENE_1</scene_id>
<narration_span>
Exact narration text span from the slide content covered by this scene.
</narration_span>
<layout_category>
single_visual_hero | two_item_split_comparison | multi_panel_grid | main_plus_supporting_inset
</layout_category>
<layout_rationale>
Brief explanation of why this layout category is most suitable for this scene.
</layout_rationale>
<required_slots>
- slot_name: role and what must be visible for learner understanding
</required_slots>
</scene>

<!-- Repeat <scene> blocks as needed in narration order -->

</layout_plan>

</output>
"""


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Layout Planning Agent",
        "function_name": "generate_layout_plan_for_slide",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def generate_layout_plan_for_slide(
    course_name,
    topic_name,
    subtopic_name,
    slide_title,
    slide_content,
    slide_type="",
    llm="gemini_3_flash_thinking"
):
    """
    Generate layout plan for a single slide row.

    :return: Tuple(layout_plan_xml, evaluation_breakdown_text)
    """
    planner = Chain(llm=llm, tags=["output", "layout_plan", "evaluation_breakdown"])
    planner.add_message(
        role="user",
        content=layout_planning_prompt.format(
            course_name=course_name,
            topic_name=topic_name,
            subtopic_name=subtopic_name,
            slide_type=slide_type or "",
            slide_title=slide_title,
            slide_content=slide_content
        )
    )

    response = planner.run()
    full_output = response.get("output", "")
    full_text = response.get("text", "")

    print(f"\nSlide Title: {slide_title}\n")
    print("📤 Layout Planning Agent Full Response from LLM:\n")
    print(full_text)
    print("\n" + "=" * 100 + "\n")

    eval_match = re.search(r"<evaluation_breakdown>(.*?)</evaluation_breakdown>", full_output, re.DOTALL | re.IGNORECASE)
    layout_match = re.search(r"<layout_plan>(.*?)</layout_plan>", full_output, re.DOTALL | re.IGNORECASE)

    evaluation_breakdown = eval_match.group(1).strip() if eval_match else ""
    layout_plan = layout_match.group(1).strip() if layout_match else ""
    return layout_plan, evaluation_breakdown


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Layout Planning Agent",
        "function_name": "process_layout_plan_row",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def process_layout_plan_row(index, row, course_name, llm="gemini_3_flash_thinking"):
    try:
        topic_name = str(row.get("Topic", "")).strip()
        subtopic_name = str(row.get("Subtopic", "")).strip()
        slide_title = str(row.get("Slide Chunk Title", "")).strip()
        slide_content = str(row.get("Slide Chunk", "")).strip()
        slide_type = str(row.get("Slide Type", "")).strip()
        if slide_type == "nan":
            slide_type = ""

        if not slide_content or slide_content == "nan":
            return index, "", ""

        layout_plan, evaluation_breakdown = generate_layout_plan_for_slide(
            course_name=course_name,
            topic_name=topic_name,
            subtopic_name=subtopic_name,
            slide_title=slide_title,
            slide_content=slide_content,
            slide_type=slide_type,
            llm=llm
        )
        return index, layout_plan, evaluation_breakdown
    except Exception as e:
        print(f"Error processing layout plan row {index}: {e}")
        return index, f"ERROR: {str(e)}", f"ERROR: {str(e)}"


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Layout Planning Agent",
        "function_name": "run_layout_planning_agent_for_all_rows",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def run_layout_planning_agent_for_all_rows(sheet, llm="gemini_3_flash_thinking", max_workers=50):
    worksheet_name = "Slide Chunks"

    _, course_info_df = get_sheet_data_and_df(sheet, "Course info")
    course_name = course_info_df.loc[0, "Course Name"]

    worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)

    if "layout_plan" not in df.columns:
        df["layout_plan"] = ""

    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, row in df.iterrows():
            slide_content = str(row.get("Slide Chunk", "")).strip()
            layout_plan = str(row.get("layout_plan", "")).strip()
            if not slide_content or slide_content == "nan":
                continue
            if layout_plan and layout_plan != "nan":
                continue
            future = executor.submit(process_layout_plan_row, index, row, course_name, llm)
            futures_map[future] = index

        if not futures_map:
            print("All rows already processed or no valid slide content found for layout planning.")
            return

        progress = SmartProgressBar(
            total_tasks=len(futures_map),
            description="Generating layout plans",
            save_interval=5
        )

        for future in as_completed(futures_map):
            index = futures_map[future]
            try:
                row_index, layout_plan, _evaluation = future.result()
                df.at[row_index, "layout_plan"] = layout_plan
                progress.update()
                if progress.should_save():
                    print(f"Saving partial progress after {progress.completed_count} tasks.")
                    save_to_sheet(worksheet, df)
            except Exception as e:
                print(f"Error getting layout plan result for row {index}: {e}")
                df.at[index, "layout_plan"] = f"ERROR: {str(e)}"
                progress.update()

    save_to_sheet(worksheet, df)
    format_worksheet(worksheet)
    print("✅ Layout planning complete and saved to sheet.")


def delete_layout_plan_columns(sheet):
    """
    Remove layout planning columns from Slide Chunks worksheet.
    """
    worksheet_name = "Slide Chunks"
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    columns_to_delete = ["layout_plan", "layout_plan_evaluation"]
    existing = [col for col in columns_to_delete if col in df.columns]
    if existing:
        df = df.drop(columns=existing)
        clear_worksheet(ws)
        save_to_sheet(ws, df)
        print(f"🗑️ Deleted columns: {', '.join(existing)} from '{worksheet_name}' worksheet")
    else:
        print(f"ℹ️ No layout planning columns found in '{worksheet_name}' worksheet")
