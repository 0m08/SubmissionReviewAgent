from services.sheets_service import get_sheet_data_and_df, save_to_sheet, format_worksheet
from tqdm import tqdm
import re
import streamlit as st
from services.smart_progress_bar import SmartProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed



from agents.graphics_definition.define_graphics.generate_graphics_definition import generate_graphics_definition
from agents.graphics_definition.define_graphics.generate_graphics_definition import get_previous_graphics_definition_as_str
from agents.graphics_definition.define_graphics.generate_complexity_review import generate_complexity_review
from agents.graphics_definition.define_graphics.generate_missing_sentences_review import generate_missing_sentences_review
from agents.graphics_definition.define_graphics.generate_accuracy_review import generate_accuracy_review
from agents.graphics_definition.define_graphics.generate_reuse_previous_review import generate_reuse_previous_graphics_review
from agents.graphics_definition.define_graphics.generate_reviser_output import generate_reviser_output_for_slide

# Pre-Exec function to create the "Reference Description" column
def ensure_reference_description_column(sheet, worksheet_name):
    """
    Adds the 'Reference Description' column if missing in Slide Chunks sheet.
    This is designed to run as a pre_exec_func before the main agent starts.
    """
    worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)
    if "Reference Description" not in df.columns:
        df["Reference Description"] = ""
        save_to_sheet(worksheet=worksheet, df=df)
        print("✅ 'Reference Description' column added.")
    else:
        print("ℹ️ 'Reference Description' column already exists.")

def run_generate_graphics_definition(sheet, worksheet_name, course_name, target_audience, progress, llm="gemini_2_flash"):
    # Read the sheet and df
    slide_chunks_sheet, slide_chunks_df = get_sheet_data_and_df(sheet, worksheet_name)

    # Ensure required columns exist
    for col in ['graphics_definition', 'complexity_review', 'missing_sentences_review', 'accuracy_review', 'reuse_previous_graphics_review', 'human_review', 'revised_graphics_definition']:
        if col not in slide_chunks_df.columns:
            slide_chunks_df[col] = ""

    # To store all the previous graphics definition
    previous_graphics_definition = ""

    for index, row in slide_chunks_df.iterrows():
        if row['graphics_definition'].strip():
            continue

        references = row['Reference Description'].strip()

        if index > 0:
            revised_graphics_definition = slide_chunks_df.loc[index - 1, 'revised_graphics_definition'].strip()
            previous_graphics_definition = get_previous_graphics_definition_as_str(
                slide_no = index,  
                text = revised_graphics_definition,
                existing_definition_str = previous_graphics_definition
            )
        else:
            revised_graphics_definition = ""
            previous_graphics_definition = ""

        graphics_definition = generate_graphics_definition(
            course_name=course_name,
            target_audience=target_audience,
            slide_title=row['final_slide_title'],
            slide_content=row['final_slide_content'],
            previous_graphics_definition=previous_graphics_definition,
            references=references,
            llm=llm
        )

        # if isinstance(graphics_definition, list):
        #     graphics_definition = "\n".join(graphics_definition)

        results = {'graphics_definition': graphics_definition}
        progress.update()  # Update progress after generating graphics definition

        def generate_review(review_func, *args):
            review_text = review_func(*args)
            # if isinstance(review_text, list):
            #     review_text = "\n".join(review_text)
            verdict_match = re.search(r"<verdict>\s*(.*?)\s*</verdict>", review_text, re.DOTALL)
            verdict = verdict_match.group(1).strip().lower() if verdict_match else ""
            return review_text if verdict == "fail" else ""

        with ThreadPoolExecutor() as executor:
            future_to_review = {
                executor.submit(generate_review, generate_complexity_review, course_name, target_audience, row['final_slide_title'], row['final_slide_content'], graphics_definition, llm): 'complexity_review',
                executor.submit(generate_review, generate_missing_sentences_review, course_name, target_audience, row['final_slide_title'], row['final_slide_content'], graphics_definition, llm): 'missing_sentences_review',
                executor.submit(generate_review, generate_accuracy_review, course_name, target_audience, row['final_slide_title'], row['final_slide_content'], graphics_definition, llm): 'accuracy_review',
                executor.submit(generate_review, generate_reuse_previous_graphics_review, course_name, target_audience, row['final_slide_title'], row['final_slide_content'], graphics_definition, previous_graphics_definition, llm): 'reuse_previous_graphics_review'
            }

            for future in as_completed(future_to_review):
                review_type = future_to_review[future]
                try:
                    results[review_type] = future.result()
                except Exception as e:
                    results[review_type] = f"Error: {str(e)}"
                progress.update()

        slide_chunks_df.loc[index, 'graphics_definition'] = results['graphics_definition']
        slide_chunks_df.loc[index, 'complexity_review'] = results['complexity_review']
        slide_chunks_df.loc[index, 'missing_sentences_review'] = results['missing_sentences_review']
        slide_chunks_df.loc[index, 'accuracy_review'] = results['accuracy_review']
        slide_chunks_df.loc[index, 'reuse_previous_graphics_review'] = results['reuse_previous_graphics_review']
        
    
        save_to_sheet(worksheet = slide_chunks_sheet, df = slide_chunks_df)

        format_worksheet(slide_chunks_sheet)

        print(f"Please enter manual feedback for this graphics definition: row_number - {index + 1}")
        return  # Stop execution after one row is processed

    return True

def run_revise_generated_graphics_definition(sheet, worksheet_name, course_name, target_audience, progress,  llm="gemini_2_flash"):
    """
    Process revised graphics definition for each row and generate graphics definition for the next row if needed.
    """
    slide_chunks_sheet, slide_chunks_df = get_sheet_data_and_df(sheet, worksheet_name)
    previous_graphics_definition = ""

    for index, row in slide_chunks_df.iterrows():
        # Skip processing if revised_graphics_definition is already populated
        if row['revised_graphics_definition'].strip():
            print(f"✅ Slide {index + 1}: Revised Graphics Definition already populated. Skipping.")
            continue
        references = row['Reference Description'].strip()
        graphics_definition = row['graphics_definition'].strip()
        complexity_review = row['complexity_review'].strip()
        missing_sentences_review = row['missing_sentences_review'].strip()
        accuracy_review = row['accuracy_review'].strip()
        reuse_previous_graphics_review = row['reuse_previous_graphics_review'].strip()
        human_review = row['human_review'].strip()

        # Skip the reviser agent if all review columns are empty
        if not (complexity_review or missing_sentences_review or accuracy_review or reuse_previous_graphics_review or human_review):
            print("✅ No issues detected. Skipping Reviser Agent and using original Graphics Definition.\n")
            revised_graphics_definition = graphics_definition  # Copy graphics definition directly
            progress.update()  

            # Update previous graphics definition even when reviser agent is skipped
            previous_graphics_definition = get_previous_graphics_definition_as_str(
                slide_no = index,
                text = revised_graphics_definition,
                existing_definition_str = previous_graphics_definition
            )

        # slide_chunks_df.loc[index, 'revised_graphics_definition'] = revised_graphics_definition
        # slide_chunks_df = slide_chunks_df.astype(str)
        # slide_chunks_sheet.update([slide_chunks_df.columns.values.tolist()] + slide_chunks_df.values.tolist())
        # continue  # Skip reviser agent and move to the next slide

        # Run the revised graphics definition function if any review is present
        print("⏳ Generating Revised Graphics Definition\n")

        revised_graphics_definition = generate_reviser_output_for_slide(
            course_name = course_name,
            target_audience = target_audience,
            slide_title = row['final_slide_title'],
            slide_content = row['final_slide_content'],
            graphics_definition = graphics_definition,
            complexity_review = complexity_review,
            missing_sentences_review = missing_sentences_review,
            accuracy_review = accuracy_review,
            reuse_previous_graphics_review = reuse_previous_graphics_review,
            human_review = human_review,
            previous_graphics_definition = previous_graphics_definition,
            references = references,
            llm = llm
        )
        # if isinstance(revised_graphics_definition, list):
        #     revised_graphics_definition = "\n".join(revised_graphics_definition)

        print("✅ Generated Revised Graphics Definition\n")
        progress.update()  # Update progress after generating revised graphics definition
        print("-"*100)
        
        # Add this to the df
        slide_chunks_df.loc[index, 'revised_graphics_definition'] = revised_graphics_definition

        save_to_sheet(worksheet = slide_chunks_sheet, df = slide_chunks_df)
    
        # Preserve the human review in the sheet before updating the DataFrame
        slide_chunks_df.loc[index, 'human_review'] = human_review

        
            

        # Update previous graphics definition
        # Update previous graphics definition even when reviser agent is skipped
        previous_graphics_definition = get_previous_graphics_definition_as_str(
            slide_no = index,
            text = revised_graphics_definition,
            existing_definition_str = previous_graphics_definition
        )
        
        return
    return True



def run_generate_and_revise_graphics(sheet, worksheet_name, course_name, target_audience, skip_manual_step=False, llm="gemini_2_flash"):
    """
    Combined function to:
    1. Generate graphics definitions if missing.
    2. Revise graphics definitions if needed.
    3. Ensure each slide is processed in order.
    
    The function stops when a graphics definition needs manual review.
    :param sheet: The sheet object.
    :param worksheet_name: The worksheet name.
    :param course_name: The course name.
    :param target_audience: The target audience.
    :param:param skip_manual_step: Bool. If True, it will assume all the graphics definitionas have been generated.
    :param llm: The language model to use.
    :return: True if all slides have been processed, False otherwise.
    """

    # Read the sheet and dataframe
    _, slide_chunks_df = get_sheet_data_and_df(sheet, worksheet_name)

    # Ensure required columns exist
    for col in ['graphics_definition', 'complexity_review', 'missing_sentences_review', 
                'accuracy_review', 'reuse_previous_graphics_review', 'human_review', 'revised_graphics_definition']:
        if col not in slide_chunks_df.columns:
            slide_chunks_df[col] = ""

    total_tasks = (len(slide_chunks_df) * 6) + 1  # 6 tasks per slide

    if not hasattr(run_generate_and_revise_graphics, "progress"):
        run_generate_and_revise_graphics.progress = SmartProgressBar(total_tasks=total_tasks, description="Percent complete")

    progress = run_generate_and_revise_graphics.progress

    for index, row in tqdm(slide_chunks_df.iterrows(), total=total_tasks):  # Track rows, not tasks
        graphics_definition = row['graphics_definition'].strip()
        revised_graphics_definition = row['revised_graphics_definition'].strip()

        # If graphics definition exists, check if revision is needed
        if graphics_definition:
            if revised_graphics_definition:
                continue  # Both exist, move to the next row

            # Run reviser function since revised definition is missing
            print(f"⏳ Revising Graphics Definition for Slide {index + 1}\n")
            run_revise_generated_graphics_definition(sheet, worksheet_name, course_name, target_audience, progress, llm)
            continue  # Move to the next row after revising

        # If graphics definition is missing, generate it
        print(f"⏳ Generating Graphics Definition for Slide {index + 1}\n")
        run_generate_graphics_definition(sheet, worksheet_name, course_name, target_audience, progress, llm)

# Show message and stop execution if manual review is needed
        st.write(f"✔ Graphics Definition generated and revised for Slide {index + 1}. Please enter review comments before continuing.(Optional)")

        # Stop execution to allow user manual review unless skip_manual_step is True
        if not skip_manual_step:
            return  # Stop execution to allow user to review the generated graphics definition

        # After generating, immediately revise the generated graphics definition if it's missing
        print(f"⏳ Revising Generated Graphics Definition for Slide {index + 1} (after generation)\n")
        run_revise_generated_graphics_definition(sheet, worksheet_name, course_name, target_audience, progress, llm)

    
    print('All slides processed.')
    return True
