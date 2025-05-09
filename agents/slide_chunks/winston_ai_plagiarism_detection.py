import requests
from gspread_dataframe import set_with_dataframe
from tqdm import tqdm
from services.smart_progress_bar import SmartProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import time
from services.sheets_service import get_sheet_data_and_df, save_to_sheet
import os


# Winston AI API Configuration
WINSTON_API_URL = "https://api.gowinston.ai/v2/plagiarism"
WINSTON_API_KEY = os.environ.get("WINSTON_API_KEY")

# Function to check plagiarism for a given slide content
def check_plagiarism_winston(text):
    """
    Sends slide content to Winston AI and retrieves plagiarism results.

    :param text: Slide content from 'checklist_based_slide_content' column.
    :return: Dictionary containing extracted plagiarism details.
    """
    headers = {
        "Authorization": f"Bearer {WINSTON_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "text": text,
        "language": "en",
        "country": "us"
    }

    print("\n🟢 Sending request to Winston AI...")
    print(f"🔹 Text Preview: {text}...")  # Print first 100 characters for validation

    try:
        response = requests.post(WINSTON_API_URL, headers=headers, json=data)

        print(f"🟡 API Response Code: {response.status_code}")
        response.raise_for_status()  # Raise an error for HTTP failures

        result = response.json()

        # Extract plagiarism details
        plagiarism_score = result.get("result", {}).get("score", 0)  # Percentage score
        total_words = result.get("result", {}).get("textWordCounts", 0)  # Word count
        total_plagiarized_words = result.get("result", {}).get("totalPlagiarismWords", 0)
        identical_matches = result.get("result", {}).get("identicalWordCounts", 0)
        similar_matches = result.get("result", {}).get("similarWordCounts", 0)
        sources = result.get("sources", [])  # Detected sources

        # Extract formatted source details
        detected_sources = "; ".join(
            [f"{src.get('title', 'Unknown Source')} ({src.get('url', 'No URL')})" for src in sources]
        ) if sources else "No sources detected"

        # Extract "Plagiarized Text" (if any)
        plagiarized_text = []
        for src in sources:
            if "plagiarismFound" in src:
                plagiarized_text.extend([match["sequence"] for match in src["plagiarismFound"]])

        plagiarized_text_str = ", ".join(plagiarized_text) if plagiarized_text else "No Plagiarized Text"

        # Format plagiarism score
        plagiarism_score_text = f"{plagiarism_score}%"

        print(f"✅ Plagiarism Score: {plagiarism_score_text}")
        print(f"✅ Total Words: {total_words}")
        print(f"✅ Plagiarized Words: {total_plagiarized_words}")
        print(f"✅ Identical Matches: {identical_matches}")
        print(f"✅ Similar Matches: {similar_matches}")
        print(f"✅ Sources: {detected_sources}")
        print(f"✅ Plagiarized Text: {plagiarized_text_str}\n")

        return {
            "Plagiarism Score": plagiarism_score_text,
            "Total Words": total_words,
            "Plagiarized Words": total_plagiarized_words,
            "Identical Matches": identical_matches,
            "Similar Matches": similar_matches,
            "Sources Found": detected_sources,
            "Plagiarized Text": plagiarized_text_str
        }

    except requests.exceptions.RequestException as e:
        print(f"❌ Error checking plagiarism: {e}")
        return {
            "Plagiarism Score": "Error",
            "Total Words": "Error",
            "Plagiarized Words": "Error",
            "Identical Matches": "Error",
            "Similar Matches": "Error",
            "Sources Found": "Error",
            "Plagiarized Text": "Error"
        }

def run_plagiarism_detection(sheet, worksheet_name):
    """
    Run plagiarism detection for each row in 'Content' and update the sheet.

    :param sheet: The Google Sheets object.
    :param slide_chunks_df: The DataFrame containing slide content.
    :return: Updated DataFrame after plagiarism check.
    """
    # Ensure required columns exist
    required_columns = [
        "Plagiarism Score", "Total Words", "Plagiarized Words",
        "Identical Matches", "Similar Matches", "Sources Found", "Plagiarized Text"
    ]
    slide_chunks_sheet, slide_chunks_df = get_sheet_data_and_df(sheet,worksheet_name)

    # ✅ Add missing columns if they do not exist
    for col in required_columns:
        if col not in slide_chunks_df.columns:
            slide_chunks_df[col] = ""

    # ✅ **Ensure column headers are updated in Google Sheet before row-wise updates**
    save_to_sheet(slide_chunks_sheet, slide_chunks_df)


    # Prepare for parallel processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit tasks for each row
        for index, row in slide_chunks_df.iterrows():
            slide_content = row["Content"]

            # Skip if slide content is empty or already checked
            if not slide_content or pd.isna(slide_content) or slide_chunks_df.at[index, "Plagiarism Score"]:
                print(f"⏭️ Skipping Slide {index + 1} (Already Checked or Empty Content)\n")
                continue

            # Submit the task
            future = executor.submit(
                check_plagiarism_winston,
                slide_content
            )

            # Map the Future to the index
            futures_map[future] = index

        # Collect the results as they complete
        total_tasks = len(futures_map)

        # Initialize the progress tracker
        progress = SmartProgressBar(total_tasks=total_tasks, description="Percent complete")

        # Process completed tasks
        for future in tqdm(as_completed(futures_map), total=total_tasks):
            index = futures_map[future]  # retrieve the index
            plagiarism_results = future.result()

            # Store results in the dataframe
            for key, value in plagiarism_results.items():
                slide_chunks_df.at[index, key] = value

            # ✅ Immediately update the sheet for this row (row-by-row update)
            save_to_sheet(slide_chunks_sheet, slide_chunks_df)

            print(f"📌 Google Sheet updated for Slide {index + 1} ✅\n")

            # Update progress
            progress.update()

            # Delay to avoid API rate limits (adjust if needed)
            time.sleep(1)

    # ✅ Final save to ensure all results are written before hiding columns
    print("📄 Finalizing sheet with all plagiarism results...")
    save_to_sheet(slide_chunks_sheet, slide_chunks_df)

    print("\n✅ Plagiarism detection completed and columns hidden")

    return slide_chunks_df  # Return the updated DataFrame




