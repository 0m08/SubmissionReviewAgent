import gspread
from typing import Dict, Any, List, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.submission_reviewer.image_identifier import (
    generate_image_description,
    score_description_match,
)

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator



def get_column_titles_for_model(model_choice: str) -> Tuple[str, str]:
    """Returns (model_output_column_title, match_score_column_title) for a given model choice."""
    m = str(model_choice).lower()
    if "3.7" in m or "gemini" in m:
        out_title = "Model output\n(gemini 3.7 flash)"
        match_title = "Gemini 3.7 Flash\nMatch (Yes / No)"
    elif "luna" in m or "5.6" in m:
        out_title = "Image description\nModel output\n(gpt 5.6 luna max)"
        match_title = "Scoring agent\nMatch (Yes / No)"
    else:
        clean_name = str(model_choice).strip()
        out_title = f"Model output\n({clean_name})"
        match_title = f"{clean_name}\nMatch (Yes / No)"

    return out_title, match_title


def resolve_column_indices(header_row: List[str], model_choice: str = "gemini-3.7-flash") -> Dict[str, int]:
    """Find 1-indexed column positions for HVAC benchmark columns."""
    col_map: Dict[str, int] = {}
    norm_headers = [str(h).strip().lower().replace("\n", " ") for h in header_row]

    exp_out, exp_match = get_column_titles_for_model(model_choice)
    norm_exp_out = exp_out.lower().replace("\n", " ")
    norm_exp_match = exp_match.lower().replace("\n", " ")

    for idx, norm in enumerate(norm_headers, start=1):
        if "image link" in norm or "image_link" in norm or "media link" in norm or norm == "image url":
            col_map.setdefault("image_link", idx)
        elif "prompt" in norm or norm == "evaluation prompt":
            col_map.setdefault("prompt", idx)
        elif norm == "image description" or "human provided" in norm or "ground truth" in norm or "human description" in norm:
            col_map.setdefault("human_desc", idx)

    # 1. Target model output column matching
    for idx, norm in enumerate(norm_headers, start=1):
        if norm_exp_out in norm or ("3.7" in model_choice.lower() and "3.7" in norm and "model output" in norm):
            col_map["model_output"] = idx
            break
        elif "luna" in model_choice.lower() and ("luna" in norm or "5.6" in norm) and "model output" in norm:
            col_map["model_output"] = idx
            break

    if "model_output" not in col_map:
        for idx, norm in enumerate(norm_headers, start=1):
            if "model output" in norm or "ai description" in norm:
                col_map["model_output"] = idx
                break

    # 2. Target match score column matching
    for idx, norm in enumerate(norm_headers, start=1):
        if norm_exp_match in norm or ("3.7" in model_choice.lower() and "gemini" in norm and "match" in norm):
            col_map["match_score"] = idx
            break
        elif "luna" in model_choice.lower() and ("scoring agent" in norm or "luna" in norm) and "match" in norm:
            col_map["match_score"] = idx
            break

    if "match_score" not in col_map:
        for idx, norm in enumerate(norm_headers, start=1):
            if "match" in norm:
                col_map["match_score"] = idx
                break

    # Fallbacks if image_link or human_desc missing
    if "image_link" not in col_map:
        for idx, norm in enumerate(norm_headers, start=1):
            if "link" in norm or "url" in norm:
                col_map["image_link"] = idx
                break
    if "human_desc" not in col_map:
        for idx, norm in enumerate(norm_headers, start=1):
            if "description" in norm and "model" not in norm:
                col_map["human_desc"] = idx
                break

    return col_map


def ensure_eval_columns_exist(worksheet: gspread.Worksheet, header_row: List[str], model_choice: str = "gemini-3.7-flash") -> Dict[str, int]:
    """Ensure output columns exist in Row 1 of Google Sheet for target model."""
    col_map = resolve_column_indices(header_row, model_choice)
    model_col_title, match_col_title = get_column_titles_for_model(model_choice)

    new_cols = []
    if "model_output" not in col_map:
        new_cols.append(model_col_title)
        col_map["model_output"] = len(header_row) + len(new_cols)
    if "match_score" not in col_map:
        new_cols.append(match_col_title)
        col_map["match_score"] = len(header_row) + len(new_cols)

    for i, col_title in enumerate(new_cols):
        worksheet.update_cell(1, len(header_row) + i + 1, col_title)

    return col_map


@traceable(name="Agent 1 Pipeline")
def run_agent1_pipeline(
    sheet: gspread.Spreadsheet,
    worksheet_name: str,
    model_choice: str = "gemini-3.7-flash",
    overwrite_existing: bool = False,
    session_creds=None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    max_workers: int = 5,
) -> List[Dict[str, Any]]:
    """Run Agent 1 (Image Description Generator) across unprocessed sheet rows."""
    from agents.submission_reviewer.submission_reviewer_ui import fetch_media_and_text_from_url

    worksheet = sheet.worksheet(worksheet_name)
    all_values = worksheet.get_all_values()
    if not all_values:
        return []

    col_map = ensure_eval_columns_exist(worksheet, all_values[0], model_choice)
    link_col, prompt_col, out_col = col_map.get("image_link"), col_map.get("prompt"), col_map["model_output"]

    # Filter for unprocessed rows (rows where output column is empty, unless overwrite_existing=True)
    unprocessed_rows = []
    for r_idx, row in enumerate(all_values[1:], start=2):
        existing_val = str(row[out_col - 1]).strip() if out_col and len(row) >= out_col else ""
        if overwrite_existing or not existing_val or existing_val.startswith("⚠️"):
            unprocessed_rows.append((r_idx, row))

    total = len(unprocessed_rows)
    if total == 0:
        if progress_callback:
            progress_callback(0, 0, "No unprocessed rows found for Agent 1.")
        return []

    def _process_row(r_idx: int, row: List[str]) -> Tuple[int, str]:
        url = row[link_col - 1] if link_col and len(row) >= link_col else ""
        prompt = row[prompt_col - 1] if prompt_col and len(row) >= prompt_col else ""
        if not url or not url.strip():
            return r_idx, "⚠️ Missing Image Link"

        images, _ = fetch_media_and_text_from_url(url.strip(), session_creds=session_creds, max_images=3)
        if not images:
            return r_idx, "⚠️ Could not download image from URL"

        ai_desc = generate_image_description(prompt=prompt.strip(), images=images, model_choice=model_choice)
        return r_idx, ai_desc

    results, cell_updates = [], []

    if progress_callback:
        progress_callback(0, total, f"Agent 1 processing {total} unprocessed rows with {model_choice}...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_row, r_idx, row): r_idx for r_idx, row in unprocessed_rows}
        for count, future in enumerate(as_completed(futures), start=1):
            r_idx, ai_desc = future.result()
            cell_updates.append(gspread.Cell(row=r_idx, col=out_col, value=ai_desc))
            results.append({"row": r_idx, "ai_description": ai_desc})
            if progress_callback:
                progress_callback(count, total, f"Agent 1 processed row {r_idx} ({count}/{total})")

    if cell_updates:
        worksheet.update_cells(cell_updates)

    return results


@traceable(name="Agent 2 Pipeline")
def run_agent2_pipeline(
    sheet: gspread.Spreadsheet,
    worksheet_name: str,
    target_model: str = "gemini-3.7-flash",
    scoring_model: str = "gemini-3.7-flash",
    overwrite_existing: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    max_workers: int = 5,
) -> List[Dict[str, Any]]:
    """Run Agent 2 (Scoring Judge) across unprocessed sheet rows for the target model."""
    worksheet = sheet.worksheet(worksheet_name)
    all_values = worksheet.get_all_values()
    if not all_values:
        return []

    col_map = ensure_eval_columns_exist(worksheet, all_values[0], target_model)
    human_col, out_col, match_col = col_map.get("human_desc"), col_map.get("model_output"), col_map["match_score"]

    # Filter for unprocessed rows (rows where match column is empty, unless overwrite_existing=True)
    unprocessed_rows = []
    for r_idx, row in enumerate(all_values[1:], start=2):
        existing_match = str(row[match_col - 1]).strip() if match_col and len(row) >= match_col else ""
        ai_desc = str(row[out_col - 1]).strip() if out_col and len(row) >= out_col else ""
        if ai_desc and (overwrite_existing or not existing_match):
            unprocessed_rows.append((r_idx, row))

    total = len(unprocessed_rows)
    if total == 0:
        if progress_callback:
            progress_callback(0, 0, "No unprocessed rows found for Agent 2.")
        return []

    def _process_row(r_idx: int, row: List[str]) -> Tuple[int, str, str]:
        ai_desc = row[out_col - 1] if out_col and len(row) >= out_col else ""
        human_desc = row[human_col - 1] if human_col and len(row) >= human_col else ""
        match_score, rationale = score_description_match(ai_desc, human_desc, model_choice=scoring_model)
        return r_idx, match_score, rationale

    results, cell_updates = [], []

    if progress_callback:
        progress_callback(0, total, f"Agent 2 scoring {total} unprocessed rows for {target_model}...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_row, r_idx, row): r_idx for r_idx, row in unprocessed_rows}
        for count, future in enumerate(as_completed(futures), start=1):
            r_idx, match_score, rationale = future.result()
            cell_updates.append(gspread.Cell(row=r_idx, col=match_col, value=match_score))
            results.append({"row": r_idx, "match_score": match_score, "rationale": rationale})
            if progress_callback:
                progress_callback(count, total, f"Agent 2 scored row {r_idx} ({count}/{total}): {match_score}")

    if cell_updates:
        worksheet.update_cells(cell_updates)

    return results


@traceable(name="Full HVAC Evaluation Pipeline")
def run_full_evaluation_pipeline(
    sheet: gspread.Spreadsheet,
    worksheet_name: str,
    agent1_model: str = "gemini-3.7-flash",
    agent2_model: str = "gemini-3.7-flash",
    overwrite_existing: bool = False,
    session_creds=None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    max_workers: int = 5,
) -> Dict[str, Any]:
    """Sequential execution of Agent 1 followed by Agent 2."""
    res1 = run_agent1_pipeline(sheet, worksheet_name, agent1_model, overwrite_existing, session_creds, progress_callback, max_workers)
    res2 = run_agent2_pipeline(sheet, worksheet_name, agent1_model, agent2_model, overwrite_existing, progress_callback, max_workers)
    return {"agent1_results": res1, "agent2_results": res2}
