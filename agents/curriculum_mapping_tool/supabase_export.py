"""
Data transformation for exporting curriculum mapping results to Supabase.
"""

import json
import logging
from collections import Counter
from typing import Dict, List, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)

NO_MATCH_VALUES = {"No relevant match found", ""}


def _get_skillcat_course_name(row: pd.Series) -> Optional[str]:
    """
    Extract the SkillCat course name for a row.

    Logic:
    1. If Consolidated Resource is from SkillCat (source == "skillcat"), use its name
    2. Otherwise, fall back to the SkillCat Resource column
    3. Skip if the value indicates no match was found

    Args:
        row: A DataFrame row

    Returns:
        The SkillCat course name, or None if not available
    """
    # Try to get from Consolidated Resource first
    consolidated_json = str(row.get("Consolidated Resource", "")).strip()
    if consolidated_json:
        try:
            consolidated = json.loads(consolidated_json)
            if consolidated.get("source") == "skillcat":
                name = consolidated.get("name", "").strip()
                if name and name not in NO_MATCH_VALUES:
                    return name
        except (json.JSONDecodeError, AttributeError):
            pass

    # Fall back to SkillCat Resource column
    skillcat_resource = str(row.get("SkillCat Resource", "")).strip()
    if skillcat_resource and skillcat_resource not in NO_MATCH_VALUES:
        return skillcat_resource

    return None


def transform_df_for_supabase(df: pd.DataFrame, curriculum_name: str) -> Dict[str, Any]:
    """
    Transform the mapping results DataFrame into the format expected by Supabase.

    Groups rows by SkillCat course name, aggregates concepts, and determines
    the category for each course.

    Args:
        df: The mapping results DataFrame with columns:
            - Category: The category name
            - Course: The concept/topic name
            - SkillCat Resource: The best SkillCat course name
            - Consolidated Resource: JSON with the final selected resource

        curriculum_name: The name for this curriculum

    Returns:
        Dict with structure:
        {
            "curriculum_name": "...",
            "courses": [
                {"course_name": "...", "category_name": "...", "concepts": ["...", "..."]},
                ...
            ]
        }
    """
    # Validate required columns
    required_columns = {"Category", "Course"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Group data by SkillCat course
    course_data: Dict[str, Dict[str, Any]] = {}
    skipped_count = 0

    for _, row in df.iterrows():
        # Get the SkillCat course name for this row
        skillcat_name = _get_skillcat_course_name(row)

        if not skillcat_name:
            skipped_count += 1
            continue

        # Get the concept and category
        concept = str(row.get("Course", "")).strip()
        category = str(row.get("Category", "")).strip()

        if not concept:
            continue

        # Add to grouped data
        if skillcat_name not in course_data:
            course_data[skillcat_name] = {
                "concepts": [],
                "categories": []
            }

        course_data[skillcat_name]["concepts"].append(concept)
        if category:
            course_data[skillcat_name]["categories"].append(category)

    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} rows without valid SkillCat course")

    # Build the final courses list
    courses = []
    for course_name, data in course_data.items():
        # Determine the category (most common, or first if no categories)
        categories = data["categories"]
        if categories:
            category_counts = Counter(categories)
            category_name = category_counts.most_common(1)[0][0]
        else:
            category_name = ""

        # Remove duplicate concepts while preserving order
        seen = set()
        unique_concepts = []
        for c in data["concepts"]:
            if c not in seen:
                seen.add(c)
                unique_concepts.append(c)

        courses.append({
            "course_name": course_name,
            "category_name": category_name,
            "concepts": unique_concepts
        })

    logger.info(f"Transformed {len(df)} rows into {len(courses)} unique SkillCat courses")

    return {
        "curriculum_name": curriculum_name,
        "courses": courses
    }
