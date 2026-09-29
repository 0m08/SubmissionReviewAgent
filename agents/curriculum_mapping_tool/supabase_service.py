"""
Supabase service for saving curriculum data to the SkillCat platform.
"""

import os
import logging
import requests
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

SUPABASE_URL = "https://mdiulditosmnaqhimcdh.supabase.co"
load_dotenv()


def _get_api_key() -> str:
    """Get Supabase API key from environment variable."""
    api_key = os.getenv("SUPABASE_SECRET_API_KEY")
    if not api_key:
        raise ValueError("SUPABASE_SECRET_API_KEY environment variable is not set")
    return api_key


def _get_headers() -> Dict[str, str]:
    """Get standard headers for Supabase API requests."""
    api_key = _get_api_key()
    return {
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }


def insert_curriculum(curriculum_name: str) -> Dict[str, Any]:
    """
    Insert a new curriculum into the custom_curriculums table.

    Args:
        curriculum_name: The name of the curriculum

    Returns:
        Dict with success status and curriculum_id if successful
    """
    url = f"{SUPABASE_URL}/rest/v1/custom_curriculums"
    headers = _get_headers()

    data = {
        "curriculum_name": curriculum_name
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

        result = response.json()
        curriculum_id = result[0]["id"] if result else None

        logger.info(f"Created curriculum '{curriculum_name}' with id={curriculum_id}")

        return {
            "success": True,
            "curriculum_id": curriculum_id,
            "data": result
        }
    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        if hasattr(e, 'response') and e.response is not None:
            error_msg = e.response.text
        logger.error(f"Failed to insert curriculum: {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "curriculum_id": None
        }


def insert_curriculum_courses(curriculum_id: int, courses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Insert courses into the curriculum_courses table.

    Args:
        curriculum_id: The ID of the parent curriculum
        courses: List of course dicts with keys: course_name, category_name, concepts

    Returns:
        Dict with success status
    """
    url = f"{SUPABASE_URL}/rest/v1/curriculum_courses"
    headers = _get_headers()

    # Prepare the data with curriculum_id and position
    records = []
    for idx, course in enumerate(courses):
        records.append({
            "curriculum_id": curriculum_id,
            "course_name": course.get("course_name", ""),
            "category_name": course.get("category_name", ""),
            "concepts": course.get("concepts", []),
            "position": idx
        })

    try:
        response = requests.post(url, headers=headers, json=records)
        response.raise_for_status()

        logger.info(f"Inserted {len(records)} courses for curriculum_id={curriculum_id}")

        return {
            "success": True,
            "count": len(records),
            "data": response.json() if response.text else None
        }
    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        if hasattr(e, 'response') and e.response is not None:
            error_msg = e.response.text
        logger.error(f"Failed to insert courses: {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "count": 0
        }


def save_curriculum_to_supabase(curriculum_name: str, courses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Save a complete curriculum with courses to Supabase.

    This is the main entry point that:
    1. Creates a curriculum record
    2. Inserts all courses linked to that curriculum

    Args:
        curriculum_name: The name of the curriculum
        courses: List of course dicts with keys: course_name, category_name, concepts

    Returns:
        Dict with success status and details
    """
    # Validate inputs
    if not curriculum_name:
        return {
            "success": False,
            "error": "Curriculum name is required"
        }

    if not courses:
        return {
            "success": False,
            "error": "At least one course is required"
        }

    # Step 1: Create the curriculum
    curriculum_result = insert_curriculum(curriculum_name)

    if not curriculum_result["success"]:
        return {
            "success": False,
            "error": f"Failed to create curriculum: {curriculum_result.get('error', 'Unknown error')}"
        }

    curriculum_id = curriculum_result["curriculum_id"]

    # Step 2: Insert the courses
    courses_result = insert_curriculum_courses(curriculum_id, courses)

    if not courses_result["success"]:
        return {
            "success": False,
            "error": f"Created curriculum but failed to insert courses: {courses_result.get('error', 'Unknown error')}",
            "curriculum_id": curriculum_id
        }

    return {
        "success": True,
        "curriculum_id": curriculum_id,
        "courses_count": courses_result["count"],
        "message": f"Successfully saved '{curriculum_name}' with {courses_result['count']} courses"
    }
