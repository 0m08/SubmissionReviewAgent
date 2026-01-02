"""
Consolidation Agent for Curriculum Mapping

This module provides category-level consolidation of mapped curriculum resources.
It analyzes course assignments within a category and reduces course diversity
by consolidating minority courses into majority courses when appropriate.
"""

import json
import logging
import regex as re
from typing import Any, Dict, List, Optional
from modules.chain import Chain

logger = logging.getLogger(__name__)

consolidation_prompt = """You are an expert curriculum consolidation reviewer.

Your task is to analyze course assignments within a category and consolidate them to minimize the number of unique courses while maintaining educational coverage quality.

<category>
{category_name}
</category>

<concepts_and_courses>
{concepts_and_courses}
</concepts_and_courses>

<course_metadata>
{course_metadata}
</course_metadata>

## Analysis Instructions:

1. **Identify Course Distribution:**
   - List all unique courses assigned to concepts in this category
   - Count how many concepts each course is assigned to
   - Identify "majority" courses (assigned to most concepts) and "minority" courses (assigned to fewer)

2. **Evaluate Coverage Potential:**
   For each concept assigned to a minority course, evaluate if a majority course could adequately cover that concept:
   - Compare the concept name/topic with the majority course's description, topics, and learning objectives
   - Consider if the majority course addresses the concept directly, tangentially, or not at all
   - A course covers a concept well if the concept topic is explicitly mentioned or closely related to the course content

3. **Make Consolidation Decisions:**
   - If a majority course can cover a minority concept WELL or ADEQUATELY, recommend consolidation to that course
   - If no majority course covers the concept well, keep the original assignment
   - If ALL courses are different (no clear majority), evaluate which course has the broadest coverage and could serve as a consolidation target
   - Prefer consolidation when it reduces total unique courses without sacrificing coverage quality

4. **Edge Cases:**
   - If only one unique course exists: keep all as-is
   - If all concepts have completely different courses with no overlap potential: keep original assignments
   - If a course has "No relevant match found" or similar: keep as-is, do not consolidate into it

5. **Preference factor:**
   - User preference is as follows: Source - skillcat courses > nextech courses > youtube videos.
   - Give slight preference to consolidating into higher-preference sources when coverage is adequate.
   
## Output Format:

<detailed_overall_analysis>
Provide an overall analysis of the course distribution and consolidation rationale here. It is okay for this section to be quite long for thoroughness.
</detailed_overall_analysis>

For EACH concept in the category, provide a decision in the exact format below:

<consolidation_decisions>
<decision>
<concept>exact concept name from input</concept>
<original_course>original course name</original_course>
<consolidated_course>consolidated course name (same as original if no change)</consolidated_course>
<reason>brief explanation of why this decision was made</reason>
</decision>

[... repeat for each concept ...]

</consolidation_decisions>

<summary>
Brief summary of consolidation actions taken for this category (e.g., "Consolidated 3 concepts from Course B to Course A which covers all topics adequately")
</summary>

IMPORTANT: You MUST provide a decision for EVERY concept listed in the input. The consolidated_course should be the exact course name.
"""


def get_unique_courses_from_category(concepts_data: List[Dict]) -> Dict[str, Dict]:
    """
    Extract unique courses and their metadata from category data.

    Args:
        concepts_data: List of dicts with 'concept' and 'best_resource' (JSON string)

    Returns:
        Dict mapping course_name -> {metadata, count, concepts, original_json}
    """
    courses = {}
    for item in concepts_data:
        concept = item.get("concept", "")
        best_resource_str = item.get("best_resource", "")

        if not best_resource_str or best_resource_str.strip() == "":
            continue

        try:
            resource = json.loads(best_resource_str)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse Best Resource JSON for concept: {concept}")
            continue

        course_name = resource.get("name", "Unknown")

        # Skip invalid/empty course names
        if not course_name or course_name.lower() in ["unknown", "no relevant match found", "none", ""]:
            continue

        if course_name not in courses:
            courses[course_name] = {
                "metadata": resource,
                "count": 0,
                "concepts": [],
                "original_json": best_resource_str
            }

        courses[course_name]["count"] += 1
        courses[course_name]["concepts"].append(concept)

    return courses


def format_concepts_for_prompt(concepts_data: List[Dict]) -> str:
    """Format concepts and their assigned courses for the prompt."""
    lines = []
    for i, item in enumerate(concepts_data):
        concept = item.get("concept", "Unknown")
        best_resource_str = item.get("best_resource", "")

        course_name = "No assignment"
        if best_resource_str:
            try:
                resource = json.loads(best_resource_str)
                course_name = resource.get("name", "Unknown")
            except json.JSONDecodeError:
                course_name = "Invalid JSON"

        lines.append(f"{i+1}. Concept: {concept}")
        lines.append(f"   Assigned Course: {course_name}")
        lines.append("")

    return "\n".join(lines)


def format_course_metadata(concepts_data: List[Dict]) -> str:
    """Extract and format unique course metadata for the prompt."""
    courses = get_unique_courses_from_category(concepts_data)

    if not courses:
        return "No valid course metadata available."

    lines = []
    # Sort by count (descending) to show majority courses first
    sorted_courses = sorted(courses.items(), key=lambda x: x[1]["count"], reverse=True)

    for course_name, data in sorted_courses:
        metadata = data["metadata"]
        lines.append(f"### Course: {course_name}")
        lines.append(f"Assigned to {data['count']} concept(s): {', '.join(data['concepts'])}")

        # Include relevant metadata for evaluation (increased limits for better context)
        if metadata.get("description"):
            desc = metadata["description"]
            if len(desc) > 1000:
                desc = desc[:1000] + "..."
            lines.append(f"Description: {desc}")

        if metadata.get("topics"):
            lines.append(f"Topics: {metadata['topics']}")

        if metadata.get("learning_objectives"):
            objectives = metadata["learning_objectives"]
            if len(objectives) > 600:
                objectives = objectives[:600] + "..."
            lines.append(f"Learning Objectives: {objectives}")

        if metadata.get("category"):
            lines.append(f"Category: {metadata['category']}")

        if metadata.get("source"):
            lines.append(f"Source: {metadata['source']}")

        lines.append("")

    return "\n".join(lines)


def parse_consolidation_response(response: Dict, concepts_data: List[Dict]) -> List[Dict[str, Any]]:
    """
    Parse the LLM response into structured consolidation decisions.

    Args:
        response: Dict from Chain.run() containing 'consolidation_decisions' and 'summary'
        concepts_data: Original concepts data for validation

    Returns:
        List of dicts with concept, original_course, consolidated_course, reason
    """
    decisions_text = response.get("consolidation_decisions", "")
    summary = response.get("summary", "")

    results = []

    # Parse each decision block using regex
    decision_pattern = r"<decision>(.*?)</decision>"
    decisions = re.findall(decision_pattern, decisions_text, re.DOTALL)

    for decision in decisions:
        concept_match = re.search(r"<concept>(.*?)</concept>", decision, re.DOTALL)
        original_match = re.search(r"<original_course>(.*?)</original_course>", decision, re.DOTALL)
        consolidated_match = re.search(r"<consolidated_course>(.*?)</consolidated_course>", decision, re.DOTALL)
        reason_match = re.search(r"<reason>(.*?)</reason>", decision, re.DOTALL)

        if concept_match and consolidated_match:
            results.append({
                "concept": concept_match.group(1).strip(),
                "original_course": original_match.group(1).strip() if original_match else "",
                "consolidated_course": consolidated_match.group(1).strip(),
                "reason": reason_match.group(1).strip() if reason_match else ""
            })

    # Log summary for debugging
    if summary:
        logger.info(f"Consolidation summary: {summary}")

    return results


def build_consolidated_resource_json(
    original_resource_json: str,
    consolidated_course_name: str,
    concepts_data: List[Dict]
) -> str:
    """
    Build the Consolidated Resource JSON.

    If consolidated course matches original, return original JSON.
    Otherwise, find the metadata for the consolidated course from the category data.

    Args:
        original_resource_json: The original Best Resource JSON string
        consolidated_course_name: The name of the consolidated course
        concepts_data: All concepts data in the category (to find course metadata)

    Returns:
        JSON string for the consolidated resource
    """
    if not original_resource_json:
        return ""

    try:
        original = json.loads(original_resource_json)
    except json.JSONDecodeError:
        return ""

    # If same course, return original
    if original.get("name") == consolidated_course_name:
        return original_resource_json

    # Find the consolidated course's metadata from category data
    courses = get_unique_courses_from_category(concepts_data)
    if consolidated_course_name in courses:
        return courses[consolidated_course_name]["original_json"]

    # Fallback: return original if consolidated course not found
    logger.warning(f"Consolidated course '{consolidated_course_name}' not found in category data")
    return original_resource_json


def consolidate_category(
    category_name: str,
    concepts_data: List[Dict[str, Any]],
    llm: str = "gemini_2_5_flash"
) -> List[Dict[str, Any]]:
    """
    Consolidate course assignments for a single category.

    Args:
        category_name: Name of the category being consolidated
        concepts_data: List of dicts with 'concept' and 'best_resource' (JSON string)
        llm: LLM model to use for consolidation

    Returns:
        List of dicts with:
        - concept: concept name
        - original_course: original course name
        - consolidated_course: consolidated course name
        - consolidated_resource_json: full JSON for consolidated resource
        - reason: explanation for the decision
    """
    # Edge case: Empty category
    if not concepts_data:
        logger.info(f"Category '{category_name}' has no concepts to consolidate")
        return []

    # Get unique courses
    courses = get_unique_courses_from_category(concepts_data)

    # Edge case: No valid courses found
    if not courses:
        logger.info(f"Category '{category_name}' has no valid course assignments")
        return [
            {
                "concept": item.get("concept", ""),
                "original_course": "",
                "consolidated_course": "",
                "consolidated_resource_json": item.get("best_resource", ""),
                "reason": "No valid course assignment to consolidate"
            }
            for item in concepts_data
        ]

    # Edge case: Already consolidated (all same course or only one unique course)
    if len(courses) == 1:
        course_name = list(courses.keys())[0]
        logger.info(f"Category '{category_name}' already consolidated - single course: {course_name}")
        return [
            {
                "concept": item.get("concept", ""),
                "original_course": course_name,
                "consolidated_course": course_name,
                "consolidated_resource_json": item.get("best_resource", ""),
                "reason": "Already consolidated - single course assignment"
            }
            for item in concepts_data
        ]

    # Format inputs for LLM
    concepts_formatted = format_concepts_for_prompt(concepts_data)
    metadata_formatted = format_course_metadata(concepts_data)

    # Build and run chain
    chain = Chain(
        llm=llm,
        tags=["consolidation_decisions", "summary"]
    )

    chain.add_message(
        role="system",
        content="You are an expert curriculum consolidation reviewer. Analyze course assignments and provide consolidation recommendations."
    )

    chain.add_message(
        role="user",
        content=consolidation_prompt.format(
            category_name=category_name,
            concepts_and_courses=concepts_formatted,
            course_metadata=metadata_formatted
        )
    )

    logger.info(f"Running consolidation for category: {category_name} ({len(concepts_data)} concepts, {len(courses)} unique courses)")

    response = chain.run()

    # Parse response
    parsed_decisions = parse_consolidation_response(response, concepts_data)

    # Build final results with consolidated JSON
    results = []

    # Create lookup from parsed decisions by concept name
    decisions_lookup = {d["concept"].lower().strip(): d for d in parsed_decisions}

    for item in concepts_data:
        concept = item.get("concept", "")
        original_json = item.get("best_resource", "")

        # Find the decision for this concept
        decision = decisions_lookup.get(concept.lower().strip())

        if decision:
            consolidated_course = decision["consolidated_course"]
            reason = decision["reason"]
            original_course = decision["original_course"]
        else:
            # Fallback if decision not found (shouldn't happen normally)
            logger.warning(f"No consolidation decision found for concept: {concept}")
            try:
                original = json.loads(original_json) if original_json else {}
                original_course = original.get("name", "")
            except json.JSONDecodeError:
                original_course = ""
            consolidated_course = original_course
            reason = "No consolidation decision from LLM"

        # Build consolidated JSON
        consolidated_json = build_consolidated_resource_json(
            original_json,
            consolidated_course,
            concepts_data
        )

        results.append({
            "concept": concept,
            "original_course": original_course,
            "consolidated_course": consolidated_course,
            "consolidated_resource_json": consolidated_json,
            "reason": reason
        })

    # Log consolidation stats
    changes = sum(1 for r in results if r["original_course"] != r["consolidated_course"] and r["original_course"])
    logger.info(f"Category '{category_name}': {changes} of {len(results)} assignments changed")

    return results
