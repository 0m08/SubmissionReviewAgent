"""
Example usage of the Graphics Workflow V2 system.

This script demonstrates how to use the hierarchical agent system to
generate graphics definitions for educational slides.
"""

from agents.graphics_workflow_v2 import (
    run_graphics_workflow,
    get_final_definition,
    get_workflow_summary,
)


def example_with_superheat_slide():
    """
    Example using the superheat slide from HVAC course.
    """
    # Example slide content
    slide_content = """Superheat happens after the refrigerant has fully evaporated into a vapor in the evaporator. At this point, it continues to absorb heat, making it hotter than its boiling point. This extra heat is called superheat - and it's important because it confirms that only vapor, not liquid, is entering the compressor. Liquid in the compressor can cause serious damage, so checking superheat helps ensure the system is running safely."""

    print("=" * 80)
    print("GRAPHICS WORKFLOW V2 - EXAMPLE USAGE")
    print("=" * 80)
    print("\nSlide Content:")
    print(slide_content)
    print("\n" + "=" * 80)

    # Note: This requires Google Drive authentication
    # Uncomment and setup when ready to test with actual Drive instance
    """
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive

    # Authenticate
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

    # Run workflow
    print("\nRunning graphics workflow...")
    result = run_graphics_workflow(
        slide_chunk=slide_content,
        drive=drive,
        course_context={
            "course_name": "HVAC Fundamentals",
            "module": "Refrigeration Cycle",
            "topic": "Superheat"
        }
    )

    # Get results
    final_def = get_final_definition(result)
    summary = get_workflow_summary(result)

    print("\n" + "=" * 80)
    print("WORKFLOW SUMMARY")
    print("=" * 80)
    print(f"Status: {summary['status']}")
    print(f"Total Segments: {summary['total_segments']}")
    print(f"Completed: {summary['completed_segments']}")
    print(f"Flagged: {summary['flagged_segments']}")
    print(f"Total References: {summary['total_references']}")
    print(f"Unique References: {summary['unique_references']}")
    print(f"Avg Iterations/Segment: {summary['avg_iterations_per_segment']}")

    if summary['flags']:
        print(f"\nFlags ({summary['total_flags']}):")
        for flag in summary['flags']:
            print(f"  - {flag}")

    print("\n" + "=" * 80)
    print("FINAL GRAPHICS DEFINITION")
    print("=" * 80)
    print(final_def)

    # Save to file
    with open("graphics_definition_output.md", "w") as f:
        f.write(final_def)
    print("\n✓ Saved to graphics_definition_output.md")
    """

    print("\nNOTE: Actual execution requires Google Drive authentication.")
    print("Uncomment the code above and setup Drive to run the full workflow.")


def example_mock_workflow():
    """
    Example showing the structure without actual execution.
    This can run without Drive authentication.
    """
    print("\n" + "=" * 80)
    print("MOCK WORKFLOW STRUCTURE")
    print("=" * 80)

    print("""
The workflow follows this pattern:

1. SEGMENTATION
   Input: Full slide content
   Output: List of VO segments
   Example:
   - Segment 0: "Superheat happens after..."
   - Segment 1: "At this point, it continues..."
   - Segment 2: "This extra heat is called superheat..."
   - Segment 3: "Liquid in the compressor..."

2. SEGMENT PROCESSING (for each segment)
   a. Define Graphics
      → Creates detailed graphics instructions

   b. Search References
      → Finds images/videos from vector store
      → Checks for reusable references from previous segments

   c. Review Quality
      → Validates against criteria checklist

   d. Revise if Rejected
      → Up to 3 revision attempts

   e. Finalize
      → Mark as complete

3. FINAL ASSEMBLY
   → Combines all segments
   → Formats as Markdown/JSON/Plain text
   → Includes all references with metadata

Expected Output Structure:
{
    "status": "completed",
    "segments": [
        {
            "segment_index": 0,
            "vo_text": "...",
            "graphics_definition": "...",
            "references": [...]
        },
        ...
    ],
    "final_definition": "...",
    "flags": [...]
}
""")


def example_configuration():
    """
    Example showing how to customize configuration.
    """
    from agents.graphics_workflow_v2.config import settings

    print("\n" + "=" * 80)
    print("CONFIGURATION EXAMPLES")
    print("=" * 80)

    print("\nCurrent Model Configuration:")
    print(f"  Slide Supervisor: {settings.SLIDE_SUPERVISOR_MODEL}")
    print(f"  Segment Processor: {settings.SEGMENT_PROCESSOR_MODEL}")
    print(f"  Search Agent: {settings.SEARCH_AGENT_MODEL}")

    print("\nCurrent Limits:")
    print(f"  Max Iterations/Segment: {settings.MAX_ITERATIONS_PER_SEGMENT}")
    print(f"  Max Search Iterations: {settings.MAX_SEARCH_ITERATIONS}")
    print(f"  Supervisor Recursion Limit: {settings.SLIDE_SUPERVISOR_RECURSION_LIMIT}")

    print("\nSearch Configuration:")
    print(f"  Results per Query: {settings.SEARCH_K}")
    print(f"  Min References/Segment: {settings.MIN_REFERENCES_PER_SEGMENT}")
    print(f"  Relevance Threshold: {settings.RELEVANCE_THRESHOLD}")

    print("\nTo customize, modify agents/graphics_workflow_v2/config/settings.py")
    print("Or pass kwargs to run_graphics_workflow():")
    print("""
    result = run_graphics_workflow(
        slide_chunk=slide,
        drive=drive,
        max_iterations_per_segment=5,  # Override default
        recursion_limit=150,            # Custom limit
    )
    """)


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "GRAPHICS WORKFLOW V2 - EXAMPLES" + " " * 26 + "║")
    print("╚" + "=" * 78 + "╝")

    # Run examples
    example_with_superheat_slide()
    example_mock_workflow()
    example_configuration()

    print("\n" + "=" * 80)
    print("For more information, see README.md")
    print("=" * 80 + "\n")
