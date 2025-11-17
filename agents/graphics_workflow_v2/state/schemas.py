"""
State schemas for the Graphics Workflow V2 system.

This module defines all state schemas used across the three-level hierarchical agent system:
- Level 1: Slide Supervisor
- Level 2: Segment Processor
- Level 3: Search Agent
"""

from typing import TypedDict, List, Optional, Dict, Any, Literal
from pydantic import Field
from langgraph.prebuilt.chat_agent_executor import AgentState


# ============================================================================
# Data Models (TypedDict)
# ============================================================================

class ReferenceData(TypedDict, total=False):
    """Metadata for a single reference (image/video)."""
    reference_id: str                     # Unique identifier
    url: str                              # Access URL
    type: Literal["image", "video"]       # Reference type
    title: str                            # Reference title/name
    description: str                      # What it shows
    source: str                           # Where it came from (Drive, YouTube, etc.)
    timestamp: Optional[str]              # For videos: start-end timestamps (e.g., "9:23-9:50")
    relevance_score: float                # How well it matches (0-1)
    search_query: str                     # What query found this
    metadata: Dict[str, Any]              # Additional metadata
    reused_from_segment: Optional[int]    # If reused, which segment originally found it


class SegmentData(TypedDict, total=False):
    """Data for a single VO (voiceover) segment."""
    segment_index: int                    # Position in the slide (0-indexed)
    vo_text: str                          # The voiceover text for this moment
    graphics_definition: str              # Graphics instructions (what + how)
    references: List[ReferenceData]       # Found images/videos
    status: Literal[
        "pending",                        # Not started
        "in_progress",                    # Currently being processed
        "completed",                      # Approved and finalized
        "needs_revision",                 # Rejected, needs work
        "flagged"                         # Issues requiring human intervention
    ]
    review_verdict: str                   # Latest review result
    review_feedback: str                  # Detailed feedback if rejected
    iteration_count: int                  # Number of revision attempts
    timestamp: Optional[float]            # When this segment was completed


# ============================================================================
# Agent State Schemas (extends AgentState)
# ============================================================================

class GraphicsSlideState(AgentState):
    """
    State for Slide Supervisor agent (Level 1).

    Manages the overall workflow for processing an entire slide's graphics definition.
    Inherits 'messages' field from AgentState for conversation history.
    """

    # Input
    slide_chunk: str = Field(default="", description="Full slide content to process")
    course_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional course context (course name, module, etc.)"
    )

    # Segmentation
    vo_segments: List[str] = Field(
        default_factory=list,
        description="List of VO text segments parsed from slide"
    )
    segmentation_method: Literal["auto", "manual"] = Field(
        default="auto",
        description="How segments were created: auto (LLM-parsed) or manual (pre-provided)"
    )

    # Processing state
    current_segment_index: int = Field(
        default=0,
        description="Which segment is currently being processed"
    )
    segments: List[SegmentData] = Field(
        default_factory=list,
        description="All segment data with their processing status"
    )

    # Cross-segment tracking
    all_references: Dict[str, ReferenceData] = Field(
        default_factory=dict,
        description="All found references indexed by reference_id"
    )
    reference_reuse_map: Dict[str, List[int]] = Field(
        default_factory=dict,
        description="Maps reference_id to list of segment indices using it"
    )

    # Control & safety
    max_iterations_per_segment: int = Field(
        default=3,
        description="Maximum revision attempts per segment"
    )
    total_tool_calls: int = Field(
        default=0,
        description="Total tool calls made (safety counter)"
    )
    flags: List[str] = Field(
        default_factory=list,
        description="Issues flagged for human attention"
    )

    # Output
    final_definition: Optional[str] = Field(
        default=None,
        description="Combined formatted graphics definition (when complete)"
    )
    status: Literal[
        "initialized",
        "segmenting",
        "processing_segments",
        "finalizing",
        "completed",
        "failed"
    ] = Field(
        default="initialized",
        description="Overall workflow status"
    )

    # Search infrastructure (passed to child agents)
    drive: Optional[Any] = Field(
        default=None,
        description="Google Drive instance for image/video search"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional filters for image search"
    )
    root_folder_id: Optional[str] = Field(
        default='1QS6PmCESfgFWNNEpRJDUB0E-t8iMAatH',
        description="Root folder ID for vector store"
    )
    search_k: int = Field(
        default=10,
        description="Number of search results to return per query"
    )


class SegmentProcessorState(AgentState):
    """
    State for Segment Processor agent (Level 2).

    Processes a single VO segment through define→search→review loop.
    Inherits 'messages' field from AgentState for conversation history.
    """

    # Input
    segment_index: int = Field(default=0, description="Which segment this is (0-indexed)")
    vo_text: str = Field(default="", description="VO text for THIS segment")
    slide_chunk: str = Field(default="", description="Full slide for context")

    # Context from previous segments
    previous_segments: List[SegmentData] = Field(
        default_factory=list,
        description="Completed segments before this one (for context)"
    )
    available_references: Dict[str, ReferenceData] = Field(
        default_factory=dict,
        description="References that could be reused from previous segments"
    )

    # Working data
    graphics_definition: str = Field(
        default="",
        description="Current graphics definition (evolves through revisions)"
    )
    references: List[ReferenceData] = Field(
        default_factory=list,
        description="References found/selected for this segment"
    )

    # Review & revision
    review_verdict: Literal["pending", "approved", "rejected"] = Field(
        default="pending",
        description="Current review status"
    )
    review_feedback: str = Field(
        default="",
        description="Detailed feedback from review"
    )
    iteration_count: int = Field(
        default=0,
        description="Number of revision attempts for this segment"
    )
    max_iterations: int = Field(
        default=3,
        description="Maximum allowed revision attempts"
    )

    # Flags
    cross_segment_issue: Optional[str] = Field(
        default=None,
        description="Description of any conflict with earlier segments"
    )

    # Search infrastructure (passed to Search Agent)
    drive: Optional[Any] = Field(
        default=None,
        description="Google Drive instance for image/video search"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional filters for image search"
    )
    root_folder_id: Optional[str] = Field(
        default=None,
        description="Root folder ID for vector store"
    )
    search_k: int = Field(
        default=10,
        description="Number of search results to return per query"
    )


class SearchAgentState(AgentState):
    """
    State for Search Agent (Level 3).

    Finds and validates visual references through iterative search refinement.
    Inherits 'messages' field from AgentState for conversation history.
    """

    # Input
    graphics_definition: str = Field(
        default="",
        description="Graphics definition text to search for"
    )
    visual_elements: List[str] = Field(
        default_factory=list,
        description="Parsed list of visual elements that need references"
    )

    # Available for reuse
    available_references: Dict[str, ReferenceData] = Field(
        default_factory=dict,
        description="References from previous segments that might be reused"
    )

    # Working data
    search_queries: List[str] = Field(
        default_factory=list,
        description="Generated search queries"
    )
    search_results: List[ReferenceData] = Field(
        default_factory=list,
        description="All search results found"
    )
    selected_references: List[ReferenceData] = Field(
        default_factory=list,
        description="Final selected references to return"
    )

    # Control
    search_iteration: int = Field(
        default=0,
        description="Current search refinement iteration"
    )
    max_search_iterations: int = Field(
        default=5,
        description="Maximum search refinement attempts"
    )
    min_references_needed: int = Field(
        default=1,
        description="Minimum references to find (default: 1 per visual element)"
    )

    # Search infrastructure
    drive: Optional[Any] = Field(
        default=None,
        description="Google Drive instance for image/video search"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional filters for image search"
    )
    root_folder_id: Optional[str] = Field(
        default=None,
        description="Root folder ID for vector store"
    )
    search_k: int = Field(
        default=10,
        description="Number of search results to return per query"
    )
