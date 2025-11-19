"""
State schema for the paraphraser agent orchestrator.

Uses LangGraph's MessagesState as base for message history tracking.
"""

from typing import TypedDict, Annotated, Optional, List, Dict
from langgraph.graph import MessagesState
from typing_extensions import NotRequired


class ParaphraserState(MessagesState):
    """
    State that flows through the orchestrator graph.

    Extends MessagesState to include message history tracking
    plus custom fields for paraphrasing workflow.
    """

    # Input parameters
    original_text: str
    """The text to paraphrase"""

    trade: str
    """Trade context (e.g., 'HVAC', 'Electrical')"""

    specialization: NotRequired[Optional[str]]
    """Optional sub-specialization (e.g., 'Residential HVAC')"""

    preserve_formatting: NotRequired[bool]
    """Whether to maintain structure (lists, paragraphs, etc.)"""

    target_length: NotRequired[Optional[str]]
    """Target length: 'similar', 'concise', or 'expanded'"""

    # Working data
    paraphrased_text: NotRequired[Optional[str]]
    """Current paraphrased version"""

    quality_report: NotRequired[Optional[Dict]]
    """Reviewer's assessment with pass/fail, issues, score, feedback"""

    refinement_count: NotRequired[int]
    """Number of refinement iterations performed"""

    # Output
    final_text: NotRequired[Optional[str]]
    """Final approved output"""

    status: NotRequired[str]
    """Current status: 'in_progress', 'completed', 'failed'"""

    # Metadata for debugging
    iteration_history: NotRequired[List[Dict]]
    """Track all iterations with timestamps and decisions"""
