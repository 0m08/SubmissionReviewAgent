"""
Configuration settings for the Graphics Workflow V2 system.

This module contains all configurable parameters for the hierarchical agent system,
including model selections, limits, thresholds, and behavior flags.
"""

from typing import Literal


# ============================================================================
# Model Configuration
# ============================================================================

# Highest intelligence for orchestration
SLIDE_SUPERVISOR_MODEL = "openai:gpt-5-mini"

# Good balance for segment work
SEGMENT_PROCESSOR_MODEL = "openai:gpt-5-mini"

# Fast/cheap for search operations
SEARCH_AGENT_MODEL = "openai:gpt-5-mini"

# Good evaluation capabilities for review
REVIEW_CHAIN_MODEL = "gpt5_mini_thinking"

# Chain model for definition generation
DEFINITION_CHAIN_MODEL = "gpt5_mini_thinking"

# Chain model for segmentation
SEGMENTATION_CHAIN_MODEL = "gpt5_mini_thinking"

# ============================================================================
# Iteration & Recursion Limits
# ============================================================================

# Maximum revision attempts per segment
MAX_ITERATIONS_PER_SEGMENT = 3

# Maximum search refinement iterations
MAX_SEARCH_ITERATIONS = 5

# Total tool calls allowed for Slide Supervisor (entire slide processing)
SLIDE_SUPERVISOR_RECURSION_LIMIT = 100

# Tool calls allowed per Segment Processor
SEGMENT_PROCESSOR_RECURSION_LIMIT = 30

# Tool calls allowed for Search Agent
SEARCH_AGENT_RECURSION_LIMIT = 25


# ============================================================================
# Search Parameters
# ============================================================================

# Number of results to retrieve per search query
SEARCH_K = 10

# Minimum references needed per segment
MIN_REFERENCES_PER_SEGMENT = 1

# Maximum references to return per segment
MAX_REFERENCES_PER_SEGMENT = 10

# Minimum similarity/relevance score to consider a reference (0-1)
RELEVANCE_THRESHOLD = 0.6

# Number of search queries to generate per visual element
SEARCH_QUERIES_PER_ELEMENT = 3


# ============================================================================
# Quality Thresholds
# ============================================================================

# Minimum characters in a VO segment
MIN_SEGMENT_LENGTH = 10

# Maximum characters in a VO segment
MAX_SEGMENT_LENGTH = 500

# Minimum characters in a graphics definition
MIN_DEFINITION_LENGTH = 50

# Maximum number of segments per slide (safety limit)
MAX_SEGMENTS_PER_SLIDE = 20


# ============================================================================
# Timeout Settings (seconds)
# ============================================================================

# Timeout for processing a single segment
SEGMENT_TIMEOUT_SECONDS = 300  # 5 minutes

# Timeout for search operations
SEARCH_TIMEOUT_SECONDS = 120  # 2 minutes


# ============================================================================
# Behavior Flags
# ============================================================================

# Allow reusing references from previous segments
ENABLE_REFERENCE_REUSE = True

# Check for contradictions between segments
ENABLE_CROSS_SEGMENT_VALIDATION = True

# Automatically flag segments that hit max iterations
AUTO_FLAG_ON_MAX_ITERATIONS = True

# Enable verbose logging for debugging
ENABLE_VERBOSE_LOGGING = False

# Enable LangSmith tracing
ENABLE_LANGSMITH_TRACING = True


# ============================================================================
# Prompt Configuration
# ============================================================================

# Strategy for segmenting slide into VO moments
# Options: "concept_based" | "sentence_based" | "time_based"
SEGMENTATION_STRATEGY: Literal["concept_based", "sentence_based", "time_based"] = "concept_based"

# Level of detail in graphics definitions
# Options: "high" | "medium" | "low"
DEFINITION_DETAIL_LEVEL: Literal["high", "medium", "low"] = "high"

# Diversity of search queries
# Options: "high" | "medium" | "low"
SEARCH_QUERY_DIVERSITY: Literal["high", "medium", "low"] = "high"

# Strictness of review process
# Options: "strict" | "medium" | "lenient"
REVIEW_STRICTNESS: Literal["strict", "medium", "lenient"] = "medium"


# ============================================================================
# Output Formatting
# ============================================================================

# Include timestamps in segment data
INCLUDE_TIMESTAMPS = True

# Include metadata in final output
INCLUDE_METADATA = True

# Format for final definition output
# Options: "markdown" | "plain_text" | "structured_json"
OUTPUT_FORMAT: Literal["markdown", "plain_text", "structured_json"] = "markdown"


# ============================================================================
# Helper Functions
# ============================================================================

def get_model_config() -> dict:
    """Returns model configuration as a dictionary."""
    return {
        "slide_supervisor": SLIDE_SUPERVISOR_MODEL,
        "segment_processor": SEGMENT_PROCESSOR_MODEL,
        "search_agent": SEARCH_AGENT_MODEL,
        "review_chain": REVIEW_CHAIN_MODEL,
        "definition_chain": DEFINITION_CHAIN_MODEL,
        "segmentation_chain": SEGMENTATION_CHAIN_MODEL,
    }


def get_limit_config() -> dict:
    """Returns limit configuration as a dictionary."""
    return {
        "max_iterations_per_segment": MAX_ITERATIONS_PER_SEGMENT,
        "max_search_iterations": MAX_SEARCH_ITERATIONS,
        "slide_supervisor_recursion_limit": SLIDE_SUPERVISOR_RECURSION_LIMIT,
        "segment_processor_recursion_limit": SEGMENT_PROCESSOR_RECURSION_LIMIT,
        "search_agent_recursion_limit": SEARCH_AGENT_RECURSION_LIMIT,
    }


def get_search_config() -> dict:
    """Returns search configuration as a dictionary."""
    return {
        "search_k": SEARCH_K,
        "min_references": MIN_REFERENCES_PER_SEGMENT,
        "max_references": MAX_REFERENCES_PER_SEGMENT,
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "queries_per_element": SEARCH_QUERIES_PER_ELEMENT,
    }


def get_all_config() -> dict:
    """Returns complete configuration as a dictionary."""
    return {
        "models": get_model_config(),
        "limits": get_limit_config(),
        "search": get_search_config(),
        "quality_thresholds": {
            "min_segment_length": MIN_SEGMENT_LENGTH,
            "max_segment_length": MAX_SEGMENT_LENGTH,
            "min_definition_length": MIN_DEFINITION_LENGTH,
            "max_segments_per_slide": MAX_SEGMENTS_PER_SLIDE,
        },
        "timeouts": {
            "segment_timeout": SEGMENT_TIMEOUT_SECONDS,
            "search_timeout": SEARCH_TIMEOUT_SECONDS,
        },
        "flags": {
            "enable_reference_reuse": ENABLE_REFERENCE_REUSE,
            "enable_cross_segment_validation": ENABLE_CROSS_SEGMENT_VALIDATION,
            "auto_flag_on_max_iterations": AUTO_FLAG_ON_MAX_ITERATIONS,
            "enable_verbose_logging": ENABLE_VERBOSE_LOGGING,
            "enable_langsmith_tracing": ENABLE_LANGSMITH_TRACING,
        },
        "prompts": {
            "segmentation_strategy": SEGMENTATION_STRATEGY,
            "definition_detail_level": DEFINITION_DETAIL_LEVEL,
            "search_query_diversity": SEARCH_QUERY_DIVERSITY,
            "review_strictness": REVIEW_STRICTNESS,
        },
        "output": {
            "include_timestamps": INCLUDE_TIMESTAMPS,
            "include_metadata": INCLUDE_METADATA,
            "output_format": OUTPUT_FORMAT,
        },
    }
