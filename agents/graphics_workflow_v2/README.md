# Graphics Workflow V2

A hierarchical multi-agent system built with LangGraph for automatically generating graphics definitions for educational video slides.

## Overview

The Graphics Workflow V2 uses a three-level ReAct agent hierarchy to process educational slide content and produce detailed graphics definitions with visual references.

### Architecture

```
Level 1: Slide Supervisor
  ├─ Segments slide into VO (voiceover) moments
  ├─ Processes segments sequentially
  ├─ Manages cross-segment context
  └─ Assembles final definition

Level 2: Segment Processor (per segment)
  ├─ Creates graphics definition
  ├─ Manages revision loop
  └─ Validates quality

Level 3: Search Agent (per search)
  ├─ Generates search queries
  ├─ Executes searches iteratively
  └─ Validates references
```

## Installation

The system is part of the Content-Generation-Workflow project and requires:

- Python 3.9+
- LangGraph
- LangChain
- Required API keys (OpenAI, Anthropic, Cohere)
- Google Drive access for image search

## Quick Start

```python
from agents.graphics_workflow_v2 import run_graphics_workflow
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Setup Google Drive
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

# Your slide content
slide_content = """
Superheat happens after the refrigerant has fully evaporated
into a vapor in the evaporator. At this point, it continues to
absorb heat, making it hotter than its boiling point...
"""

# Run workflow
result = run_graphics_workflow(
    slide_chunk=slide_content,
    drive=drive,
    course_context={
        "course_name": "HVAC Fundamentals",
        "module": "Refrigeration Cycle"
    }
)

# Get final definition
from agents.graphics_workflow_v2 import get_final_definition, get_workflow_summary

definition = get_final_definition(result)
summary = get_workflow_summary(result)

print(definition)
print(f"Processed {summary['total_segments']} segments")
print(f"Found {summary['total_references']} references")
```

## Configuration

Configuration is centralized in `config/settings.py`:

### Model Selection

```python
SLIDE_SUPERVISOR_MODEL = "claude-sonnet-4"      # Orchestration
SEGMENT_PROCESSOR_MODEL = "claude-sonnet-3-5"   # Segment processing
SEARCH_AGENT_MODEL = "gpt-4o-mini"              # Search operations
```

### Iteration Limits

```python
MAX_ITERATIONS_PER_SEGMENT = 3     # Revisions per segment
MAX_SEARCH_ITERATIONS = 5          # Search refinements
SLIDE_SUPERVISOR_RECURSION_LIMIT = 100  # Total tool calls
```

### Search Parameters

```python
SEARCH_K = 10                      # Results per query
MIN_REFERENCES_PER_SEGMENT = 1     # Minimum references needed
RELEVANCE_THRESHOLD = 0.6          # Minimum similarity score
```

### Behavior Flags

```python
ENABLE_REFERENCE_REUSE = True              # Allow reusing references
ENABLE_CROSS_SEGMENT_VALIDATION = True     # Check for contradictions
AUTO_FLAG_ON_MAX_ITERATIONS = True         # Flag problematic segments
```

## Output Format

The system produces a formatted graphics definition with:

### Markdown Format (default)

```markdown
# GRAPHICS DEFINITION

## Metadata
- Generated: 2025-01-15 10:30:00
- Total Segments: 4
- Total References: 7

---

## SEGMENT 1 of 4

**When VO:** "Superheat happens after..."

### GRAPHICS:
[Detailed graphics instructions]

### REFERENCES:
**[1] Evaporator Diagram**
- URL: https://...
- Type: image
- Relevance: 0.92

---
```

### Programmatic Access

```python
# Access individual segments
for segment in result["segments"]:
    print(f"Segment {segment['segment_index']}:")
    print(f"  VO: {segment['vo_text']}")
    print(f"  Definition: {segment['graphics_definition']}")
    print(f"  References: {len(segment['references'])}")

# Access all references
all_refs = result["all_references"]
for ref_id, ref in all_refs.items():
    print(f"{ref['title']}: {ref['url']}")
```

## Workflow Details

### 1. Segmentation

The Slide Supervisor segments the slide into VO moments using:
- **Concept-based**: Segments by major ideas (default)
- **Sentence-based**: 1-3 sentences per segment
- **Time-based**: 10-15 seconds per segment

### 2. Segment Processing

Each segment goes through:
1. **Define** - Create graphics definition
2. **Search** - Find visual references
3. **Review** - Validate quality
4. **Revise** - If rejected (max 3 iterations)
5. **Finalize** - Mark as complete

### 3. Quality Criteria

Segments are reviewed against:
- ✓ Definition aligns with VO text
- ✓ At least 1 relevant reference per visual element
- ✓ No contradictions with previous segments
- ✓ Specific enough for animator to implement
- ✓ All references are accessible

### 4. Reference Reuse

The system automatically:
- Tracks all found references
- Checks if previous references can be reused
- Avoids redundant searches
- Maintains reference→segment mapping

## Error Handling

### Automatic Flags

The system flags issues for human review:
- Segments reaching max iterations without approval
- Cross-segment contradictions detected
- Search failures or missing references
- Processing errors

### Graceful Degradation

- Failed segments are flagged but don't block processing
- Workflow continues with best attempts
- All segments processed (completed or flagged)
- Final definition includes flags section

## Advanced Usage

### Custom Configuration

```python
result = run_graphics_workflow(
    slide_chunk=slide,
    drive=drive,
    max_iterations_per_segment=5,      # Override default
    recursion_limit=150,                # Allow more tool calls
    tags=["custom_run", "experiment"],  # LangSmith tags
)
```

### Filters for Search

```python
result = run_graphics_workflow(
    slide_chunk=slide,
    drive=drive,
    filters={
        "course_name": "HVAC Fundamentals",
        "image_type": ["diagram", "photo"],
        "stock_type": "custom"  # vs "stock"
    }
)
```

### Manual Segmentation

```python
# Pre-segment your slide
manual_segments = [
    "First VO moment...",
    "Second VO moment...",
    "Third VO moment..."
]

# Then join and pass
slide = "\n\n".join(manual_segments)
result = run_graphics_workflow(slide_chunk=slide, drive=drive)
```

## Monitoring & Debugging

### LangSmith Tracing

Enable in `config/settings.py`:
```python
ENABLE_LANGSMITH_TRACING = True
```

View traces at: https://smith.langchain.com

### Verbose Logging

```python
ENABLE_VERBOSE_LOGGING = True
```

### Accessing Agent Messages

```python
# See conversation history
for msg in result["messages"]:
    print(f"{msg['role']}: {msg['content'][:100]}...")
```

## File Structure

```
graphics_workflow_v2/
├── __init__.py                    # Package exports
├── README.md                      # This file
├── state/
│   └── schemas.py                 # State definitions
├── config/
│   └── settings.py                # Configuration
├── agents/
│   ├── slide_supervisor.py        # Level 1 agent
│   ├── segment_processor.py       # Level 2 agent
│   └── search_agent.py            # Level 3 agent
├── tools/
│   ├── slide_supervisor/
│   │   └── supervisor_tools.py
│   ├── segment_processor/
│   │   └── segment_tools.py
│   └── search_agent/
│       └── search_tools.py
├── prompts/
│   ├── slide_supervisor_prompts.py
│   ├── segment_processor_prompts.py
│   └── search_agent_prompts.py
└── utils/
    └── (future utilities)
```

## Performance Considerations

### Cost Optimization

- Supervisor uses highest-tier model (claude-sonnet-4)
- Segment processing uses mid-tier (claude-sonnet-3-5)
- Search uses cheapest (gpt-4o-mini)
- Average cost per slide: ~$0.50-2.00 (varies by complexity)

### Latency

- Typical slide (4-6 segments): 3-5 minutes
- Simple slide (1-2 segments): 1-2 minutes
- Complex slide (8-10 segments): 6-10 minutes

### Optimization Tips

1. Use cheaper models for non-critical operations
2. Reduce `MAX_SEARCH_ITERATIONS` if search quality is good
3. Lower `SEARCH_K` if finding too many irrelevant results
4. Enable reference reuse to avoid redundant searches

## Troubleshooting

### "No segments could be extracted"

- Check slide content is not empty
- Ensure MIN_SEGMENT_LENGTH is appropriate
- Try manual segmentation

### "Search failed: Drive not authenticated"

- Verify Google Drive authentication
- Check `drive` instance is valid
- Ensure internet connection

### "Max iterations reached without approval"

- Review flagged segments in output
- Check if review criteria are too strict
- Consider increasing MAX_ITERATIONS_PER_SEGMENT

### "Cross-segment contradiction"

- Review flagged segments
- May require human intervention to resolve
- Consider adjusting earlier segments manually

## Contributing

When extending the system:

1. Follow the hierarchical agent pattern
2. Use Command objects for state updates
3. Add configuration to `settings.py`
4. Document new tools and agents
5. Add tests for new functionality

## Future Enhancements

Potential improvements:
- [ ] Parallel segment processing (with dependency management)
- [ ] Video timestamp extraction
- [ ] Animation script generation
- [ ] Quality scoring and ranking
- [ ] Human-in-the-loop feedback integration
- [ ] Export to video editing formats

## License

Part of the Content-Generation-Workflow project.

## Support

For issues or questions:
- Check this README first
- Review configuration in `config/settings.py`
- Check LangSmith traces for debugging
- Review flagged segments in output
