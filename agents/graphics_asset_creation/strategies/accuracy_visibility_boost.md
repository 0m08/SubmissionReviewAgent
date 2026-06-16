---
id: accuracy_visibility_boost
name: Component Visibility Boost
applicable: accuracy
trigger: consecutive_failures
failure_class: visibility
priority: 2
min_consecutive_failures: 3
min_stale_rounds: 0
---

## Description
Activates after 3 consecutive accuracy failures. A distinct failure mode from component
factual error: the component IS technically present, but it is too small, too peripheral,
too obscured, or badly lit to be confirmed by either the reviewer LLM or a human viewer.
In this case, editing the component itself is lower-value than reframing the scene to
bring it into clear, dominant view.

The reviewer's recommended_instructions should direct the editor to change the
composition and framing, NOT to modify the component's functional state.

## Accuracy Reviewer Directive

[MANDATORY STRATEGY SHIFT: Visibility Boost Mode — the failing component cannot be confirmed visually]

After multiple correction rounds, the target component remains unclear or unconfirmable
in the image. This is a FRAMING problem, not a component state problem. You MUST now
instruct the editor to reframe the composition so the component is dominant and clear.

REQUIRED output in recommended_instructions:
- subject_focus: MANDATORY — specify a deliberate crop/zoom/reframe. Examples:
    "Zoom into the pressure gauge area so it fills at least 40% of the frame.
     The gauge face, needle position, and scale markings must be fully legible."
    "Reframe the composition to place the valve assembly at the centre foreground.
     It must not be obscured by pipes, panels, or other components."
    "Crop to show only the control panel and the specific button being described.
     Remove all background machinery that is not relevant to the voiceover."
  Specify WHICH component to bring forward and HOW MUCH of the frame it should fill.
- additional_comments: MANDATORY — add lighting specification:
    "Apply direct, flat, even lighting on [component name] to eliminate shadows
     that obscure surface detail. No dramatic or directional lighting."
- visual_style: MANDATORY EMPTY — set to empty string. Style changes are irrelevant here.

DO NOT suggest changing the component's functional state in this round.
DO NOT suggest colour, style, or background changes.
Rationale: A clearly visible correct component always scores better with the accuracy
reviewer than a well-edited but poorly framed component that cannot be confirmed.

## Example Feedback
Representative reviewer phrases that should classify to `visibility` and activate this strategy:
- "The valve handle appears to be present but it is too small to confirm its position clearly"
- "Cannot see the gauge face clearly — it is obscured by reflections and the narrow camera angle"
- "The component is barely visible and hard to distinguish from the background equipment"
- "Hard to confirm the state of the indicator — it is blocked by adjacent piping"
