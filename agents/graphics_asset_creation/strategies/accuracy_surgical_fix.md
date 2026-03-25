---
id: accuracy_surgical_fix
name: Surgical Component Fix
applicable: accuracy
trigger: consecutive_failures
failure_class: factual
priority: 1
min_consecutive_failures: 2
min_stale_rounds: 0
---

## Description
Activates after 2 consecutive accuracy rejections. Normal reviewer thinking tends to
suggest multiple concurrent changes, which makes it harder to isolate the actual failure
and often introduces regressions by touching parts of the image that were already
correct. After 2 failures, this strategy constrains the reviewer to identify and
specify ONE fix only — the single most critical factual error.

The reviewer's recommended_instructions should become hyper-specific and minimal,
not a list of improvements but a precisely scoped, field-level correction request.

## Accuracy Reviewer Directive

[MANDATORY STRATEGY SHIFT: Surgical Fix Mode — previous broad corrections have not resolved the issue]

Your previous recommended_instructions addressed multiple changes, which has not resolved
the accuracy problem. You MUST now narrow your output to a single, precisely targeted fix.

REQUIRED output in recommended_instructions:
- additional_comments: MANDATORY — describe ONE component fix with clinical precision.
  Format: "[Component name] shows [current wrong state]. It must show [correct state]."
  Examples:
    "The pressure gauge needle points to 120 PSI. It must point to 180 PSI — the
     maximum operational pressure described in the voiceover."
    "The valve handle is shown in the OPEN position (horizontal). It must be in the
     CLOSED position (vertical) as described in the voiceover procedure."
    "The LCD display shows '00:00'. It must show the active timer state as described."
  Be specific: name the component, describe the current wrong state, describe the
  required correct state. One sentence. One component. No ambiguity.
- subject_focus: MANDATORY EMPTY — set to empty string. Do not add subject changes.
- visual_style: MANDATORY EMPTY — set to empty string. Style is not the problem.

DO NOT suggest compositional, environmental, lighting, or background changes.
DO NOT list multiple issues. Fix the single most critical factual error ONLY.
Rationale: Focused single-component instructions have a higher execution success rate
because the editor does not need to prioritise between conflicting changes.

## Example Feedback
Representative reviewer phrases that should classify to `factual` and activate this strategy:
- "The pressure gauge needle points to 0 PSI but should read 180 PSI as described in the voiceover"
- "The valve handle is shown in the OPEN position (horizontal) but must be CLOSED (vertical)"
- "The LCD display shows '00:00' — it must show the active timer state described in the procedure"
- "The component label reads 'Input' but should read 'Output' according to the voiceover"
