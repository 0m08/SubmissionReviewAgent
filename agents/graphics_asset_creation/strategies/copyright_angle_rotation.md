---
id: copyright_angle_rotation
name: Angle Rotation
applicable: copyright
trigger: stale_similarity
failure_class: structural
priority: 1
min_consecutive_failures: 2
min_stale_rounds: 2
---

## Description
Activates when copyright similarity scores have not decreased over 2+ consecutive rounds,
meaning the current transformation approach is not creating sufficient visual
differentiation. The primary lever is a major camera perspective shift — not a tweak,
but a full axis change. This is the first copyright strategy to try because perspective
shifts reliably change the visual fingerprint without affecting functional accuracy.

Applicable when: similarity plateau detected, fewer than 2 prior copyright strategies tried.

## Copyright Reviewer Directive

[MANDATORY STRATEGY SHIFT: Angle Rotation — previous transformation approach exhausted]

Your previous transformation_instructions have not reduced visual similarity adequately.
You MUST pivot to a camera angle rotation as your PRIMARY transformation this round.

REQUIRED output in transformation_instructions:
- perspective_and_camera: MANDATORY — specify a major, named axis change. Examples:
    "Rotate from frontal face-on view to a 45-degree isometric perspective"
    "Shift from eye-level product shot to overhead bird's-eye view"
    "Change from straight-on to a low 3/4 profile angle"
  The angle change must be at least 45 degrees. Name the before-state and after-state.
- visual_style: Set to empty string — do NOT request a style change alongside rotation.
  One major change per round prevents conflicting instructions reaching the editor.
- subject_focus: Set to empty string unless the subject needs repositioning for the
  new angle to make sense.
- All other fields: Set to empty string. The perspective shift IS the transformation.

Rationale: A single, decisive structural change is more reliably executed than
simultaneous multi-axis changes. Confirm the angle rotation is unmistakably different.

## Example Feedback
Representative reviewer phrases that should classify to `structural` and activate this strategy:
- "The composition is nearly identical to the reference — same frontal angle and product silhouette"
- "The perspective and camera viewpoint match the reference image too closely"
- "The overall form factor and profile are recognisably similar to the original product"
- "Same angle as the source — the product shape is still immediately identifiable"
