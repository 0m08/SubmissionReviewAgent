---
id: copyright_color_transform
name: Color Palette Transform
applicable: copyright
trigger: stale_similarity
failure_class: color
priority: 3
min_consecutive_failures: 2
min_stale_rounds: 1
---

## Description
A targeted, lower-cost strategy for subjects where the primary copyright identifier
is brand colour (e.g., a product's signature orange casing, a brand's unmistakable
blue). Complete palette replacement can be tried earlier than structural changes
because it's low-risk for functional accuracy — it doesn't change shape, composition,
or the subject's educational legibility.

Best used as a parallel strategy to angle/style changes (different priority). When
similarity persists after colour transforms, escalate to structural strategies.

Applicable when: subject has a dominant branded colour, or angle rotation is either
          not suitable (subject is symmetric) or already being tried.

## Copyright Reviewer Directive

[MANDATORY STRATEGY SHIFT: Color Palette Transform — brand colour identity must be replaced]

The subject's colour palette is a primary copyright identifier. You MUST specify a
complete hue-family replacement as your PRIMARY transformation this round.

REQUIRED output in transformation_instructions:
- color_grading: MANDATORY — specify a complete palette replacement. Examples:
    "Replace all orange/red tones with a cool industrial gunmetal grey and silver
     palette. Apply uniformly to subject body, surface details, and housing."
    "Shift the entire palette to a neutral technical white-and-black scheme with
     blue accent highlights on operational indicators only."
    "Desaturate all brand colours to a flat matte grey, then re-apply a deep
     navy-blue as the dominant hue across all surfaces."
  Specify the SOURCE colour family being eliminated AND the TARGET replacement family.
- visual_style: Set to empty string — do NOT request a style change alongside colour.
- perspective_and_camera: Set to empty string — do NOT change the angle.
- All other fields: Set to empty string.

Rationale: Hue-family replacement changes the perceptual colour signature completely.
Combined with prior structural changes (if any), creates compounding differentiation.
This is a lower-cost change for the editor and should not introduce accuracy regressions.

## Example Feedback
Representative reviewer phrases that should classify to `color` and activate this strategy:
- "The orange colour palette is a distinctive brand identifier and has not been sufficiently changed"
- "The image retains the signature blue hue strongly associated with this product line"
- "The colour scheme is recognisably similar to the reference — the brand palette is intact"
- "Colour grading still matches the original — the dominant hue family must be replaced"
