---
id: copyright_abstraction_escalation
name: Concept Abstraction
applicable: copyright
trigger: stale_similarity
failure_class: any
priority: 4
min_consecutive_failures: 4
min_stale_rounds: 3
---

## Description
Last-resort copyright strategy. When all structural, medium, and colour approaches have
failed — typically because the subject is an extremely distinctive iconic product — the
only remaining path is to abstract the subject into a generic, archetypal representation
of the CATEGORY rather than the specific branded product.

This strategy accepts a trade-off: some product-specific fidelity is lost in exchange
for guaranteed copyright safety. The educational concept is preserved. The specific
product identity is not.

Use ONLY when: min 4 copyright failures AND min 3 stale similarity rounds.
All prior strategies must have been exhausted first (priority 1–3).

## Copyright Reviewer Directive

[MANDATORY STRATEGY SHIFT: Concept Abstraction — all prior transformation approaches exhausted]

Structural angle changes, medium migration, and palette changes have not sufficiently
reduced similarity. You MUST now specify a subject-level abstraction as your PRIMARY
transformation. The goal is to replace the specific branded product with a generic
archetypal equivalent.

REQUIRED output in transformation_instructions:
- subject_focus: MANDATORY — describe the archetypal replacement. Examples:
    "Replace the specific branded product with a generic, universal equivalent of
     the same device category — no brand-identifiable form factors, ergonomic styling,
     or distinctive industrial design. Use simplified geometric forms."
    "Render a generic reference-quality schematic of this type of component, as it
     would appear in a neutral technical training manual from the 1990s — completely
     unbranded, simplified, universally recognisable as the category."
  Specify BOTH what to remove (specific branded identity) AND what to replace it with
  (generic archetypal representation of the same category).
- visual_style: MANDATORY — set to "Clean technical diagram style, neutral and
  completely unbranded. No manufacturer-specific design languages."
- color_grading: MANDATORY — set to "Neutral grey/silver palette. No brand colours."
- perspective_and_camera: OPTIONAL — choose the angle that best serves the
  educational/generic representation, not the original reference angle.
- All remaining fields: Set to empty.

Educational accuracy preservation: The component MUST still serve the same educational
purpose. If a pressure gauge was being taught, a generic pressure gauge must still be
present, correctly labelled, and functionally legible.

## Example Feedback
Representative phrases indicating all prior strategies exhausted — matches `any` failure class:
- "All previous transformation approaches have not reduced visual similarity sufficiently"
- "The product is still immediately recognisable despite angle, style, and colour changes"
- "This is an iconic branded product — its industrial design is unmistakably identifiable"
- "After multiple rounds of transformation, the image still closely resembles the original reference"
