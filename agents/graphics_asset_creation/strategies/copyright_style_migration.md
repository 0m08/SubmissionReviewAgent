---
id: copyright_style_migration
name: Style Migration
applicable: copyright
trigger: stale_similarity
failure_class: structural
priority: 2
min_consecutive_failures: 3
min_stale_rounds: 2
---

## Description
Activates after 3+ copyright failures, when perspective shifts have not been sufficient.
Forces a medium change — converting the image from photographic realism to a technical
illustration, vector diagram, or engineering render. This eliminates the photographic
fingerprint entirely, which is the most reliable way to defeat pixel-level similarity
when the subject's shape and composition are inherently identifiable.

Applicable when: angle rotation was tried and failed, or subject is a well-known
          product where structural similarity persists despite angle changes.

## Copyright Reviewer Directive

[MANDATORY STRATEGY SHIFT: Style Migration — structural transformations have been insufficient]

Camera angle changes and colour adjustments have not reduced similarity enough. You MUST
now specify a full visual medium change as your PRIMARY transformation this round.

REQUIRED output in transformation_instructions:
- visual_style: MANDATORY — specify a complete medium change. Examples:
    "Convert from photographic realism to high-fidelity technical illustration with
     clean linework, flat colour fills, and schematic-style labels"
    "Render as a professional engineering exploded-view diagram — no photographic
     textures, no real-world lighting, no brand surface finishes"
    "Transform to a clean vector-art cutaway diagram on a white background"
  The output must be unmistakably non-photographic. Name the target medium explicitly.
- perspective_and_camera: Set to empty string — do NOT request an angle change
  simultaneously. The style migration IS the transformation this round.
- subject_focus: OPTIONAL — only if technical components need to be labelled or
  annotated consistent with the diagram/illustration style you specified.
- color_grading: OPTIONAL — suggest a neutral technical palette (e.g., grey-scale
  with accent blue) ONLY if it reinforces the illustration style.
- All other fields: Set to empty string.

Rationale: Medium changes (photo → diagram) have near-zero visual similarity to the
original because they eliminate all photographic texture, lighting, and surface cues.

## Example Feedback
Representative reviewer phrases that should classify to `structural` after angle rotation has been tried:
- "Despite the angle change, the photographic realism still closely resembles the reference product"
- "Structural similarity persists — the surface textures and physical form factor are still recognisable"
- "Perspective was changed but the image is still identifiable as the same branded product"
- "The silhouette and physical form remain too similar to the original even after rotation"
