---
id: accuracy_scene_reframe
name: Camera Perspective Reframe
applicable: accuracy
trigger: consecutive_failures
failure_class: scene_mismatch
priority: 3
min_consecutive_failures: 4
min_stale_rounds: 0
---

## Description
Last-resort accuracy strategy. After 4+ consecutive failures where surgical fixes and
visibility boosts have not resolved the issue, the root cause is likely that the
current camera angle or composition is fundamentally mismatched to what the voiceover
describes — key components are out of frame, the wrong side is shown, or the viewing
angle makes the subject unrecognisable.

In this case, the reviewer must specify a decisive camera perspective shift. The
editor will apply this as a perspective/framing edit to the existing image — NOT
a full scene regeneration. The image is edited in place; only the camera angle
and composition change.

## Accuracy Reviewer Directive

[MANDATORY STRATEGY SHIFT: Camera Perspective Reframe — change viewing angle to
expose the correct components. Do NOT request full scene regeneration.]

After 4+ correction rounds, the issue appears to be the camera angle or composition:
the current viewpoint either hides the relevant components, shows the wrong side of
the subject, or makes the key elements unrecognisable. You MUST now specify a
significant perspective change to expose what the voiceover describes.

REQUIRED output in recommended_instructions:
- perspective_and_camera: MANDATORY — specify a significant, named camera shift.
  Always state the CURRENT angle first, then describe the target angle.
  Format: "Currently [current viewpoint]. Rotate/shift to [target viewpoint] so
  that [component] becomes visible/legible."
  Examples:
    "Currently a front-facing eye-level view of the control panel. Rotate 40°
     clockwise around the vertical axis to a front-right 3/4 view so the valve
     handle position is visible on the right side."
    "Currently a straight-on overhead view that hides the gauge face. Lower the
     camera 35° to an elevated 3/4 angle so the gauge face and needle are
     directly readable."
    "Currently a wide-angle shot where the target component occupies <10% of
     frame. Zoom in 2× and reframe to fill 50% of frame with the component."
  The angle change must be at least 30°, or a reframe/zoom that significantly
  changes what is visible. Name both the before-state and after-state.
- subject_focus: MANDATORY EMPTY — set to empty string.
- additional_comments: MANDATORY EMPTY — set to empty string.
- visual_style: MANDATORY EMPTY — set to empty string.

DO NOT suggest anything other than a camera/perspective change.
DO NOT request regenerating or re-creating the scene from scratch.
The editor will apply this as an in-place perspective edit to the existing image.
Rationale: A decisive camera shift changes which components are in frame and
readable, often resolving scene-mismatch failures without touching content.

## Example Feedback
Representative reviewer phrases that should classify to `scene_mismatch` and activate this strategy:
- "The image shows the wrong side of the equipment — the valve is on the back and not visible"
- "The current overhead angle hides the gauge face completely"
- "The component described in the voiceover is present but out of frame/cropped out"
- "The camera angle makes it impossible to see the relevant part of the subject"
