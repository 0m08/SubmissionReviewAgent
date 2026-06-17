DEFAULT_WORKSHEET = "Slide Chunks"
FINAL_GRAPHICS_COLUMN = "final_graphics_definition"
HUMAN_FEEDBACK_COLUMN = "human_feedback"
HUMAN_FEEDBACK_STATUS_COLUMN = "human_feedback_status"
HUMAN_FEEDBACK_TRACKING_COLUMN = "human_feedback_revision_tracking"
HUMAN_REVIEW_ACTIONS_COLUMN = "human_review_actions"

ACTION_NONE = "unreviewed"
ACTION_APPROVE = "approve"
ACTION_REJECT_DRIVE_HVAC = "reject_drive_hvac"
ACTION_REJECT_ALL = "reject_all"
ACTION_REJECT_AI = "reject_ai"
AI_NO_FEEDBACK_MARKER = "No Feedback"

DEFAULT_REJECT_FEEDBACK = (
    "I did not like the visual that is currently assigned. Find and assign a "
    "better visual that is relevant for this voiceover part"
)

MODE_TO_ACTION = {
    "drive": ACTION_REJECT_DRIVE_HVAC,
    "all": ACTION_REJECT_ALL,
    "ai": ACTION_REJECT_AI,
}

ACTION_TO_MODE = {v: k for k, v in MODE_TO_ACTION.items()}
