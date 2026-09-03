from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class ChecklistResultItem(BaseModel):
    item_id: str = Field(description="Unique identifier or index of the checklist item/rule.")
    instruction_or_condition: str = Field(description="The original instruction or checklist rule text being evaluated.")
    followed: bool = Field(description="True if the user followed this instruction / checklist requirement, False if missed or failed.")
    is_fail_if_condition: bool = Field(default=False, description="True if this rule represents an explicit 'fail if' condition.")
    comment: str = Field(description="Short informal human-like comment explaining why this item passed or failed based on evidence.")


class SubmissionReviewOutput(BaseModel):
    checklist_evaluations: List[ChecklistResultItem] = Field(
        description="Detailed list of evaluations for every checklist item and activity instruction."
    )
    agent_comment: str = Field(
        description="Short, informal, natural language summary explaining the overall decision and what fell short or passed."
    )
    agent_grade: Literal["Pass", "Fail", "Pass (Unsure)", "Fail (Unsure)"] = Field(
        description="Overall grade: Pass if core goals are met, Fail if major requirements were missed or a fail-if condition triggered, Pass (Unsure) if effort is visible but evidence is ambiguous/cropped, Fail (Unsure) if major requirements appear missed but evidence is blurry/partially unreadable."
    )
