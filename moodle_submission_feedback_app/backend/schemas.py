# -*- coding: utf-8 -*-
"""Pydantic schemas for Moodle Submission Review API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChecklistItem(BaseModel):
    passed: bool
    instruction: str
    comment: str = ""


class SubmissionItem(BaseModel):
    name: str
    status: str
    grade: str
    agent_grade: str
    online_text: str = ""
    media_folder: str = ""
    attempt_number: int = 1
    last_modified: str = ""
    feedback_comment: str = ""
    agent_checklist: str = ""
    checklist_items: List[ChecklistItem] = []
    review_status: str = "Pending Review"
    mentor_reviewer: str = ""
    mentor_reviewed_at: str = ""


class ActivitySummary(BaseModel):
    activity_name: str
    total_submissions: int
    pending_count: int
    approved_count: int
    overridden_count: int
    pass_count: int
    fail_count: int
    unsure_count: int
    pass_rate: float


class UpdateReviewRequest(BaseModel):
    student_name: str = Field(..., min_length=1)
    attempt_number: int = Field(default=1)
    grade: str = Field(..., min_length=1)
    feedback_comment: str = Field(default="")
    review_status: str = Field(default="Approved by Mentor")
    mentor_name: str = Field(default="Mentor")
    edge_case: Optional[str] = None
    guideline: Optional[str] = None


class ActivityRuleRequest(BaseModel):
    activity_name: Optional[str] = None
    edge_case: Optional[str] = None
    guideline: Optional[str] = None
    mentor_name: Optional[str] = None


class BatchApproveRequest(BaseModel):
    mentor_name: str = Field(default="Mentor")


class SyncLocalRequest(BaseModel):
    assignment_folder: Optional[str] = None
    activity_name: str = Field(default="System Identification")
