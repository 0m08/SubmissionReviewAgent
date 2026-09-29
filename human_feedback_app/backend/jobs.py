"""Background revision job queue."""

from __future__ import annotations

import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobCancelled(Exception):
    """Raised when a revision job is aborted (e.g. logout) before it can finish writing."""


@dataclass
class RevisionJob:
    id: str
    session_id: str
    label: str
    status: JobStatus = JobStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: _now())
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    slide_index: int = 0
    segment_index: int = 0
    step_index: int = 0
    kind: str = ""  # "visual" | "segmentation" | "layout"
    row_index: Optional[int] = None
    visual_id: str = ""
    scene_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "slide_index": self.slide_index,
            "segment_index": self.segment_index,
            "step_index": self.step_index,
            "kind": self.kind,
            "row_index": self.row_index,
            "visual_id": self.visual_id,
            "scene_id": self.scene_id,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RevisionJobQueue:
    def __init__(self, max_workers: int = 10) -> None:
        self._jobs: Dict[str, RevisionJob] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def register(
        self,
        session_id: str,
        label: str,
        slide_index: int,
        segment_index: int,
        step_index: int,
        *,
        kind: str = "",
        row_index: Optional[int] = None,
        visual_id: str = "",
        scene_id: str = "",
        job_id: Optional[str] = None,
    ) -> RevisionJob:
        """Register a pending job without starting it (so reconcile can see it as active)."""
        job_id = (job_id or "").strip() or uuid.uuid4().hex[:12]
        job = RevisionJob(
            id=job_id,
            session_id=session_id,
            label=label,
            slide_index=slide_index,
            segment_index=segment_index,
            step_index=step_index,
            kind=kind or "",
            row_index=row_index,
            visual_id=visual_id or "",
            scene_id=scene_id or "",
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def start(self, job_id: str, worker: Callable[[], Dict[str, Any]]) -> Optional[RevisionJob]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status == JobStatus.CANCELLED:
                return job
        self._executor.submit(self._run_job, job_id, worker)
        return job

    def submit(
        self,
        session_id: str,
        label: str,
        worker: Callable[[], Dict[str, Any]],
        slide_index: int,
        segment_index: int,
        step_index: int,
        *,
        kind: str = "",
        row_index: Optional[int] = None,
        visual_id: str = "",
        scene_id: str = "",
        job_id: Optional[str] = None,
    ) -> RevisionJob:
        job = self.register(
            session_id=session_id,
            label=label,
            slide_index=slide_index,
            segment_index=segment_index,
            step_index=step_index,
            kind=kind,
            row_index=row_index,
            visual_id=visual_id,
            scene_id=scene_id,
            job_id=job_id,
        )
        self.start(job.id, worker)
        return job

    def get(self, job_id: str, session_id: str) -> Optional[RevisionJob]:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None or job.session_id != session_id:
            return None
        return job

    def list_jobs(self, session_id: str) -> List[RevisionJob]:
        with self._lock:
            jobs = [j for j in self._jobs.values() if j.session_id == session_id]
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            return job is not None and job.status == JobStatus.CANCELLED

    def has_active_visual_job(self, row_index: int, visual_id: str) -> bool:
        visual_id = (visual_id or "").strip()
        if not visual_id:
            return False
        with self._lock:
            for job in self._jobs.values():
                if (
                    job.kind == "visual"
                    and job.row_index == row_index
                    and (job.visual_id or "").strip() == visual_id
                    and job.status in (JobStatus.PENDING, JobStatus.RUNNING)
                ):
                    return True
        return False

    def cancel_job(self, job_id: str, *, error: str = "Cancelled") -> Optional[RevisionJob]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status not in (JobStatus.PENDING, JobStatus.RUNNING):
                return job
            job.status = JobStatus.CANCELLED
            job.error = error
            job.finished_at = _now()
            return job

    def cancel_session_jobs(self, session_id: str) -> List[RevisionJob]:
        """Mark all pending/running jobs for a session as cancelled. Returns those jobs."""
        cancelled: List[RevisionJob] = []
        with self._lock:
            for job in self._jobs.values():
                if job.session_id != session_id:
                    continue
                if job.status not in (JobStatus.PENDING, JobStatus.RUNNING):
                    continue
                job.status = JobStatus.CANCELLED
                job.error = "Cancelled on logout"
                job.finished_at = _now()
                cancelled.append(job)
        return cancelled

    def _run_job(self, job_id: str, worker: Callable[[], Dict[str, Any]]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if job.status == JobStatus.CANCELLED:
                return
            job.status = JobStatus.RUNNING
            job.started_at = _now()
        try:
            result = worker()
            with self._lock:
                if job.status == JobStatus.CANCELLED:
                    # Logout won the race: discard any result and do not mark completed.
                    return
                job.status = JobStatus.COMPLETED
                job.result = result
                job.finished_at = _now()
        except JobCancelled as exc:
            with self._lock:
                job.status = JobStatus.CANCELLED
                job.error = str(exc) or "Cancelled"
                job.finished_at = _now()
        except Exception as exc:
            with self._lock:
                if job.status == JobStatus.CANCELLED:
                    return
                job.status = JobStatus.FAILED
                job.error = str(exc)
                job.finished_at = _now()
            traceback.print_exc()


revision_queue = RevisionJobQueue(max_workers=10)
