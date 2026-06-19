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


@dataclass
class RevisionJob:
    id: str
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
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RevisionJobQueue:
    def __init__(self, max_workers: int = 10) -> None:
        self._jobs: Dict[str, RevisionJob] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit(
        self,
        label: str,
        worker: Callable[[], Dict[str, Any]],
        slide_index: int,
        segment_index: int,
        step_index: int,
    ) -> RevisionJob:
        job_id = uuid.uuid4().hex[:12]
        job = RevisionJob(
            id=job_id,
            label=label,
            slide_index=slide_index,
            segment_index=segment_index,
            step_index=step_index,
        )
        with self._lock:
            self._jobs[job_id] = job
        self._executor.submit(self._run_job, job_id, worker)
        return job

    def get(self, job_id: str) -> Optional[RevisionJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> List[RevisionJob]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs

    def _run_job(self, job_id: str, worker: Callable[[], Dict[str, Any]]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = JobStatus.RUNNING
            job.started_at = _now()
        try:
            result = worker()
            with self._lock:
                job.status = JobStatus.COMPLETED
                job.result = result
                job.finished_at = _now()
        except Exception as exc:
            with self._lock:
                job.status = JobStatus.FAILED
                job.error = str(exc)
                job.finished_at = _now()
            traceback.print_exc()


revision_queue = RevisionJobQueue(max_workers=10)
