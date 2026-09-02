"""
Scene-level parallelism helpers for hero / multi-visual overlay steps.

Two-level parallelism:
  - Outer pool: up to max_row_workers rows at once (default 50)
  - Inner pool: all scenes within a row in parallel
  - Global API semaphore: caps concurrent scene workers doing API work (default 30)
"""

from __future__ import annotations

import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    from streamlit.errors import NoSessionContext
except ImportError:
    NoSessionContext = None

try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
except ImportError:
    add_script_run_ctx = None
    get_script_run_ctx = None

from services.smart_progress_bar import SmartProgressBar

# Marker for scenes that still need an async API call.
PENDING = object()

# Max concurrent scene workers across all rows (Gemini / Drive throttle).
GLOBAL_API_SEMAPHORE_LIMIT = 30

SceneSpec = Tuple[int, str, Any]  # (sort_key, scene_id, preset_text | PENDING)


class OverlayStepProgress:
    """
    Thread-safe progress for nested scene parallelism in overlay steps.

    Worker threads only increment an internal counter via update(). The Streamlit
    widget is refreshed from the main thread via refresh_ui() so SmartProgressBar
    stays unchanged for all other agents.
    """

    def __init__(self, description="Processing", save_interval=5):
        self._bar = SmartProgressBar(total_tasks=1, description=description, save_interval=save_interval)
        self._save_interval = save_interval
        self._done = 0
        self._total = 0
        self._lock = threading.Lock()

    def add_tasks(self, n):
        if n and n > 0:
            with self._lock:
                self._total += n

    def update(self, increment=1):
        """Record completed work (safe from worker threads; no Streamlit calls)."""
        with self._lock:
            self._done += increment

    def refresh_ui(self):
        """Redraw the Streamlit bar on the main script thread."""
        with self._lock:
            done = self._done
            total = self._total if self._total > 0 else 1
        fraction = min(1.0, done / total)
        try:
            self._bar.update_progress(fraction)
        except Exception as exc:
            if NoSessionContext is not None and isinstance(exc, NoSessionContext):
                return
            if type(exc).__name__ == "NoSessionContext":
                return
            raise

    def should_save(self):
        with self._lock:
            done = self._done
        if self._save_interval <= 0 or done <= 0:
            return False
        return done % self._save_interval == 0

    def finish(self):
        try:
            self._bar.progress_bar.empty()
        except Exception:
            pass


class SceneRowAggregator:
    """Collect per-scene results for one sheet column on one row."""

    def __init__(self):
        self._lock = threading.Lock()
        self._rows: Dict[Any, dict] = {}

    def register_row(self, row_index, ordered_specs: Sequence[SceneSpec]) -> None:
        results = {}
        pending = 0
        for sort_key, scene_id, preset in ordered_specs:
            if preset is not PENDING:
                results[sort_key] = (scene_id, preset)
            else:
                pending += 1
        with self._lock:
            self._rows[row_index] = {
                "ordered_specs": list(ordered_specs),
                "results": results,
                "pending": pending,
                "flushed": False,
            }

    def complete_scene(self, row_index, sort_key, scene_id, text) -> None:
        with self._lock:
            row = self._rows[row_index]
            row["results"][sort_key] = (scene_id, text)
            row["pending"] -= 1

    def build_row_text(
        self,
        row_index,
        scene_block_fn: Callable[[str, str], str],
        empty_fallback: str = "-",
    ) -> str:
        row = self._rows[row_index]
        parts = []
        for sort_key, scene_id, _preset in row["ordered_specs"]:
            sid, body = row["results"][sort_key]
            parts.append(scene_block_fn(sid, body))
        merged = "\n\n".join(parts).strip()
        return merged if merged else empty_fallback


def build_cached_row_text(
    ordered_specs: Sequence[SceneSpec],
    scene_block_fn: Callable[[str, str], str],
    empty_fallback: str = "-",
) -> str:
    """Merge preset-only scene specs without running workers."""
    parts = []
    for _sort_key, scene_id, preset in ordered_specs:
        if preset is not PENDING:
            parts.append(scene_block_fn(scene_id, preset))
    merged = "\n\n".join(parts).strip()
    return merged if merged else empty_fallback


def _attach_streamlit_context(streamlit_ctx) -> None:
    """Copy the main Streamlit script context onto the current worker thread."""
    if streamlit_ctx is not None and add_script_run_ctx is not None:
        add_script_run_ctx(threading.current_thread(), streamlit_ctx)


def _semaphore_wrapped_worker(
    worker_fn: Callable[[dict], Tuple[Any, int, str, str]],
    streamlit_ctx=None,
) -> Callable[[dict], Tuple[Any, int, str, str]]:
    """Acquire the row's shared API semaphore for the duration of one scene worker."""

    def run(task: dict) -> Tuple[Any, int, str, str]:
        _attach_streamlit_context(streamlit_ctx)
        sem = task.get("api_semaphore")
        if sem is None:
            return worker_fn(task)
        sem.acquire()
        try:
            return worker_fn(task)
        finally:
            sem.release()

    return run


def _refresh_progress_ui(progress) -> None:
    """Update the Streamlit progress widget from the main thread."""
    if progress is None:
        return
    refresh = getattr(progress, "refresh_ui", None)
    if callable(refresh):
        refresh()


def run_row_scene_tasks(
    row_index,
    ordered_specs: Sequence[SceneSpec],
    tasks: Sequence[dict],
    worker_fn: Callable[[dict], Tuple[Any, int, str, str]],
    scene_block_fn: Callable[[str, str], str],
    api_semaphore: threading.Semaphore,
    empty_fallback: str = "-",
    on_scene_done: Optional[Callable[[], None]] = None,
    streamlit_ctx=None,
) -> str:
    """
    Run all async scene tasks for one row (inner pool), then return merged cell text.

    :param row_index: DataFrame row index.
    :param ordered_specs: Full ordered scene list including cached presets.
    :param tasks: Async scene task dicts (PENDING scenes only).
    :param worker_fn: Scene worker returning (row_index, sort_key, scene_id, text).
    :param scene_block_fn: Builds ---Scene ID--- blocks.
    :param api_semaphore: Shared global semaphore for API throttling.
    :param empty_fallback: Cell value when no blocks.
    :param on_scene_done: Optional callback after each scene completes (e.g. progress).
    :return: Merged row cell text.
    """
    aggregator = SceneRowAggregator()
    aggregator.register_row(row_index, ordered_specs)

    if not tasks:
        return aggregator.build_row_text(row_index, scene_block_fn, empty_fallback)

    wrapped = _semaphore_wrapped_worker(worker_fn, streamlit_ctx=streamlit_ctx)
    for task in tasks:
        task["api_semaphore"] = api_semaphore

    with ThreadPoolExecutor(max_workers=len(tasks)) as scene_pool:
        futures = {scene_pool.submit(wrapped, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            sort_key = task["sort_key"]
            scene_id = task["scene_id"]
            try:
                _row_index, sort_key, scene_id, text = future.result()
            except Exception as exc:
                traceback.print_exc()
                text = f"ERROR: {str(exc)}"
            aggregator.complete_scene(row_index, sort_key, scene_id, text)
            if on_scene_done:
                on_scene_done()

    return aggregator.build_row_text(row_index, scene_block_fn, empty_fallback)


def execute_nested_row_scene_batch(
    row_jobs: Sequence[dict],
    worker_by_phase: Dict[str, Callable[[dict], Tuple[Any, int, str, str]]],
    scene_block_fn: Callable[[str, str], str],
    on_row_complete: Callable[[str, Any, str], None],
    max_row_workers: int = 50,
    api_semaphore_limit: int = GLOBAL_API_SEMAPHORE_LIMIT,
    progress=None,
    log_label: str = "",
    log_every: int = 5,
    on_checkpoint: Optional[Callable[[], None]] = None,
    empty_fallback: str = "-",
    sheet_lock: Optional[threading.Lock] = None,
) -> None:
    """
    Two-level parallel batch: rows in outer pool, scenes per row in inner pool.

    Each row_job dict must include:
      - phase: str
      - row_index: Any
      - ordered_specs: list of (sort_key, scene_id, preset|PENDING)
      - tasks: list of async scene task dicts

    Optional keys on row_job are copied into each task (e.g. drive).

    :param row_jobs: One job per (phase, row) that needs scene processing.
    :param worker_by_phase: Maps phase name -> scene worker function.
    :param scene_block_fn: Builds ---Scene ID--- blocks.
    :param on_row_complete: (phase, row_index, merged_text) -> write one cell.
    :param max_row_workers: Outer pool size (rows in parallel).
    :param api_semaphore_limit: Max concurrent scene API workers globally.
    :param progress: Optional object with update() and should_save().
    :param log_label: Short label for terminal progress logs.
    :param log_every: Print every N completed scenes (0 disables).
    :param on_checkpoint: Called when progress.should_save() is true.
    :param empty_fallback: Cell value when a row has no scene blocks.
    :param sheet_lock: Optional lock serializing on_row_complete / on_checkpoint (df + sheet).
    """
    if not row_jobs:
        return

    streamlit_ctx = get_script_run_ctx() if get_script_run_ctx else None

    def _safe_row_complete(phase, row_index, text):
        if sheet_lock:
            with sheet_lock:
                on_row_complete(phase, row_index, text)
        else:
            on_row_complete(phase, row_index, text)

    def _safe_checkpoint():
        if not on_checkpoint:
            return
        if sheet_lock:
            with sheet_lock:
                on_checkpoint()
        else:
            on_checkpoint()

    api_semaphore = threading.Semaphore(api_semaphore_limit)
    total_scenes = sum(len(job.get("tasks") or []) for job in row_jobs)
    done_scenes = 0
    progress_lock = threading.Lock()

    if log_label:
        print(
            f"{log_label}: {len(row_jobs)} row job(s), {total_scenes} scene task(s), "
            f"row_workers={max_row_workers}, api_semaphore={api_semaphore_limit}"
        )

    task_field_keys = ("drive", "llm", "model")

    def _on_scene_done():
        nonlocal done_scenes
        with progress_lock:
            done_scenes += 1
            if progress is not None:
                progress.update()
            if log_label and log_every and done_scenes % log_every == 0:
                print(f"{log_label}: {done_scenes}/{total_scenes} scene(s) done")
            should_checkpoint = (
                on_checkpoint
                and progress is not None
                and progress.should_save()
            )
        if should_checkpoint:
            print(f"{log_label}: saving sheet progress ({done_scenes}/{total_scenes} scenes)")
            _safe_checkpoint()

    def _process_row(job: dict):
        _attach_streamlit_context(streamlit_ctx)
        phase = job["phase"]
        worker = worker_by_phase[phase]
        tasks = list(job.get("tasks") or [])
        for task in tasks:
            for key in task_field_keys:
                if key in job and key not in task:
                    task[key] = job[key]
        text = run_row_scene_tasks(
            row_index=job["row_index"],
            ordered_specs=job["ordered_specs"],
            tasks=tasks,
            worker_fn=worker,
            scene_block_fn=scene_block_fn,
            api_semaphore=api_semaphore,
            empty_fallback=empty_fallback,
            on_scene_done=_on_scene_done if tasks else None,
            streamlit_ctx=streamlit_ctx,
        )
        return phase, job["row_index"], text

    with ThreadPoolExecutor(max_workers=max_row_workers) as row_pool:
        futures = {row_pool.submit(_process_row, job): job for job in row_jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                phase, row_index, text = future.result()
            except Exception as exc:
                traceback.print_exc()
                phase = job["phase"]
                row_index = job["row_index"]
                text = f"ERROR: {str(exc)}"
            _safe_row_complete(phase, row_index, text)
            _refresh_progress_ui(progress)

    _refresh_progress_ui(progress)

    if log_label and total_scenes:
        print(f"{log_label}: all {total_scenes} scene task(s) complete")
