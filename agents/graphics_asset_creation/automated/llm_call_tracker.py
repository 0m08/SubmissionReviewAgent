"""
llm_call_tracker.py
===================
Thread-safe, session-scoped tracker for LLM API calls made during the
automated Graphics Creation pipeline.

Usage
-----
    from agents.graphics_asset_creation.automated.llm_call_tracker import tracker

    # Option A – context manager (measures latency automatically):
    with tracker.call("gemini-3-flash-preview", role="Accuracy Reviewer"):
        response = chat.send_message(...)

    # Option B – manual record after the call:
    tracker.record(model="grok-imagine-image", role="Image Editor (Grok)", latency_s=2.31)

In the UI, call:
    tracker.render_stats_panel()

to display a live statistics dashboard.
"""
from __future__ import annotations

import copy
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CallRecord:
    model: str
    role: str
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    success: bool = True
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ModelStats:
    model: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_latency_s: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    @property
    def avg_latency_s(self) -> float:
        return self.total_latency_s / self.total_calls if self.total_calls else 0.0

    @property
    def success_rate(self) -> float:
        return (self.successful_calls / self.total_calls * 100) if self.total_calls else 0.0


class CallContext:
    """Helper for capturing usage data from an LLM response."""
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0

    def set_response(self, response):
        """Automatically extract token usage from a Google GenAI response."""
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            self.input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            self.output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
        elif hasattr(response, "usage") and response.usage:
            # Fallback for other SDKs / structures
            self.input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            self.output_tokens = getattr(response.usage, "completion_tokens", 0) or 0

    def __getitem__(self, key):
        # Backward compatibility with dict-based usage
        if key == "input": return self.input_tokens
        if key == "output": return self.output_tokens
        raise KeyError(key)

    def __setitem__(self, key, value):
        if key == "input": self.input_tokens = value
        elif key == "output": self.output_tokens = value
        else: raise KeyError(key)

    def get(self, key, default=0):
        try: return self[key]
        except KeyError: return default


# ---------------------------------------------------------------------------
# Tracker singleton
# ---------------------------------------------------------------------------

class LLMCallTracker:
    """
    Singleton tracker – safe to import at module level.  Per-run data lives in
    st.session_state so it persists across Streamlit reruns and is isolated
    per browser session.
    """

    _lock = threading.Lock()
    _SS_KEY = "llm_stats"

    # ── Internal helpers ───────────────────────────────────────────────────
    def __init__(self):
        self._internal_state = self._empty_state()

    def _get_state(self) -> dict:
        """Return the shared stats dict. Sync with session_state for UI persistence."""
        try:
            import streamlit as st
            # If we're in a Streamlit thread, ensure session_state has a reference to our shared dict
            # so it survives reruns if the module is somehow reloaded (rare but safer).
            if self._SS_KEY not in st.session_state:
                st.session_state[self._SS_KEY] = self._internal_state
            return st.session_state[self._SS_KEY]
        except Exception:
            # Running in a background thread or outside Streamlit
            return self._internal_state

    @staticmethod
    def _empty_state() -> dict:
        return {
            "records": [],          # list[CallRecord]
            "by_model": {},         # model_name → ModelStats
            "by_role": {},          # role_label → ModelStats
            "total_calls": 0,
            "total_latency_s": 0.0,
            "pipeline_start": None,
            "pipeline_end": None,
        }

    # ── Public write API ───────────────────────────────────────────────────

    def reset(self):
        """Clear all stats. Call at the start of each automation run."""
        with self._lock:
            # We must update the dictionary in-place so all references (including session_state) see the change
            state = self._get_state()
            state.clear()
            state.update(self._empty_state())
            state["pipeline_start"] = time.time()
            # Also reset our internal backup just in case they were different objects
            if state is not self._internal_state:
                self._internal_state.clear()
                self._internal_state.update(state)

    def mark_pipeline_end(self):
        """Stamp the end time of the pipeline."""
        with self._lock:
            state = self._get_state()
            state["pipeline_end"] = time.time()
            if state is not self._internal_state:
                 self._internal_state["pipeline_end"] = state["pipeline_end"]

    @contextmanager
    def call(self, model: str, role: str):
        """
        Context manager that measures latency and records the result.

        Usage::
            with tracker.call("model-name", "Role") as usage:
                resp = client.generate_content(...)
                usage.set_response(resp)
        """
        t0 = time.time()
        success = True
        error_msg = None
        ctx = CallContext()
        try:
            yield ctx
        except Exception as exc:
            success = False
            error_msg = str(exc)
            raise
        finally:
            self.record(
                model=model,
                role=role,
                latency_s=time.time() - t0,
                input_tokens=ctx.input_tokens,
                output_tokens=ctx.output_tokens,
                success=success,
                error=error_msg,
            )

    def record(
        self,
        model: str,
        role: str,
        latency_s: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        success: bool = True,
        error: Optional[str] = None,
    ):
        """Manually append a completed LLM call record."""
        rec = CallRecord(
            model=model,
            role=role,
            latency_s=latency_s,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            success=success,
            error=error,
        )
        with self._lock:
            state = self._get_state()

            if state["pipeline_start"] is None:
                state["pipeline_start"] = time.time()

            state["records"].append(rec)
            state["total_calls"] += 1
            state["total_latency_s"] += latency_s

            # Per-model aggregation
            if model not in state["by_model"]:
                state["by_model"][model] = ModelStats(model=model)
            ms: ModelStats = state["by_model"][model]
            ms.total_calls += 1
            ms.total_latency_s += latency_s
            ms.total_input_tokens += input_tokens
            ms.total_output_tokens += output_tokens
            if success:
                ms.successful_calls += 1
            else:
                ms.failed_calls += 1

            # Per-role aggregation
            if role not in state["by_role"]:
                state["by_role"][role] = ModelStats(model=role)
            rs: ModelStats = state["by_role"][role]
            rs.total_calls += 1
            rs.total_latency_s += latency_s
            rs.total_input_tokens += input_tokens
            rs.total_output_tokens += output_tokens
            if success:
                rs.successful_calls += 1
            else:
                rs.failed_calls += 1

    # ── Public read API ────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Return a deep copy of the current stats (thread-safe)."""
        with self._lock:
            return copy.deepcopy(self._get_state())

    @property
    def wall_time_s(self) -> float:
        state = self._get_state()
        start = state.get("pipeline_start")
        if start is None:
            return 0.0
        end = state.get("pipeline_end") or time.time()
        return end - start

    # ── Streamlit UI rendering ─────────────────────────────────────────────

    def render_stats_panel(self):
        """
        Render a rich live statistics panel inside Streamlit.
        Safe to call repeatedly – it reads from session_state each time.
        """
        import streamlit as st
        import pandas as pd

        state = self._get_state()
        total_calls: int = state["total_calls"]
        total_latency: float = state["total_latency_s"]
        by_model: dict[str, ModelStats] = state["by_model"]
        by_role: dict[str, ModelStats] = state["by_role"]
        records: list[CallRecord] = state["records"]
        wall = self.wall_time_s

        # ── Top-level KPI row ──────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "⏱️ Wall Time",
            f"{wall / 60:.1f} min" if wall >= 60 else f"{wall:.1f} s",
        )
        c2.metric("📞 Total LLM Calls", total_calls)
        c3.metric(
            "⚡ Avg Latency / Call",
            f"{(total_latency / total_calls):.1f} s" if total_calls else "—",
        )
        c4.metric("🤖 Distinct Models", len(by_model))

        if not by_model:
            st.info("No LLM calls recorded yet. Run the automation pipeline to see live stats.")
            return

        # ── Per-model table ────────────────────────────────────────────
        st.markdown("#### 📊 Breakdown by Model")
        model_rows = [
            {
                "Model": m,
                "Calls": ms.total_calls,
                "✅ OK": ms.successful_calls,
                "❌ Errors": ms.failed_calls,
                "Input Tokens": f"{ms.total_input_tokens:,}",
                "Output Tokens": f"{ms.total_output_tokens:,}",
                "Σ Time": f"{ms.total_latency_s:.1f} s",
                "Avg / call": f"{ms.avg_latency_s:.1f} s",
                "Success": f"{ms.success_rate:.0f}%",
            }
            for m, ms in sorted(by_model.items(), key=lambda x: -x[1].total_calls)
        ]
        st.dataframe(pd.DataFrame(model_rows), width='stretch', hide_index=True)

        # ── Per-role / agent table ─────────────────────────────────────
        st.markdown("#### 🎭 Breakdown by Agent Role")
        role_rows = [
            {
                "Agent Role": role,
                "Calls": rs.total_calls,
                "✅ OK": rs.successful_calls,
                "❌ Errors": rs.failed_calls,
                "Σ Time": f"{rs.total_latency_s:.1f} s",
                "Avg / call": f"{rs.avg_latency_s:.1f} s",
            }
            for role, rs in sorted(by_role.items(), key=lambda x: -x[1].total_calls)
        ]
        st.dataframe(pd.DataFrame(role_rows), width='stretch', hide_index=True)

        # ── Donut-like bar chart (using Streamlit's built-ins) ─────────
        if len(by_model) > 1:
            st.markdown("#### 📈 Call Volume Comparison")
            chart_data = pd.DataFrame(
                {
                    "Model / Role": list(by_model.keys()),
                    "Calls": [ms.total_calls for ms in by_model.values()],
                }
            ).set_index("Model / Role")
            st.bar_chart(chart_data)

        # ── Recent call timeline ───────────────────────────────────────
        if records:
            with st.expander(
                f"🕐 Recent Calls – last {min(50, len(records))} of {len(records)}",
                expanded=False,
            ):
                pipeline_start = state.get("pipeline_start") or records[0].timestamp
                timeline_rows = []
                for rec in reversed(records[-50:]):
                    elapsed = rec.timestamp - pipeline_start
                    status = "✅" if rec.success else f"❌ {(rec.error or '')[:60]}"
                    timeline_rows.append(
                        {
                            "T+ (s)": f"{elapsed:.1f}",
                            "Model": rec.model,
                            "Agent Role": rec.role,
                            "Latency (s)": f"{rec.latency_s:.2f}",
                            "Status": status,
                        }
                    )
                st.dataframe(
                    pd.DataFrame(timeline_rows),
                    width='stretch',
                    hide_index=True,
                )


# Module-level singleton – import this everywhere
tracker = LLMCallTracker()
