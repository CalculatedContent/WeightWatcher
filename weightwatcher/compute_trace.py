import contextlib
import datetime as _dt
import json
import os
import threading
import uuid

_state = threading.local()


def _env_enabled():
    return os.getenv("WW_COMPUTE_TRACE", "").strip().lower() in {"1", "true", "yes", "on"}


class ComputeTrace(contextlib.AbstractContextManager):
    def __init__(self, enabled=None, log_path=None):
        self.enabled = _env_enabled() if enabled is None else bool(enabled)
        self.log_path = log_path or os.getenv("WW_COMPUTE_TRACE_FILE")
        self.events = []
        self.run_id = str(uuid.uuid4())
        self._fh = None

    def __enter__(self):
        _state.active_trace = self if self.enabled else None
        if self.enabled and self.log_path:
            self._fh = open(self.log_path, "a", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb):
        if getattr(_state, "active_trace", None) is self:
            _state.active_trace = None
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        return False

    def add_event(self, event_type, **fields):
        if not self.enabled:
            return
        evt = {
            "run_id": self.run_id,
            "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            "event_type": event_type,
        }
        evt.update(fields)
        self.events.append(evt)
        if self._fh is not None:
            self._fh.write(json.dumps(evt, default=str) + "\n")
            self._fh.flush()

    def summary(self):
        return summarize_trace(self.events)


def active_trace():
    return getattr(_state, "active_trace", None)


def trace_event(event_type, **fields):
    trace = active_trace()
    if trace is not None:
        trace.add_event(event_type, **fields)


def summarize_trace(events):
    svd_end = [e for e in events if e.get("event_type") == "svd_full_end"]
    by_phase = {"randomize_model": 0, "analyze_traps": 0, "remove_traps": 0}
    by_layer = {}
    for e in svd_end:
        phase = e.get("phase")
        if phase in by_phase:
            by_phase[phase] += 1
        lid = e.get("layer_id")
        if lid is not None:
            by_layer[str(lid)] = by_layer.get(str(lid), 0) + 1
    return {
        "total_svd_calls": len(svd_end),
        "total_svd_elapsed_ms": float(sum(float(e.get("elapsed_ms", 0.0) or 0.0) for e in svd_end)),
        "svd_calls_by_phase": by_phase,
        "svd_calls_by_layer": by_layer,
        "full_bulk_metric_calls": sum(1 for e in events if e.get("event_type") == "full_bulk_metrics"),
        "fast_bulk_metric_calls": sum(1 for e in events if e.get("event_type") == "fast_bulk_metrics"),
        "original_basis_metric_calls": sum(1 for e in events if e.get("event_type") == "original_basis_metrics"),
        "collect_trap_artifacts_calls": sum(1 for e in events if e.get("event_type") == "collect_trap_artifacts_start"),
        "max_sampled_bulk_modes": max([int(e.get("sampled_bulk_modes", 0) or 0) for e in events if e.get("event_type") == "fast_bulk_metrics"] or [0]),
    }
