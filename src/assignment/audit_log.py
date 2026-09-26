"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store input + start timestamp keyed by request_id/user_id."""
        req_id = request_id or f"req-{len(self.logs) + len(self._open) + 1}"
        self._open[req_id] = {
            "request_id": req_id,
            "user_id": user_id,
            "input": text,
            "start_time": time.time(),
            "timestamp": utc_now_iso(),
        }
        return req_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store output, layer decision, latency; append to self.logs."""
        start_info = self._open.pop(request_id, None) if request_id else None
        start_time = start_info["start_time"] if start_info else time.time()
        latency_ms = (time.time() - start_time) * 1000

        entry = {
            "request_id": request_id or (start_info["request_id"] if start_info else f"req-{len(self.logs)+1}"),
            "user_id": user_id,
            "input": start_info["input"] if start_info else "",
            "response": text,
            "blocked": blocked,
            "layer": layer,
            "latency_ms": round(latency_ms, 2),
            "timestamp": utc_now_iso(),
        }
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        target = Path(filepath or default_audit_log_path())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.logs, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(target)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
