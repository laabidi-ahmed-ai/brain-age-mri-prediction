"""Small shared helpers (case ids + deterministic demo signal/timeline).

Ported from the previous Django `common.base` so the mock/demo payloads keep
producing the same shapes the frontend visualizers expect.
"""
from __future__ import annotations

import secrets
from typing import Callable


def new_case_id() -> str:
    """Short human-readable case id, e.g. ``BRN-1A2B3C``."""
    return "BRN-" + secrets.token_hex(3).upper()


def _rand(seed: int) -> Callable[[], float]:
    """Tiny LCG so demo signals are reproducible without numpy."""
    state = [seed & 0xFFFFFFFF or 1]

    def _next() -> float:
        state[0] = (1103515245 * state[0] + 12345) & 0x7FFFFFFF
        return state[0] / 0x7FFFFFFF

    return _next


def make_signal(n: int = 120, seed: int = 7) -> list[dict]:
    """Deterministic pseudo-random waveform for the frontend signal chart."""
    rnd = _rand(seed)
    out = []
    v = 0.0
    for t in range(n):
        v = 0.85 * v + (rnd() - 0.5) * 0.6
        out.append({"t": t, "v": round(v, 4)})
    return out


def make_timeline(events: list[tuple[int, str, str]]) -> list[dict]:
    return [{"t": t, "label": label, "severity": sev} for t, label, sev in events]
