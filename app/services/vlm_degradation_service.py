from __future__ import annotations

from dataclasses import dataclass, field
from threading import BoundedSemaphore, Lock
import time
from typing import Any

from app.config import settings


DEGRADATION_STATE_VERSION = "vlm_degradation_state_v1"


@dataclass
class BackendHealthState:
    backend_key: str
    recent_failures: list[dict[str, Any]] = field(default_factory=list)
    circuit_open_until: float = 0.0
    last_success_at: float | None = None


class VlmDegradationState:
    def __init__(self) -> None:
        self._states: dict[str, BackendHealthState] = {}
        self._lock = Lock()

    def backend_health(self, backend_key: str) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            state = self._state_for(backend_key)
            self._prune_locked(state, now=now)
            return self._serialize_state_locked(state, now=now)

    def is_backend_available(self, backend_key: str) -> tuple[bool, str | None, dict[str, Any]]:
        now = time.time()
        with self._lock:
            state = self._state_for(backend_key)
            self._prune_locked(state, now=now)
            if state.circuit_open_until > now:
                return (
                    False,
                    "vlm_backend_circuit_open",
                    self._serialize_state_locked(state, now=now),
                )
            return True, None, self._serialize_state_locked(state, now=now)

    def record_success(self, backend_key: str) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            state = self._state_for(backend_key)
            state.last_success_at = now
            state.recent_failures = []
            state.circuit_open_until = 0.0
            return self._serialize_state_locked(state, now=now)

    def record_failure(self, backend_key: str, *, reason: str) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            state = self._state_for(backend_key)
            self._prune_locked(state, now=now)
            state.recent_failures.append({"at": now, "reason": reason})
            threshold = max(1, int(settings.vlm_recent_failure_threshold or 1))
            if len(state.recent_failures) >= threshold:
                state.circuit_open_until = now + max(
                    1,
                    int(settings.vlm_circuit_breaker_cooldown_seconds or 1),
                )
            return self._serialize_state_locked(state, now=now)

    def reset(self) -> None:
        with self._lock:
            self._states = {}

    def _state_for(self, backend_key: str) -> BackendHealthState:
        state = self._states.get(backend_key)
        if state is None:
            state = BackendHealthState(backend_key=backend_key)
            self._states[backend_key] = state
        return state

    def _prune_locked(self, state: BackendHealthState, *, now: float) -> None:
        window_seconds = max(1, int(settings.vlm_circuit_breaker_window_seconds or 1))
        state.recent_failures = [
            failure
            for failure in state.recent_failures
            if now - float(failure.get("at", 0.0)) <= window_seconds
        ]
        if state.circuit_open_until <= now:
            state.circuit_open_until = 0.0

    def _serialize_state_locked(self, state: BackendHealthState, *, now: float) -> dict[str, Any]:
        return {
            "state_version": DEGRADATION_STATE_VERSION,
            "backend_key": state.backend_key,
            "recent_failure_count": len(state.recent_failures),
            "recent_failure_reasons": [
                str(failure.get("reason", "unknown")) for failure in state.recent_failures[-5:]
            ],
            "circuit_open": state.circuit_open_until > now,
            "circuit_open_remaining_seconds": max(0, round(state.circuit_open_until - now, 3)),
            "last_success_age_seconds": (
                None if state.last_success_at is None else max(0, round(now - state.last_success_at, 3))
            ),
        }


class VlmConcurrencyGate:
    def __init__(self) -> None:
        self._limit = 0
        self._semaphore: BoundedSemaphore | None = None
        self._lock = Lock()

    def acquire(
        self,
        *,
        timeout_seconds: float = 0.0,
        max_concurrent_inferences: int | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        limit = int(
            settings.vlm_max_concurrent_inferences
            if max_concurrent_inferences is None
            else max_concurrent_inferences
        )
        if limit <= 0:
            return False, self._trace(limit=limit, acquired=False, reason="vlm_concurrency_disabled")

        semaphore = self._semaphore_for_limit(limit)
        acquired = semaphore.acquire(timeout=max(0.0, float(timeout_seconds or 0.0)))
        reason = None if acquired else "vlm_concurrency_budget_exhausted"
        return acquired, self._trace(limit=limit, acquired=acquired, reason=reason)

    def release(self) -> None:
        semaphore = self._semaphore
        if semaphore is None:
            return
        try:
            semaphore.release()
        except ValueError:
            pass

    def _semaphore_for_limit(self, limit: int) -> BoundedSemaphore:
        with self._lock:
            if self._semaphore is None or self._limit != limit:
                self._limit = limit
                self._semaphore = BoundedSemaphore(limit)
            return self._semaphore

    def _trace(self, *, limit: int, acquired: bool, reason: str | None) -> dict[str, Any]:
        return {
            "max_concurrent_inferences": limit,
            "acquire_timeout_seconds": settings.vlm_concurrency_acquire_timeout_seconds,
            "slot_acquired": acquired,
            "reason": reason,
            "serialization_guard_enabled": settings.vlm_serialization_guard_enabled,
        }


vlm_degradation_state = VlmDegradationState()
vlm_concurrency_gate = VlmConcurrencyGate()
