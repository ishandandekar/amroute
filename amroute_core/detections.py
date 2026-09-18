"""Thread-safe state for combining siren and ambulance detections."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class Detections:
    """Record audio/vision events and identify a fresh fused pair.

    A pair is fused when its audio and vision events occurred within ``window``
    seconds of one another and the newest event has not already gone stale.
    ``snapshot`` returns the event counters as a stable identity so callers can
    avoid announcing the same pair more than once.
    """

    def __init__(self, window: float, clock: Callable[[], float] = time.monotonic):
        if window <= 0:
            raise ValueError("window must be greater than zero")
        self.window = float(window)
        self._clock = clock
        self.lock = threading.Lock()
        self.last_siren = 0.0
        self.last_ambulance = 0.0
        self.siren_count = 0
        self.ambulance_count = 0

    def siren_fire(self) -> None:
        with self.lock:
            self.last_siren = self._clock()
            self.siren_count += 1

    def ambulance_fire(self) -> None:
        with self.lock:
            self.last_ambulance = self._clock()
            self.ambulance_count += 1

    def snapshot(self) -> tuple[int, int] | None:
        """Return the identity of the current fresh fused pair, if any."""
        with self.lock:
            if self.last_siren == 0.0 or self.last_ambulance == 0.0:
                return None
            if abs(self.last_siren - self.last_ambulance) > self.window:
                return None
            if self._clock() - max(self.last_siren, self.last_ambulance) > self.window:
                return None
            return self.siren_count, self.ambulance_count

    def fused(self) -> bool:
        """Return whether a fresh audio/vision pair currently exists."""
        return self.snapshot() is not None
