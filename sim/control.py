"""Preemption trigger (knob R) for the green-corridor study (ticket #4).

A signal on the EV's route preempts to green only when the EV is within R meters
of it. R=inf is the full pre-knowledge ceiling: each signal is held green from the
moment it becomes the next one, yielding an unbroken green corridor. R=0 is the
no-intervention baseline.

Primitive used: `traci.trafficlight.setLinkState` sets the tls program to "online"
and the state is maintained until `setProgram()` restores the static plan.
"""

from __future__ import annotations

import math

import traci

EV_ID = "ev"
GREEN = "G"
RESTORE_PROGRAM = "0"


def format_r(r: float) -> str:
    """Serialize an R knob for file names / summaries ('inf', '200', '12.5')."""
    if math.isinf(r):
        return "inf"
    if r == int(r):
        return str(int(r))
    return f"{r:g}"


class PreemptionController:
    """Hold the EV's next signal green once the EV is within R meters of it.

    Each step reads the EV's nearest upcoming traffic light via `getNextTLS`
    (tls_id, link_index, distance, state). When distance <= R the signal's EV-approach
    link is forced green and held; once the EV passes it (the nearest tls changes)
    the static program is restored.
    """

    def __init__(self, r: float) -> None:
        self.r = float(r)
        self._held: dict[str, int] = {}

    def step(self) -> None:
        if EV_ID not in traci.vehicle.getIDList():
            return
        try:
            nxt = traci.vehicle.getNextTLS(EV_ID)
        except traci.TraCIException:
            nxt = ()
        if not nxt:
            return
        tls_id, link_idx, dist, _state = nxt[0]
        link_idx = int(link_idx)
        dist = float(dist)

        for held_id in [t for t in self._held if t != tls_id]:
            self.release(held_id)

        if dist <= self.r and self._held.get(tls_id) != link_idx:
            traci.trafficlight.setLinkState(tls_id, link_idx, GREEN)
            self._held[tls_id] = link_idx

    def release(self, tls_id: str) -> None:
        try:
            traci.trafficlight.setProgram(tls_id, RESTORE_PROGRAM)
        except traci.TraCIException:
            pass
        self._held.pop(tls_id, None)

    def reset(self) -> None:
        for tls_id in list(self._held):
            self.release(tls_id)