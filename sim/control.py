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

import sumolib
import traci

EV_ID = "ev"
GREEN = "G"
RESTORE_PROGRAM = "0"

# SUMO lane-change mode bits (LCA_*)
LCA_STRATEGIC = 0x1
LCA_COOPERATIVE = 0x2
LCA_SPEEDGAIN = 0x4
LCA_URGENT = 0x40


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


class LaneClearanceController:
    """Shunt blocking vehicles out of the EV's lane inside the detection zone.

    Zone scoping (a vehicle is a shunt target only when ALL hold):
      (a) it sits on an edge of the EV's fixed route (never off-route),
      (b) it is ahead of the EV along that route,
      (c) it is within R meters of route distance (R=inf = whole route ahead),
      (d) it is on the lane the EV currently drives, and
      (e) it moves slower than a cutoff (fast traffic is left alone).

    Each target is first asked for a safe (urgent) lane change into the adjacent
    same-direction lane; if a block stays in the EV's lane past a grace period,
    it is hard-moved into the adjacent lane so the queued lane physically
    empties ahead of the ambulance. The adjacent lane is recomputed per step,
    so whichever of the corridor's 2 facing lanes the EV happens to be in is the
    one that gets drained ("cars pull over" onto the lane the ambulance is not
    using). Nothing behind the EV or on the opposite carriageway is touched.
    """

    def __init__(
        self,
        r: float,
        route_edges: list[str],
        net_path,
        shunt_speed: float = 10.0,
        grace: float = 2.0,
        change_dur: float = 0.5,
        hard_window: float = 80.0,
    ) -> None:
        self.r = float(r)
        self.shunt_speed = shunt_speed
        self.grace = grace
        self.change_dur = change_dur
        self.hard_window = hard_window
        self.stop_dur = 3.0  # s a hard-shoved car stays stopped before resuming
        net = sumolib.net.readNet(str(net_path))
        lengths: dict[str, float] = {}
        nlanes: dict[str, int] = {}
        for eid in route_edges:
            edge = net.getEdge(eid)
            lengths[eid] = edge.getLength()
            nlanes[eid] = len(edge.getLanes())
        cum: dict[str, float] = {}
        acc = 0.0
        for eid in route_edges:
            cum[eid] = acc
            acc += lengths[eid]
        self.nlanes = nlanes
        self._cum = cum
        self._ev_dist: float | None = None
        self._ev_lane = 0
        self._pending: dict[str, float] = {}  # veh -> time of last request
        self._orig_mode: dict[str, int] = {}  # veh -> LC mode before urgent
        self._stopped: dict[str, float] = {}  # veh -> when hard-shoved at v=0
        # telemetry counters
        self.n_req = 0
        self.n_done = 0
        self.n_failed = 0
        self.shunted_ids: set[str] = set()  # vehicles hard-shoved this run

    def step(self) -> None:
        if EV_ID not in traci.vehicle.getIDList():
            self.reset()
            return
        now = traci.simulation.getTime()
        self._update_ev()
        if self._ev_dist is None:
            return
        for veh, ahead in self._blockers():
            self._shunt(veh, ahead, now)
        self._sweep()
        self._resume_stopped(now)

    def _update_ev(self) -> None:
        edge = traci.vehicle.getRoadID(EV_ID)
        if edge in self._cum:
            self._ev_dist = self._cum[edge] + traci.vehicle.getLanePosition(EV_ID)
            self._ev_lane = traci.vehicle.getLaneIndex(EV_ID)

    def _blockers(self) -> list[tuple[str, float]]:
        if self._ev_dist is None:
            return []
        out = []
        for veh in traci.vehicle.getIDList():
            if veh == EV_ID:
                continue
            try:
                speed = traci.vehicle.getSpeed(veh)
                edge = traci.vehicle.getRoadID(veh)
                lane = traci.vehicle.getLaneIndex(veh)
                pos = traci.vehicle.getLanePosition(veh)
            except traci.TraCIException:
                continue
            if speed >= self.shunt_speed or lane != self._ev_lane:
                continue
            if edge not in self._cum or edge.startswith(":"):
                continue
            ahead = self._cum[edge] + pos - self._ev_dist
            if 0.0 < ahead <= self.r:
                out.append((veh, ahead))
        out.sort(key=lambda t: t[1])
        return out

    def _shunt(self, veh: str, ahead: float, now: float) -> None:
        edge = traci.vehicle.getRoadID(veh)
        n_lanes = self.nlanes.get(edge, 2)
        if n_lanes < 2:
            self.n_failed += 1
            return
        target = 1 - self._ev_lane
        if target < 0 or target >= n_lanes or target == self._ev_lane:
            # EV on an unusually-laned internal edge: do not shove inside a
            # junction box (would feed cross-traffic into it); skip instead.
            self.n_failed += 1
            return
        if traci.vehicle.getLaneIndex(veh) != self._ev_lane:
            return
        last = self._pending.get(veh)
        if last is not None and now - last < self.grace:
            return  # a change is in flight; don't spray requests
        if last is None:
            try:
                orig = traci.vehicle.getLaneChangeMode(veh)
            except traci.TraCIException:
                orig = LCA_STRATEGIC | LCA_COOPERATIVE | LCA_SPEEDGAIN
            self._orig_mode[veh] = orig
            self._pending[veh] = now
            try:
                traci.vehicle.setLaneChangeMode(
                    veh, orig | LCA_URGENT | LCA_COOPERATIVE
                )
                traci.vehicle.changeLane(veh, target, self.change_dur)
            except traci.TraCIException:
                pass
            self.n_req += 1
            return
        if ahead > self.hard_window:
            # beyond the hard-shove window: keep asking for the safe change
            self._pending[veh] = now
            return
        # grace elapsed: hard-shove into a free slot of the adjacent lane so the
        # EV's lane physically empties; never into an occupied spot (no ramming)
        try:
            edge = traci.vehicle.getRoadID(veh)
            tgt_lane = f"{edge}_{target}"
            lane_len = traci.lane.getLength(tgt_lane)
            pos = traci.vehicle.getLanePosition(veh)
            spot = self._free_spot(
                tgt_lane,
                pos,
                traci.vehicle.getLength(veh),
                max_scan=min(60.0, lane_len),
            )
            if spot is None:
                self.n_failed += 1
            else:
                traci.vehicle.moveTo(veh, tgt_lane, spot)
                traci.vehicle.setSpeed(veh, 0.0)  # pulled over, not rolling
                self._stopped[veh] = now
                self.shunted_ids.add(veh)
                self.n_done += 1
        except traci.TraCIException:
            self.n_failed += 1
        self._restore(veh)

    def _resume_stopped(self, now: float) -> None:
        for veh, t0 in list(self._stopped.items()):
            if veh not in traci.vehicle.getIDList():
                self._stopped.pop(veh, None)
                continue
            if now - t0 >= self.stop_dur:
                try:
                    traci.vehicle.setSpeed(veh, -1.0)  # give control back
                except traci.TraCIException:
                    pass
                self._stopped.pop(veh, None)

    @staticmethod
    def _free_spot(
        lane_id: str, pos: float, veh_len: float, max_scan: float = 60.0
    ) -> float | None:
        """Nearest free slot (>= current pos) on a lane, or None past max_scan."""
        occ = []
        for vid in traci.lane.getLastStepVehicleIDs(lane_id):
            if vid not in traci.vehicle.getIDList():
                continue
            try:
                p = traci.vehicle.getLanePosition(vid)
                l = traci.vehicle.getLength(vid)
            except traci.TraCIException:
                continue
            occ.append((p - 3.0, p + l + 3.0))
        occ.sort()
        cand = max(0.0, pos)
        for lo, hi in occ:
            if hi < cand:
                continue
            if lo - cand >= veh_len + 3.0:
                return cand
            cand = max(cand, hi)
        return cand if max_scan - cand >= veh_len + 3.0 else None

    def _sweep(self) -> None:
        # drop bookkeeping for targets that already left the EV's lane or sim
        for veh in list(self._pending):
            try:
                on_lane = (
                    veh in traci.vehicle.getIDList()
                    and traci.vehicle.getLaneIndex(veh) == self._ev_lane
                )
            except traci.TraCIException:
                on_lane = False
            if not on_lane:
                self._restore(veh)

    def _restore(self, veh: str) -> None:
        mode = self._orig_mode.pop(veh, None)
        if mode is not None:
            try:
                traci.vehicle.setLaneChangeMode(veh, mode)
            except traci.TraCIException:
                pass
        self._pending.pop(veh, None)

    def reset(self) -> None:
        for veh in list(self._orig_mode):
            self._restore(veh)
        for veh in list(self._stopped):
            try:
                traci.vehicle.setSpeed(veh, -1.0)
            except traci.TraCIException:
                pass
        self._stopped.clear()
        self._ev_dist = None
