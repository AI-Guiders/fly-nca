"""Nervous Impulse Trace — substrate-wide synaptic wave observability for Fly-NCA (parity with CASA-ADR-0080).

Records the exact biological wave across a 4.038 ms cognitive frame:
- which sensory receptors fired (`activate`),
- which KCs / LHON branches / MBON compartments were inhibited (`suppress`, e.g. APL 95.4% GABA or GluCla cLTM shunt),
- what competing EPSP/IPSP potentials accumulated (`score`),
- which biological thresholds were crossed (`threshold`, e.g. LC4->GF escape or dFB sleep switch),
- which topological wedges / valence gates won (`select`),
- and how the wave routed into the 3 VNC autopilots (`route`).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
import json
import os
from pathlib import Path
import time
from typing import Any, Iterator
import uuid

from fly_nca.nervous.channels import BiologicalPort, ImpulseChannel, ImpulseOutcome

_SCHEMA = "fly_nca.impulse.v1"
_RING_MAX = int(os.environ.get("FLY_IMPULSE_RING", "1000") or "1000")


class DebugLevel(StrEnum):
    OFF = "off"
    SUMMARY = "summary"
    FULL = "full"


def resolve_debug_level(explicit: str | None = None) -> DebugLevel:
    raw = (explicit or os.environ.get("FLY_IMPULSE_TRACE") or "off").strip().lower()
    try:
        return DebugLevel(raw)
    except ValueError:
        return DebugLevel.OFF


def activate_impulse_trace(
    level: DebugLevel | str = DebugLevel.SUMMARY,
    *,
    only_if_unset: bool = True,
) -> DebugLevel:
    """Enable nervous impulse trace globally or for replay/verification."""
    if only_if_unset and os.environ.get("FLY_IMPULSE_TRACE"):
        resolved = resolve_debug_level()
    elif isinstance(level, DebugLevel):
        resolved = level
        os.environ["FLY_IMPULSE_TRACE"] = resolved.value
    else:
        try:
            resolved = DebugLevel(str(level).strip().lower())
        except ValueError:
            resolved = DebugLevel.SUMMARY
        os.environ["FLY_IMPULSE_TRACE"] = resolved.value
    _GLOBAL.level = resolved
    return resolved


@dataclass
class SynapticEvent:
    """Single synaptic/compartment event inside an impulse wave."""

    seq: int
    node: str
    kind: str  # "activate" | "score" | "suppress" | "threshold" | "select" | "route"
    organ: str = ""
    port: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    potentials: list[dict[str, Any]] = field(default_factory=list)
    selected: list[str] = field(default_factory=list)
    rejected_above_floor: int = 0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "seq": self.seq,
            "node": self.node,
            "kind": self.kind,
        }
        if self.organ:
            out["organ"] = self.organ
        if self.port:
            out["port"] = self.port
        if self.inputs:
            out["inputs"] = self.inputs
        if self.potentials:
            out["potentials"] = self.potentials
        if self.selected:
            out["selected"] = self.selected
        if self.rejected_above_floor:
            out["rejected_above_floor"] = self.rejected_above_floor
        if self.reason:
            out["reason"] = self.reason
        return out


@dataclass
class ImpulseRecord:
    header: dict[str, Any]
    events: list[SynapticEvent] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, debug_level: DebugLevel) -> dict[str, Any]:
        payload = dict(self.header)
        payload["summary"] = dict(self.summary)
        if debug_level == DebugLevel.FULL:
            payload["events"] = [event.to_dict() for event in self.events]
        elif debug_level == DebugLevel.SUMMARY:
            payload["events"] = [
                event.to_dict()
                for event in self.events
                if event.kind in {"select", "threshold", "suppress", "route"}
            ]
        return payload


class ImpulseChannelWriter:
    __slots__ = ("_recorder", "_record", "_finished", "_organs_touched")

    def __init__(self, recorder: "ImpulseRecorder", record: ImpulseRecord) -> None:
        self._recorder = recorder
        self._record = record
        self._finished = False
        self._organs_touched: list[str] = []

    @property
    def impulse_id(self) -> str:
        return str(self._record.header.get("impulse_id", ""))

    def _append(
        self,
        *,
        node: str,
        kind: str,
        organ: str = "",
        port: BiologicalPort | str = "",
        inputs: dict[str, Any] | None = None,
        potentials: list[dict[str, Any]] | None = None,
        selected: list[str] | None = None,
        rejected_above_floor: int = 0,
        reason: str = "",
    ) -> None:
        if self._finished or self._recorder.level == DebugLevel.OFF:
            return
        if organ and organ not in self._organs_touched:
            self._organs_touched.append(organ)
        seq = len(self._record.events) + 1
        self._record.events.append(
            SynapticEvent(
                seq=seq,
                node=node,
                kind=kind,
                organ=organ,
                port=str(port.value if isinstance(port, BiologicalPort) else port),
                inputs=dict(inputs or {}),
                potentials=list(potentials or []),
                selected=list(selected or []),
                rejected_above_floor=rejected_above_floor,
                reason=reason,
            )
        )

    def activate(
        self,
        node: str,
        *,
        organ: str = "",
        port: BiologicalPort | str = BiologicalPort.PORT1_ACH,
        inputs: dict[str, Any] | None = None,
    ) -> None:
        self._append(node=node, kind="activate", organ=organ, port=port, inputs=inputs)

    def score(
        self,
        node: str,
        *,
        organ: str = "",
        port: BiologicalPort | str = BiologicalPort.PORT1_ACH,
        potentials: list[dict[str, Any]] | None = None,
        inputs: dict[str, Any] | None = None,
    ) -> None:
        self._append(
            node=node,
            kind="score",
            organ=organ,
            port=port,
            potentials=potentials,
            inputs=inputs,
        )

    def suppress(
        self,
        node: str,
        *,
        organ: str = "",
        port: BiologicalPort | str = BiologicalPort.PORT2_GLUCLA_GABA,
        reason: str = "",
        rejected_above_floor: int = 0,
        inputs: dict[str, Any] | None = None,
    ) -> None:
        self._append(
            node=node,
            kind="suppress",
            organ=organ,
            port=port,
            reason=reason,
            rejected_above_floor=rejected_above_floor,
            inputs=inputs,
        )

    def threshold(
        self,
        node: str,
        *,
        organ: str = "",
        port: BiologicalPort | str = BiologicalPort.PORT4_SHAKB_GAP,
        reason: str = "",
        inputs: dict[str, Any] | None = None,
    ) -> None:
        self._append(node=node, kind="threshold", organ=organ, port=port, reason=reason, inputs=inputs)

    def select(
        self,
        node: str,
        *,
        organ: str = "",
        port: BiologicalPort | str = BiologicalPort.PORT1_ACH,
        selected: list[str] | None = None,
        potentials: list[dict[str, Any]] | None = None,
        inputs: dict[str, Any] | None = None,
        rejected_above_floor: int = 0,
    ) -> None:
        self._append(
            node=node,
            kind="select",
            organ=organ,
            port=port,
            selected=selected,
            potentials=potentials,
            inputs=inputs,
            rejected_above_floor=rejected_above_floor,
        )

    def route(
        self,
        node: str,
        *,
        organ: str = "VNC",
        port: BiologicalPort | str = BiologicalPort.PORT1_ACH,
        inputs: dict[str, Any] | None = None,
        selected: list[str] | None = None,
        potentials: list[dict[str, Any]] | None = None,
    ) -> None:
        self._append(
            node=node,
            kind="route",
            organ=organ,
            port=port,
            inputs=inputs,
            selected=selected,
            potentials=potentials,
        )

    def finish(
        self,
        outcome: ImpulseOutcome | str,
        *,
        organs_touched: list[str] | None = None,
    ) -> dict[str, Any]:
        if self._finished:
            return self._record.summary
        self._finished = True
        if self._recorder.level == DebugLevel.OFF:
            return self._record.summary
        finished_ms = int(time.time() * 1000)
        self._record.header["finished_at_ms"] = finished_ms
        self._record.header["outcome"] = str(
            outcome.value if isinstance(outcome, ImpulseOutcome) else outcome
        )
        touched = organs_touched if organs_touched is not None else self._organs_touched
        self._record.header["organs_touched"] = list(touched)
        self._record.summary = self._recorder._build_summary(self._record)
        self._recorder._commit(self._record)
        return self._record.summary


class ImpulseRecorder:
    """Per-cognitive-frame (4.038 ms) nervous impulse recorder; zero-overhead no-op when OFF."""

    def __init__(self, *, level: DebugLevel | None = None, store_dir: Path | None = None) -> None:
        self.level = level if level is not None else resolve_debug_level()
        self._tick_id: int = 0
        self._time_ms: float = 0.0
        self._store_dir: Path | None = store_dir.resolve() if store_dir is not None else None
        self._impulses: list[ImpulseRecord] = []

    @property
    def is_active(self) -> bool:
        return self.level != DebugLevel.OFF

    def begin_tick(self, tick_id: int, *, time_ms: float = 0.0, store_dir: Path | None = None) -> None:
        if not self.is_active:
            return
        self._tick_id = int(tick_id)
        self._time_ms = float(time_ms)
        if store_dir is not None:
            self._store_dir = store_dir.resolve()

    @contextmanager
    def channel(
        self,
        channel: ImpulseChannel | str = ImpulseChannel.COGNITIVE_FRAME,
        *,
        store_dir: Path | None = None,
    ) -> Iterator[ImpulseChannelWriter]:
        if store_dir is not None:
            self._store_dir = store_dir.resolve()
        if not self.is_active:
            yield ImpulseChannelWriter(self, ImpulseRecord(header={}))
            return
        started_ms = int(time.time() * 1000)
        record = ImpulseRecord(
            header={
                "schema": _SCHEMA,
                "impulse_id": str(uuid.uuid4()),
                "tick_id": self._tick_id,
                "sim_time_ms": round(self._time_ms, 3),
                "channel": str(channel.value if isinstance(channel, ImpulseChannel) else channel),
                "store_dir": str(self._store_dir or ""),
                "debug_level": self.level.value,
                "started_at_ms": started_ms,
                "finished_at_ms": 0,
                "outcome": ImpulseOutcome.HOVER_STABILIZE.value,
                "organs_touched": [],
            }
        )
        writer = ImpulseChannelWriter(self, record)
        try:
            yield writer
        finally:
            if self.is_active and not writer._finished:
                writer.finish(ImpulseOutcome.HOVER_STABILIZE)

    def last_impulses(self, limit: int | None = None) -> list[ImpulseRecord]:
        if limit is None:
            return list(self._impulses)
        return list(self._impulses[-max(1, limit) :])

    def clear(self) -> None:
        self._impulses.clear()

    def _build_summary(self, record: ImpulseRecord) -> dict[str, Any]:
        winners: list[str] = []
        suppressed: list[str] = []
        thresholds: list[str] = []
        routes: list[str] = []
        for event in record.events:
            if event.kind == "select" and event.selected:
                winners.extend(event.selected)
            elif event.kind == "suppress" and event.reason:
                suppressed.append(event.reason)
            elif event.kind == "threshold" and event.reason:
                thresholds.append(event.reason)
            elif event.kind == "route" and event.selected:
                routes.extend(event.selected)
        return {
            "impulse_id": record.header.get("impulse_id"),
            "tick_id": record.header.get("tick_id"),
            "sim_time_ms": record.header.get("sim_time_ms"),
            "channel": record.header.get("channel"),
            "outcome": record.header.get("outcome"),
            "organs_touched": record.header.get("organs_touched", []),
            "winners": winners[:8],
            "suppressed": suppressed[:6],
            "thresholds": thresholds[:4],
            "routes": routes[:4],
            "event_count": len(record.events),
        }

    def _commit(self, record: ImpulseRecord) -> None:
        self._impulses.append(record)
        if len(self._impulses) > _RING_MAX:
            self._impulses = self._impulses[-_RING_MAX:]
        if self._store_dir is None:
            return
        trace_dir = self._store_dir / "nervous"
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / "impulse_trace.jsonl"
        line = json.dumps(
            record.to_dict(debug_level=self.level),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        self._rotate(trace_path)

    @staticmethod
    def _rotate(trace_path: Path) -> None:
        if _RING_MAX <= 0 or not trace_path.is_file():
            return
        try:
            lines = trace_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        if len(lines) <= _RING_MAX:
            return
        trimmed = lines[-_RING_MAX:]
        trace_path.write_text("\n".join(trimmed) + "\n", encoding="utf-8")


_GLOBAL = ImpulseRecorder()


def impulse_recorder(*, level: DebugLevel | None = None, store_dir: Path | None = None) -> ImpulseRecorder:
    if level is not None or store_dir is not None:
        return ImpulseRecorder(level=level or resolve_debug_level(), store_dir=store_dir)
    _GLOBAL.level = resolve_debug_level()
    return _GLOBAL


def read_impulse_trace(
    store_dir: Path,
    *,
    impulse_id: str | None = None,
    tick_id: int | None = None,
    channel: str | None = None,
    outcome: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Read persisted nervous impulse trace rows from `<store_dir>/nervous/impulse_trace.jsonl`."""
    path = store_dir.resolve() / "nervous" / "impulse_trace.jsonl"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            continue
        if impulse_id and doc.get("impulse_id") != impulse_id:
            continue
        if tick_id is not None and int(doc.get("tick_id") or -1) != tick_id:
            continue
        if channel and doc.get("channel") != channel:
            continue
        if outcome and doc.get("outcome") != outcome:
            continue
        rows.append(doc)
    return rows[-max(1, int(limit)) :]


def format_impulse_replay(records: list[ImpulseRecord | dict[str, Any]]) -> str:
    """Format a human-readable synaptic wave trace across the fly's organs and ports."""
    lines: list[str] = []
    for item in records:
        doc = item.to_dict(debug_level=DebugLevel.FULL) if isinstance(item, ImpulseRecord) else item
        summary = doc.get("summary") or {}
        tick_id = doc.get("tick_id", 0)
        sim_ms = doc.get("sim_time_ms", 0.0)
        outcome = doc.get("outcome", "")
        organs = " -> ".join(doc.get("organs_touched") or [])
        lines.append(
            f"[T={tick_id:03d} | {sim_ms:6.1f} ms] outcome={outcome.upper()} | wave: {organs}"
        )
        for ev in doc.get("events") or []:
            seq = ev.get("seq", 0)
            kind = ev.get("kind", "")
            organ = ev.get("organ", "")
            port = ev.get("port", "")
            node = ev.get("node", "")
            detail_parts: list[str] = []
            if ev.get("selected"):
                detail_parts.append(f"selected={ev['selected']}")
            if ev.get("reason"):
                detail_parts.append(f"reason={ev['reason']}")
            if ev.get("rejected_above_floor"):
                detail_parts.append(f"suppressed_cells={ev['rejected_above_floor']}")
            if ev.get("potentials"):
                pots = ", ".join(
                    f"{p.get('id')}={p.get('score'):.3f}"
                    if isinstance(p.get("score"), (int, float))
                    else f"{p.get('id')}={p.get('score')}"
                    for p in ev["potentials"][:4]
                )
                detail_parts.append(f"potentials=[{pots}]")
            if ev.get("inputs"):
                inps = ", ".join(
                    f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in list(ev["inputs"].items())[:4]
                )
                detail_parts.append(f"in=({inps})")
            detail = " · ".join(detail_parts)
            lines.append(f"  #{seq:02d} [{organ}:{kind}] ({port}) {node} -> {detail}")
        if summary.get("winners"):
            lines.append(f"  => winners: {', '.join(summary['winners'])} | routes: {', '.join(summary.get('routes') or [])}")
    return "\n".join(lines)
