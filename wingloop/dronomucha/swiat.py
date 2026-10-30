"""The one place the drone's tasks reach into the lab.

The lab's arena knows its kinds, names and scenery as module settings
(:data:`~wingloop.lab.arena.KINDS`, :data:`~wingloop.lab.autopilot.POLISH`,
:data:`~wingloop.lab.watch.SCENERY`, the thermal camera's settings). A task
puts its own in place while it is flown, and puts the lab's back after --
whatever happens -- so the lab's own missions and lessons are never touched.
"""

from __future__ import annotations

from contextlib import contextmanager

from wingloop.lab import arena, autopilot, watch

from .zadanie import Zadanie


@contextmanager
def swiat(z: Zadanie):
    """Make the arena this task's world while in the ``with``; then put the lab's back."""
    saved = (
        dict(arena.KINDS),
        dict(autopilot.POLISH),
        dict(autopilot.NAMES),
        dict(watch.SCENERY),
        (watch.IR_HEAT, watch.IR_CLUTTER, watch.THERMAL_SPAN_MIN, watch.THERMAL_SPAN),
    )
    arena.KINDS.clear()
    arena.KINDS.update(z.kinds)
    autopilot.POLISH.clear()
    autopilot.POLISH.update({k: v["pl"] for k, v in z.kinds.items()})
    autopilot.NAMES.update({v["pl"]: k for k, v in z.kinds.items()})
    watch.SCENERY.update(z.scenery)
    watch.IR_HEAT, watch.IR_CLUTTER = z.column_heat, z.clutter
    watch.THERMAL_SPAN_MIN, watch.THERMAL_SPAN = z.span
    try:
        yield z
    finally:
        kinds, polish, names, scenery, thermal = saved
        for live, was in (
            (arena.KINDS, kinds),
            (autopilot.POLISH, polish),
            (autopilot.NAMES, names),
            (watch.SCENERY, scenery),
        ):
            live.clear()
            live.update(was)
        watch.IR_HEAT, watch.IR_CLUTTER, watch.THERMAL_SPAN_MIN, watch.THERMAL_SPAN = thermal


def oko(z: Zadanie):
    """The drone's eye for a task, looking through what the task says."""
    with swiat(z):
        e = watch.FlysEye(every_ms=autopilot.EYE_MS[True])
    e.thermal = z.eye == "termowizja"
    return e
