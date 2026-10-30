"""Flying a task: a search area, the drone ready to search it, searches, and a film."""

from __future__ import annotations

from wingloop.lab import arena, autopilot

from .dodatki import MUCHA, Dodatki
from .mozg import mozg
from .swiat import oko, swiat
from .zadanie import Zadanie


def obszar(z: Zadanie, seed: int, moving: float = 0.0):
    """A search area: ``per_kind`` of the target and of each other kind, around the start."""
    with swiat(z):
        return arena.mission(seed, kinds=tuple(z.kinds), per_kind=z.per_kind, moving=moving)


def pilot(fly_eye, mb, valence, z: Zadanie, items, dodatki: Dodatki = MUCHA):
    """The drone, ready to search this area for the task's target, with ``dodatki``."""
    with swiat(z):
        fly_eye.set_items(items)
        p = autopilot._pilot(items, mb, fly_eye, valence, {z.target}, True, True)
    p.nav.pattern = tuple(z.pattern)
    for name, value in z.nawigacja.items():
        setattr(p.nav, name, value)
    if dodatki.mapa is not None:
        from .mapa import Mapa

        p.nav.planner = Mapa(**{**z.mapa, **dodatki.mapa})
    p.operator = None
    if dodatki.operator is not None:
        from .operator import Operator

        p.operator = Operator(p.nav.mushroom_body, fly_eye, items, z.target, **dodatki.operator)
        p.operator.nav = p.nav
        p.nav.mushroom_body = p.operator
    if dodatki.pamiec:
        from .pamiec import Pamiec

        p.nav.mushroom_body = p.nav.memory = Pamiec(p.nav.mushroom_body, fly_eye)
        p.nav.memory.nav = p.nav
    return p


def _czas(p) -> float:
    """How long the search has taken: its flight, and hovering for the operator's answers."""
    return p.flight.t + (p.operator.czekanie if p.operator is not None else 0.0)


def szukaj(
    z: Zadanie,
    seeds,
    progress=print,
    fly_eye=None,
    moving: float = 0.0,
    dodatki: Dodatki = MUCHA,
) -> list:
    """Search each area, with ``dodatki``; one row per area.

    ``found`` of ``wanted`` targets, ``wrong``: anything else it stopped at
    -- a false alarm -- ``first``: when it found the first target, s, and
    ``t``: how long the search took, s.
    """
    fly_eye = fly_eye or oko(z)
    mb, valence = mozg(fly_eye, z, progress=progress, dodatki=dodatki)
    rows = []
    for seed in seeds:
        items = obszar(z, seed, moving)
        p = pilot(fly_eye, mb, valence, z, items, dodatki)
        p.nav.reward = None
        first = None
        with swiat(z):
            while _czas(p) < autopilot.MISSION_S[True] and not p.done:
                p.step()
                if first is None and any(it.kind == z.target for it, d in p.landed if d < 30):
                    first = _czas(p)
            reached = p.reached()
        wanted = [it for it in items if it.kind == z.target]
        row = dict(
            seed=seed,
            found=sum(it in reached for it in wanted),
            wanted=len(wanted),
            wrong=sum(it not in wanted for it in reached),
            first=None if first is None else round(first, 1),
            t=round(_czas(p), 1),
        )
        if p.operator is not None:
            row.update(
                pytania=len(p.operator.pytania),
                zle_odpowiedzi=sum(a != b for _, a, b in p.operator.pytania),
            )
        rows.append(row)
        if progress:
            progress(
                f"  obszar {seed}: znalezione {row['found']}/{row['wanted']}, "
                f"fałszywe alarmy {row['wrong']}, pierwsze po {row['first']} s, "
                f"lot {row['t']} s"
            )
    return rows


def film(
    z: Zadanie,
    path,
    seed: int = 0,
    thermal: str | None = "bw",
    progress=print,
    size=(1280, 720),
    dodatki: Dodatki = MUCHA,
):
    """Record one search, with the thermal camera's picture in the corner, ``size`` pixels."""
    fly_eye = oko(z)
    mb, valence = mozg(fly_eye, z, progress=progress, dodatki=dodatki)
    p = pilot(fly_eye, mb, valence, z, obszar(z, seed), dodatki)
    with swiat(z):
        seconds = autopilot.MISSION_S[True]
        out = autopilot.record(
            path, p, True, seconds, 33.0, *size, thermal=thermal, distance=24.0
        )
        if progress:
            progress(p.summary())
    return out
