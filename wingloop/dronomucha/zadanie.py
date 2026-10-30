"""What the drone is for: a task, and the register of tasks.

A task is everything that makes one use of the drone different from another
-- what is in its world, what it looks for, what its eye sees through, how
its world looks -- and nothing else: the drone, its eye, its brain and its
navigator are the lab's (:mod:`wingloop.lab`), the same for every task. A new
use of the drone is a new :class:`Zadanie` in :mod:`.zadania`, registered with
:func:`dodaj`; nothing in the lab changes for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Zadanie:
    """One use of the drone.

    ``kinds``: what is in its world, as the lab's :data:`~wingloop.lab.arena.KINDS`
    -- each with its Polish name ``pl``, colour ``rgb`` for the person
    watching, ``shape`` (``sphere``, ``box`` or ``bar``), ``size`` (a radius,
    or a box's half-sizes, mm), where its middle is (``z``, mm), and how much
    warmer than the ground it is to a thermal camera (``heat``, a range, K).
    ``target``: the kind it looks for; the others are what it must not take
    for it. ``eye``: what the eye looks through, ``"termowizja"`` or
    ``"widzialne"``. ``scenery``: the 3-D view's floor and sky
    (:data:`~wingloop.lab.watch.SCENERY`). ``column_heat``, ``clutter``,
    ``span``: the landscape's columns' warmth, the ground's unevenness and the
    thermal camera's least and most span, K. ``per_kind``: how many of each
    kind in a search area. ``lessons``: tethered views and close looks it is
    taught on. ``pattern``: places, mm from the start, it flies to in turn
    to look around from when it has nothing left to look at
    (:attr:`~wingloop.lab.navigate.Navigator.pattern`). ``nawigacja``: the
    navigator's settings this task flies with instead of the drone's own
    (:class:`~wingloop.lab.navigate.Navigator`'s fields). ``ground_c``: the
    ground's temperature, C, for a thermal camera's screen (:mod:`.termo`).
    ``mapa``: the task's search map (:class:`~.mapa.Mapa`'s settings) --
    where its target is likely to be, and how far a look finds it -- for a
    drone flying with one (:attr:`~.dodatki.Dodatki.mapa`).
    """

    name: str
    pl: str
    target: str
    kinds: dict
    eye: str = "termowizja"
    scenery: dict = field(default_factory=dict)
    column_heat: float = 1.5
    clutter: float = 0.3
    span: tuple = (6.0, 20.0)
    per_kind: int = 2
    lessons: tuple = (90, 300)
    pattern: tuple = ()
    nawigacja: dict = field(default_factory=dict)
    ground_c: float = 0.0
    mapa: dict = field(default_factory=dict)
    about: str = ""


def kwadrat(step: float, legs: int) -> tuple:
    """An expanding square search (IAMSAR): out ``step``, turn left, legs growing by ``step``.

    The places it turns at, mm from the start: (s, 0), (s, s), (-s, s),
    (-s, -s), (2s, -s), (2s, 2s), ... -- the pattern a rescue aircraft flies
    round a datum, each turn a place to look around from.
    """
    x = y = 0.0
    out = []
    for i in range(legs):
        length = step * (i // 2 + 1)
        dx, dy = ((1, 0), (0, 1), (-1, 0), (0, -1))[i % 4]
        x, y = x + dx * length, y + dy * length
        out.append((x, y))
    return tuple(out)


#: Every task there is, by name.
ZADANIA: dict[str, Zadanie] = {}


def dodaj(zadanie: Zadanie) -> Zadanie:
    """Register a task; it is then known by name everywhere (:func:`zadanie`, the CLI)."""
    if zadanie.target not in zadanie.kinds:
        raise ValueError(f"{zadanie.name}: cel {zadanie.target!r} nie jest wśród rodzajów")
    ZADANIA[zadanie.name] = zadanie
    return zadanie


def zadanie(name: str) -> Zadanie:
    """A registered task by name."""
    from . import zadania  # noqa: F401 -- registers the tasks shipped

    if name not in ZADANIA:
        raise SystemExit(f"Nie znam zadania {name!r}. Znane: {', '.join(sorted(ZADANIA))}")
    return ZADANIA[name]
