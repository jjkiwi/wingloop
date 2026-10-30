"""Things to fly to: the objects of a mission, shared by the 3-D view and the eye.

The scenery columns are bright on purpose and stay background. Mission
objects are dark, because flies fixate dark objects and because the eye's
object detector (:meth:`~wingloop.lab.vision.FlyEye.dark`) looks for exactly
that. Three kinds that differ the way the classic fly experiments' shapes do --
round, horizontal, vertical: a ball, a flat slab and a thin pole. A cube was
tried first in the slab's place and the eye could not tell it from the ball:
both are compact dark blobs, and the mushroom body valued them alike.

Sizes are set by the eye, not by taste. An ommatidium sees about five degrees,
so an object has to fill most of one to read as dark at all: a 5 mm pole
vanished at 120 mm, and a slab 4.4 mm thick, seen edge-on from flight height,
was flown past without being noticed.

One list of :class:`Item` drives both the MuJoCo scene the person watches
and the numpy scene the fly's eye renders, so what is drawn and what is seen
cannot drift apart.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

#: The floor of the arena, mm (the fly starts at the origin).
FLOOR = -12.0
#: Kinds, their colour in the 3-D view, and their size, mm.
KINDS = {
    "sphere": {"rgb": (0.45, 0.07, 0.07), "size": 8.0, "pl": "kula"},
    "slab": {"rgb": (0.07, 0.12, 0.42), "size": (14.0, 14.0, 6.0), "pl": "płytka"},
    "bar": {"rgb": (0.12, 0.12, 0.12), "size": 5.0, "pl": "słupek"},
}
#: Height of a pole, mm, from the floor.
BAR_HEIGHT = 40.0


@dataclass(frozen=True)
class Item:
    kind: str
    x: float
    y: float
    z: float = 5.0
    label: str = ""
    #: Camouflage, 0-1: painted in the floor's checker instead of its colour
    #: (:attr:`~.scene.Thing.camo`). Only the eye sees it.
    camo: float = 0.0
    #: How fast it moves, mm/s: round the start, on its own circle, anticlockwise
    #: if positive. ``x`` and ``y`` are where it is at the start (:meth:`at`).
    speed: float = 0.0

    def at(self, t: float) -> tuple[float, float]:
        """Where it is ``t`` seconds into the mission."""
        if not self.speed:
            return self.x, self.y
        r = float(np.hypot(self.x, self.y))
        a = np.arctan2(self.y, self.x) + self.speed * t / r
        return float(r * np.cos(a)), float(r * np.sin(a))

    @property
    def size(self):
        return KINDS[self.kind]["size"]

    @property
    def rgb(self) -> tuple:
        return KINDS[self.kind]["rgb"]

    @property
    def name(self) -> str:
        return self.label or self.kind

    def bearing_from(self, x: float, y: float) -> float:
        """Compass heading from (x, y) to this item, degrees."""
        return float(np.degrees(np.arctan2(self.y - y, self.x - x)))

    def distance_from(self, x: float, y: float) -> float:
        return float(np.hypot(self.x - x, self.y - y))


def luminance(rgb) -> float:
    """What a green-sensitive photoreceptor makes of an RGB colour."""
    r, g, b = rgb
    return 0.3 * r + 0.55 * g + 0.15 * b


def mission(
    seed: int = 0,
    kinds=("sphere", "slab", "bar"),
    per_kind: int = 1,
    distance=(50.0, 80.0),
    height: float = 5.0,
    camo: float = 0.0,
    moving: float = 0.0,
) -> tuple[Item, ...]:
    """A random arena: ``per_kind`` of each kind around the start, well apart.

    ``camo``: every object painted that much in the floor's checker. The
    arena is otherwise the same as without it. ``moving``: every object goes
    round the start at that many mm/s, each its own way round (the same
    objects, starting where they stand without it).

    Headings are spread evenly, each jittered by up to 10 degrees, so that
    seen from the start no two objects are closer than 360/n - 20 degrees
    (100 for three objects, 40 for six). That keeps one object in the middle
    of the eye at a time while the fly looks around.
    """
    rng = np.random.default_rng(seed)
    wanted = [k for k in kinds for _ in range(per_kind)]
    n = len(wanted)
    slots = np.linspace(0, 360, n, endpoint=False) + rng.uniform(0, 360)
    rng.shuffle(wanted)
    ways = np.random.default_rng(seed + 1_000_003).choice([-1.0, 1.0], n)
    items = []
    for kind, heading, way in zip(wanted, slots, ways, strict=True):
        h = np.deg2rad(heading + rng.uniform(-10, 10))
        d = rng.uniform(*distance)
        items.append(
            Item(
                kind,
                round(float(d * np.cos(h)), 1),
                round(float(d * np.sin(h)), 1),
                height,
                camo=camo,
                speed=float(way * moving),
            )
        )
    return tuple(items)


def move_geoms(model, items, t: float) -> None:
    """Put the moving items of a MuJoCo scene (:func:`add_to_mjcf`) where they are at ``t``."""
    for i, it in enumerate(items):
        if it.speed:
            model.geom_pos[model.geom(f"flylab_item{i}").id][:2] = it.at(t)


def shape_of(kind: str) -> str:
    """The solid a kind is drawn as: ``sphere``, ``box`` or ``bar`` (a standing cylinder)."""
    return KINDS[kind].get("shape", {"bar": "bar", "slab": "box"}.get(kind, "sphere"))


def _placed(it) -> tuple[float, float, float]:
    """Where an item's middle is: its kind can fix the height (lying on the ground, afloat)."""
    return it.x, it.y, KINDS[it.kind].get("z", it.z)


def add_to_mjcf(world: ET.Element, items) -> None:
    """Draw the items in the MuJoCo scene (no collisions: they are scenery)."""
    for i, it in enumerate(items):
        rgba = " ".join(f"{c:.3f}" for c in it.rgb) + " 1"
        common = dict(name=f"flylab_item{i}", rgba=rgba, contype="0", conaffinity="0")
        if it.camo:
            # The floor's checker, tinted by what is left of its own colour.
            tint = [it.camo + (1 - it.camo) * c for c in it.rgb]
            common.update(
                rgba=" ".join(f"{c:.3f}" for c in tint) + " 1", material="flylab_floor"
            )
        x, y, z = _placed(it)
        shape = shape_of(it.kind)
        if shape == "sphere":
            ET.SubElement(
                world, "geom", type="sphere", size=f"{it.size}", pos=f"{x} {y} {z}", **common
            )
        elif shape == "box":
            sx, sy, sz = it.size
            ET.SubElement(
                world, "geom", type="box", size=f"{sx} {sy} {sz}", pos=f"{x} {y} {z}", **common
            )
        else:
            h = KINDS[it.kind].get("height", BAR_HEIGHT)
            ET.SubElement(
                world,
                "geom",
                type="cylinder",
                size=f"{it.size} {h / 2}",
                pos=f"{x} {y} {FLOOR + h / 2}",
                **common,
            )


def eye_things(items) -> list:
    """The same items as :class:`~wingloop.lab.scene.Thing` for the eye's renderer.

    A kind with a ``heat`` range, kelvin above the ground, is that much warmer
    to a thermal camera, each item its own share of the range.
    """
    from .scene import Thing

    out = []
    for it in items:
        x, y, z = _placed(it)
        common = dict(brightness=luminance(it.rgb), label=it.name, camo=it.camo)
        shape = shape_of(it.kind)
        if shape == "bar":
            h = KINDS[it.kind].get("height", BAR_HEIGHT)
            thing = Thing("bar", [x, y, FLOOR], it.size, height=h, **common)
        elif shape == "box":
            thing = Thing("box", [x, y, z], np.asarray(it.size), **common)
        else:
            thing = Thing("sphere", [x, y, z], it.size, **common)
        if "heat" in KINDS[it.kind]:
            lo, hi = KINDS[it.kind]["heat"]
            u = (np.sin(12.9898 * it.x + 78.233 * it.y) * 43758.5453) % 1.0
            thing.heat = float(lo + (hi - lo) * u)
        out.append(thing)
    return out
