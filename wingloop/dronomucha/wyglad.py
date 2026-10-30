"""How the drone's jobs look to the person watching: textures, models, mountains.

All of it is drawing (:data:`~wingloop.lab.watch.SCENERY`,
:func:`~wingloop.lab.arena.add_parts`): the drone's eye and sensors see the
kinds' own solids, of the same size, and nothing here changes what it
finds. Textures and the mountains' height map are made here, once, from
fixed seeds, into a cache folder.

The arena's scale is 1:50: a millimetre is 5 cm.
"""

from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

#: Where the made textures are kept.
CACHE = Path(tempfile.gettempdir()) / "dronomucha_wyglad"


def _tileable(n: int, waves, seed: int) -> np.ndarray:
    """A smooth pattern on an n x n picture that tiles: sums of whole waves across it."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n] / n
    out = np.zeros((n, n))
    for fx, fy, amp in waves:
        out += amp * np.sin(2 * np.pi * (fx * x + fy * y) + rng.uniform(0, 2 * np.pi))
    return out


def _save(name: str, rgb: np.ndarray) -> Path:
    from PIL import Image

    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(path)
    return path


@lru_cache(maxsize=None)
def snieg() -> Path:
    """Snow: white, faintly blue in its hollows, with a fine grain."""
    n = 512
    rng = np.random.default_rng(11)
    drifts = _tileable(n, [(1, 2, 1.0), (3, -1, 0.6), (2, 5, 0.35), (7, 3, 0.2)], 1)
    grain = rng.normal(0, 1, (n, n))
    base = np.array([238.0, 241.0, 247.0])
    shade = (drifts[..., None] * np.array([6.0, 5.0, 2.0])) + grain[..., None] * 2.5
    return _save("snieg.png", base + shade)


@lru_cache(maxsize=None)
def woda() -> Path:
    """Sea: deep blue, lighter on the crests of swells crossing a chop, a little foam."""
    n = 1024
    rng = np.random.default_rng(4)
    waves = [(1, 2, 1.0), (2, -1, 0.8), (3, 1, 0.5)]
    waves += [
        (int(a), int(b), c)
        for a, b, c in zip(
            rng.integers(4, 31, 14),
            rng.integers(-30, 31, 14),
            rng.uniform(0.08, 0.3, 14),
            strict=True,
        )
    ]
    crest = _tileable(n, waves, 2)
    crest = crest / np.abs(crest).max()
    base = np.array([20.0, 58.0, 92.0])
    light = np.array([30.0, 46.0, 52.0])
    foam = np.clip(crest - 0.72, 0, None)[..., None] * np.array([160.0, 160.0, 150.0])
    return _save("woda.png", base + crest[..., None] * light + foam)


@lru_cache(maxsize=None)
def gory_mapa() -> Path:
    """A height map: flat in the middle, a ring of peaks beyond, as a picture (white high)."""
    n = 256
    y, x = (np.mgrid[0:n, 0:n] - (n - 1) / 2) / ((n - 1) / 2)
    r = np.hypot(x, y)
    a = np.arctan2(y, x)
    rng = np.random.default_rng(5)
    ridge = sum(
        rng.uniform(0.3, 1.0) * np.cos(k * a + rng.uniform(0, 2 * np.pi))
        for k in (3, 5, 7, 11, 17)
    )
    rough = _tileable(n, [(5, 3, 1.0), (9, 7, 0.6), (17, 13, 0.3), (31, 23, 0.15)], 6)
    rise = np.clip((r - 0.42) / 0.25, 0, 1) ** 1.5 * np.clip((1.05 - r) / 0.2, 0, 1)
    h = rise * (0.6 + 0.15 * ridge + 0.12 * rough)
    h = np.clip(h / max(h.max(), 1e-9), 0, 1)
    return _save("gory.png", np.repeat(255 * h[..., None], 3, axis=-1))


def gory_w_tle(root) -> None:
    """Snowy mountains all round the horizon, beyond the floor (``SCENERY["decor"]``)."""
    import xml.etree.ElementTree as ET

    from wingloop.lab.arena import FLOOR

    asset = root.find("asset")
    ET.SubElement(
        asset, "hfield", name="dronomucha_gory", file=str(gory_mapa()), size="2600 2600 200 1"
    )
    ET.SubElement(
        asset,
        "material",
        name="dronomucha_gory",
        rgba="0.86 0.88 0.92 1",
        specular="0.1",
        shininess="0.1",
    )
    ET.SubElement(
        root.find("worldbody"),
        "geom",
        name="dronomucha_gory",
        type="hfield",
        hfield="dronomucha_gory",
        pos=f"0 0 {FLOOR - 1.2}",
        material="dronomucha_gory",
        contype="0",
        conaffinity="0",
    )


# ---------------------------------------------------------------- models
# Each kind's parts, relative to its middle (the kind's own solid is the box
# or ball the eye sees; these fill about the same space). Lengths along x.

SKIN = (0.86, 0.66, 0.52, 1)


def czlowiek_lezacy(jacket=(0.85, 0.16, 0.07, 1), trousers=(0.16, 0.18, 0.26, 1)) -> list:
    """A climber lying on its back, 1.7 m along x, helmet on (34 x 10 x 6 mm)."""
    parts = [
        dict(type="sphere", size=2.3, pos=(14.4, 0, 0.2), rgba=SKIN),
        dict(
            type="ellipsoid",
            size=(2.5, 2.5, 1.6),
            pos=(14.6, 0, 1.0),
            rgba=(0.98, 0.78, 0.08, 1),
        ),
        dict(type="capsule", size=3.1, fromto=(2.5, 0, 0, 10.5, 0, 0), rgba=jacket),
        dict(
            type="ellipsoid", size=(3.6, 2.6, 1.2), pos=(6.5, 0, -1.6), rgba=(0.2, 0.25, 0.3, 1)
        ),  # rucksack under it
    ]
    for side in (-1, 1):
        parts += [
            dict(
                type="capsule",
                size=1.25,
                fromto=(10, 4.2 * side, 0.2, 3, 4.8 * side, -0.4),
                rgba=jacket,
            ),
            dict(type="sphere", size=1.0, pos=(2.2, 4.9 * side, -0.5), rgba=(0.1, 0.1, 0.1, 1)),
            dict(
                type="capsule",
                size=1.55,
                fromto=(2.0, 1.7 * side, -0.2, -12.5, 2.1 * side, -0.8),
                rgba=trousers,
            ),
            dict(
                type="capsule",
                size=1.3,
                fromto=(-12.8, 2.1 * side, -0.8, -15.5, 2.2 * side, 0.6),
                rgba=(0.2, 0.12, 0.08, 1),
            ),
        ]
    return parts


def rozbitek() -> list:
    """A castaway afloat on the back in an orange life jacket; the water hides the rest."""
    jacket = (0.98, 0.42, 0.04, 1)
    parts = [
        dict(type="sphere", size=2.1, pos=(13.2, 0, 0.9), rgba=SKIN),
        dict(
            type="ellipsoid",
            size=(2.2, 2.2, 1.0),
            pos=(13.4, 0, 2.0),
            rgba=(0.25, 0.18, 0.12, 1),
        ),  # wet hair
        dict(type="capsule", size=2.8, fromto=(3.0, 0, 0.4, 10.8, 0, 0.6), rgba=jacket),
        dict(type="ellipsoid", size=(2.4, 3.6, 1.2), pos=(11.4, 0, 1.4), rgba=jacket),
    ]
    for side in (-1, 1):
        parts += [
            dict(
                type="capsule",
                size=1.1,
                fromto=(9, 3.0 * side, -0.2, 5, 8.0 * side, -1.3),
                rgba=(0.12, 0.14, 0.2, 1),
            ),
            dict(
                type="capsule",
                size=1.4,
                fromto=(2.0, 1.5 * side, -1.4, -12, 2.2 * side, -2.4),
                rgba=(0.1, 0.12, 0.22, 1),
            ),
        ]
    return parts


def kozica() -> list:
    """A chamois standing, head up, along x (20 x 6 x 14 mm)."""
    fur, dark = (0.42, 0.29, 0.16, 1), (0.08, 0.07, 0.06, 1)
    parts = [
        dict(type="ellipsoid", size=(6.8, 2.6, 3.2), pos=(0, 0, 1.6), rgba=fur),
        dict(type="capsule", size=1.5, fromto=(5, 0, 3, 7.6, 0, 6.3), rgba=fur),
        dict(
            type="ellipsoid",
            size=(2.6, 1.3, 1.3),
            pos=(8.9, 0, 6.6),
            euler=(0, 25, 0),
            rgba=(0.82, 0.75, 0.62, 1),
        ),  # pale face
        dict(type="ellipsoid", size=(0.6, 0.3, 0.9), pos=(-7.0, 0, 3.0), rgba=dark),  # tail
    ]
    for side in (-1, 1):
        parts.append(
            dict(
                type="capsule",
                size=0.32,
                fromto=(8.9, 0.55 * side, 7.6, 8.6, 0.7 * side, 10.2),
                rgba=dark,
            )
        )
        for front in (-4.8, 4.8):
            leg = (front, 1.6 * side, 0.4, front, 1.6 * side, -6.6)
            parts.append(dict(type="capsule", size=0.8, fromto=leg, rgba=fur))
    return parts


def kamien() -> list:
    """A boulder, a little snow on its top (14 mm across)."""
    return [
        dict(
            type="ellipsoid",
            size=(7.2, 6.2, 4.6),
            pos=(0, 0, -0.4),
            euler=(0, 0, 20),
            rgba=(0.31, 0.29, 0.27, 1),
        ),
        dict(
            type="ellipsoid",
            size=(4.6, 3.8, 3.4),
            pos=(2.6, -2.2, 1.6),
            euler=(10, 0, 50),
            rgba=(0.38, 0.36, 0.33, 1),
        ),
        dict(
            type="ellipsoid",
            size=(4.8, 4.0, 1.0),
            pos=(-0.8, 0.8, 3.8),
            rgba=(0.95, 0.96, 0.98, 1),
        ),
    ]


def foka() -> list:
    """A seal afloat, head up, 1.4 m along x."""
    coat = (0.36, 0.34, 0.33, 1)
    parts = [
        dict(type="ellipsoid", size=(11.5, 3.6, 2.2), pos=(0, 0, 0), rgba=coat),
        dict(type="sphere", size=2.5, pos=(11.6, 0, 1.4), rgba=(0.3, 0.28, 0.27, 1)),
        dict(type="sphere", size=0.35, pos=(13.8, 0.9, 2.0), rgba=(0.02, 0.02, 0.02, 1)),
        dict(type="sphere", size=0.35, pos=(13.8, -0.9, 2.0), rgba=(0.02, 0.02, 0.02, 1)),
    ]
    for side in (-1, 1):
        parts += [
            dict(
                type="ellipsoid",
                size=(2.6, 1.0, 0.4),
                pos=(5, 4.0 * side, 0),
                euler=(0, 0, 30 * side),
                rgba=coat,
            ),
            dict(
                type="ellipsoid",
                size=(2.2, 1.1, 0.3),
                pos=(-12.4, 1.2 * side, 0),
                euler=(0, 0, 25 * side),
                rgba=coat,
            ),
        ]
    return parts


def smieci() -> list:
    """Planks and a plastic crate afloat, a metre across."""
    wood = (0.52, 0.38, 0.24, 1)
    return [
        dict(type="box", size=(9.5, 1.8, 0.5), pos=(0, -5, 0), euler=(0, 0, 6), rgba=wood),
        dict(
            type="box",
            size=(8.5, 1.8, 0.5),
            pos=(-0.5, 0, 0.2),
            euler=(0, 0, -4),
            rgba=(0.46, 0.33, 0.2, 1),
        ),
        dict(type="box", size=(9.0, 1.6, 0.5), pos=(0.5, 4.8, 0), euler=(0, 0, 3), rgba=wood),
        dict(type="box", size=(3.0, 2.4, 2.0), pos=(5.5, 3.5, 1.2), rgba=(0.12, 0.38, 0.82, 1)),
    ]


def iglica() -> list:
    """A rock spire for a column (3 mm across, 60 tall in the eye), from its foot."""
    rock, light = (0.36, 0.35, 0.34, 1), (0.44, 0.43, 0.41, 1)
    return [
        dict(type="ellipsoid", size=(5.0, 4.4, 15), pos=(0, 0, 14), rgba=rock),
        dict(type="ellipsoid", size=(3.6, 3.2, 13), pos=(0.4, -0.3, 35), rgba=light),
        dict(type="ellipsoid", size=(2.4, 2.2, 9), pos=(-0.2, 0.2, 51), rgba=rock),
        dict(
            type="ellipsoid",
            size=(2.2, 2.0, 1.4),
            pos=(-0.2, 0.2, 59.5),
            rgba=(0.95, 0.96, 0.98, 1),
        ),
    ]


#: The jobs' scenery for the person watching (:data:`~wingloop.lab.watch.SCENERY`).
def scena_gory() -> dict:
    return dict(
        floor_texture=snieg(),
        floor_repeat="40 40",
        floor_reflectance=0.04,
        sky=("0.42 0.6 0.86", "0.88 0.93 1"),
        column_parts=iglica(),
        fog=(500.0, 2600.0, "0.86 0.9 0.97 1"),
        decor=gory_w_tle,
        map=dict(bg=(238, 241, 247), grid=(214, 220, 232), metre=20.0),
    )


def scena_morze() -> dict:
    return dict(
        floor_texture=woda(),
        floor_repeat="14 14",
        floor_reflectance=0.22,
        sky=("0.4 0.56 0.78", "0.86 0.9 0.96"),
        fog=(250.0, 1300.0, "0.74 0.8 0.88 1"),
        map=dict(bg=(176, 206, 228), grid=(156, 190, 216), metre=20.0),
    )


#: How warm each part looks to a thermal camera, as a share of its item's
#: warmth (:mod:`.termo`), by the part's colour: the face warmest, a jacket
#: warm, trousers less, boots, a helmet and a rucksack cold; a life jacket's
#: foam keeps the warmth in; snow on a rock is as cold as the ground.
CIEPLO = {
    SKIN: 1.25,
    (0.98, 0.78, 0.08, 1): 0.35,  # helmet
    (0.2, 0.25, 0.3, 1): 0.25,  # rucksack
    (0.1, 0.1, 0.1, 1): 0.6,  # gloves
    (0.16, 0.18, 0.26, 1): 0.75,  # trousers
    (0.2, 0.12, 0.08, 1): 0.25,  # boots
    (0.98, 0.42, 0.04, 1): 0.6,  # life jacket
    (0.25, 0.18, 0.12, 1): 0.9,  # wet hair
    (0.12, 0.14, 0.2, 1): 0.8,  # sleeves in the water
    (0.1, 0.12, 0.22, 1): 0.6,  # legs in the water
    (0.82, 0.75, 0.62, 1): 1.2,  # a chamois' face
    (0.08, 0.07, 0.06, 1): 0.3,  # horns, tail
    (0.38, 0.36, 0.33, 1): 0.85,  # the rock's shaded side
    (0.95, 0.96, 0.98, 1): 0.0,  # snow on it
    (0.3, 0.28, 0.27, 1): 1.2,  # a seal's head
    (0.12, 0.38, 0.82, 1): 0.6,  # a plastic crate
}
