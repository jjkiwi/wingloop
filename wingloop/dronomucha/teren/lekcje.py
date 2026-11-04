"""Lessons for the fly's brain at real scale: warm spots seen from the air, and what they were.

The drone, its camera looking down, is shown search areas it will not
search: from 12-110 m up -- the heights it searches from and comes down to
-- over people, chamois, foxes and sun-warmed boulders, and over bare
ground. Each warm spot found (:func:`~.wykrywanie.plamy`)
is a lesson, labelled by what really lies there -- a person, or not -- as
the dopamine of a real find would label it. The search areas are the same
valley's on other days' placings (seeds from :data:`ZIARNO`); a search is
flown on seeds below it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..platformy import Kamera
from . import kamera as K
from . import obiekty as O
from . import teren as T
from . import wykrywanie as W

#: Lessons are flown over areas from this seed on; searches below it.
ZIARNO = 1000
DATA = Path(__file__).resolve().parent.parent / "data"


def nagraj(
    kamera: Kamera, ujecia: int = 1500, wysokosci=(12.0, 110.0), seed: int = 0, teren=None
):
    """(measures, is it a person, what it was) of every warm spot in ``ujecia`` pictures."""
    rng = np.random.default_rng(seed + ZIARNO)
    t = teren or T.dolina()
    pg = T.Pogoda()
    gt = t.temperatura(pg)
    feats, people, kinds = [], [], []
    area = None
    for i in range(ujecia):
        if i % 20 == 0:
            area = O.uloz(t, pg, seed=ZIARNO + seed * 1000 + i)
        if rng.uniform() < 0.75:
            o = area[rng.integers(len(area))]
            if rng.uniform() < 0.5:
                o = [a for a in area if a.czlowiek][0]
            x, y = o.x + rng.uniform(-15, 15), o.y + rng.uniform(-15, 15)
        else:
            x, y = rng.uniform(-500, 500, 2)
        h = rng.uniform(*wysokosci)
        z = float(t.wysokosc(x, y)) + h
        widok = K.zdjecie(t, gt, area, kamera, (x, y, z), rng.uniform(-np.pi, np.pi), rng=rng)
        for p in W.plamy(widok):
            best, dist = None, np.inf
            for o in area:
                d = np.hypot(p.gdzie[0] - o.x, p.gdzie[1] - o.y)
                if d < dist:
                    best, dist = o, d
            hit = best is not None and dist < best.zasieg() + max(1.0, 2 * p.cechy[5])
            feats.append(p.cechy)
            people.append(bool(hit and best.czlowiek))
            kinds.append(best.rodzaj if hit else "tlo")
    return np.array(feats), np.array(people), np.array(kinds)


def lekcje(kamera_nazwa: str, kamera: Kamera, ujecia: int = 1500, fresh=False, teren=None):
    """A camera's lessons, flown once and kept in ``data/teren_<camera>.npz``."""
    path = DATA / f"teren_{kamera_nazwa}.npz"
    if path.exists() and not fresh:
        with np.load(path) as d:
            return d["cechy"], d["ludzie"], d["rodzaje"]
    f, p, k = nagraj(kamera, ujecia, teren=teren)
    np.savez_compressed(path, cechy=f, ludzie=p, rodzaje=k)
    return f, p, k
