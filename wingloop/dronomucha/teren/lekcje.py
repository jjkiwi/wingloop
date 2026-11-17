"""Lessons for the fly's brain at real scale: warm spots seen from the air, and what they were.

The drone, its camera looking down, is shown search areas it will not
search: from 12-110 m up -- the heights it searches from and comes down to
-- over people, chamois, foxes and sun-warmed boulders, and over bare
ground. Each warm spot found (:func:`~.wykrywanie.plamy`)
is a lesson, labelled by what really lies there -- a person, or not -- as
the dopamine of a real find would label it. The lessons are flown in other
valleys than the one searched (:data:`DOLINY`), on other days
(:func:`pogoda_lekcji`: a hard frost to a hot summer afternoon, clear to overcast): what the
brain is then tried on, it has not seen.
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
#: The valleys the lessons are flown in (:func:`~.teren.dolina`'s seeds): not the one
#: searched, which is 0.
DOLINY = (1, 2, 3)
#: Raised whenever the ground, the objects, the camera or the measures change what a
#: lesson holds: kept lessons of another version are flown again.
WERSJA = 4
#: How many pictures the lessons are: with days from frost to heat, 1500 held too few
#: people for the hot ones (docs/teren.md).
UJECIA = 3000
#: How many of the lesson pictures are taken over an object, the rest anywhere. A search
#: sees bare ground nearly all the time: with three in four over objects, the brain taught
#: on hot days too took sun-warmed ground for people and spent the flight checking it.
NAD_OBIEKTEM = 0.5
DATA = Path(__file__).resolve().parent.parent / "data"


def pogoda_lekcji(rng) -> T.Pogoda:
    """A day for a lesson: between a hard frost and a hot summer afternoon, the sun low or
    high, clear or overcast. The brain is to know a person on any of them -- and on a hot
    day a sunlit boulder is warmer than a person."""
    return T.Pogoda(
        powietrze=float(rng.uniform(-12.0, 25.0)),
        slonce_az=float(rng.uniform(100.0, 260.0)),
        slonce_el=float(rng.uniform(8.0, 58.0)),
        chmury=float(rng.uniform(0.0, 0.9)),
    )


def nagraj(
    kamera: Kamera,
    ujecia: int = 1500,
    wysokosci=(12.0, 110.0),
    seed: int = 0,
    doliny=DOLINY,
    nad_obiektem: float = NAD_OBIEKTEM,
):
    """(measures, is it a person, what it was) of every warm spot in ``ujecia`` pictures.

    Every twenty pictures another placing of people, animals and boulders,
    in the next of the valleys ``doliny``, on another day
    (:func:`pogoda_lekcji`). ``nad_obiektem`` of the pictures are taken over
    an object -- a person, a chamois, a fox or a boulder, each as often -- the
    rest anywhere; two in five from 12-40 m, the heights of a closer look,
    the rest from up to 110 m.
    """
    rng = np.random.default_rng(seed + ZIARNO)
    valleys = [T.dolina(seed=s) for s in doliny]
    feats, people, kinds = [], [], []
    t = pg = gt = area = None
    for i in range(ujecia):
        if i % 20 == 0:
            t = valleys[(i // 20) % len(valleys)]
            pg = pogoda_lekcji(rng)
            gt = t.temperatura(pg, seed=i)
            area = O.uloz(t, pg, seed=ZIARNO + seed * 1000 + i)
        if rng.uniform() < nad_obiektem:
            # Over one of a kind, each kind as often: by their numbers three
            # pictures in four would be of boulders and one in fifty of a fox
            # -- and a fox is what a person is taken for.
            kind = ("czlowiek", "kozica", "lis", "glaz")[rng.integers(4)]
            some = [a for a in area if a.rodzaj.startswith(kind)]
            o = some[rng.integers(len(some))]
            x, y = o.x + rng.uniform(-15, 15), o.y + rng.uniform(-15, 15)
        else:
            x, y = rng.uniform(-500, 500, 2)
        # Two pictures in five from as low as it comes to look again: that
        # look is what a report rests on.
        low = min(wysokosci[0] + 28.0, wysokosci[1])
        h = (
            rng.uniform(wysokosci[0], low)
            if rng.uniform() < 0.4
            else rng.uniform(low, wysokosci[1])
        )
        z = float(t.wysokosc(x, y)) + h
        widok = K.zdjecie(t, gt, area, kamera, (x, y, z), rng.uniform(-np.pi, np.pi), rng=rng)
        for p in W.plamy(widok, powietrze=pg.powietrze):
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


def _klucz(kamera: Kamera, ujecia: int, doliny=DOLINY) -> str:
    """What a kept file of lessons was flown with: another, and it is flown again."""
    what = (
        WERSJA,
        kamera.px,
        kamera.fov,
        kamera.netd,
        W.PROG,
        W.BLOK,
        tuple(doliny),
        ujecia,
        NAD_OBIEKTEM,
    )
    return "|".join(str(v) for v in what)


def lekcje(
    kamera_nazwa: str,
    kamera: Kamera,
    ujecia: int = UJECIA,
    fresh=False,
    progress=print,
    doliny=DOLINY,
):
    """A camera's lessons, flown once and kept in ``data/teren_<camera>.npz`` -- and flown
    again when what they were flown with has changed since (:data:`WERSJA`)."""
    path = DATA / f"teren_{kamera_nazwa}.npz"
    key = _klucz(kamera, ujecia, doliny)
    if path.exists() and not fresh:
        with np.load(path) as d:
            if "klucz" in d and str(d["klucz"]) == key:
                return d["cechy"], d["ludzie"], d["rodzaje"]
        if progress:
            progress(f"Lekcje {path.name} są z innej wersji symulacji: nagrywam od nowa.")
    f, p, k = nagraj(kamera, ujecia, doliny=doliny, nad_obiektem=NAD_OBIEKTEM)
    np.savez_compressed(path, cechy=f, ludzie=p, rodzaje=k, klucz=key)
    return f, p, k
