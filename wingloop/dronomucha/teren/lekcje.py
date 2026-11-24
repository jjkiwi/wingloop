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

Then the drone flies searches of its own over those valleys (:func:`loty`), with the
brain the pictures taught, and every spot it judged -- going over and coming down to
look -- is labelled the same way. These teach the compartment that confirms
(:attr:`~.wykrywanie.Mozg.z_bliska`): they hold the boulders, chamois and foxes the
brain took for people and flew down to, which pictures picked at random hardly ever do.
:func:`mozg` is the brain taught on both.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..platformy import Kamera
from . import kamera as K
from . import lot as LOT
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
#: The searches flown for :func:`loty`: the first two in three by the plane, the rest by
#: the tiltrotor, both from 60 m.
LOTY = 90
SAMOLOTEM = 60
WYSOKOSC_LOTOW = 60.0
#: Of the spots on bare ground seen in those searches, the share kept: one in ten -- a
#: search sees bare ground nearly all the time, and every spot of it kept would bury the
#: few hundred close looks at boulders and people.
TLO_Z_LOTOW = 0.1
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


class _Zapis:
    """A brain that notes every spot it is asked about: (close look?, where, measures)."""

    def __init__(self, mozg, log, z_bliska=False):
        self.mozg, self.log, self.blisko = mozg, log, z_bliska
        if not z_bliska:
            close = getattr(mozg, "z_bliska", None)
            self.z_bliska = _Zapis(close, log, True) if close is not None else None

    def ocena(self, plama) -> float:
        self.log.append((self.blisko, plama.gdzie.copy(), plama.cechy.copy()))
        return self.mozg.ocena(plama)


def _lot(kamera_nazwa: str, seed: int, doliny=DOLINY):
    """(measures, is it a person, what it was, close look?) of every spot judged in one
    search of :func:`loty` -- over a lesson area, on a lesson day."""
    from ..platformy import KAMERY, PLATFORMY

    kamera = KAMERY[kamera_nazwa]
    f, p, _ = lekcje(kamera_nazwa, kamera, progress=None, doliny=doliny)
    rng = np.random.default_rng(5000 + seed)
    t = T.dolina(seed=doliny[seed % len(doliny)])
    pg = pogoda_lekcji(rng)
    area = O.uloz(t, pg, seed=ZIARNO + 500 + seed)
    gt = t.temperatura(pg, seed=seed)
    log = []
    plat = PLATFORMY["samolot" if seed < SAMOLOTEM else "vtol"]
    brain = _Zapis(W.naucz(f, p), log)
    LOT.szukaj(
        plat, kamera, t, gt, area, brain, agl=WYSOKOSC_LOTOW, seed=seed, powietrze=pg.powietrze
    )
    feats, people, kinds, close = [], [], [], []
    for blisko, gdzie, cechy in log:
        best, dist = None, np.inf
        for o in area:
            d = np.hypot(gdzie[0] - o.x, gdzie[1] - o.y)
            if d < dist:
                best, dist = o, d
        hit = best is not None and dist < best.zasieg() + max(1.0, 2 * cechy[5])
        feats.append(cechy)
        people.append(bool(hit and best.czlowiek))
        kinds.append(best.rodzaj if hit else "tlo")
        close.append(blisko)
    return np.array(feats), np.array(people, bool), np.array(kinds), np.array(close, bool)


def loty(
    kamera_nazwa: str,
    kamera: Kamera,
    ile: int = LOTY,
    fresh=False,
    progress=print,
    doliny=DOLINY,
    procesy: int | None = None,
):
    """(measures, is it a person, what it was, close look?) of the spots judged in ``ile``
    searches flown over the lesson valleys with the brain the pictures taught, bare ground
    thinned to :data:`TLO_Z_LOTOW` -- flown once and kept in ``data/teren_<camera>_loty.npz``.
    The searches are flown ``procesy`` at a time (all the computer's cores by default):
    for 90 of them that is still the best part of an hour."""
    path = DATA / f"teren_{kamera_nazwa}_loty.npz"
    key = "|".join(
        [
            _klucz(kamera, UJECIA, doliny),
            str(ile),
            str(SAMOLOTEM),
            str(WYSOKOSC_LOTOW),
            str(TLO_Z_LOTOW),
            str(W.BLISKO),
        ]
    )
    if path.exists() and not fresh:
        with np.load(path) as d:
            if "klucz" in d and str(d["klucz"]) == key:
                return d["cechy"], d["ludzie"], d["rodzaje"], d["z_bliska"]
        if progress:
            progress(f"Loty {path.name} są z innej wersji symulacji: latam od nowa.")
    # The pictures first, once: else every search of the pool would fly them.
    lekcje(kamera_nazwa, kamera, progress=progress, doliny=doliny)
    if progress:
        progress(f"Latam {ile} poszukiwań na lekcje dla kamery {kamera_nazwa} ...")
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    one = partial(_lot, kamera_nazwa, doliny=tuple(doliny))
    if procesy == 1:
        parts = list(map(one, range(ile)))
    else:
        with ProcessPoolExecutor(procesy) as pool:
            parts = list(pool.map(one, range(ile)))
    f, p, k, c = (np.concatenate([part[i] for part in parts]) for i in range(4))
    f = f.reshape(-1, len(W.CECHY))
    keep = (k != "tlo") | (np.random.default_rng(0).uniform(size=len(k)) < TLO_Z_LOTOW)
    f, p, k, c = f[keep], p[keep], k[keep], c[keep]
    np.savez_compressed(path, cechy=f, ludzie=p, rodzaje=k, z_bliska=c, klucz=key)
    return f, p, k, c


def mozg(kamera_nazwa: str, kamera: Kamera, progress=print) -> W.Mozg:
    """The brain a search flies with: taught on the camera's pictures (:func:`lekcje`), its
    confirming compartment on its own searches too (:func:`loty`)."""
    f, p, _ = lekcje(kamera_nazwa, kamera, progress=progress)
    fl, pl, _, _ = loty(kamera_nazwa, kamera, progress=progress)
    return W.naucz(f, p, z_lotow=(fl, pl))
