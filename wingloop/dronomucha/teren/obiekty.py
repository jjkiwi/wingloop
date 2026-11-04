"""What lies on the ground: people, animals and boulders, as parts.

Each model is a list of parts -- capsules (``a`` to ``b``, radius ``r``) and
ellipsoids (centre ``c``, radii ``r``) in metres, in the object's own frame
(x along it, y to its left, z up, the ground at z = 0) -- each with a colour
for a person watching and a surface temperature for a thermal camera. The
same parts are drawn in 3-D (:mod:`.film`) and seen from above by the
thermal camera (:mod:`.kamera`): what the camera sees of a person is the
parts of them that are warm, where they are -- a face at 30 C, a jacket
some 10 K above the air, boots barely warmer than the snow -- not a box of
one temperature.

Kinds: a person lying on their back, lying on their side, sitting and
standing; a chamois; a fox; a boulder warmed by the sun. A person may be
partly under snow: those parts are not there for the camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import teren as T


@dataclass
class Czesc:
    """One part: ``typ`` ``"kapsula"`` (from ``a`` to ``b``, radius ``r``) or ``"elipsoida"``
    (centre ``a``, radii ``r``); its colour and surface temperature, C."""

    typ: str
    a: tuple
    b: tuple | None
    r: object
    kolor: tuple
    temp: float
    pod_sniegiem: bool = False


@dataclass
class Obiekt:
    """An object on the ground: its kind, where (x, y, the ground's z), which way it faces
    (radians), and its parts."""

    rodzaj: str
    x: float
    y: float
    z: float
    yaw: float
    czesci: list = field(default_factory=list)

    @property
    def czlowiek(self) -> bool:
        return self.rodzaj.startswith("czlowiek")

    def w_swiecie(self, p) -> np.ndarray:
        """A point of the object's frame in the world's."""
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        p = np.asarray(p, dtype=float)
        return np.array(
            [self.x + c * p[0] - s * p[1], self.y + s * p[0] + c * p[1], self.z + p[2]]
        )

    def zasieg(self) -> float:
        """How far from (x, y) any part reaches, m."""
        far = 0.0
        for c in self.czesci:
            for p in (c.a, c.b) if c.b is not None else (c.a,):
                far = max(far, float(np.hypot(p[0], p[1])) + float(np.max(c.r)))
        return far


SKORA = (0.86, 0.66, 0.52)
KURTKI = (
    (0.85, 0.15, 0.1),
    (0.95, 0.5, 0.05),
    (0.1, 0.35, 0.8),
    (0.95, 0.8, 0.1),
    (0.2, 0.55, 0.3),
)


def _kaps(a, b, r, kolor, temp):
    return Czesc("kapsula", tuple(a), tuple(b), float(r), kolor, float(temp))


def _elip(c, r, kolor, temp):
    return Czesc("elipsoida", tuple(c), None, tuple(r), kolor, float(temp))


def czlowiek(poza: str, powietrze: float, rng) -> list:
    """A person in a pose: ``"lezy"`` (on the back), ``"bok"`` (on the side, knees bent),
    ``"siedzi"`` or ``"stoi"``. Their clothes' surfaces are as warm as clothing
    over a living body is in the cold; someone cold for hours is cooler."""
    jacket = KURTKI[rng.integers(len(KURTKI))]
    trousers = (0.15, 0.15, 0.18)
    boots = (0.25, 0.15, 0.08)
    hat = KURTKI[rng.integers(len(KURTKI))]
    cold = rng.uniform(0.6, 1.0)  # 1: warm; less: hours in the cold
    a = powietrze
    face = a + (32.0 - a) * cold
    tj = a + rng.uniform(9, 14) * cold
    tt = a + rng.uniform(6, 9) * cold
    tb = a + 3.0 * cold
    th = a + 8.0 * cold
    p = []
    if poza == "lezy":
        z = 0.14
        p += [_kaps((0.05, 0, z), (0.55, 0, z), 0.17, jacket, tj)]
        p += [_elip((0.78, 0, 0.11), (0.11, 0.09, 0.1), SKORA, face)]
        p += [_elip((0.82, 0, 0.13), (0.07, 0.095, 0.07), hat, th)]
        for s in (1, -1):
            p += [_kaps((0.5, 0.25 * s, 0.1), (0.1, 0.3 * s, 0.08), 0.055, jacket, tj)]
            p += [_kaps((0.1, 0.3 * s, 0.08), (-0.18, 0.28 * s, 0.07), 0.045, jacket, tj)]
            p += [_kaps((0.0, 0.1 * s, 0.1), (-0.45, 0.12 * s, 0.09), 0.085, trousers, tt)]
            p += [_kaps((-0.45, 0.12 * s, 0.09), (-0.88, 0.14 * s, 0.07), 0.06, trousers, tt)]
            p += [_kaps((-0.9, 0.14 * s, 0.07), (-0.95, 0.15 * s, 0.17), 0.055, boots, tb)]
    elif poza == "bok":
        p += [_kaps((0.05, 0, 0.2), (0.5, 0, 0.2), 0.16, jacket, tj)]
        p += [_elip((0.72, 0.02, 0.18), (0.11, 0.1, 0.1), SKORA, face)]
        p += [_kaps((0.45, 0.05, 0.3), (0.65, 0.3, 0.1), 0.05, jacket, tj)]
        p += [_kaps((0.45, -0.05, 0.1), (0.3, -0.3, 0.06), 0.05, jacket, tj)]
        p += [_kaps((0.0, 0, 0.2), (-0.25, 0.35, 0.15), 0.085, trousers, tt)]
        p += [_kaps((-0.25, 0.35, 0.15), (-0.65, 0.25, 0.1), 0.06, trousers, tt)]
        p += [_kaps((0.0, -0.05, 0.1), (-0.42, -0.05, 0.08), 0.085, trousers, tt)]
        p += [_kaps((-0.42, -0.05, 0.08), (-0.82, -0.1, 0.07), 0.06, trousers, tt)]
        p += [_kaps((-0.65, 0.25, 0.1), (-0.75, 0.28, 0.1), 0.055, boots, tb)]
    elif poza == "siedzi":
        p += [_kaps((0, 0, 0.3), (-0.05, 0, 0.75), 0.17, jacket, tj)]
        p += [_elip((-0.03, 0, 0.98), (0.1, 0.09, 0.11), SKORA, face)]
        p += [_elip((-0.04, 0, 1.06), (0.1, 0.095, 0.06), hat, th)]
        p += [
            _elip((-0.25, 0, 0.55), (0.12, 0.17, 0.22), (0.2, 0.2, 0.25), a + 1.0)
        ]  # backpack
        for s in (1, -1):
            p += [_kaps((0, 0.11 * s, 0.25), (0.45, 0.12 * s, 0.28), 0.085, trousers, tt)]
            p += [_kaps((0.45, 0.12 * s, 0.28), (0.5, 0.13 * s, 0.05), 0.06, trousers, tt)]
            p += [_kaps((-0.02, 0.22 * s, 0.68), (0.25, 0.15 * s, 0.4), 0.05, jacket, tj)]
    else:  # "stoi"
        p += [_kaps((0, 0, 0.95), (0, 0, 1.4), 0.17, jacket, tj)]
        p += [_elip((0.02, 0, 1.63), (0.1, 0.09, 0.12), SKORA, face)]
        p += [_elip((0.02, 0, 1.72), (0.1, 0.095, 0.06), hat, th)]
        p += [_elip((-0.22, 0, 1.2), (0.11, 0.17, 0.24), (0.2, 0.2, 0.25), a + 1.0)]
        for s in (1, -1):
            p += [_kaps((0, 0.1 * s, 0.9), (0.02, 0.11 * s, 0.05), 0.075, trousers, tt)]
            p += [_kaps((0, 0.23 * s, 1.35), (0.03, 0.27 * s, 0.85), 0.05, jacket, tj)]
            p += [_kaps((0.02, 0.11 * s, 0.05), (0.15, 0.11 * s, 0.04), 0.05, boots, tb)]
    return p


def kozica(powietrze: float, rng) -> list:
    """A chamois in its winter coat: the fur 6-10 K above the air, its head warmer."""
    fur = (0.28, 0.2, 0.13)
    tf = powietrze + rng.uniform(6, 10)
    p = [_elip((0, 0, 0.6), (0.55, 0.18, 0.22), fur, tf)]
    p += [_kaps((0.42, 0, 0.72), (0.6, 0, 0.92), 0.08, fur, tf)]
    p += [_elip((0.68, 0, 0.98), (0.13, 0.07, 0.08), (0.85, 0.82, 0.75), tf + 4)]
    for s in (1, -1):
        p += [_kaps((0.68, 0.03 * s, 1.04), (0.66, 0.04 * s, 1.16), 0.015, (0.1, 0.1, 0.1), tf)]
        for fx in (0.35, -0.35):
            p += [_kaps((fx, 0.1 * s, 0.5), (fx + 0.02, 0.1 * s, 0.0), 0.035, fur, tf - 4)]
    return p


def lis(powietrze: float, rng) -> list:
    """A fox: small, warm, low."""
    red = (0.8, 0.4, 0.15)
    tf = powietrze + rng.uniform(7, 11)
    p = [_elip((0, 0, 0.3), (0.3, 0.1, 0.12), red, tf)]
    p += [_elip((0.36, 0, 0.36), (0.1, 0.06, 0.07), red, tf + 3)]
    p += [_kaps((-0.28, 0, 0.3), (-0.65, 0, 0.18), 0.06, red, tf - 3)]
    for s in (1, -1):
        for fx in (0.18, -0.18):
            p += [_kaps((fx, 0.06 * s, 0.25), (fx, 0.06 * s, 0.0), 0.025, (0.2, 0.1, 0.05), tf)]
    return p


def glaz(powietrze: float, pogoda: T.Pogoda, rng) -> list:
    """A boulder, 0.5-2.5 m, dark rock lit by the sun: as warm as a jacket, or warmer."""
    rx, ry = rng.uniform(0.3, 1.25), rng.uniform(0.3, 1.0)
    rz = rng.uniform(0.25, 0.8)
    lit = rng.uniform(0.3, 1.0) * np.sin(np.deg2rad(pogoda.slonce_el) + 0.5)
    t = powietrze - 3.0 + 25.0 * lit * (1 - 0.8 * pogoda.chmury) + rng.uniform(0, 6)
    grey = rng.uniform(0.3, 0.45)
    return [_elip((0, 0, rz * 0.6), (rx, ry, rz), (grey, grey * 0.97, grey * 0.93), t)]


#: How many of each in a search area, and where they are put.
SKLAD = {"czlowiek": (1, 2), "kozica": (3, 6), "lis": (1, 3), "glaz": (12, 20)}
POZY = ("lezy", "lezy", "bok", "siedzi", "stoi")


def uloz(t: T.Teren, pogoda: T.Pogoda, seed: int, w=500.0) -> list:
    """A search area's objects, within ``w`` m of its middle.

    People: half near the trail -- where people get lost or hurt -- half
    anywhere a person could have fallen or wandered (not on cliffs or the
    tarn), a third of them partly under snow if they lie on it. Chamois on
    steep ground high up; foxes in the dwarf pine and meadows; boulders on
    scree, rock and meadow below the cliffs.
    """
    rng = np.random.default_rng(seed)
    slope = t.nachylenie()
    out = []

    def place(ok, near_trail=False):
        for _ in range(4000):
            if near_trail:
                k = rng.integers(len(t.szlak) - 1)
                f = rng.uniform()
                px, py = t.szlak[k] + f * (t.szlak[k + 1] - t.szlak[k])
                px, py = px + rng.normal(0, 30), py + rng.normal(0, 30)
            else:
                px, py = rng.uniform(-w, w, 2)
            if abs(px) > w or abs(py) > w:
                continue
            c = int(t.pole(t.pokrycie.astype(float), px, py, order=0))
            s = float(t.pole(slope, px, py))
            if ok(c, s):
                return px, py
        raise RuntimeError("no place found")

    a = pogoda.powietrze
    n_people = rng.integers(SKLAD["czlowiek"][0], SKLAD["czlowiek"][1] + 1)
    for i in range(n_people):
        xy = place(lambda c, s: c not in (T.SKALA, T.WODA) and s < 38, near_trail=i % 2 == 0)
        poza = POZY[rng.integers(len(POZY))]
        parts = czlowiek(poza, a, rng)
        c = int(t.pole(t.pokrycie.astype(float), *xy, order=0))
        if c == T.SNIEG and poza in ("lezy", "bok") and rng.uniform() < 0.33:
            for part in parts:  # the legs, and perhaps more, under snow
                if part.a[0] < -0.1 or rng.uniform() < 0.25:
                    part.pod_sniegiem = True
        out.append(Obiekt("czlowiek_" + poza, *xy, 0.0, rng.uniform(-np.pi, np.pi), parts))
    for _ in range(rng.integers(*SKLAD["kozica"])):
        xy = place(lambda c, s: c in (T.SNIEG, T.PIARG, T.SKALA, T.TRAWA) and 20 < s < 45)
        out.append(Obiekt("kozica", *xy, 0.0, rng.uniform(-np.pi, np.pi), kozica(a, rng)))
    for _ in range(rng.integers(*SKLAD["lis"])):
        xy = place(lambda c, s: c in (T.KOSO, T.TRAWA, T.SZLAK) and s < 30)
        out.append(Obiekt("lis", *xy, 0.0, rng.uniform(-np.pi, np.pi), lis(a, rng)))
    for _ in range(rng.integers(*SKLAD["glaz"])):
        xy = place(lambda c, s: c in (T.PIARG, T.SKALA, T.TRAWA, T.SNIEG) and s < 42)
        out.append(Obiekt("glaz", *xy, 0.0, rng.uniform(-np.pi, np.pi), glaz(a, pogoda, rng)))
    for o in out:
        o.z = float(t.wysokosc(o.x, o.y))
    return out
