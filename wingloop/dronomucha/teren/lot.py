"""The two platforms flying a search of a square kilometre, and coming lower to look again.

Both fly lanes north-south, along the valley and so mostly along the
slopes' contours, spaced so the camera's strips touch; both hold a height
above the ground ahead of them (they look ahead along their way on the
terrain map, as ArduPilot's terrain following does, and climb no faster
than they can); both take a picture a second and judge every warm spot in
it (:mod:`.wykrywanie`). What the brain likes is a candidate, and they go
to look again from lower down:

- the **quadcopter** flies to it, comes down to :data:`NIZEJ` ``["quad"]`` m,
  hovers and looks three times; it flies the lanes nearest the trail first
  (where people are lost most) -- it cannot fly them all on one battery;
- the **plane** cannot stop: it flies over it at :data:`NIZEJ`
  ``["samolot"]`` m, looking four times a second while it is close, and
  comes round again if it missed; it flies every other lane or more, so
  that it can turn into the next (it turns no tighter than its bank
  allows).

What the closer look likes is reported -- where, for the team; what it does
not is not looked at again; what it had no time to look at again is
reported as to be checked, for the operator. The lanes end at four fifths of
the flight; it goes on looking again at what it found until 92%.
Kinematics only: a body that flies where it is pointed, as fast as it
can, turning and climbing no faster than it can -- no wind, no physics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..platformy import Kamera, Platforma
from . import kamera as K
from . import wykrywanie as W

#: How low it comes to look again, m above the ground.
NIZEJ = {"quad": 15.0, "samolot": 35.0}
#: How fast each climbs and comes down, m/s.
WZNOSZENIE = {"quad": 3.0, "samolot": 5.0}
OPADANIE = {"quad": 2.5, "samolot": 5.0}
#: How far along its way each looks at the ground ahead to hold its height, m, and the
#: least height over the ground it is let down to, m.
PRZED = {"quad": 60.0, "samolot": 800.0}
NAJNIZEJ = {"quad": 5.0, "samolot": 25.0}
#: The area searched: this many metres each way of its middle.
POLE = 500.0
G = 9.81


@dataclass
class Kandydat:
    """A warm spot the brain liked: where, how much, and what became of it."""

    gdzie: np.ndarray
    ocena: float
    t: float
    stan: str = "czeka"  # "czeka", "potwierdzony", "odrzucony"
    oceny: list = field(default_factory=list)
    przeloty: int = 0


@dataclass
class Wynik:
    """A search: the path (t, x, y, z, height above ground, phase), every candidate, how
    much of the area was seen, the pictures taken."""

    slad: list
    kandydaci: list
    pokrycie: np.ndarray
    zdjecia: int
    czas: float


class Lot:
    """A platform in the air: where it is, which way it heads, how fast it goes."""

    def __init__(self, platforma: Platforma, teren, start, agl: float):
        self.p, self.t = platforma, teren
        x, y = start
        self.x, self.y = float(x), float(y)
        self.z = float(teren.wysokosc(x, y)) + 2.0
        self.kurs = 0.0
        self.v = 0.0 if platforma.zawis else platforma.predkosc
        self.agl = agl

    def wysokosc_potrzebna(self, agl: float, ile: float, ku: float) -> float:
        """How high it must be now to keep ``agl`` m over the ground for the next ``ile`` m
        of its way (bearing ``ku``, radians), climbing no faster than it can: the
        highest ground ahead, less what it can climb before it gets there."""
        d = np.linspace(0, ile, 32)
        xs = self.x + d * np.cos(ku)
        ys = self.y + d * np.sin(ku)
        climb = WZNOSZENIE[self.p.nazwa] * d / max(self.v, 1.0)
        return float(np.max(self.t.wysokosc(xs, ys) + agl - climb))

    def krok(self, dt: float, cel, agl: float, stop: bool = False) -> float:
        """Fly ``dt`` s toward ``cel`` (x, y) at ``agl`` m above the ground; return the distance
        left. ``stop``: a quadcopter slows to a hover on it."""
        dx, dy = cel[0] - self.x, cel[1] - self.y
        dist = float(np.hypot(dx, dy))
        want = np.arctan2(dy, dx)
        turn = (want - self.kurs + np.pi) % (2 * np.pi) - np.pi
        if self.p.zawis:
            self.kurs += np.clip(turn, -np.pi / 2 * dt, np.pi / 2 * dt)
            target_v = self.p.predkosc if not stop else min(self.p.predkosc, 0.8 * dist)
            if abs(turn) > np.pi / 3:
                target_v = min(target_v, 1.0)
            ahead = 5.0 if stop and dist < 10.0 else PRZED["quad"]
        else:
            rate = G * np.tan(np.deg2rad(35.0)) / self.v
            self.kurs += np.clip(turn, -rate * dt, rate * dt)
            target_v = self.v
            ahead = PRZED["samolot"]
        # Along its way to where it is going, not along its nose: turning at a
        # lane's end it points at the valley's wall, and climbed for it.
        z_want = self.wysokosc_potrzebna(agl, min(ahead, max(dist, 5.0)), want)
        if self.p.zawis and z_want - self.z > 5.0:
            target_v = min(target_v, 2.0)  # a quadcopter slows to climb
        if self.p.zawis:
            self.v += np.clip(target_v - self.v, -3.0 * dt, 3.0 * dt)
        self.x += self.v * np.cos(self.kurs) * dt
        self.y += self.v * np.sin(self.kurs) * dt
        up, down = WZNOSZENIE[self.p.nazwa], OPADANIE[self.p.nazwa]
        self.z += np.clip(z_want - self.z, -down * dt, up * dt)
        here = float(self.t.wysokosc(self.x, self.y))
        # It does not fly into the ground: a pilot, or the autopilot's floor, pulls it up.
        self.z = max(self.z, here + NAJNIZEJ[self.p.nazwa])
        return float(np.hypot(cel[0] - self.x, cel[1] - self.y))

    @property
    def nad_ziemia(self) -> float:
        return self.z - float(self.t.wysokosc(self.x, self.y))


def pasy(platforma: Platforma, kamera: Kamera, agl: float, szlak_x: float = 60.0) -> list:
    """The lanes' ends, in the order flown: [(x, y from), (x, y to)], ... ."""
    from ..platformy import UZYTECZNY_PAS

    spacing = kamera.pas(agl) * UZYTECZNY_PAS
    n = int(np.ceil(2 * POLE / spacing))
    xs = -POLE + spacing * (np.arange(n) + 0.5)
    if platforma.zawis:
        order = list(np.argsort(np.abs(xs - szlak_x), kind="stable"))
    else:
        radius = platforma.predkosc**2 / (G * np.tan(np.deg2rad(35.0)))
        k = max(1, int(np.ceil(2 * radius / spacing)))
        order, done = [], set()
        for start in range(k + 1):
            for i in range(start, n, k + 1):
                if i not in done:
                    order.append(i)
                    done.add(i)
    lanes = []
    for j, i in enumerate(order):
        ends = [(-POLE - 20.0), (POLE + 20.0)]
        if j % 2:
            ends.reverse()
        lanes.append([(xs[i], ends[0]), (xs[i], ends[1])])
    return lanes


def _w_kadrze(rogi, gx, gy) -> np.ndarray:
    """Which points (gx, gy) lie inside a picture's footprint on the ground: a convex
    quadrilateral, either way round -- every edge has them on the same side."""
    sides = []
    for i in range(4):
        (x0, y0), (x1, y1) = rogi[i][:2], rogi[(i + 1) % 4][:2]
        sides.append((x1 - x0) * (gy - y0) - (y1 - y0) * (gx - x0))
    sides = np.stack(sides)
    return (sides >= 0).all(axis=0) | (sides <= 0).all(axis=0)


def szukaj(
    platforma: Platforma,
    kamera: Kamera,
    teren,
    ground_temp,
    obiekty,
    mozg: W.Mozg,
    agl: float | None = None,
    potwierdzaj: bool = True,
    dt: float = 0.2,
    seed: int = 0,
    klatka=None,
    krok=None,
) -> Wynik:
    """Fly one search of the area; ``klatka(lot, widok, plamy, faza)`` sees every picture,
    ``krok(t, lot, faza, cel, kandydaci)`` every step -- and ends it by returning True."""
    rng = np.random.default_rng(seed)
    agl = agl or platforma.wysokosc
    low = NIZEJ[platforma.nazwa]
    lot = Lot(platforma, teren, teren.baza, agl)
    lanes = [p for lane in pasy(platforma, kamera, agl) for p in lane]
    cells = np.arange(-POLE + 5, POLE, 10.0)
    gx, gy = np.meshgrid(cells, cells)
    seen = np.zeros(gx.shape, bool)
    candidates: list[Kandydat] = []
    path, shots = [], 0
    limit = platforma.zapas() * 60.0
    # The lanes end at ``limit``; what is still to be looked at again, until ``late``.
    late = 0.92 * platforma.lot_min * 60.0
    merge = 8.0 if platforma.zawis else 15.0
    t, next_shot, wp = 0.0, 0.0, 0
    phase, target, resume = "pasy", None, None

    def look(where_from_phase):
        nonlocal shots
        shots += 1
        v = K.zdjecie(
            teren, ground_temp, obiekty, kamera, (lot.x, lot.y, lot.z), lot.kurs, rng=rng
        )
        spots = W.plamy(v)
        if klatka is not None:
            klatka(lot, v, spots, where_from_phase)
        return v, spots

    while t < limit + 900.0:
        if phase == "pasy":
            if wp >= len(lanes) or t > limit:
                pending = [c for c in candidates if c.stan == "czeka"]
                if pending and potwierdzaj and t < late:
                    target, resume, phase = pending[0], teren.baza, "potwierdz"
                else:
                    phase = "powrot"
                continue
            if lot.krok(dt, lanes[wp], agl) < (8.0 if platforma.zawis else 30.0):
                wp += 1
            if t >= next_shot and lot.nad_ziemia > 0.5 * agl:
                next_shot = t + 1.0
                v, spots = look("pasy")
                seen |= _w_kadrze(v.rogi, gx, gy)
                for s in spots:
                    val = W.Mozg.ocena(mozg, s)
                    near = [
                        c for c in candidates if np.hypot(*(c.gdzie[:2] - s.gdzie[:2])) < merge
                    ]
                    if near or val <= 0:
                        continue
                    c = Kandydat(s.gdzie, val, t)
                    candidates.append(c)
                    if not potwierdzaj:
                        c.stan = "potwierdzony"
                    elif target is None:
                        target, resume, phase = c, (lot.x, lot.y), "potwierdz"
        elif phase == "potwierdz":
            gx_, gy_ = target.gdzie[:2]
            left = lot.krok(dt, (gx_, gy_), low, stop=platforma.zawis)
            near = 4.0 if platforma.zawis else 25.0
            ready = (
                abs(lot.nad_ziemia - low) < 3.0
                if platforma.zawis
                else lot.nad_ziemia < low + 25
            )

            if (
                left < near
                and ready
                and t >= next_shot
                and (lot.v < 1.5 or not platforma.zawis)
            ):
                next_shot = t + (0.5 if platforma.zawis else 0.25)
                v, spots = look("potwierdz")
                close = [s for s in spots if np.hypot(*(s.gdzie[:2] - target.gdzie[:2])) < 10.0]
                if close:
                    best = min(close, key=lambda s: np.hypot(*(s.gdzie[:2] - target.gdzie[:2])))
                    target.oceny.append(W.Mozg.ocena(mozg, best))
                else:
                    target.oceny.append(-1.0)
            done_quad = platforma.zawis and len(target.oceny) >= 3
            passed = not platforma.zawis and target.oceny and left > 30.0
            if not platforma.zawis and left > 30.0 and not target.oceny and t - target.t > 90:
                passed = True  # it could not get over it
            if done_quad or passed:
                good = [o for o in target.oceny if o > -1.0]
                target.stan = "potwierdzony" if good and np.mean(good) > 0 else "odrzucony"
                pending = [c for c in candidates if c.stan == "czeka"]
                if pending and t < late:
                    target = pending[0]
                else:
                    target, phase = None, "wroc"
            if t > late:
                phase = "powrot"
        elif phase == "wroc":
            if lot.krok(dt, resume, agl) < (8.0 if platforma.zawis else 40.0):
                phase = "pasy"
        elif phase == "powrot":
            if lot.krok(dt, teren.baza, agl, stop=platforma.zawis) < 15.0:
                break
        path.append((t, lot.x, lot.y, lot.z, lot.nad_ziemia, phase))
        t += dt
        if krok is not None and krok(t, lot, phase, target, candidates):
            break
    return Wynik(path, candidates, seen, shots, t)


def ocen(wynik: Wynik, obiekty, promien: float = 8.0) -> dict:
    """How the search went: people found of those there, false alarms, when the first was
    found, how much of the area was seen."""
    people = [o for o in obiekty if o.czlowiek]
    reports = [c for c in wynik.kandydaci if c.stan == "potwierdzony"]
    found, wrong, first = set(), 0, None
    for c in reports:
        d = [np.hypot(c.gdzie[0] - o.x, c.gdzie[1] - o.y) for o in people]
        if d and min(d) < promien:
            found.add(int(np.argmin(d)))
            first = c.t if first is None else min(first, c.t)
        else:
            wrong += 1
    unsure = [c for c in wynik.kandydaci if c.stan == "czeka"]
    near_person = sum(
        any(np.hypot(c.gdzie[0] - o.x, c.gdzie[1] - o.y) < promien for o in people)
        for c in unsure
    )
    return dict(
        do_sprawdzenia=len(unsure),
        do_sprawdzenia_ludzie=near_person,
        znalezieni=len(found),
        ludzie=len(people),
        falszywe=wrong,
        pierwszy=first,
        pokrycie=float(wynik.pokrycie.mean()),
        kandydaci=len(wynik.kandydaci),
        czas=wynik.czas,
    )
