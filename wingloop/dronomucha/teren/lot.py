"""The two platforms flying a search of a square kilometre, and coming lower to look again.

The plane flies lanes along the contours (:func:`poziomice`): the ground
below a lane stays at one height, so it holds its height over it without
climbing and coming down as a lane across a steep slope makes it. The
quadcopter flies lanes north-south (:func:`pasy`), those nearest the trail
first. The lanes lie so the camera's strips touch; both hold a height
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
  comes round again if it missed; it goes on to a lane it can turn into
  (it turns no tighter than its bank allows).

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

    def wysokosc_potrzebna(self, agl: float, ile: float, ku: float, droga=None) -> float:
        """How high it must be now to keep ``agl`` m over the ground for the next ``ile`` m
        of its way (bearing ``ku``, radians -- or along ``droga``, the points it will
        fly through), climbing no faster than it can: the highest ground ahead, less
        what it can climb before it gets there."""
        if droga is None:
            d = np.linspace(0, ile, 32)
            xs = self.x + d * np.cos(ku)
            ys = self.y + d * np.sin(ku)
        else:
            xs, ys, d = _po_drodze((self.x, self.y), droga, ile)
        climb = WZNOSZENIE[self.p.nazwa] * d / max(self.v, 1.0)
        return float(np.max(self.t.wysokosc(xs, ys) + agl - climb))

    def krok(self, dt: float, cel, agl: float, stop: bool = False, droga=None) -> float:
        """Fly ``dt`` s toward ``cel`` (x, y) at ``agl`` m above the ground; return the distance
        left. ``stop``: a quadcopter slows to a hover on it. ``droga``: the points it
        flies through after ``cel`` -- it holds its height along them, not along a line."""
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
        if droga is None:
            z_want = self.wysokosc_potrzebna(agl, min(ahead, max(dist, 5.0)), want)
        else:
            z_want = self.wysokosc_potrzebna(agl, ahead, want, [cel, *droga])
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


def _po_drodze(start, droga, ile: float, co: float = 25.0):
    """Points every ``co`` m along the way from ``start`` through ``droga``, for ``ile`` m:
    (x, y, distance along the way)."""
    pts = np.vstack([np.asarray(start, float)[None, :2], np.asarray(droga, float)[:, :2]])
    seg = np.hypot(*np.diff(pts, axis=0).T)
    along = np.concatenate([[0.0], np.cumsum(seg)])
    d = np.arange(0.0, min(ile, along[-1]) + 1e-6, co)
    return np.interp(d, along, pts[:, 0]), np.interp(d, along, pts[:, 1]), d


def poziomice(
    platforma: Platforma, kamera: Kamera, agl: float, teren, start=None, co: float = 40.0
) -> list:
    """Lanes along the contours, in the order flown: each a list of points (x, y).

    On a lane the ground below stays at one height, so the plane holds its height
    over it without climbing and coming down as a lane across the slopes makes it.
    The contours are of the ground smoothed over 40 m -- a plane turns no tighter
    than its bank lets it, and wiggles smaller than that it cannot fly. Their
    heights are chosen so the lanes lie :data:`~..platformy.UZYTECZNY_PAS` of the
    camera's strip apart *on the ground*: how far apart two contours lie is the
    height between them over the slope's steepness, and that is measured band by
    band of height (its median) -- gentle on the valley's floor, so contours a few
    metres apart there; steep on the walls, so tens of metres. Lanes shorter than
    80 m are left out. Their order: from ``start`` (the base) to the nearest end
    of a lane not yet flown, either way along it -- but not one closer than the
    plane can turn into (two turning radii), unless it is far shorter.
    """
    import contourpy
    from scipy.ndimage import gaussian_filter

    from ..platformy import UZYTECZNY_PAS

    spacing = kamera.pas(agl) * UZYTECZNY_PAS
    zs = gaussian_filter(np.asarray(teren.z, float), 40.0 / teren.krok)
    a = -teren.polowa + teren.krok * np.arange(teren.n)
    m = (a >= -POLE - 10) & (a <= POLE + 10)
    zc, ac = zs[np.ix_(m, m)], a[m]
    gy, gx = np.gradient(zc, teren.krok)
    steep = np.hypot(gx, gy)
    band = 10.0
    edges = np.arange(zc.min(), zc.max() + band, band)
    grad = []
    for e0, e1 in zip(edges[:-1], edges[1:], strict=True):
        inside = (zc >= e0) & (zc < e1)
        grad.append(np.median(steep[inside]) if inside.sum() > 20 else 0.3)
    across = np.concatenate([[0.0], np.cumsum(band / np.clip(grad, 0.03, None))])
    levels = np.interp(np.arange(spacing / 2, across[-1], spacing), across, edges)
    gen = contourpy.contour_generator(ac, ac, zc)
    lanes = []
    for level in levels:
        for line in gen.lines(level):
            seg = np.hypot(*np.diff(line, axis=0).T)
            along = np.concatenate([[0.0], np.cumsum(seg)])
            if along[-1] < 80.0:
                continue
            d = np.linspace(0.0, along[-1], max(2, int(np.ceil(along[-1] / co)) + 1))
            lanes.append(
                np.stack(
                    [np.interp(d, along, line[:, 0]), np.interp(d, along, line[:, 1])], axis=1
                )
            )
    radius = platforma.predkosc**2 / (G * np.tan(np.deg2rad(35.0)))
    here = np.asarray(start if start is not None else teren.baza, float)
    order, left = [], list(range(len(lanes)))
    while left:
        best, cost, flip = None, np.inf, False
        for i in left:
            for rev, end in ((False, lanes[i][0]), (True, lanes[i][-1])):
                d = float(np.hypot(*(end - here)))
                c = d + (4 * radius if d < 2 * radius else 0.0)
                if c < cost:
                    best, cost, flip = i, c, rev
        lane = lanes[best][::-1] if flip else lanes[best]
        order.append([tuple(p) for p in lane])
        here = lane[-1]
        left.remove(best)
    return order


def trasa_lotu(platforma: Platforma, kamera: Kamera, agl: float, teren, trasa=None):
    """(which, the lanes): the plane flies the contours, the quadcopter north-south lanes,
    unless ``trasa`` ("pasy", "poziomice") says otherwise."""
    trasa = trasa or ("pasy" if platforma.zawis else "poziomice")
    if trasa == "poziomice":
        return trasa, poziomice(platforma, kamera, agl, teren)
    return trasa, pasy(platforma, kamera, agl)


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
    trasa: str | None = None,
) -> Wynik:
    """Fly one search of the area; ``klatka(lot, widok, plamy, faza)`` sees every picture,
    ``krok(t, lot, faza, cel, kandydaci)`` every step -- and ends it by returning True.
    ``trasa``: "pasy" (north-south, :func:`pasy`) or "poziomice" (along the contours,
    :func:`poziomice`); by default the plane flies the contours, the quadcopter lanes."""
    rng = np.random.default_rng(seed)
    agl = agl or platforma.wysokosc
    low = NIZEJ[platforma.nazwa]
    lot = Lot(platforma, teren, teren.baza, agl)
    trasa, route = trasa_lotu(platforma, kamera, agl, teren, trasa)
    lanes = [p for lane in route for p in lane]
    lane_of = [i for i, lane in enumerate(route) for _ in lane]
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
            nxt = wp + 1 < len(lanes) and lane_of[wp + 1] == lane_of[wp]
            first = wp == 0 or lane_of[wp - 1] != lane_of[wp]
            ahead = (
                [lanes[i] for i in range(wp + 1, min(wp + 30, len(lanes)))]
                if trasa == "poziomice"
                else None
            )
            if lot.krok(dt, lanes[wp], agl, droga=ahead) < (8.0 if platforma.zawis else 30.0):
                wp += 1
            elif nxt and not first:
                # Along a lane, a point it has gone by is not turned back for.
                a_, b_ = np.asarray(lanes[wp]), np.asarray(lanes[wp + 1])
                if np.hypot(lot.x - b_[0], lot.y - b_[1]) < np.hypot(*(b_ - a_)):
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
