"""The two platforms flying a search of a square kilometre, and coming lower to look again.

The plane flies lanes along the contours (:func:`poziomice`): the ground
below a lane stays at one height, so it holds its height over it without
climbing and coming down as a lane across a steep slope makes it. The
quadcopter flies lanes north-south (:func:`pasy`), those nearest the trail
first. The lanes lie so the camera's strips touch. Both take a picture a
second and judge every warm spot in it (:mod:`.wykrywanie`). What the
brain likes in two of the pictures a place is in (:data:`RAZY`) is a
candidate, and they go to look again from lower down:

- the **quadcopter** flies to it, comes down to :data:`NIZEJ` ``["quad"]`` m,
  hovers and looks three times; it flies the lanes nearest the trail first
  (where people are lost most) -- it cannot fly them all on one battery;
- the **tiltrotor** (:data:`~..platformy.VTOL`) searches as the plane does and
  looks as the quadcopter does: it tilts its rotors up, stops over the
  candidate, comes down to :data:`NIZEJ` ``["vtol"]`` m (35) and looks three
  times, then climbs in a hover to where it can fly as a plane again; a
  minute in a hover costs it several of cruising (:attr:`Lot.energia`);
- the **plane** cannot stop: it flies over it along its contour
  (:meth:`~.teren.Teren.poziomica`), at :data:`NIZEJ` ``["samolot"]`` m,
  looking four times a second while it is in the picture -- along the
  contour, because that is the way it can come down to 35 m beside a
  slope and fly on; once more from the other end if it saw nothing.

What the closer look likes is reported -- where, for the team; what it does
not is not looked at again; what it could not look at from close -- no
time, or no way down to it -- stays a candidate, reported as to be checked,
for the operator. The lanes end at four fifths of the flight; it goes on
looking again at what it found until 92%, and turns for home whatever it
is doing while its battery still gets it there.

**Height.** Each holds its height over the ground of the way it will fly:
the plane works out that way -- turning as tightly as it can toward where
it is going, then along its lane -- and climbs ahead of the ground, as
ArduPilot's terrain following does; where it could not keep even its least
height (:data:`NAJNIZEJ`) climbing as fast as it can, it circles away from
the slope and climbs first. Nothing lifts it over a slope it flew at: one
that meets the ground has crashed (:attr:`Lot.rozbity`), and the search
ends there. Kinematics only: a body that flies where it is pointed, as
fast as it can, turning and climbing no faster than it can -- no wind, no
physics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..platformy import PIKSELE_NA_CZLOWIEKA, Kamera, Platforma
from . import kamera as K
from . import wykrywanie as W

#: How low it comes to look again, m above the ground, and how much higher than that a
#: look still counts as one from close.
#: The tiltrotor has two sets of these: ``"vtol"`` flying as a plane, ``"vtol_zawis"``
#: in a hover (:attr:`Lot.klucz`).
#: The tiltrotor hovers at 35 m: from 15 m straight over them, the T256s's brain turned
#: down every person sitting or standing (0 of 11); from 35 m it found 60% of all, not 46%.
NIZEJ = {"quad": 15.0, "samolot": 35.0, "vtol": 35.0, "vtol_zawis": 35.0}
LUZ = {"quad": 3.0, "samolot": 8.0, "vtol": 8.0, "vtol_zawis": 3.0}
#: How fast each climbs and comes down, m/s.
WZNOSZENIE = {"quad": 3.0, "samolot": 5.0, "vtol": 5.0, "vtol_zawis": 3.0}
OPADANIE = {"quad": 2.5, "samolot": 5.0, "vtol": 5.0, "vtol_zawis": 2.5}
#: How far along its way each looks at the ground ahead to hold its height, m, and the
#: least height over the ground it is to keep, m.
PRZED = {"quad": 60.0, "samolot": 800.0, "vtol": 800.0, "vtol_zawis": 60.0}
NAJNIZEJ = {"quad": 5.0, "samolot": 25.0, "vtol": 25.0, "vtol_zawis": 5.0}
#: How fast the tiltrotor moves in a hover, m/s, and how fast it speeds up once its
#: rotors tilt forward, m/s2.
ZAWIS_V, ROZPED = 8.0, 3.0
#: How far past the end of the way it knows the plane keeps its least height: room to turn.
ZAWROT = 150.0
#: Within this of a point of its way it has reached it, m; the plane turns no tighter
#: than about 33 m, so a point it must reach (a lane's first, home) is given more.
BLISKO = {"quad": 8.0, "samolot": 30.0, "vtol": 30.0, "vtol_zawis": 8.0}
DOLOT = {"quad": 8.0, "samolot": 45.0, "vtol": 45.0, "vtol_zawis": 8.0}
#: Home is flown to with this many times the time the straight way takes left.
ZAPAS = 1.5
#: How long it tries to get over one candidate, s.
PROBA = 120.0
#: In how many pictures from the lane the brain must like a spot before it is a
#: candidate: every place is in three or four pictures as it flies over, and what
#: the brain likes in one of them only is the picture's noise more often than a person.
RAZY = 2
#: The area searched: this many metres each way of its middle.
POLE = 500.0
G = 9.81
PRZECHYL = 35.0
#: A pixel no bigger than this on the ground, m, can find a person: ground seen with a
#: bigger one is not searched (:data:`~..platformy.PIKSELE_NA_CZLOWIEKA`).
PIKSEL = 0.5 / (PIKSELE_NA_CZLOWIEKA / 2)


@dataclass(eq=False)
class Kandydat:
    """A warm spot the brain liked: where, how much, and what became of it.

    ``stan``: ``"czeka"`` (not judged from close: to be looked at, or, once
    ``proba`` is set, tried and not managed -- for the operator),
    ``"potwierdzony"``, ``"odrzucony"``. ``oceny``: what the brain thought at
    each closer look, ``wysokosci``: from how high each was, m. ``razy``: in how
    many pictures from the lane the brain liked it.
    """

    gdzie: np.ndarray
    ocena: float
    t: float
    stan: str = "czeka"
    oceny: list = field(default_factory=list)
    wysokosci: list = field(default_factory=list)
    proba: bool = False
    razy: int = 1


@dataclass
class Wynik:
    """A search: the path (t, x, y, z, height above ground, phase), every candidate, which
    of the area's cells were searched, the pictures taken, how long it flew, s;
    where it crashed (t, x, y) or None, how long it flew below its least
    height, s, whether it got home before its battery ended; the battery used, in
    seconds of cruising, and how often the tiltrotor tilted its rotors."""

    slad: list
    kandydaci: list
    pokrycie: np.ndarray
    zdjecia: int
    czas: float
    rozbicie: tuple | None = None
    za_nisko: float = 0.0
    wrocil: bool = True
    energia: float = 0.0
    przejscia: int = 0


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Lot:
    """A platform in the air: where it is, which way it heads, how fast it goes.

    ``rozbity``: it met the ground. ``za_nisko``: seconds flown below its least
    height (:data:`NAJNIZEJ`) since it first climbed above it. ``krazy``: the
    plane is circling to climb (+1 left, -1 right), 0 otherwise. ``zawis``: it
    hovers -- the quadcopter always, the tiltrotor when told (:meth:`tryb`).
    ``energia``: the battery used, in seconds of cruising: a second of the
    tiltrotor's hover is :attr:`~..platformy.Platforma.zawis_koszt` of them.
    """

    def __init__(self, platforma: Platforma, teren, start, agl: float, kurs: float = 0.0):
        self.p, self.t = platforma, teren
        x, y = start
        self.x, self.y = float(x), float(y)
        self.z = float(teren.wysokosc(x, y)) + 2.0
        self.kurs = float(kurs)
        # The tiltrotor takes off as a quadcopter does: straight up, in a hover.
        self.zawis = platforma.zawis or platforma.vtol
        self.v = 0.0 if self.zawis else platforma.predkosc
        self.agl = agl
        self.energia = 0.0
        self.przejscia = 0
        self.wystartowal = False
        self.rozbity = False
        self.za_nisko = 0.0
        self.krazy = 0

    @property
    def klucz(self) -> str:
        """Which of the tables' rows it flies by now: the tiltrotor has one for each mode."""
        if self.p.vtol:
            return "vtol_zawis" if self.zawis else "vtol"
        return self.p.nazwa

    def tryb(self, zawis: bool) -> None:
        """The tiltrotor: hover, or fly as a plane. Leaving a hover low down it climbs in
        the hover first, to where it can fly as a plane (:data:`NAJNIZEJ` and 5 m)."""
        if not self.p.vtol:
            return
        if not zawis and self.zawis and self.nad_ziemia < NAJNIZEJ["vtol"] + 5.0:
            zawis = True
        if zawis != self.zawis:
            self.zawis = zawis
            self.krazy = 0
            self.przejscia += 1

    @property
    def skret(self) -> float:
        """How fast the plane turns, rad/s, banked :data:`PRZECHYL` degrees."""
        return G * math.tan(math.radians(PRZECHYL)) / max(self.v, 1.0)

    def tor(self, cel, droga=None, ile: float | None = None):
        """The way the plane will fly, from where it is: (x, y, metres along it, known).

        Turning as tightly as it can toward ``cel``, on through ``droga``
        (the points after it), then straight on for :data:`ZAWROT` m -- past
        the end of what it knows (``known`` False) it has only to have room
        to turn. A point every second of flight, for ``ile`` m.
        """
        pts = [cel, *(droga if droga is not None else ())]
        ile = PRZED[self.klucz] if ile is None else ile
        step = max(self.v, 1.0)
        rate = self.skret
        x, y, k = self.x, self.y, self.kurs
        near2 = BLISKO[self.klucz] ** 2
        xs, ys, known = [x], [y], [True]
        i, past = 0, 0.0
        for _ in range(int(ile / step)):
            while i < len(pts) and (pts[i][0] - x) ** 2 + (pts[i][1] - y) ** 2 < near2:
                i += 1
            if i < len(pts):
                turn = _wrap(math.atan2(pts[i][1] - y, pts[i][0] - x) - k)
                k += max(-rate, min(rate, turn))
            else:
                past += step
                if past > ZAWROT:
                    break
            x += step * math.cos(k)
            y += step * math.sin(k)
            xs.append(x)
            ys.append(y)
            known.append(i < len(pts))
        d = step * np.arange(len(xs))
        return np.array(xs), np.array(ys), d, np.array(known, bool)

    def wysokosc_potrzebna(self, agl: float, ile: float, ku: float) -> float:
        """How high it must be now to keep ``agl`` m over the ground for the next ``ile`` m
        straight along the bearing ``ku`` (radians), climbing no faster than it
        can: the highest ground ahead, less what it can climb before it gets there."""
        d = np.linspace(0, ile, 32)
        xs = self.x + d * np.cos(ku)
        ys = self.y + d * np.sin(ku)
        climb = WZNOSZENIE[self.klucz] * d / max(self.v, 1.0)
        return float(np.max(self.t.wysokosc(xs, ys) + agl - climb))

    def krok(
        self, dt: float, cel, agl: float, stop: bool = False, droga=None, szuka: bool = False
    ) -> float:
        """Fly ``dt`` s toward ``cel`` (x, y) at ``agl`` m above the ground; return the distance
        left. ``stop``: a quadcopter slows to a hover on it. ``droga``: the points it
        flies through after ``cel`` -- the plane holds its height along them.
        ``szuka``: it is searching -- a quadcopter slows where the ground falls away
        faster than it can come down, to stay at its height over it."""
        if self.rozbity:
            return float(np.hypot(cel[0] - self.x, cel[1] - self.y))
        name = self.klucz
        up, down, floor = WZNOSZENIE[name], OPADANIE[name], NAJNIZEJ[name]
        self.energia += dt * (self.p.zawis_koszt if self.zawis and self.p.vtol else 1.0)
        dx, dy = cel[0] - self.x, cel[1] - self.y
        dist = float(np.hypot(dx, dy))
        want = math.atan2(dy, dx)
        turn = _wrap(want - self.kurs)
        here = float(self.t.wysokosc(self.x, self.y))
        if self.zawis:
            top = ZAWIS_V if self.p.vtol else self.p.predkosc
            self.kurs += float(np.clip(turn, -np.pi / 2 * dt, np.pi / 2 * dt))
            target_v = top if not stop else min(top, 0.8 * dist)
            if abs(turn) > np.pi / 3:
                target_v = min(target_v, 1.0)
            ahead = 5.0 if stop and dist < 10.0 else PRZED["quad"]
            z_want = self.wysokosc_potrzebna(agl, min(ahead, max(dist, 5.0)), want)
            if z_want - self.z > 5.0 or (szuka and self.z - z_want > max(5.0, 0.25 * agl)):
                # A quadcopter slows to climb -- and, searching, to come down
                # where the ground falls away faster than it can: from twice
                # its height it finds nothing.
                target_v = min(target_v, 2.0)
            if self.z < here + floor:
                target_v = 0.0  # and under its least height stops, and climbs
            self.v += float(np.clip(target_v - self.v, -3.0 * dt, 3.0 * dt))
        else:
            if self.v < self.p.predkosc:  # the tiltrotor, its rotors tilting forward
                self.v = min(self.p.predkosc, self.v + ROZPED * dt)
            xs, ys, d, known = self.tor(cel, droga)
            ground = self.t.wysokosc(xs, ys) - up * d / max(self.v, 1.0)
            least = float(np.max(ground)) + floor
            z_want = max(float(np.max(ground[known])) + agl, least)
            if not self.wystartowal:
                z_want = max(z_want, here + floor)  # thrown: straight up to its least height
            elif self.krazy:
                if self.z >= least + 5.0:
                    self.krazy = 0
            elif self.z < least - 2.0:
                # It cannot keep its least height on this way however it climbs:
                # it circles, on the side where the ground is lower, and climbs.
                r = self.v / self.skret
                sides = [
                    float(
                        self.t.wysokosc(
                            self.x - s * r * math.sin(self.kurs),
                            self.y + s * r * math.cos(self.kurs),
                        )
                    )
                    for s in (1, -1)
                ]
                self.krazy = 1 if sides[0] <= sides[1] else -1
            rate = self.skret * dt
            if self.krazy:
                self.kurs += self.krazy * rate
                z_want = self.z + up
            else:
                self.kurs += max(-rate, min(rate, turn))
        self.x += self.v * math.cos(self.kurs) * dt
        self.y += self.v * math.sin(self.kurs) * dt
        self.z += float(np.clip(z_want - self.z, -down * dt, up * dt))
        over = self.nad_ziemia
        if not self.wystartowal:
            self.wystartowal = over >= floor - 0.5
        elif over < 0.5:
            self.rozbity = True  # nothing lifts it over a slope it flew at
        elif over < floor - 0.5:
            self.za_nisko += dt
        return float(np.hypot(cel[0] - self.x, cel[1] - self.y))

    @property
    def nad_ziemia(self) -> float:
        return self.z - float(self.t.wysokosc(self.x, self.y))


def pasy(
    platforma: Platforma, kamera: Kamera, agl: float, szlak_x: float = 60.0, start=None
) -> list:
    """The lanes' ends, in the order flown: [(x, y from), (x, y to)], ... -- the first from
    its end nearer ``start`` (x, y): the way to its far end would be flown twice."""
    from ..platformy import UZYTECZNY_PAS

    spacing = kamera.pas(agl) * UZYTECZNY_PAS
    n = int(np.ceil(2 * POLE / spacing))
    xs = -POLE + spacing * (np.arange(n) + 0.5)
    if platforma.zawis:
        order = list(np.argsort(np.abs(xs - szlak_x), kind="stable"))
    else:
        radius = platforma.predkosc**2 / (G * np.tan(np.deg2rad(PRZECHYL)))
        k = max(1, int(np.ceil(2 * radius / spacing)))
        order, done = [], set()
        for offset in range(k + 1):
            for i in range(offset, n, k + 1):
                if i not in done:
                    order.append(i)
                    done.add(i)
    lanes = []
    for j, i in enumerate(order):
        ends = [(-POLE - 20.0), (POLE + 20.0)]
        if (j % 2 == 1) != (start is not None and start[1] > 0):
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

    from ..platformy import UZYTECZNY_PAS

    spacing = kamera.pas(agl) * UZYTECZNY_PAS
    zs = teren.gladki(40.0)[0]
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
    radius = platforma.predkosc**2 / (G * np.tan(np.deg2rad(PRZECHYL)))
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
    return trasa, pasy(platforma, kamera, agl, start=teren.baza)


def _w_kadrze(rogi, gx, gy) -> np.ndarray:
    """Which points (gx, gy) lie inside a picture's footprint on the ground: a convex
    quadrilateral, either way round -- every edge has them on the same side."""
    sides = []
    for i in range(4):
        (x0, y0), (x1, y1) = rogi[i][:2], rogi[(i + 1) % 4][:2]
        sides.append((x1 - x0) * (gy - y0) - (y1 - y0) * (gx - x0))
    sides = np.stack(sides)
    return (sides >= 0).all(axis=0) | (sides <= 0).all(axis=0)


def przeszukane(widok, seen: np.ndarray, komorka: float = 10.0) -> None:
    """Mark the cells of the area (``komorka`` m, ``seen``) this picture searched: those a
    pixel fell in that was small enough on the ground to find a person
    (:data:`PIKSEL`). A slope seen from 180 m up is in the picture, and not searched."""
    good = widok.gsd[::4, ::4] <= PIKSEL
    pts = widok.ziemia[::4, ::4][good]
    ix = np.floor((pts[:, 0] + POLE) / komorka).astype(int)
    iy = np.floor((pts[:, 1] + POLE) / komorka).astype(int)
    ok = (ix >= 0) & (iy >= 0) & (ix < seen.shape[1]) & (iy < seen.shape[0])
    seen[iy[ok], ix[ok]] = True


class _Trasa:
    """A way through points, flown: which point it is flying to, lane by lane."""

    def __init__(self, lanes: list):
        self.pts = [p for lane in lanes for p in lane]
        self.pas = [i for i, lane in enumerate(lanes) for _ in lane]
        self.i = 0

    @property
    def koniec(self) -> bool:
        return self.i >= len(self.pts)

    def krok(self, lot: Lot, dt: float, agl: float, wzdluz: bool, szuka: bool = False) -> None:
        """One step toward the point it is flying to, and on to the next once there.

        A lane's first point must be reached (:data:`DOLOT`); one along a
        lane it has gone by is not turned back for. ``wzdluz``: the plane
        holds its height along the points to come.
        """
        i, name = self.i, lot.klucz
        first = i == 0 or self.pas[i - 1] != self.pas[i]
        more = i + 1 < len(self.pts) and self.pas[i + 1] == self.pas[i]
        ahead = self.pts[i + 1 : i + 30] if wzdluz else None
        left = lot.krok(dt, self.pts[i], agl, droga=ahead, szuka=szuka)
        if left < (DOLOT[name] if first else BLISKO[name]):
            self.i += 1
        elif more and not first:
            a, b = self.pts[i], self.pts[i + 1]
            if math.hypot(lot.x - b[0], lot.y - b[1]) < math.hypot(b[0] - a[0], b[1] - a[1]):
                self.i += 1


def _przelot(teren, lot: Lot, gdzie, przed: float = 100.0, za: float = 60.0) -> list:
    """The plane's way over a candidate: along its contour, from the end nearer the plane
    -- ``przed`` m to come down on before it, ``za`` m after; where the ground is
    too level to have a contour, straight on over it."""
    co = 20.0
    line = teren.poziomica(float(gdzie[0]), float(gdzie[1]), ile=przed, co=co)
    if line is None:
        ku = math.atan2(gdzie[1] - lot.y, gdzie[0] - lot.x)
        d = np.arange(-przed, przed + 1.0, co)
        line = np.stack([gdzie[0] + d * math.cos(ku), gdzie[1] + d * math.sin(ku)], axis=1)
    ends = [math.hypot(lot.x - e[0], lot.y - e[1]) for e in (line[0], line[-1])]
    if ends[1] < ends[0]:
        line = line[::-1]
    middle = len(line) // 2
    return [tuple(p) for p in line[: middle + int(round(za / co)) + 1]]


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
    powietrze: float = -2.0,
) -> Wynik:
    """Fly one search of the area; ``klatka(lot, widok, plamy, faza)`` sees every picture,
    ``krok(t, lot, faza, cel, kandydaci)`` every step -- and ends it by returning True.
    ``trasa``: "pasy" (north-south, :func:`pasy`) or "poziomice" (along the contours,
    :func:`poziomice`); by default the plane flies the contours, the quadcopter lanes.
    ``powietrze``: the air's temperature that day, C -- the drone's thermometer."""
    rng = np.random.default_rng(seed)
    name = platforma.nazwa
    agl = agl or platforma.wysokosc
    low = NIZEJ[name]
    # How it looks again: stopped in the air (quadcopter, tiltrotor), or flying by.
    hover = platforma.zawis or platforma.vtol
    trasa, lanes = trasa_lotu(platforma, kamera, agl, teren, trasa)
    route = _Trasa(lanes)
    first = route.pts[0]
    # Thrown, or taken off, toward where it is going.
    lot = Lot(
        platforma,
        teren,
        teren.baza,
        agl,
        kurs=math.atan2(first[1] - teren.baza[1], first[0] - teren.baza[0]),
    )
    seen = np.zeros((int(2 * POLE / 10), int(2 * POLE / 10)), bool)
    # The close looks are judged by the compartment taught on them, if the brain has one.
    judge = getattr(mozg, "z_bliska", None) or mozg
    candidates: list[Kandydat] = []
    traces: list[Kandydat] = []  # liked in one picture so far: not yet candidates
    path, shots = [], 0
    limit = platforma.zapas() * 60.0
    # The lanes end at ``limit``; what is still to be looked at again, until ``late``.
    late = 0.92 * platforma.lot_min * 60.0
    merge = 8.0 if platforma.zawis else 15.0
    t, next_shot = 0.0, 0.0
    phase, target, resume = "pasy", None, None
    over, since, tries = None, 0.0, 0  # the plane's way over a candidate; this try's start
    home = False

    def look(where_from_phase):
        nonlocal shots
        shots += 1
        v = K.zdjecie(
            teren, ground_temp, obiekty, kamera, (lot.x, lot.y, lot.z), lot.kurs, rng=rng
        )
        spots = W.plamy(v, powietrze=powietrze)
        if klatka is not None:
            klatka(lot, v, spots, where_from_phase)
        return v, spots

    def waiting():
        return [c for c in candidates if c.stan == "czeka" and not c.proba]

    def take(c):
        """Go and look at a candidate from close."""
        nonlocal target, phase, over, since, tries
        target, phase, since, tries = c, "potwierdz", t, 1
        over = None if hover else _Trasa([_przelot(teren, lot, c.gdzie)])

    def leave():
        """This candidate is judged, or given up: the next, or back to the lanes, or home."""
        nonlocal target, phase
        target = None
        rest = waiting()
        if rest and lot.energia < late:
            take(rest[0])
        elif route.koniec or lot.energia > limit:
            phase = "powrot"
        else:
            phase = "wroc"

    battery = platforma.lot_min * 60.0
    while lot.energia < battery:
        lot.tryb(phase == "potwierdz")
        if phase == "pasy":
            if route.koniec or lot.energia > limit:
                rest = waiting()
                if rest and potwierdzaj and lot.energia < late:
                    resume = teren.baza
                    take(rest[0])
                else:
                    phase = "powrot"
                continue
            route.krok(lot, dt, agl, wzdluz=trasa == "poziomice", szuka=True)
            if t >= next_shot and lot.nad_ziemia > 0.5 * agl:
                next_shot = t + 1.0
                v, spots = look("pasy")
                przeszukane(v, seen)
                for s in spots:
                    val = mozg.ocena(s)

                    def close(c, s=s):
                        return np.hypot(*(c.gdzie[:2] - s.gdzie[:2])) < merge

                    if val <= 0 or any(close(c) for c in candidates):
                        continue
                    c = next((c for c in traces if close(c)), None)
                    if c is None:
                        traces.append(Kandydat(s.gdzie, val, t))
                        continue
                    c.razy += 1
                    if c.razy < RAZY:
                        continue
                    traces.remove(c)
                    c.gdzie, c.ocena, c.t = s.gdzie, max(c.ocena, val), t
                    candidates.append(c)
                    if not potwierdzaj:
                        c.stan = "potwierdzony"
                    elif target is None:
                        resume = (lot.x, lot.y)
                        take(c)
        elif phase == "potwierdz":
            tx, ty = target.gdzie[:2]
            if hover:
                left = lot.krok(dt, (tx, ty), low, stop=True)
                ready = (
                    left < 4.0
                    and lot.zawis
                    and abs(lot.nad_ziemia - low) < LUZ[lot.klucz]
                    and lot.v < 1.5
                )
            else:
                over.krok(lot, dt, low, wzdluz=True)
                left = math.hypot(tx - lot.x, ty - lot.y)
                ready = left < 25.0 and lot.nad_ziemia < low + LUZ[name]
            if ready and t >= next_shot:
                next_shot = t + (0.5 if hover else 0.25)
                v, spots = look("potwierdz")
                # A look counts if the candidate's place is in the picture.
                if _w_kadrze(v.rogi, tx, ty):
                    near = [
                        s for s in spots if np.hypot(*(s.gdzie[:2] - target.gdzie[:2])) < 10.0
                    ]
                    best = min(
                        near,
                        key=lambda s: np.hypot(*(s.gdzie[:2] - target.gdzie[:2])),
                        default=None,
                    )
                    target.oceny.append(judge.ocena(best) if best is not None else None)
                    target.wysokosci.append(lot.nad_ziemia)
            done = len(target.oceny) >= 3 if hover else over.koniec
            if done and not target.oceny and not hover and tries < 2:
                # It flew by without the place in its picture, or too high: once
                # more, from the end it is at now.
                tries, since = 2, t
                over = _Trasa([_przelot(teren, lot, target.gdzie)])
            elif done or t - since > PROBA:
                good = [o for o in target.oceny if o is not None]
                if not target.oceny:
                    target.proba = True  # not seen from close: it stays a candidate
                else:
                    target.stan = "potwierdzony" if good and np.mean(good) > 0 else "odrzucony"
                leave()
            if lot.energia > late and phase == "potwierdz":
                phase = "powrot"
        elif phase == "wroc":
            if lot.krok(dt, resume, agl) < DOLOT[lot.klucz]:
                phase = "pasy"
        elif phase == "powrot":
            if lot.krok(dt, teren.baza, agl, stop=platforma.zawis) < 2 * BLISKO[lot.klucz]:
                home = True
                break
        path.append((t, lot.x, lot.y, lot.z, lot.nad_ziemia, phase))
        t += dt
        if phase != "powrot":
            # Whatever it is doing: home while the battery still gets it there.
            away = math.hypot(lot.x - teren.baza[0], lot.y - teren.baza[1])
            if lot.energia + ZAPAS * away / platforma.predkosc + 20.0 > battery:
                phase = "powrot"
        if lot.rozbity:
            break
        if krok is not None and krok(t, lot, phase, target, candidates):
            break
    crash = (t, lot.x, lot.y) if lot.rozbity else None
    return Wynik(
        path, candidates, seen, shots, t, crash, lot.za_nisko, home, lot.energia, lot.przejscia
    )


def ocen(wynik: Wynik, obiekty, promien: float = 8.0) -> dict:
    """How the search went: people found of those there, false alarms, when the first of
    them was first seen, how much of the area was searched; candidates left to be
    checked and how many of them are people; whether it crashed, whether it got
    home, how long it flew below its least height, and from how high its closer
    looks were (their middle)."""
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
    looks = [h for c in wynik.kandydaci for h in c.wysokosci]
    return dict(
        rozbicie=wynik.rozbicie is not None,
        wrocil=wynik.wrocil,
        przejscia=wynik.przejscia,
        za_nisko=wynik.za_nisko,
        z_bliska=float(np.median(looks)) if looks else None,
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
