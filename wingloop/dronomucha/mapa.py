"""Where to search next: a map of where the target may be, as search theory keeps one.

Rescue services plan a search on a map of probabilities: each cell's chance
that the missing person is there (the probability of area, POA), from where
they were last seen and what people like them do (lost-person statistics).
A look from somewhere finds what is there with a probability that falls with
distance (the probability of detection, POD); a look that finds nothing
leaves each cell less likely by that much -- Bayes:

    POA <- POA * (1 - POD) / (1 - sum(POA * POD))

and the next place to search from is where the most probability can be
covered for the time it takes to get there and look (Koopman's search
theory, the basis of IAMSAR's planning). The expanding square
(:func:`~.zadanie.kwadrat`) is what is flown without a map: the same turns
whatever the drone has already seen.

:class:`Mapa` is the navigator's planner
(:attr:`~wingloop.lab.navigate.Navigator.planner`): everything the drone has
swept with its eye looking around -- wherever it did, at home, at the places
it landed or turned away from, at the places the map sent it to -- lowers
the map there, and when it has nothing left to look at it asks the map
where next.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wingloop.lab.navigate import FIXATE

#: Measured slower than this, mm/s, what it looks for is taken to stand still.
STOI = 2.0

#: How the probability spreads for a target that moves at ``ruch`` mm/s: as
#: far as it can go in this many seconds, s, after that many seconds, and
#: like a random walk after -- the same however often the map is spread.
#: And how seldom it is spread, s: each spread lets places swept before be
#: swept again.
SKALA_CZASU, CO_ILE = 10.0, 3.0


@dataclass
class Mapa:
    """A search map, mm from where the search starts -- from the navigator's home, which
    is where the search starts (0, 0).

    ``prior``: where the target may be before the search, ``"pierscien"``
    (a ring ``promien`` mm out, ``szerokosc`` wide: lost-person statistics
    give a distance from where they were last seen, not a direction) or
    ``"rowny"`` (anywhere within ``zasieg_mapy``). ``wykrycie``: how far a look
    around finds the target, mm -- its probability of detection is ``pod`` at
    the drone's feet, falling as ``exp(-(d / wykrycie)^2)``. ``predkosc``:
    the drone's speed between places, mm/s, and ``rozgladanie``: how long a
    look around takes, s. ``koniec``: the share of the probability left at
    which it stops searching. A place within ``odstep`` mm of one it has
    looked around from is not sent to again -- or, when targets move, not
    within ``powrot`` seconds of looking from there; and an area where they
    move is never called cleared: it searches to the end.

    ``ruch``: how fast what it looks for may move, mm/s. A place looked at
    and found empty does not stay empty: a castaway drifts, so the
    probability spreads back over it as time goes by, by about ``ruch`` mm
    a second, and a place looked at long ago is worth looking at again. None:
    as fast as the objects it has fixed move (:mod:`.pamiec`), or 0 without
    a memory of objects.
    """

    prior: str = "pierscien"
    promien: float = 65.0
    szerokosc: float = 20.0
    zasieg_mapy: float = 110.0
    krok: float = 5.0
    wykrycie: float = 70.0
    pod: float = 0.85
    predkosc: float = 40.0
    rozgladanie: float = 5.0
    koniec: float = 0.05
    odstep: float = 30.0
    ruch: float | None = None
    powrot: float = 15.0
    #: Every place it looked around from, and when it last did.
    przejrzane: dict = field(default_factory=dict)
    #: Every place the map sent it to, and when.
    cele: list = field(default_factory=list)

    def __post_init__(self):
        n = int(round(self.zasieg_mapy / self.krok))
        a = np.arange(-n, n + 1) * self.krok
        self.x, self.y = np.meshgrid(a, a)
        r = np.hypot(self.x, self.y)
        self.inside = r <= self.zasieg_mapy
        if self.prior == "pierscien":
            p = np.exp(-0.5 * ((r - self.promien) / self.szerokosc) ** 2)
        else:
            p = np.ones_like(r)
        p = np.where(self.inside, p, 0.0)
        self.p = p / p.sum()
        self._read = 0
        self._t = 0.0
        self._swept = {}  # each place's cells already swept since the map last spread
        self._pod = {}  # each place's probability of detection, cell by cell
        self._speed = 0.0 if self.ruch is None else self.ruch

    def pod_z(self, x, y) -> np.ndarray:
        """Each cell's probability of detection by a look around from (x, y)."""
        key = (float(x), float(y))
        if key not in self._pod:
            d = np.hypot(self.x - x, self.y - y)
            self._pod[key] = self.pod * np.exp(-((d / self.wykrycie) ** 2))
        return self._pod[key]

    def poa(self) -> np.ndarray:
        """Each cell's probability now: the chance the target is there and unfound."""
        return self.p

    def zostalo(self) -> float:
        """The probability not yet looked away: the chance the target is still unfound."""
        return float(self.p.sum())

    def _spread(self, t: float) -> None:
        """Up to ``t``: the probability spreads as far as the target may have moved."""
        from scipy.ndimage import gaussian_filter

        dt = t - self._t
        if dt < CO_ILE or self._speed <= 0:
            return  # and the time is spread once a speed is known
        self._t = t
        mass = self.p.sum()
        sigma = self._speed * np.sqrt(dt * SKALA_CZASU) / self.krok
        p = gaussian_filter(self.p, sigma, mode="constant")
        p = np.where(self.inside, p, 0.0)
        self.p = p * mass / max(p.sum(), 1e-12)
        self._swept = {}  # what was swept before may have been driven into

    def zapisz(self, log) -> None:
        """Take in what the eye swept since last time: every look while turning on the spot.

        Each cell a look around sweeps is made less likely by its probability
        of detection from there, once a look around (Bayes, unnormalised: the
        sum left is the chance the target is still unfound).
        """
        for entry in log[self._read :]:
            if entry["phase"] != "scan":
                continue
            self._spread(entry.get("t", 0.0))
            key = (round(entry["x"] / 10) * 10, round(entry["y"] / 10) * 10)
            self.przejrzane[key] = entry.get("t", 0.0)
            to = np.degrees(np.arctan2(self.y - entry["y"], self.x - entry["x"]))
            off = np.abs((to - entry["heading"] + 180) % 360 - 180)
            done = self._swept.setdefault(key, np.zeros(self.p.shape, dtype=bool))
            new = (off < FIXATE) & ~done
            if new.any():
                self.p = self.p * (1.0 - self.pod_z(*key) * new)
                done |= new
        self._read = len(log)

    def _measured_speed(self, nav) -> float:
        """How fast the objects it has fixed move: the middle of their tracks' speeds.

        Below :data:`STOI` mm/s they stand: that is how far apart a still
        object's fixes come, as the gimbal finds another edge of it.
        """
        tracks = getattr(getattr(nav, "memory", None), "slady", ())
        speeds = [np.hypot(*s.predkosc()) for s in tracks if s.t[-1] - s.t[0] >= 1.0]
        speed = float(np.median(speeds)) if speeds else 0.0
        return speed if speed >= STOI else 0.0

    def __call__(self, nav, here):
        """The next place to look around from, mm, or None: the planner's question."""
        if self.ruch is None:
            self._speed = self._measured_speed(nav)
        self.zapisz(nav.log)
        if nav.log:
            self._spread(nav.log[-1].get("t", 0.0))
        if self.zostalo() < self.koniec and self._speed <= 0:
            # Cleared -- of a target that stands. One that moves can drive
            # back into what was looked at, or be the thing turned down
            # there: measured, a search that called its area cleared ended
            # at 53-87 s with targets driving round it.
            return None
        poa = self.poa()
        hx, hy = nav._home
        now = nav.log[-1].get("t", 0.0) if nav.log else 0.0
        # A drifting target can come back to where it was looked for: then
        # only where it looked from, or was sent to, lately is not worth
        # flying to.
        places = [*self.przejrzane.items(), *self.cele]
        been = [p for p, t in places if self._speed <= 0 or now - t < self.powrot]
        been.append(tuple(here))
        been = np.array(been, dtype=float).reshape(-1, 2)
        cand = self.krok * 2
        best, score = None, 0.0
        for cx in np.arange(-self.zasieg_mapy, self.zasieg_mapy + 1, cand):
            for cy in np.arange(-self.zasieg_mapy, self.zasieg_mapy + 1, cand):
                if np.hypot(cx, cy) > self.zasieg_mapy:
                    continue
                gap = np.hypot(been[:, 0] - hx - cx, been[:, 1] - hy - cy)
                if len(gap) and gap.min() < self.odstep:
                    continue
                gain = float((poa * self.pod_z(cx, cy)).sum())
                t = np.hypot(hx + cx - here[0], hy + cy - here[1]) / self.predkosc
                s = gain / (t + self.rozgladanie)
                if s > score:
                    best, score = (hx + cx, hy + cy), s
        if best is None:
            return None
        self.cele.append((best, now))
        return best
