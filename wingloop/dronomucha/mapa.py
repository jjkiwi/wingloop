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


@dataclass
class Mapa:
    """A search map, mm from where the search starts.

    ``prior``: where the target may be before the search, ``"pierscien"``
    (a ring ``promien`` mm out, ``szerokosc`` wide: lost-person statistics
    give a distance from where they were last seen, not a direction) or
    ``"rowny"`` (anywhere within ``zasieg_mapy``). ``wykrycie``: how far a look
    around finds the target, mm -- its probability of detection is ``pod`` at
    the drone's feet, falling as ``exp(-(d / wykrycie)^2)``. ``predkosc``:
    the drone's speed between places, mm/s, and ``rozgladanie``: how long a
    look around takes, s. ``koniec``: the share of the probability left at
    which it stops searching.
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
    #: Every place it looked around from, and the cells it swept from there.
    przejrzane: dict = field(default_factory=dict)
    #: Every place the map sent it to.
    cele: list = field(default_factory=list)

    def __post_init__(self):
        n = int(round(self.zasieg_mapy / self.krok))
        a = np.arange(-n, n + 1) * self.krok
        self.x, self.y = np.meshgrid(a, a)
        r = np.hypot(self.x, self.y)
        inside = r <= self.zasieg_mapy
        if self.prior == "pierscien":
            p = np.exp(-0.5 * ((r - self.promien) / self.szerokosc) ** 2)
        else:
            p = np.ones_like(r)
        p = np.where(inside, p, 0.0)
        self.poa0 = p / p.sum()
        self._read = 0

    def pod_z(self, x, y) -> np.ndarray:
        """Each cell's probability of detection by a look around from (x, y)."""
        d = np.hypot(self.x - x, self.y - y)
        return self.pod * np.exp(-((d / self.wykrycie) ** 2))

    def poa(self) -> np.ndarray:
        """Each cell's probability now, after everything swept so far."""
        p = self.poa0.copy()
        for (x, y), swept in self.przejrzane.items():
            p *= 1.0 - self.pod_z(x, y) * swept
        return p / max(p.sum(), 1e-12) * self.zostalo()

    def zostalo(self) -> float:
        """The probability not yet looked away: the chance the target is still unfound."""
        p = self.poa0.copy()
        for (x, y), swept in self.przejrzane.items():
            p *= 1.0 - self.pod_z(x, y) * swept
        return float(p.sum())

    def zapisz(self, log) -> None:
        """Take in what the eye swept since last time: every look while turning on the spot."""
        for entry in log[self._read :]:
            if entry["phase"] != "scan":
                continue
            key = (round(entry["x"] / 10) * 10, round(entry["y"] / 10) * 10)
            to = np.degrees(np.arctan2(self.y - entry["y"], self.x - entry["x"]))
            off = np.abs((to - entry["heading"] + 180) % 360 - 180)
            seen = off < FIXATE
            if key not in self.przejrzane:
                self.przejrzane[key] = np.zeros_like(self.poa0)
            self.przejrzane[key] = np.maximum(self.przejrzane[key], seen)
        self._read = len(log)

    def __call__(self, nav, here):
        """The next place to look around from, mm, or None: the planner's question."""
        self.zapisz(nav.log)
        if self.zostalo() < self.koniec:
            return None
        poa = self.poa()
        hx, hy = nav._home
        been = np.array(list(self.przejrzane) + self.cele, dtype=float).reshape(-1, 2)
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
        self.cele.append(best)
        return best
