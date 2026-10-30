"""A memory of objects, not of places: what it has been to, wherever it is now.

The fly's navigator remembers where it landed and where it turned something
down, and does not look that way again
(:meth:`~wingloop.lab.navigate.Navigator._already_visited`).
For a still world that is the same as remembering the objects. For a
moving one it is not: measured in the lab, with everything moving at 10 mm/s
it found 36 of 72 targets against 71 still -- what it had turned down drove
into places it had not, and was looked at and landed on again, and a target
that drove into a place it had turned something down at was never looked at.

Here every object the eye fixates is also found by the gimbal and the
rangefinder (:func:`~.zmysly.polozenie`), and the fixes of one object, joined
up, are a track: where it was when, and so where it is going. What the drone
lands on or turns down is its track's, and a direction is "already been to"
if it points at where such a track is now -- as a rescue drone's tracker
follows a drifting castaway. A drone without a gimbal fix of what it went
to keeps the place, as the fly does.

:class:`Pamiec` sits between the navigator and its brain: it is asked what
the brain thinks of each view, which is when the eye is on an object and
it fixes it, and it is the navigator's
:attr:`~wingloop.lab.navigate.Navigator.memory`.
"""

from __future__ import annotations

import numpy as np

from .zmysly import polozenie

#: Fastest an object is taken to move, mm/s: how far apart two fixes of it
#: may be. How far back a fix still says where it is going, s.
NAJSZYBCIEJ, HORYZONT = 15.0, 6.0


class Slad:
    """One object's fixes: when, and where."""

    def __init__(self, t: float, xy):
        self.t, self.xy = [t], [tuple(xy)]
        self.stan = None  # "byl": been to; "odrzucony": turned down

    def predkosc(self) -> np.ndarray:
        """Its velocity, mm/s, from its fixes over the last :data:`HORYZONT` s; 0 from one."""
        t = np.array(self.t)
        keep = t >= t[-1] - HORYZONT
        t, xy = t[keep], np.array(self.xy)[keep]
        if len(t) < 2 or t[-1] - t[0] < 0.4:
            return np.zeros(2)
        v = np.polyfit(t - t[-1], xy, 1)[0]
        speed = np.hypot(*v)
        return v if speed <= NAJSZYBCIEJ else v * NAJSZYBCIEJ / speed

    def gdzie(self, t: float) -> np.ndarray:
        """Where it is at ``t``, by its last fix and its velocity."""
        ahead = float(np.clip(t - self.t[-1], 0.0, HORYZONT))
        return np.array(self.xy[-1]) + self.predkosc() * ahead


class Pamiec:
    """The brain, asked through this, and the navigator's memory of objects."""

    def __init__(self, brain, eye, seed: int = 0):
        self.brain, self.eye = brain, eye
        self.rng = np.random.default_rng(seed)
        self.slady: list[Slad] = []
        self.nav = None
        self._last = None
        self._seen = (0, 0)  # how many visited and rejected places it has taken in
        self._places = []  # (x, y, radius, least) kept as places: no fix to go by

    def __getattr__(self, name):
        return getattr(self.brain, name)

    def _t(self) -> float:
        return self.nav.log[-1]["t"] if self.nav is not None and self.nav.log else 0.0

    def valence(self, f) -> float:
        self.namierz()
        return self.brain.valence(f)

    def peek(self, f) -> float:
        """What the brain thinks of a view, without fixing anything: for a screen."""
        return getattr(self.brain, "peek", self.brain.valence)(f)

    def learn(self, f, dopamine) -> None:
        self.brain.learn(f, dopamine)

    def namierz(self) -> None:
        """Fix what the eye is on, and add the fix to its object's track."""
        xy = polozenie(self.eye, self.rng)
        self._last = None
        if xy is None:
            return
        t = self._t()
        best, gap = None, np.inf
        for s in self.slady:
            dt = t - s.t[-1]
            off = np.hypot(*(s.gdzie(t) - xy))
            if off < min(8.0 + NAJSZYBCIEJ * dt, 40.0) and off < gap:
                best, gap = s, off
        if best is None:
            best = Slad(t, xy)
            self.slady.append(best)
        else:
            best.t.append(t)
            best.xy.append(xy)
        self._last = best

    def _take_in(self, nav) -> None:
        """What it has landed on or turned down since: the track of what it judged last."""
        fresh = self._last
        nv, nr = self._seen
        for place in nav.visited[nv:]:
            if fresh is not None:
                fresh.stan = "byl"
            else:
                self._places.append((*place, 30.0, 25.0))
        for place in nav.rejected[nr:]:
            if fresh is not None:
                fresh.stan = "odrzucony"
            else:
                self._places.append((*place, 30.0, 25.0))
        self._seen = len(nav.visited), len(nav.rejected)

    def __call__(self, nav, here, heading_deg: float) -> bool:
        """Does this direction point at something it has been to or turned down?"""
        self.nav = nav
        self._take_in(nav)
        t = self._t()
        places = list(self._places)
        places += [(*s.gdzie(t), 30.0, 25.0) for s in self.slady if s.stan]
        for x, y, r, least in places:
            dx, dy = x - here[0], y - here[1]
            to = np.degrees(np.arctan2(dy, dx))
            cone = max(least, np.degrees(np.arctan2(r, np.hypot(dx, dy))))
            if abs((to - heading_deg + 180) % 360 - 180) < cone:
                return True
        return False
