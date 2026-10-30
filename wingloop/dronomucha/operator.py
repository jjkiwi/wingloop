"""Asking a person: the drone says how sure it is, and when it is not, asks.

A rescue drone is not alone: an operator watches its thermal picture at the
ground station. Everything it judges, it judges with a margin -- the brain's
valence, how far above or below the line it is -- and where the margin is
small, a wrong landing (a false alarm: a team sent to a rock) or a wrong
turn-down (a person left) is likeliest. There, instead of deciding, it
hovers, sends the picture and asks: "czy to człowiek?". The operator's
answer decides.

The operator here is simulated: a person who, looking at the thermal
picture from close by, is right ``trafnosc`` of the time and takes
``zwloka`` seconds to answer, the drone hovering meanwhile -- the search
loses that time (:func:`~.misja.szukaj`). Only close looks are asked about:
they are what a landing rests on.

Every close look is also a report, :attr:`Operator.zgloszenia`: what it
thinks it is, how sure, and where -- what a rescue drone sends its team.
"""

from __future__ import annotations

import numpy as np

from wingloop.lab.navigate import nearest

from .zmysly import polozenie


def pewnosc(v: float, skala: float = 0.1) -> float:
    """How sure the drone is that a view is its target, 0-1, from the decision's valence."""
    return float(1.0 / (1.0 + np.exp(-v / skala)))


class Operator:
    """The brain asked through this; when a close look is within ``prog`` of the line,
    the operator is asked instead.

    ``items``, ``target``: the search area and what is looked for -- what the
    operator, seeing the picture, can tell. ``prog``: how close to the line,
    in valence, a close look must be to ask.
    """

    def __init__(
        self,
        brain,
        eye,
        items,
        target: str,
        prog: float = 0.1,
        trafnosc: float = 0.95,
        zwloka: float = 5.0,
        seed: int = 0,
    ):
        self.brain, self.eye, self.items, self.target = brain, eye, items, target
        self.prog, self.trafnosc, self.zwloka = prog, trafnosc, zwloka
        self.rng = np.random.default_rng(seed)
        self.nav = None
        #: Every question: (t, was it the target, what the operator said).
        self.pytania: list = []
        #: Every close look: (t, valence, how sure, where it is or None, asked).
        self.zgloszenia: list = []

    def __getattr__(self, name):
        return getattr(self.brain, name)

    @property
    def czekanie(self) -> float:
        """Seconds hovered waiting for answers."""
        return self.zwloka * len(self.pytania)

    def peek(self, f) -> float:
        """What the brain thinks of a view, without asking anyone: for a screen."""
        return getattr(self.brain, "peek", self.brain.valence)(f)

    def learn(self, f, dopamine) -> None:
        self.brain.learn(f, dopamine)

    def valence(self, f) -> float:
        v = float(self.brain.valence(f))
        nav = self.nav
        if nav is None or nav.phase != "approach":
            return v
        t = nav.log[-1]["t"] if nav.log else 0.0
        where = polozenie(self.eye, self.rng)
        asked = abs(v) < self.prog
        self.zgloszenia.append((t, v, pewnosc(v), where, asked))
        if not asked:
            return v
        truth = self._is_target(where, nav)
        right = self.rng.random() < self.trafnosc
        said = truth if right else not truth
        self.pytania.append((t, truth, said))
        return 0.5 if said else -0.5

    def _is_target(self, where, nav) -> bool:
        """What the operator sees: what is where the gimbal points, or, without a fix, ahead."""
        t = nav.log[-1]["t"] if nav.log else 0.0
        if where is not None:
            it, d = nearest(self.items, *where, t)
            if d < 25.0:
                return it.kind == self.target
        x, y, h = nav.log[-1]["x"], nav.log[-1]["y"], nav.log[-1]["heading"]
        best, off = None, np.inf
        for it in self.items:
            ix, iy = it.at(t)
            o = abs((np.degrees(np.arctan2(iy - y, ix - x)) - h + 180) % 360 - 180)
            if o < off:
                best, off = it, o
        return best is not None and best.kind == self.target
