"""The fly's brain with functions added by people: a hybrid brain for the drone.

The core is the fly's: the optic lobe (:mod:`wingloop.lab.vision`) and the
mushroom body, which learns what to like by dopamine
(:mod:`wingloop.lab.learning`). Added to it, each a part that can be put in
or left out and is measured against the fly alone:

- **senses in the calyx** (:mod:`.zmysly`): what the drone's thermal camera
  and rangefinder read -- how warm, how far, how big -- taken into the
  mushroom body next to the eye's features, as a fly's calyx takes smell
  next to sight. The fly's own rule learns what they mean;
- **a cortex** (:class:`Kora`): a learning machine of the people's kind, on
  the same eye and senses, giving a second opinion the decision weighs.

:class:`Hybryda` is what the navigator calls its mushroom body: it reads the
senses where the eye looks each time it is asked what it thinks of a view.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .zmysly import Zmysly, kod


def wejscie(f, readings, gain: int) -> np.ndarray:
    """The calyx's input: the eye's features, then the senses' lines (none if ``gain`` 0)."""
    f = np.atleast_2d(f)
    if not gain:
        return f
    return np.concatenate([f, np.atleast_2d(kod(readings, gain))], axis=1)


class Kora:
    """A cortex: gradient-boosted trees on the eye's features and the senses' readings.

    Not a model of a fly: what a person would build to tell a target from
    the rest, from the same lessons. Its opinion is the probability it gives
    the target, as a valence: ``2p - 1``. Readings it lacks (NaN) it takes as
    missing, which the trees handle.
    """

    def __init__(self, seed: int = 0, senses: bool = True):
        self.seed, self.senses = seed, senses
        self.model = None

    def _x(self, F, S):
        F = np.atleast_2d(F)
        return np.concatenate([F, np.atleast_2d(S)], axis=1) if self.senses else F

    def fit(self, F, S, y, weight=None) -> Kora:
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.model = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=self.seed
        )
        self.model.fit(self._x(F, S), np.asarray(y, dtype=int), sample_weight=weight)
        return self

    def opinion(self, F, S) -> np.ndarray:
        """``2p - 1`` per view: +1 sure it is the target, -1 sure it is not."""
        return 2.0 * self.model.predict_proba(self._x(F, S))[:, 1] - 1.0


@dataclass
class Hybryda:
    """The brain the navigator asks: the fly's mushroom body, the senses, a cortex.

    ``mb``: the mushroom body, taught on :func:`wejscie` with ``gain``.
    ``kora``: a :class:`Kora`, or None; its opinion counts ``waga`` against
    the mushroom body's valence. ``eye``: the eye whose pose the senses read
    from (:class:`~wingloop.lab.watch.FlysEye`).
    """

    mb: object
    eye: object = None
    gain: int = 0
    kora: Kora | None = None
    waga: float = 0.0
    zmysly: Zmysly = field(default_factory=Zmysly)
    #: Every view judged: (the mushroom body's valence, the cortex's opinion
    #: or None, the readings) -- for the operator, and for measuring.
    oceny: list = field(default_factory=list)

    def __post_init__(self):
        self._last = None

    def senses(self) -> np.ndarray:
        return self.zmysly(self.eye)

    def parts(self, f) -> tuple:
        """(the mushroom body's valence, the cortex's opinion or None, the readings)."""
        s = self.senses() if (self.gain or self.kora) else None
        self._last = s
        v = float(self.mb.valence(wejscie(f, s, self.gain)[0]))
        c = float(self.kora.opinion(f, s)[0]) if self.kora is not None else None
        return v, c, s

    def valence(self, f) -> float:
        v, c, s = self.parts(f)
        self.oceny.append((v, c, s))
        return laczona(v, c, self.waga)

    def peek(self, f) -> float:
        """What it thinks of a view, not kept: for a screen."""
        v, c, _ = self.parts(f)
        return laczona(v, c, self.waga)

    def learn(self, f, dopamine) -> None:
        self.mb.learn(wejscie(f, self._last, self.gain)[0], dopamine)


def laczona(v, c, waga: float) -> float:
    """The decision's valence: the mushroom body's, and the cortex's opinion by ``waga``.

    The cortex's opinion runs -1 to +1 and the mushroom body's valence about
    -0.3 to +0.3; ``waga`` 0.3 gives them about equal say.
    """
    if c is None or not waga:
        return float(v)
    return float(np.clip(v + waga * c, -1.0, 1.0))


def naucz(far, near, target: str, gain: int = 0, kora: float = 0.0, seed: int = 0):
    """The hybrid brain taught on a task's lessons with their readings: (Hybryda, valence).

    ``far``, ``near``: (features, kinds, readings) of the tethered views and
    the close looks (:func:`~.mozg.lekcje` with ``zmysly``). The mushroom
    body is taught as the fly's always is
    (:func:`~wingloop.lab.autopilot.teach_targets`), on the eye's features
    and ``gain`` of the senses' lines; the cortex, if ``kora``, on both
    lessons, each kind weighed so the target and the rest count alike.
    Call it in the task's world (:func:`~.swiat.swiat`).
    """
    from wingloop.lab import autopilot

    F, K, S = far
    CF, CK, CS = near
    mb, valence = autopilot.teach_targets(
        wejscie(F, S, gain), K, {target}, seed=seed, close=(wejscie(CF, CS, gain), CK)
    )
    cortex = None
    if kora:
        X = np.concatenate([F, CF])
        R = np.concatenate([S, CS])
        y = np.concatenate([K, CK]) == target
        weight = np.where(y, 0.5 / y.mean(), 0.5 / (1 - y.mean()))
        cortex = Kora(seed=seed).fit(X, R, y, weight)
    return Hybryda(mb, gain=gain, kora=cortex, waga=kora), valence
