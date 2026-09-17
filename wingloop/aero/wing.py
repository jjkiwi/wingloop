"""Wing geometry, reduced to the few numbers aerodynamics actually needs.

A blade-element model does not need a wing outline. It needs the chord at each
spanwise station, because every force it computes is an integral of chord
against some power of radius -- and those integrals, not the shape, are what
distinguish one wing from another.

Everything here is millimetres and milligrams, matching NeuroMechFly's MuJoCo
units, so a force comes out in the units MuJoCo expects to be handed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Wing length of *Drosophila melanogaster*, millimetres.
WING_LENGTH_MM = 2.5

#: Mean chord, millimetres. Aspect ratio R/c is then about 3.2, which is the
#: usual figure for this animal.
MEAN_CHORD_MM = 0.78

#: Air density at sea level in this model's units, g/mm^3.
#:
#: The unit system was read off the real NeuroMechFly model rather than assumed:
#: its total body mass is 1.027e-3 and a fly weighs about a milligram, so mass
#: is in **grams**; its wing mesh spans 2.40 along one axis and a wing is about
#: 2.5 mm, so length is in **millimetres**; its gravity is 9810, which is
#: mm/s^2. Hence 1.225 kg/m^3 = 1.225e-6 g/mm^3.
#:
#: An earlier value here was wrong by six orders of magnitude and nothing in
#: the code noticed. :func:`wingloop.aero.blade_element.hover_check` exists
#: because of that: it is the only test that catches a unit error, since a
#: dimensionally broken model still returns a plausible-looking float.
AIR_DENSITY = 1.225e-6


@dataclass(frozen=True)
class Wing:
    """A wing as its spanwise chord distribution.

    ``stations`` are the *centres* of equal-width blade elements measured from
    the hinge, and ``chords`` the chord at each. Uniform spacing is assumed,
    which keeps the integrals a plain sum and is accurate enough at the element
    counts anyone would use.
    """

    stations: np.ndarray  # mm from the hinge
    chords: np.ndarray  # mm
    length: float = WING_LENGTH_MM

    def __post_init__(self):
        if len(self.stations) != len(self.chords):
            raise ValueError("stations and chords must be the same length")
        if len(self.stations) < 2:
            raise ValueError("a blade-element wing needs at least two elements")
        if np.any(self.chords <= 0):
            raise ValueError("chord must be positive everywhere")
        if np.any(np.diff(self.stations) <= 0):
            raise ValueError("stations must increase from the hinge outward")

    @property
    def dr(self) -> float:
        """Width of one blade element."""
        return float(self.length / len(self.stations))

    @property
    def area(self) -> float:
        """Planform area of one wing, mm^2."""
        return float(np.sum(self.chords) * self.dr)

    @property
    def mean_chord(self) -> float:
        return float(self.area / self.length)

    def moment(self, k: int) -> float:
        """The k-th moment of area, ``integral c(r) r^k dr``.

        The second moment (k=2) is the one that sets translational force, and
        the third (k=3) sets the aerodynamic torque about the hinge. Reporting
        them is how two wing shapes are compared meaningfully.
        """
        return float(np.sum(self.chords * self.stations**k) * self.dr)


def elliptical_wing(n: int = 20, *, length: float = WING_LENGTH_MM,
                    mean_chord: float = MEAN_CHORD_MM) -> Wing:
    """A wing whose chord follows a half-ellipse along the span.

    Not the real *Drosophila* outline, which is blunter at the base and tapers
    late. It is a stand-in with the right length, the right area and a
    plausible second moment, to be replaced by the chord distribution measured
    off the NeuroMechFly wing mesh -- see :func:`wing_from_mesh`.
    """
    edges = np.linspace(0.0, length, n + 1)
    stations = 0.5 * (edges[:-1] + edges[1:])
    shape = np.sqrt(np.clip(1.0 - (stations / length) ** 2, 0.0, None))
    # Scale so the mean chord comes out as asked.
    chords = shape * (mean_chord / shape.mean())
    return Wing(stations=stations, chords=chords, length=length)


def wing_from_mesh(vertices: np.ndarray, *, n: int = 20,
                   span_axis: int = 0, chord_axis: int = 1) -> Wing:
    """Chord distribution measured off a wing mesh, one bin at a time.

    This is how the model stops being a guess: NeuroMechFly ships an ``RWing``
    mesh, and its chord at each spanwise station is a fact about that mesh
    rather than an assumed outline.
    """
    v = np.asarray(vertices, dtype=float)
    if v.ndim != 2 or v.shape[1] < 3:
        raise ValueError("vertices must be an (N, 3) array")
    span = v[:, span_axis] - v[:, span_axis].min()
    length = float(span.max())
    if length <= 0:
        raise ValueError("the mesh has no extent along the span axis")
    edges = np.linspace(0.0, length, n + 1)
    stations, chords = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        inside = (span >= lo) & (span <= hi)
        if inside.sum() < 2:
            continue
        c = v[inside, chord_axis]
        stations.append(0.5 * (lo + hi))
        chords.append(float(c.max() - c.min()))
    if len(stations) < 2:
        raise ValueError("too few populated spanwise bins; lower n")
    chords = np.asarray(chords)
    # An empty tip bin would otherwise give a zero chord and fail validation.
    chords[chords <= 0] = chords[chords > 0].min()
    return Wing(stations=np.asarray(stations), chords=chords, length=length)
