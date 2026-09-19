"""The connectome's steering command, for a body that flies.

`flyloop` steers a walking fly on ``DNa02_R - DNa02_L``. That readout does not
transfer: DNa02 supplies **0.21%** of the descending drive onto the wing
steering muscles. Flight runs on a different set, and which member of it
carries steering had to be measured rather than assumed.

Two things were measured, and they divide the labour cleanly.

**DNg02 is the actuator and it is blind.** It is the largest single descending
input to the wing steering muscles -- 9.07%, more than double the next -- and
about 29 cells across seven subtypes, which is what population-coded amplitude
control looks like from the wiring. It receives **1.48%** of its input from
visual neurons and 0.14% from the LC/LPLC projection types, and its
right-minus-left difference is **exactly zero at every bearing from -90 to
+90**. Nothing about where an object is reaches it.

**DNbe001 is the one that sees.** 20.4% visual input, 10.5% from LC/LPLC, the
second largest descending drive onto the wing muscles at 4.06%, and the
steepest bearing tuning of any candidate: a range of 0.175 across the visual
field against DNa10's 0.134 and DNge107's 0.042, with the sign the right way
round -- an object to the right drives the right side harder.

So the command path this module implements is **vision -> DNbe001 -> wing
amplitude asymmetry -> roll**, and the last arrow is the measured 38.2 of roll
torque per unit of asymmetry from :mod:`wingloop.body.control`.

**The curve is precomputed, and that is not a shortcut.** One pass through the
rate model costs about 0.16 s while a wingbeat is 4.6 ms, so a per-step readout
would be three hundred times slower than the animal. It is also the wrong
model of the animal: a fly's visual system does not resolve individual
wingbeats. Sampling the open-loop tuning once and interpolating is the same
thing `flyloop` calls the controller's open-loop shape, used here as a lookup.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Descending neurons whose bearing tuning was compared. The winner is first.
CANDIDATES = ("DNbe001", "DNa10", "DNp09", "DNge107", "DNa08", "DNg02")

#: The type this reads steering from, and the one it does not.
STEERING_TYPE = "DNbe001"
AMPLITUDE_TYPE = "DNg02"

#: Bearings sampled when building the curve, degrees, positive to the right.
DEFAULT_BEARINGS = tuple(float(b) for b in range(-90, 91, 15))

_HINT = (
    "The flight readout needs flyloop for the connectome and the rate model:\n"
    "    pip install -e /path/to/flyloop"
)


def tuning(
    connectome,
    *,
    types=CANDIDATES,
    bearings=DEFAULT_BEARINGS,
    hops: int = 5,
    half_width: float = 5.0,
):
    """Right-minus-left difference for each type, against object bearing.

    The mirror-pair quantity `flyloop` established: taking one side alone
    confounds the stimulus with a wiring lean, and the difference cancels it.
    """
    try:
        from flyloop.brain.rate import population_index, rate_brain, steady_state
        from flyloop.vision.hexproject import HexWorldView
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(_HINT) from exc

    view = HexWorldView(connectome)
    record = {}
    for side in ("L", "R"):
        record.update(population_index(connectome, tuple(types), side=side))
    brain = rate_brain(connectome, view.sensory, num_layers=hops)

    out = {"bearing": np.asarray(bearings, dtype=float)}
    columns = {name: [] for name in types}
    for bearing in bearings:
        peak = brain.run(
            steady_state(view.pattern([(bearing, half_width)]), hops), record=record
        ).peak()
        for name in types:
            columns[name].append(
                float(peak.get(f"{name}_R", 0.0) - peak.get(f"{name}_L", 0.0))
            )
    out.update({k: np.asarray(v) for k, v in columns.items()})
    return out


@dataclass
class FlightReadout:
    """Bearing in, steering command out, through the connectome.

    ``gain`` converts the descending difference into the wing amplitude
    asymmetry the body takes. At 1.0 the raw difference is used, which spans
    about +-0.09 across the visual field and is therefore already in a usable
    range -- the measured roll authority is 38.2 per unit, so the full field is
    worth about 3.4 of roll torque against a pitch disturbance of 3.5. That is
    a coincidence of scale, not a calibration, and it is why the gain exists.
    """

    bearings: np.ndarray
    command: np.ndarray
    gain: float = 1.0
    source: str = STEERING_TYPE
    extra: dict = field(default_factory=dict)

    @classmethod
    def measure(cls, connectome, *, gain: float = 1.0, source: str = STEERING_TYPE, **kw):
        curve = tuning(connectome, **kw)
        if source not in curve:
            raise KeyError(f"{source!r} was not among the types measured")
        return cls(
            bearings=curve["bearing"],
            command=curve[source],
            gain=gain,
            source=source,
            extra={k: v for k, v in curve.items() if k not in ("bearing", source)},
        )

    def asymmetry(self, bearing: float) -> float:
        """Wing amplitude asymmetry for an object at this bearing, degrees.

        Beyond the sampled field the command is held at its edge value rather
        than extrapolated: the tuning is not linear out there and inventing a
        slope would put the strongest commands where the measurements stop.
        """
        return float(
            self.gain * np.interp(bearing, self.bearings, self.command)
        )

    @property
    def span(self) -> float:
        """How much command the whole visual field is worth."""
        return float(self.command.max() - self.command.min())
